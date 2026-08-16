"""教授給的 14 份 segmentation 長什麼樣 —— 一顆一張 ID map,外加一張總覽。

教授在 Drive 的 wsky_cubes/ 底下,除了 cube 還附了每一顆的

    DATACUBE_FINAL_{N}_pseudo_r.fits    偵測用的底圖(nosky cube 在 5625-6825 A
                                        的等權平均)
    DATACUBE_FINAL_{N}_seg.fits         在那張底圖上跑出來的 segmentation

這支只做一件事:把兩者疊起來畫。底圖用**教授自己的 pseudo_r**,不用我們的
whitelight —— 遮罩是在 pseudo_r 上長出來的,配另一張底圖會看到對不齊的錯覺。

繪製本身完全交給 step4_catalog.id_map,不另外寫一套:同一種圖在專案裡只能有
一種畫法,否則兩份圖擺在一起會因為拉伸、配色、標號規則不同而看起來像不同的資料。

    conda run -n astro python src/skymodel/experiments/prof_seg_maps.py
    conda run -n astro python src/skymodel/experiments/prof_seg_maps.py -n 4 8 12
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step4_catalog import id_map  # noqa: E402

ROOT    = Path(__file__).resolve().parents[3]
SEGDIR  = ROOT / "data/wsky_seg"
FIGURES = ROOT / "results/skymodel/figures/prof_seg"


def load(n):
    seg = fits.getdata(SEGDIR / f"DATACUBE_FINAL_{n}_seg.fits").astype(int)
    img = np.asarray(fits.getdata(SEGDIR / f"DATACUBE_FINAL_{n}_pseudo_r.fits"), float)
    return seg, img


def rows_from(seg, min_area):
    """id_map 要的 rows:每個源的編號與形心。"""
    ids, cnt = np.unique(seg[seg > 0], return_counts=True)
    out = []
    for i, c in zip(ids, cnt):
        if c < min_area:
            continue
        y, x = np.nonzero(seg == i)
        # group 欄位 id_map 只在 by_group=True 時用得到,但 rows 的欄位要齊全,
        # 否則 step4_catalog 的 GROUP_COLOR 會 KeyError。
        out.append(dict(id=int(i), x=float(x.mean()), y=float(y.mean()),
                        group="galaxy"))
    return out, len(ids)


def overview(ns, out_path):
    """14 顆一起看 —— 單張圖回答不了「這套遮罩在整個 mosaic 上一致嗎」。"""
    ncol = 5
    nrow = int(np.ceil(len(ns) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.5 * nrow))
    for a in np.ravel(ax):
        a.axis("off")
    for k, n in enumerate(ns):
        seg, img = load(n)
        a = np.ravel(ax)[k]
        a.axis("on"); a.set_xticks([]); a.set_yticks([])
        # 拉伸和 id_map 一致(asinh,以 99.5 分位的 2% 當軟閾值),
        # 這樣總覽和單張圖看起來是同一份資料。
        v = np.nanpercentile(img[np.isfinite(img) & (img != 0)], 99.5)
        a.imshow(np.arcsinh(img / (0.02 * v)), origin="lower", cmap="gray",
                 vmin=0, vmax=np.arcsinh(1 / 0.02))
        a.contour(seg > 0, levels=[0.5], colors="#1f77b4", linewidths=0.6)
        n_src = int(seg.max())
        a.set_title(f"#{n}   {n_src} src   "
                    f"{100 * (seg > 0).mean():.1f}% masked", fontsize=9)
    fig.suptitle("professor's segmentation on pseudo-r  (blue = source)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="教授 segmentation 的 ID map")
    ap.add_argument("-n", type=int, nargs="+", default=list(range(1, 15)),
                    help="要畫哪幾顆")
    ap.add_argument("--min-area", type=int, default=1,
                    help="只標面積 >= 這個值的源。小到幾個像素的標上去會互相蓋住")
    ap.add_argument("--no-overview", action="store_true")
    args = ap.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    print(f"{'':>4}{'源數':>6}{'源像素':>10}{'佔視野':>8}"
          f"{'最大三塊佔源流量':>18}")
    for n in args.n:
        seg, img = load(n)
        rows, n_all = rows_from(seg, args.min_area)
        out = FIGURES / f"id_map_p{n:02d}.png"
        id_map(seg, img, rows, out, by_group=False)

        # 主源被拆成幾塊,是這批資料反覆出現的問題 —— 順手量出來,
        # 不必再開一支腳本。流量用 pseudo_r 積分(那就是偵測用的量)。
        flux = {int(i): float(np.nansum(np.where(seg == i, img, 0)))
                for i in np.unique(seg) if i > 0}
        tot  = sum(flux.values()) or 1.0
        top  = sorted(flux.values(), reverse=True)[:3]
        print(f"#{n:<3}{n_all:>6}{int((seg > 0).sum()):>10,}"
              f"{100 * (seg > 0).mean():>7.1f}%"
              + "".join(f"{100 * f / tot:>6.1f}%" for f in top))
    print(f"\n每顆一張 -> {FIGURES}")
    if not args.no_overview:
        overview(args.n, FIGURES / "overview.png")


if __name__ == "__main__":
    main()
