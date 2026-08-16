"""學天光的範圍 —— 一條直線把視場切開,畫在影像上讓人直接看。

**用殘留星系光**定界線:

    ① 看主源遮罩的質心在 y、x 兩軸偏離視場中心多遠,取偏得較嚴重的那一軸
    ② 界線沿那一軸每次外移 5 px 掃過去,每次量一次留下來的 blank 平均比遠場
       高多少(= 殘留星系光)
    ③ 取「還滿足污染 <= max_contam」之中保留最多格的那一條

界線因此有一句可以講清楚的意義:**這條線以外,殘留星系光不超過 max_contam**。
取整到 5 的倍數 —— 半個像素的精度在這裡沒有意義,整數界線好記、好寫進文件。

輸出的界線可以直接餵給 step3 的 --xlim / --ylim。

    conda run -n astro python src/skymodel/experiments/sky_region_bound.py
    conda run -n astro python src/skymodel/experiments/sky_region_bound.py --max-contam 0.1
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import main_source_group  # noqa: E402

ROOT    = Path(__file__).resolve().parents[3]
SEGDIR  = ROOT / "data/wsky_seg"
FIGURES = ROOT / "results/skymodel/evaluation/masking/prof_seg"


def bound(main, base_ok, img, base, max_contam, round_to=5):
    """挑界線 —— 直接用「殘留污染」定,不用「離遮罩多遠」。

    掃描:界線每次往外挪 round_to 個像素,量一次留下來的 blank 平均比遠場
    高多少,取「還滿足污染 <= max_contam」之中**保留最多格**的那一條。這樣界線
    的意義是可以講清楚的一句話 —— 「這條線以外,殘留星系光不超過 max_contam」。

    回傳 (軸, 條件字串, keep 布林圖, 界線座標, 污染)。軸 0 = y,1 = x。
    """
    ny, nx = main.shape
    ys, xs = np.nonzero(main)
    cen = [(ys.mean() - ny / 2) / (ny / 2), (xs.mean() - nx / 2) / (nx / 2)]
    ax  = 0 if abs(cen[0]) >= abs(cen[1]) else 1
    n   = (ny, nx)[ax]
    yy, xx = np.mgrid[0:ny, 0:nx]
    c = (yy, xx)[ax]

    best = None
    for b in range(0, n + 1, round_to):
        keep = (c < b) if cen[ax] > 0 else (c >= b)
        m = base_ok & keep
        if m.sum() < 500:
            continue
        ex = float(np.mean(img[m]) - base)
        if ex <= max_contam and (best is None or m.sum() > best[0].sum()):
            best = (m, b, ex)
    if best is None:                       # 沒有任何界線達標:回報最好的那條
        for b in range(0, n + 1, round_to):
            keep = (c < b) if cen[ax] > 0 else (c >= b)
            m = base_ok & keep
            if m.sum() < 500:
                continue
            ex = float(np.mean(img[m]) - base)
            if best is None or ex < best[2]:
                best = (m, b, ex)
    if best is None:
        return ax, "—", np.zeros_like(main), 0, np.nan
    m, b, ex = best
    op = "<" if cen[ax] > 0 else ">="
    return ax, f"{'yx'[ax]} {op} {b}", m, b, ex


def main():
    ap = argparse.ArgumentParser(description="用主源範圍推出學天光的界線")
    ap.add_argument("-n", type=int, nargs="+", default=list(range(1, 15)))
    ap.add_argument("--max-contam", type=float, default=0.05,
                    help="界線之外容許的殘留星系光(和遠場基線的差,單位同 pseudo-r)")
    ap.add_argument("--round-to", type=int, default=5)
    ap.add_argument("--margin", type=float, default=10.0, help="離視野邊緣多遠")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    ncol = 5
    nrow = int(np.ceil(len(args.n) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.6 * nrow),
                           squeeze=False)
    for a in np.ravel(ax):
        a.axis("off")

    print(f"{'':>4}{'界線':>12}{'blank 格數':>12}{'佔視野':>9}{'殘留污染':>10}")
    for k, n in enumerate(args.n):
        seg = fits.getdata(SEGDIR / f"DATACUBE_FINAL_{n}_seg.fits").astype(int)
        img = np.asarray(fits.getdata(
            SEGDIR / f"DATACUBE_FINAL_{n}_pseudo_r.fits"), float)
        valid = np.isfinite(img) & (img != 0)
        main, _, _ = main_source_group(seg, np.where(valid, img, np.nan))
        edge = ndimage.distance_transform_edt(valid)
        d_main = ndimage.distance_transform_edt(~main)
        # 基線只定一次,用**全視場離主源最遠的一成** —— 若跟著界線走,
        # 每條候選線都拿不同的基線去比,掃描出來的數字就不可比。
        cand = valid & (seg == 0) & (edge > args.margin)
        base = float(np.median(img[cand & (d_main >
                                           np.percentile(d_main[cand], 90))]))
        axis, txt, blank, b, ex = bound(main, cand, img, base,
                                        args.max_contam, args.round_to)

        if blank.sum() < 500:
            # 主源幾乎橫跨整個視場(或遮罩本身有問題)時,單軸切法切不出東西。
            # 這不是程式錯誤,是這條規則對這一顆不適用 —— 要讓它講出來,
            # 不能悄悄回一個空的範圍。
            print(f"#{n:<3}{txt:>12}{int(blank.sum()):>12,}"
                  f"{100 * blank.sum() / valid.sum():>8.1f}%{'切不出範圍':>12}")
        else:
            print(f"#{n:<3}{txt:>12}{int(blank.sum()):>12,}"
                  f"{100 * blank.sum() / valid.sum():>8.1f}%{ex:>10.3f}"
                  + ("" if ex <= args.max_contam else "  <- 未達標"))

        a = np.ravel(ax)[k]
        a.axis("on"); a.set_xticks([]); a.set_yticks([])
        v = np.nanpercentile(img[valid], 99.5)
        a.imshow(np.arcsinh(img / (0.02 * v)), origin="lower", cmap="gray",
                 vmin=0, vmax=np.arcsinh(1 / 0.02))
        a.contour(seg > 0, levels=[0.5], colors="#1f77b4", linewidths=0.5)
        a.contour(main, levels=[0.5], colors="#ff7f0e", linewidths=1.0)
        # 保留區塗一層綠 —— 只畫一條線的話,看的人要自己想「留哪一邊」。
        rgba = np.zeros(seg.shape + (4,))
        rgba[blank] = [0.2, 0.8, 0.3, 0.20]
        a.imshow(rgba, origin="lower")
        (a.axhline if axis == 0 else a.axvline)(b, color="#d62728", lw=1.6)
        a.set_title(f"#{n}   keep {txt}   {100 * blank.sum() / valid.sum():.0f}%",
                    fontsize=9)
    fig.suptitle("sky-learning region: one-axis cut, chosen so the residual "
                 f"galaxy light stays <= {args.max_contam:g}\n"
                 "orange = main source,  blue = all sources,  green = blank used",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.out) if args.out else FIGURES / f"region_bound_c{args.max_contam:g}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
