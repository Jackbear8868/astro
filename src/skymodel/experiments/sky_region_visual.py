"""給眼睛看的視場圖 —— 讓人自己決定學天光的界線該畫在哪裡。

這一支**不挑界線**,只把「星系的光延伸到哪裡」畫到眼睛能直接判讀,
座標軸標好格線,由人自己讀出要切在哪一格。

三件事讓暈看得見
----------------
① 拉伸貼著背景,不貼著峰值
   核心比背景亮四個量級。用峰值定上限的話,暈整片都是同一個顏色。這裡
   上限只取 背景 + 8 sigma —— 核心一定過曝,那正是要的:過曝換來暈的層次。

② 等光度線(isophote)畫在**平滑過**的影像上
   單像素的雜訊會讓等高線碎成一堆圈圈,看不出形狀。先高斯平滑 2 px 再畫,
   線條才代表「這一片區域的亮度」而不是個別像素。
   四條線分別是背景之上 1 / 2 / 5 / 15 sigma —— 1 sigma 那條就是「肉眼可及的
   星系邊界」,它比 SExtractor 的遮罩往外多少,一眼就看得到。

③ sigma 由**沒有源的地方**量
   全圖量的話會被星系本身撐大,等光度線就會被推到太裡面。

#1 / #2 / #14 的底圖
-------------------
教授那三顆的 pseudo-r 沒有扣天空,和其他 11 顆不在同一個基準上。這支優先讀
重建版 `*_pseudo_r_nosky.fits`(從 nosky cube 在 5625-6825 A 等權平均),
沒有才退回教授的原檔。

    conda run -n astro python src/skymodel/experiments/sky_region_visual.py
    conda run -n astro python src/skymodel/experiments/sky_region_visual.py -n 4
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
from utils import main_source_group, scale  # noqa: E402

ROOT    = Path(__file__).resolve().parents[3]
SEGDIR  = ROOT / "data/wsky_seg"
FIGURES = ROOT / "results/skymodel/figures/prof_seg"

LEVELS = (1, 2, 5, 15)          # 背景之上幾個 sigma
LCOL   = ("#2ca02c", "#1f77b4", "#ff7f0e", "#d62728")


def load(n):
    """優先用重建版(已扣天空),沒有才用教授原檔。回傳 (影像, 來源說明)。"""
    p = SEGDIR / f"DATACUBE_FINAL_{n}_pseudo_r_nosky.fits"
    if p.exists():
        return np.asarray(fits.getdata(p), float), "rebuilt from nosky"
    p = SEGDIR / f"DATACUBE_FINAL_{n}_pseudo_r.fits"
    return np.asarray(fits.getdata(p), float), "professor's pseudo-r"


def prep(n, smooth, margin):
    seg = fits.getdata(SEGDIR / f"DATACUBE_FINAL_{n}_seg.fits").astype(int)
    img, src = load(n)
    valid = np.isfinite(img) & (img != 0)
    main, _, _ = main_source_group(seg, np.where(valid, img, np.nan))

    # 背景與 sigma 只從「沒有源、離主源遠、離視野邊緣遠」的地方量
    d_main = ndimage.distance_transform_edt(~main)
    edge   = ndimage.distance_transform_edt(valid)
    ref = valid & (seg == 0) & (edge > 10) & (d_main > np.percentile(d_main[valid], 75))
    if ref.sum() < 500:
        ref = valid & (seg == 0) & (edge > 10)
    bg, sd = float(np.median(img[ref])), float(scale(img[ref]))

    # 平滑要避開無效格,否則邊緣會被拉黑:分子分母各自平滑再相除
    f = np.where(valid, img - bg, 0.0)
    num = ndimage.gaussian_filter(f, smooth)
    den = ndimage.gaussian_filter(valid.astype(float), smooth)
    sm  = np.where(den > 0.2, num / np.maximum(den, 1e-6), np.nan)
    # 視野最外圈不畫等高線 —— 那裡曝光次數少、值偏高又偏吵,bg 與 sigma 本來就
    # 是排除它之後才量的。留著畫的話,會沿整個邊框描出一圈亮線,在縮圖裡看起來
    # 像整個視場都超過門檻,而那不是天上的光,是拼接的邊界。
    sm[edge <= margin] = np.nan
    return seg, img, valid, main, bg, sd, sm


def draw(a, n, seg, img, valid, main, bg, sd, sm, tick, src, hi):
    a.imshow(np.where(valid, img, np.nan), origin="lower", cmap="gray",
             vmin=bg - sd, vmax=bg + hi * sd)
    for lv, c in zip(LEVELS, LCOL):
        a.contour(sm, levels=[lv * sd], colors=c, linewidths=0.9)
    a.contour(main, levels=[0.5], colors="w", linewidths=0.8, linestyles="--")
    ny, nx = seg.shape
    a.set_xticks(np.arange(0, nx + 1, tick))
    a.set_yticks(np.arange(0, ny + 1, tick))
    a.grid(color="#00e5ff", lw=0.45, alpha=0.55)
    a.tick_params(labelsize=7)
    a.set_title(f"#{n}   {nx} x {ny}   bg {bg:.2f}  sigma {sd:.3f}   [{src}]",
                fontsize=9)


def main():
    ap = argparse.ArgumentParser(description="給眼睛判讀界線用的視場圖")
    ap.add_argument("-n", type=int, nargs="+", default=list(range(1, 15)))
    ap.add_argument("--smooth", type=float, default=2.0, help="等光度線的平滑(px)")
    ap.add_argument("--tick", type=int, default=25, help="格線間隔(px)")
    ap.add_argument("--hi", type=float, default=8.0, help="顯示上限 = 背景 + hi*sigma")
    ap.add_argument("--edge-margin", type=float, default=10.0,
                    help="視野邊緣這麼多 px 之內不畫等高線(那裡曝光少、不可信)")
    ap.add_argument("--single", action="store_true", help="另外每顆各存一張大圖")
    args = ap.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    ncol = 4
    nrow = int(np.ceil(len(args.n) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 5.2 * nrow),
                           squeeze=False)
    for a in np.ravel(ax):
        a.axis("off")

    print(f"{'':>4}{'背景':>9}{'sigma':>9}{'1sigma 等光度線的範圍':>26}   底圖")
    for k, n in enumerate(args.n):
        seg, img, valid, main, bg, sd, sm = prep(n, args.smooth, args.edge_margin)
        _, src = load(n)
        a = np.ravel(ax)[k]
        a.axis("on")
        draw(a, n, seg, img, valid, main, bg, sd, sm, args.tick, src, args.hi)

        halo = np.isfinite(sm) & (sm > LEVELS[0] * sd)
        # 只報**主源那一團**的 1 sigma 範圍,不含零星小源
        lab, _ = ndimage.label(halo)
        keep = np.isin(lab, np.unique(lab[main & (lab > 0)]))
        ys, xs = np.nonzero(keep)
        print(f"#{n:<3}{bg:>9.3f}{sd:>9.3f}"
              f"   y {ys.min():>3}-{ys.max():<3}  x {xs.min():>3}-{xs.max():<3}"
              f"   ({100 * keep.sum() / valid.sum():.0f}% of field)   {src}")

        if args.single:
            f1, a1 = plt.subplots(figsize=(9, 9))
            draw(a1, n, seg, img, valid, main, bg, sd, sm, args.tick, src, args.hi)
            a1.set_xlabel("x [px]"); a1.set_ylabel("y [px]")
            a1.legend(handles=[plt.Line2D([], [], color=c, lw=2,
                                          label=f"{lv} sigma above bg")
                               for lv, c in zip(LEVELS, LCOL)]
                      + [plt.Line2D([], [], color="w", lw=2, ls="--",
                                    label="SExtractor main mask")],
                      loc="upper right", fontsize=8, framealpha=0.85)
            f1.tight_layout()
            f1.savefig(FIGURES / f"visual_p{n:02d}.png", dpi=140,
                       bbox_inches="tight")
            plt.close(f1)

    fig.suptitle("where does the galaxy actually end?   isophotes at "
                 + ", ".join(f"{lv} sigma" for lv in LEVELS)
                 + "   (white dashed = SExtractor main mask)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = FIGURES / "visual_overview.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")
    if args.single:
        print(f"每顆一張 -> {FIGURES}/visual_pNN.png")


if __name__ == "__main__":
    main()
