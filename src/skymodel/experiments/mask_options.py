"""四種 step3 訓練遮罩並排 —— 到底各自留下哪些 spaxel。

要比較的四個選項
----------------
    ① x0-170                          現行的做法。避開 Haro 11 的暈。
    ② x0-170 + 離小源 > 15 px         再排除其他源的 PSF 翼。
    ③ ② + 離 Haro 11 > 50 px          補掉 x=170 附近仍在暈內的一小塊。
    ④ 純距離式:小源 15 + Haro 50       不用 x 切,只用「離源多遠」。

兩個半徑各自要排除的東西(1 px = 0.2 角秒):
    小源的 PSF 翼      compact 源周圍還留著 PSF 翼的一圈
    Haro 11 的延展暈   暈還沒回到遠場基線的一圈

「可學習」的定義與 step3 一致:視場內、非源、而且**光譜 100% 完整**。step3 會把
部分覆蓋的 spaxel 丟掉(大氣色散讓有效視野逐波長平移,視野邊緣有一圈只在部分
波長有資料),所以拿 white != 0 去數會高估。

    conda run -n astro python src/skymodel/experiments/mask_options.py
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
CUBE    = ROOT / "data/Haro11_NEpointing_wsky.fits"
FIGURES = ROOT / "results/skymodel/figures/s_field"


def complete_mask(cube, cache):
    """每個 spaxel 的光譜是不是 3801 個通道都有資料。掃一次 cube,存起來重用。"""
    if cache.exists():
        return fits.getdata(cache).astype(bool)
    with fits.open(cube, memmap=True) as h:
        nz = h["DATA"].header["NAXIS3"]
        ny, nx = h["DATA"].header["NAXIS2"], h["DATA"].header["NAXIS1"]
        good = np.ones(ny * nx, bool)
        for j in range(0, nz, 200):
            good &= np.isfinite(
                np.asarray(h["DATA"].data[j:j+200], np.float32)).all(axis=0).ravel()
    out = good.reshape(ny, nx)
    fits.writeto(cache, out.astype(np.uint8), overwrite=True)
    print(f"  (掃描完成,存到 {cache.name} 供重用)")
    return out


def main():
    ap = argparse.ArgumentParser(description="四種訓練遮罩並排")
    ap.add_argument("--xcut", type=int, default=170)
    ap.add_argument("--r-oth", type=float, default=15.0)
    ap.add_argument("--r-haro", type=float, default=50.0)
    ap.add_argument("--figdir", default=str(FIGURES))
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits").astype(int)
    white = fits.getdata(STEP01 / "whitelight.fits")
    ny, nx = seg.shape
    print("檢查光譜完整性…")
    comp = complete_mask(CUBE, STEP01 / "complete.fits")

    valid = white != 0
    blank = valid & (seg == 0) & comp            # step3 真正能用的樣本
    # distance_transform_edt 對每個 True 算「離最近的 False 多遠」
    d_oth = ndimage.distance_transform_edt(~((seg > 0) & (seg != 1)))
    d_har = ndimage.distance_transform_edt(seg != 1)
    xx    = np.mgrid[0:ny, 0:nx][1]

    hx = np.nonzero(seg == 1)[1]
    print(f"blank 且光譜完整:{int(blank.sum()):,}")
    print(f"Haro 11 足跡 x = {hx.min()}–{hx.max()},離 x={args.xcut} 最近 "
          f"{hx.min() - args.xcut} px")

    # 圖上的字一律英文 —— matplotlib 的預設字型沒有中文字符,會畫成方框。
    opts = [
        (f"(1)  x < {args.xcut}\ncurrent (professor)", blank & (xx < args.xcut)),
        (f"(2)  x < {args.xcut}   +   > {args.r_oth:g} px from compact sources",
         blank & (xx < args.xcut) & (d_oth > args.r_oth)),
        (f"(3)  x < {args.xcut}   +   {args.r_oth:g} px compact   "
         f"+   > {args.r_haro:g} px from Haro 11",
         blank & (xx < args.xcut) & (d_oth > args.r_oth) & (d_har > args.r_haro)),
        (f"(4)  distance only:  {args.r_oth:g} px compact  +  "
         f"{args.r_haro:g} px Haro 11\n(no x cut)",
         blank & (d_oth > args.r_oth) & (d_har > args.r_haro)),
    ]
    n_all = int(blank.sum())
    print(f"\n{'選項':>52}{'樣本':>9}{'佔全部':>9}")
    for t, m in opts:
        print(f"{t.replace(chr(10), ' '):>52}{int(m.sum()):>9,}"
              f"{100*m.sum()/n_all:>8.1f}%")

    # 0 視場外  1 源  2 blank 被排除  3 blank 保留
    cmap = ListedColormap(["#ffffff", "#3b3b3b", "#f7c9c9", "#1a9e8f"])
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 12.6))
    for a, (title, m) in zip(ax.ravel(), opts):
        img = np.zeros((ny, nx), int)
        img[valid & (seg > 0)] = 1
        img[blank] = 2
        img[m] = 3
        a.imshow(img, origin="lower", cmap=cmap, vmin=-0.5, vmax=3.5,
                 interpolation="nearest")
        # 為什麼被排除,用兩條輔助線標出來
        a.axvline(args.xcut, color="#1f4fd8", ls="--", lw=1.3)
        a.contour(d_har, levels=[args.r_haro], colors="#ff7f0e", linewidths=1.3)
        a.contour(seg == 1, levels=[0.5], colors="k", linewidths=0.8)
        a.set_title(f"{title}\n{int(m.sum()):,} spaxels "
                    f"({100 * m.sum() / n_all:.1f}% of {n_all:,})", fontsize=10)
        a.set_xticks([]); a.set_yticks([])
    handles = [Patch(fc="#1a9e8f", label="kept — the sky basis learns from these"),
               Patch(fc="#f7c9c9", label="blank but excluded"),
               Patch(fc="#3b3b3b", label="source footprint (never used)"),
               plt.Line2D([], [], color="#1f4fd8", ls="--",
                          label=f"x = {args.xcut}"),
               plt.Line2D([], [], color="#ff7f0e",
                          label=f"{args.r_haro:g} px from Haro 11 "
                                f"(where its halo fades into the background)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("which spaxels does the sky model learn from?   "
                 "(blank spaxels with complete spectra — what step3 actually uses)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    out = Path(args.figdir); out.mkdir(parents=True, exist_ok=True)
    p = out / "10_mask_options.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
