"""遮罩邊界上的階梯有多高 —— 膨脹要長到多遠才看不見。

階梯從哪來
----------
遮罩**裡面**:模型是 Σa·T + s·C_sky + Σc·L,s 鎖在 s_fix,源的光由模板接走,
而且 A·T **不進 sky_model**,所以留下來。
遮罩**外面**:blank 分支,整個擬合結果都當天空扣掉。

真實的源光在邊界上是連續的,處理方式卻在邊界上跳變,所以殘差必然出現階梯。
**膨脹消不掉它,只能把它推到源夠暗的地方** —— 階梯高度永遠等於該半徑處還剩多少
源光。所以問題是「長到多遠才低於雜訊」,而那是可以量的。

怎麼量
------
對每個 run,取它自己遮罩最外面那一圈(dist 落在 (r-1, r])與外面第一圈
(dist 落在 (r, r+1]),兩者中位數之差就是階梯。

兩個尺度都要報,因為它們給出的答案差很多:

    逐像素   step / sigma_pixel            肉眼看單一像素會不會發現
    整圈     step / (sigma_pixel / sqrt(n))  眼睛沿著整圈積分之後會不會發現

階梯即使遠低於單像素雜訊仍可能看得見,因為它沿整圈同號,眼睛會自動把幾十個
像素積起來。

只取遠離 Haro 11 與視野邊界的源:那兩者各自有更強的效應,會把階梯蓋掉。

    conda run -n astro python src/skymodel/experiments/boundary_step.py
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/ne_pointing/step01"
STEP03  = ROOT / "results/skymodel/ne_pointing/step03"
STEP05  = ROOT / "results/skymodel/ne_pointing/step05"
FIGURES = ROOT / "results/skymodel/figures/edge_oversub"

DEFAULT = [(0, "blank_svdK30_tpl68_s1"),
           (4, "blank_dil4_svdK30_tpl68_s1"),
           (6, "blank_dil6_svdK30_tpl68_s1"),
           (8, "blank_dil8_svdK30_tpl68_s1")]


def main():
    ap = argparse.ArgumentParser(description="量遮罩邊界上的階梯")
    ap.add_argument("--band", type=float, nargs=2, default=(5000, 6000))
    ap.add_argument("--min-haro", type=float, default=25.0,
                    help="只取離 Haro 11 這麼遠的源")
    ap.add_argument("--min-edge", type=float, default=12.0,
                    help="只取離視野邊界這麼遠的源")
    ap.add_argument("--rmax", type=int, default=16)
    ap.add_argument("--figdir", default=str(FIGURES))
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits").astype(int)
    white = fits.getdata(STEP01 / "whitelight.fits")
    wl    = np.load(STEP03 / "wavelength.npy")
    m     = (wl >= args.band[0]) & (wl < args.band[1])

    dist = ndimage.distance_transform_edt(seg == 0)       # 離原始足跡多遠
    edge = ndimage.distance_transform_edt(white != 0)     # 離視野邊界多遠
    dh   = ndimage.distance_transform_edt(seg != 1)       # 離 Haro 11 多遠
    base = (edge > args.min_edge) & (dh > args.min_haro) & (seg != 1)

    prof, sig_px = {}, {}
    for r, run in DEFAULT:
        f = STEP05 / run / "sky_subtracted.fits"
        if not f.exists():
            print(f"跳過(還沒跑){run}")
            continue
        with fits.open(f, memmap=True) as h:
            img = np.nanmean(np.asarray(h[0].data[m], np.float32), axis=0)
        ok = base & np.isfinite(img)
        b  = ok & (seg == 0) & (dist > args.rmax)         # 乾淨的 blank 當雜訊尺度
        sig_px[r] = 1.4826 * np.median(np.abs(img[b] - np.median(img[b])))
        prof[r] = [(np.median(img[ok & (dist > k) & (dist <= k + 1)]),
                    int((ok & (dist > k) & (dist <= k + 1)).sum()))
                   for k in range(args.rmax)]

    print(f"{args.band[0]:.0f}-{args.band[1]:.0f} A   "
          f"只取離 Haro 11 > {args.min_haro:g} px、離視野邊界 > {args.min_edge:g} px 的源")
    print(f"\n{'膨脹 r':>7}{'邊界內':>9}{'邊界外':>9}{'階梯':>9}"
          f"{'/sigma_px':>11}{'整圈 n':>9}{'整圈顯著度':>12}{'blank 少了':>11}")
    n_blank0 = int(((seg == 0) & (white != 0)).sum())
    for r, run in DEFAULT:
        if r not in prof:
            continue
        if r == 0:
            inn = np.median(np.nan_to_num([prof[r][0][0]]))   # 只有外面第一圈可用
            # r=0 的「邊界內」取足跡最外一層
            d_in = ndimage.distance_transform_edt(seg > 0)
            with fits.open(STEP05 / run / "sky_subtracted.fits", memmap=True) as h:
                img = np.nanmean(np.asarray(h[0].data[m], np.float32), axis=0)
            ok = base & np.isfinite(img)
            inn = float(np.median(img[ok & (seg > 0) & (d_in <= 1)]))
            out, n = prof[r][0]
        else:
            inn = prof[r][r - 1][0]
            out, n = prof[r][r]
        step = inn - out
        add  = int(((seg == 0) & (dist <= r)).sum())
        print(f"{r:>7}{inn:>9.3f}{out:>9.3f}{step:>9.3f}"
              f"{step / sig_px[r]:>11.2f}{n:>9,}"
              f"{step / (sig_px[r] / np.sqrt(max(n, 1))):>12.1f}"
              f"{100 * add / n_blank0:>10.1f}%")

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for r, run in DEFAULT:
        if r not in prof:
            continue
        y = [p[0] for p in prof[r]]
        ax.plot(np.arange(args.rmax) + 0.5, y, "o-", ms=4, lw=1.5,
                label=f"dilation r = {r}")
        if r:
            ax.axvline(r, ls=":", lw=1.0, color=ax.lines[-1].get_color())
    s = np.mean(list(sig_px.values()))
    ax.axhspan(-s, s, color="0.85", zorder=0, label="$\\pm\\sigma$ per pixel")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("distance from the original footprint  [px]")
    ax.set_ylabel(f"median residual, {args.band[0]:.0f}-{args.band[1]:.0f} A")
    ax.set_title("the step follows the mask edge — dilation moves it outward, "
                 "it never disappears", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(args.figdir) / "29_step_vs_radius.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
