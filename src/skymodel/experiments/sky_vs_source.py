"""一個源的光譜拆成三塊:天空(含線)、天空連續譜、剩下的源。

回答教授的問題「how much of that is sky continuum?」——
step4b 的圖畫的是 `observed − 1.0·C_sky`,已經扣過天空,所以那條線的高度
容易被誤讀成「源本來就這麼亮」。這張圖把扣掉的東西一起畫出來。

三條線,把算式講完:observed = C_sky + source
    observed (source)  源區域的每 spaxel 平均光譜,含天空 —— 原始輸入
    C_sky              天空連續譜(step3 的迭代結果),實際被扣掉的那條
    source             observed − 1.0 × C_sky,step4b 拿去擬合的那條

blank 區的 mean sky 預設不畫(--with-mean-sky 可加回)——它是 C_sky 的來源而不是
被扣掉的東西,而且連續譜和 C_sky 幾乎重合,疊上去只會混淆。

線性刻度、單一面板。天光線遠高於連續譜,照實畫會把連續譜壓成一條線,所以直接
截斷 —— 這張圖要回答的是「連續譜和源的比例」,不是天光線有多高。

    conda run -n astro python src/skymodel/experiments/sky_vs_source.py --id 35
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import load_line_masks

ROOT    = Path(__file__).resolve().parents[3]
STEP02  = ROOT / "results/skymodel/ne_pointing/step02"
STEP02B = ROOT / "results/skymodel/ne_pointing/step02b"
STEP03  = ROOT / "results/skymodel/ne_pointing/step03"
FIGURES = ROOT / "results/skymodel/figures"


def main():
    ap = argparse.ArgumentParser(description="天空 vs 連續譜 vs 源")
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--window", type=float, nargs=2, default=(4600.0, 6000.0),
                    metavar=("LO", "HI"), help="統計數字用的波長視窗")
    ap.add_argument("--line-mask-iter", type=int, default=1)
    ap.add_argument("--raw-mask", action="store_true")
    ap.add_argument("--s-fix", type=float, default=1.0)
    ap.add_argument("--aperture", action="store_true")
    ap.add_argument("--ylim", type=float, nargs=2, default=(0.0, 100.0),
                    metavar=("LO", "HI"), help="下方線性面板的 y 範圍")
    ap.add_argument("--xlim", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"), help="只畫這一段波長(預設全波段)")
    ap.add_argument("--mark-window", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="標出另一個候選視窗的界線,界線外打底色")
    ap.add_argument("--with-mean-sky", action="store_true",
                    help="把 blank 區的平均天空也畫上去(預設不畫)")
    ap.add_argument("--width", type=float, default=22.0)
    args = ap.parse_args()

    wl   = np.load(STEP03 / "wavelength.npy")
    msky = np.load(STEP03 / "mean_sky.npy")
    Csky = np.load(STEP03 / "sky_continuum.npy")
    line = load_line_masks(STEP03 / "iter_line_mask.npy",
                           cumulative=not args.raw_mask)[args.line_mask_iter - 1]

    src_dir = STEP02B if args.aperture else STEP02
    ids = np.load(src_dir / "object_ids.npy")
    k   = int(np.flatnonzero(ids == args.id)[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        nsp = np.load(src_dir / "object_nspax.npy")[k]
        obs = np.load(src_dir / "object_flux.npy")[k] / nsp
    source = obs - args.s_fix * Csky

    clean = (wl >= args.window[0]) & (wl < args.window[1]) & ~line

    # ---------------- 數字 ----------------
    o, c, s = (float(np.nanmedian(x[clean])) for x in (obs, Csky, source))
    print(f"ID {args.id}   {args.window[0]:.0f}-{args.window[1]:.0f} A, "
          f"{int(clean.sum())} sky-line-free channels   median flux per spaxel")
    print(f"{'':>26}{'median':>10}{'of observed':>13}")
    print("-" * 50)
    print(f"{'observed (source region)':>26}{o:>10.2f}{100.0:>12.0f}%")
    print(f"{'sky continuum C_sky':>26}{c:>10.2f}{100*c/o:>12.1f}%")
    print(f"{'source = observed - C_sky':>26}{s:>10.2f}{100*s/o:>12.1f}%")
    print(f"\n  sky continuum is {c/s:.2f} x the source")
    print(f"  a 1% error on C_sky = {0.01*c:.2f} = {100*0.01*c/s:.1f}% of the source")

    print(f"\n{'range':>14}{'observed':>11}{'C_sky':>9}{'source':>9}{'sky %':>8}")
    for lo, hi in [(4600, 5000), (5000, 5400), (5400, 6000),
                   (6000, 7000), (7000, 9350)]:
        m = (wl >= lo) & (wl < hi) & ~line
        oo, cc, ss = (float(np.nanmedian(x[m])) for x in (obs, Csky, source))
        print(f"{f'{lo}-{hi}':>14}{oo:>11.2f}{cc:>9.2f}{ss:>9.2f}{100*cc/oo:>7.1f}%")

    # ---------------- 圖 ----------------
    # 單一面板、線性刻度。天光線遠高於連續譜,照實畫會把連續譜壓成一條線 ——
    # 直接截斷,因為這張圖要回答的是「連續譜和源的比例」,不是天光線多高。
    fig, a = plt.subplots(figsize=(args.width, 7))
    # 三條就把算式講完了:observed = C_sky + source。
    # blank 區的 mean sky 不在這個算式裡(它是 C_sky 的來源,不是被扣掉的東西),
    # 而且它的連續譜和 C_sky 幾乎重合,疊上去只會讓人分不清哪條是哪條。
    curves = [("observed (source region)", obs, "#1f77b4", 0.5),
              ("C_sky (sky continuum, subtracted)", Csky, "#ff9500", 1.8),
              (f"source = observed − {args.s_fix}·C_sky", source, "0.45", 0.6)]
    if args.with_mean_sky:
        curves.insert(0, ("mean sky (blank spaxels)", msky, "#2ca02c", 0.5))

    for nm, y, col, lw in curves:
        a.plot(wl, y, lw=lw, color=col, label=nm)
    a.axhline(0, color="0.3", lw=0.8)
    xlo, xhi = args.xlim if args.xlim else (wl.min(), wl.max())
    a.set_xlim(xlo, xhi)
    a.set_ylim(*args.ylim)
    # 另一個候選視窗:界線畫成虛線,界線外(會被砍掉的部分)打上淡淡的底色。
    # 重點是被砍掉的那一段長什麼樣,不是界線本身。
    if args.mark_window:
        mlo, mhi = args.mark_window
        for v in (mlo, mhi):
            a.axvline(v, color="#d62728", ls="--", lw=1.6, zorder=4)
        for lo2, hi2 in [(xlo, mlo), (mhi, xhi)]:
            if hi2 > lo2:
                a.axvspan(lo2, hi2, color="#d62728", alpha=0.07, zorder=0)

    a.grid(alpha=0.3)
    a.set_xlabel("wavelength [$\\AA$]")
    a.set_ylabel("flux per spaxel  [$10^{-20}$ cgs]")
    a.legend(fontsize=11, loc="upper left")
    a.set_title(f"ID {args.id}", fontsize=14)

    out = FIGURES / "sky_vs_source"
    out.mkdir(parents=True, exist_ok=True)
    # 檔名編碼 xlim 與 mark-window —— 不編的話換個範圍重跑會靜靜蓋掉上一張。
    nm = f"id{args.id:02d}"
    if args.xlim:
        nm += f"_{args.xlim[0]:.0f}-{args.xlim[1]:.0f}"
    if args.mark_window:
        nm += f"_mark{args.mark_window[0]:.0f}-{args.mark_window[1]:.0f}"
    p = out / f"{nm}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
