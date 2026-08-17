"""指定座標,比較兩顆(以上)cube 的光譜。

用途:拿我們扣完天空的結果和 ESO 的版本在**同一個位置**對打。box_spectra.py
的方框是依內容自動挑的,問「哪裡出問題」很好用,但要問「這一個位置怎麼樣」
就不行 —— 這支補上那個缺口。

    conda run -n astro python src/skymodel/experiments/compare_spectra.py --at 280 84
    conda run -n astro python src/skymodel/experiments/compare_spectra.py \\
        --at 280 84 --at 120 200 --half 2

預設就是「p01(NE)的最新結果 vs ESO nosky」,所以只給座標就能跑。

兩件事刻意這樣做
----------------
**① 一定檢查兩顆 cube 的網格一致。**  不同 pointing 的視場大小不同(316x337 到
385x327 都有)。網格不同還硬比同一個 (x, y),比到的是天上兩個不同的位置,而圖
看起來完全正常 —— 這種錯不會自己現形,所以必須擋在最前面。

**② 預設 half=0(單一 spaxel)。**  單格很吵,但那是誠實的:使用者問的是「這個
位置」。要平均請自己給 --half,而且圖上會標出用了幾格。

--diff 那一格畫的是**差**,不是比值。兩者在 0 附近的行為完全不同 —— 天空扣完
之後 blank 的值本來就在 0 附近震盪,比值會炸掉。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import spectrum_stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
# 圖與量測值一律寫中央,檔名帶 pointing —— 放在各自的工作區裡的話,
# 要比較幾顆就得開幾個目錄,而且檔名相同排不到一起。
EVAL = ROOT / "results/skymodel/evaluation/subtraction_check"

Z_HARO = 0.0204
LINES  = [("[O II]", 3727.0), ("Hb", 4861.3), ("[O III]", 4958.9),
          ("[O III]", 5006.8), ("Ha", 6562.8), ("[S II]", 6716.4)]
COL    = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def open_cube(path):
    """回傳 (hdulist, hdu)。step5 寫在 primary HDU,ESO 的在名為 DATA 的 HDU。"""
    h = fits.open(path, memmap=True)
    return h, (h["DATA"] if "DATA" in h else h[0])


def spec_at(hdu, x, y, half):
    """(x, y) 附近 (2*half+1)² 格的平均光譜。NaN 的格自動略過。"""
    y0, y1 = max(y - half, 0), y + half + 1
    x0, x1 = max(x - half, 0), x + half + 1
    sub = np.asarray(hdu.data[:, y0:y1, x0:x1], np.float32)
    n = int(np.isfinite(sub).all(axis=0).sum())
    with np.errstate(invalid="ignore"):
        return np.nanmean(sub.reshape(sub.shape[0], -1), axis=1), n


def smooth(a, w):
    """畫圖用的移動平均。表格裡的統計量都沒有平滑過。"""
    if w <= 1:
        return a
    k = np.ones(w) / w
    return (np.convolve(np.nan_to_num(a), k, mode="same")
            / np.convolve(np.isfinite(a).astype(float), k, mode="same"))


def main():
    ap = argparse.ArgumentParser(description="指定座標比較兩顆 cube 的光譜")
    ap.add_argument("--at", type=int, nargs=2, action="append", metavar=("X", "Y"),
                    required=True, help="要看的位置,可以給多次")
    ap.add_argument("--half", type=int, default=0,
                    help="取 (2*half+1)² 格平均。預設 0 = 單一 spaxel")
    ap.add_argument("--cube", action="append", metavar="NAME=PATH",
                    help="要比的 cube。省略 = p01 的最新結果 vs ESO nosky")
    ap.add_argument("--work", default="results/skymodel/p01")
    ap.add_argument("--smooth", type=int, default=1, help="畫圖用的移動平均寬度")
    ap.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--no-diff", action="store_true", help="不畫差值那一格")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    W = ROOT / args.work
    if args.cube:
        cubes = [tuple(c.split("=", 1)) for c in args.cube]
    else:
        run = sorted((W / "step05").glob("*_sfield"))[0]
        cubes = [("ours", str(run / "sky_subtracted.fits")),
                 ("ESO nosky", "data/Haro11_NEpointing_esonosky.fits")]
        print(f"預設比較:{run.name}  vs  ESO nosky")
    wl = np.load(W / "step03/wavelength.npy")

    specs, shapes = {}, {}
    for nm, p in cubes:
        h, hdu = open_cube(ROOT / p if not Path(p).is_absolute() else p)
        shapes[nm] = hdu.data.shape
        specs[nm] = [spec_at(hdu, x, y, args.half) for x, y in args.at]
        h.close()
        print(f"  {nm:<12} {shapes[nm]}  {p}")

    # 網格不同就停 —— 硬比會比到天上兩個不同的位置,而圖看起來完全正常
    uniq = {v for v in shapes.values()}
    if len(uniq) > 1:
        raise SystemExit(f"★ cube 的網格不同 {shapes} —— 同一個 (x, y) 不是同一個天區,"
                         f"不能直接比。要比請先重取樣到同一個 WCS。")
    if specs and len(wl) != shapes[cubes[0][0]][0]:
        raise SystemExit(f"★ 波長軸長度 {len(wl)} 與 cube 的 {shapes[cubes[0][0]][0]} 不符")

    nrow = len(args.at)
    ncol = 1 if args.no_diff else 2
    fig, ax = plt.subplots(nrow, ncol, figsize=(9.5 * ncol, 3.2 * nrow),
                           squeeze=False)
    for r, (x, y) in enumerate(args.at):
        a = ax[r][0]
        a.axhline(0, color="0.5", lw=0.6)
        for k, (nm, _) in enumerate(cubes):
            s, n = specs[nm][r]
            a.plot(wl, smooth(s, args.smooth), lw=0.6, color=COL[k % len(COL)],
                   alpha=0.85, label=nm)
        for _, lam in LINES:                       # 紅移後的位置,天空模型扣不掉
            a.axvline(lam * (1 + Z_HARO), color="0.8", lw=0.5, ls=":")
        a.set_xlim(*(args.xlim if args.xlim else (wl[0], wl[-1])))
        a.grid(alpha=0.25); a.set_ylabel("flux", fontsize=8)
        a.tick_params(labelsize=7)
        # 圖上一律英文 —— matplotlib 預設字型沒有中文字符,中文會變成一格格的豆腐
        a.set_title(f"(x, y) = ({x}, {y})"
                    + (f"   mean of {(2*args.half+1)**2} spaxels" if args.half
                       else "   single spaxel"), fontsize=9)
        if r == 0:
            a.legend(fontsize=8, loc="upper right", framealpha=0.85)

        if not args.no_diff:
            b = ax[r][1]
            base = specs[cubes[0][0]][r][0]
            b.axhline(0, color="0.5", lw=0.6)
            for k, (nm, _) in enumerate(cubes[1:], start=1):
                d = base - specs[nm][r][0]
                b.plot(wl, smooth(d, args.smooth), lw=0.6,
                       color=COL[k % len(COL)], alpha=0.85,
                       label=f"{cubes[0][0]} − {nm}")
            for _, lam in LINES:
                b.axvline(lam * (1 + Z_HARO), color="0.8", lw=0.5, ls=":")
            b.set_xlim(*(args.xlim if args.xlim else (wl[0], wl[-1])))
            b.grid(alpha=0.25); b.tick_params(labelsize=7)
            b.set_title("difference", fontsize=9)
            if r == 0:
                b.legend(fontsize=8, loc="upper right", framealpha=0.85)
    for c in range(ncol):
        ax[-1][c].set_xlabel("wavelength [$\\AA$]")

    print(f"\n{'位置':>12}{'cube':>14}{'mean':>10}{'sigma':>9}{'rms':>9}")
    print("-" * 54)
    for r, (x, y) in enumerate(args.at):
        for nm, _ in cubes:
            st = spectrum_stats(specs[nm][r][0])
            print(f"{f'({x}, {y})':>12}{nm:>14}{st['mean']:>10.3f}"
                  f"{st['sigma']:>9.3f}{st['rms_from_zero']:>9.3f}")

    fig.suptitle(f"spectra at fixed positions   {args.work}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    tag = "_".join(f"{x}-{y}" for x, y in args.at)
    out = (Path(args.out) if args.out
           else EVAL / f"compare_spectra_{tag}_{W.name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
