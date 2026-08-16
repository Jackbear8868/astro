"""reminder ②:blank 區的擬合改用「只在天光線通道上算 chi2」會不會比較好?

四個變體(兩個選項的 2x2),在固定的 10 個 blank spaxel 上比較:

    未加權 + 全部通道     基準(step5 目前的預設)
    chi2   + 全部通道     --blank-chi2
    未加權 + line1        --blank-region line1
    chi2   + line1        兩個都開

「line1」是 estimate_continuum 第一輪的 line mask,只抓到最強的線。限縮只改
「解係數用哪些通道」,天空模型仍在全部通道上求值 —— 每個通道都要扣天空。

畫圖的設計:不要把四條殘差疊在一起(四條雜訊曲線疊起來看不出東西),
改畫「變體 − 基準」。那等於兩個天空模型相減,光子雜訊完全消掉,
剩下的就是這個選項實際改變了什麼,是平滑可讀的曲線。

    conda run -n astro python src/skymodel/experiments/compare_blank_options.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import lsq_linear
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_blank_spectra import FIXED_SPAXELS

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/ne_pointing/step01"
STEP03  = ROOT / "results/skymodel/ne_pointing/step03"
FIGURES = ROOT / "results/skymodel/evaluation/sky_basis"
WSKY    = ROOT / "data/Haro11_NEpointing_wsky.fits"
ESO     = ROOT / "data/Haro11_NEpointing_esonosky.fits"

# 標籤只用 ASCII —— matplotlib 的 DejaVu Sans 沒有中日韓字形,中文會變成方框
VARIANTS = [("unweighted, all",   False, False, "#1f77b4"),
            ("chi2, all",         True,  False, "#ff7f0e"),
            ("unweighted, line1", False, True,  "#2ca02c"),
            ("chi2, line1",       True,  True,  "#9467bd")]


def fit_one(d, var, sky, rows):
    """和 step5 的 fit_blank 同一套:選定的通道上解係數,事後補 s >= 0。

    rows 只縮小「解係數用哪些通道」;回傳的係數之後仍會在全部通道上求值。
    """
    g = np.isfinite(d) & rows
    if var is not None:
        g &= np.isfinite(var) & (var > 0)
    K = sky.shape[0]
    if g.sum() <= K:
        return np.full(K, np.nan)
    A = sky[:, g].T
    y = d[g]
    if var is not None:
        w = 1.0 / np.sqrt(var[g])
        A, y = A * w[:, None], y * w
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    if c[0] < 0:
        lb = np.r_[0.0, np.full(K - 1, -np.inf)]
        c = lsq_linear(A, y, bounds=(lb, np.full(K, np.inf)), method="bvls").x
    return c


def main():
    ap = argparse.ArgumentParser(description="reminder ②:blank 擬合的四個變體")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--mode", choices=["flux", "diff"], default="flux",
                    help="flux = 直接畫四個變體扣完天空後的殘差光譜;"
                         "diff = 畫「變體 − 基準」(兩個天空模型相減,雜訊相消,"
                         "看得出每個選項改變了什麼,但看不到絕對的流量)")
    ap.add_argument("--ylim", type=float, nargs=2, default=None,
                    help="y 範圍;不給則 flux 用 -20 40、diff 用 -6 6")
    ap.add_argument("--panels", type=int, default=5)
    ap.add_argument("--width", type=float, default=22.0)
    ap.add_argument("--height", type=float, default=3.2)
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    wl    = np.load(STEP03 / "wavelength.npy")
    lm    = np.load(STEP03 / "iter_line_mask.npy")[0]
    sky   = np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                       np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")])
    nx    = seg.shape[1]
    idx   = [y * nx + x for y, x in FIXED_SPAXELS[:args.n]]

    with fits.open(WSKY, memmap=True) as h:
        nz = h["DATA"].data.shape[0]
        D  = h["DATA"].data.reshape(nz, -1)[:, idx].astype(np.float64)
        V  = h["STAT"].data.reshape(nz, -1)[:, idx].astype(np.float64)
    E = fits.getdata(ESO).reshape(nz, -1)[:, idx].astype(np.float64)

    allrows = np.ones(nz, bool)
    R = {}
    for name, use_chi2, only_line, _ in VARIANTS:
        rows = lm if only_line else allrows
        R[name] = np.column_stack([
            D[:, j] - sky.T @ fit_one(D[:, j], V[:, j] if use_chi2 else None, sky, rows)
            for j in range(D.shape[1])])

    def stat(X):
        return (np.median(np.abs(X.mean(axis=0))), np.median(X[lm].std(axis=0)),
                np.median(X[~lm].std(axis=0)),
                np.median(np.sqrt((X ** 2).mean(axis=0))),
                np.median(np.abs(np.linalg.lstsq(sky.T, X, rcond=None)[0][0])) * 0)

    print(f"basis={args.basis} K={args.K}   固定 {args.n} 個 blank spaxel"
          f"   line1 {int(lm.sum())}/{nz} 通道\n")
    print(f"{'設定':>16}{'|mean|':>10}{'線內 sigma':>12}{'線外 sigma':>12}{'rms':>9}"
          f"{'贏 ESO':>9}")
    print("-" * 70)
    eo, ei, ex, er, _ = stat(E)
    print(f"{'ESO nosky':>16}{eo:>10.4f}{ei:>12.4f}{ex:>12.4f}{er:>9.4f}")
    print("-" * 70)
    e_rms = np.sqrt((E ** 2).mean(axis=0))
    for name, *_ in VARIANTS:
        o, i_, x, r, _ = stat(R[name])
        win = int((np.sqrt((R[name] ** 2).mean(axis=0)) < e_rms).sum())
        print(f"{name:>16}{o:>10.4f}{i_:>12.4f}{x:>12.4f}{r:>9.4f}{f'{win}/{args.n}':>9}")

    # ---- 圖:一個變體一個資料夾,每張圖把所有波段疊成一疊 ----
    # 四個變體不放在同一張圖 —— 它們用的是同一份資料,雜訊完全相同,
    # 放在一起只會互相干擾。各自和 ESO 比,檔名跨資料夾相同,可直接對照。
    # 顏色固定:ESO 一律紅、我們一律深藍(不隨方法或變體改變),這樣跨圖比較時
    # 眼睛不必重新對應圖例。
    C_OURS, C_ESO = "#0d47a1", "#d62728"
    ylim = args.ylim or ((-20.0, 40.0) if args.mode == "flux" else (-6.0, 6.0))
    base = VARIANTS[0][0]
    root = FIGURES / f"blank_options_{args.basis}_K{args.K}"
    edges = np.linspace(wl[0], wl[-1], args.panels + 1)
    nfig = 0
    for name, _, _, _ in VARIANTS:
        out = root / name.replace(", ", "_").replace(" ", "_")
        out.mkdir(parents=True, exist_ok=True)
        for j_ in range(args.n):
            y0, x0 = FIXED_SPAXELS[j_]
            e = E[:, j_]
            v = (R[name][:, j_] if args.mode == "flux"
                 else R[name][:, j_] - R[base][:, j_])
            fig, axes = plt.subplots(args.panels, 1,
                                     figsize=(args.width,
                                              args.height * args.panels + 0.6))
            axes = np.atleast_1d(axes)
            for b, a in enumerate(axes):
                m = (wl >= edges[b]) & (wl <= edges[b + 1])
                a.axhline(0, color="0.55", lw=0.6, zorder=1)
                if args.mode == "flux":
                    a.plot(wl[m], e[m], lw=0.7, color=C_ESO, alpha=0.85, zorder=2,
                           label=f"ESO nosky          mean {e.mean():+.3f}   "
                                 f"sigma {e.std():.3f}   "
                                 f"rms {np.sqrt((e**2).mean()):.3f}")
                a.plot(wl[m], v[m], lw=0.7, color=C_OURS, alpha=0.9, zorder=3,
                       label=f"{name:<18} mean {v.mean():+.3f}   "
                             f"sigma {v.std():.3f}   "
                             f"rms {np.sqrt((v**2).mean()):.3f}")
                a.set_xlim(edges[b], edges[b + 1])
                a.set_ylim(*ylim)
                a.set_ylabel("flux  [$10^{-20}$ cgs]" if args.mode == "flux"
                             else "difference  [$10^{-20}$ cgs]", fontsize=9)
                a.tick_params(labelsize=9)
                if b == 0:
                    a.legend(fontsize=9, loc="upper right", framealpha=0.9)
                    a.set_title(f"blank spaxel #{j_+1}   (y={y0}, x={x0})   "
                                f"{args.basis} K={args.K}   —   {name}", fontsize=11)
            axes[-1].set_xlabel("wavelength [$\\AA$]", fontsize=10)
            fig.tight_layout(h_pad=0.4)
            fig.savefig(out / f"spaxel_{j_+1:02d}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)
            nfig += 1
    print(f"\nsaved {nfig} figures -> {root}/<option>/")


if __name__ == "__main__":
    main()
