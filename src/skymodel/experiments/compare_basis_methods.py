"""四種分解方法在同一個 blank spaxel 上的逐條對照。

不疊在一張圖上 —— 四條殘差用的是同一份資料,雜訊完全相同,疊起來只會
看到四條一樣的雜訊。改成一個方法一列,每列都畫上 ESO 當共同的參照,
這樣列與列之間可以直接對齊比較。

波長也切段:3801 個通道畫在一張 22 吋、140 dpi 的圖上是每通道 0.8 px,
通道解析不出來。切 N 段之後每段一張圖,每張圖 4 列 —— 同一段波長、
四個方法上下並排,差異才看得見。

    conda run -n astro python src/skymodel/experiments/compare_basis_methods.py
    conda run -n astro python src/skymodel/experiments/compare_basis_methods.py --n 1
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
STEP01  = ROOT / "results/skymodel/step01"
STEP03  = ROOT / "results/skymodel/step03"
FIGURES = ROOT / "results/skymodel/figures"
WSKY    = ROOT / "data/Haro11_NEpointing_wsky.fits"
ESO     = ROOT / "data/Haro11_NEpointing_esonosky.fits"

COLORS = {"svd": "#1f77b4", "pca": "#2ca02c", "nmf": "#ff7f0e", "rpca": "#9467bd"}


def fit_one(d, sky):
    """和 step5 的 fit_blank 同一套:未加權最小平方,事後補 s >= 0。"""
    g = np.isfinite(d)
    K = sky.shape[0]
    c = np.linalg.lstsq(sky[:, g].T, d[g], rcond=None)[0]
    if c[0] < 0:
        lb = np.r_[0.0, np.full(K - 1, -np.inf)]
        c = lsq_linear(sky[:, g].T, d[g], bounds=(lb, np.full(K, np.inf)),
                       method="bvls").x
    return c


def main():
    ap = argparse.ArgumentParser(description="四種分解方法的逐條對照")
    ap.add_argument("--methods", nargs="+", default=["svd", "pca", "nmf", "rpca"])
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--n", type=int, default=10, help="畫幾個 spaxel")
    ap.add_argument("--panels", type=int, default=5, help="波長切幾段(每段一張圖)")
    ap.add_argument("--ylim", type=float, nargs=2, default=(-20.0, 40.0))
    ap.add_argument("--width", type=float, default=22.0)
    ap.add_argument("--height", type=float, default=3.2, help="每一列的高(吋)")
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    wl    = np.load(STEP03 / "wavelength.npy")
    lm    = np.load(STEP03 / "iter_line_mask.npy")[0]
    C_sky = np.load(STEP03 / "sky_continuum.npy")
    nx    = seg.shape[1]
    idx   = [y * nx + x for y, x in FIXED_SPAXELS[:args.n]]

    with fits.open(WSKY, memmap=True) as h:
        nz = h["DATA"].data.shape[0]
        D  = h["DATA"].data.reshape(nz, -1)[:, idx].astype(np.float64)
    E = fits.getdata(ESO).reshape(nz, -1)[:, idx].astype(np.float64)

    R = {}
    for m in args.methods:
        f = STEP03 / f"sky_basis_{m}_K{args.K}.npy"
        if not f.exists():
            print(f"(缺 {f.name},跳過 {m})")
            continue
        sky = np.vstack([C_sky, np.load(f)])
        R[m] = np.column_stack([D[:, j] - sky.T @ fit_one(D[:, j], sky)
                                for j in range(D.shape[1])])

    print(f"K={args.K}   固定 {args.n} 個 blank spaxel   線內 {int(lm.sum())}/{nz} 通道\n")
    print(f"{'方法':>8}{'|mean|':>10}{'線內 sigma':>12}{'線外 sigma':>12}"
          f"{'rms':>9}{'贏 ESO':>9}")
    print("-" * 62)
    e_rms = np.sqrt((E ** 2).mean(axis=0))
    print(f"{'ESO':>8}{np.median(np.abs(E.mean(0))):>10.4f}"
          f"{np.median(E[lm].std(0)):>12.4f}{np.median(E[~lm].std(0)):>12.4f}"
          f"{np.median(e_rms):>9.4f}")
    print("-" * 62)
    for m, X in R.items():
        r = np.sqrt((X ** 2).mean(axis=0))
        print(f"{m:>8}{np.median(np.abs(X.mean(0))):>10.4f}"
              f"{np.median(X[lm].std(0)):>12.4f}{np.median(X[~lm].std(0)):>12.4f}"
              f"{np.median(r):>9.4f}{f'{int((r < e_rms).sum())}/{args.n}':>9}")

    out = FIGURES / f"basis_methods_K{args.K}"
    out.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(wl[0], wl[-1], args.panels + 1)
    nrow = len(R)
    for j in range(args.n):
        y0, x0 = FIXED_SPAXELS[j]
        for b in range(args.panels):
            m_ = (wl >= edges[b]) & (wl <= edges[b + 1])
            fig, axes = plt.subplots(nrow, 1, sharex=True, sharey=True,
                                     figsize=(args.width, args.height * nrow + 0.6))
            axes = np.atleast_1d(axes)
            for a, (meth, X) in zip(axes, R.items()):
                v = X[:, j]
                a.axhline(0, color="0.55", lw=0.6, zorder=1)
                a.plot(wl[m_], E[m_, j], lw=0.7, color="#d62728", alpha=0.55, zorder=2,
                       label=f"ESO nosky            mean {E[:, j].mean():+.3f}   "
                             f"sigma {E[:, j].std():.3f}   "
                             f"rms {np.sqrt((E[:, j]**2).mean()):.3f}")
                a.plot(wl[m_], v[m_], lw=0.8, color=COLORS.get(meth, "0.2"),
                       alpha=0.9, zorder=3,
                       label=f"{meth:<6} K={args.K}      mean {v.mean():+.3f}   "
                             f"sigma {v.std():.3f}   "
                             f"rms {np.sqrt((v**2).mean()):.3f}")
                a.set_xlim(edges[b], edges[b + 1])
                a.set_ylim(*args.ylim)
                a.set_ylabel("flux  [$10^{-20}$ cgs]", fontsize=9)
                a.legend(fontsize=9, loc="upper right", framealpha=0.9, ncol=2)
                a.tick_params(labelsize=9)
            axes[0].set_title(f"blank spaxel #{j+1}   (y={y0}, x={x0})   K={args.K}   "
                              f"band {b+1}/{args.panels}   "
                              f"{edges[b]:.0f} - {edges[b+1]:.0f} A", fontsize=11)
            axes[-1].set_xlabel("wavelength [$\\AA$]", fontsize=10)
            fig.tight_layout(h_pad=0.3)
            fig.savefig(out / f"spaxel_{j+1:02d}_band{b+1}.png", dpi=140,
                        bbox_inches="tight")
            plt.close(fig)
    print(f"\nsaved {args.n * args.panels} figures -> {out}")


if __name__ == "__main__":
    main()
