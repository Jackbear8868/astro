"""Plot the K sky basis vectors learned by step3, one figure each, all saved into one
folder.

Questions it answers:
  - at which wavelengths the structure of each basis vector lies (compare with
    mean_sky.png)
  - whether the number of vectors is enough (if the later ones are nothing but noise,
    that means going beyond the true degrees of freedom)
  - whether any basis vector is wasted (all the energy concentrated in a single
    channel = a bad pixel, not a sky emission line)
  - the difference between the decompositions: svd's right singular vectors are
    orthonormal (each has norm = 1), while pca's vector 0 is the mean spectrum (not
    normalised) and only the rest are principal components

Written under results/skymodel/evaluation/sky_basis/basis_{method}/:
    mean_sky.png   the mean sky and the continuum, for comparison
    e00.png ...    one figure per basis vector

Usage:
    conda run -n astro python src/skymodel/evaluation/plot_basis.py --basis svd
    conda run -n astro python src/skymodel/evaluation/plot_basis.py --basis svd --ylim -0.1 0.3
    conda run -n astro python src/skymodel/evaluation/plot_basis.py --basis pca --ylim-sky 0 60
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT, pointing_dir


def main():
    ap = argparse.ArgumentParser(description="Plot sky basis vectors learned by step3")
    # The working directory has to be specified. It used to be hard-coded to
    # ne_pointing, that directory has been deleted, and more importantly: once it is
    # hard-coded the figure always shows the same pointing, which does not match "what
    # this pointing's basis looks like".
    ap.add_argument("--work", required=True,
                    help="pointing work directory, e.g. results/skymodel/p01")
    ap.add_argument("--basis", default="svd", help="pca / svd")
    ap.add_argument("-K", type=int, default=30, help="number of basis vectors; must match step3")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="y-axis range for basis plots; auto-scaled per figure if omitted")
    ap.add_argument("--ylim-sky", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="y-axis range for mean_sky plot; defaults to 0 to 99.5 percentile")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    W = ROOT / args.work
    STEP03 = W / "step03"
    wl = np.load(STEP03 / "wavelength.npy")
    B  = np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")
    C  = np.load(STEP03 / "sky_continuum.npy")
    lm = np.load(STEP03 / "line_mask.npy")
    K  = B.shape[0]          # not hard-coded to 10 -- step3's -K can change the count

    # --- diagnostic: the energy concentration exposes basis vectors wasted on bad
    #     pixels ---
    print(f"basis={args.basis}   K={K}   {wl.size} channels "
          f"({wl.min():.1f}-{wl.max():.1f} A air)")
    print(f"line_mask {lm.sum()}/{lm.size} ({100*lm.mean():.1f}%)\n")
    print(f"{'#':>3}{'L2 norm':>11}{'peak':>10}{'neg%':>8}"
          f"{'chan for 90% energy':>20}{'peak wl':>12}")
    print("-" * 66)
    for k in range(K):
        e   = np.sort(B[k] ** 2)[::-1]
        n90 = int(np.searchsorted(np.cumsum(e), 0.9 * e.sum()) + 1)
        i   = int(np.argmax(np.abs(B[k])))
        print(f"{k:>3}{np.linalg.norm(B[k]):>11.4g}{np.abs(B[k]).max():>10.4g}"
              f"{100.0 * (B[k] < 0).mean():>7.0f}%{n90:>16}{wl[i]:>12.1f}")
    print("\n90% energy in only a few channels = that basis vector is dominated by a single bad pixel, not a sky line.")

    # ------- figures: one per basis vector, all into the same folder -------
    ms  = np.load(STEP03 / "mean_sky.npy")
    out = pointing_dir(W.name, "basis")
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(out / name, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)      # they pile up if not closed, eating all memory for big K

    fig, a = plt.subplots(figsize=(15, 4))
    a.plot(wl, ms, lw=0.4, color="0.3", label="mean sky (blank)")
    a.plot(wl, C,  lw=0.8, color="#d62728", label="continuum $C_{sky}$")
    # the sky emission line peaks are tens of times higher than the continuum, so an
    # automatic range would squash the continuum at 18-30 into a single line
    a.set_ylim(*(args.ylim_sky if args.ylim_sky else (0, np.percentile(ms, 99.5))))
    a.set_xlabel("observed wavelength (air) [$\\AA$]")
    a.set_ylabel("flux")
    a.legend(fontsize=8, loc="upper right")
    a.set_title(f"mean sky   basis={args.basis}   K={K}   "
                f"line_mask {lm.sum()}/{lm.size} ({100*lm.mean():.1f}%)")
    save(fig, "mean_sky.png")

    for k in range(K):
        b    = B[k].astype(np.float64)
        flat = b.mean() ** 2 * b.size / (b ** 2).sum()   # energy a constant explains

        fig, a = plt.subplots(figsize=(15, 4))
        a.plot(wl, b, lw=0.5, color="#1f77b4")
        a.axhline(0, color="0.6", lw=0.5)    # zero line: at a glance, does this basis
                                             # cross zero
        # if args.ylim:
        #     a.set_ylim(*args.ylim)
        a.set_ylim(np.min(b), np.max(b)*1.1)
        a.set_xlabel("observed wavelength (air) [$\\AA$]")
        a.set_ylabel(f"$e_{{{k}}}$")
        # the title uses ASCII only -- DejaVu Sans has no CJK glyphs, and Chinese
        # would turn into boxes
        save(fig, f"e{k:02d}.png")

    print(f"\nsaved {K + 1} figures -> {out}")


if __name__ == "__main__":
    main()