"""Plot the K sky basis vectors learned by step3, one figure each, all saved into one
folder.

Questions it answers: where each vector's structure lies in wavelength (compare with
mean_sky.png), whether K is enough (later vectors that are pure noise are past the true
degrees of freedom), and whether a vector is wasted (all its energy in one channel is a
bad pixel, not a sky line). The decompositions differ too: svd's right singular vectors
are orthonormal, while pca's vector 0 is the unnormalised mean spectrum and only the
rest are principal components.

Written under results/skymodel/evaluation/sky_basis/basis_{method}/:
    mean_sky.png   the mean sky, for comparison
    e00.png ...    one figure per basis vector

    conda run -n astro python src/skymodel/evaluation/plot_basis.py --basis svd --ylim -0.1 0.3
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# common imports utils, which lives one level up -- without this the script only
# runs when PYTHONPATH already points at src/skymodel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, pointing_dir  # noqa: E402

# Amplitude scale for every basis figure. A percentile rather than the maximum --
# see the comment where span is computed.
SPAN_PCT, SPAN_MULT = 99.9, 1.05


def basis_colour(k):
    """Colour for vector k -- a qualitative palette, not a gradient.

    A gradient would say the index is a quantity; it is only a rank. Distinct hues
    make the same vector recognisable between the stacked figures and its own single
    figure. tab10 repeats every ten, and each vector has its own lane, so a repeat ten
    rows away is never ambiguous.
    """
    return plt.get_cmap("tab10").colors[k % 10]


def main():
    ap = argparse.ArgumentParser(description="Plot sky basis vectors learned by step3")
    # The working directory is required: hard-coded, the figure would always show the
    # same pointing, not "what this pointing's basis looks like".
    ap.add_argument("--work", required=True,
                    help="pointing work directory, e.g. results/skymodel/p01")
    ap.add_argument("--basis", default="svd", help="pca / svd")
    ap.add_argument("-K", type=int, default=30, help="number of basis vectors; must match step3")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="y-axis range for basis plots; defaults to one range "
                         "shared by all K, symmetric about zero")
    ap.add_argument("--ylim-sky", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="y-axis range for mean_sky plot; defaults to the full range")
    ap.add_argument("--overview-width", type=float, default=22,
                    help="width in inches of the stacked figures")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="multiply the traces in the stacked figures; > 1 trades "
                         "overlap between neighbours for visible structure")
    ap.add_argument("--top", type=int, default=5,
                    help="how many leading vectors go into the second stacked "
                         "figure, drawn with room to read them")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    W = ROOT / args.work
    STEP03 = W / "step03"
    wl = np.load(STEP03 / "wavelength.npy")
    B  = np.load(STEP03 / f"sky_line_basis_{args.basis}_K{args.K}.npy")
    # The mask is stored per iteration; the last row is the one step3 finished with.
    lm = np.load(STEP03 / "continuum_iterations.npz")["line_mask"][-1]
    K  = B.shape[0]          # not hard-coded to 10 -- step3's -K can change the count

    # --- diagnostic: energy concentration exposes vectors wasted on bad pixels ---
    print(f"basis={args.basis}   K={K}   {wl.size} channels "
          f"({wl.min():.1f}-{wl.max():.1f} A air)")
    print(f"sky line mask {lm.sum()}/{lm.size} ({100*lm.mean():.1f}%)\n")
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
    ms  = np.load(STEP03 / "blank_mean_spectrum.npy")
    out = pointing_dir(W.name, "basis")
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(out / name, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)      # they pile up if not closed, eating all memory for big K

    fig, a = plt.subplots(figsize=(15, 4))
    a.plot(wl, ms, lw=0.4, color="#6baed6")
    # Whole range by default: the brightest sky lines are tens of times the continuum,
    # and cutting them off hides how much of the sky is line rather than continuum.
    a.set_ylim(*(args.ylim_sky if args.ylim_sky else (0, ms.max() * 1.04)))
    a.set_xlim(wl.min(), wl.max())
    a.set_xlabel("wavelength [$\\AA$]")
    a.set_ylabel("flux")
    save(fig, "mean_sky.png")

    # One range for all K, so the figures compare: the vectors are orthonormal, so
    # their amplitudes already share a scale, and rescaling each figure to its own
    # vector would make a nearly flat one look as structured as the first. Symmetric
    # about zero because a singular vector's sign is arbitrary. The scale is a high
    # percentile, not the maximum -- one bad-pixel channel would flatten all the real
    # structure, so instead the few channels past the range run off the top.
    span = SPAN_MULT * np.percentile(np.abs(B), SPAN_PCT)
    ylim = args.ylim if args.ylim else (-span, span)

    for k in range(K):
        b    = B[k].astype(np.float64)
        flat = b.mean() ** 2 * b.size / (b ** 2).sum()   # energy a constant explains

        fig, a = plt.subplots(figsize=(15, 4))
        a.plot(wl, b, lw=0.5, color=basis_colour(k))
        a.axhline(0, color="0.6", lw=0.5)    # zero line: at a glance, does this basis
                                             # cross zero
        a.set_xlim(wl.min(), wl.max())
        a.set_ylim(*ylim)
        a.set_xlabel("wavelength [$\\AA$]")
        a.set_ylabel(f"$e_{{{k}}}$")
        # labels stay ASCII -- DejaVu Sans has no CJK glyphs
        save(fig, f"e{k:02d}.png")

    # ---- several vectors on one figure ----
    # Stacked with a constant vertical offset rather than overplotted, which would be
    # one unreadable band of colour. The offset is a fixed multiple of the largest
    # excursion in the whole basis and does not depend on which vectors are drawn, so
    # traces keep their relative amplitudes across every stacked figure.
    step = 2.2 * np.percentile(np.abs(B), SPAN_PCT)

    def stacked(ks, name, per_trace, lw):
        """Vectors on one axis, e0 at the top, each in its own lane.

        Traces are clipped to half a lane. An offset set from the maximum would leave
        room for every excursion, but the maximum is one channel of one vector and
        everything else then reads flat. From a percentile the ordinary structure fills
        its lane, and a flat top says "off scale" without looking like the neighbour's.
        """
        half = 0.5 * step
        fig, a = plt.subplots(figsize=(args.overview_width, per_trace * len(ks) + 1.6))
        for row, k in enumerate(ks):
            # e0 at the top: the vectors are ordered by singular value.
            y0 = (len(ks) - 1 - row) * step
            a.axhline(y0, color="0.85", lw=0.5, zorder=1)
            trace = np.clip(args.gain * B[k].astype(np.float64), -half, half)
            a.plot(wl, trace + y0, lw=lw, color=basis_colour(k), zorder=2)
        a.set_xlim(wl.min(), wl.max())
        a.set_ylim(-step, len(ks) * step)
        a.set_yticks([])      # the vertical axis is only an offset, not a quantity
        a.set_xlabel("wavelength [$\\AA$]")
        save(fig, name)

    stacked(range(K), "overview.png", 0.42, 0.5)
    n_top = min(args.top, K)
    stacked(range(n_top), f"top{n_top}.png", 1.5, 0.7)

    print(f"\nsaved {K + 3} figures -> {out}")


if __name__ == "__main__":
    main()