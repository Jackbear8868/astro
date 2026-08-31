"""What step3 learned the sky lines to be, and what it learned them from.

`--which basis` draws the K basis vectors, one figure each plus two stacked figures and
the mean sky to read them against. `--which residual` draws what the decomposition was
handed: the mean sky with the sky continuum taken out. `--which all` draws both.

Questions the vectors answer: where each vector's structure lies in wavelength (compare
with mean_sky.png), whether K is enough (later vectors that are pure noise are past the
true degrees of freedom), and whether a vector is wasted (all its energy in one channel
is a bad pixel, not a sky line). The decompositions differ too: svd's right singular
vectors are orthonormal, while pca's vector 0 is the unnormalised mean spectrum and only
the rest are principal components.

The residual is where those vectors come from. step3 subtracts the continuum from every
blank spaxel and learns K basis vectors from what is left. The per-spaxel training matrix
is never written to disk, but its sigma-clipped mean is exactly mean_sky - C_sky, since
C_sky is one value per channel and shifts every spaxel alike, and that is the curve drawn
here. Being a mean, it shows the lines the basis has to describe but not how much they
vary between spaxels, which is the part the basis exists to capture; step3's threshold is
drawn as a band for that.

Written into two places, which is deliberate -- see the comment at FIGURES:

    evaluation/<pointing>/basis/
        mean_sky.png              the mean sky, for comparison
        e00.png ...               one figure per basis vector
        overview.png, topN.png    the vectors stacked
    evaluation/sky_basis/
        line_residual_<pointing>.png

    conda run -n astro python src/skymodel/evaluation/sky_basis.py \\
        --work results/skymodel/p01 --basis svd --ylim -0.1 0.3
    conda run -n astro python src/skymodel/evaluation/sky_basis.py \\
        --work results/skymodel/p01 --which residual --ylim -10 80 --no-band
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
# common imports utils, which lives one level up -- without this the script only
# runs when PYTHONPATH already points at src/skymodel.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL  # noqa: E402
from products import Run  # noqa: E402

# Only the residual goes here. The basis vectors are about this one pointing and go
# beside its run, through Run.figdir; the residual is one of fourteen curves that are
# read against each other, so it is filed with the other cross-pointing sky figures,
# next to pointing_curves.py's. Two destinations for one script, on purpose.
FIGURES = EVAL / "sky_basis"

# Amplitude scale for every basis figure. A percentile rather than the maximum --
# see the comment where span is computed.
SPAN_PCT, SPAN_MULT = 99.9, 1.05

C_RES, C_BAND, C_ZERO = "#b30000", "#f4a582", "0.45"

# --dpi is one option for both halves, so its default cannot be: the residual is a
# single wide figure carrying every channel, and is written finer than the per-vector
# figures are.
DPI_BASIS, DPI_RESIDUAL = 140, 220


def basis_colour(k):
    """Colour for vector k -- a qualitative palette, not a gradient.

    A gradient would say the index is a quantity; it is only a rank. Distinct hues
    make the same vector recognisable between the stacked figures and its own single
    figure. tab10 repeats every ten, and each vector has its own lane, so a repeat ten
    rows away is never ambiguous.
    """
    return plt.get_cmap("tab10").colors[k % 10]


def vectors(run, args):
    """The basis step3 learned: one figure per vector, the two stacked ones, mean_sky."""
    wl = run.wl
    B  = run.basis(args.basis, args.K)
    # The mask is stored per iteration; run.line_mask is the one step3 finished with.
    lm = run.line_mask
    K  = B.shape[0]          # not hard-coded to 10 -- step3's -K can change the count
    dpi = args.dpi if args.dpi else DPI_BASIS

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
    ms  = run.mean_sky
    out = run.figdir("basis")
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(out / name, dpi=dpi, bbox_inches="tight")
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


def residual(run, args):
    """What the basis was learned from: the mean sky with the continuum taken out."""
    wl = run.wl
    ms = run.mean_sky
    C  = run.continuum
    # The threshold is stored per iteration; the last row is what step3 finished with.
    sg = run.iterations["threshold"][-1]
    res = ms - C
    dpi = args.dpi if args.dpi else DPI_RESIDUAL

    print(f"{args.work}: {wl.size} channels {wl.min():.1f}-{wl.max():.1f} A")
    print(f"  residual   median {np.median(res):7.3f}   max {res.max():9.2f}"
          f"   min {res.min():8.2f}")
    print(f"  continuum  median {np.median(C):7.3f}")
    # The residual is not centred on zero and is not meant to be: the continuum is
    # fitted only on channels that survived the line mask, so elsewhere it interpolates
    # and the lines it did not describe are what is left.
    print(f"  channels above zero {int((res > 0).sum()):,} / {res.size:,}"
          f"  ({100 * (res > 0).mean():.1f}%)")

    fig, a = plt.subplots(figsize=args.figsize)
    if not args.no_band:
        # The threshold is the running median of |mean sky - continuum|, so the band
        # is the typical channel-to-channel scatter, not an error on the mean.
        a.fill_between(wl, -sg, sg, color=C_BAND, alpha=0.55, lw=0,
                       label="$\\pm\\sigma$ (running median of $|$residual$|$)")
    a.axhline(0, lw=0.8, color=C_ZERO)
    a.plot(wl, res, lw=0.6, color=C_RES, label="mean sky $-$ continuum")
    a.set_xlim(wl.min(), wl.max())
    if args.ylim:
        a.set_ylim(*args.ylim)
    a.set_xlabel("wavelength [$\\AA$]")
    a.set_ylabel("flux")
    a.legend(fontsize=11, loc="upper left", bbox_to_anchor=(1.005, 1.0),
             borderaxespad=0, frameon=False)

    name = Path(args.work).name
    out = Path(args.out) if args.out else FIGURES / f"line_residual_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


def main():
    ap = argparse.ArgumentParser(
        description="The sky-line basis step3 learned, and the residual it was learned from")
    # The working directory is required: hard-coded, the figure would always show the
    # same pointing, not "what this pointing's basis looks like".
    ap.add_argument("--work", required=True,
                    help="pointing work directory, e.g. results/skymodel/p01")
    ap.add_argument("--which", choices=["basis", "residual", "all"], default="basis",
                    help="basis: the K vectors, the two stacked figures and mean_sky. "
                         "residual: the curve they were learned from, mean sky minus "
                         "continuum. all: both")
    ap.add_argument("--basis", default="svd", help="--which basis only: pca / svd")
    ap.add_argument("-K", type=int, default=30,
                    help="--which basis only: number of basis vectors; must match step3")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="y-axis range of whichever curves are drawn: the basis "
                         "vectors, defaulting to one range shared by all K and "
                         "symmetric about zero, or the residual, defaulting to its "
                         "full range. The two are in different units, so under "
                         "--which all one value cannot suit both")
    ap.add_argument("--ylim-sky", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="--which basis only: y-axis range for mean_sky plot; "
                         "defaults to the full range")
    ap.add_argument("--overview-width", type=float, default=22,
                    help="--which basis only: width in inches of the stacked figures")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="--which basis only: multiply the traces in the stacked "
                         "figures; > 1 trades overlap between neighbours for visible "
                         "structure")
    ap.add_argument("--top", type=int, default=5,
                    help="--which basis only: how many leading vectors go into the "
                         "second stacked figure, drawn with room to read them")
    ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(24, 7),
                    help="--which residual only: the basis figures size themselves "
                         "from --overview-width and how many vectors are stacked")
    ap.add_argument("--no-band", action="store_true",
                    help="--which residual only: drop the +/- sigma band and draw the "
                         "mean residual alone")
    ap.add_argument("--out", default=None,
                    help="--which residual only: the basis figures are a folder, not "
                         "one file, and always go to the run's own figure directory")
    ap.add_argument("--dpi", type=int, default=None,
                    help=f"default {DPI_BASIS} for the basis figures, "
                         f"{DPI_RESIDUAL} for the residual")
    args = ap.parse_args()

    run = Run(args.work)
    if args.which in ("basis", "all"):
        vectors(run, args)
    if args.which == "all":
        print()     # the two halves each print a heading; run together they read as one
    if args.which in ("residual", "all"):
        residual(run, args)


if __name__ == "__main__":
    main()
