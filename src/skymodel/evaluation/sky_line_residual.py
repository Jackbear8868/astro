"""What the sky-line basis is learned from: mean sky minus the sky continuum.

step3 subtracts the continuum from every blank spaxel and learns K basis vectors
from what is left, so the residual is the sky-line spectrum with the smooth part
taken out. The training matrix itself is per spaxel and is never written to disk
-- it is nz x n_blank, tens of thousands of columns -- but its sigma-clipped mean
is exactly mean_sky - C_sky, because C_sky is one value per channel and shifts
every spaxel by the same amount. That is the curve drawn here.

Being the mean, it shows the lines the basis has to describe but not how much they
vary between spaxels, which is the part the basis exists to capture. The sky line
threshold, also written by step3, is drawn as a band for that.

    conda run -n astro python src/skymodel/evaluation/sky_line_residual.py \\
        --work results/skymodel/p01
    conda run -n astro python src/skymodel/evaluation/sky_line_residual.py \\
        --work results/skymodel/p01 --ylim -10 80 --no-band
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL, ROOT  # noqa: E402

FIGURES = EVAL / "sky_basis"

C_RES, C_BAND, C_ZERO = "#b30000", "#f4a582", "0.45"


def main():
    ap = argparse.ArgumentParser(
        description="mean sky minus continuum -- the residual the sky-line basis is learned from")
    ap.add_argument("--work", required=True,
                    help="pointing work directory, e.g. results/skymodel/p01")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="y range; default is the full range of the residual")
    ap.add_argument("--no-band", action="store_true",
                    help="drop the +/- sigma band and draw the mean residual alone")
    ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(24, 7))
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = ROOT / args.work / "step03"
    wl = np.load(d / "wavelength.npy")
    ms = np.load(d / "blank_mean_spectrum.npy")
    C  = np.load(d / "sky_continuum.npy")
    # The threshold is stored per iteration; the last row is the one step3 finished
    # with.
    sg = np.load(d / "sky_line_threshold_per_iteration.npy")[-1]
    res = ms - C

    print(f"{args.work}: {wl.size} channels {wl.min():.1f}-{wl.max():.1f} A")
    print(f"  residual   median {np.median(res):7.3f}   max {res.max():9.2f}"
          f"   min {res.min():8.2f}")
    print(f"  continuum  median {np.median(C):7.3f}")
    # The residual is not centred on zero and is not meant to be: the continuum is
    # fitted only on the channels that survived the line mask, so on line channels
    # it is an interpolation, and the lines it did not describe are what is left.
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
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
