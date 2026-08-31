"""The source templates step4 fits with, drawn on one figure.

    --kind galaxy   the Bolton et al. 2012 galaxy eigenspectra, 4 components
    --kind qso      the QSO eigenspectra, 4 components
    --kind star     the stellar library step4 draws its stellar candidates from,
                    one curve per template, ordered by spectral type

The curves come from evaluating the same splines the fit uses, over each spline's own
domain, not from re-reading the files. Whatever spline construction trimmed -- padding
at the ends of the eigenspectra, zero-filled gaps in the stellar files -- is absent
here too, so what is drawn is what the fit can see. Outside its own domain a curve is
NaN and stops. They differ in amplitude by orders of magnitude, so --mode chooses how
they share one pair of axes:

    raw      as they are; the first component dominates and the rest sit on zero
    offset   each shifted down a fixed step; shapes comparable, levels not
    norm     each divided by its own peak |value|; feature sizes comparable too
    panels   no sharing -- one row per component, each with its own y range

    conda run -n astro python src/skymodel/evaluation/plot_eigen.py --kind star --mode panels
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
from utils import (DWARF_DIR, load_ascii_template, load_eigen_galaxy,  # noqa: E402
                   load_eigen_qso)

FIGURES = EVAL / "templates"

# Where each set lives and how it is read, kept together so a figure cannot claim one
# source while reading another.
EIGEN = {
    "galaxy": (ROOT / "data/eigen_galaxy_Bolton2012.fits", load_eigen_galaxy),
    "qso":    (ROOT / "data/qso_eigen_linear_55732.dat",   load_eigen_qso),
}
KINDS = sorted(EIGEN) + ["star"]

# Harvard order: sorting the filenames alphabetically would interleave the sequence
# (a0v before o5v), and the point of the figure is to read the trend along it.
SPECTRAL_ORDER = "OBAFGKMLTY"


def spline_domain(sp):
    """The wavelength range a spline is actually defined over."""
    return float(sp.t[sp.k]), float(sp.t[-sp.k - 1])


def class_label(stem):
    """The Harvard class letter a stellar template file name starts with (g5v -> G)."""
    return stem[0].upper()


def load_curves(kind, class_labels=False):
    """(lo, hi, sample, labels) for one template set.

    sample(lam) returns (len(lam), n_curve), NaN outside a curve's own domain rather
    than extrapolated, so no template is drawn past its data.
    """
    if kind in EIGEN:
        path, loader = EIGEN[kind]
        if not path.exists():
            raise SystemExit(f"{path} does not exist")
        sp = loader(path)
        lo, hi = spline_domain(sp)
        n = np.atleast_2d(sp([lo])).shape[-1]
        print(f"{kind}: {path.name}   rest {lo:.1f}-{hi:.1f} A")
        return lo, hi, (lambda lam: np.asarray(sp(lam, extrapolate=False))), \
            [f"component {j + 1}" for j in range(n)]

    files = sorted(DWARF_DIR.glob("*.dat"),
                   key=lambda f: (SPECTRAL_ORDER.find(f.stem[0].upper()), f.stem))
    if not files:
        raise SystemExit(f"no .dat templates under {DWARF_DIR}")
    sps = [load_ascii_template(f) for f in files]
    doms = [spline_domain(sp) for sp in sps]
    lo, hi = min(d[0] for d in doms), max(d[1] for d in doms)
    print(f"star: {DWARF_DIR.relative_to(ROOT)}   {len(files)} templates, "
          f"rest {lo:.1f}-{hi:.1f} A")
    for f, (a, b) in zip(files, doms):
        print(f"    {f.stem:<8} {a:>8.1f}-{b:<8.1f} A")
    labels = [class_label(f.stem) if class_labels else f.stem for f in files]
    return lo, hi, \
        (lambda lam: np.column_stack([sp(lam, extrapolate=False) for sp in sps])), \
        labels

Z_HARO = 0.0204
MUSE_RANGE = (4600.0, 9350.0)     # observed frame, the span the cubes cover
N_SAMPLE = 6000
LABEL_TOP = 0.86        # axes fraction the corner label is kept clear down to


def main():
    ap = argparse.ArgumentParser(description="Plot the eigenspectra used as source templates")
    ap.add_argument("--kind", choices=KINDS, default="galaxy")
    ap.add_argument("--mode", choices=["raw", "offset", "norm", "panels"], default="raw",
                    help="how the components share the y axis, or panels for one row "
                         "each; see the module docstring")
    ap.add_argument("--muse", action="store_true",
                    help="restrict to the rest wavelengths MUSE covers at --z, "
                         "i.e. the part of each component the fit can ever use")
    ap.add_argument("--z", type=float, default=Z_HARO,
                    help="redshift used to convert the MUSE range to rest wavelength")
    ap.add_argument("--xlim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="rest wavelength range to draw; overrides --muse")
    ap.add_argument("--only-full", action="store_true",
                    help="drop curves that do not span the whole drawn range. step4 "
                         "applies the same kind of test to its stellar candidates -- "
                         "a template with a hole in it leaves NaN channels that the "
                         "solve then discards for every spaxel -- so this is roughly "
                         "the set that is actually fitted, though step4 also allows "
                         "for its redshift search margin")
    ap.add_argument("--class-labels", action="store_true",
                    help="name the stellar curves by Harvard class letter (g5v -> G) "
                         "instead of the file name; --kind star only, and only "
                         "unambiguous while the library holds one template per class")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None)
    ap.add_argument("--step", type=float, default=None,
                    help="vertical step for --mode offset; default is one span per component")
    ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(20, 8))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lo, hi, sample, labels = load_curves(args.kind, args.class_labels)

    if args.xlim:
        x0, x1 = args.xlim
    elif args.muse:
        x0, x1 = (w / (1.0 + args.z) for w in MUSE_RANGE)
    else:
        x0, x1 = lo, hi
    x0, x1 = max(x0, lo), min(x1, hi)
    if not x1 > x0:
        raise SystemExit(f"the requested range lies outside {lo:.1f}-{hi:.1f} A")

    lam = np.linspace(x0, x1, N_SAMPLE)
    F = sample(lam)                              # (N_SAMPLE, n_curve)
    n = F.shape[1]
    print(f"\n  drawing {n} curves over {x0:.1f}-{x1:.1f} A rest"
          + (f"  (MUSE {MUSE_RANGE[0]:.0f}-{MUSE_RANGE[1]:.0f} A at z={args.z})"
             if args.muse and not args.xlim else ""))
    keep = []
    for j in range(n):
        y = F[:, j]
        cov = 100 * np.isfinite(y).mean()
        full = cov > 99.999
        print(f"    {labels[j]:<14} min {np.nanmin(y):>10.4f}   max {np.nanmax(y):>10.4f}"
              f"   peak |value| {np.nanmax(np.abs(y)):>10.4f}"
              + ("" if full else f"   covers {cov:.1f}% of the range"
                 + ("  -> dropped" if args.only_full else "")))
        if full or not args.only_full:
            keep.append(j)
    if args.only_full and len(keep) < n:
        F = F[:, keep]
        labels = [labels[j] for j in keep]
        n = len(keep)

    # tab10 rather than common.qualitative: tab20 alternates a dark and a light shade
    # of the same hue, so four curves would read as two pairs.
    tab10 = plt.get_cmap("tab10").colors
    colours = [tab10[j % len(tab10)] for j in range(n)]

    if args.mode == "panels":
        # One row per component, autoscaling on its own values: they differ by orders
        # of magnitude, and a shared y axis trades one's detail for another's.
        fig, axes = plt.subplots(n, 1, sharex=True,
                                 figsize=(args.figsize[0], args.figsize[1] * 0.45 * n))
        for j, ax in enumerate(axes):
            ax.axhline(0, lw=0.8, color="0.7", zorder=1)
            ax.plot(lam, F[:, j], lw=0.9, color=colours[j], zorder=2)
            ax.set_xlim(x0, x1)
            y = F[:, j]
            if args.ylim:
                ax.set_ylim(*args.ylim)
            else:
                # Every label sits in the same corner, so room is made per row: the
                # axis stretches up until the curve's right-hand end clears LABEL_TOP.
                y0, y1 = ax.get_ylim()
                e = max(1, F.shape[0] // 5)
                top = float(np.nanmax(y[-e:]))
                room = y0 + (top - y0) / LABEL_TOP if top > y0 else y1
                ax.set_ylim(y0, max(y1 + 0.10 * (y1 - y0), room))
            ax.set_ylabel("flux")
            ax.grid(alpha=0.2)
            # In the corner, not a legend: one curve to a row needs no key.
            ax.text(0.996, 0.94, labels[j], transform=ax.transAxes,
                    ha="right", va="top", fontsize=13, color=colours[j])
        axes[-1].set_xlabel("rest wavelength [$\\AA$]")
        fig.subplots_adjust(hspace=0.08)
    else:
        # Each mode is a different y axis, so the label changes with it. nan-aware
        # throughout: a short curve is NaN outside its own domain, and a plain max
        # would poison the scale for all of them.
        if args.mode == "norm":
            Y = F / np.nanmax(np.abs(F), axis=0)
            ylab = "flux / peak |flux|"
        elif args.mode == "offset":
            # One step for every curve, from the largest span, so none overlap.
            step = args.step if args.step else float(
                np.nanmax(np.nanmax(F, axis=0) - np.nanmin(F, axis=0)))
            Y = F - step * np.arange(n)
            ylab = f"flux, offset by {step:.3g} per curve"
        else:
            Y, ylab = F, "flux"

        fig, ax = plt.subplots(figsize=args.figsize)
        ax.axhline(0, lw=0.8, color="0.7", zorder=1)
        for j in range(n):
            ax.plot(lam, Y[:, j], lw=0.9, color=colours[j], zorder=2 + j,
                    label=labels[j])
        ax.set_xlim(x0, x1)
        if args.ylim:
            ax.set_ylim(*args.ylim)
        ax.set_xlabel("rest wavelength [$\\AA$]")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=11, frameon=False, ncol=min(n, 8), loc="lower left",
                  bbox_to_anchor=(0, 1.005), borderaxespad=0)
        ax.grid(alpha=0.2)

    # Mode and range both change what the figure shows, so both go in the name;
    # otherwise a second run silently replaces the first.
    rng = "muse" if (args.muse and not args.xlim) else ("cut" if args.xlim else "full")
    stem = (f"eigen_{args.kind}_{args.mode}_{rng}"
            + ("_onlyfull" if args.only_full else ""))
    out = Path(args.out) if args.out else FIGURES / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
