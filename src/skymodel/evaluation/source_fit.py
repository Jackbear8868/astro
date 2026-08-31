"""What step4 fits the sources with, and how the fit went.

`--which templates` draws the source templates, all of one set on one figure:

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

`--which scan` draws what those templates then did on one source: step4's redshift scan,
reduced chi2 against z, stars and galaxy side by side. step4 classifies by scanning both
model families over redshift and keeping whichever reaches the lower reduced chi2 on the
same channel set. There is no absolute threshold, so the decision is the two minima of
this figure and their separation is the confidence in it. The families get separate
panels because their scan ranges differ by orders of magnitude -- a stellar radial
velocity against a galaxy redshift -- and on one x axis the stellar scan would be a
single vertical line. The shared y axis is the axis the comparison is made on.

Written into two places, which is deliberate -- see the comment at FIGURES:

    evaluation/templates/
        eigen_<kind>_<mode>_<range>.png
    evaluation/<pointing>/template_fit/
        chi2_scan_id<N>.png

    conda run -n astro python src/skymodel/evaluation/source_fit.py \\
        --which templates --kind star --mode panels
    conda run -n astro python src/skymodel/evaluation/source_fit.py \\
        --which scan --work results/skymodel/p01 --id all
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
from products import Run  # noqa: E402
from spectra import Z_HARO  # noqa: E402
from utils import (DWARF_DIR, load_ascii_template, load_eigen_galaxy,  # noqa: E402
                   load_eigen_qso, load_scan)

# Only the templates go here. They are the same library for every pointing and so
# belong to none of them; a scan is one source of one run and goes beside that run,
# through Run.figdir. Two destinations for one script, on purpose.
FIGURES = EVAL / "templates"

# Where each set lives and how it is read, kept together so a figure cannot claim one
# source while reading another.
EIGEN = {
    "galaxy": (ROOT / "data/eigen_galaxy_Bolton2012.fits", load_eigen_galaxy),
    "qso":    (ROOT / "data/qso_eigen_linear_55732.dat",   load_eigen_qso),
}
KINDS = sorted(EIGEN) + ["star"]

# Harvard order: sorting the filenames alphabetically would interleave the sequence
# (a0v before o5v), and the point of the figure is to read the trend along it. Both
# halves order the stellar library by it, and they have to agree or one template gets
# one colour in the template figure and another in the scan.
SPECTRAL_ORDER = "OBAFGKMLTY"

MUSE_RANGE = (4600.0, 9350.0)     # observed frame, the span the cubes cover
N_SAMPLE = 6000
LABEL_TOP = 0.86        # axes fraction the corner label is kept clear down to

C_GAL = "#e8710a"
C_MIN = "0.35"
LOG_RATIO = 20.0        # span above which the shared y axis switches to log

# --figsize and --dpi are each one option for both halves, so their defaults cannot be:
# the template figure is one wide axis carrying a whole library, the scan is two panels
# holding one source between them.
FIGSIZE_TEMPLATES, FIGSIZE_SCAN = (20, 8), (16, 6)
DPI_TEMPLATES, DPI_SCAN = 200, 180


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


def scan_files(step04, sid):
    """(stars, galaxy) scan arrays for one source, either possibly None.

    One file per branch, named after it, because z means a radial velocity on the star
    side and a redshift on the galaxy side.
    """
    def one(branch):
        try:
            return load_scan(step04, branch, sid)
        except SystemExit:
            return None
    return one("star"), one("galaxy")


def curves(d, spectral=False):
    """(label, z, reduced chi2, colour index) per template, each sorted by z.

    The saved rows are ordered by chi2, not by z, so file order would draw a curve
    jumping back and forth. spectral=True puts the stellar library in Harvard order and
    ties each colour to a place in that sequence, so a template keeps its colour.
    """
    out = []
    for tpl in dict.fromkeys(d["template"].tolist()):
        m = d["template"] == tpl
        z, r = d["z"][m], d["red_chi2"][m]
        o = np.argsort(z)
        out.append([tpl, z[o], r[o], 0])
    if spectral:
        out.sort(key=lambda c: (SPECTRAL_ORDER.find(c[0][0].upper()), c[0]))
        # Spectral class alone: subtype and luminosity class are shared across the
        # library, so printing them repeats one fact in every legend entry.
        for c in out:
            c[0] = c[0][0].upper()
    else:
        out.sort(key=lambda c: np.nanmin(c[2]))
    for i, c in enumerate(out):
        c[3] = i
    return out


def draw(ax, cs, colours, side="left"):
    """One family's scan, with each curve's own minimum marked.

    The legend sits above the axes: any curve can be the lowest, so no corner is
    reliably free and a box inside would cover one of the curves being compared. The
    two panels anchor to opposite ends, or the wider legend lands on the other panel.
    """
    for tpl, z, r, ci in cs:
        col = colours[ci % len(colours)]
        ax.plot(z, r, lw=1.0, color=col, label=tpl)
        k = int(np.nanargmin(r))
        ax.plot(z[k], r[k], "o", ms=4, color=col, zorder=5)
    ax.set_xlabel("z")
    ax.legend(fontsize=9, frameon=False, ncol=len(cs), borderaxespad=0,
              handlelength=1.2, columnspacing=0.9,
              loc=f"lower {side}", bbox_to_anchor=(0 if side == "left" else 1, 1.005))
    ax.grid(alpha=0.2)


def templates(args):
    """The library step4 fits with: one set of curves on one figure."""
    figsize = args.figsize if args.figsize else FIGSIZE_TEMPLATES
    dpi = args.dpi if args.dpi else DPI_TEMPLATES

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
                                 figsize=(figsize[0], figsize[1] * 0.45 * n))
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

        fig, ax = plt.subplots(figsize=figsize)
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
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


def scan(args):
    """How the fit went: reduced chi2 against z, one figure per source."""
    # Not required by argparse, because the template half draws a library that belongs
    # to no pointing; a scan is one run's own and cannot be found without being told.
    if not args.work:
        raise SystemExit("--which scan needs --work, e.g. --work results/skymodel/p01")
    figsize = args.figsize if args.figsize else FIGSIZE_SCAN
    dpi = args.dpi if args.dpi else DPI_SCAN

    run = Run(args.work)
    name = Path(args.work).name
    step04 = Path(args.step04) if args.step04 else run.work / "step04"
    fit_file = step04 / "source_fits.npz"
    if not fit_file.exists():
        raise SystemExit(f"no source_fits.npz under {step04}")
    source_fits = np.load(fit_file)
    # step4 writes the whole chi2 curve only when asked, so its absence is a config
    # to change, not a lost file.
    if not (step04 / "scans_galaxy.npz").exists():
        raise SystemExit(
            f"no scans_*.npz under {step04} -- this figure is the redshift scans "
            "themselves, and step4 keeps them only when source_fit.keep_scans is "
            "true in the config; set it and rerun step4")
    print(f"{name}: {fit_file}")

    ids = source_fits["id"].tolist() if args.id == "all" else [int(args.id)]
    out_dir = Path(args.out_dir) if args.out_dir else run.figdir("template_fit")
    out_dir.mkdir(parents=True, exist_ok=True)
    tab10 = plt.get_cmap("tab10").colors

    for sid in ids:
        d1, d2 = scan_files(step04, sid)
        if d1 is None and d2 is None:
            print(f"  id {sid}: no scan files")
            continue
        k = source_fits["id"].tolist().index(sid)
        won, tpl, z_best = (str(source_fits["group"][k]), str(source_fits["template"][k]),
                            float(source_fits["z"][k]))
        rs, rg = float(source_fits["star_red_chi2"][k]), float(source_fits["gal_red_chi2"][k])
        margin = max(rs, rg) / min(rs, rg)

        fig, (axs, axg) = plt.subplots(1, 2, sharey=True, figsize=figsize,
                                       gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.03})
        lo = hi = None
        if d1 is not None:
            cs = curves(d1, spectral=True)
            draw(axs, cs, tab10)
            lo, hi = (np.nanmin([np.nanmin(c[2]) for c in cs]),
                      np.nanmax([np.nanmax(c[2]) for c in cs]))
        if d2 is not None:
            cg = curves(d2)
            draw(axg, cg, [C_GAL], side="right")
            g = (np.nanmin([np.nanmin(c[2]) for c in cg]),
                 np.nanmax([np.nanmax(c[2]) for c in cg]))
            lo = g[0] if lo is None else min(lo, g[0])
            hi = g[1] if hi is None else max(hi, g[1])
        axs.set_ylabel("reduced $\\chi^2$")

        # The lower minimum is the classification, so the line crosses both panels.
        for ax in (axs, axg):
            ax.axhline(min(rs, rg), lw=0.9, ls="--", color=C_MIN, zorder=1)
        if args.logy == "on" or (args.logy == "auto" and lo and hi / lo > LOG_RATIO):
            axs.set_yscale("log")
        else:
            pad = 0.06 * (hi - lo)
            axs.set_ylim(lo - pad, hi + pad)

        out = out_dir / f"chi2_scan_id{sid}.png"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  id {sid:>3}  {won:<7} {tpl:<7} z={z_best:>7.4f}  "
              f"star {rs:>10.3f}  galaxy {rg:>10.3f}  margin {margin:.2f}x  -> {out.name}")


def main():
    ap = argparse.ArgumentParser(
        description="The source templates step4 fits with, and its redshift scan for one source")
    ap.add_argument("--which", choices=["templates", "scan"], default="templates",
                    help="templates: the eigenspectra and the stellar library step4 "
                         "fits with. scan: step4's reduced chi2 against z for one "
                         "source, stars and galaxy side by side")
    ap.add_argument("--kind", choices=KINDS, default="galaxy",
                    help="--which templates only: which set to draw")
    ap.add_argument("--mode", choices=["raw", "offset", "norm", "panels"], default="raw",
                    help="--which templates only: how the components share the y axis, "
                         "or panels for one row each; see the module docstring")
    ap.add_argument("--muse", action="store_true",
                    help="--which templates only: restrict to the rest wavelengths "
                         "MUSE covers at --z, i.e. the part of each component the fit "
                         "can ever use")
    ap.add_argument("--z", type=float, default=Z_HARO,
                    help="--which templates only: redshift used to convert the MUSE "
                         "range to rest wavelength")
    ap.add_argument("--xlim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="--which templates only: rest wavelength range to draw; "
                         "overrides --muse")
    ap.add_argument("--only-full", action="store_true",
                    help="--which templates only: drop curves that do not span the "
                         "whole drawn range. step4 applies the same kind of test to "
                         "its stellar candidates -- a template with a hole in it "
                         "leaves NaN channels that the solve then discards for every "
                         "spaxel -- so this is roughly the set that is actually "
                         "fitted, though step4 also allows for its redshift search "
                         "margin")
    ap.add_argument("--class-labels", action="store_true",
                    help="--which templates only: name the stellar curves by Harvard "
                         "class letter (g5v -> G) instead of the file name; --kind "
                         "star only, and only unambiguous while the library holds one "
                         "template per class")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="--which templates only: y-axis range; the scan's y axis is "
                         "the comparison itself and is set from the curves")
    ap.add_argument("--step", type=float, default=None,
                    help="--which templates only: vertical step for --mode offset; "
                         "default is one span per component")
    ap.add_argument("--work", default=None,
                    help="--which scan only: pointing work directory, e.g. "
                         "results/skymodel/p01")
    ap.add_argument("--id", default="all", help="--which scan only: source id, or all")
    ap.add_argument("--step04", default=None,
                    help="--which scan only: the step4 directory to draw, e.g. "
                         "results/skymodel/p01/step04/mask_iter2; default is the "
                         "work directory's own step04")
    ap.add_argument("--logy", choices=["auto", "on", "off"], default="auto",
                    help="--which scan only: auto uses log whenever the values span "
                         "more than 20x")
    ap.add_argument("--out-dir", default=None,
                    help="--which scan only: the scan is one figure per source, so it "
                         "names a directory; the templates are one file and take --out")
    ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=None,
                    help=f"default {FIGSIZE_TEMPLATES} for the templates, "
                         f"{FIGSIZE_SCAN} for a scan; under --mode panels the height "
                         f"is per row rather than for the whole figure")
    ap.add_argument("--dpi", type=int, default=None,
                    help=f"default {DPI_TEMPLATES} for the templates, "
                         f"{DPI_SCAN} for a scan")
    ap.add_argument("--out", default=None,
                    help="--which templates only: a scan writes one file per source "
                         "and takes --out-dir instead")
    args = ap.parse_args()

    if args.which == "templates":
        templates(args)
    else:
        scan(args)


if __name__ == "__main__":
    main()
