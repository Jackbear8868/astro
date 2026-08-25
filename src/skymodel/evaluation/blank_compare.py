"""What is left in the blank region after the sky is taken out -- ours against ESO's.

Blank has no source in it, so after a perfect sky subtraction its mean spectrum is
zero: no continuum, no residual sky lines, nothing but noise averaged down by the
number of spaxels. Anything else is what the sky model got wrong. Two pipelines can
therefore be compared directly, because both are being measured against the same right
answer rather than against each other.

    ours   the mean of our sky_subtracted cube over the blank spaxels
    ESO    the mean of the ESO nosky cube over the same spaxels

--diff adds a lower panel with their difference. Since both are the same data minus a
sky,

    (data - our sky) - (data - ESO sky) = ESO sky - our sky

it is the difference between the two sky models as they were actually applied, with
the data cancelled out of it. It is off by default: when both curves are being read
against zero, subtracting one from the other answers a question nobody asked -- how
far each is from zero is already in the panel above.

--mode sky compares the inputs instead: our blank mean sky (step3's mean_sky, the sky
as observed) against wsky - nosky (the sky ESO chose to remove). That answers a
different question -- what the two think the sky *is*, rather than what each leaves
behind -- and its --diff panel is mean(nosky) in blank.

Same spaxels, one procedure
---------------------------
The blank mask is rebuilt from step03/meta.json -- the same seg and the same --xlim /
--ylim / --exclude-box -- so both curves are averaged over identical spaxels with
identical sigma-clipping. A spaxel must be spectrally complete in both cubes: a
channel missing from either has no comparison to make. How many that leaves is
printed, since our cube and ESO's do not carry NaN in the same places.

    conda run -n astro python src/skymodel/evaluation/blank_compare.py --work results/skymodel/p01
    conda run -n astro python src/skymodel/evaluation/blank_compare.py --work results/skymodel/p07 \\
        --ylim -3 3
    conda run -n astro python src/skymodel/evaluation/blank_compare.py --work results/skymodel/p01 \\
        --mode sky
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, pointing_dir  # noqa: E402
from products import spectrum_stats  # noqa: E402

C_OURS, C_ESO, C_RESID, C_ZERO = "#1f77b4", "#e8710a", "#b30000", "0.55"

# What products.spectrum_stats returns, in the order it is shown, with the label used on
# the figure. rms_from_zero keeps its full name: "rms" alone is unambiguous only while
# sigma is next to it, and a number quoted out of the figure loses that.
#
# One format for every row and both blocks. With %g the two columns choose their own
# precision from their own magnitude -- 0.032115 next to 0.5693 -- and the eye reads
# the difference in decimal places as a difference in the numbers.
STATS = [("mean", "mean"), ("sigma", "sigma"), ("skewness", "skewness"),
         ("kurtosis", "kurtosis"), ("rms_from_zero", "rms_from_zero")]
FMT = "{:.4f}"
CHUNK = 200


def blank_mask(work, meta):
    """The blank spaxels step3 used, rebuilt from its meta.

    Not "blank" in general -- *that* run's blank. A comparison made on a different
    set of spaxels than the number it is compared against is not a comparison.
    """
    white = fits.getdata(work / "step01/whitelight.fits")
    seg_p = ROOT / meta["seg"] if not Path(meta["seg"]).is_absolute() else Path(meta["seg"])
    seg = fits.getdata(seg_p)
    valid = white != 0
    m = valid & ~((seg > 0) & valid)
    n_all = int(m.sum())
    yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
    if meta.get("xlim"):
        m &= (xx >= meta["xlim"][0]) & (xx < meta["xlim"][1])
    if meta.get("ylim"):
        m &= (yy >= meta["ylim"][0]) & (yy < meta["ylim"][1])
    if meta.get("exclude_box"):
        y0, y1, x0, x1 = meta["exclude_box"]
        m &= ~((yy >= y0) & (yy <= y1) & (xx >= x0) & (xx <= x1))
    return m, n_all, seg_p


def data_hdu(h):
    return h["DATA"] if "DATA" in h else h[0]


def our_cube(work, pattern=None):
    """Our sky_subtracted for this pointing, and the run directory it came from.

    step6 writes it into step06; the 14-pointing runs wrote it into named directories
    under step05 whose names carry that pointing's template count, so no literal name
    fits all 14 and the pattern is a glob. Without one the newest wins, and the name is
    returned so the caller can print which run the figure is actually about.
    """
    d = Path(work) / "step05"
    runs = [x for x in d.glob(pattern if pattern else "*")
            if x.is_dir() and (x / "sky_subtracted.fits").exists()]
    if pattern is None and (Path(work) / "step06/sky_subtracted.fits").exists():
        runs.append(Path(work) / "step06")
    if not runs:
        return None
    if len(runs) > 1:
        runs.sort(key=lambda x: json.loads((x / "meta.json").read_text()).get("created", ""))
    return runs[-1]


def collapse(x, clip, statistic):
    """Collapse the blank spaxels of a chunk of channels into one spectrum.

    mean -- step3's rule, verbatim: robust centre and spread to decide what to reject,
    but the final average is the mean. The mean is the unbiased estimate of the level,
    which is what "did the sky come out" is asking.

    median -- the level that half the blank spaxels are above. It cannot be moved by
    any minority of spaxels, however extreme, so it answers a different question:
    what a typical blank spaxel looks like, rather than what the region as a whole
    sums to. On a right-skewed cross-spaxel distribution the two are not the same
    number, and the gap between them is itself a measurement.

    Returns (spectrum, rejected, total) -- the last two are 0 for the median, which
    rejects nothing.
    """
    if statistic == "median":
        return np.median(x, axis=1).astype(np.float64), 0, 0
    p16, med, p84 = np.percentile(x, [16, 50, 84], axis=1)
    sg = np.maximum((p84 - p16) / 2, 1e-6)
    keep = np.abs(x - med[:, None]) <= clip * sg[:, None]
    return ((x * keep).sum(axis=1, dtype=np.float64) / keep.sum(axis=1),
            int((~keep).sum()), keep.size)


def robust_range(y, pct=0.5, pad=0.35):
    """A y range set by the spectrum, not by its worst channel.

    Percentiles rather than min/max: a handful of dead or hot channels is enough to
    stretch the axis until everything real is a flat line on zero. Zero is always
    inside the range -- it is the value these spectra are being read against.
    """
    lo, hi = np.percentile(y[np.isfinite(y)], [pct, 100 - pct])
    m = pad * max(hi - lo, 1e-9)
    return min(lo - m, 0.0), max(hi + m, 0.0)


def check_against_step3(work, wl, ours):
    """Does the reconstruction land on step3's own mean_sky?

    Not expected to be zero: step3 kept the spaxels complete in wsky, this keeps the
    ones complete in both cubes, and a different sample gives a different mean. It is
    reported as a fraction of the level, and with where the worst of it is, because on
    a bright sky line a small fractional change is a large number.
    """
    saved = np.load(Path(work) / "step03/mean_sky.npy")
    d = np.abs(ours - saved)
    k = int(np.argmax(d))
    print(f"  vs step03/mean_sky.npy (different sample, see above): "
          f"median |diff| {np.median(d):.4g} on a typical level of {np.median(saved):.2f} "
          f"({100 * np.median(d) / max(abs(np.median(saved)), 1e-9):.3f}%)")
    print(f"    worst channel {wl[k]:.1f} A: {saved[k]:.2f} -> {ours[k]:.2f} "
          f"({100 * d[k] / max(abs(saved[k]), 1e-9):+.2f}%);  "
          f"{int((d > 1).sum())} channels differ by more than 1")


def main():
    ap = argparse.ArgumentParser(
        description="Blank-region mean sky: ours against the sky the ESO pipeline removed")
    ap.add_argument("--work", required=True, help="pointing work directory, e.g. results/skymodel/p01")
    ap.add_argument("--statistic", choices=["mean", "median"], default="mean",
                    help="how the blank spaxels are collapsed into one spectrum per "
                         "channel. mean is step3's sigma-clipped mean; median is the "
                         "level half the spaxels are above")
    ap.add_argument("--mode", choices=["residual", "sky"], default="residual",
                    help="residual: what each pipeline leaves in blank (both should be "
                         "zero). sky: what each thinks the sky is")
    ap.add_argument("--run", default=None,
                    help="glob naming the run directory under step05 that holds our "
                         "sky_subtracted.fits; without it the newest run is used")
    ap.add_argument("--nosky", default=None,
                    help="ESO sky-subtracted cube; by default derived from the wsky "
                         "filename recorded in step03/meta.json")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="y range of the upper panel; default is a robust range in "
                         "residual mode, autoscale in sky mode")
    ap.add_argument("--alpha", type=float, default=0.75,
                    help="opacity of the ESO curve; ours is always drawn solid")
    ap.add_argument("--diff", action="store_true",
                    help="add a lower panel with ours minus ESO")
    ap.add_argument("--resid-ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="y range of the --diff panel; default is a robust range of it")
    ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(22, 9))
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    W = ROOT / args.work
    name = Path(args.work).name
    meta = json.loads((W / "step03/meta.json").read_text())
    wsky = Path(meta["cube"])
    if not wsky.is_absolute():
        wsky = ROOT / wsky
    if args.nosky:
        nosky = Path(args.nosky)
        if not nosky.is_absolute():
            nosky = ROOT / nosky
    else:
        # DATACUBE_FINAL_7.fits -> DATACUBE_FINAL_ESOSKY_7.fits. Derived rather than
        # guessed at by pointing number: the pointing directory name and the cube
        # number are not the same thing in every run.
        mo = re.fullmatch(r"DATACUBE_FINAL_(\d+)\.fits", wsky.name)
        if not mo:
            raise SystemExit(f"cannot derive the ESO cube from {wsky.name}; pass --nosky")
        nosky = ROOT / "data/nosky" / f"DATACUBE_FINAL_ESOSKY_{mo.group(1)}.fits"
    if not nosky.exists():
        raise SystemExit(f"{nosky} does not exist")

    # The two cubes to average, and what each curve is called. In residual mode the
    # left one is our own product; in sky mode it is the raw cube, and the ESO curve
    # becomes a difference, which is why the second entry is a pair.
    if args.mode == "residual":
        run = our_cube(W, args.run)
        if run is None:
            raise SystemExit(
                f"no sky_subtracted.fits under {W}/step05 or {W}/step06 -- "
                f"this pointing has not been through step6")
        cube_a, cube_b = run / "sky_subtracted.fits", nosky
        lab_a, lab_b = "ours", "ESO"
        lab_d = "ours $-$ ESO  $=$  ESO sky $-$ our sky"
        src = run.relative_to(ROOT)
    else:
        run = None
        cube_a, cube_b = wsky, nosky
        lab_a, lab_b = "ours", "ESO"
        lab_d = "ours $-$ ESO  $=$  mean nosky in blank"
        src = wsky.relative_to(ROOT)

    m, n_all, seg_p = blank_mask(W, meta)
    clip = float(meta.get("clip_sigma", 30.0))
    wl = np.load(W / "step03/wavelength.npy")
    nz = wl.size
    print(f"{name}:  mode {args.mode}   ours {src}   ESO {nosky.name}")
    print(f"  seg {seg_p.name}   blank {n_all:,} -> {int(m.sum()):,} used "
          f"(xlim={meta.get('xlim')} ylim={meta.get('ylim')} "
          f"exclude_box={meta.get('exclude_box')})")

    with fits.open(cube_a, memmap=True) as hw, fits.open(cube_b, memmap=True) as hn:
        dw, dn = data_hdu(hw), data_hdu(hn)
        if dw.data.shape != dn.data.shape:
            raise SystemExit(f"cube shapes differ: {dw.data.shape} vs {dn.data.shape}")

        # Pass 1 -- which blank spaxels are complete in both cubes. Read in chunks:
        # the full blank matrix would be nz x n_blank floats, of order a gigabyte.
        cw = np.ones(int(m.sum()), bool)
        cn = np.ones(int(m.sum()), bool)
        for j in range(0, nz, CHUNK):
            a = np.asarray(dw.data[j:j + CHUNK], np.float32)[:, m]
            b = np.asarray(dn.data[j:j + CHUNK], np.float32)[:, m]
            cw &= np.isfinite(a).all(axis=0)
            cn &= np.isfinite(b).all(axis=0)
            print(f"    coverage {min(j + CHUNK, nz)}/{nz}", end="\r", flush=True)
        print(" " * 34, end="\r")
        complete = cw & cn
        # The two counts are reported separately because they are different facts.
        # cw is step3's own sample -- meta's n_blank_used is the mask before this
        # filter, not after it. cn is how much of blank the ESO cube still has a
        # complete spectrum for, and it is the smaller of the two.
        print(f"  spectrally complete: ours {int(cw.sum()):,}   ESO {int(cn.sum()):,}"
              f"   both {int(complete.sum()):,}   of {int(m.sum()):,} blank")

        # Pass 2 -- the two averages, over exactly those spaxels.
        ours = np.empty(nz)
        eso  = np.empty(nz)
        rej = tot = 0
        for j in range(0, nz, CHUNK):
            a = np.asarray(dw.data[j:j + CHUNK], np.float32)[:, m][:, complete]
            b = np.asarray(dn.data[j:j + CHUNK], np.float32)[:, m][:, complete]
            ours[j:j + a.shape[0]], r, t = collapse(a, clip, args.statistic)
            rej += r; tot += t
            # sky mode's ESO curve is a difference of cubes, residual mode's is the
            # ESO cube itself.
            eso[j:j + a.shape[0]], _, _ = collapse(
                a - b if args.mode == "sky" else b, clip, args.statistic)
            print(f"    averaging {min(j + CHUNK, nz)}/{nz}", end="\r", flush=True)
        print(" " * 34, end="\r")
    if tot:
        print(f"  sigma-clip {clip:g}: rejected {rej:,} / {tot:,} ({100 * rej / tot:.6f}%)")
    else:
        print(f"  statistic: per-channel median across spaxels (nothing rejected)")

    # The reconstruction has to land on step3's own answer, or the figure is comparing
    # ESO against something this script invented.
    if args.mode == "sky" and args.statistic == "mean":
        check_against_step3(W, wl, ours)

    resid = ours - eso
    lw = max(len(lab) for _, lab in STATS)
    print("    " + f"{'':<14}" + "".join(f"{lab:>{lw + 2}}" for _, lab in STATS))
    rows = [("ours", ours), ("ESO", eso)] + ([("ours - ESO", resid)] if args.diff else [])
    for lab, y in rows:
        st = spectrum_stats(y)
        print(f"    {lab:<14}"
              + "".join(f"{FMT.format(st[k]):>{lw + 2}}" for k, _ in STATS))

    # One row when the difference is off, so the two curves get the whole canvas
    # instead of the figure keeping an empty half.
    h = args.figsize[1] if args.diff else args.figsize[1] * 0.62
    fig = plt.figure(figsize=(args.figsize[0], h))
    if args.diff:
        gs = fig.add_gridspec(2, 2, width_ratios=[6, 1], height_ratios=[1.5, 1],
                              hspace=0.08, wspace=0.02)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
        sax = fig.add_subplot(gs[:, 1])
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[6, 1], wspace=0.02)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = None
        sax = fig.add_subplot(gs[0, 1])

    # ours thick, ESO thin on top: at the same width the two would coincide and a
    # hidden line could not be told from a line that was never drawn -- which is the
    # question the panel is there to answer.
    if args.mode == "residual":
        # In residual mode zero is the answer both are being measured against, so the
        # line is drawn and the panel is not allowed to autoscale away from it.
        ax0.axhline(0, lw=0.9, color=C_ZERO)
    # ESO underneath, thicker and faded; ours on top, thin and solid. Whichever is
    # drawn last wins where they overlap, and our curve is the one being examined.
    # The alpha is on the lower curve only: on the upper one it would blend our blue
    # with whatever is behind it, exactly where it needs to be legible.
    ax0.plot(wl, eso, lw=1.3, color=C_ESO, alpha=args.alpha, zorder=2, label=lab_b)
    ax0.plot(wl, ours, lw=0.7, color=C_OURS, zorder=4, label=lab_a)
    ax0.set_ylabel("flux")
    # Drawing order -- ESO, then ours -- so the legend reads the same way round as
    # the figure was built. It sits above the axes rather than inside: with the y range
    # tightened onto our curve the panel is full at the top, and a legend in the corner
    # would cover the ESO residuals that the corner is there to show.
    ax0.legend(fontsize=11, loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2,
               frameon=False, borderaxespad=0)
    ax0.grid(alpha=0.2)
    if args.ylim:
        ax0.set_ylim(*args.ylim)
    elif args.mode == "residual":
        ax0.set_ylim(*robust_range(np.concatenate([ours, eso])))
    ax0.set_xlim(wl.min(), wl.max())

    if ax1 is None:
        ax0.set_xlabel("wavelength [$\\AA$]")
    else:
        plt.setp(ax0.get_xticklabels(), visible=False)
        ax1.axhline(0, lw=0.9, color=C_ZERO)
        ax1.plot(wl, resid, lw=0.6, color=C_RESID, label=lab_d)
        ax1.set_xlabel("wavelength [$\\AA$]")
        ax1.set_ylabel("flux")
        ax1.set_xlim(wl.min(), wl.max())
        ax1.legend(fontsize=11, loc="upper left", frameon=False)
        ax1.grid(alpha=0.2)
        ax1.set_ylim(*(args.resid_ylim if args.resid_ylim else robust_range(resid)))

    sax.axis("off")
    # Right-aligned values in a fixed-width column, so the two blocks line up digit
    # for digit and can be compared by eye without reading the numbers.
    w = max(len(FMT.format(v)) for y in (ours, eso, resid)
            for v in spectrum_stats(y).values())

    def block(lab, y, colour, y0):
        st = spectrum_stats(y)
        sax.text(0.02, y0, f"[{lab}]\n" + "\n".join(
            f"{name:<{lw}} = {FMT.format(st[k]):>{w}}" for k, name in STATS),
            transform=sax.transAxes, color=colour, va="top",
            family="monospace", fontsize=10)
    # Same order as the legend, which is the order the two were drawn in. Three lists
    # of the same pair on one figure disagreeing about which comes first is a way to
    # misread a number as belonging to the other curve.
    block(lab_b, eso, C_ESO, 0.98)
    block(lab_a, ours, C_OURS, 0.72)
    if args.diff:
        block("ours - ESO", resid, C_RESID, 0.46)

    # The mode is in the filename because the two modes are two different figures,
    # and the run because a pointing can hold several.
    stem = (f"blank_{args.mode}_{args.statistic}_vs_eso"
            + (f"_{run.name}" if run is not None else "")
            + ("_diff" if args.diff else ""))
    out = Path(args.out) if args.out else pointing_dir(name, "sky") / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
