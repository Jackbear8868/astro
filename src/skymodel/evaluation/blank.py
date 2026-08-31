"""What is left in the blank region after the sky is taken out -- ours against ESO's.

Blank has no source in it, so after a perfect sky subtraction its mean spectrum is
zero: no continuum, no residual sky lines, only noise averaged down. Anything else is
what the sky model got wrong, and both pipelines are measured against that same answer.

    ours   the mean of our sky_subtracted cube over the blank spaxels
    ESO    the mean of the ESO nosky cube over the same spaxels

That measurement answers two questions, and `--view` says which one is being asked:
how big the residual is, and whether a residual that size is noise. The first needs
nothing but the two curves; the second needs what a single spaxel looks like, which is
the same spaxels read a second way.

--view mean -- how big is it
----------------------------
The two mean spectra, drawn together and read against zero.

--diff adds a lower panel with their difference. Both curves are the same data minus a
sky, so it is (data - our sky) - (data - ESO sky) = ESO sky - our sky, the two sky
models differenced with the data cancelled out. Off by default: read against zero, the
panel above already says how far each is.

--mode sky compares the inputs instead -- our blank mean sky (step3's
blank_mean_spectrum, the sky as observed) against wsky - nosky, the sky ESO chose to
remove -- and its --diff panel is mean(nosky) in blank.

--view floor -- is it noise, or is it wrong
-------------------------------------------
A mean over tens of thousands of blank spaxels looks flat whether the subtraction was
good or bad, as long as the error is random: averaging N spaxels divides random scatter
by sqrt(N) and leaves a systematic offset untouched. So each panel draws the mean
residual against its own noise floor:

    scatter   the spread across blank spaxels within one channel, (p84 - p16) / 2 --
              what a single spaxel actually looks like.
    floor     scatter / sqrt(N). Inside this band the channel's residual is
              indistinguishable from the same spaxels averaged with no systematic
              error; outside it, the same mistake was made in every spaxel.

The band is what makes the two pipelines comparable: rms and mean both shrink with the
number of spaxels averaged, so they say as much about the size of the blank region as
about the sky model, while the ratio to the floor does not. The top panel draws the two
scatters together, so how much of the separation below is systematic can be judged.

Same spaxels, one procedure
---------------------------
The blank mask is rebuilt from step03/meta.json, the same seg and the same --xlim /
--ylim / --exclude-box, so both curves are averaged over identical spaxels with
identical sigma-clipping. A spaxel must be spectrally complete in both cubes; how many
that leaves is printed, because the two do not carry NaN in the same places.

    conda run -n astro python src/skymodel/evaluation/blank.py --work results/skymodel/p01
    conda run -n astro python src/skymodel/evaluation/blank.py --work results/skymodel/p07 \\
        --ylim -3 3
    conda run -n astro python src/skymodel/evaluation/blank.py --work results/skymodel/p01 \\
        --mode sky
    conda run -n astro python src/skymodel/evaluation/blank.py --work results/skymodel/p01 \\
        --view floor --n-floor 3 --ylim -0.4 0.4

This is what blank_compare.py and blank_noise_floor.py were: the same blank mask, the
same two cubes, the same spaxels complete in both. One asked how far from zero the
mean lands, the other whether that distance is more than averaging noise leaves behind.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, data_hdu  # noqa: E402
from config import resolve_path  # noqa: E402
from products import Run, latest_run, spectrum_stats  # noqa: E402
from spectra import robust_range  # noqa: E402
from zones import blank_mask  # noqa: E402

C_OURS, C_ESO, C_RESID = "#1f77b4", "#e8710a", "#b30000"
# Zero, and the noise-floor band. Zero is drawn darker in the floor view because there
# it lies on top of the band; at the mean view's grey the two greys would read as one.
C_ZERO, C_ZERO_BAND, C_BAND = "0.55", "0.45", "0.72"

# What products.spectrum_stats returns, in display order, with the figure's label.
# rms_from_zero keeps its full name: "rms" alone is unambiguous only while sigma is
# next to it. One format for every row and both blocks -- with %g each column would
# pick its own precision, and a gap in decimal places reads as a gap in the numbers.
STATS = [("mean", "mean"), ("sigma", "sigma"), ("skewness", "skewness"),
         ("kurtosis", "kurtosis"), ("rms_from_zero", "rms_from_zero")]
FMT = "{:.4f}"
CHUNK = 200
# The two views do not draw the same figure, so neither can hold the other's size: the
# mean view has one panel, or two with --diff, and the floor view always has three.
FIGSIZE = {"mean": (22, 9), "floor": (22, 11)}


def eso_cube(pointing, given):
    """ESO's cube as the run's config names it. Deriving it from the wsky filename,
    which is what this did, only ever finds data kept inside the repository."""
    nosky = resolve_path(given) if given else pointing.nosky
    if not nosky.exists():
        raise SystemExit(f"{nosky} does not exist")
    return nosky


def collapse(x, clip, statistic):
    """Collapse the blank spaxels of a chunk of channels into one spectrum.

    mean -- step3's rule verbatim: a robust centre and spread decide what to reject, but
    the average is the mean, the unbiased estimate of the level the question asks for.

    median -- the level half the blank spaxels are above, which no minority can move
    however extreme. It says what a typical blank spaxel looks like rather than what
    the region sums to; on a skewed distribution the gap between them is a measurement.

    Returns (spectrum, rejected, total); the last two are 0 for the median.
    """
    if statistic == "median":
        return np.median(x, axis=1).astype(np.float64), 0, 0
    p16, med, p84 = np.percentile(x, [16, 50, 84], axis=1)
    sg = np.maximum((p84 - p16) / 2, 1e-6)
    keep = np.abs(x - med[:, None]) <= clip * sg[:, None]
    return ((x * keep).sum(axis=1, dtype=np.float64) / keep.sum(axis=1),
            int((~keep).sum()), keep.size)


def check_against_step3(run, wl, ours):
    """Does the reconstruction land on step3's own blank mean spectrum?

    Not expected to be zero: step3 kept the spaxels complete in wsky, this keeps those
    complete in both cubes, and a different sample gives a different mean. Reported as a
    fraction and with the worst channel, a bright sky line making a small fraction large.
    """
    saved = run.mean_sky
    d = np.abs(ours - saved)
    k = int(np.argmax(d))
    print(f"  vs step03/blank_mean_spectrum.npy (different sample, see above): "
          f"median |diff| {np.median(d):.4g} on a typical level of {np.median(saved):.2f} "
          f"({100 * np.median(d) / max(abs(np.median(saved)), 1e-9):.3f}%)")
    print(f"    worst channel {wl[k]:.1f} A: {saved[k]:.2f} -> {ours[k]:.2f} "
          f"({100 * d[k] / max(abs(saved[k]), 1e-9):+.2f}%);  "
          f"{int((d > 1).sum())} channels differ by more than 1")


def channel_stats(hdu, mask, keep, nz):
    """Per channel: the mean across the kept blank spaxels, and their robust scatter.

    (p84 - p16) / 2 rather than the standard deviation: a handful of bad spaxels would
    set the standard deviation, and a floor built from it would be so wide that nothing
    could ever leave the band.
    """
    mean = np.empty(nz)
    scat = np.empty(nz)
    for j in range(0, nz, CHUNK):
        x = np.asarray(hdu.data[j:j + CHUNK], np.float32)[:, mask][:, keep]
        mean[j:j + x.shape[0]] = x.mean(axis=1, dtype=np.float64)
        p16, p84 = np.percentile(x, [16, 84], axis=1)
        scat[j:j + x.shape[0]] = (p84 - p16) / 2
        print(f"    {min(j + CHUNK, nz)}/{nz}", end="\r", flush=True)
    print(" " * 24, end="\r")
    return mean, scat


def view_mean(args, pointing, meta):
    """The mean residual spectrum of each pipeline, drawn against zero."""
    W = pointing.work
    name = pointing.name
    wsky = pointing.wsky
    nosky = eso_cube(pointing, args.nosky)

    # The two cubes to average, and what each curve is called. In sky mode the left one
    # is the raw cube and the ESO curve becomes a difference.
    if args.mode == "residual":
        run = latest_run(W, "sky_subtracted.fits", "step06", args.run)
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
    wl = pointing.wl
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
        # Two counts, two facts: cw is step3's own sample (meta's n_blank_used is the
        # mask before this filter), cn is what the ESO cube still has complete.
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
            # In sky mode the ESO curve is a difference of cubes, not the ESO cube.
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
        check_against_step3(pointing, wl, ours)

    resid = ours - eso
    lw = max(len(lab) for _, lab in STATS)
    print("    " + f"{'':<14}" + "".join(f"{lab:>{lw + 2}}" for _, lab in STATS))
    rows = [("ours", ours), ("ESO", eso)] + ([("ours - ESO", resid)] if args.diff else [])
    for lab, y in rows:
        st = spectrum_stats(y)
        print(f"    {lab:<14}"
              + "".join(f"{FMT.format(st[k]):>{lw + 2}}" for k, _ in STATS))

    # One row when the difference is off, so the two curves get the whole canvas.
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

    if args.mode == "residual":
        # In residual mode zero is the answer both are measured against, so it is drawn.
        ax0.axhline(0, lw=0.9, color=C_ZERO)
    # ESO underneath, thicker and faded; ours on top, thin and solid, since whatever is
    # drawn last wins where they overlap and equal widths would hide one curve outright.
    # The alpha is on the lower curve only, or our blue would blend with the background.
    ax0.plot(wl, eso, lw=1.3, color=C_ESO, alpha=args.alpha, zorder=2, label=lab_b)
    ax0.plot(wl, ours, lw=0.7, color=C_OURS, zorder=4, label=lab_a)
    ax0.set_ylabel("flux")
    # The legend follows the drawing order, and sits above the axes rather than inside:
    # the panel is full at the top, where a legend would cover the ESO residuals.
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
    # Right-aligned in a fixed-width column, so the blocks line up digit for digit and
    # can be compared by eye without reading the numbers.
    w = max(len(FMT.format(v)) for y in (ours, eso, resid)
            for v in spectrum_stats(y).values())

    def block(lab, y, colour, y0):
        st = spectrum_stats(y)
        sax.text(0.02, y0, f"[{lab}]\n" + "\n".join(
            f"{name:<{lw}} = {FMT.format(st[k]):>{w}}" for k, name in STATS),
            transform=sax.transAxes, color=colour, va="top",
            family="monospace", fontsize=10)
    # Same order as the legend and the drawing: three lists of the same pair in
    # different orders is a way to read a number as the other curve's.
    block(lab_b, eso, C_ESO, 0.98)
    block(lab_a, ours, C_OURS, 0.72)
    if args.diff:
        block("ours - ESO", resid, C_RESID, 0.46)

    # The mode is in the filename because the two modes are two different figures,
    # and the run because a pointing can hold several.
    stem = (f"blank_{args.mode}_{args.statistic}_vs_eso"
            + (f"_{run.name}" if run is not None else "")
            + ("_diff" if args.diff else ""))
    out = Path(args.out) if args.out else pointing.figdir("sky") / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


def view_floor(args, pointing, meta):
    """The same means, per channel, against the noise floor of their own spaxels."""
    W = pointing.work
    name = pointing.name
    run = latest_run(W, "sky_subtracted.fits", "step06", args.run)
    if run is None:
        raise SystemExit(f"no sky_subtracted.fits under {W}/step05 or {W}/step06")
    nosky = eso_cube(pointing, args.nosky)

    m, n_all, _ = blank_mask(W, meta)
    wl = pointing.wl
    nz = wl.size
    print(f"{name}:  ours {run.relative_to(ROOT)}   ESO {nosky.name}")

    with fits.open(run / "sky_subtracted.fits", memmap=True) as ha, \
         fits.open(nosky, memmap=True) as he:
        da, de = data_hdu(ha), data_hdu(he)
        ca = np.ones(int(m.sum()), bool)
        ce = np.ones(int(m.sum()), bool)
        for j in range(0, nz, CHUNK):
            ca &= np.isfinite(np.asarray(da.data[j:j + CHUNK], np.float32)[:, m]).all(axis=0)
            ce &= np.isfinite(np.asarray(de.data[j:j + CHUNK], np.float32)[:, m]).all(axis=0)
            print(f"    coverage {min(j + CHUNK, nz)}/{nz}", end="\r", flush=True)
        print(" " * 30, end="\r")
        keep = ca & ce
        N = int(keep.sum())
        print(f"  blank {n_all:,} -> {int(m.sum()):,} in the step3 mask -> "
              f"{N:,} complete in both cubes   sqrt(N) = {np.sqrt(N):.1f}")
        mo, so = channel_stats(da, m, keep, nz)
        me, se = channel_stats(de, m, keep, nz)

    fo, fe = so / np.sqrt(N), se / np.sqrt(N)
    print(f"\n    {'':<6}{'scatter':>10}{'floor':>10}{'|mean|/floor':>15}"
          f"{'channels > ' + str(args.n_floor) + 'x':>18}")
    for lab, mm, sc, fl in (("ours", mo, so, fo), ("ESO", me, se, fe)):
        r = np.abs(mm) / fl
        print(f"    {lab:<6}{np.median(sc):>10.3f}{np.median(fl):>10.4f}"
              f"{np.median(r):>15.2f}"
              f"{f'{int((r > args.n_floor).sum()):,} / {nz:,}':>18}"
              f"  ({100 * (r > args.n_floor).mean():.1f}%)")

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=args.figsize,
                           gridspec_kw={"height_ratios": [1, 1.3, 1.3], "hspace": 0.09})

    ax[0].plot(wl, so, lw=0.6, color=C_OURS, label="ours")
    ax[0].plot(wl, se, lw=0.6, color=C_ESO, label="ESO")
    ax[0].set_ylabel("scatter across\nblank spaxels")
    ax[0].legend(fontsize=11, loc="upper left", frameon=False)
    ax[0].grid(alpha=0.2)
    if args.scatter_ylim:
        ax[0].set_ylim(*args.scatter_ylim)
    else:
        ax[0].set_ylim(0, np.percentile(np.concatenate([so, se]), 99.5) * 1.15)

    # The two residual panels share a y range, or the larger residual is squeezed to
    # look like the smaller one.
    lim = args.ylim if args.ylim else robust_range(np.concatenate([mo, me]))
    for a, (lab, mm, fl, c) in zip(ax[1:], (("ours", mo, fo, C_OURS),
                                            ("ESO", me, fe, C_ESO))):
        a.fill_between(wl, -args.n_floor * fl, args.n_floor * fl, color=C_BAND,
                       lw=0, label=f"$\\pm${args.n_floor:g} $\\times$ noise floor")
        a.axhline(0, lw=0.8, color=C_ZERO_BAND)
        a.plot(wl, mm, lw=0.6, color=c, label=f"{lab}: mean over blank")
        a.set_ylabel("flux")
        a.set_ylim(*lim)
        a.legend(fontsize=11, loc="upper left", frameon=False, ncol=2)
        a.grid(alpha=0.2)
    ax[2].set_xlabel("wavelength [$\\AA$]")
    ax[2].set_xlim(wl.min(), wl.max())

    out = (Path(args.out) if args.out
           else pointing.figdir("sky") / f"blank_noise_floor_{run.name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


def main():
    ap = argparse.ArgumentParser(
        description="What the blank region has left in it: how large it is, and "
                    "whether it is more than noise -- ours against ESO's")
    ap.add_argument("--work", required=True, help="pointing work directory, e.g. results/skymodel/p01")
    ap.add_argument("--view", choices=["mean", "floor"], default="mean",
                    help="mean: the mean spectrum over blank, ours against ESO's. "
                         "floor: the same mean per channel against the noise floor of "
                         "the spaxels it was taken over")
    ap.add_argument("--run", default=None,
                    help="glob naming the run directory under step05 that holds our "
                         "sky_subtracted.fits; without it the newest run is used")
    ap.add_argument("--nosky", default=None,
                    help="ESO sky-subtracted cube; by default derived from the wsky "
                         "filename recorded in step03/meta.json")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="mean view: y range of the upper panel, by default a robust "
                         "range in residual mode and autoscale in sky mode. floor "
                         "view: y range of the two residual panels, shared")
    ap.add_argument("--statistic", choices=["mean", "median"], default="mean",
                    help="mean view: how the blank spaxels are collapsed into one "
                         "spectrum per channel. mean is step3's sigma-clipped mean; "
                         "median is the level half the spaxels are above")
    ap.add_argument("--mode", choices=["residual", "sky"], default="residual",
                    help="mean view: residual: what each pipeline leaves in blank "
                         "(both should be zero). sky: what each thinks the sky is")
    ap.add_argument("--alpha", type=float, default=0.75,
                    help="mean view: opacity of the ESO curve; ours is always drawn solid")
    ap.add_argument("--diff", action="store_true",
                    help="mean view: add a lower panel with ours minus ESO")
    ap.add_argument("--resid-ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="mean view: y range of the --diff panel; default is a robust "
                         "range of it")
    ap.add_argument("--n-floor", type=float, default=1.0,
                    help="floor view: width of the drawn band in noise floors. 1 is "
                         "'the mean of this many spaxels with no systematic error'")
    ap.add_argument("--scatter-ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="floor view: y range of the scatter panel")
    ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=None,
                    help=f"default {FIGSIZE['mean']} in the mean view, {FIGSIZE['floor']} "
                         f"in the floor view: the mean view draws one panel, two with "
                         f"--diff, and the floor view always three")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.figsize is None:
        args.figsize = FIGSIZE[args.view]

    pointing = Run(args.work)
    meta = pointing.meta(3)
    (view_mean if args.view == "mean" else view_floor)(args, pointing, meta)


if __name__ == "__main__":
    main()
