"""The mean spectrum of a region of the field, from one cube or from several.

Was the sky removed cleanly, and did the source survive? Residual size alone cannot
tell one from the other -- subtracting the source away flattens the residual the same
way -- so both readings have to land in the same figure. A region at a time is how: in
blank the curve should sit on zero, and on the galaxy it should be a galaxy spectrum
with its lines standing up rather than pushed down.

Every figure here is the same three steps: choose a set of spaxels, average each set
over one cube or over several, draw the curves together and print what they add up to.
Only the choosing differs, and `--zones` is what says it.

    zone_spectra.py --work results/skymodel/p01
    zone_spectra.py --work results/skymodel/p01 --zones outside --cubes ours eso
    zone_spectra.py --work results/skymodel/p01 --zones galaxy --map
    zone_spectra.py --work results/skymodel/p01 --zones blank
    zone_spectra.py --work results/skymodel/p01 --zones blank --view floor
    zone_spectra.py --work results/skymodel/p01 --zones box
    zone_spectra.py --work results/skymodel/p01 --zones box --half 0

--zones all | galaxy | outside -- the field's own zones
------------------------------------------------------
The zones zone_labels builds: the brightness layers of the galaxy, the rings outside
its boundary, or both, one panel each. `--cubes` says which cubes to draw in them.

Colour carries whichever of the two is varying. With one cube the panels differ by
zone, so the colour runs inner to outer through viridis and the zone map is drawn in
the same colours. With several, the colour names the cube and is the same in every
panel, since what is being read is the difference between them and it must mean one
thing from panel to panel.

--zones blank -- what is left where there is no source, ours against ESO's
-------------------------------------------------------------------------
Blank has no source in it, so after a perfect sky subtraction its mean spectrum is
zero: no continuum, no residual sky lines, only noise averaged down. Anything else is
what the sky model got wrong, and both pipelines are measured against that same answer.

    ours   the mean of our sky_subtracted cube over the blank spaxels
    ESO    the mean of the ESO nosky cube over the same spaxels

The mask is `zones.blank_mask`, rebuilt from step03/meta.json -- that run's blank, not
the zone construction's outermost ring, since two numbers over different spaxels are
not a comparison. The same seg and the same --xlim / --ylim / --exclude-box, so both
curves are averaged over identical spaxels with identical sigma-clipping. A spaxel must
be spectrally complete in both cubes; how many that leaves is printed, because the two
do not carry NaN in the same places.

The two curves are the two pipelines, so blank takes no `--cubes`: `--mode` says which
pair. `--mode sky` compares the inputs instead -- our blank mean sky (step3's
blank_mean_spectrum, the sky as observed) against wsky - nosky, the sky ESO chose to
remove -- and its --diff panel is mean(nosky) in blank.

--diff adds a lower panel with the two curves' difference. In residual mode both are
the same data minus a sky, so it is (data - our sky) - (data - ESO sky) = ESO sky - our
sky, the two sky models differenced with the data cancelled out. Off by default: read
against zero, the panel above already says how far each is.

--zones box -- boxes picked by content, one figure each
-------------------------------------------------------
Not typed-in coordinates: core/halo from the white-light brightness inside the main
source group, src edge from the distance to that group, blank sampled uniformly along
the same distance. A box must lie entirely inside its class's region (checked with
minimum_filter), or the label and the content would not match. The map says where they
landed, and is drawn whether or not it was asked for.

    blank pixels    the line sits on 0  -> the sky was removed cleanly
    source pixels   a galaxy spectrum   -> the source survived; a line pushed down is
                                           over-subtraction

--half 0 is a single spaxel, chosen by the same rule but not averaged. A box beats the
noise down as 1/sqrt(N) and is good for the zero point; a single point is noisy, but no
averaging can hide a systematic offset in it. Both write the same filenames, so the
output goes to evaluation/pNN/box/ or evaluation/pNN/point/ accordingly.

--pdf draws this run's own files instead: one page per box, raw and sky model
overplotted above, their difference below. It needs no reference cube at all, so it
works for a pointing ESO never subtracted.

--view floor -- is the residual noise, or is it wrong
-----------------------------------------------------
A mean over tens of thousands of spaxels looks flat whether the subtraction was good or
bad, as long as the error is random: averaging N spaxels divides random scatter by
sqrt(N) and leaves a systematic offset untouched. So each panel draws the mean residual
against its own noise floor:

    scatter   the spread across the zone's spaxels within one channel, (p84 - p16) / 2
              -- what a single spaxel actually looks like.
    floor     scatter / sqrt(N). Inside this band the channel's residual is
              indistinguishable from the same spaxels averaged with no systematic
              error; outside it, the same mistake was made in every spaxel.

The band is what makes the two pipelines comparable: rms and mean both shrink with the
number of spaxels averaged, so they say as much about the size of the region as about
the sky model, while the ratio to the floor does not. The top panel draws the two
scatters together, so how much of the separation below is systematic can be judged.

Nothing in that reading is particular to blank -- it is the reading for any set of
spaxels averaged down -- so `--view floor` takes any `--zones`, with two `--cubes` to
fill its two residual panels. Blank is where it was written, and where it is read.

This is what halo_spectra.py, outside_compare.py, blank_compare.py,
blank_noise_floor.py, blank.py and box_spectra.py were. The first two were the same
program: the same zones from the same construction, the same panel, the same markers,
the same report of what fell off it. One drew every zone with one curve, the other drew
the outer zones with two. The next two were the same blank mask, the same two cubes,
the same spaxels complete in both; one asked how far from zero the mean lands, the
other whether that distance is more than averaging noise leaves behind. blank.py was
already those two under one --view, and box_spectra.py chose its spaxels differently
and did the same three steps with them.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, arcsinh_stretch, data_hdu, slug  # noqa: E402
from config import resolve_path  # noqa: E402
from products import Run, latest_run, spectrum_stats  # noqa: E402
from spectra import (A_LINE, C_ESO, C_LINE, C_OURS, C_ZERO, LINES,  # noqa: E402
                     Z_HARO, despiked_range, panel_ylim, robust_range)
from utils import DZ_MAX, main_source_group, robust_spread  # noqa: E402
from zones import blank_mask, zone_labels, zone_means  # noqa: E402

# The three ways of choosing the spaxels, grouped for the options that belong to one of
# them. `ZONE` are the kinds zone_labels builds and numbers together.
ZONE = ("all", "galaxy", "outside")
BLANK = ("blank",)
BOX = ("box",)

# Opacity of every curve after the first. The first is the one being examined and is
# drawn solid on top; the rest are references underneath, and at full strength a pair
# that agrees reads as one stroke rather than as agreement.
A_REF = 0.75
# Colours after the first two. Reached by --cubes with three or more entries, and by a
# box curve whose label COL has no role for.
C_MORE = ["#2ca02c", "#9467bd", "#8c564b"]
# The difference curve, and the noise-floor band. Zero is drawn darker in the floor view
# because there it lies on top of the band; at C_ZERO the two greys would read as one.
C_RESID, C_ZERO_BAND, C_BAND = "#b30000", "0.45", "0.72"

# What products.spectrum_stats returns, in display order, with the figure's label.
# rms_from_zero keeps its full name: "rms" alone is unambiguous only while sigma is
# next to it. One format for every row and both blocks -- with %g each column would
# pick its own precision, and a gap in decimal places reads as a gap in the numbers.
STATS = [("mean", "mean"), ("sigma", "sigma"), ("skewness", "skewness"),
         ("kurtosis", "kurtosis"), ("rms_from_zero", "rms_from_zero")]
FMT = "{:.4f}"
CHUNK = 200
# The two views of one region do not draw the same figure, so neither can hold the
# other's size: the curves view has one panel, or two with --diff, and the floor view
# always has three. The zone kinds size themselves from --width and --panel-height
# instead, there being as many panels as zones.
FIGSIZE = {"curves": (22, 9), "floor": (22, 11)}

# The lines the box figures mark. Not spectra.LINES: these three are what they have
# always marked, and the fainter two would change every box figure ever made.
BOX_LINES = [("[O III]", 5006.8), ("Ha", 6562.8), ("[S II]", 6716.4)]

# The colours encode roles: external reference orange, old approach grey, ours blue.
# The labels stay English -- matplotlib's default font has no Chinese glyphs.
COL = {"ESO nosky": "#ff7f0e", "s per-spaxel (old)": "0.35", "ours": "#1f77b4"}
SHORT = {"ESO nosky": "ESO", "s per-spaxel (old)": "old", "ours": "ours"}

# These positions get a large canvas (see the loop). It only gives each channel more
# screen pixels; the layout is unchanged.
BIG_BOXES = {"core"}
BIG_FIG, BIG_DPI = (30, 8.4), 200
def select(names, which):
    """The zone numbers to draw, and their names.

    zone_labels numbers every zone it builds, and the numbers are what the spectra are
    keyed by, so a subset keeps its original numbers rather than being renumbered --
    otherwise the rings drawn here and the rings in the full figure would be different
    zones with the same names.
    """
    if which == "all":
        keep = list(range(len(names)))
    elif which == "galaxy":
        keep = [i for i, nm in enumerate(names) if nm.startswith("galaxy")]
    else:
        keep = [i for i, nm in enumerate(names) if nm.startswith("outside")]
    if not keep:
        raise SystemExit(f"the zone construction produced nothing matching --zones {which}")
    return [i + 1 for i in keep], [names[i] for i in keep]


def pick_boxes(seg, white, half, n_blank, edge_targets, margin, step04):
    """Pick the boxes by content.

    Returns ({name: (y0, y1, x0, x1)}, {name: note}, the main source group, its
    brightest pixel), box endpoints included.

    Blank boxes are numbered rather than named after their distance: that distance is
    a different number in every pointing, so it makes a different filename in each.
    #1 is always the nearest, and the distance survives in the note the figure prints.
    """
    main, ids, peak = main_source_group(seg, white, step04)
    valid = white != 0
    size  = 2 * half + 1

    # "the whole box is inside a class's region" = the class's boolean map survives a
    # minimum_filter, which on a boolean map asks "is there any False in this window".
    def whole(m):
        return ndimage.minimum_filter(m.astype(np.uint8), size=size).astype(bool)

    d_main   = ndimage.distance_transform_edt(~main)
    # The edge of the field of view has to be avoided: exposures are few there and
    # step5 writes NaN below 90% coverage, so a box there measures the mosaic boundary.
    # The main source group needs the same cut -- halo takes the faintest place inside
    # it, and the group can reach the edge, faint for want of exposure.
    d_edge   = ndimage.distance_transform_edt(valid)
    far_edge = d_edge > half + margin
    in_main  = whole(main & valid) & far_edge
    in_blank = whole((seg == 0) & valid) & far_edge

    boxes = {}
    notes = {}

    def add(name, cy, cx, note=""):
        boxes[name] = (cy - half, cy + half, cx - half, cx + half)
        notes[name] = note

    # --- main source group: brightest core, faintest fully-inside place (halo) ---
    wm = ndimage.uniform_filter(np.where(valid, white, 0.0), size=size)
    if in_main.any():
        cand = np.flatnonzero(in_main.ravel())
        add("core", *divmod(int(cand[np.argmax(wm.ravel()[cand])]), seg.shape[1]))
        add("halo", *divmod(int(cand[np.argmin(wm.ravel()[cand])]), seg.shape[1]))
    else:               # main source group too small for the box -> put it on the peak
        add("core", *peak)

    # --- the ring just outside the source: where over-subtraction shows ---
    # The box lies entirely in blank, so its centre gets no closer than half+1.
    cand = np.flatnonzero(in_blank.ravel())
    dm   = d_main.ravel()[cand]
    for t in edge_targets:
        j = int(cand[np.argmin(np.abs(dm - t))])
        add(f"src edge d={d_main.ravel()[j]:.0f}px", *divmod(j, seg.shape[1]))

    # --- other sources: the two brightest, to see if only Haro 11 improves ---
    others = [i for i in np.unique(seg[seg > 0]) if i not in ids]
    flux   = {i: float(np.nansum(np.where(seg == i, white, 0))) for i in others}
    n_added = 0
    for i in sorted(flux, key=flux.get, reverse=True):
        if n_added == 2:
            break
        m = (seg == i) & valid
        k = np.unravel_index(np.nanargmax(np.where(m, white, -np.inf)), seg.shape)
        if d_edge[k] <= half + margin:      # avoid the edge of the field of view too
            continue
        add(f"source #{int(i)}", int(k[0]), int(k[1]))
        n_added += 1

    # --- blank: sampled uniformly in distance from the main source group, for the
    #     trend "the farther away, the cleaner" ---
    lo, hi = float(dm.min()), float(dm.max())
    for k, q in enumerate(np.linspace(0.25, 1.0, n_blank), start=1):
        j = int(cand[np.argmin(np.abs(dm - (lo + q * (hi - lo))))])
        add(f"blank #{k}", *divmod(j, seg.shape[1]),
            note=f"d={d_main.ravel()[j]:.0f}px from the main source")
    return boxes, notes, main, peak


def cubes_and_labels(run, args):
    """The cubes to draw and what to call them.

    Each kind has its own default pair, used when --cubes is not given: the box figures
    colour by the label and abbreviate it in the statistics column, so their default
    has to carry the names COL and SHORT know. Naming the cubes replaces both, since a
    label kept from another cube would name the wrong curve.
    """
    default = (["eso", "ours"], ["ESO nosky", "ours"]) if args.zones in BOX \
        else (["ours"], ["ours"])
    cubes, labels = default if args.cubes is None else (args.cubes, args.cubes)
    labels = args.labels or labels
    if len(labels) != len(cubes):
        raise SystemExit(f"{len(cubes)} cubes but {len(labels)} labels")
    paths = [run.named_cube(c) for c in cubes]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"{p} does not exist")
    return paths, labels


def eso_cube(run, given):
    """ESO's cube as the run's config names it. Deriving it from the wsky filename,
    which is what this did, only ever finds data kept inside the repository."""
    nosky = resolve_path(given) if given else run.nosky
    if not nosky.exists():
        raise SystemExit(f"{nosky} does not exist")
    return nosky


def collapse(x, clip, statistic):
    """Collapse the blank spaxels of a chunk of channels into one spectrum.

    clipped -- step3's rule verbatim: a robust centre and spread decide what to reject,
    but the average is the mean, the unbiased estimate of the level the question asks
    for.

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
    """Per channel: the mean across the kept spaxels, and their robust scatter.

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


def box_mean(hdu, y0, y1, x0, x1):
    """Mean spectrum of one box. NaN spaxels are skipped automatically (nanmean)."""
    sub = np.asarray(hdu.data[:, y0:y1 + 1, x0:x1 + 1], np.float32)
    with np.errstate(invalid="ignore"):
        return np.nanmean(sub.reshape(sub.shape[0], -1), axis=1)


def smooth(a, w):
    """Moving average used for plotting. Purely to make things legible; none of the
    numbers in the table on the right have been smoothed."""
    if w <= 1:
        return a
    k = np.ones(w) / w
    return np.convolve(np.nan_to_num(a), k, mode="same") / np.convolve(
        np.isfinite(a).astype(float), k, mode="same")


def curve_colours(n_cubes, n_zones):
    """One colour per curve, carrying whichever of cube and zone is varying.

    Returns a function of (cube index, zone index). With one cube the zones are what
    differ between panels, and a sequential map says which way is outwards; a
    qualitative palette would hide that order. With several, the colour is the cube's
    and is the same in every panel, or a difference between two panels would be
    unreadable.
    """
    if n_cubes == 1:
        # The pale end of viridis is left out; it does not read on white.
        cols = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, n_zones))
        return lambda c, z: cols[z]
    fixed = [C_OURS, C_ESO] + C_MORE
    return lambda c, z: fixed[c % len(fixed)]


def panel_range(curves, rule, pct):
    """The y range of one panel, given every curve that goes in it.

    One range for all of them, or the larger residual is squeezed until it looks like
    the smaller. They can differ by orders of magnitude, so keeping every channel of
    every curve spends the height on one and draws the rest flat.

    medfilt   follows the median-filtered extremes of all the curves. Everything stays
              on the panel, at the cost of letting one curve's outliers flatten the
              others.
    first-pct the first curve is never allowed off the panel and the rest reach only to
              their percentiles, so a reference's deepest sky-line residuals may run
              off. The first curve is the one being examined, which is what makes this
              the default where there is more than one.
    """
    if rule == "medfilt" or len(curves) == 1:
        return panel_ylim(np.concatenate(curves))
    lo, hi = despiked_range(curves[0])
    for y in curves[1:]:
        lo = min(lo, float(np.nanpercentile(y, pct)))
        hi = max(hi, float(np.nanpercentile(y, 100 - pct)))
    m = 0.08 * max(hi - lo, 1e-9)
    return lo - m, hi + m


def mark_lines(ax, marker):
    """Haro 11's lines, at the top of the panel or across it.

    tick: a short stroke at the top, which names the wavelength without crossing the
    spectrum at all. line: a full-height rule, which puts the wavelength next to the
    data at the cost of drawing over it.
    """
    for _, lam in LINES:
        if marker == "tick":
            # ymin/ymax are axes fractions, so the stroke keeps to the top of the panel
            # whatever the range is.
            ax.axvline(lam * (1 + Z_HARO), ymin=0.92, ymax=1.0, lw=1.4,
                       color=C_LINE, zorder=3)
        else:
            ax.axvline(lam * (1 + Z_HARO), lw=0.8, color=C_LINE, alpha=A_LINE, zorder=0)


def draw_map(run, zones, keys, names, colour, main, out, dpi):
    """Where the regions are, in the colours the curves were drawn in.

    Not optional decoration: a mean spectrum with no picture of where it came from
    cannot be checked against the field.
    """
    seg, white, valid = run.seg, run.white, run.valid
    stretched, vmax = arcsinh_stretch(white, valid, soft=0.004)
    fig, ax = plt.subplots(figsize=(9, 9 * seg.shape[0] / seg.shape[1]))
    ax.imshow(stretched, origin="lower", cmap="gray", vmin=0, vmax=vmax)
    rgba = np.zeros(seg.shape + (4,))
    for z, k in enumerate(keys):
        rgba[zones == k] = list(colour(0, z)[:3]) + [0.55]
    ax.imshow(rgba, origin="lower")
    # The boundary the rings are measured from, in a colour none of the zones uses.
    ax.contour(main, levels=[0.5], colors="#ff7f0e", linewidths=1.2)
    ax.legend(handles=[mpatches.Patch(color=colour(0, z), label=names[z])
                       for z in range(len(keys))],
              fontsize=9, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              borderaxespad=0, frameon=False)
    ax.set_axis_off()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def s_field_of(run, args):
    """The s field to draw beside the boxes, or None if this run has none.

    Optional rather than required: a pointing that has not been through step 5 can
    still be looked at, and the map then has the one panel the original drew for it.
    """
    f = run.fit_dir / "sky_continuum_amplitude_field.npy"
    return np.load(f) if f.exists() else None


def box_map(white, seg, amplitude_field, boxes, out_path, title):
    """The boxes drawn on the white light image and on the s field -- reporting only
    the coordinates would not show which patch of the sky that is.

    Not draw_map: this is two panels, and the second one is why. A box sitting on a
    stripe of the s field is measuring that stripe, and the box figure cannot be read
    without seeing whether it does. The zone map above has one panel and no s field,
    which is right for a zone built out of the segmentation and wrong for a rectangle
    put down by hand.
    """
    n = 1 if amplitude_field is None else 2
    fig, axes = plt.subplots(1, n, figsize=(8.5 * n, 8), squeeze=False)
    q0 = max(float(np.nanpercentile(white[white != 0], 20)), 1e-3)
    d  = np.arcsinh(white / q0)
    axes[0][0].imshow(d, origin="lower", cmap="gray_r",
                      vmin=float(np.nanpercentile(d, 30)),
                      vmax=float(np.nanpercentile(d, 99.7)))
    axes[0][0].set_title("whitelight", fontsize=10)
    if amplitude_field is not None:
        # Centred on 1.0, half width 3 x the robust spread. Percentiles would be
        # asymmetric about the centre and narrower, so the same s field would look
        # more strongly striped -- a difference of the colour scale, not of the data.
        v = 3 * robust_spread(amplitude_field[np.isfinite(amplitude_field)])
        im = axes[0][1].imshow(amplitude_field, origin="lower", cmap="RdBu_r",
                               vmin=1 - v, vmax=1 + v)
        plt.colorbar(im, ax=axes[0][1], fraction=0.045, pad=0.01)
        axes[0][1].set_title("s field (new continuum map)", fontsize=10)
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax in axes[0]:
        ax.contour(seg > 0, levels=[0.5], colors="#2ca02c", linewidths=0.45)
        for i, (nm, (y0, y1, x0, x1)) in enumerate(boxes.items()):
            c = cmap[i % 10]
            if y0 == y1 and x0 == x1:
                # a single spaxel drawn as a box is one pixel, invisible at this scale
                ax.plot(x0, y0, "+", color=c, ms=13, mew=2.2)
            else:
                ax.add_patch(mpatches.Rectangle((x0 - .5, y0 - .5), x1 - x0 + 1,
                                                y1 - y0 + 1, fill=False, ec=c,
                                                lw=1.8))
            ax.annotate(nm, (x0 + (x1 - x0) / 2, y1), xytext=(0, 5),
                        textcoords="offset points", color=c, fontsize=8,
                        fontweight="bold", ha="center",
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        ax.set_xlabel("x [pix]")
    axes[0][0].set_ylabel("y [pix]")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def draw_pdf(boxes, notes, wl, tri, out_path, title, smooth_w):
    """One box per page, two panels stacked. A small residual is not good news by
    itself, since subtracting the source away shrinks it too, so read both:

        upper panel   raw and model overplotted -- does the model sit on the data
        lower panel   resid = raw - model, enlarged. Blank should keep only noise; a
                      source should keep its shape, and a flattened one is
                      over-subtraction.

    The panels do not share a y scale: the sky continuum is orders of magnitude larger
    than the residual, so on a common scale the residual would be a line stuck on 0.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out_path) as pdf:
        for nm, b in boxes.items():
            raw, mod, res = tri[nm]
            fig, ax = plt.subplots(2, 1, figsize=(13, 7.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1.25, 1],
                                                "hspace": 0.08})
            # raw thick, model thin on top. At equal widths a hidden curve and a curve
            # that was never drawn look the same, which is the question this panel
            # answers; different widths leave a grey rim where the two coincide.
            ax[0].plot(wl, smooth(raw, smooth_w), lw=1.4, color="0.45",
                       label="raw (wsky)")
            ax[0].plot(wl, smooth(mod, smooth_w), lw=0.5, color="#d62728",
                       label="sky model")
            ax[0].set_ylabel("flux", fontsize=9)
            ax[0].legend(fontsize=9, loc="upper right", framealpha=0.85)
            ax[0].grid(alpha=0.25)

            ax[1].axhline(0, color="0.5", lw=0.7)
            ax[1].plot(wl, smooth(res, smooth_w), lw=0.5, color="#1f77b4",
                       label="residual = raw − model")
            ax[1].set_ylabel("residual", fontsize=9)
            ax[1].set_xlabel("wavelength [$\\AA$]")
            ax[1].legend(fontsize=9, loc="upper right", framealpha=0.85)
            ax[1].grid(alpha=0.25)
            v = res[np.isfinite(res)]
            lo, hi = (float(x) for x in np.percentile(v, [0.5, 99.5]))
            pad = 0.30 * max(hi - lo, 1e-9)
            ax[1].set_ylim(min(lo - pad, 0.0), max(hi + pad, 0.0))
            ax[1].set_xlim(wl[0], wl[-1])
            for _, lam in BOX_LINES:
                for a_ in ax:
                    a_.axvline(lam * (1 + Z_HARO), color="0.75", lw=0.5, ls=":")

            st = spectrum_stats(res)
            fig.suptitle(
                f"{title}\n{nm}{'  ' + notes[nm] if notes.get(nm) else ''}"
                f"   y {b[0]}-{b[1]}  x {b[2]}-{b[3]}   "
                f"{(b[1]-b[0]+1)*(b[3]-b[2]+1)} spaxels   |   residual: "
                f"mean {st['mean']:+.3f}  sigma {st['sigma']:.3f}  "
                f"rms_from_zero {st['rms_from_zero']:.3f}", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)
    print(f"saved -> {out_path}")


def zone_regions(args, run):
    """The zones of the field, their numbers and their names, and the header that says
    what was read -- the spaxel counts and median brightness, which is what tells a
    zone that came out empty from one that is genuinely dark."""
    seg, white, valid = run.seg, run.white, run.valid
    main, ids, _ = main_source_group(seg, np.where(valid, white, np.nan),
                                     Path(args.step04) if args.step04 else None,
                                     args.dz_max)
    zones, all_names = zone_labels(seg, white, valid, main, args.layers, args.rings)
    keys, names = select(all_names, args.zones)
    print(f"{run.name}: main group {len(ids)} ids {ids}, {int(main.sum()):,} px")
    return zones, keys, names, main


def zone_curves(args, run):
    """One panel per zone, every cube drawn in each."""
    paths, labels = cubes_and_labels(run, args)
    zones, keys, names, main_ = zone_regions(args, run)
    n = len(keys)
    wl = run.wl
    white = run.white

    for lab, p in zip(labels, paths):
        print(f"  {lab:<10}{p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
    print(f"  {wl.size} channels {wl.min():.1f}-{wl.max():.1f} A\n")
    print(f"    {'zone':<22}{'spaxels':>9}{'median white':>14}")
    for z, k in enumerate(keys):
        m = zones == k
        wv = white[m]
        print(f"    {names[z]:<22}{int(m.sum()):>9,}"
              f"{(np.median(wv) if wv.size else np.nan):>14.3f}")

    print("\n  averaging the cubes ...")
    spec = [zone_means(p, zones, keys, wl.size) for p in paths]

    # The same lines the figure marks, so the dropped channels are exactly what a
    # reader sees marked. All False unless --exclude-source-lines was given.
    src = np.zeros(wl.size, bool)
    if args.exclude_source_lines:
        for _lab, lam in LINES:
            src |= np.abs(wl - lam * (1 + Z_HARO)) <= args.exclude_source_lines
        print(f"\n  source lines removed from the second set of columns: "
              f"+-{args.exclude_source_lines:g} A around {len(LINES)} lines at "
              f"z={Z_HARO:g}, {int(src.sum())} of {wl.size} channels "
              f"({100 * src.mean():.1f}%)")

    wid = max(6, max(len(l) for l in labels) + 2)
    head = f"\n    {'':<22}{'':<{wid}}{'mean':>10}{'sigma':>10}{'rms_from_zero':>16}"
    if src.any():
        head += f"{'mean':>12}{'sigma':>10}{'rms_from_zero':>16}    source lines out"
    print(head)
    for z in range(n):
        for c, lab in enumerate(labels):
            st = spectrum_stats(spec[c][z])
            row = (f"    {names[z] if c == 0 else '':<22}{lab:<{wid}}"
                   f"{st['mean']:>10.4f}{st['sigma']:>10.4f}{st['rms_from_zero']:>16.4f}")
            if src.any():
                sk = spectrum_stats(spec[c][z][~src])
                row += (f"{sk['mean']:>12.4f}{sk['sigma']:>10.4f}"
                        f"{sk['rms_from_zero']:>16.4f}")
            print(row)

    if args.smooth > 1:
        kk = np.ones(args.smooth) / args.smooth
        spec = [np.array([np.convolve(y, kk, mode="same") for y in s]) for s in spec]

    rule = args.ylim_rule or ("medfilt" if len(paths) == 1 else "first-pct")
    ph = args.panel_height or (1.9 if len(paths) == 1 else 2.4)
    colour = curve_colours(len(paths), n)
    if args.separate:
        # A zone per file. The full canvas height is what makes small residuals
        # readable; the price is independent y ranges, so zones cannot be compared.
        figs = [plt.subplots(figsize=(args.width, ph * 2.2)) for _ in keys]
        axes = [a for _, a in figs]
    else:
        fig, axes = plt.subplots(n, 1, sharex=True,
                                 figsize=(args.width, ph * n + 1.2),
                                 gridspec_kw={"hspace": 0.10})
        axes = list(np.atleast_1d(axes))

    clipped = []
    for z, ax in enumerate(axes):
        if not args.no_lines:
            mark_lines(ax, args.marker)
        ax.axhline(0, lw=0.8, color=C_ZERO)
        # Drawn last to first: the references go underneath and thicker, the curve
        # being examined on top and thinner, so an orange rim around the blue reads as
        # agreement rather than as one line.
        # Alone, the curve is as fine as it reads; over a reference it has to be
        # thick enough to stay visible on top of it, and the reference thicker still
        # so an orange rim around the blue reads as agreement rather than one line.
        for c in range(len(paths) - 1, -1, -1):
            lw = 0.6 if len(paths) == 1 else (0.7 if c == 0 else 1.3)
            ax.plot(wl, spec[c][z], lw=lw, color=colour(c, z),
                    label=labels[c], alpha=1.0 if c == 0 else A_REF,
                    zorder=4 if c == 0 else 2)
        lo, hi = panel_range([spec[c][z] for c in range(len(paths))], rule, args.pct)
        ax.set_ylim(lo, hi)
        for c, lab in enumerate(labels):
            y = spec[c][z]
            off = np.flatnonzero((y < lo) | (y > hi))
            if off.size:
                # Named, not silently cropped: a channel drawn off the panel is a
                # measurement about the cube, and the reader cannot see it in the figure.
                worst = off[np.argmax(np.abs(y[off]))]
                clipped.append(f"    {names[z]:<22}{lab:<{wid}}{off.size:>4} channel(s) "
                               f"off the panel; the largest is {y[worst]:.1f} at "
                               f"{wl[worst]:.1f} A")
        ax.set_ylabel("flux", fontsize=9)
        if not args.separate:
            # Stacked, the corner text is the only thing telling the panels apart. One
            # zone per file names itself in the filename, so the text is dropped there.
            ax.text(0.004, 0.93, names[z], transform=ax.transAxes, fontsize=10,
                    va="top", ha="left",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.6))
        if len(paths) > 1:
            if args.separate:
                # Above the axes, so it never lands on a residual the panel exists to show.
                ax.legend(fontsize=11, loc="lower left", frameon=False, ncol=len(paths),
                          bbox_to_anchor=(0, 1.005), borderaxespad=0)
            elif z == 0:
                ax.legend(fontsize=10, loc="upper right", frameon=False, ncol=len(paths))
        if args.separate:
            ax.set_xlabel("wavelength [$\\AA$]")
        ax.set_xlim(*(args.xlim if args.xlim else (wl.min(), wl.max())))
    if not args.separate:
        axes[-1].set_xlabel("wavelength [$\\AA$]")
        axes[-1].set_xlim(*(args.xlim if args.xlim else (wl.min(), wl.max())))
    if clipped:
        print(f"\n  drawn off the panel (y range rule {rule}"
              + (f", pct {args.pct:g}" if rule == "first-pct" else "") + "):")
        print("\n".join(clipped))

    # The name says what is in the figure: which zones, and which cubes against which.
    span = f"_{args.xlim[0]:.0f}-{args.xlim[1]:.0f}" if args.xlim else ""
    stem = f"{args.zones}_" + "_vs_".join(slug(l) for l in labels)
    # --out may name either a directory or the file itself, so the directory to make
    # is not known until that is decided; each branch below makes its own.
    d = Path(args.out) if args.out else run.figdir("halo")
    if args.separate:
        for (f, _), nm in zip(figs, names):
            o = d / f"{stem}_{slug(nm)}{span}.png"
            o.parent.mkdir(parents=True, exist_ok=True)
            f.savefig(o, dpi=args.dpi, bbox_inches="tight")
            plt.close(f)
            print(f"saved -> {o}")
    else:
        o = d if d.suffix == ".png" else d / f"{stem}{span}.png"
        o.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(o, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"\nsaved -> {o}")

    if (len(paths) == 1 or args.map) and not args.no_map:
        draw_map(run, zones, keys, names, colour, main_,
                 (o if not args.separate else d / f"{stem}{span}.png")
                 .with_name(f"{stem}{span}_map.png"), args.dpi)


def blank_curves(args, run):
    """The mean residual spectrum of each pipeline over blank, drawn against zero."""
    W = run.work
    meta = run.meta(3)
    nosky = eso_cube(run, args.nosky)

    # The two cubes to average, and what each curve is called. In sky mode the left one
    # is the raw cube and the ESO curve becomes a difference.
    if args.mode == "residual":
        rundir = latest_run(W, "sky_subtracted.fits", "step06", args.run)
        if rundir is None:
            raise SystemExit(
                f"no sky_subtracted.fits under {W}/step05 or {W}/step06 -- "
                f"this pointing has not been through step6")
        cube_a, cube_b = rundir / "sky_subtracted.fits", nosky
        lab_d = "ours $-$ ESO  $=$  ESO sky $-$ our sky"
        src = rundir.relative_to(ROOT)
    else:
        rundir = None
        cube_a, cube_b = run.wsky, nosky
        lab_d = "ours $-$ ESO  $=$  mean nosky in blank"
        src = run.wsky.relative_to(ROOT)
    lab_a, lab_b = "ours", "ESO"

    m, n_all, seg_p = blank_mask(W, meta)
    clip = float(meta.get("clip_sigma", 30.0))
    wl = run.wl
    nz = wl.size
    print(f"{run.name}:  mode {args.mode}   ours {src}   ESO {nosky.name}")
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
    if args.mode == "sky" and args.statistic == "clipped":
        check_against_step3(run, wl, ours)

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
    # and the run because a pointing can hold several. The sigma-clipped mean is
    # written "mean" as it always has been: it is step3's mean, and a renamed file is
    # a figure someone has to go and find again.
    tag = "mean" if args.statistic == "clipped" else args.statistic
    stem = (f"blank_{args.mode}_{tag}_vs_eso"
            + (f"_{rundir.name}" if rundir is not None else "")
            + ("_diff" if args.diff else ""))
    out = Path(args.out) if args.out else run.figdir("sky") / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


def floor_view(args, run, mask, region, counted, cube_a, cube_b, labels, out):
    """One region's mean, per channel, against the noise floor of its own spaxels.

    `region` is what the panels call the spaxels they were taken over, and `counted`
    is how the caller wants them accounted for; this adds what survives the
    completeness cut, since only spaxels complete in both cubes take part and the two
    do not carry NaN in the same places.
    """
    wl = run.wl
    nz = wl.size
    lab_a, lab_b = labels

    with fits.open(cube_a, memmap=True) as ha, fits.open(cube_b, memmap=True) as he:
        da, de = data_hdu(ha), data_hdu(he)
        ca = np.ones(int(mask.sum()), bool)
        ce = np.ones(int(mask.sum()), bool)
        for j in range(0, nz, CHUNK):
            ca &= np.isfinite(np.asarray(da.data[j:j + CHUNK], np.float32)[:, mask]).all(axis=0)
            ce &= np.isfinite(np.asarray(de.data[j:j + CHUNK], np.float32)[:, mask]).all(axis=0)
            print(f"    coverage {min(j + CHUNK, nz)}/{nz}", end="\r", flush=True)
        print(" " * 30, end="\r")
        keep = ca & ce
        N = int(keep.sum())
        print(f"{counted}{N:,} complete in both cubes   sqrt(N) = {np.sqrt(N):.1f}")
        mo, so = channel_stats(da, mask, keep, nz)
        me, se = channel_stats(de, mask, keep, nz)

    fo, fe = so / np.sqrt(N), se / np.sqrt(N)
    print(f"\n    {'':<6}{'scatter':>10}{'floor':>10}{'|mean|/floor':>15}"
          f"{'channels > ' + str(args.n_floor) + 'x':>18}")
    for lab, mm, sc, fl in ((lab_a, mo, so, fo), (lab_b, me, se, fe)):
        r = np.abs(mm) / fl
        print(f"    {lab:<6}{np.median(sc):>10.3f}{np.median(fl):>10.4f}"
              f"{np.median(r):>15.2f}"
              f"{f'{int((r > args.n_floor).sum()):,} / {nz:,}':>18}"
              f"  ({100 * (r > args.n_floor).mean():.1f}%)")

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=args.figsize,
                           gridspec_kw={"height_ratios": [1, 1.3, 1.3], "hspace": 0.09})

    ax[0].plot(wl, so, lw=0.6, color=C_OURS, label=lab_a)
    ax[0].plot(wl, se, lw=0.6, color=C_ESO, label=lab_b)
    ax[0].set_ylabel(f"scatter across\n{region} spaxels")
    ax[0].legend(fontsize=11, loc="upper left", frameon=False)
    ax[0].grid(alpha=0.2)
    if args.scatter_ylim:
        ax[0].set_ylim(*args.scatter_ylim)
    else:
        ax[0].set_ylim(0, np.percentile(np.concatenate([so, se]), 99.5) * 1.15)

    # The two residual panels share a y range, or the larger residual is squeezed to
    # look like the smaller one.
    lim = args.ylim if args.ylim else robust_range(np.concatenate([mo, me]))
    for a, (lab, mm, fl, c) in zip(ax[1:], ((lab_a, mo, fo, C_OURS),
                                            (lab_b, me, fe, C_ESO))):
        a.fill_between(wl, -args.n_floor * fl, args.n_floor * fl, color=C_BAND,
                       lw=0, label=f"$\\pm${args.n_floor:g} $\\times$ noise floor")
        a.axhline(0, lw=0.8, color=C_ZERO_BAND)
        a.plot(wl, mm, lw=0.6, color=c, label=f"{lab}: mean over {region}")
        a.set_ylabel("flux")
        a.set_ylim(*lim)
        a.legend(fontsize=11, loc="upper left", frameon=False, ncol=2)
        a.grid(alpha=0.2)
    ax[2].set_xlabel("wavelength [$\\AA$]")
    ax[2].set_xlim(wl.min(), wl.max())

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


def blank_floor(args, run):
    """The blank means, per channel, against the noise floor of their own spaxels."""
    W = run.work
    meta = run.meta(3)
    rundir = latest_run(W, "sky_subtracted.fits", "step06", args.run)
    if rundir is None:
        raise SystemExit(f"no sky_subtracted.fits under {W}/step05 or {W}/step06")
    nosky = eso_cube(run, args.nosky)

    m, n_all, _ = blank_mask(W, meta)
    print(f"{run.name}:  ours {rundir.relative_to(ROOT)}   ESO {nosky.name}")
    out = (Path(args.out) if args.out
           else run.figdir("sky") / f"blank_noise_floor_{rundir.name}.png")
    floor_view(args, run, m, "blank",
               f"  blank {n_all:,} -> {int(m.sum()):,} in the step3 mask -> ",
               rundir / "sky_subtracted.fits", nosky, ("ours", "ESO"), out)


def zone_floor(args, run):
    """The same reading on the field's own zones: one figure per zone drawn."""
    paths, labels = cubes_and_labels(run, args)
    # Two panels of residual, so two cubes: the figure is one pipeline read against
    # another, and a single curve has nothing to be separated from.
    if len(paths) != 2:
        raise SystemExit(f"--view floor draws two residual panels and so needs "
                         f"two --cubes; {len(paths)} given")
    zones, keys, names, _ = zone_regions(args, run)
    for lab, p in zip(labels, paths):
        print(f"  {lab:<10}{p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
    stem = f"{args.zones}_" + "_vs_".join(slug(l) for l in labels)
    d = Path(args.out) if args.out else run.figdir("halo")
    for z, k in enumerate(keys):
        m = zones == k
        print(f"\n  {names[z]}")
        floor_view(args, run, m, names[z], f"    {int(m.sum()):,} spaxels -> ",
                   paths[0], paths[1], labels,
                   d / f"{stem}_floor_{slug(names[z])}.png")


def box_regions(args, run):
    """The boxes, printed: they are chosen by content, and a reader cannot see that
    choice in the figure."""
    # The grouping redshifts must come from the fit being drawn -- a workspace can hold
    # several step4 runs. meta records the classification file's path from the repo
    # root, so its parent is that run; older products name the key "best".
    meta = run.meta(6)
    step04 = ROOT / Path(meta.get("classification") or meta["best"]).parent
    boxes, notes, main, peak = pick_boxes(run.seg, run.white, args.half, args.n_blank,
                                          args.edge, args.margin, step04)
    print(f"main source {int(main.sum()):,} px, brightest pixel (y, x) = {peak}")
    for nm, (y0, y1, x0, x1) in boxes.items():
        print(f"  {nm:<18} y {y0}-{y1}  x {x0}-{x1}   {notes.get(nm, '')}")
    return boxes, notes, main


def box_curves(args, run):
    """One figure per box, every cube drawn in it, its statistics beside it."""
    paths, labels = cubes_and_labels(run, args)
    boxes, notes, main = box_regions(args, run)
    wl = run.wl

    curves = {nm: {} for nm in boxes}
    for lab, path in zip(labels, paths):
        with fits.open(path, memmap=True) as h:
            hdu = data_hdu(h)
            for nm, b in boxes.items():
                curves[nm][lab] = box_mean(hdu, *b)
        print(f"loaded {lab}")

    # half=0 is a single spaxel, a different kind of sampling, and both write the same
    # filenames -- mixed in one directory the later run would overwrite the earlier.
    kind = "box" if args.half else "point"
    outdir = Path(args.out) if args.out else run.figdir(kind)
    outdir.mkdir(parents=True, exist_ok=True)
    # The label is the key itself: "rms" alone is unambiguous only while sigma is next
    # to it, and a number quoted out of the figure loses that.
    keys = ("mean", "sigma", "rms_from_zero")
    # The statistics column is laid out from a character count, not from fixed
    # coordinates: the number of columns depends on how many cubes were named.
    NW, VW = len(max(keys, key=len)) + 2, 10   # label column width, value column width
    ncol   = len(paths)
    span   = NW + VW * ncol
    # COL says what the three known roles look like, so each keeps its colour whichever
    # order the cubes were named in. A label it has no role for takes one of C_MORE,
    # which those roles do not use: two curves drawn the same colour read as one.
    spare = iter(C_MORE * len(labels))
    colours = {lab: COL[lab] if lab in COL else next(spare) for lab in labels}
    for nm, b in boxes.items():
        # core is drawn large: at the default size several channels share one screen
        # pixel and a MUSE line is narrower than that, so the detail is thrown away
        # before it is drawn. Only core needs it -- the other positions are noise.
        big = nm in BIG_BOXES
        w, h, dpi = (BIG_FIG + (BIG_DPI,)) if big else (15, 4.2, 135)
        sc = w / 15.0        # font size follows the canvas, or it is unreadable when big
        fig = plt.figure(figsize=(w, h))
        # 0.04 is the width fraction of one monospace character at fontsize 8.5, so
        # span sizes the axis: too narrow truncates, too wide strands the numbers.
        gs  = fig.add_gridspec(1, 2, width_ratios=[6, 0.04 * span], wspace=0.02,
                               left=0.055, right=0.985,
                               top=1 - 0.59 / h, bottom=0.55 / h)
        ax = fig.add_subplot(gs[0, 0])
        ax.axhline(0, color="0.5", lw=0.6)
        for lab, y in curves[nm].items():
            ax.plot(wl, smooth(y, args.smooth), lw=0.5, alpha=0.8, label=lab,
                    color=colours[lab])
        # The range is asymmetric: on a source pixel the whole spectrum is above 0, and
        # forcing symmetry would squash the curve into half the canvas. Percentiles
        # rather than min/max, or a handful of bad channels set the range; 0 stays
        # inside it either way, being the line "clean sky" is read against. The padding
        # is generous because the percentiles choose what sets the scale, not what is
        # visible, and a spike cut off would be a residual sky line.
        v = np.concatenate([y[np.isfinite(y)] for y in curves[nm].values()])
        lo, hi = (float(x) for x in np.percentile(v, [args.ypct, 100 - args.ypct]))
        pad = args.ypad * max(hi - lo, 1e-9)
        ax.set_ylim(min(lo - pad, 0.0), max(hi + pad, 0.0))
        ax.set_xlim(wl[0], wl[-1])
        for lname, lam in BOX_LINES:    # redshifted; the sky model cannot remove these
            ax.axvline(lam * (1 + Z_HARO), color="0.75", lw=0.5, ls=":")
        ax.set_ylabel("flux", fontsize=9 * sc)
        ax.set_xlabel("wavelength [$\\AA$]", fontsize=9 * sc)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8 * sc)
        ax.legend(fontsize=9 * sc, loc="upper left", ncol=3, framealpha=0.85)

        sax = fig.add_subplot(gs[0, 1]); sax.axis("off")
        sax.text(0.0, 0.98, "\n".join([""] + list(keys)),
                 va="top", ha="left", family="monospace", fontsize=8.5 * sc,
                 transform=sax.transAxes)
        for j, (lab, y) in enumerate(curves[nm].items()):
            st = spectrum_stats(y)
            sax.text((NW + VW * (j + 1)) / span, 0.98,
                     "\n".join([SHORT.get(lab, lab[:9])]
                               + [f"{st[k]:.3f}" for k in keys]),
                     va="top", ha="right", family="monospace", fontsize=8.5 * sc,
                     color=colours[lab],
                     transform=sax.transAxes)
        npx = (b[1] - b[0] + 1) * (b[3] - b[2] + 1)
        where = (f"y {b[0]}  x {b[2]}" if npx == 1 else
                 f"y {b[0]}-{b[1]}  x {b[2]}-{b[3]}")
        note = notes.get(nm, "")
        fig.suptitle(f"{run.name}  {nm}   {where}   "
                     f"{npx} spaxel{'' if npx == 1 else 's'}"
                     + (f"   {note}" if note else ""), fontsize=12 * sc)
        o = outdir / f"{slug(nm)}.png"
        fig.savefig(o, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {o}")

    # Not optional here, and so not behind --map: the boxes are chosen by content, and
    # where they landed is the only thing that says the label matches what is in them.
    if not args.no_map:
        box_map(run.white, run.seg, s_field_of(run, args), boxes,
                outdir / "map.png", f"{kind} locations   {run.name}")


def box_pdf(args, run):
    """The same boxes on this run's own raw, model and residual."""
    boxes, notes, main = box_regions(args, run)
    tri = {}
    for kind, path in (("raw", ROOT / run.meta(6)["cube"]),
                       ("mod", run.sky_model), ("res", run.cube)):
        with fits.open(path, memmap=True) as h:
            hdu = data_hdu(h)
            for nm, b in boxes.items():
                tri.setdefault(nm, {})[kind] = box_mean(hdu, *b)
        print(f"loaded {kind}  {path.name}")
    tri = {nm: (d["raw"], d["mod"], d["res"]) for nm, d in tri.items()}
    out = (Path(args.out) / "box_raw_model_resid.pdf" if args.out
           else run.figdir("box") / "box_raw_model_resid.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    draw_pdf(boxes, notes, run.wl, tri, out,
             f"{args.work}  [{run.cube_dir.name}]", args.smooth)
    if not args.no_map:
        box_map(run.white, run.seg, s_field_of(run, args), boxes,
                out.with_suffix(".map.png"),
                f"{'box' if args.half else 'point'} locations   {run.name}")


def main():
    ap = argparse.ArgumentParser(
        description="Mean spectrum of a region of the field, from one cube or several")
    # ---- which spaxels, read from which cubes -------------------------------
    ap.add_argument("--work", required=True,
                    help="pointing work directory, e.g. results/skymodel/p01")
    ap.add_argument("--zones", choices=["all", "galaxy", "outside", "blank", "box"],
                    default="all",
                    help="which spaxels: the galaxy's brightness layers, the rings "
                         "outside its boundary, both, blank -- the spaxels step3 "
                         "learned the sky from, rebuilt from its own meta -- or box, "
                         "boxes picked by content around the field")
    ap.add_argument("--view", choices=["curves", "floor"], default="curves",
                    help="curves: the mean spectrum of each region. floor: the same "
                         "mean per channel against the noise floor of the spaxels it "
                         "was taken over, which needs exactly two --cubes. Not for "
                         "--zones box, whose figures already draw one box's residual")
    ap.add_argument("--cubes", nargs="+", default=None, metavar="CUBE",
                    help="ours (step06), eso (the nosky the config names), wsky (the "
                         "input), model (the sky taken out), run:GLOB (a run under "
                         "step05), or a path. The first is the one being examined. "
                         "Default ours, and eso ours for --zones box. Not for --zones "
                         "blank, whose two curves are the two pipelines and are chosen "
                         "by --mode")
    ap.add_argument("--labels", nargs="+", default=None, metavar="NAME",
                    help="names for the curves; defaults to what --cubes says")

    ap.add_argument("--statistic", choices=["mean", "median", "clipped"], default=None,
                    help="how the spaxels of a region are collapsed into one spectrum "
                         "per channel. --zones blank: clipped, step3's sigma-clipped "
                         "mean, is the default and what its figures are named for; "
                         "median is the level half the spaxels are above. The other "
                         "zones go through zones.zone_means, which averages, so mean "
                         "is all they take")

    # ---- --zones all/galaxy/outside: how the zones are built ----------------
    ap.add_argument("--layers", type=int, default=4,
                    help="--zones all/galaxy/outside: equal-count brightness layers "
                         "inside the main source group. Given even with --zones "
                         "outside, so the rings match the layer figure exactly")
    ap.add_argument("--rings", type=int, nargs="+", default=[0, 10, 25, 50],
                    help="--zones all/galaxy/outside: ring edges in px outside the "
                         "boundary; N edges give N-1 rings")
    ap.add_argument("--step04", default=None,
                    help="--zones all/galaxy/outside: step04 directory; given, the "
                         "main group keeps only members matching the main source's "
                         "redshift")
    ap.add_argument("--dz-max", type=float, default=DZ_MAX,
                    help="--zones all/galaxy/outside: maximum redshift difference from "
                         "the main source, with --step04")

    # ---- --zones box: how the boxes are picked, and what is drawn in them ---
    ap.add_argument("--half", type=int, default=6,
                    help="--zones box: box half-width; box width = 2*half+1")
    ap.add_argument("--n-blank", type=int, default=4,
                    help="--zones box: how many blank boxes, sampled uniformly in "
                         "distance from the main source group")
    ap.add_argument("--edge", type=float, nargs="+", default=[7, 20],
                    help="--zones box: target distance (px) from the main source for "
                         "outer boxes")
    ap.add_argument("--margin", type=float, default=10,
                    help="--zones box: minimum distance (px) from the field-of-view "
                         "edge. Edges have low exposure, and step5 writes NaN for "
                         "spaxels with coverage < 90%%")
    ap.add_argument("--ypct", type=float, default=0.5,
                    help="--zones box: percentile that sets the y range; the range "
                         "runs from this to 100 minus this. Smaller keeps more of the "
                         "spikes")
    ap.add_argument("--ypad", type=float, default=0.55,
                    help="--zones box: extra room above and below, as a fraction of "
                         "that range")
    ap.add_argument("--pdf", action="store_true",
                    help="--zones box: output PDF, one page per box, upper panel raw "
                         "vs sky model, lower panel residual. Uses only this run's own "
                         "files (raw / model / resid), no ESO or other run needed, so "
                         "it works for every pointing. The raw path is read from the "
                         "run's meta.json")

    # ---- which run, and which reference cube --------------------------------
    ap.add_argument("--run", default=None,
                    help="--zones blank: glob naming the run directory under step05 "
                         "that holds our sky_subtracted.fits; without it the newest "
                         "run is used. --zones box: the run directory under step05 to "
                         "draw, which also names the sky model --pdf reads; default is "
                         "the pipeline's own step05/step06")
    ap.add_argument("--nosky", default=None,
                    help="--zones blank: ESO sky-subtracted cube; by default the one "
                         "the run's config names")
    # ---- --zones blank: which pair, and the panels it draws them in ---------
    ap.add_argument("--mode", choices=["residual", "sky"], default="residual",
                    help="--zones blank: residual: what each pipeline leaves in blank "
                         "(both should be zero). sky: what each thinks the sky is")
    ap.add_argument("--diff", action="store_true",
                    help="--zones blank --view curves: add a lower panel with ours "
                         "minus ESO")
    ap.add_argument("--alpha", type=float, default=0.75,
                    help="--zones blank --view curves: opacity of the ESO curve; ours "
                         "is always drawn solid")
    ap.add_argument("--resid-ylim", type=float, nargs=2, metavar=("LO", "HI"),
                    default=None,
                    help="--zones blank --view curves: y range of the --diff panel; "
                         "default is a robust range of it")
    # ---- the y ranges and canvas of those two views -------------------------
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="--zones blank --view curves: y range of the upper panel, by "
                         "default a robust range in residual mode and autoscale in sky "
                         "mode. --view floor: y range of the two residual panels, shared")
    ap.add_argument("--n-floor", type=float, default=1.0,
                    help="--view floor: width of the drawn band in noise floors. 1 is "
                         "'the mean of this many spaxels with no systematic error'")
    ap.add_argument("--scatter-ylim", type=float, nargs=2, metavar=("LO", "HI"),
                    default=None,
                    help="--view floor: y range of the scatter panel")
    ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=None,
                    help=f"--zones blank and --view floor: default {FIGSIZE['curves']} "
                         f"in the curves view, {FIGSIZE['floor']} in the floor view -- "
                         f"the curves view draws one panel, two with --diff, and the "
                         f"floor view always three. The zone kinds size themselves "
                         f"from --width and --panel-height")

    # ---- --zones all/galaxy/outside: what the panels look like --------------
    ap.add_argument("--exclude-source-lines", type=float, nargs="?", const=12.0,
                    default=None, metavar="HW",
                    help="--zones all/galaxy/outside: also report the statistics with "
                         "Haro 11's own lines removed, +-HW Angstrom around each line "
                         "the figure marks (default 12). Outside the boundary is not "
                         "outside the galaxy's line emission, and those channels make "
                         "the residual columns favour whichever cube keeps less source "
                         "light")
    ap.add_argument("--smooth", type=int, default=1,
                    help="--zones all/galaxy/outside and --zones box: running-mean "
                         "width in channels for what is drawn; 1 leaves the spectrum "
                         "alone, and the statistics are never smoothed")
    ap.add_argument("--ylim-rule", choices=["first-pct", "medfilt"], default=None,
                    help="--zones all/galaxy/outside: see panel_range. Defaults to "
                         "medfilt for one cube and first-pct for more")
    ap.add_argument("--pct", type=float, default=5.0,
                    help="--zones all/galaxy/outside: percentile the reference curves "
                         "reach to, with first-pct")
    ap.add_argument("--xlim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="--zones all/galaxy/outside: wavelength range drawn, and a "
                         "suffix on the filename")
    ap.add_argument("--no-lines", action="store_true",
                    help="--zones all/galaxy/outside: drop the redshifted line markers")
    ap.add_argument("--marker", choices=["tick", "line"], default="tick",
                    help="--zones all/galaxy/outside: see mark_lines")
    ap.add_argument("--separate", action="store_true",
                    help="--zones all/galaxy/outside: one file per zone instead of one "
                         "stacked figure")
    ap.add_argument("--map", action="store_true",
                    help="--zones all/galaxy/outside: also draw where the zones are. "
                         "On by default with one cube, which is the figure that is "
                         "read against the field")
    ap.add_argument("--no-map", action="store_true",
                    help="drop the map. --zones box draws it whatever else was asked "
                         "for, the boxes being chosen by content")
    ap.add_argument("--width", type=float, default=20,
                    help="--zones all/galaxy/outside: figure width in inches")
    ap.add_argument("--panel-height", type=float, default=None,
                    help="--zones all/galaxy/outside: default 1.9 for one cube, 2.4 "
                         "for more -- two curves in a panel need the height to stay "
                         "apart")
    # ---- every kind ---------------------------------------------------------
    ap.add_argument("--dpi", type=int, default=180,
                    help="--zones box sets its own for the box figures -- core needs "
                         "200, see BIG_BOXES -- and takes this for the map only")
    ap.add_argument("--out", default=None,
                    help="--zones all/galaxy/outside: a directory, or the file itself. "
                         "--zones blank: the file. --zones box: a directory, one "
                         "figure per box in it")
    args = ap.parse_args()

    # An option belonging to another kind of region is a mistake, not a no-op: blank is
    # rebuilt from step3's meta and takes none of the zone construction's settings,
    # the boxes are picked by content and take none of the zone display options, and
    # the zone kinds read their cubes through zones.zone_means, which averages. Saying
    # so beats passing --statistic median and getting the mean back unremarked.
    CURVES = ZONE + BOX                     # the kinds that draw one cube against another
    misplaced = [(f, where) for f, given, kinds, where in (
        ("--cubes", args.cubes is not None, CURVES,
         "--zones all/galaxy/outside and --zones box"),
        ("--labels", args.labels is not None, CURVES,
         "--zones all/galaxy/outside and --zones box"),
        ("--layers", args.layers != 4, ZONE, "--zones all/galaxy/outside"),
        ("--rings", args.rings != [0, 10, 25, 50], ZONE, "--zones all/galaxy/outside"),
        ("--step04", args.step04 is not None, ZONE, "--zones all/galaxy/outside"),
        ("--dz-max", args.dz_max != DZ_MAX, ZONE, "--zones all/galaxy/outside"),
        ("--run", args.run is not None, BLANK + BOX, "--zones blank and --zones box"),
        ("--nosky", args.nosky is not None, BLANK, "--zones blank"),
        ("--mode", args.mode != "residual", BLANK, "--zones blank"),
        ("--diff", args.diff, BLANK, "--zones blank"),
        ("--alpha", args.alpha != 0.75, BLANK, "--zones blank"),
        ("--resid-ylim", args.resid_ylim is not None, BLANK, "--zones blank"),
        ("--exclude-source-lines", args.exclude_source_lines is not None, ZONE,
         "--zones all/galaxy/outside"),
        ("--smooth", args.smooth != 1, CURVES,
         "--zones all/galaxy/outside and --zones box"),
        ("--ylim-rule", args.ylim_rule is not None, ZONE, "--zones all/galaxy/outside"),
        ("--pct", args.pct != 5.0, ZONE, "--zones all/galaxy/outside"),
        ("--xlim", args.xlim is not None, ZONE, "--zones all/galaxy/outside"),
        ("--no-lines", args.no_lines, ZONE, "--zones all/galaxy/outside"),
        ("--marker", args.marker != "tick", ZONE, "--zones all/galaxy/outside"),
        ("--separate", args.separate, ZONE, "--zones all/galaxy/outside"),
        ("--map", args.map, ZONE, "--zones all/galaxy/outside"),
        ("--no-map", args.no_map, CURVES,
         "--zones all/galaxy/outside and --zones box"),
        ("--width", args.width != 20, ZONE, "--zones all/galaxy/outside"),
        ("--panel-height", args.panel_height is not None, ZONE,
         "--zones all/galaxy/outside"),
        ("--half", args.half != 6, BOX, "--zones box"),
        ("--n-blank", args.n_blank != 4, BOX, "--zones box"),
        ("--edge", args.edge != [7, 20], BOX, "--zones box"),
        ("--margin", args.margin != 10, BOX, "--zones box"),
        ("--ypct", args.ypct != 0.5, BOX, "--zones box"),
        ("--ypad", args.ypad != 0.55, BOX, "--zones box"),
        ("--pdf", args.pdf, BOX, "--zones box"),
        ("--view floor", args.view == "floor", ZONE + BLANK,
         "--zones all/galaxy/outside and --zones blank"),
    ) if given and args.zones not in kinds]
    if args.statistic not in (None, "mean") and args.zones not in BLANK:
        misplaced.append(("--statistic " + args.statistic, "--zones blank"))
    if misplaced:
        raise SystemExit("\n".join(f"{f}: {w} only" for f, w in misplaced))

    if args.figsize is None:
        args.figsize = FIGSIZE[args.view]
    # blank collapses its spaxels here and has always done it step3's way; the zone
    # kinds hand the job to zones.zone_means, which takes the plain mean.
    if args.statistic is None:
        args.statistic = "clipped" if args.zones in BLANK else "mean"

    # Only --zones box names a run directory to Run: blank reaches its own through
    # latest_run, which takes a glob, and the zone kinds name theirs in --cubes.
    run = Run(args.work, args.run if args.zones in BOX else None)
    if args.zones in BLANK:
        (blank_curves if args.view == "curves" else blank_floor)(args, run)
    elif args.zones in BOX:
        (box_pdf if args.pdf else box_curves)(args, run)
    elif args.view == "floor":
        zone_floor(args, run)
    else:
        zone_curves(args, run)


if __name__ == "__main__":
    main()
