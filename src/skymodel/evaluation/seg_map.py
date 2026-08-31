"""The source map of any segmentation: the whole field, a zoom on the main galaxy, or
the fourteen segmentations the professor supplied.

products.id_map()'s rows normally carry the group, redshift and similar fit fields, but
sometimes the question is only which things a mask circles as sources, with nothing to
do with the fitting -- comparing two segmentations of the same field, say.

So this assembles the rows id_map() needs (id and centroid) from one seg image and
leaves the locator drawing to products.id_map, with no second implementation of it.

--style locator is the map used while working, and its axes are part of that job:
reading a source's x/y off the figure is how you go and look at that spaxel. --style
slide is the same field with nothing around it. On a slide nothing outside the image is
read, so title, axes and frame are noise, and the image fills the canvas; the default
colour is green rather than the locator map's pale red, which sits too close to
greyscale in brightness to separate when projected.

What --style must never change is the asinh stretch and the translucent fill plus
contour. Two figures of one field drawn by different rules would look like different
data because the stretch, colours or labelling differ, so only the colour and the
surroundings are the style's to decide.

--crop main is that same field, cropped. The whole-field maps are locator maps, and at
that scale the halo and the few-pixel detections on it are too small to answer the two
questions the crop is for: does the segmentation stop where the light stops, and what
else did SExtractor detect inside and around that extended light? So it crops to the
main source group's bounding box plus a margin and stretches the background hard enough
for the halo to show. The main group is drawn as a contour only -- a translucent fill
would sit on exactly the light the figure exists to show. Every other source in the crop
is filled, outlined and labelled with its seg ID, so anything found here can be looked
up in the whole-field map.

The main group is the connected blob containing the brightest pixel
(utils.main_source_group); with --step04 its members must also share the main source's
redshift. Adjacency alone can chain through touching neighbours, so the printed ID list
and bbox make a chained blob visible. The margin is clipped to the data, and the
printout says how much room each side actually had.

--professor is a mode over pointings rather than over one run. Each of the fourteen
comes with two files besides the cube:

    DATACUBE_FINAL_{N}_pseudo_r.fits    the background image used for detection, the
                                        equally weighted average of the nosky cube
                                        over 5625-6825 A
    DATACUBE_FINAL_{N}_seg.fits         the segmentation produced on that image

It overlays the two and draws them on the professor's own pseudo_r rather than our
whitelight -- the mask grew out of pseudo_r, and a different background would give the
illusion of a misalignment -- plus one overview, because a single figure cannot answer
"is this set of masks consistent across the whole mosaic".

    conda run -n astro python src/skymodel/evaluation/seg_map.py \\
        --work results/skymodel/p01 --seg data/wsky_seg/DATACUBE_FINAL_1_seg.fits
    conda run -n astro python src/skymodel/evaluation/seg_map.py --style slide \\
        --work results/skymodel/p01 --labels --min-x 15 --color '#22d3ee'
    conda run -n astro python src/skymodel/evaluation/seg_map.py \\
        --work results/skymodel/p01 --crop main --margin 60 --soft 0.005
    conda run -n astro python src/skymodel/evaluation/seg_map.py --professor -n 4 8 12
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
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL, ROOT, map_name, qualitative, seg_and_background  # noqa: E402
from products import Run, id_map  # noqa: E402
from utils import DZ_MAX, arcsinh_stretch, main_source_group  # noqa: E402

FIGURES = EVAL / "masking"
# The professor's set is not one pointing's, and overview.png belongs to no single
# pointing even inside it, so it gets a level of its own.
PROF_FIGURES = EVAL / "masking/prof_seg"
SEGDIR = ROOT / "data/wsky_seg"

# All fourteen pointings, which is what --professor draws when it is not told which.
POINTINGS = list(range(1, 15))

COLOR = "#4ade80"
# At slide size the fill only says where the sources are, so it stays light enough to
# leave the bright source body visible underneath. The IDs are off by default for the
# same reason: crowded numbers are unreadable from the back of a room.
ALPHA_SLIDE = 0.25
# Fill opacity of the neighbouring sources in the main crop. They are drawn over sky,
# not over the halo's interesting part, so they can be solid enough to read at a glance.
ALPHA_MAIN = 0.40
# asinh softening. The percentile inside arcsinh_stretch sets the bright end; this
# sets how far the faint end is pulled up, and so whether faint sources show at all.
# It is passed in rather than changed there, where every figure shares the default.
# arcsinh_stretch's project default is set for the whole field; in the main crop the
# halo is the subject, so the faint end is pulled up further.
SOFT_SLIDE, SOFT_MAIN = 0.01, 0.004
# Saved resolution of the slide figure -- also what products.id_map saves the locator
# map at. The main crop is written finer: it is read up close, not from the back of a
# room. --dpi, --soft and --alpha each serve two modes that want different values, so
# the default belongs to the mode and not to the option, the way sky_basis.py keeps
# one --dpi with a default per half.
DPI_SLIDE, DPI_MAIN = 150, 180
# The main group's outline, in the colour utils.plot_main_group uses for the same
# object, so the two figures cannot be misread as two different footprints.
C_MAIN = "#ff7f0e"
# How far past the main group's bounding box to show. The margin exists for the light
# outside the boundary, so it must be a real fraction of the galaxy, not framing.
MARGIN = 45
# Width in inches of the main crop; its height follows the crop's aspect ratio.
WIDTH = 13


def source_ids(seg, min_area=1, min_x=None):
    """Which sources to draw -- returns (the IDs kept, how many the seg holds in all).

    The same question in every mode, so it is asked once.
    """
    all_ids, cnt = np.unique(seg[seg > 0], return_counts=True)
    ids = []
    for i, c in zip(all_ids, cnt):
        if c < min_area:
            continue
        if min_x is not None and np.nonzero(seg == i)[1].mean() < min_x:
            continue
        ids.append(int(i))
    return ids, len(all_ids)


def id_rows(seg, ids):
    """The rows id_map needs: the number and the centroid of each source."""
    rows = []
    for i in ids:
        y, x = np.nonzero(seg == i)
        # id_map reads group only when by_group=True, but the field still has to be
        # present or GROUP_COLOR over in products raises KeyError.
        rows.append(dict(id=int(i), x=float(x.mean()), y=float(y.mean()),
                         group="galaxy"))
    return rows


def slide_map(seg, white, out, color=COLOR, alpha=ALPHA_SLIDE, labels=False,
              ids=None, dpi=DPI_SLIDE, soft=SOFT_SLIDE, background=True,
              per_source=False):
    """Draw the map and write it to `out`.

    ids limits which sources are drawn; None draws every label in seg.

    background=False drops the white light and leaves the mask alone on black, so the
    figure says only which spaxels are source. With nothing to see through, the fill is
    drawn solid. per_source=True gives each source its own colour from a qualitative
    palette, for that mask figure, where no background separates neighbours. Over the
    white light the contours already do, and many colours would read as if the colour
    meant something about the source.
    """
    if ids is None:
        ids = np.unique(seg[seg > 0])
    keep = np.isin(seg, ids) & (seg > 0)

    h, w = white.shape
    fig, ax = plt.subplots(figsize=(12, 12 * h / w))
    # The project's one stretch, as products.id_map uses for the locator style, so the
    # two figures of the same field cannot end up looking like different data.
    if background:
        bg, vmax = arcsinh_stretch(white, soft=soft)
        ax.imshow(bg, origin="lower", cmap="gray", vmin=0, vmax=vmax)
    else:
        ax.imshow(np.zeros(seg.shape), origin="lower", cmap="gray", vmin=0, vmax=1)
        alpha = 1.0

    # One fill for all sources, one contour per source: the fill shows extent, the
    # contour keeps small sources visible, and only the contour needs repeating.
    palette = qualitative(len(ids)) if per_source else None
    rgba = np.zeros(seg.shape + (4,))
    if per_source:
        for k, i in enumerate(ids):
            rgba[seg == i] = list(palette[k % len(palette)]) + [alpha]
    else:
        rgba[keep] = list(matplotlib.colors.to_rgb(color)) + [alpha]
    ax.imshow(rgba, origin="lower")
    for k, i in enumerate(ids):
        m = seg == i
        c = palette[k % len(palette)] if per_source else color
        ax.contour(m, levels=[0.5], colors=[c], linewidths=1.0)
        if labels:
            y, x = np.nonzero(m)
            ax.text(x.mean(), y.mean(), str(int(i)), color="white", fontsize=11,
                    fontweight="bold", ha="center", va="center",
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="black")])

    ax.set_axis_off()                   # ticks, labels and the frame together
    fig.subplots_adjust(0, 0, 1, 1)     # image fills the canvas -- no margin on the slide
    # facecolor black, not the default white: the figure is saved without
    # bbox_inches="tight", so any canvas left uncovered would show as a white edge.
    fig.savefig(out, dpi=dpi, facecolor="black")
    plt.close(fig)


def field(args):
    """The whole pointing, in the two styles it gets read in."""
    alpha = ALPHA_SLIDE if args.alpha is None else args.alpha
    soft  = SOFT_SLIDE if args.soft is None else args.soft
    dpi   = DPI_SLIDE if args.dpi is None else args.dpi
    labels = False if args.labels is None else args.labels

    seg, white, seg_path = seg_and_background(args.work, args.seg, args.white)

    # Which sources to draw is the same question in both styles, so it is asked once.
    # --min-x cannot reach here under --style locator, so that style's printed count
    # is still the one --min-area alone explains.
    ids, n_all = source_ids(seg, args.min_area, args.min_x)

    name = map_name(args.work, seg_path)
    prefix = "id_map" if args.style == "locator" else "slide"
    out = Path(args.out) if args.out else FIGURES / f"{prefix}_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.style == "locator":
        rows = id_rows(seg, ids)
        id_map(seg, white, rows, out, by_group=False)
        print(f"{name}: {n_all} sources, plotted {len(rows)} (area >= {args.min_area} px)")
    else:
        slide_map(seg, white, out, color=args.color, alpha=alpha,
                  labels=labels, ids=ids, dpi=dpi, soft=soft,
                  background=not args.no_background,
                  per_source=args.per_source_color)
        print(f"{name}: {n_all} sources, drawn {len(ids)}")
    print(f"  source spaxels {int((seg > 0).sum()):,}  ({100 * (seg > 0).mean():.1f}% of field)")
    print(f"saved -> {out}")


def crop_main(args):
    """Zoom on the main galaxy: what is detected on and around its extended light."""
    alpha = ALPHA_MAIN if args.alpha is None else args.alpha
    soft  = SOFT_MAIN if args.soft is None else args.soft
    dpi   = DPI_MAIN if args.dpi is None else args.dpi
    labels = True if args.labels is None else args.labels

    # Run only for where the figure goes; the two crops read the field the same way,
    # so --seg and --white reach this one too. The dtypes are products.Run's, which
    # both this and the whole-field crop rely on: percentiles and nanmean must not
    # depend on what the FITS happened to be written with.
    run = Run(args.work)
    seg, white, _ = seg_and_background(args.work, args.seg, args.white)
    seg, white = seg.astype(int), np.asarray(white, float)
    valid = white != 0
    wn = np.where(valid, white, np.nan)
    step04 = Path(args.step04) if args.step04 else None
    mg, ids, _ = main_source_group(seg, wn, step04, args.dz_max)

    H, Xw = seg.shape
    ys, xs = np.nonzero(mg)
    # np.clip, not max/min by hand: the margin can run off the field.
    y0, y1 = int(np.clip(ys.min() - args.margin, 0, H)), int(np.clip(ys.max() + 1 + args.margin, 0, H))
    x0, x1 = int(np.clip(xs.min() - args.margin, 0, Xw)), int(np.clip(xs.max() + 1 + args.margin, 0, Xw))
    sub = np.s_[y0:y1, x0:x1]

    print(f"{run.name}: field {H}x{Xw}, main group {len(ids)} ids {ids}, "
          f"{int(mg.sum()):,} px")
    print(f"  bbox y {ys.min()}-{ys.max()}  x {xs.min()}-{xs.max()}")
    # Room actually available on each side. Less than --margin means the crop is cut
    # by the field, and the light beyond it was never observed.
    print(f"  margin {args.margin} px requested; available  "
          f"left {int(xs.min())}  right {int(Xw - 1 - xs.max())}  "
          f"bottom {int(ys.min())}  top {int(H - 1 - ys.max())}")
    print(f"  crop y {y0}-{y1 - 1}  x {x0}-{x1 - 1}   ({y1 - y0} x {x1 - x0} px)")

    # Distance to the nearest main-group pixel, so each neighbour reports how far
    # outside the boundary it is; area and centroid alone do not say that.
    dist = ndimage.distance_transform_edt(~mg)

    others = []
    for i in np.unique(seg[sub][seg[sub] > 0]):
        if int(i) in [int(k) for k in ids]:
            continue
        m = seg == i
        mc = m[sub]
        yy, xx = np.nonzero(mc)
        others.append(dict(id=int(i), area=int(m.sum()), in_crop=int(mc.sum()),
                           x=float(xx.mean()), y=float(yy.mean()),
                           dist=float(dist[m].min())))
    others.sort(key=lambda r: r["dist"])

    print(f"\n  {len(others)} other sources inside the crop"
          f"   ({int(np.unique(seg[seg > 0]).size)} in the whole field)")
    print(f"    {'id':>4}{'area':>7}{'in crop':>9}{'x':>7}{'y':>7}{'gap to main':>13}")
    for r in others:
        print(f"    {r['id']:>4}{r['area']:>7}{r['in_crop']:>9}"
              f"{r['x'] + x0:>7.1f}{r['y'] + y0:>7.1f}{r['dist']:>13.1f}")

    # The stretch comes from the whole field, not the crop: a percentile of a crop
    # centred on the galaxy is the galaxy, so the reference would move with the crop.
    stretched, vmax = arcsinh_stretch(white, valid, soft=soft)
    bg = stretched[sub]

    h, w = bg.shape
    fig, ax = plt.subplots(figsize=(args.width, args.width * h / w))
    ax.imshow(bg, origin="lower", cmap="gray", vmin=0, vmax=vmax)

    palette = qualitative(len(others))
    rgba = np.zeros(bg.shape + (4,))
    for k, r in enumerate(others):
        rgba[(seg == r["id"])[sub]] = list(palette[k]) + [alpha]
    if args.fill_main:
        rgba[mg[sub]] = list(matplotlib.colors.to_rgb(C_MAIN)) + [alpha]
    ax.imshow(rgba, origin="lower")

    for k, r in enumerate(others):
        ax.contour((seg == r["id"])[sub], levels=[0.5], colors=[palette[k]], linewidths=1.0)
    ax.contour(mg[sub], levels=[0.5], colors=C_MAIN, linewidths=1.8)

    if labels:
        for r in others:
            if r["in_crop"] < args.min_area:
                continue
            ax.text(r["x"], r["y"], str(r["id"]), color="white", fontsize=10,
                    fontweight="bold", ha="center", va="center",
                    path_effects=[pe.withStroke(linewidth=2.4, foreground="black")])

    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)

    out = Path(args.out) if args.out else run.figdir("masking") / "halo_sources.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="black")
    plt.close(fig)
    print(f"\nsaved -> {out}")


def load(n):
    """One professor pointing: its segmentation and the pseudo_r it was detected on."""
    seg = fits.getdata(SEGDIR / f"DATACUBE_FINAL_{n}_seg.fits").astype(int)
    img = np.asarray(fits.getdata(SEGDIR / f"DATACUBE_FINAL_{n}_pseudo_r.fits"), float)
    return seg, img


def overview(ns, out_path):
    """All 14 pointings seen together -- a single figure cannot answer "is this set of
    masks consistent across the whole mosaic"."""
    ncol = 5
    nrow = int(np.ceil(len(ns) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.5 * nrow))
    for a in np.ravel(ax):
        a.axis("off")
    for k, n in enumerate(ns):
        seg, img = load(n)
        a = np.ravel(ax)[k]
        a.axis("on"); a.set_xticks([]); a.set_yticks([])
        # the stretch matches id_map, so the overview and the single figures look
        # like the same data.
        bg, vmax = arcsinh_stretch(img)
        a.imshow(bg, origin="lower", cmap="gray", vmin=0, vmax=vmax)
        a.contour(seg > 0, levels=[0.5], colors="#1f77b4", linewidths=0.6)
        n_src = int(seg.max())
        a.set_title(f"#{n}   {n_src} src   "
                    f"{100 * (seg > 0).mean():.1f}% masked", fontsize=9)
    fig.suptitle("professor's segmentation on pseudo-r  (blue = source)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def professor(args):
    """The segmentations the professor supplied: one ID map per pointing, plus one
    overview."""
    ns = POINTINGS if args.n is None else args.n

    print(f"{'':>4}{'n_src':>6}{'src_px':>10}{'field%':>8}"
          f"{'top-3 flux share':>18}")
    for n in ns:
        seg, img = load(n)
        ids, n_all = source_ids(seg, args.min_area)
        rows = id_rows(seg, ids)
        # EVAL and not Run.figdir: this reads the segmentation input, not any
        # run's products, so there is no run directory for the figure to sit beside.
        out = (EVAL / f"p{n:02d}"); out.mkdir(parents=True, exist_ok=True)
        out = out / "segmentation_map.png"
        id_map(seg, img, rows, out, by_group=False)

        # A main source split into several pieces shows up as a divided flux share,
        # so measure it here rather than in a separate script. The flux is integrated
        # over pseudo_r, the quantity detection used.
        flux = {int(i): float(np.nansum(np.where(seg == i, img, 0)))
                for i in np.unique(seg) if i > 0}
        tot  = sum(flux.values()) or 1.0
        top  = sorted(flux.values(), reverse=True)[:3]
        print(f"#{n:<3}{n_all:>6}{int((seg > 0).sum()):>10,}"
              f"{100 * (seg > 0).mean():>7.1f}%"
              + "".join(f"{100 * f / tot:>6.1f}%" for f in top))
    print("\none map per pointing -> results/skymodel/evaluation/pNN/segmentation_map.png")
    if args.overview is not False:
        overview(ns, PROF_FIGURES / "overview.png")


def main():
    ap = argparse.ArgumentParser(
        description="Source map of any segmentation: the whole field, a zoom on the "
                    "main galaxy, or the fourteen the professor supplied")

    # ---- which figure -------------------------------------------------------
    ap.add_argument("--professor", action="store_true",
                    help="the fourteen segmentations the professor supplied: one ID "
                         "map per pointing over that pointing's own pseudo_r, plus "
                         "one overview. Reads data/wsky_seg and so takes -n, not --work")
    ap.add_argument("--crop", choices=["field", "main"], default="field",
                    help="field: the whole pointing. main: the main source group's "
                         "bounding box plus --margin, stretched harder, for what is "
                         "detected on and around the galaxy's extended light")
    ap.add_argument("--style", choices=["locator", "slide"], default="locator",
                    help="--crop field only. locator: axes, title and ID labels, the "
                         "map to read a source's x/y off while working. slide: the "
                         "same field with nothing around it. See the module docstring "
                         "for what the two share")

    # ---- what to read (both crops) ------------------------------------------
    ap.add_argument("--work", default=None,
                    help="pointing work directory, e.g. results/skymodel/p01. "
                         "Required for both crops, and not taken by --professor; "
                         "seg and background default to its step01/")
    ap.add_argument("--seg", default=None,
                    help="both crops: segmentation image to plot; defaults to "
                         "step01/segmentation_input.fits in the work directory")
    ap.add_argument("--white", default=None,
                    help="both crops: background image; defaults to "
                         "step01/whitelight_nosky.fits. when using a different seg, "
                         "the background must still be that pointing's own whitelight "
                         "-- a mismatched background creates a false impression of "
                         "misalignment")
    ap.add_argument("--min-area", type=int, default=1,
                    help="every mode: only plot sources with area >= this value; "
                         "tiny sources overlap when labeled. Under --crop main it is "
                         "the labels alone that go, and on the area inside the crop -- "
                         "the outline is what says a detection is there")

    # ---- shared by --style slide and --crop main ----------------------------
    ap.add_argument("--alpha", type=float, default=None,
                    help=f"fill opacity; default {ALPHA_SLIDE} on the slide, "
                         f"{ALPHA_MAIN} for the neighbouring sources of --crop main. "
                         "--style slide and --crop main only")
    ap.add_argument("--soft", type=float, default=None,
                    help="asinh softening of the background; smaller pulls the "
                         f"faint end up further. Default {SOFT_SLIDE} on the slide, "
                         f"{SOFT_MAIN} for --crop main, where the halo is the subject. "
                         "--style slide and --crop main only, and ignored with "
                         "--no-background")
    ap.add_argument("--labels", action=argparse.BooleanOptionalAction, default=None,
                    help="write each source's ID on it. --style slide and --crop main "
                         "only; off by default on the slide -- see ALPHA_SLIDE -- and "
                         "on by default in the crop, where the IDs are what lets a "
                         "source be looked up in the whole-field map. The locator map "
                         "is a lookup table and always labels")
    ap.add_argument("--dpi", type=int, default=None,
                    help=f"default {DPI_SLIDE} for the slide, {DPI_MAIN} for --crop "
                         "main; the locator map is saved at the dpi products.id_map "
                         "draws every ID map at. --style slide and --crop main only")

    # ---- --style slide only -------------------------------------------------
    ap.add_argument("--color", default=COLOR, help="source colour. --style slide only")
    ap.add_argument("--no-background", action="store_true",
                    help="mask only, on black: which spaxels are source and nothing "
                         "else. --style slide only")
    ap.add_argument("--per-source-color", action="store_true",
                    help="one colour per source instead of one for all, so "
                         "neighbours stay apart with no background to separate them. "
                         "--style slide only")
    ap.add_argument("--min-x", type=int, default=None,
                    help="drop sources whose centroid is left of this column. The "
                         "field edge produces a strip of narrow detections that are "
                         "not sources; area alone does not separate them. "
                         "--style slide only")

    # ---- --crop main only ---------------------------------------------------
    ap.add_argument("--margin", type=int, default=MARGIN,
                    help="--crop main only: pixels of sky to keep outside the main "
                         "group's bounding box")
    ap.add_argument("--step04", default=None,
                    help="--crop main only: step04 directory, e.g. "
                         "results/skymodel/p01/step04. Given, the main group keeps "
                         "only members matching the main source's redshift; omitted, "
                         "adjacency alone decides")
    ap.add_argument("--dz-max", type=float, default=DZ_MAX,
                    help="--crop main only: maximum redshift difference from the main "
                         "source, with --step04")
    ap.add_argument("--fill-main", action="store_true",
                    help="--crop main only: fill the main group too. Off by default: "
                         "the fill covers the extended light that crop is for")
    ap.add_argument("--width", type=float, default=WIDTH,
                    help="--crop main only: figure width in inches; the height follows "
                         "the crop's aspect ratio")

    # ---- --professor only ---------------------------------------------------
    ap.add_argument("-n", type=int, nargs="+", default=None,
                    help="--professor only: which pointings to plot; all fourteen "
                         "by default")
    ap.add_argument("--overview", action=argparse.BooleanOptionalAction, default=None,
                    help="--professor only: the fourteen pointings on one figure, "
                         "which is the only way to see whether the set of masks is "
                         "consistent across the mosaic. On by default")

    ap.add_argument("--out", default=None,
                    help="both crops; --professor writes one file per pointing plus "
                         "the overview, so there is no single path to give it")
    args = ap.parse_args()

    mode = ("professor" if args.professor
            else "main" if args.crop == "main"
            else args.style)
    CROPS = ("locator", "slide", "main")    # every mode that draws one run's field
    FIELD = ("locator", "slide")
    ZOOMED = ("slide", "main")              # the two that draw the pixels themselves

    # An option belonging to another mode is a mistake, not a no-op: --style locator is
    # drawn by products.id_map, which takes none of the slide's options, and
    # --professor reads no run at all. Saying so beats passing --color and getting the
    # same figure back without being told why.
    misplaced = [(f, where) for f, given, modes, where in (
        ("--work", args.work is not None, CROPS, "--crop field and --crop main"),
        ("--seg", args.seg is not None, CROPS, "--crop field and --crop main"),
        ("--white", args.white is not None, CROPS, "--crop field and --crop main"),
        ("--out", args.out is not None, CROPS, "--crop field and --crop main"),
        ("--style", args.style != "locator", FIELD, "--crop field"),
        ("--alpha", args.alpha is not None, ZOOMED, "--style slide and --crop main"),
        ("--soft", args.soft is not None, ZOOMED, "--style slide and --crop main"),
        ("--labels", args.labels is not None, ZOOMED, "--style slide and --crop main"),
        ("--dpi", args.dpi is not None, ZOOMED, "--style slide and --crop main"),
        ("--color", args.color != COLOR, ("slide",), "--style slide"),
        ("--no-background", args.no_background, ("slide",), "--style slide"),
        ("--per-source-color", args.per_source_color, ("slide",), "--style slide"),
        ("--min-x", args.min_x is not None, ("slide",), "--style slide"),
        ("--margin", args.margin != MARGIN, ("main",), "--crop main"),
        ("--step04", args.step04 is not None, ("main",), "--crop main"),
        ("--dz-max", args.dz_max != DZ_MAX, ("main",), "--crop main"),
        ("--fill-main", args.fill_main, ("main",), "--crop main"),
        ("--width", args.width != WIDTH, ("main",), "--crop main"),
        ("-n", args.n is not None, ("professor",), "--professor"),
        ("--overview", args.overview is not None, ("professor",), "--professor"),
    ) if given and mode not in modes]
    if misplaced:
        raise SystemExit("\n".join(f"{f}: {w} only" for f, w in misplaced))

    if args.professor:
        professor(args)
    elif args.work is None:
        # Not required=True on the option: --professor is over the fourteen
        # segmentations as they were given, and has no run directory to be pointed at.
        raise SystemExit("--work is required unless --professor")
    elif args.crop == "main":
        crop_main(args)
    else:
        field(args)


if __name__ == "__main__":
    main()
