"""The source map with nothing around it -- for slides.

This is a different figure from seg_id_map, not an option on it. seg_id_map is the
locator map used while working: the axes are part of what it does, because reading a
source's x/y off it is how you go and look at that spaxel. On a slide nothing outside
the image is read, so the title, the axes and the frame are noise, and the image is
made to fill the canvas so no white margin appears against the slide background.

What the two share is deliberate: the same asinh stretch, the same translucent fill
plus contour. Two figures of the same field drawn by different rules would look like
different data. Only the colour and the surroundings differ.

The default colour is green rather than the locator map's pale red, which sits too
close to greyscale in brightness to separate cleanly when projected.

    conda run -n astro python src/skymodel/evaluation/seg_slide_map.py \\
        --work results/skymodel/p01
    conda run -n astro python src/skymodel/evaluation/seg_slide_map.py \\
        --work results/skymodel/p01 --labels --min-x 15 --color '#22d3ee'
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL, ROOT  # noqa: E402
from utils import arcsinh_stretch  # noqa: E402

FIGURES = EVAL / "masking"


def qualitative(n):
    """n colours that neighbouring sources will not be confused between.

    The three tab20 families together give 60 before anything repeats, which
    covers a MUSE pointing's source count. Past that the cycle restarts, and two
    sources sharing a colour is only a problem if they are adjacent -- the ID
    order is not spatial, so a repeat lands somewhere else in the field.
    """
    cols = (list(plt.get_cmap("tab20").colors)
            + list(plt.get_cmap("tab20b").colors)
            + list(plt.get_cmap("tab20c").colors))
    return [cols[i % len(cols)] for i in range(n)]
COLOR = "#4ade80"
# At slide size the fill is there to say where the sources are, not to be read
# through -- 0.25 marks the extent while leaving the body of Haro 11 visible
# underneath. The IDs are off by default for the same reason: 60 numbers, many of
# them overlapping along the field edge, are unreadable from the back of a room.
ALPHA = 0.25
# asinh softening. The percentile inside arcsinh_stretch sets what counts as the
# bright end; this sets how far the faint end is pulled up, and it is the one that
# decides whether the faint sources are visible at all. arcsinh_stretch's own
# default is shared by every figure in the project, so it is passed in here rather
# than changed there.
SOFT = 0.01


def slide_map(seg, white, out, color=COLOR, alpha=ALPHA, labels=False,
              ids=None, dpi=150, soft=SOFT, background=True, per_source=False):
    """Draw the map and write it to `out`.

    ids limits which sources are drawn; None draws every label in seg.

    background=False drops the white light and leaves the mask alone on black --
    the figure then says "these spaxels are source, those are not" and nothing
    else. With nothing underneath to see through, the fill is drawn solid.

    per_source=True gives each source its own colour from a qualitative palette,
    so neighbouring sources stay apart from each other. It is for the mask figure,
    where there is no background to separate them; over the white light the
    contours already do that, and many colours there would read as if the colour
    meant something about the source.
    """
    if ids is None:
        ids = np.unique(seg[seg > 0])
    keep = np.isin(seg, ids) & (seg > 0)

    h, w = white.shape
    fig, ax = plt.subplots(figsize=(12, 12 * h / w))
    # The project's one stretch, the same one utils.id_map uses, so the two
    # figures of the same field cannot end up looking like different data.
    if background:
        bg, vmax = arcsinh_stretch(white, soft=soft)
        ax.imshow(bg, origin="lower", cmap="gray", vmin=0, vmax=vmax)
    else:
        ax.imshow(np.zeros(seg.shape), origin="lower", cmap="gray", vmin=0, vmax=1)
        alpha = 1.0

    # One fill for all sources, one contour per source: the fill shows extent and
    # the contour keeps small sources visible, but the fill does not need to be
    # drawn once per source the way the contour does.
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


def main():
    ap = argparse.ArgumentParser(description="Source map without axes, for slides")
    ap.add_argument("--work", required=True,
                    help="pointing work directory, e.g. results/skymodel/p01. "
                         "seg and background default to its step01/")
    ap.add_argument("--seg", default=None,
                    help="segmentation image; defaults to step01/seg.fits")
    ap.add_argument("--white", default=None,
                    help="background image; defaults to step01/whitelight.fits. "
                         "with a different seg the background must still be this "
                         "pointing's own whitelight, or the two look misaligned")
    ap.add_argument("--color", default=COLOR, help="source colour")
    ap.add_argument("--alpha", type=float, default=ALPHA, help="fill opacity")
    ap.add_argument("--soft", type=float, default=SOFT,
                    help="asinh softening of the background; smaller pulls the "
                         "faint end up further. Ignored with --no-background")
    ap.add_argument("--no-background", action="store_true",
                    help="mask only, on black: which spaxels are source and nothing else")
    ap.add_argument("--per-source-color", action="store_true",
                    help="one colour per source instead of one for all, so "
                         "neighbours stay apart with no background to separate them")
    ap.add_argument("--labels", action="store_true",
                    help="write each source's ID on it. Off by default -- see ALPHA")
    ap.add_argument("--min-x", type=int, default=None,
                    help="drop sources whose centroid is left of this column. The "
                         "field edge produces a strip of narrow detections that are "
                         "not sources; area alone does not separate them")
    ap.add_argument("--min-area", type=int, default=1,
                    help="drop sources smaller than this")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    STEP01 = ROOT / args.work / "step01"
    white = fits.getdata(Path(args.white) if args.white
                         else STEP01 / "whitelight.fits")
    seg_path = Path(args.seg) if args.seg else STEP01 / "seg.fits"
    seg = fits.getdata(seg_path)
    if seg.shape != white.shape:
        raise SystemExit(f"seg {seg.shape} and whitelight {white.shape} have different dimensions")

    all_ids, cnt = np.unique(seg[seg > 0], return_counts=True)
    ids = []
    for i, c in zip(all_ids, cnt):
        if c < args.min_area:
            continue
        if args.min_x is not None and np.nonzero(seg == i)[1].mean() < args.min_x:
            continue
        ids.append(int(i))

    # The output name carries both the pointing and the seg's parent directory: two
    # different segmentations are both called seg.fits, so the stem alone collides
    # and one silently overwrites the other.
    name = f"{Path(args.work).name}_{seg_path.parent.name}_{seg_path.stem}"
    out = Path(args.out) if args.out else FIGURES / f"slide_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    slide_map(seg, white, out, color=args.color, alpha=args.alpha,
              labels=args.labels, ids=ids, dpi=args.dpi, soft=args.soft,
              background=not args.no_background,
              per_source=args.per_source_color)
    print(f"{name}: {len(all_ids)} sources, drawn {len(ids)}")
    print(f"  source spaxels {int((seg > 0).sum()):,}  ({100 * (seg > 0).mean():.1f}% of field)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
