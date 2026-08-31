"""Zoom on the main galaxy: what is detected on and around its extended light.

The whole-field figures (seg_id_map, seg_slide_map) are locator maps, and at that
scale the halo and the few-pixel detections on it are too small to answer the two
questions this figure is for: does the segmentation stop where the light stops, and
what else did SExtractor detect inside and around that extended light?

So this crops to the main source group's bounding box plus a margin and stretches the
background hard enough for the halo to show. The main group is drawn as a contour only
-- a translucent fill would sit on exactly the light the figure exists to show. Every
other source in the crop is filled, outlined and labelled with its seg ID, so anything
found here can be looked up in the whole-field map.

The main group is the connected blob containing the brightest pixel
(utils.main_source_group); with --step04 its members must also share the main source's
redshift. Adjacency alone can chain through touching neighbours, so the printed ID list
and bbox make a chained blob visible. The margin is clipped to the data, and the
printout says how much room each side actually had.

    conda run -n astro python src/skymodel/evaluation/halo_sources.py --work results/skymodel/p01 \\
        --margin 60 --soft 0.005
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL, ROOT, arcsinh_stretch, load_field, pointing_dir, qualitative  # noqa: E402
from utils import DZ_MAX, main_source_group  # noqa: E402

FIGURES = EVAL / "masking"

# The main group's outline, in the colour utils.plot_main_group uses for the same
# object, so the two figures cannot be misread as two different footprints.
C_MAIN = "#ff7f0e"
# How far past the main group's bounding box to show. The margin exists for the light
# outside the boundary, so it must be a real fraction of the galaxy, not framing.
MARGIN = 45
# asinh softening of the background. arcsinh_stretch's project default is set for the
# whole field; here the halo is the subject, so the faint end is pulled up further.
SOFT = 0.004
# Fill opacity of the neighbouring sources. They are drawn over sky, not over the
# halo's interesting part, so they can be solid enough to read at a glance.
ALPHA = 0.40


def main():
    ap = argparse.ArgumentParser(
        description="Zoomed map of the sources on and around the main galaxy's extended light")
    ap.add_argument("--work", required=True,
                    help="pointing work directory, e.g. results/skymodel/p01")
    ap.add_argument("--margin", type=int, default=MARGIN,
                    help="pixels of sky to keep outside the main group's bounding box")
    ap.add_argument("--soft", type=float, default=SOFT,
                    help="asinh softening; smaller pulls the faint end up further")
    ap.add_argument("--alpha", type=float, default=ALPHA,
                    help="fill opacity of the neighbouring sources")
    ap.add_argument("--step04", default=None,
                    help="step04 directory, e.g. results/skymodel/p01/step04. Given, "
                         "the main group keeps only members matching the main source's "
                         "redshift; omitted, adjacency alone decides")
    ap.add_argument("--dz-max", type=float, default=DZ_MAX,
                    help="maximum redshift difference from the main source, with --step04")
    ap.add_argument("--min-area", type=int, default=1,
                    help="do not label sources smaller than this; their labels overlap")
    ap.add_argument("--no-labels", action="store_true",
                    help="drop the ID numbers and keep the outlines alone")
    ap.add_argument("--fill-main", action="store_true",
                    help="fill the main group too. Off by default: the fill covers the "
                         "extended light this figure is for")
    ap.add_argument("--width", type=float, default=13,
                    help="figure width in inches; the height follows the crop's aspect ratio")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    W = ROOT / args.work
    seg, white, valid = load_field(W)
    wn = np.where(valid, white, np.nan)
    step04 = Path(args.step04) if args.step04 else None
    mg, ids, pk = main_source_group(seg, wn, step04, args.dz_max)

    H, Xw = seg.shape
    ys, xs = np.nonzero(mg)
    # np.clip, not max/min by hand: the margin can run off the field.
    y0, y1 = int(np.clip(ys.min() - args.margin, 0, H)), int(np.clip(ys.max() + 1 + args.margin, 0, H))
    x0, x1 = int(np.clip(xs.min() - args.margin, 0, Xw)), int(np.clip(xs.max() + 1 + args.margin, 0, Xw))
    sub = np.s_[y0:y1, x0:x1]

    print(f"{Path(args.work).name}: field {H}x{Xw}, main group {len(ids)} ids {ids}, "
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
    stretched, vmax = arcsinh_stretch(white, valid, soft=args.soft)
    bg = stretched[sub]

    h, w = bg.shape
    fig, ax = plt.subplots(figsize=(args.width, args.width * h / w))
    ax.imshow(bg, origin="lower", cmap="gray", vmin=0, vmax=vmax)

    palette = qualitative(len(others))
    rgba = np.zeros(bg.shape + (4,))
    for k, r in enumerate(others):
        rgba[(seg == r["id"])[sub]] = list(palette[k]) + [args.alpha]
    if args.fill_main:
        rgba[mg[sub]] = list(matplotlib.colors.to_rgb(C_MAIN)) + [args.alpha]
    ax.imshow(rgba, origin="lower")

    for k, r in enumerate(others):
        ax.contour((seg == r["id"])[sub], levels=[0.5], colors=[palette[k]], linewidths=1.0)
    ax.contour(mg[sub], levels=[0.5], colors=C_MAIN, linewidths=1.8)

    if not args.no_labels:
        for r in others:
            if r["in_crop"] < args.min_area:
                continue
            ax.text(r["x"], r["y"], str(r["id"]), color="white", fontsize=10,
                    fontweight="bold", ha="center", va="center",
                    path_effects=[pe.withStroke(linewidth=2.4, foreground="black")])

    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)

    name = Path(args.work).name
    out = Path(args.out) if args.out else pointing_dir(W, "masking") / "halo_sources.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor="black")
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
