"""The white light image of a pointing's cubes, drawn on one colour scale.

    whitelight.py --work results/skymodel/p01 --cubes wsky
    whitelight.py --work results/skymodel/p01 --cubes ours eso

`--cubes` says which cubes to draw, one panel each, and the two invocations above are
the two questions this figure answers -- one before the pipeline and one after.

The input cube, the sky still in it, is the field before anything was done to it. What
is the spatial structure of the sky (flat? striped? how large?), and does the outline of
seg cover the range the galaxy extends to? Whatever it does not cover is taken as blank
and used to learn the sky, which amounts to learning the galaxy's own light as sky and
then subtracting it.

The sky-subtracted cubes side by side are the result. The per-ring and per-box numbers
say by how much two subtractions differ, not where. A uniform offset over the whole
field is a different disease from an extra ring subtracted next to the source, and both
can give the same regional average; an image lays the spatial structure out.

    blank region      whoever is closer to black had their sky removed more cleanly
    on and around     the same brightness on both sides = the source was preserved
    the sources       everywhere; darker on our side = we subtracted the source away

Every panel is drawn on the ruler of the first cube listed -- its blank spread and its
zero point -- or the difference between two panels could not be told apart from the
ruler itself. asinh is applied last, which puts the sky's striping, the faint sources
and the galaxy core on one scale:

    z = (flux - pedestal) / sigma_blank

What the pedestal is follows from which cube is being examined rather than from a flag;
the comment where it is computed says why.

The difference between two panels is printed rather than drawn: a difference image is
dominated by its field-wide zero point, which takes the whole colour scale and hides the
structure.

Bad voxels are rejected by common.collapse (a sigma-clip across spaxels, clipping only
blank), so cosmic rays leave no spurious bright spots.

This is what whitelight_wsky.py and whitelight_compare.py were: the same field, the same
stretch, the same outline, the same colour bar, differing in how many panels they drew
and in what 0 on the scale meant.
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
from common import (ROOT, asinh_bar, band_tag, collapse, seg_outline,  # noqa: E402
                    sigma_image, slug)
from products import Run  # noqa: E402
from utils import main_source_group, robust_spread  # noqa: E402

# Panel size in inches and saved resolution. One panel has the canvas to itself; several
# share a figure a screen still has to hold, so each is drawn smaller -- and the wider
# figure is written coarser, or the file grows without putting more field on the screen.
PANEL_ONE, PANEL_MANY = (9.5, 9.0, 150), (7.25, 6.8, 140)


def main():
    ap = argparse.ArgumentParser(
        description="White light image of a pointing's cubes, on one colour scale")
    ap.add_argument("--work", required=True,
                    help="pointing work directory, e.g. results/skymodel/p01")
    ap.add_argument("--cubes", nargs="+", default=["ours", "eso"], metavar="CUBE",
                    help="ours (step06), eso (the nosky the config names), wsky (the "
                         "input), model (the sky taken out), run:GLOB (a run under "
                         "step05), or a path. The first is the one being examined")
    ap.add_argument("--labels", nargs="+", default=None, metavar="NAME",
                    help="names for the panels; defaults to what --cubes says")
    ap.add_argument("--band", type=float, nargs=2, default=(4600, 9350))
    ap.add_argument("--vmin-sigma", type=float, default=-3.0,
                    help="colour scale lower bound, in units of blank sigma")
    ap.add_argument("--dpi", type=int, default=None,
                    help=f"default {PANEL_ONE[2]} for one panel, "
                         f"{PANEL_MANY[2]} for several")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Run(args.work)
    labels = args.labels or args.cubes
    if len(labels) != len(args.cubes):
        raise SystemExit(f"{len(args.cubes)} cubes but {len(labels)} labels")
    paths = [run.named_cube(c) for c in args.cubes]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"{p} does not exist")
    n = len(paths)
    wid = max(6, max(len(l) for l in labels) + 2)

    # The field of view is the pointing's, not each image's: a sky-subtracted cube is
    # legitimately 0 wherever the subtraction came out exact, and reading 0 as "outside
    # the field" would throw those spaxels away. One mask for every panel besides, or
    # which panel is brighter would be a statement about two different footprints.
    seg, valid = run.seg, run.valid
    blank = valid & (seg == 0)

    # Named before the cubes are read, which is the slow part: what the figure compares
    # is the first thing to check, and it is not worth a minute's wait to see it.
    print(f"{run.name}   {args.band[0]:.0f}-{args.band[1]:.0f} A")
    for lab, p in zip(labels, paths):
        print(f"  {lab:<{wid}}{p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")

    imgs = []
    for lab, p in zip(labels, paths):
        img, nbad, ntot = collapse(p, args.band, run.wl, seg)
        imgs.append(np.where(valid, img, np.nan))
        print(f"  {lab:<{wid}}rejected bad voxels {nbad:,}/{ntot:,} "
              f"({100*nbad/ntot:.3f}%)")

    # What 0 on the colour scale means, taken from which cube is being examined rather
    # than from a flag. The input cube and the sky model still have the sky in them, and
    # the sky continuum is a large pedestal with the structure of interest a small
    # fraction of it: stretched as they stand, the whole colour scale goes on the
    # pedestal and the figure comes out uniform -- flat because the scale was wasted,
    # not because the data are. So for those two the sky level is measured off blank and
    # taken out, and the scale reads "signal above sky". A sky-subtracted cube's 0 is
    # already its true value, so nothing is taken out and any uniform background left
    # behind stays visible as brightness instead of being hidden.
    with_sky = paths[0] in (run.wsky, run.sky_model)
    sky = float(np.nanmedian(imgs[0][blank])) if with_sky else 0.0

    # Percentiles rather than the standard deviation: any source light falling inside
    # blank would inflate it, loosening the colour scale until the striping is lost.
    sig = robust_spread(imgs[0][blank])
    z = [(img - sky) / sig for img in imgs]
    hi = np.nanmax(z[0])

    if with_sky:
        # Only with the sky still in the cube: both "% of sky" and "x sky" divide by the
        # pedestal, which a sky-subtracted cube has none of.
        peak = np.nanmax(imgs[0])
        print(f"  sky {sky:.2f}   blank spread {sig:.3f} ({100*sig/sky:.2f}% of sky)"
              f"   peak {peak:.0f} = {peak/sky:.1f}x sky")

    if n > 1:
        # Which sources are the galaxy is read only for the difference statistics, so a
        # one-panel figure does not depend on step 4 products it never looks at.
        main_, _, _ = main_source_group(seg, np.where(valid, run.white, np.nan),
                                        run.step04)
        for lab, img in zip(labels[1:], imgs[1:]):
            # The difference carries a field-wide zero point, the same on blank and on
            # the sources; only with it gone does the source residual mean "how much
            # more of the source's light we kept". The median, not the mean -- the
            # source residual would drag the mean away.
            d0  = imgs[0] - img
            off = float(np.nanmedian(d0[blank]))
            d   = d0 - off
            print(f"  {labels[0]} - {lab}: blank median {off:+.4f} "
                  f"(field-wide zero point)")
            print(f"      after removing zero point  main source median "
                  f"{np.nanmedian(d[main_]):+.4f}   blank {np.nanmedian(d[blank]):+.4f}")

    pw, ph, dpi = PANEL_ONE if n == 1 else PANEL_MANY
    fig, axes = plt.subplots(1, n, figsize=(pw * n, ph))
    for a, lab, zi in zip(np.atleast_1d(axes), labels, z):
        im = sigma_image(a, zi, args.vmin_sigma, hi)
        a.set_title(lab, fontsize=12)
        asinh_bar(fig, im, a,
                  f"signal{' above sky' if with_sky else ''}   [$\\sigma$ of blank]",
                  args.vmin_sigma, float(hi))
        # Only the segmentation is outlined: the main group is a source in seg like any
        # other, and a second outline would look like it meant something else.
        seg_outline(a, seg)
    fig.suptitle(f"{run.name}    {args.band[0]:.0f}-{args.band[1]:.0f} A", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # The name says what is in the figure: which cubes, against which.
    stem = "_vs_".join(slug(l) for l in labels)
    d = Path(args.out) if args.out else run.figdir("whitelight")
    o = d if d.suffix == ".png" else d / f"{stem}{band_tag(args.band)}.png"
    o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(o, dpi=args.dpi or dpi, bbox_inches="tight")
    print(f"saved -> {o}")


if __name__ == "__main__":
    main()
