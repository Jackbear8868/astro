"""The sky-subtracted white light image -- ESO and ours side by side.

The per-ring and per-box numbers say by how much the two differ, not where. A uniform
offset over the whole field is a different disease from an extra ring subtracted next
to the source, and both can give the same regional average; an image lays the spatial
structure out.

The two panels share one colour scale, or the difference between them could not be
told apart from the ruler. The ruler is arcsinh(flux / blank spread), in units of
sigma, where 0 is the true value once the sky is removed cleanly. No pedestal is
subtracted, so any uniform background left behind stays visible as brightness.

    blank region      whoever is closer to black had their sky removed more cleanly
    on and around     the same brightness on both sides = the source was preserved
    the sources       everywhere; darker on our side = we subtracted the source away

The difference itself is printed rather than drawn: a difference image is dominated by
its field-wide zero point, which takes the whole colour scale and hides the structure.

    conda run -n astro python src/skymodel/evaluation/whitelight_compare.py --work results/skymodel/p01
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import SEG_COLOR, asinh_bar, collapse  # noqa: E402
from config import resolve_path  # noqa: E402
from products import Run  # noqa: E402
from utils import main_source_group, robust_spread  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Sky-subtracted white light image: ESO vs ours")
    ap.add_argument("--work", required=True)
    ap.add_argument("--run", default=None,
                    help="alternative run directory under step05; default is the pipeline's own step05/step06")
    ap.add_argument("--eso", default=None,
                    help="ESO nosky cube; defaults to inferring from pNN number")
    ap.add_argument("--band", type=float, nargs=2, default=(4600, 9350))
    args = ap.parse_args()

    run = Run(args.work, args.run)
    # ESO's cube out of the run's own config, so a run reading data from outside the
    # checkout compares against the file it was actually given.
    eso = resolve_path(args.eso) if args.eso else run.nosky

    seg, white, valid = run.seg, run.white, run.valid
    main, ids, _ = main_source_group(seg, np.where(valid, white, np.nan), run.step04)

    imgs = {}
    for lab, p in (("ESO nosky", eso), ("ours", run.cube)):
        img, nbad, ntot = collapse(p, args.band, run.wl, seg)
        imgs[lab] = np.where(valid, img, np.nan)
        print(f"  {lab:>10}  rejected bad voxels {nbad:,}/{ntot:,} ({100*nbad/ntot:.3f}%)")

    # The difference carries a field-wide zero point, the same on blank and on the
    # sources; only with it gone does the source residual mean "how much more of the
    # source's light we kept". The median, not the mean -- the source residual would
    # drag the mean away.
    d0  = imgs["ours"] - imgs["ESO nosky"]
    off = float(np.nanmedian(d0[valid & (seg == 0)]))
    d   = d0 - off
    print(f"  diff: blank median {off:+.4f} (field-wide zero point)")
    print(f"      after removing zero point  main source median {np.nanmedian(d[main]):+.4f}   "
          f"blank {np.nanmedian(d[valid & (seg == 0)]):+.4f}")

    # The ruler: our blank spread, with no pedestal subtracted. Once the sky is gone
    # the true value of blank is 0, so 0 serves as the reference, and any uniform
    # background left behind stays visible instead of being hidden. Both panels share
    # the ruler, or "which is brighter" would be an artefact of the colour scale.
    sig = robust_spread(imgs["ours"][valid & (seg == 0)])
    zmax = np.arcsinh(np.nanmax(imgs["ours"][valid]) / sig)
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.8))
    for a, lab in zip(ax, ("ESO nosky", "ours")):
        im = a.imshow(np.arcsinh(imgs[lab] / sig), origin="lower", cmap="magma",
                      vmin=np.arcsinh(-3.0), vmax=zmax)
        a.set_title(lab, fontsize=12)
        asinh_bar(fig, im, a, "signal   [$\\sigma$ of blank]",
                  -3.0, float(np.nanmax(imgs["ours"][valid]) / sig))

    for a in ax:
        # Only the segmentation is drawn: the main group is a source in seg like any
        # other, and a second outline would look like it meant something else.
        a.contour(seg > 0, levels=[0.5], colors=SEG_COLOR, linewidths=0.5,
                  alpha=.75)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{run.name}    {args.band[0]:.0f}-{args.band[1]:.0f} A", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    band = ("" if tuple(args.band) == (4600.0, 9350.0)
            else f"_{args.band[0]:.0f}-{args.band[1]:.0f}")
    o = run.figdir("whitelight") / f"compare{band}.png"
    fig.savefig(o, dpi=140, bbox_inches="tight")
    print(f"saved -> {o}")


if __name__ == "__main__":
    main()
