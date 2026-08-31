"""The white light image of the input cube (with sky) -- what the sky looks like, and
whether the mask covers the galaxy.

This draws the raw data before any sky subtraction, not our product, and answers two
questions that belong before the pipeline: what the spatial structure of the sky is
(flat? striped? how large?), and whether the outline of seg covers the range the galaxy
extends to. Whatever it does not cover is taken as blank and used to learn the sky,
which amounts to learning the galaxy's own light as sky and then subtracting it.

The sky continuum is a large pedestal and the structure of interest a small fraction of
it, so stretching the raw values would spend the whole colour scale on the pedestal and
give a uniform figure -- flat because the scale was wasted, not because the data are.
The blank median is subtracted first, the result divided by the robust spread of blank,
and asinh applied last:

    z = (flux − sky) / sigma_blank

0 is the sky and the unit of the colour scale becomes "how many sigma above the sky",
which puts the sky's striping, the faint sources and the galaxy core on one scale.

Bad voxels are rejected by common.collapse (a sigma-clip across spaxels, clipping only
blank), so cosmic rays leave no spurious bright spots.

    conda run -n astro python src/skymodel/evaluation/whitelight_wsky.py --work results/skymodel/p01
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (asinh_bar, band_tag, collapse, seg_outline,  # noqa: E402
                    sigma_image)
from config import resolve_path  # noqa: E402
from products import Run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="White light image of the input cube (with sky)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--cube", default=None,
                    help="input cube; defaults to data/wsky/DATACUBE_FINAL_N.fits inferred from pNN")
    ap.add_argument("--band", type=float, nargs=2, default=(4600, 9350))
    ap.add_argument("--vmin-sigma", type=float, default=-3.0,
                    help="colour scale lower bound, in units of blank sigma")
    args = ap.parse_args()

    run = Run(args.work)
    # The cube the run was actually given, out of its own config -- not the pointing
    # number and a path rebuilt from it, which only finds data kept inside the repo.
    cube = resolve_path(args.cube) if args.cube else run.wsky

    seg = run.seg
    img, nbad, ntot = collapse(cube, args.band, run.wl, seg)
    valid = np.isfinite(img) & (img != 0)
    img = np.where(valid, img, np.nan)

    blank = valid & (seg == 0)
    sky = float(np.nanmedian(img[blank]))
    # Percentiles rather than the standard deviation: any source light falling inside
    # blank would inflate it, loosening the colour scale until the striping is lost.
    sig = float(np.nanpercentile(img[blank], 84)
                - np.nanpercentile(img[blank], 16)) / 2
    z = (img - sky) / sig
    print(f"  {cube.name}  rejected bad voxels {nbad:,}/{ntot:,} ({100*nbad/ntot:.3f}%)")
    print(f"  sky {sky:.2f}   blank spread {sig:.3f} ({100*sig/sky:.2f}% of sky)"
          f"   peak {np.nanmax(img):.0f} = {np.nanmax(img)/sky:.1f}x sky")

    fig, ax = plt.subplots(figsize=(9.5, 9))
    im = sigma_image(ax, z, args.vmin_sigma, np.nanmax(z))
    seg_outline(ax, seg, lw=0.6, alpha=.8)
    asinh_bar(fig, im, ax, "signal above sky   [$\\sigma$ of blank]",
              args.vmin_sigma, float(np.nanmax(z)))
    ax.set_title(f"{run.name}   {cube.name} (with sky)   "
                 f"{args.band[0]:.0f}-{args.band[1]:.0f} A", fontsize=13)
    fig.tight_layout()
    o = run.figdir("whitelight") / f"wsky{band_tag(args.band)}.png"
    fig.savefig(o, dpi=150, bbox_inches="tight")
    print(f"saved -> {o}")


if __name__ == "__main__":
    main()
