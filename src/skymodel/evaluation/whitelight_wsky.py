"""The white light image of the input cube (with sky) -- what the sky looks like, and
whether the mask covers the galaxy.

What this draws is **the raw data before any sky subtraction**, not our product. It
answers two questions that should be settled before entering the pipeline:

    the spatial structure   is it flat? is there striping? how large is the
    of the sky              amplitude?
    is the mask big         does the outline of seg cover the range the galaxy
    enough                  actually extends to -- the part it does not cover gets
                            taken as blank and used to learn the sky, which amounts
                            to learning the galaxy's own light as sky and then
                            subtracting it

Where the contrast comes from
-----------------------------
The sky continuum is a pedestal of about 52, while the structure we want to look at is
only 1% of it. Stretching the raw values directly, the pedestal would use up the whole
colour scale and the figure would be a uniform colour -- not because the data are
flat, but because the colour scale was wasted.

So the blank median is subtracted first, then it is divided by the robust spread of
blank, and only then asinh is applied:

    z = (flux − sky) / sigma_blank

0 is the sky, the unit of the colour scale becomes "how many sigma above the sky", and
the three levels -- the striping of the sky (about 1 sigma), the faint sources (2-4)
and the core of the galaxy (about 9) -- are all visible at the same time.

Bad voxels are rejected by common.collapse (a sigma-clip across spaxels, clipping only
blank), so cosmic rays do not leave spurious bright spots on the figure.

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
from common import (ROOT, SEG_COLOR, asinh_bar, collapse, load_field,
                    pointing_dir)  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="White light image of the input cube (with sky)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--cube", default=None,
                    help="input cube; defaults to data/wshy/DATACUBE_FINAL_N.fits inferred from pNN")
    ap.add_argument("--band", type=float, nargs=2, default=(4600, 9350))
    ap.add_argument("--vmin-sigma", type=float, default=-3.0,
                    help="colour scale lower bound, in units of blank sigma")
    args = ap.parse_args()

    W = ROOT / args.work
    n = int(W.name[1:])
    cube = ROOT / (args.cube or f"data/wshy/DATACUBE_FINAL_{n}.fits")

    seg, _, _ = load_field(W)
    wl = np.load(W / "step03/wavelength.npy")
    img, nbad, ntot = collapse(cube, args.band, wl, seg)
    valid = np.isfinite(img) & (img != 0)
    img = np.where(valid, img, np.nan)

    blank = valid & (seg == 0)
    sky = float(np.nanmedian(img[blank]))
    # The spread uses percentiles rather than the standard deviation: the part of the
    # galaxy spilling outside the mask also falls inside blank, and it would inflate
    # the standard deviation, loosening the colour scale with it until the striping is
    # no longer visible.
    sig = float(np.nanpercentile(img[blank], 84)
                - np.nanpercentile(img[blank], 16)) / 2
    z = (img - sky) / sig
    print(f"  {cube.name}  rejected bad voxels {nbad:,}/{ntot:,} ({100*nbad/ntot:.3f}%)")
    print(f"  sky {sky:.2f}   blank spread {sig:.3f} ({100*sig/sky:.2f}% of sky)"
          f"   peak {np.nanmax(img):.0f} = {np.nanmax(img)/sky:.1f}x sky")

    fig, ax = plt.subplots(figsize=(9.5, 9))
    im = ax.imshow(np.arcsinh(z), origin="lower", cmap="magma",
                   vmin=np.arcsinh(args.vmin_sigma),
                   vmax=np.arcsinh(np.nanmax(z)))
    ax.contour(seg > 0, levels=[0.5], colors=SEG_COLOR, linewidths=0.6, alpha=.8)
    ax.set_xticks([]); ax.set_yticks([])
    asinh_bar(fig, im, ax, "signal above sky   [$\\sigma$ of blank]",
              args.vmin_sigma, float(np.nanmax(z)))
    ax.set_title(f"{W.name}   {cube.name} (with sky)   "
                 f"{args.band[0]:.0f}-{args.band[1]:.0f} A", fontsize=13)
    fig.tight_layout()
    band = ("" if tuple(args.band) == (4600.0, 9350.0)
            else f"_{args.band[0]:.0f}-{args.band[1]:.0f}")
    o = pointing_dir(W.name, "whitelight") / f"wsky{band}.png"
    fig.savefig(o, dpi=150, bbox_inches="tight")
    print(f"saved -> {o}")


if __name__ == "__main__":
    main()
