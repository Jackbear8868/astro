"""The sky-subtracted white light image -- ESO and ours side by side.

Why look at the white light image
---------------------------------
The per-ring and per-box numbers tell you "by how much it differs", but they cannot
tell you "where it differs". Sky that was not removed cleanly, if it is an offset
uniform over the whole field, is a completely different disease from "an extra ring
subtracted next to the source", and the two may give the same number in a regional
average. An image lays the spatial structure out.

The two figures
---------------
The same colour scale, so that "which one is brighter" can be seen -- with different
colour scales, the difference between the two sides could not be told apart from the
ruler. The ruler is arcsinh(flux / blank spread), where 0 is the true value of "the
sky was removed cleanly", in units of sigma. No pedestal is subtracted: the layer of
background ESO leaves behind shows up in the figure as a uniform brightness, and that
is real information.

How to read it
--------------
    blank region      whoever is closer to black had their sky removed more cleanly
    on and around     the same brightness on both sides = the source was preserved
    the sources       everywhere; darker on our side = we subtracted the source away

The difference between the two is printed to the terminal (the whole-field zero point,
and how much is left on the source after that zero point is removed) rather than drawn
as a figure -- a difference image is dominated by that zero point, which takes the
whole colour scale, and the spatial structure becomes invisible instead.

    conda run -n astro python src/skymodel/evaluation/whitelight_compare.py --work results/skymodel/p01
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (ROOT, SEG_COLOR, asinh_bar, collapse, load_field,
                    pointing_dir)  # noqa: E402
from utils import fit_dirs, main_source_group, scale  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Sky-subtracted white light image: ESO vs ours")
    ap.add_argument("--work", required=True)
    ap.add_argument("--run", default=None,
                    help="alternative run directory under step05; default is the pipeline's own step05/step06")
    ap.add_argument("--eso", default=None,
                    help="ESO nosky cube; defaults to inferring from pNN number")
    ap.add_argument("--band", type=float, nargs=2, default=(4600, 9350))
    args = ap.parse_args()

    W = ROOT / args.work
    _, run = fit_dirs(W, args.run)
    n = int(W.name[1:])
    eso = ROOT / (args.eso or f"data/nosky/DATACUBE_FINAL_ESOSKY_{n}.fits")

    seg, white, valid = load_field(W)
    main, ids, _ = main_source_group(seg, np.where(valid, white, np.nan),
                                     W / "step04")
    wl = np.load(W / "step03/wavelength.npy")

    imgs = {}
    for lab, p in (("ESO nosky", eso), ("ours", run / "sky_subtracted.fits")):
        img, nbad, ntot = collapse(p, args.band, wl, seg)
        imgs[lab] = np.where(valid, img, np.nan)
        print(f"  {lab:>10}  rejected bad voxels {nbad:,}/{ntot:,} ({100*nbad/ntot:.3f}%)")

    # The difference between the two has a zero point uniform over the whole field:
    # ESO leaves a layer of background across the entire field of view, the same on
    # blank and on the sources. Only after removing it is what remains on the source
    # "how much more of the source's light we kept than ESO did".
    # What is subtracted is the median, not the mean -- the source residual would drag
    # the mean away.
    d0  = imgs["ours"] - imgs["ESO nosky"]
    off = float(np.nanmedian(d0[valid & (seg == 0)]))
    d   = d0 - off
    print(f"  diff: blank median {off:+.4f} (field-wide zero point)")
    print(f"      after removing zero point  main source median {np.nanmedian(d[main]):+.4f}   "
          f"blank {np.nanmedian(d[valid & (seg == 0)]):+.4f}")

    # The ruler: divide by **our** blank spread, with no pedestal subtracted. After
    # the sky is removed the true value of blank is 0, so 0 can be used directly as
    # the reference -- and the layer of background ESO leaves behind shows up in the
    # figure as a uniform brightness, which is real information, so subtracting a
    # pedestal would only hide it.
    # Both panels share the same ruler, otherwise "which one is brighter" would be an
    # artefact created by the colour scale.
    sig = scale(imgs["ours"][valid & (seg == 0)])
    zmax = np.arcsinh(np.nanmax(imgs["ours"][valid]) / sig)
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.8))
    for a, lab in zip(ax, ("ESO nosky", "ours")):
        im = a.imshow(np.arcsinh(imgs[lab] / sig), origin="lower", cmap="magma",
                      vmin=np.arcsinh(-3.0), vmax=zmax)
        a.set_title(lab, fontsize=12)
        asinh_bar(fig, im, a, "signal   [$\\sigma$ of blank]",
                  -3.0, float(np.nanmax(imgs["ours"][valid]) / sig))

    for a in ax:
        # Only the segmentation is drawn. The main source group gets no extra outline
        # -- it is a source in seg just like the others, and singling it out would
        # make people think that line stands for something else.
        a.contour(seg > 0, levels=[0.5], colors=SEG_COLOR, linewidths=0.5,
                  alpha=.75)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{W.name}    {args.band[0]:.0f}-{args.band[1]:.0f} A", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    band = ("" if tuple(args.band) == (4600.0, 9350.0)
            else f"_{args.band[0]:.0f}-{args.band[1]:.0f}")
    o = pointing_dir(W.name, "whitelight") / f"compare{band}.png"
    fig.savefig(o, dpi=140, bbox_inches="tight")
    print(f"saved -> {o}")


if __name__ == "__main__":
    main()
