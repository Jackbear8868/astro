"""Collapse a cube along wavelength into a white light image.

Everything downstream that has to say "where is the source" works on this image
rather than on the cube: the segmentation is checked against it, the main source is
the blob holding its brightest pixel, and the evaluation figures use it as their
background.
"""
from pathlib import Path
from typing import NamedTuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import matplotlib
matplotlib.use("Agg")              # must be set before importing pyplot: render to file, not screen
import matplotlib.pyplot as plt


class WhiteLight(NamedTuple):
    """What this step hands the ones after it.

    The header travels with the image because the segmentation check asks where
    each pixel points on the sky, which the array on its own cannot answer; every
    other consumer reads only `data`.
    """
    data: np.ndarray          # (ny, nx), the collapsed image, 0 outside the field
    header: fits.Header       # the cube's celestial WCS


def whitelight(cube, out, rows=32, keep_intermediate=True):
    """Collapse `cube` along wavelength; return the image and its WCS.

    The image comes back rather than being read from the file again by everyone
    downstream: a step that reads its input from disk can be handed a file an
    earlier run left there, and nothing says so.

    With keep_intermediate the same image is written to `out` as whitelight.fits
    plus a preview png, which is what the evaluation scripts read.

    rows is the number of image rows collapsed at a time. Affects only memory and
    speed, not the result.
    """
    cube, out = Path(cube), Path(out)
    if keep_intermediate:
        out.mkdir(parents=True, exist_ok=True)
    white_fits = out / "whitelight.fits"
    white_png = out / "whitelight.png"

    with fits.open(cube, memmap=True) as hdul:
        data = hdul["DATA"].data
        # Collapse a band of image rows at a time. nanmean copies its input to
        # replace the NaNs, so calling it on the whole cube holds a second copy of
        # the cube plus its mask; a band holds only its own share of that. The
        # split is spatial, so each band still accumulates over the full wavelength
        # axis in one call -- the summation order per pixel is untouched, and the
        # bands only have to be laid back next to each other. Splitting along
        # wavelength instead would change that order and with it the last bits.
        white = np.concatenate([np.nanmean(data[:, y:y + rows, :], axis=0)
                                for y in range(0, data.shape[1], rows)])
        white = np.nan_to_num(white, nan=0.0)
        # Carry over the celestial WCS from the cube. Without it the white light
        # image is a bare array -- downstream checks that the segmentation map and
        # the white light sit on the same pixel grid can only compare shapes, and
        # matching shapes do not guarantee alignment. celestial extracts the two sky
        # axes and drops the wavelength axis.
        hdr = WCS(hdul["DATA"].header).celestial.to_header()

    if keep_intermediate:
        fits.writeto(white_fits, white, hdr, overwrite=True)

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(white, origin="lower", cmap="gray",
                   vmin=np.nanpercentile(white, 5),
                   vmax=np.nanpercentile(white, 99))
        plt.colorbar()
        fig.savefig(white_png, dpi=130)
        # Closed explicitly: whitelight() is called in-process by the pipeline, and figures left
        # open accumulate for the whole run instead of dying with a short-lived process.
        plt.close(fig)
        print(f"saved -> {white_fits}")

    print(f"white light {white.shape} {white.dtype}")
    return WhiteLight(white, hdr)


# Without this the file would import and exit 0 when run, which reads as having
# done the step. There is one way into the pipeline, and this says where it is.
if __name__ == "__main__":
    raise SystemExit(
        "★ the steps are not run on their own; run the pipeline:\n"
        "      python src/skymodel/run_pipeline.py configs/pNN.yaml")
