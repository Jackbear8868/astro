"""Collapse a cube along wavelength into a white light image.

Everything downstream that has to say "where is the source" works on this image
rather than on the cube: the segmentation is checked against it, the main source is
the blob holding its brightest pixel, and the evaluation figures use it as their
background.

whitelight() does the work and can be called directly; main() is the same
thing driven from the command line.

    conda run -n astro python src/skymodel/step1_whitelight.py \\
        data/nosky/DATACUBE_FINAL_ESOSKY_1.fits --out results/skymodel/p01/step01
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

import matplotlib
matplotlib.use("Agg")              # must be set before importing pyplot: render to file, not screen
import matplotlib.pyplot as plt


def whitelight(cube, out, rows=32):
    """Write whitelight.fits and its preview png into `out`; return the fits path.

    The path comes back rather than being rebuilt by the caller: the pipeline hands
    it to the segmentation check, and a filename spelled out in two places drifts.

    rows is the number of image rows collapsed at a time. Affects only memory and
    speed, not the result.
    """
    cube, out = Path(cube), Path(out)
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
    return white_fits


def main():
    ap = argparse.ArgumentParser(
        description="cube -> whitelight image (whitelight.fits + preview png)")
    ap.add_argument("cube", type=Path, help="input cube (.fits)")
    # --out has no default. The alternative would be to derive a directory name from
    # the cube filename, but that means feeding the wrong cube silently creates an
    # unrecognisable directory under results/ with no warning.
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    args = ap.parse_args()
    whitelight(args.cube, args.out)


if __name__ == "__main__":
    main()
