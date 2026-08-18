from pathlib import Path
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import argparse

import matplotlib
matplotlib.use("Agg")              # must be set before importing pyplot: render to file, not screen
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description="cube -> whitelight image (whitelight.fits + preview png)")
    ap.add_argument("cube", type=Path, help="input cube (.fits)")
    # --out has no default. The alternative would be to derive a directory name from
    # the cube filename, but that means feeding the wrong cube silently creates an
    # unrecognisable directory under results/ with no warning.
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    NOSKY_CUBE = args.cube
    WHITE_LIGHT_CUBE = out / "whitelight.fits"
    WHITE_LIGHT_IMAGE = out / "whitelight.png"

    with fits.open(NOSKY_CUBE) as hdul: # hdu list
        data = hdul["DATA"].data
        white = np.nanmean(data, axis=0)
        white = np.nan_to_num(white, nan=0.0)
        # Carry over the celestial WCS from the cube. Without it the white light
        # image is a bare array -- downstream checks that the segmentation map and
        # the white light sit on the same pixel grid can only compare shapes, and
        # matching shapes do not guarantee alignment. celestial extracts the two sky
        # axes and drops the wavelength axis.
        hdr = WCS(hdul["DATA"].header).celestial.to_header()
        fits.writeto(WHITE_LIGHT_CUBE, white, hdr, overwrite=True)

        plt.figure(figsize=(6, 6))
        plt.imshow(white, origin="lower", cmap="gray",
                vmin=np.nanpercentile(white, 5),
                vmax=np.nanpercentile(white, 99))
        plt.colorbar()
        plt.savefig(WHITE_LIGHT_IMAGE, dpi=130)


if __name__ == "__main__":
    main()