"""
Separate SOURCE regions from BACKGROUND regions with SEP -- simplest version.

Workflow:
    1. Pick a wavelength range (in Angstrom) and SUM only those channels into a 2D image.
    2. Run a basic SEP background estimate + detection (no STAT, no smoothing kernel).
    3. Use the segmentation map to split the image into:
         - source region      (segmap > 0)
         - background region   (segmap == 0)

Run with:  uv run python segment_background.py
"""

import numpy as np
import sep
from astropy.io import fits
import matplotlib

matplotlib.use("Agg")  # non-interactive backend, so we can save PNGs headlessly
import matplotlib.pyplot as plt

# --- Configuration -----------------------------------------------------------
FITS_FILE = "Haro11_nosky.fits"
WAVE_MIN = 4750.0      # start of the wavelength band to sum, in Angstrom (cube starts at ~4750)
WAVE_MAX = 7000.0      # end   of the wavelength band to sum, in Angstrom
THRESH = 5.0           # detection threshold in units of background RMS (sigma)
DEBLEND_CONT = 0.005   # deblending strength (higher = less splitting)
MINAREA = 5            # minimum number of connected pixels to count as a source
OUTPUT_IMAGE = "segment_result.png"


def load_band_image(path, wave_min, wave_max):
    """Sum a chosen wavelength band of a MUSE cube into a single 2D image.

    The cube's spectral axis is linear in wavelength (CTYPE3 = AWAV):
        wavelength(channel) = CRVAL3 + (channel + 1 - CRPIX3) * CD3_3
    We invert that to turn a wavelength range into channel indices, then sum
    only those channels (instead of all 3679).
    """
    hdul = fits.open(path)
    cube = hdul["DATA"].data            # 3D array: (wavelength, y, x)
    hdr = hdul["DATA"].header

    # Build the wavelength value of every channel from the WCS keywords.
    crval, crpix, cd = hdr["CRVAL3"], hdr["CRPIX3"], hdr["CD3_3"]
    n = cube.shape[0]
    wavelengths = crval + (np.arange(n) + 1 - crpix) * cd  # Angstrom per channel

    # Select the channels that fall inside [wave_min, wave_max].
    band = (wavelengths >= wave_min) & (wavelengths <= wave_max)
    print(f"Summing {band.sum()} channels over {wave_min}-{wave_max} Angstrom")

    # Sum only that band. nansum ignores NaNs; cast to native float32 for SEP.
    image = np.nansum(cube[band], axis=0).astype(np.float32)
    hdul.close()
    return image


def main():
    # 1. Build the 2D detection image from a wavelength band -----------------
    image = load_band_image(FITS_FILE, WAVE_MIN, WAVE_MAX)

    # The cube is padded with zeros outside the field of view; mark the real
    # data region so those empty border pixels are excluded from the background.
    valid = image != 0

    # 2. Basic background estimate + subtraction -----------------------------
    bkg = sep.Background(image)
    image_sub = image - bkg
    print(f"Background level: {bkg.globalback:.3f}, RMS: {bkg.globalrms:.3f}")

    # 3. Detect sources and also return the segmentation map -----------------
    sep.set_sub_object_limit(4096)
    objects, segmap = sep.extract(
        image_sub,
        THRESH,
        err=bkg.globalrms,
        minarea=MINAREA,
        deblend_cont=DEBLEND_CONT,
        segmentation_map=True,   # the labeled image: 0 = background, >0 = source id
    )
    print(f"Detected {len(objects)} sources")

    # 4. Split into source vs background regions -----------------------------
    source_mask = segmap > 0           # every labeled pixel belongs to a source
    background_mask = valid & ~source_mask  # real data pixels that are not sources

    print(f"Source pixels:     {int(source_mask.sum()):>8d}")
    print(f"Background pixels:  {int(background_mask.sum()):>8d}")

    # 5. Visualize the separation (4 panels) ---------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    vmin, vmax = np.percentile(image_sub[valid], [5, 99])

    axes[0, 0].imshow(image_sub, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"Band image {WAVE_MIN:.0f}-{WAVE_MAX:.0f} A (bkg-subtracted)")

    axes[0, 1].imshow(segmap, origin="lower", cmap="nipy_spectral")
    axes[0, 1].set_title(f"Segmentation map ({len(objects)} objects)")

    axes[1, 0].imshow(source_mask, origin="lower", cmap="gray")
    axes[1, 0].set_title("Source region (segmap > 0)")

    axes[1, 1].imshow(background_mask, origin="lower", cmap="gray")
    axes[1, 1].set_title("Background region (segmap == 0)")

    for ax in axes.ravel():
        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("y [pixel]")
    fig.tight_layout()
    fig.savefig(OUTPUT_IMAGE, dpi=130)
    print(f"Figure saved to {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
