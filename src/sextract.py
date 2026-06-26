"""
Source Extraction with SEP (the Python port of Source Extractor / SExtractor).

This script demonstrates the standard SEP workflow on a MUSE data cube:
    1. Load the cube and collapse it into a 2D "white-light" image.
    2. Estimate and subtract the sky background.
    3. Detect sources (objects) above a noise threshold.
    4. Measure their positions and fluxes.
    5. Save a catalog and a figure with the detected sources circled.

Run with:  uv run python sextract.py
Docs:      https://sep.readthedocs.io/
"""
import os
import numpy as np
import sep
from astropy.io import fits
import matplotlib

matplotlib.use("Agg")  # use a non-interactive backend so the script can save a PNG headlessly
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# --- Configuration -----------------------------------------------------------
FITS_FILE = "Haro11_nosky.fits"  # input MUSE cube
THRESH = 4.0                     # detection threshold, in units of the background RMS (sigma)
DEBLEND_CONT = 0.005             # deblending: higher = less aggressive splitting (1.0 disables it)
OUTPUT_DIR = "results"                    # directory to save the output image and catalog
OUTPUT_IMAGE = "sextract_result.png"
OUTPUT_CATALOG = "sextract_catalog.txt"


def load_white_light_image(path):
    """Open a MUSE cube and collapse the spectral axis into a single 2D image.

    The cube has shape (wavelength, y, x). Summing over axis 0 produces a
    "white-light" image: the total flux per spatial pixel across all wavelengths.
    """
    hdul = fits.open(path)
    cube = hdul["DATA"].data  # 3D array: (wavelength, y, x)

    # Collapse the wavelength axis. nansum ignores NaNs (masked/edge pixels).
    image = np.nansum(cube, axis=0)

    # SEP requires a C-contiguous array in the machine's *native* byte order.
    # MUSE FITS files are stored big-endian ('>f4'), so we convert to native float32.
    image = image.astype(np.float32)  # .astype already returns native byte order + C-contiguous

    hdul.close()
    return image


def main():
    # 1. Load and collapse the cube ------------------------------------------
    image = load_white_light_image(FITS_FILE)
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")

    # 2. Estimate the background ---------------------------------------------
    # sep.Background models the spatially-varying sky background and its noise (RMS).
    bkg = sep.Background(image)
    print(f"Background global level: {bkg.globalback:.3f}, "
          f"global RMS (noise): {bkg.globalrms:.3f}")

    # Subtract the background to get a flat, sky-free image for detection.
    image_sub = image - bkg  # subtracting the Background object subtracts its 2D model

    # 3. Detect sources -------------------------------------------------------
    # Bright extended galaxies can produce many sub-objects during deblending;
    # raise the internal limit so SEP does not overflow on clumpy targets.
    sep.set_sub_object_limit(4096)

    # Pixels brighter than THRESH * (local background RMS) are flagged, then
    # grouped into objects. `err=bkg.globalrms` sets the per-pixel noise level.
    # `deblend_cont` controls how readily a blended source is split apart.
    objects = sep.extract(
        image_sub, THRESH, err=bkg.globalrms, deblend_cont=DEBLEND_CONT
    )
    print(f"Detected {len(objects)} sources")

    # 4. Measure aperture photometry -----------------------------------------
    # Sum the flux inside a circular aperture (radius = 3 px) around each source.
    # `gain` would convert to electrons for proper Poisson errors; omitted here.
    flux, fluxerr, flag = sep.sum_circle(
        image_sub, objects["x"], objects["y"], 3.0, err=bkg.globalrms
    )

    # 5a. Write a simple text catalog ----------------------------------------
    with open(os.path.join(OUTPUT_DIR, OUTPUT_CATALOG), "w") as f:
        f.write("# id  x_pixel  y_pixel  flux_r3px  flux_err\n")
        for i in range(len(objects)):
            f.write(f"{i:4d}  {objects['x'][i]:8.2f}  {objects['y'][i]:8.2f}  "
                    f"{flux[i]:12.3f}  {fluxerr[i]:10.3f}\n")
    print(f"Catalog written to {OUTPUT_CATALOG}")

    # 5b. Plot the image with detected sources circled ------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    # Scale the display between the 5th and 99th percentiles for good contrast.
    vmin, vmax = np.percentile(image_sub, [5, 99])
    ax.imshow(image_sub, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    # Draw an ellipse for each source using SEP's shape parameters (a, b, theta).
    for i in range(len(objects)):
        e = Ellipse(
            xy=(objects["x"][i], objects["y"][i]),
            width=6 * objects["a"][i],   # 6 = scale factor for visibility
            height=6 * objects["b"][i],
            angle=np.degrees(objects["theta"][i]),
        )
        e.set_facecolor("none")
        e.set_edgecolor("red")
        ax.add_artist(e)

    ax.set_title(f"SEP: {len(objects)} sources detected (thresh = {THRESH} sigma)")
    ax.set_xlabel("x [pixel]")
    ax.set_ylabel("y [pixel]")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, OUTPUT_IMAGE), dpi=150)
    print(f"Figure saved to {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
