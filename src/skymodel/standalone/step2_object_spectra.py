"""Step 2, on its own: sum every source's spectrum over the spaxels its
segmentation ID covers.

These summed spectra are what step 4 classifies: one spectrum per source, with its
variance and the number of contributing spaxels per channel. They come from a
sky-subtracted cube -- classifying a spectrum that still holds the sky gives output
that looks entirely normal with every template and redshift wrong.

    python step2_object_spectra.py --work results/skymodel/p01 \
                                   --cube data/nosky/DATACUBE_FINAL_ESOSKY_1.fits

The white light and the segmentation are read from <work>/step01, which step 1 wrote.

This is the standalone copy of `Pipeline.sum_spectra_by_id` and
`Pipeline.source_spectra`.
"""

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

import step_io
from step_io import SourceSpectra
from utils import wavelength_grid


def sum_spectra_by_id(cube_path, seg, ids, chunk=200, var_path=None):
    """Sum the spectra of all spaxels belonging to the same segmentation ID.

    Parameters
    ----------
    cube_path : path-like
        MUSE cube; requires a DATA extension.
    var_path : path-like or None
        Where the STAT (variance) is read from; defaults to cube_path. A sky-subtracted
        cube has only DATA -- subtracting a deterministic sky model does not change the
        pixel variance, so the original cube's STAT is correct.
    seg : ndarray, shape (ny, nx)
        Segmentation map, 0 meaning no source. Pixels outside the field of view must be
        set to 0 before calling or they enter the summation.
    ids : ndarray, shape (n_ids,)
    chunk : int
        Wavelength planes read at once; memory and speed only.

    Returns
    -------
    flux, var, nspax : ndarray, shape (n_ids, nz)
        The summed spectra, the summed variance (for independent pixels variance is
        what adds, not sigma), and the number of spaxels with valid data per ID per
        channel. nspax is below an ID's total spaxel count wherever spaxels are bad or
        the band ends, and it varies with wavelength, so a sum becomes a mean by
        dividing by it and never by the total: mean_flux = flux / nspax, and
        mean_var = var / nspax**2 because the variance of the mean is 1/n^2 times the
        variance of the sum.

    All three count the same set of spaxels, because the ok mask zeros unusable
    positions before summing. np.nansum on flux and var separately would let each skip
    different positions -- a pixel with valid flux but NaN variance entering flux and
    not var -- making the two inconsistent.
    """
    seg_flat = seg.ravel()
    members  = [np.flatnonzero(seg_flat == i) for i in ids]

    with fits.open(cube_path, memmap=True) as hdul, \
         fits.open(var_path or cube_path, memmap=True) as vdul:
        nz   = hdul["DATA"].header["NAXIS3"]
        flux = np.zeros((len(ids), nz))
        var  = np.zeros((len(ids), nz))
        nspax = np.zeros((len(ids), nz))

        for j in range(0, nz, chunk):
            # Left at the cube's own float32: the widening the sums need happens per
            # source below, so doing it here would double the chunk for nothing.
            d = np.asarray(hdul["DATA"].data[j:j+chunk], np.float32).reshape(-1, seg_flat.size)
            v = np.asarray(vdul["STAT"].data[j:j+chunk], np.float32).reshape(-1, seg_flat.size)

            ok = np.isfinite(d) & np.isfinite(v) & (v > 0)
            d  = np.where(ok, d, 0.0)
            v  = np.where(ok, v, 0.0)

            for k, idx in enumerate(members):
                # Widened before the sum, not during it: a source covers thousands of
                # spaxels and float32 accumulation loses digits, while sum()'s float64
                # accumulator depends on a blocking it does not promise.
                flux[k,  j:j+chunk] = d[:, idx].astype(np.float64).sum(axis=1)
                var[k,   j:j+chunk] = v[:, idx].astype(np.float64).sum(axis=1)
                nspax[k, j:j+chunk] = ok[:, idx].sum(axis=1)

    return flux, var, nspax


def source_spectra(white, seg, cube, out, var_cube=None, top=20,
                   keep_intermediate=True):
    """Sum every source's spectrum, print an SNR table, write the bundle.

    top sets how many rows of the SNR table are printed and changes nothing that is
    saved. The table is there to notice a source far weaker than the rest, which no
    saved array announces on its own.
    """
    out = Path(out)
    if keep_intermediate:
        out.mkdir(parents=True, exist_ok=True)
    print(f"spectra -> {out}   cube {Path(cube).name}")

    white, seg = white.data, seg.data

    valid_mask  = white != 0
    source_mask = (seg > 0) & valid_mask
    seg_valid   = np.where(valid_mask, seg, 0)      # outside FoV -> 0, excluded from sum

    ids, counts = np.unique(seg_valid[source_mask], return_counts=True)
    print(f"{len(ids)} sources, {counts.sum()} source spaxels")

    print(f"DATA <- {Path(cube).name}   STAT <- {Path(var_cube or cube).name}")
    flux, var, nspax = sum_spectra_by_id(cube, seg_valid, ids, var_path=var_cube)

    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.nanmedian(flux / np.sqrt(var), axis=1)

    order = np.argsort(snr)[::-1]
    print(f"{'ID':>5} {'N':>7} {'sqrt(N)':>9} {'median SNR':>12}")
    for k in order[:top]:
        print(f"{ids[k]:>5d} {counts[k]:>7d} {np.sqrt(counts[k]):>9.1f} {snr[k]:>12.2f}")

    if keep_intermediate:
        # One bundle, not four arrays: the four have to be read together to mean
        # anything -- ids is the row order of the rest, and the flux is a sum that
        # needs spaxel_count to become a mean -- and the wavelength axis rides along so
        # the file can be opened without also finding step03.
        np.savez(out / "source_spectra.npz",
                 ids=ids, flux_sum=flux, variance_sum=var, spaxel_count=nspax,
                 wavelength=wavelength_grid(fits.getheader(cube, "DATA")))
        print("saved ->", out / "source_spectra.npz")
    return SourceSpectra(ids, flux, var, nspax, out)


def main():
    ap = argparse.ArgumentParser(
        description="sum each source's spectrum over the spaxels its seg ID covers")
    ap.add_argument("--work", type=Path, required=True,
                    help="the run directory; step01's products are read from it and "
                         "step02 is written into it")
    ap.add_argument("--cube", type=Path, required=True,
                    help="the sky-subtracted cube the spectra are summed from")
    ap.add_argument("--var-cube", type=Path, default=None,
                    help="where STAT is read from; defaults to --cube. A cube that "
                         "lost its STAT takes the original's, a deterministic sky "
                         "subtraction not having changed the pixel variance")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory; defaults to <work>/step02")
    ap.add_argument("--top", type=int, default=20,
                    help="rows of the SNR table to print; changes nothing saved")
    args = ap.parse_args()

    out = args.out or args.work / "step02"
    source_spectra(step_io.white(args.work), step_io.seg(args.work),
                   args.cube, out, var_cube=args.var_cube, top=args.top)


if __name__ == "__main__":
    main()
