from pathlib import Path
import numpy as np
from astropy.io import fits


def sum_spectra_by_id(cube_path, seg, ids, chunk=200, var_path=None):
    """Sum the spectra of all spaxels belonging to the same segmentation ID.

    Parameters
    ----------
    cube_path : path-like
        MUSE cube; requires a DATA extension.
    var_path : path-like or None
        Where to read the STAT (variance) from; defaults to cube_path. A
        sky-subtracted cube has only DATA, no STAT -- subtracting a
        deterministic sky model does not change the pixel variance, so
        re-using the original cube's STAT is correct.
    seg : ndarray, shape (ny, nx)
        Segmentation map; each pixel stores its source ID, 0 means no source.
        Pixels outside the field of view must be set to 0 before calling,
        otherwise they will be included in the summation.
    ids : ndarray, shape (n_ids,)
        List of IDs to process.
    chunk : int
        Number of wavelength planes to read at once. Affects only memory and
        speed, not the result.

    Returns
    -------
    flux : ndarray, shape (n_ids, nz)
        Summed spectra.
    var : ndarray, shape (n_ids, nz)
        Summed variance. For independent pixels the additive quantity is
        variance, not sigma; take the square root to get the noise of the
        summed spectrum.
    nspax : ndarray, shape (n_ids, nz)
        Number of spaxels with valid data per ID per wavelength channel.
        Bad spaxels and edge-of-band NaNs make this lower than the total
        spaxel count for each ID, and it varies with wavelength. Downstream
        conversions from sum to mean must divide by this, not by the total
        spaxel count:

            mean_flux = flux / nspax
            mean_var  = var / nspax**2

        Variance is divided by nspax squared because the variance of the
        mean is 1/n^2 times the variance of the sum.

    Notes
    -----
    flux, var, and nspax are guaranteed to count the same set of spaxels:
    the ok mask zeros out unusable positions before summing. Using np.nansum
    on flux and var separately would let each skip different positions
    independently (e.g. a pixel with valid flux but NaN variance would enter
    flux but not var), making variance and flux inconsistent.
    """
    seg_flat = seg.ravel()                                    # 2D -> 1D
    members  = [np.flatnonzero(seg_flat == i) for i in ids]

    with fits.open(cube_path, memmap=True) as hdul, \
         fits.open(var_path or cube_path, memmap=True) as vdul:
        nz   = hdul["DATA"].header["NAXIS3"]
        flux = np.zeros((len(ids), nz))
        var  = np.zeros((len(ids), nz))
        nspax = np.zeros((len(ids), nz))

        for j in range(0, nz, chunk):
            d = np.asarray(hdul["DATA"].data[j:j+chunk], np.float64).reshape(-1, seg_flat.size)
            v = np.asarray(vdul["STAT"].data[j:j+chunk], np.float64).reshape(-1, seg_flat.size)

            ok = np.isfinite(d) & np.isfinite(v) & (v > 0)
            d  = np.where(ok, d, 0.0)
            v  = np.where(ok, v, 0.0)

            for k, idx in enumerate(members):
                flux[k,  j:j+chunk] = d[:, idx].sum(axis=1)
                var[k,   j:j+chunk] = v[:, idx].sum(axis=1)
                nspax[k, j:j+chunk] = ok[:, idx].sum(axis=1)

    return flux, var, nspax


def main():
    import argparse
    ap = argparse.ArgumentParser(description="sum source spectra by segmentation ID")
    ap.add_argument("--cube", required=True,
                    help="cube to extract (reads its DATA). Classification needs a "
                         "sky-subtracted version; run_pipeline.py passes the ESO "
                         "nosky cube. Passing our own step05/sky_subtracted.fits "
                         "also works, but that feeds step5 output back into step2, "
                         "creating a 5->2->4->5 loop")
    ap.add_argument("--var-cube", default=None,
                    help="where to read STAT from; defaults to --cube. Our own "
                         "sky-subtracted cube only has DATA, so use this to point "
                         "to the original cube")
    ap.add_argument("--work", required=True, help="working directory for this cube")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()
    work   = Path(args.work)
    STEP01 = work / "step01"
    out    = Path(args.out) if args.out else work / "step02"
    out.mkdir(parents=True, exist_ok=True)
    print(f"workspace {work}   cube {Path(args.cube).name}")

    white = fits.getdata(STEP01 / "whitelight.fits")
    seg   = fits.getdata(STEP01 / "seg.fits")

    valid_mask  = white != 0
    source_mask = (seg > 0) & valid_mask
    seg_valid   = np.where(valid_mask, seg, 0)      # outside FoV -> 0, excluded from sum

    ids, counts = np.unique(seg_valid[source_mask], return_counts=True)
    print(f"{len(ids)} sources, {counts.sum()} source spaxels")

    print(f"DATA <- {Path(args.cube).name}   STAT <- "
          f"{Path(args.var_cube or args.cube).name}")
    flux, var, nspax = sum_spectra_by_id(args.cube, seg_valid, ids,
                                         var_path=args.var_cube)

    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.nanmedian(flux / np.sqrt(var), axis=1)

    order = np.argsort(snr)[::-1]
    print(f"{'ID':>5} {'N':>7} {'sqrt(N)':>9} {'median SNR':>12}")
    for k in order[:20]:
        print(f"{ids[k]:>5d} {counts[k]:>7d} {np.sqrt(counts[k]):>9.1f} {snr[k]:>12.2f}")

    np.save(out / "object_ids.npy",   ids)
    np.save(out / "object_flux.npy",  flux)
    np.save(out / "object_var.npy",   var)
    np.save(out / "object_nspax.npy", nspax)
    print("saved ->", out)


if __name__ == "__main__":
    main()