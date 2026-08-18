from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.interpolate import make_interp_spline

def load_sdss_template(path):
    """Read one SDSS spDR2 template and return a cubic B-spline in rest
    wavelength.

    The template is sampled uniformly in log wavelength, which is coarser
    than the MUSE grid at the red end. Linear interpolation error oscillates
    periodically with z at the template's sampling interval, creating
    spurious local maxima in chi2(z). A spline is built once in rest
    wavelength; redshifting only changes the evaluation positions.
    """
    with fits.open(path) as hdul:
        header   = hdul[0].header
        spectrum = hdul[0].data[0].astype(np.float64)

    spectrum[spectrum == 0] = np.nan
    
    lam_rest = 10.0 ** (header["COEFF0"] + header["COEFF1"] * np.arange(header["NAXIS1"]))
    good = np.isfinite(spectrum)
    return make_interp_spline(lam_rest[good], spectrum[good], k=3)

def _eigen_spline(lam_rest, F):
    """Turn (n_comp, n_wave) eigenspectra into a "batch" spline.

    The constant padding at both ends is filler, not data -- the file repeats
    the last real value all the way to the boundary. Including it in the
    spline would produce an artificial flat line there, which the fit would
    happily use to absorb residuals. The boundary of the real data is where
    all components simultaneously stop changing.

    y is passed as (n_wave, n_comp), so a single evaluation returns all
    components. The cost of a spline is dominated by the binary search for
    the knot interval, which is independent of the number of components, so
    evaluating 4 components is as fast as 1 and avoids repeating the search.

    Returns
    -------
    BSpline
        Evaluated shape is (n_out, n_comp); positions not covered are NaN.
    """
    d = np.abs(np.diff(F, axis=1)).max(axis=0)
    i = np.flatnonzero(d > 0)
    g = slice(i[0], i[-1] + 2)                  # +2: diff is one shorter, and the right endpoint must be included
    return make_interp_spline(lam_rest[g], F[:, g].T, k=3)


def load_eigen_galaxy(path):
    """Bolton et al. 2012 galaxy eigenspectra (FITS bintable) -> batch spline.

    Real data covers 1183-9840 A (rest), 4 components. The chi2 column
    contains uninitialised memory (1e-310 to 1e307) -- do not touch. The
    .spec file in the same directory is a text version of the same data,
    kept to only 4 decimal places -- higher-order components cross zero,
    so truncation causes divergent relative error; use the FITS version.
    """
    d = fits.open(path)[1].data
    lam = np.asarray(d["wave"], np.float64)
    F   = np.vstack([np.asarray(d[f"flux{i}"], np.float64) for i in range(1, 5)])
    return _eigen_spline(lam, F)


def load_eigen_qso(path):
    """QSO eigenspectra (text file: column 1 = wavelength, rest = components)
    -> batch spline.

    Real data covers 605-8356 A (rest), 4 components. The filename says
    "linear", but the wavelength grid is actually logarithmic like the galaxy
    file and SDSS templates (dlog10 = 1e-4).
    """
    q = np.loadtxt(path)
    return _eigen_spline(q[:, 0], q[:, 1:].T)


def redshift_to_grid(spline, z, lam_muse):
    """Redshift the template to z and resample onto lam_muse.

    Channels not covered by the template are NaN. A single template returns
    shape (nz,); the batch spline of eigenspectra returns (nz, n_comp).
    """
    return spline(lam_muse / (1.0 + z), extrapolate=False)
    
def template_on_grid(path, z, lam_muse):
    """Read + redshift + resample (convenience wrapper for single use)."""
    return redshift_to_grid(load_sdss_template(path), z, lam_muse)
    
def air_to_vacuum(lam_air):
    """Convert air wavelengths to vacuum (Morton 2000, IAU standard).

    The MUSE cube's CTYPE3 = AWAV (air wavelength), while SDSS templates
    use vacuum wavelength. Without conversion there is a systematic redshift
    offset of ~83 km/s.
    """
    s2 = (1e4 / lam_air) ** 2
    n = (1.0
         + 8.336624212083e-5
         + 2.408926869968e-2 / (130.1065924522 - s2)
         + 1.599740894897e-4 / (38.92568793293 - s2))
    return lam_air * n