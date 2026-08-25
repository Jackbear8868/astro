from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.interpolate import make_interp_spline

ROOT = Path(__file__).resolve().parents[2]

# The stellar library: where its files are, and the name written into the products.
# The two are defined together -- kept apart, a product could claim one library while
# the code read another directory, and that error is invisible in the output.
DWARF_DIR    = ROOT / "data/stellar_templates"      # two-column ASCII, luminosity
                                                    # class V main-sequence templates
STAR_LIBRARY = "dwarf"

AIR_MIN = 2000.0        # air wavelengths are undefined where air stops transmitting


def load_ascii_template(path, air=True):
    """Read one two-column ASCII template (wavelength, flux) and return a cubic
    B-spline in rest wavelength.

    Gaps are filled with 0 in these files, which a fit would read as "the flux
    really is zero there"; they become NaN and are dropped, so the spline's
    domain is the range that actually carries data, and redshift_to_grid
    returns NaN outside it.

    air=True converts the wavelength axis to vacuum, because everything
    downstream is evaluated at vacuum wavelengths; a template left in air is
    fitted about 83 km/s too blue. The axis is cut at AIR_MIN first: below it
    an air wavelength has no meaning, and air_to_vacuum is not even monotonic
    there (see its docstring), which would make the spline unbuildable.
    """
    lam, y = np.loadtxt(path, unpack=True)
    y = y.astype(np.float64)
    y[y == 0] = np.nan
    good = np.isfinite(y)
    if air:
        good &= lam >= AIR_MIN
        lam = air_to_vacuum(lam)
    return make_interp_spline(lam[good], y[good], k=3)


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
    file (dlog10 = 1e-4).
    """
    q = np.loadtxt(path)
    return _eigen_spline(q[:, 0], q[:, 1:].T)


def redshift_to_grid(spline, z, lam_muse):
    """Redshift the template to z and resample onto lam_muse.

    Channels not covered by the template are NaN. A single template returns
    shape (nz,); the batch spline of eigenspectra returns (nz, n_comp).
    """
    return spline(lam_muse / (1.0 + z), extrapolate=False)
    
def air_to_vacuum(lam_air):
    """Convert air wavelengths to vacuum (Morton 2000, IAU standard).

    The MUSE cube's CTYPE3 = AWAV (air wavelength), while the templates and
    eigenspectra are all in vacuum wavelength. Without conversion there is a
    systematic redshift offset of ~83 km/s.

    Valid above about 2000 A only. The two resonance terms have poles at
    1602.8 A and 876.7 A, so below that the result is neither correct nor
    monotonic -- which is not a limitation in practice, since air does not
    transmit there and "air wavelength" has no meaning either.
    """
    s2 = (1e4 / lam_air) ** 2
    n = (1.0
         + 8.336624212083e-5
         + 2.408926869968e-2 / (130.1065924522 - s2)
         + 1.599740894897e-4 / (38.92568793293 - s2))
    return lam_air * n