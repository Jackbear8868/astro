"""Everything the six pipeline steps share, section by section below.

fit_blank and fit_source both minimise the unweighted sum of squared residuals in
three tiers: one pinv where every channel is good, per-spaxel lstsq for the rest,
and per-spaxel lsq_linear only where a bound is violated.
"""

import functools
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy.interpolate import UnivariateSpline, make_interp_spline
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg")              # must be set before importing pyplot: render to file, not screen
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# thread control
# ---------------------------------------------------------------------------


def blas_single_thread(fn):
    """Run fn with BLAS, and the OpenMP runtime under it, held at one thread. A
    threaded BLAS adds a sum up in as many pieces as it has threads, so a fit's last
    bits follow the thread count. OMP_NUM_THREADS is read as each library loads, too
    early for a module to set, so the limit goes around the call instead.
    """
    @functools.wraps(fn)                # pipeline.py logs fn.__module__/__name__
    def limited(*args, **kwargs):
        with threadpool_limits(limits=1):
            return fn(*args, **kwargs)
    return limited


# ---------------------------------------------------------------------------
# wavelength axis
# ---------------------------------------------------------------------------


def wavelength_grid(header):
    """The wavelength of every channel of a cube, from its DATA header. CTYPE3 is
    checked, not assumed: on a log-sampled axis the linear formula below would still
    return a plausible grid, correct only at the reference channel.
    """
    ctype = header.get("CTYPE3")
    if ctype is not None and str(ctype).strip() != "AWAV":
        raise SystemExit(f"★ CTYPE3 is {str(ctype).strip()!r}, not 'AWAV'; this "
                         "wavelength axis is not linearly sampled in air wavelength "
                         "and cannot be built by this formula")
    nz = header["NAXIS3"]
    return (header["CRVAL3"]
            + (np.arange(nz, dtype=np.float64) + 1 - header["CRPIX3"]) * header["CD3_3"])


AIR_MIN = 2000.0        # air wavelengths are undefined where air stops transmitting


def air_to_vacuum(lam_air):
    """Convert air wavelengths to vacuum (Morton 2000, IAU standard). The cube's axis
    is air while the templates and eigenspectra are vacuum, so without this everything
    is fitted about 83 km/s off. The two resonance terms have poles below 1700 A, which
    is what AIR_MIN keeps the input above.
    """
    s2 = (1e4 / lam_air) ** 2
    n = (1.0
         + 8.336624212083e-5
         + 2.408926869968e-2 / (130.1065924522 - s2)
         + 1.599740894897e-4 / (38.92568793293 - s2))
    return lam_air * n


# ---------------------------------------------------------------------------
# sky continuum and line detection -- the spectral direction
# ---------------------------------------------------------------------------


def running_median(spectrum, window=300):
    """The median of the `window` channels around each channel, ignoring NaN. The
    window shortens at the ends rather than being padded, so the result is as long as
    `spectrum` and defined everywhere.
    """
    half = window // 2
    n = len(spectrum)
    result = np.empty(n)
    for j in range(n):
        result[j] = np.nanmedian(spectrum[max(0, j - half):j + half])
    return result


def detect_lines(mean_sky, exclude=None, thresholds = (1, 2), window=300):
    """One pass of "where is the continuum, and which channels sit off it".

    Returns (continuum, sigma, line_mask), each as long as `mean_sky`. The continuum is
    a running median smoothed by a cubic spline, sigma a running median of the distance
    to it. A channel is masked thresholds[0] sigma above or thresholds[1] sigma below,
    each side having its own because emission and absorption differ. `exclude` blanks
    channels so earlier lines cannot drag the continuum up, but the mask is tested
    against the untouched `mean_sky`, so an excluded channel can come back.
    """
    m = mean_sky.copy()
    if exclude is not None:
        m[exclude] = np.nan                        # the previous iteration's lines, kept out of the continuum
    continuum = running_median(m, window)
    
    x = np.arange(len(m))
    good = np.isfinite(m) & np.isfinite(continuum)
    xg, yg = x[good], continuum[good]
    spl = UnivariateSpline(xg, yg, k=3, s=len(xg) * 0.05**2, ext=3)
    continuum = spl(x)

    abs_diff = np.abs(m - continuum)
    sigma = running_median(abs_diff, window)

    line_mask = (mean_sky > continuum + thresholds[0] * sigma) | (mean_sky < continuum - thresholds[1] * sigma)
    return continuum, sigma, line_mask


MIN_UNMASKED_FRAC = 0.16


def estimate_continuum(mean_sky, thresholds=(1, 2), window=300, max_iter=5,
                       min_unmasked_frac=MIN_UNMASKED_FRAC):
    """detect_lines repeated, each pass excluding the last one's lines from the
    continuum. Returns (continuum, sigma, line_mask) from the final pass and the
    history of all of them; the mask is what every later step fits around.

    It stops when a pass reproduces the previous mask, after max_iter, or when a mask
    would leave less than min_unmasked_frac of the channels, past which no continuum
    can be measured.
    """
    line_mask = None
    history = []

    for i in range(max_iter):
        continuum, sigma, new_mask = detect_lines(mean_sky, exclude=line_mask, thresholds=thresholds, window=window)
        
        unmasked_frac = 1.0 - new_mask.sum() / new_mask.size
        if unmasked_frac < min_unmasked_frac:
            print(f"Iteration {i+1}: unmasked fraction {unmasked_frac:.1%} < floor {min_unmasked_frac:.0%}. Stop iteration.")
            break

        if line_mask is not None and np.array_equal(new_mask, line_mask):
            break

        line_mask = new_mask
        history.append((continuum, sigma, line_mask))
    
    if not history:
        raise ValueError(f"First iteration already masked more than {1 - min_unmasked_frac:.0%} of the spectrum.\n"
        "Check the input spectrum and parameters.")

    return history[-1][0], history[-1][1], history[-1][2], history


def load_line_masks(masks, cumulative=True):
    """The per-iteration sky-line masks step3 produced; cumulative by default.

    `masks` is either the stack step3 returns or the path of the
    continuum_iterations.npz it wrote. The saved masks are non-cumulative and must stay
    that way, since estimate_continuum stops when an iteration reproduces the previous
    mask; they are then not strictly nested, a re-estimated continuum being free to
    drop a channel back below threshold, so cumulative=True accumulates with or.
    """
    if isinstance(masks, (str, Path)):
        with np.load(masks) as z:
            m = z["line_mask"]
    else:
        m = np.asarray(masks)
    return np.logical_or.accumulate(m, axis=0) if cumulative else m


# ---------------------------------------------------------------------------
# source templates -- eigenspectra and the stellar library, read as splines
# ---------------------------------------------------------------------------


# Directory and product name together, so a product cannot name the wrong library.
DWARF_DIR    = ROOT / "data/stellar_templates"      # two-column ASCII, class V
STAR_LIBRARY = "dwarf"

EIGEN_GAL = ROOT / "data/eigen_galaxy_Bolton2012.fits"
EIGEN_QSO = ROOT / "data/qso_eigen_linear_55732.dat"


def load_ascii_template(path, air=True):
    """Read one two-column ASCII template (wavelength, flux) and return a cubic
    B-spline in rest wavelength. air=True converts the axis to vacuum, cut at AIR_MIN
    first because below that air_to_vacuum is not monotonic. Gaps are filled with 0 in
    these files, which a fit would read as real zero flux, so they become NaN and are
    dropped.
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
    """Turn (n_comp, n_wave) eigenspectra into one "batch" spline. The file pads both
    ends by repeating its last real value, and the fit would use that flat line to
    absorb residuals, so the spline is cut where it stops changing.

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
    """Bolton et al. 2012 galaxy eigenspectra (FITS bintable) -> batch spline. Real
    data covers 1183-9840 A rest, 4 components; the chi2 column is uninitialised
    memory. Not the .spec file beside it: truncated to 4 decimals, it diverges where
    the higher-order components cross zero.
    """
    d = fits.open(path)[1].data
    lam = np.asarray(d["wave"], np.float64)
    F   = np.vstack([np.asarray(d[f"flux{i}"], np.float64) for i in range(1, 5)])
    return _eigen_spline(lam, F)


def load_eigen_qso(path):
    """QSO eigenspectra (text file: column 1 = wavelength, rest = components)
    -> batch spline. Real data covers 605-8356 A rest, 4 components. The filename
    says "linear", but the grid is logarithmic like the galaxy file (dlog10 = 1e-4).
    """
    q = np.loadtxt(path)
    return _eigen_spline(q[:, 0], q[:, 1:].T)


def redshift_to_grid(spline, z, lam_muse):
    """Redshift the template to z and resample onto lam_muse; channels not covered
    are NaN. A single template returns shape (nz,), the batch spline (nz, n_comp).
    """
    return spline(lam_muse / (1.0 + z), extrapolate=False)


def build_templates(best, lam_vac):
    """Select the sources that receive a model and redshift each onto lam_vac. The
    stellar library is checked, not assumed: a template name says nothing about it.

    Returns
    -------
    dict
        {segmentation ID: model redshifted to lam_vac, shape (nz, n_comp)}
    """
    # Galaxy only: a classification's group is "galaxy" or "star", and a star is below.
    eigen = {"galaxy": load_eigen_galaxy(EIGEN_GAL)}
    # A file with no star_library field came from "sdss". Membership is asked of `best`
    # itself, not of .files, so step6 can pass step4's fields on without an npz.
    lib   = str(best["star_library"]) if "star_library" in best else "sdss"
    if lib != STAR_LIBRARY:
        raise SystemExit(
            f"★ this classification was fitted with the {lib!r} stellar library, "
            f"which is no longer available; re-run step4 to refit it with "
            f"{STAR_LIBRARY!r}")
    A = np.asarray(best["A"], float)
    # nansum() is 0.0 both for a row never solved (all NaN) and for one solved to no
    # amplitude, so the two are separated; np.abs keeps cancelling components visible.
    unsolved = np.isnan(A).all(axis=1)
    keep     = ~unsolved & (np.nansum(np.abs(A), axis=1) > 0)
    zero_amp = ~unsolved & ~keep

    out   = {}
    for i in np.flatnonzero(keep):
        g, name = str(best["group"][i]), str(best["template"][i])
        spline = eigen[g] if g in eigen else load_ascii_template(DWARF_DIR / f"{name}.dat")
        T = redshift_to_grid(spline, float(best["z"][i]), lam_vac)
        out[int(best["id"][i])] = T if T.ndim == 2 else T[:, None]
    print(f"sources with a model: {int(keep.sum())} / {A.shape[0]}"
          f"  ({int(unsolved.sum())} unsolved, {int(zero_amp.sum())} zero amplitude)")
    return out


# ---------------------------------------------------------------------------
# linear solves -- the per-spaxel solves steps 5 and 6 share
# ---------------------------------------------------------------------------


N_COMPONENTS = 4
MIN_COVERAGE = 0.9
SPAXEL_CHUNK = 256      # must be a power of two -- see fit_blank


def as_vector(s_fix, n):
    """Normalise s_fix into a length-n vector; None is returned unchanged."""
    if s_fix is None:
        return None
    return np.broadcast_to(np.asarray(s_fix, float), (n,))


# Decorated as well as its callers, so a direct caller gets the pipeline's numbers.
@blas_single_thread
def fit_blank(D, sky, fit_mask=None, s_fix=None):
    """Coefficients for blank spaxels, with a non-negativity constraint on s.

    Parameters
    ----------
    D : ndarray, shape (nz, n)
    sky : ndarray, shape (K+1, nz)
    fit_mask : ndarray or None, shape (nz,)
    s_fix : scalar, ndarray or None, shape (n,)
        s held fixed: only the line coefficients are solved, and the bound is moot.

    Returns
    -------
    ndarray, shape (K+1, n)
    """
    n    = D.shape[1]
    s    = as_vector(s_fix, n)
    C    = None if s is None else sky[0]
    A    = sky if s is None else sky[1:]
    K    = A.shape[0]
    coef = np.full((K, n), np.nan)

    good = np.isfinite(D)
    if C is not None:
        # The residual fitted is D - s*C, non-finite wherever D, C or s is.
        good &= np.isfinite(C)[:, None]
        good &= np.isfinite(s)
    if fit_mask is not None:
        good &= fit_mask[:, None]

    # A slice, not an all-true mask: a boolean index would copy every row.
    rows  = slice(None) if fit_mask is None else fit_mask
    clean = good[rows].all(axis=0)
    P     = np.linalg.pinv(A[:, rows].T)

    # Chunked so the float32 -> float64 widening covers one chunk, not the block. Widths
    # stay whole multiples of four with no lone column left: BLAS takes columns in
    # fours, and a remainder answers in different last bits. Hence positions, not a mask
    # -- a chunk is then a run of consecutive columns, which is what the widths count.
    Drows = D[rows]
    cols  = np.flatnonzero(clean)
    fit   = np.empty((K, cols.size))
    width = max(SPAXEL_CHUNK // 4, 1) * 4
    edges = list(range(0, cols.size, width)) + [cols.size]
    if len(edges) > 2 and edges[-1] - edges[-2] == 1:
        del edges[-2]
    for a, b in zip(edges, edges[1:]):
        fit[:, a:b] = P @ Drows[:, cols[a:b]]

    if C is not None:
        # Least squares is linear in its data, so taking s*C off the K coefficients
        # matches taking it off the data, without ever forming the cube-sized D - s*C.
        fit -= (P @ C[rows])[:, None] * s[clean]
    coef[:, clean] = fit

    for j in np.flatnonzero(~clean):
        g = good[:, j]
        if g.sum() <= K:
            continue
        y = D[g, j] if C is None else D[g, j] - s[j] * C[g]
        coef[:, j] = np.linalg.lstsq(A[:, g].T, y, rcond=None)[0]

    if C is not None:
        out = np.full((sky.shape[0], n), np.nan)
        out[0], out[1:] = s, coef
        return out

    lb = np.r_[0.0, np.full(K - 1, -np.inf)]
    ub = np.full(K, np.inf)
    for j in np.flatnonzero(coef[0] < 0):
        g = good[:, j]
        coef[:, j] = lsq_linear(sky[:, g].T, D[g, j],
                                bounds=(lb, ub), method="bvls").x
    return coef


# Decorated as well as its callers, so a direct caller gets the pipeline's numbers.
@blas_single_thread
def fit_source(D, sky, T, s_fix=None, progress=False):
    """A batch of source spaxels sharing the same template.

    Returns
    -------
    ndarray, shape (N_COMPONENTS+K, n)
        Fixed layout (a1...a4, s, c1...c_{K-1}).
    """
    K      = sky.shape[0]
    n_comp = 0 if T is None else T.shape[1]
    n      = D.shape[1]
    out    = np.full((N_COMPONENTS + K, n), np.nan)

    rows   = (([] if T is None else list(T.T))
              + ([] if s_fix is not None else [sky[0]])
              + list(sky[1:]))
    design = np.vstack(rows)
    p      = design.shape[0]

    lb = np.full(p, -np.inf)
    if n_comp == 1:
        lb[0] = 0.0
    if s_fix is None:
        lb[n_comp] = 0.0
    ub = np.full(p, np.inf)
    has_bounds = np.any(np.isfinite(lb))

    sv = as_vector(s_fix, n)
    y = D if sv is None else D - sv * sky[0][:, None]

    design_good = np.all(np.isfinite(design), axis=0)
    good = np.isfinite(y) & design_good[:, None]

    clean = good.all(axis=0)

    def _unpack(th, j):
        out[:n_comp, j]           = th[:n_comp]
        out[N_COMPONENTS, j]      = th[n_comp] if sv is None else (
            sv[j] if sv.ndim else float(sv))
        out[N_COMPONENTS + 1:, j] = th[n_comp + (sv is None):]

    n_clean = int(clean.sum())
    if n_clean:
        P = np.linalg.pinv(design.T)
        coef = P @ y[:, clean]
        for idx, j in enumerate(np.flatnonzero(clean)):
            _unpack(coef[:, idx], j)

    dirty = np.flatnonzero(~clean & (good.sum(axis=0) > p))
    for j in dirty:
        g = good[:, j]
        th = np.linalg.lstsq(design[:, g].T, y[g, j], rcond=None)[0]
        _unpack(th, j)

    if has_bounds:
        # lb is in design-column order, out in the fixed report order; without this map
        # a bound would be compared with someone else's coefficient.
        design_to_out = (list(range(n_comp))
                         + ([N_COMPONENTS] if s_fix is None else [])
                         + list(range(N_COMPONENTS + 1, N_COMPONENTS + K)))
        theta = out[design_to_out]
        # NaN fails every comparison, so an unsolved column needs `solved`, not the
        # bound test, to be told apart from one already inside its bounds.
        solved = np.isfinite(theta).all(axis=0)
        violators = np.flatnonzero(np.any(theta < lb[:, None], axis=0) & solved)
        for j in violators:
            g = good[:, j]
            th = lsq_linear(design[:, g].T, y[g, j],
                            bounds=(lb, ub), method="bvls").x
            _unpack(th, j)
        if progress and len(violators):
            print(f"      bounded re-solve: {len(violators)}/{n}", flush=True)

    return out


# ---------------------------------------------------------------------------
# the main source group -- the whole galaxy, reassembled from its seg IDs
# ---------------------------------------------------------------------------


C_KMS = 299792.458

# How close in redshift a member must sit to the main source to be the same galaxy:
# wide enough for its rotation and outflows, tight enough to reject background galaxies.
DZ_MAX = 0.005


def galaxy_redshifts(step04, ids):
    """Best galaxy-branch redshift for each seg ID. Returns {id: z}. It is the galaxy
    branch's own answer, not the classification file's z, which belongs to whichever
    branch won and is a radial velocity for a star.
    """
    f = Path(step04) / "source_fits.npz"
    if not f.exists():
        raise SystemExit(f"★ {f} not found -- step4 has not run for this pointing")
    d = np.load(f, allow_pickle=False)
    if "gal_z" not in d.files:
        raise SystemExit(
            f"★ {f} has no gal_z column, so it was written before step4 recorded the "
            "galaxy branch's redshift. Re-run step4 for this pointing.")
    by_id = {int(i): float(z) for i, z in zip(d["id"], d["gal_z"])}
    out = {}
    for i in ids:
        z = by_id.get(int(i))
        if z is None:
            raise SystemExit(f"★ seg ID {i} is not in {f}")
        # NaN is "could not fit at all", not a redshift; main_source_group wants it out.
        if np.isfinite(z):
            out[int(i)] = z
    return out


def load_scan(step04, branch, sid):
    """One source's whole scan of one branch, as step4 packed it: one row per candidate
    fit, `template` given back as the name rather than the index the file stores.
    branch is "star" or "galaxy" -- separate files, with different row counts and `z`
    meaning a radial velocity on the star side, a redshift on the galaxy side.
    """
    f = Path(step04) / f"scans_{branch}.npz"
    if not f.exists():
        raise SystemExit(
            f"★ {f} not found. The scans are the whole chi2 surface of every source, "
            "and step4 writes them only when source_fit.keep_scans is true in the "
            "config. Set it and re-run step4 for this pointing.")
    with np.load(f, allow_pickle=False) as z:
        if f"id{sid}" not in z.files:
            raise SystemExit(f"★ {f} holds no scan for seg ID {sid}")
        rows, names = z[f"id{sid}"], z[f"id{sid}_templates"]
    out = np.zeros(rows.shape, dtype=[(n, rows.dtype[n].str if n != "template"
                                       else names.dtype.str,
                                       rows.dtype[n].shape)
                                      for n in rows.dtype.names])
    for n in rows.dtype.names:
        out[n] = names[rows["template"]] if n == "template" else rows[n]
    return out


def scan_ids(step04, branch):
    """The seg IDs one branch's scan file holds, ascending."""
    f = Path(step04) / f"scans_{branch}.npz"
    if not f.exists():
        return []
    with np.load(f, allow_pickle=False) as z:
        return sorted(int(k[2:]) for k in z.files
                      if k.startswith("id") and not k.endswith("_templates"))


def main_source_group(seg, white, step04=None, dz_max=DZ_MAX, redshifts=None):
    """Full footprint of the main galaxy -- the connected blob holding the brightest
    pixel, less the members whose redshift does not match. Returns (mask, ID list,
    peak coordinates); why a group and not one seg ID is in README.md.

    Membership is direct adjacency, no dilation: deblended siblings touch, while a
    separate object is cut off by below-threshold background. An object superposed on
    the galaxy touches too, so members further than dz_max from the one holding the
    brightest pixel are dropped; their redshifts come as `redshifts` or from step04's
    source_fits.npz, and with neither adjacency alone decides. The mask is intersected
    with the blob because SExtractor's CLEAN merges spurious detections into its ID.
    """
    k = np.unravel_index(np.nanargmax(np.where(np.isfinite(white), white, -np.inf)),
                         white.shape)
    src = seg > 0
    lab, _ = ndimage.label(src)
    # label() leaves the background as 0, so a peak on no source would select it.
    if lab[k] == 0:
        raise SystemExit(
            f"★ the brightest pixel of the white light image, y={k[0]} x={k[1]} "
            f"(value {white[k]:.6g}), lies on no segmentation source")
    blob = lab == lab[k]
    ids = [int(i) for i in np.unique(seg[blob & src]) if i > 0]

    if step04 is not None or redshifts is not None:
        z = redshifts if redshifts is not None else galaxy_redshifts(step04, ids)
        # A handed-in mapping is held to galaxy_redshifts' standard: no silent dropping.
        missing = [i for i in ids if i not in z]
        if missing:
            raise SystemExit(
                f"★ no galaxy-branch redshift for seg ID(s) {missing}, which are "
                "part of the blob holding the brightest pixel; the main source "
                "group cannot be filtered by redshift without them")
        z0 = z[int(seg[k])]
        # dz, not c dz/(1+z0): the same criterion, with no threshold tied to a given z0.
        ids = [i for i in ids if abs(z[i] - z0) <= dz_max]

    return np.isin(seg, ids) & blob, ids, k


def main_source_mask(seg, source_id=None, main_blob=True):
    """Mask of the main source. Returns (boolean mask, seg ID used).

    With source_id omitted the largest-area source is taken; do not hard-code seg == 1,
    since SExtractor numbers sources in detection order. main_blob=True keeps only the
    largest connected component, since `CLEAN Y` can leave one ID as several patches.
    """
    if source_id is None:
        ids, cnt = np.unique(seg[seg > 0], return_counts=True)
        source_id = int(ids[np.argmax(cnt)])
    m = seg == source_id
    if main_blob:
        lab, n = ndimage.label(m)
        if n > 1:
            sz = ndimage.sum(m, lab, range(1, n + 1))
            m = lab == (int(np.argmax(sz)) + 1)
    return m, source_id


# ---------------------------------------------------------------------------
# the amplitude field -- built from the per-spaxel sky-continuum amplitude s
# ---------------------------------------------------------------------------
# Solved freely, s is how the sky model absorbs source flux: a spaxel next to a source
# can explain leaked light only by raising its own s. A field trained far away cannot.


def robust_spread(a):
    """Robust spread (p84 - p16) / 2. Not rms/std: a few spaxels with failed fits are
    extreme enough to dominate either.
    """
    a = a[np.isfinite(a)]
    return float((np.percentile(a, 84) - np.percentile(a, 16)) / 2) if a.size else np.nan


def nanmed(a, axis):
    """Median along axis, giving 0 where a whole row or column is NaN. That row has
    no training points, so 0 is "apply no correction" -- an assumption, not a
    measurement.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nan_to_num(np.nanmedian(a, axis=axis))


FIELD_ITER = 100


def median_polish(s, w, n_iter=FIELD_ITER):
    """s ~ mu + a(y) + b(x), solved by alternating medians (Tukey's median
    polish). Returns (field, a, b).

    Additive rather than a general f(x, y), which is one number per pixel and cannot
    predict where there is no data. Here a(y) is shared by every spaxel in its row,
    which is how the field reaches into a large gap; stripes and linear gradients are
    represented, anything confined to one spot stays in the residual. Medians rather
    than means still estimate a row's offset when most of it is covered by a source.
    a and b are coupled -- column offsets must come out first -- so they are alternated
    to a fixed point, past which n_iter does not matter. Only their sum is determined.
    """
    S  = np.where(w, s, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = float(np.nanmedian(S))
    a = np.zeros(s.shape[0])
    b = np.zeros(s.shape[1])
    for _ in range(n_iter):
        a = nanmed(S - mu - b[None, :], 1)
        b = nanmed(S - mu - a[:, None], 0)
    return mu + a[:, None] + b[None, :], a, b


def build_amplitude_field(s, seg, blank, r_far, r_far_haro, clip,
                          main_id=None, exclude=None, main=None, n_iter=FIELD_ITER):
    """Build a spatial field from the per-spaxel s map. Returns (s_hat, training mask).

    s is the (ny, nx) map of sky-continuum coefficients from the free solve, seg the
    segmentation (0 = blank, >0 = source), blank the usable blank spaxels; the rest say
    which spaxels train the field, whose form is median_polish's.

    Parameters
    ----------
    r_far : float
        Distance (px) from any source; nearer points carry its PSF-wing flux.
    r_far_haro : float or None
        Extra radius for the main source, whose halo reaches past the PSF wings of
        small sources and would be learned as sky. None = no extra.
    clip : float
        Spaxels with |s - median| > clip x robust spread are dropped as failed fits.
    main_id : int or None
        None takes the largest-area source (see main_source_mask).
    exclude : ndarray of bool or None
        Kept out of training but still sky-subtracted -- sub-fields too shallow to
        write anything but noise.
    main : ndarray of bool or None
        When given, main_id is not used.
    """
    train = blank & (ndimage.distance_transform_edt(seg == 0) > r_far)
    n_far = int(train.sum())
    if exclude is not None:
        train &= ~exclude
    n_kept = int(train.sum())
    if r_far_haro:
        m = main if main is not None else main_source_mask(seg, main_id)[0]
        train &= ndimage.distance_transform_edt(~m) > r_far_haro
    # With no training spaxel the field is NaN everywhere, which still looks like an
    # answer; which cut emptied the set says which parameter to change.
    if not train.any():
        raise SystemExit(
            "★ no spaxel is left to train the amplitude field. Survivors after "
                "each cut: "
            f"{n_far:,} more than {r_far:g} px from any source"
            + (f", {n_kept:,} outside the exclude mask" if exclude is not None else "")
            + (f", {int(train.sum()):,} more than {r_far_haro:g} px from the main source"
               if r_far_haro else ""))
    med = float(np.median(s[train]))
    train &= np.abs(s - med) <= clip * robust_spread(s[train])

    M, _, _ = median_polish(s, train, n_iter)
    return M, train


# ---------------------------------------------------------------------------
# figures and display
# ---------------------------------------------------------------------------


def arcsinh_stretch(img, valid=None, soft=0.02):
    """asinh stretch for display -- returns (stretched image, vmax). Linear where
    faint and logarithmic where bright, which fits a white-light image's dynamic
    range into a displayable one. vmin is always 0.
    """
    m = np.isfinite(img) & (img != 0)
    v = np.nanpercentile(img[m], 99.5)
    a = img if valid is None else np.where(valid, img, np.nan)
    return np.arcsinh(a / (soft * v)), np.arcsinh(1 / soft)


def plot_main_group(seg, white, main_mask, main_ids, all_ids, peak,
                    out_path, title=""):
    """Two-panel figure of the main source group before and after redshift filtering,
    saved to out_path. Left, every seg ID in the adjacent blob (all_ids), each in its
    own colour and labelled. Right, only the IDs that passed (main_ids), with main_mask
    filled and the blob boundary dashed. white is the background, peak the brightest
    pixel as (y, x).
    """
    valid = white != 0
    stretched, vmax = arcsinh_stretch(white, valid)

    fig, ax = plt.subplots(1, 2, figsize=(15, 7.2))
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    for a in ax:
        a.imshow(stretched, origin="lower", cmap="gray", vmin=0, vmax=vmax)
        a.set_xticks([]); a.set_yticks([])

    for k, i in enumerate(all_ids):
        m = seg == i
        rgba = np.zeros(m.shape + (4,))
        rgba[m] = list(cmap[k % 20][:3]) + [0.55]
        ax[0].imshow(rgba, origin="lower")
        yy, xx = np.nonzero(m)
        if yy.size > 40:
            ax[0].text(xx.mean(), yy.mean(), str(i), color="w", fontsize=11,
                       fontweight="bold", ha="center", va="center",
                       path_effects=[pe.withStroke(linewidth=2.5,
                                                   foreground="k")])
    ax[0].set_title(f"before ({len(all_ids)} sources)", fontsize=12)

    rgba = np.zeros(main_mask.shape + (4,))
    rgba[main_mask] = [1.0, 0.5, 0.05, 0.5]
    ax[1].imshow(rgba, origin="lower")
    ax[1].contour(main_mask, levels=[0.5], colors="#ff7f0e", linewidths=1.6)
    lab, _ = ndimage.label(seg > 0)
    ax[1].contour(lab == lab[peak], levels=[0.5],
                  colors="#00e5ff", linewidths=0.9, linestyles="--")
    ax[1].plot(peak[1], peak[0], "w+", ms=14, mew=2)
    ax[1].set_title(f"after ({len(main_ids)} sources)", fontsize=12)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
