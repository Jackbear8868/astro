"""Everything the six pipeline steps share.

The sections below, in order:

  * thread control -- the one-thread BLAS limit the fitting steps run under
  * wavelength axis -- the channel grid of a cube, and the air-to-vacuum
    conversion everything downstream is evaluated in
  * sky continuum and line detection -- the spectral direction: running
    median, iterative sky-line detection (estimate_continuum etc.)
  * source templates -- eigenspectra and the stellar library, read as
    splines, and the models step6 reconstructs its sources from
  * linear solves -- the per-spaxel solves steps 5 and 6 share
  * the main source group -- the whole galaxy, reassembled from the seg IDs
    the deblender split it into
  * the s field -- the spatial direction: a smooth field built from the
    per-spaxel sky-continuum coefficients
  * figures and display -- the asinh stretch, and the figures the pipeline
    itself draws

fit_blank and fit_source both minimise the unweighted sum of squared
residuals. The design matrix is the same for every spaxel (within blank or
within a source region), so a single pinv call solves all clean spaxels at
once; only spaxels with bad channels need per-spaxel lstsq, and only those
violating bounds need per-spaxel lsq_linear.

build_templates selects sources that received a model in step4 and
redshifts each model onto the MUSE wavelength grid.

The solves run under whatever thread limit their caller set; the steps that
call them hold BLAS at one thread (blas_single_thread), and so do the two
solves themselves.

Each figure function draws one self-contained figure and saves it to disk.
The pipeline steps call these after the corresponding computation; the
figures are not essential to the pipeline's data flow but let the user
verify intermediate results without running evaluation scripts separately.
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
    """Run fn with BLAS, and the OpenMP runtime under it, held at one thread.

    A threaded BLAS splits a sum across however many threads it is given and adds
    the pieces back in that order, so the last bits of a fit follow the thread
    count. Held at one, what a step writes depends on the step alone and not on
    the machine it ran on.

    The limit is applied when fn is called and lifted when it returns. That is
    why it is here rather than in the OMP_NUM_THREADS family of variables: those
    are read once, as each library loads, so they only bite when they are set
    before anything imports numpy -- which a module cannot arrange for whoever
    imports it, and which a step run on its own therefore never gets. Applying
    the limit around the work instead makes the two ways of running a step the
    same, and leaves the threading of a process that merely imported us alone.
    """
    @functools.wraps(fn)                # pipeline.py writes fn.__module__ and
                                        # fn.__name__ into the head of the step log
    def limited(*args, **kwargs):
        with threadpool_limits(limits=1):
            return fn(*args, **kwargs)
    return limited


# ---------------------------------------------------------------------------
# wavelength axis
# ---------------------------------------------------------------------------


def wavelength_grid(header):
    """The wavelength of every channel of a cube, from its DATA header.

    The rule lives here rather than in the step that first needs it: every later
    step has to rebuild the same grid to check that the one saved on disk belongs
    to the cube it was handed, and a formula written in several places drifts.

    CTYPE3 is checked rather than assumed. The arithmetic below is linear
    sampling, which is what AWAV declares; on a log-sampled axis it would still
    return a grid, correct only at the reference channel and wrong everywhere
    else, and nothing downstream can tell that from a real one.
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


# ---------------------------------------------------------------------------
# sky continuum and line detection -- the spectral direction
# ---------------------------------------------------------------------------


def running_median(spectrum, window=300):
    half = window // 2
    n = len(spectrum)
    result = np.empty(n)
    for j in range(n):
        result[j] = np.nanmedian(spectrum[max(0, j - half):j + half])
    return result


def detect_lines(mean_sky, exclude=None, thresholds = (1, 2), window=300):
    m = mean_sky.copy()
    if exclude is not None:
        m[exclude] = np.nan                        # lines found in previous iteration -> excluded from continuum
    continuum = running_median(m, window)
    
    x = np.arange(len(m))
    good = np.isfinite(m) & np.isfinite(continuum)
    xg, yg = x[good], continuum[good]
    spl = UnivariateSpline(xg, yg, k=3, s=len(xg) * 0.05**2, ext=3)
    continuum = spl(x)

    abs_diff = np.abs(m - continuum)
    sigma = running_median(abs_diff, window)       # running median along wavelength

    line_mask = (mean_sky > continuum + thresholds[0] * sigma) | (mean_sky < continuum - thresholds[1] * sigma)
    return continuum, sigma, line_mask


MIN_UNMASKED_FRAC = 0.16


def estimate_continuum(mean_sky, thresholds=(1, 2), window=300, max_iter=5,
                       min_unmasked_frac=MIN_UNMASKED_FRAC):
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
    """The per-iteration sky-line masks step3 produced; returns the cumulative
    version by default.

    `masks` is either the stack step3 returns or the path of the
    iter_line_mask.npy it wrote, so a script reading the products applies the
    same rule the pipeline applied.

    The saved file stores the mask each iteration actually used -- inside
    estimate_continuum's loop, new_mask is recomputed from scratch by
    detect_lines and wholly replaces the old one, not accumulated
    (`line_mask = new_mask`). The convergence condition
    `np.array_equal(new_mask, line_mask)` relies on this: a mask that can
    only grow monotonically would never equal its predecessor. So the saved
    file must remain non-cumulative; it is a faithful record of the process.

    When used as a "progressively wider" sequence, however, the non-cumulative
    version has exceptions -- after the continuum is re-estimated, individual
    channels can drop back below threshold, so iteration i does not strictly
    contain iteration i-1. cumulative=True applies a logical_or prefix
    accumulation to enforce monotonicity.

    Iteration 1 is the same either way, so call sites that read only [0] are
    unaffected.
    """
    m = np.load(masks) if isinstance(masks, (str, Path)) else np.asarray(masks)
    return np.logical_or.accumulate(m, axis=0) if cumulative else m


# ---------------------------------------------------------------------------
# source templates -- eigenspectra and the stellar library, read as splines
# ---------------------------------------------------------------------------


# The stellar library: where its files are, and the name written into the products.
# The two are defined together -- kept apart, a product could claim one library while
# the code read another directory, and that error is invisible in the output.
DWARF_DIR    = ROOT / "data/stellar_templates"      # two-column ASCII, luminosity
                                                    # class V main-sequence templates
STAR_LIBRARY = "dwarf"

EIGEN_GAL = ROOT / "data/eigen_galaxy_Bolton2012.fits"
EIGEN_QSO = ROOT / "data/qso_eigen_linear_55732.dat"


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


def build_templates(best, lam_vac):
    """Select the sources that receive a model and redshift each source model
    onto the MUSE wavelength grid.

    The stellar library the classification was fitted with is checked
    against the one available here rather than assumed: a template name says
    nothing about which library it came from, so reconstructing a source from
    the wrong library would be silent. Classifications produced by a library
    that is no longer shipped are refused instead of reinterpreted.

    Returns
    -------
    dict
        {segmentation ID: model redshifted to lam_vac, shape (nz, n_comp)}
    """
    eigen = {"galaxy": load_eigen_galaxy(EIGEN_GAL), "qso": load_eigen_qso(EIGEN_QSO)}
    # A file without this field was written before the field existed, and every one
    # of those came from the SDSS stellar library. Membership is asked of `best`
    # itself rather than of .files, so that step6 can hand over the fields step4
    # returned without writing and reading an npz to get an object with that
    # attribute; an NpzFile answers `in` the same way a dict does.
    lib   = str(best["star_library"]) if "star_library" in best else "sdss"
    if lib != STAR_LIBRARY:
        raise SystemExit(
            f"★ this classification was fitted with the {lib!r} stellar library, "
            f"which is no longer available; re-run step4 to refit it with "
            f"{STAR_LIBRARY!r}")
    A = np.asarray(best["A"], float)
    # Two different things leave a source without a model, and nansum() of a row
    # returns 0.0 for both: a row that is NaN in every component, which is a source
    # step4 never solved, and a row that is finite and adds up to zero, which is a
    # source solved to no amplitude. Neither can be redshifted onto the grid, but
    # they are not the same event, so they are separated before the sum is taken and
    # reported apart. np.abs is kept: without it a positive and a negative component
    # cancelling would read as an amplitude of zero rather than as the model it is.
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


N_SRC        = 4
MIN_COVERAGE = 0.9
# Spaxels multiplied at a time in the blank solve. A module constant rather than a
# parameter: it trades the size of one temporary against the number of BLAS calls
# and cannot change what fit_blank returns, so there is nothing for a caller to
# decide about it. A power of two, so that it stays a whole number of the column
# groups BLAS walks in (see fit_blank) whatever width those groups have here --
# they are powers of two, and a chunk that is not a whole number of them answers
# in different last bits.
SPAXEL_CHUNK = 256


def as_vector(s_fix, n):
    """Normalise s_fix into a length-n vector; None is returned unchanged."""
    if s_fix is None:
        return None
    return np.broadcast_to(np.asarray(s_fix, float), (n,))


# Decorated as well as the steps that call it: the limit belongs to the solve, so a
# diagnostic script that calls this directly gets the numbers the pipeline got.
@blas_single_thread
def fit_blank(D, sky, fit_mask=None, s_fix=None):
    """Coefficients for blank spaxels, with a non-negativity constraint on s.

    Parameters
    ----------
    D : ndarray, shape (nz, n)
    sky : ndarray, shape (K+1, nz)
    fit_mask : ndarray or None, shape (nz,)
    s_fix : scalar, ndarray or None, shape (n,)
        s held at this value. Only the line coefficients are then solved for,
        and the bound on s has nothing left to constrain.

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
        # The residual being fitted is D - s*C, which is non-finite wherever D
        # is, plus every channel where C is and every spaxel where s is.
        good &= np.isfinite(C)[:, None]
        good &= np.isfinite(s)
    if fit_mask is not None:
        good &= fit_mask[:, None]

    # An all-true row index is not free -- numpy copies every row of both arrays
    # to honour it -- and D is C-contiguous already, so with no mask the rows are
    # taken as they lie.
    rows  = slice(None) if fit_mask is None else fit_mask
    clean = good[rows].all(axis=0)
    P     = np.linalg.pinv(A[:, rows].T)

    # P is float64 and D is float32, and a mixed-dtype multiply widens its float32
    # side first, so one call covering every clean spaxel builds a double-width copy
    # of the whole block just to multiply it. Taking the spaxels a chunk at a time
    # builds one chunk of that copy instead. Output column j is read off input
    # column j and nothing else, so which columns share a call cannot change the sum
    # each column gets -- but for the sum to be the same bit for bit it has to be
    # the same instructions too. BLAS takes the columns in groups of four and
    # finishes what is left over with a separate kernel, so a chunk that is not a
    # whole number of groups wide puts its last columns through a kernel the
    # unchunked multiply never uses and their last bits move. Chunks are therefore
    # whole groups wide, and a lone final column -- a matrix-vector product, a
    # different routine again -- is folded into the chunk before it.
    #
    # The rows are selected once, above the loop: with a mask that selection copies
    # the rows it keeps, and per chunk it would copy them again every time; with no
    # mask it is a view either way. The clean spaxels are carried as positions
    # rather than as the mask so that a chunk is a run of consecutive output
    # columns, which is what those widths are counted in.
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
        # Least squares is linear in the data it is given, so
        #     pinv(A) @ (D - s*C) == pinv(A) @ D - (pinv(A) @ C) * s
        # is an identity, not an approximation. Taken from right to left it
        # subtracts s*C from the K coefficients rather than from the (nz, n)
        # data, and the cube-sized D - s*C is never formed.
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


# Decorated as well as the steps that call it: the limit belongs to the solve, so a
# diagnostic script that calls this directly gets the numbers the pipeline got.
@blas_single_thread
def fit_source(D, sky, T, s_fix=None, progress=False):
    """A batch of source spaxels sharing the same template.

    Returns
    -------
    ndarray, shape (N_SRC+K, n)
        Fixed layout (a1...a4, s, c1...c_{K-1}).
    """
    K      = sky.shape[0]
    n_comp = 0 if T is None else T.shape[1]
    n      = D.shape[1]
    out    = np.full((N_SRC + K, n), np.nan)

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
        out[:n_comp, j]    = th[:n_comp]
        out[N_SRC, j]      = th[n_comp] if sv is None else (
            sv[j] if sv.ndim else float(sv))
        out[N_SRC + 1:, j] = th[n_comp + (sv is None):]

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
        # lb is in the order the design columns were stacked; out is in the fixed
        # report order. The two are different orders and line up only when n_comp
        # happens to equal N_SRC, so the bounds are read through this map instead of
        # against out row by row -- otherwise a bound is compared with someone else's
        # coefficient and the violation it exists to catch never fires.
        design_to_out = (list(range(n_comp))
                         + ([N_SRC] if s_fix is None else [])
                         + list(range(N_SRC + 1, N_SRC + K)))
        theta = out[design_to_out]
        # A column that got neither solve leaves its design rows NaN, and NaN fails
        # every comparison; that is not a column within its bounds, it is a column
        # with nothing to re-solve.
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

# How close in redshift a member must be to the main source to count as part
# of the same galaxy. The galaxy has internal rotation and outflows, so the
# criterion is "within the galaxy's velocity range", not "identical redshift".
#
# The threshold must be loose enough not to exclude the galaxy's own bright
# knots and tight enough to reject background galaxies. The data make this
# easy: genuine members differ by only tens of km/s, while a superposed
# background galaxy differs by ~350,000 km/s -- four orders of magnitude
# apart, so any threshold between ~300 and ~3000 km/s gives the same result
# (see the scan table in evaluation/main_group_spec.py).
#
# 0.005 is chosen because step4's stellar-redshift scan half-width (--star-dz)
# uses the same value, and reusing the same number is easier to remember.
# Converted to velocity: c dz / (1+z) = 1469 km/s at z = 0.0205, which falls
# in the safe interval.
DZ_MAX = 0.005


def galaxy_redshifts(step04, ids, tag=None):
    """Best galaxy-branch redshift for each seg ID. Returns {id: z}.

    step4 stores the two branches separately; scan2 is the galaxy branch.
    The z from the classification file is not used -- that is the winning
    branch's value, and when the source is classified as a star it is a
    radial velocity, not a redshift.

    tag names one step4 run, and is the part of the classification filename
    after "classification_". Callers that hold a --best file know it and
    should pass it; without it several matches are an error rather than a
    silent pick.
    """
    out = {}
    for i in ids:
        pat = f"scan2_id{i}_*.npz" if tag is None else f"scan2_id{i}_{tag}.npz"
        f = sorted(Path(step04).glob(pat))
        if not f:
            raise SystemExit(f"{pat} not found in {step04}")
        # Multiple hits mean the directory contains results from several step4
        # runs (different windows, different mask iterations). Taking [0] picks
        # one by filename order, and since the redshift determines which members
        # belong to the main source, picking the wrong one is invisible
        # downstream. Better to stop and ask.
        if len(f) > 1:
            raise SystemExit(
                f"id{i} has {len(f)} scan2 files in {step04}:\n  "
                + "\n  ".join(x.name for x in f)
                + "\n  redshift must come from the same fit as --best; pass tag= to pick one.")
        d = np.load(f[0], allow_pickle=True)
        out[int(i)] = float(d["z"][np.argmin(d["red_chi2"])])
    return out


def main_source_group(seg, white, step04=None, dz_max=DZ_MAX, tag=None,
                      redshifts=None):
    """Full footprint of the main galaxy -- the connected blob containing the
    brightest pixel, keeping only members with matching redshifts.
    Returns (mask, ID list, peak coordinates).

    Why a single "largest-area" or "brightest" ID does not work: SExtractor's
    deblender splits the main galaxy. A merging galaxy naturally has several
    bright knots, and whether they are split and into how many pieces depends
    on that exposure's seeing and dither -- it varies between exposures. When
    split, any "pick one ID" rule gets only part of the galaxy, and downstream
    uses it to decide "which side to mask" and "how many pixels to exclude
    around it" -- picking the wrong piece misplaces both.

    Two criteria:

    (1) **Directly adjacent** (no dilation). Deblended siblings are carved from
        the same above-threshold connected region, so they touch; a different
        object is separated by below-threshold background. Dilation would blur
        this distinction, pulling in unrelated nearby sources.
    (2) **Redshift match**. Adjacency alone is not enough -- another object
        superposed on the galaxy is also deblended as a child of the same
        parent detection and therefore also touches. The galaxy-branch
        redshift from step4's fit discriminates: members differing from the
        main source by more than dz_max are not part of this galaxy. The main
        source's redshift is taken from the member containing the brightest
        pixel.

    tag is passed through to galaxy_redshifts to name one step4 run when the
    directory holds several.

    redshifts is the same {ID: z} mapping read straight from step4 instead of
    from its files, which is how the pipeline passes it; step04 and tag are for
    a script working from the products afterwards. Give one or the other.

    When neither is given, only criterion (1) is applied. The professor's
    delivered seg has no corresponding template fit, so no redshift is
    available; returning the entire adjacent blob is the only honest choice.

    The returned mask is intersected with the connected blob, not
    `isin(seg, ids)` -- SExtractor's CLEAN merges scattered spurious
    detections into the bright source's ID, and those pixels carry the main
    source's number but are not on the main source.
    """
    k = np.unravel_index(np.nanargmax(np.where(np.isfinite(white), white, -np.inf)),
                         white.shape)
    src = seg > 0
    lab, _ = ndimage.label(src)
    # label() numbers the sources and leaves the background as 0, so if the brightest
    # pixel sits on no source at all, `lab == lab[k]` selects the background instead
    # of a blob: `blob & src` is then empty and the caller is handed a mask of
    # nothing, with the exclusion radius that mask sets applied to nowhere. Either
    # the segmentation missed a source or this pixel is not source light (a cosmic
    # ray residual, a hot pixel); which of the two it is cannot be settled here.
    if lab[k] == 0:
        raise SystemExit(
            f"★ the brightest pixel of the white light image, y={k[0]} x={k[1]} "
            f"(value {white[k]:.6g}), lies on no segmentation source")
    blob = lab == lab[k]
    ids = [int(i) for i in np.unique(seg[blob & src]) if i > 0]

    if step04 is not None or redshifts is not None:
        z = redshifts if redshifts is not None else galaxy_redshifts(step04, ids, tag)
        # galaxy_redshifts stops when a member has no galaxy scan to read; a
        # mapping handed in has to be held to the same standard, or a member
        # missing from it would drop out of the group without a word.
        missing = [i for i in ids if i not in z]
        if missing:
            raise SystemExit(
                f"★ no galaxy-branch redshift for seg ID(s) {missing}, which are "
                "part of the blob holding the brightest pixel; the main source "
                "group cannot be filtered by redshift without them")
        z0 = z[int(seg[k])]
        # Comparing |dz| and |c dz/(1+z0)| is the same criterion -- both sides
        # are multiplied by the same positive number. Using the redshift
        # difference directly avoids tying the threshold to a particular z0.
        ids = [i for i in ids if abs(z[i] - z0) <= dz_max]

    return np.isin(seg, ids) & blob, ids, k


def main_source_mask(seg, source_id=None, main_blob=True):
    """Mask of the main source. Returns (boolean mask, seg ID used).

    When source_id is omitted, the largest-area source is selected -- do not
    hard-code seg == 1. SExtractor numbers sources in detection order, so a
    different pointing may give the main galaxy a different ID, and hard-coding
    would silently use some small source as the main, excluding dozens of
    pixels in the wrong place.

    main_blob=True keeps only the largest connected component. A single seg ID
    can consist of **several disconnected patches** -- that is caused by
    SExtractor's `CLEAN Y` merging pixels of objects it judges to be spurious
    into the nearby bright source. Using the entire ID for distance
    calculations would let those scattered fragments each produce an exclusion
    ring, inflating the exclusion area far beyond the main source itself.
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
# s field -- build a spatial field from the per-spaxel sky-continuum
# coefficient s
# ---------------------------------------------------------------------------
# Why this is needed: if s is solved freely per spaxel, a spaxel adjacent to
# a source that sees leaked source light can only explain it by raising its
# own s -- that per-spaxel degree of freedom is the channel through which
# the sky model absorbs source flux. Replacing it with "build a field from
# spaxels far from all sources, then extrapolate to the source vicinity"
# means one spaxel's data cannot budge the field, so source light has nowhere
# to go inside the sky model and stays in the residual (= is preserved).
#
# The functional form of the field (see rowcol_field):
#
#     s_hat(x, y) = mu + a(y) + b(x)
#
# describes axis-aligned striping caused by the instrument (extending along
# entire rows/columns, neither sky nor source).


def scale(a):
    """Robust spread (p84 - p16) / 2.

    rms/std cannot be used: s has a few spaxels with failed fits whose
    outlier values are extreme, and rms/std would be dominated by those few,
    no longer measuring the overall spread.
    """
    a = a[np.isfinite(a)]
    return float((np.percentile(a, 84) - np.percentile(a, 16)) / 2) if a.size else np.nan


def nanmed(a, axis):
    """Median along axis; returns 0 instead of NaN when an entire row/column
    is all NaN.

    All-NaN means that row has no training points, so the offset cannot be
    estimated. Setting 0 is "apply no correction" -- the only honest choice
    in that situation, though it is an assumption, not a measurement.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nan_to_num(np.nanmedian(a, axis=axis))


FIELD_ITER = 100


def rowcol_field(s, w, n_iter=FIELD_ITER):
    """s ~ mu + a(y) + b(x), solved by alternating medians (Tukey's median
    polish).

    Returns (field, a, b).

    Why additive rather than a general 2D function f(x, y): a general f is
    one number per pixel -- no compression, and therefore no ability to
    predict where there is no data. The additive form has only 1 + ny + nx
    parameters, and **a(y) is shared by all spaxels in that row** -- this is
    exactly why information can flow sideways into large gaps: a(y) for a row
    in the centre of the gap is determined by the training spaxels in the same
    row far from the source.

    Can represent: horizontal stripes, vertical stripes, any linear diagonal
    gradient (alpha*x + beta*y is separable). Cannot represent: features that
    appear only at a specific location and do not extend along an entire row
    or column (statistically: interaction terms). Those stay in the residual.

    Why median rather than mean: (1) robust to bad spaxels; (2) **natural
    for large gaps** -- even when most of a row is occupied by a source, the
    median of the remaining spaxels is still a good estimate of that row's
    offset.

    Why alternate: a and b are coupled. To measure "how much higher is this
    row", the column offsets must be subtracted first, otherwise the
    measurement reflects "which columns happen to remain in this row". The
    alternation reaches a fixed point -- the step shrinks until a and b stop
    moving at all -- so n_iter only has to be past that point; running further
    costs a few milliseconds an iteration and changes nothing.

    Note the degeneracy: adding c to every a(y) and subtracting c from every
    b(x) leaves the field unchanged. So (mu, a, b) are individually
    non-unique; only their sum is meaningful. The alternating medians
    naturally keep median(a) and median(b) near 0, with mu absorbing the
    overall level. This must be kept in mind when interpreting a(y) alone.
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


def build_s_field(s, seg, blank, r_far, r_far_haro, clip,
                  main_id=None, exclude=None, main=None, n_iter=FIELD_ITER):
    """Build a spatial field from the per-spaxel s map. Returns (s_hat,
    training mask).

    The field has the form mu + a(y) + b(x) (see rowcol_field). It has only
    1 + ny + nx parameters, and a(y) is shared across an entire row while
    b(x) is shared across an entire column -- this is exactly why it can
    extrapolate into the source region: rows that pass through the source
    still have training spaxels far away, and the parameters are determined
    from those, then applied to the centre of the gap.

    Parameters
    ----------
    s : ndarray, shape (ny, nx)
        Sky-continuum coefficients from the per-spaxel free solve. Values in
        the source region are not used.
    seg : ndarray, shape (ny, nx)
        Segmentation; 0 = blank, >0 = source.
    blank : ndarray of bool, shape (ny, nx)
        Usable blank spaxels (inside FoV, not source, spectrally complete,
        s successfully solved).
    r_far : float
        Training points must be at least this far (px) from **any** source,
        to avoid the source PSF wings; otherwise the training points
        themselves carry source flux. The only cost is fewer samples.
    r_far_haro : float or None
        Extra exclusion radius applied only to the main source. The main
        source's extended halo reaches much farther than the PSF wings of
        small sources; using the same r_far would leave training points
        inside the halo, and the model would learn the halo as sky.
        None = no extra exclusion.
    clip : float
        Spaxels with |s - median| > clip x robust spread are excluded
        (rejects failed-fit solutions).
    main_id : int or None
        Segmentation ID of the main source. None = automatically select the
        largest-area source (see main_source_mask).
    exclude : ndarray of bool or None
        Additional spaxels to exclude from training (they are still sky-
        subtracted, just not used for building the field). Purpose: mosaic
        sub-fields with insufficient exposure depth where noise is far
        higher than the outer ring -- using them to learn the sky writes
        noise into the field.
    main : ndarray of bool or None
        Mask of the main source; when given, main_id is not used. Callers
        typically have already computed this via main_source_group.
    """
    train = blank & (ndimage.distance_transform_edt(seg == 0) > r_far)
    n_far = int(train.sum())
    if exclude is not None:
        train &= ~exclude
    n_kept = int(train.sum())
    if r_far_haro:
        m = main if main is not None else main_source_mask(seg, main_id)[0]
        train &= ndimage.distance_transform_edt(~m) > r_far_haro
    # With no training spaxel the median below is NaN, every later comparison is
    # False, and the field comes out NaN everywhere -- a result that looks like an
    # answer. The count after each cut is reported because which one emptied the
    # set is what says which parameter to change.
    if not train.any():
        raise SystemExit(
            "★ no spaxel is left to train the s field. Survivors after each cut: "
            f"{n_far:,} more than {r_far:g} px from any source"
            + (f", {n_kept:,} outside the exclude mask" if exclude is not None else "")
            + (f", {int(train.sum()):,} more than {r_far_haro:g} px from the main source"
               if r_far_haro else ""))
    med = float(np.median(s[train]))
    train &= np.abs(s - med) <= clip * scale(s[train])

    M, _, _ = rowcol_field(s, train, n_iter)
    return M, train


# ---------------------------------------------------------------------------
# figures and display
# ---------------------------------------------------------------------------


def arcsinh_stretch(img, valid=None, soft=0.02):
    """asinh stretch for display -- returns (stretched image, vmax).

    Maps the dynamic range of a white-light image into a displayable
    range: linear in the faint parts, logarithmic in the bright parts.
    vmin is always 0; vmax = arcsinh(1 / soft).
    """
    m = np.isfinite(img) & (img != 0)
    v = np.nanpercentile(img[m], 99.5)
    a = img if valid is None else np.where(valid, img, np.nan)
    return np.arcsinh(a / (soft * v)), np.arcsinh(1 / soft)


def plot_main_group(seg, white, main_mask, main_ids, all_ids, peak,
                    out_path, title=""):
    """Two-panel figure: before and after redshift filtering of the main
    source group.

    Left panel: every seg ID inside the adjacent blob, each in its own
    colour and labelled with the ID number.  Right panel: only the IDs
    that passed the redshift criterion, with the connected-component
    boundary drawn as a dashed contour.

    Parameters
    ----------
    seg : 2-d int array
        Segmentation map.
    white : 2-d float array
        White-light image (used as background).
    main_mask : 2-d bool array
        Footprint of the final main source group (after filtering).
    main_ids : list of int
        Seg IDs kept after redshift filtering.
    all_ids : list of int
        Seg IDs in the adjacent blob before filtering.
    peak : tuple (y, x)
        Coordinates of the brightest pixel.
    out_path : path-like
        Where to save the figure.
    title : str
        Figure title (typically the pointing name).
    """
    valid = white != 0
    stretched, vmax = arcsinh_stretch(white, valid)

    fig, ax = plt.subplots(1, 2, figsize=(15, 7.2))
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    for a in ax:
        a.imshow(stretched, origin="lower", cmap="gray", vmin=0, vmax=vmax)
        a.set_xticks([]); a.set_yticks([])

    # left: one colour per seg ID inside the adjacent blob (before filtering)
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

    # right: after redshift filtering
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
