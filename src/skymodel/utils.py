"""Everything the six pipeline steps share.

The sections below, in order:

  * thread control -- the one-thread BLAS limit the fitting steps run under
  * wavelength axis -- the channel grid of a cube, and the air-to-vacuum
    conversion everything downstream is evaluated in
  * sky continuum and line detection -- the spectral direction: running
    median and iterative sky-line detection
  * source templates -- eigenspectra and the stellar library, read as
    splines, and the models step6 reconstructs its sources from
  * linear solves -- the per-spaxel solves steps 5 and 6 share
  * the main source group -- the whole galaxy, reassembled from the seg IDs
    the deblender split it into
  * the s field -- the spatial direction: a smooth field built from the
    per-spaxel sky-continuum coefficients
  * figures and display

fit_blank and fit_source both minimise the unweighted sum of squared
residuals. The design matrix is the same for every spaxel (within blank or
within a source region), so a single pinv call solves all clean spaxels at
once; only spaxels with bad channels need per-spaxel lstsq, and only those
violating bounds need per-spaxel lsq_linear.
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

    The limit is applied when fn is called and lifted when it returns, rather
    than through the OMP_NUM_THREADS family: those are read once as each library
    loads, so they bite only when set before anything imports numpy, which a
    module cannot arrange for whoever imports it.
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

    CTYPE3 is checked rather than assumed: the arithmetic below is linear
    sampling, which is what AWAV declares, and on a log-sampled axis it would
    still return a grid -- correct only at the reference channel, and nothing
    downstream could tell that from a real one.
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

    The MUSE cube's CTYPE3 is AWAV (air wavelength) while the templates and
    eigenspectra are all in vacuum, so without the conversion everything is
    fitted with a systematic offset of ~83 km/s.

    Valid above about 2000 A only: the two resonance terms have poles at
    1602.8 A and 876.7 A, below which the result is neither correct nor
    monotonic.
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
    """The median of the `window` channels around each channel, ignoring NaN.

    Returns an array the length of `spectrum`. The window shortens at the two
    ends rather than being padded, so the result is defined everywhere.
    """
    half = window // 2
    n = len(spectrum)
    result = np.empty(n)
    for j in range(n):
        result[j] = np.nanmedian(spectrum[max(0, j - half):j + half])
    return result


def detect_lines(mean_sky, exclude=None, thresholds = (1, 2), window=300):
    """One pass of "where is the continuum, and which channels sit off it".

    Returns (continuum, sigma, line_mask), each the length of `mean_sky`. The
    continuum is a running median smoothed by a cubic spline; sigma is a running
    median of the distance to it; a channel is masked when it lies more than
    thresholds[0] sigma above or thresholds[1] sigma below. The two sides have
    their own threshold because emission and absorption are not the same
    question of the sky.

    `exclude` blanks channels before the continuum is measured, so that lines
    found earlier do not drag it upwards. The mask itself is still tested against
    the untouched `mean_sky`, so a channel excluded this time can come back: what
    is returned is where the lines are, not where they have ever been.
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
    """detect_lines repeated, each pass keeping the last one's lines out of the
    continuum.

    Returns (continuum, sigma, line_mask) from the final pass, and the history of
    all of them -- what each iteration saw is worth keeping, since the mask is
    what every later step fits around.

    It stops when a pass reproduces the previous mask, or after max_iter, or when
    a mask would leave less than min_unmasked_frac of the channels: past that
    there is not enough continuum left to measure one from, and the pass is
    discarded rather than used. A first pass already over that floor raises,
    because there is then no answer to return at all.
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
    iter_line_mask.npy it wrote, so a script reading the products applies the
    same rule the pipeline applied.

    The saved masks are non-cumulative and must stay that way: estimate_continuum
    stops when an iteration reproduces the previous mask, which a monotonically
    growing mask could never do. Read as a "progressively wider" sequence they are
    then not strictly nested -- a re-estimated continuum can drop a marginal channel
    back below threshold -- so cumulative=True applies a logical_or prefix
    accumulation. Iteration 1 is the same either way.
    """
    m = np.load(masks) if isinstance(masks, (str, Path)) else np.asarray(masks)
    return np.logical_or.accumulate(m, axis=0) if cumulative else m


# ---------------------------------------------------------------------------
# source templates -- eigenspectra and the stellar library, read as splines
# ---------------------------------------------------------------------------


# The stellar library: where its files are, and the name written into the products.
# Defined together -- apart, a product could name one library while the code read
# another directory, and that is invisible in the output.
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
    domain is the range that actually carries data.

    air=True converts the axis to vacuum, because everything downstream is
    evaluated at vacuum wavelengths. The axis is cut at AIR_MIN first: below it
    air_to_vacuum is not monotonic, which would make the spline unbuildable.
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
    """Turn (n_comp, n_wave) eigenspectra into one "batch" spline.

    The constant padding at both ends is filler, not data -- the file repeats
    the last real value all the way to the boundary. In the spline it would be
    an artificial flat line the fit would use to absorb residuals, so the
    spline is cut where all components simultaneously stop changing.

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

    Real data covers 1183-9840 A rest, 4 components. The chi2 column holds
    uninitialised memory -- do not read it. The .spec file beside it is the same
    data truncated to 4 decimals, which diverges in relative error where the
    higher-order components cross zero; use the FITS version.
    """
    d = fits.open(path)[1].data
    lam = np.asarray(d["wave"], np.float64)
    F   = np.vstack([np.asarray(d[f"flux{i}"], np.float64) for i in range(1, 5)])
    return _eigen_spline(lam, F)


def load_eigen_qso(path):
    """QSO eigenspectra (text file: column 1 = wavelength, rest = components)
    -> batch spline.

    Real data covers 605-8356 A rest, 4 components. The filename says "linear",
    but the wavelength grid is logarithmic like the galaxy file (dlog10 = 1e-4).
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
    """Select the sources that receive a model and redshift each onto lam_vac.

    The stellar library the classification was fitted with is checked rather
    than assumed: a template name says nothing about which library it came from,
    so rebuilding a source from the wrong one would be silent.

    Returns
    -------
    dict
        {segmentation ID: model redshifted to lam_vac, shape (nz, n_comp)}
    """
    # Galaxy only: step4 scans the stellar templates against these eigenspectra and
    # nothing else, so "galaxy" and "star" are the only groups a classification can
    # carry, and a star is read from its own file below.
    eigen = {"galaxy": load_eigen_galaxy(EIGEN_GAL)}
    # A file without this field predates it, and all of those came from the SDSS
    # library. Membership is asked of `best` itself, not of .files, so step6 can
    # hand over step4's fields without going through an npz to get that attribute.
    lib   = str(best["star_library"]) if "star_library" in best else "sdss"
    if lib != STAR_LIBRARY:
        raise SystemExit(
            f"★ this classification was fitted with the {lib!r} stellar library, "
            f"which is no longer available; re-run step4 to refit it with "
            f"{STAR_LIBRARY!r}")
    A = np.asarray(best["A"], float)
    # nansum() of a row is 0.0 both for a source step4 never solved (NaN throughout)
    # and for one solved to no amplitude, so the two are separated and reported
    # apart. np.abs matters: without it, components that cancel would read as zero.
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
SPAXEL_CHUNK = 256      # must be a power of two -- see fit_blank


def as_vector(s_fix, n):
    """Normalise s_fix into a length-n vector; None is returned unchanged."""
    if s_fix is None:
        return None
    return np.broadcast_to(np.asarray(s_fix, float), (n,))


# Decorated as well as its callers: the limit belongs to the solve, so a script
# calling this directly gets the numbers the pipeline got.
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
        # The residual fitted is D - s*C, non-finite wherever D is plus every
        # channel where C is and every spaxel where s is.
        good &= np.isfinite(C)[:, None]
        good &= np.isfinite(s)
    if fit_mask is not None:
        good &= fit_mask[:, None]

    # A slice, not an all-true mask: a boolean index would copy every row.
    rows  = slice(None) if fit_mask is None else fit_mask
    clean = good[rows].all(axis=0)
    P     = np.linalg.pinv(A[:, rows].T)

    # Chunked so the float32 -> float64 widening covers one chunk, not the block.
    # Widths must stay whole multiples of four with no lone column left over: BLAS
    # takes columns in fours, and a remainder answers in different last bits. The
    # clean spaxels are carried as positions and not as the mask for the same
    # reason -- a chunk is then a run of consecutive columns, which is what those
    # widths count.
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
        # Least squares is linear in its data, so subtracting s*C from the K
        # coefficients gives exactly what subtracting it from the (nz, n) data
        # would -- and this way the cube-sized D - s*C is never formed.
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


# Decorated as well as its callers: the limit belongs to the solve, so a script
# calling this directly gets the numbers the pipeline got.
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
        # lb is in design-column order and out is in the fixed report order; the two
        # line up only when n_comp happens to equal N_SRC. Read the bounds through
        # this map, or a bound is compared with someone else's coefficient.
        design_to_out = (list(range(n_comp))
                         + ([N_SRC] if s_fix is None else [])
                         + list(range(N_SRC + 1, N_SRC + K)))
        theta = out[design_to_out]
        # A column that got neither solve is NaN, and NaN fails every comparison:
        # not a column inside its bounds, a column with nothing to re-solve.
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

# How close in redshift a member must be to the main source to count as part of the
# same galaxy. The galaxy has internal rotation and outflows, so the criterion is
# "within its velocity range": loose enough to keep its own bright knots, tight
# enough to reject background galaxies.
DZ_MAX = 0.005


def galaxy_redshifts(step04, ids, tag=None):
    """Best galaxy-branch redshift for each seg ID. Returns {id: z}.

    step4 stores the two branches separately; scan2 is the galaxy branch. The z
    in the classification file is the winning branch's, which for a star is a
    radial velocity and not a redshift, so it is not used here.

    tag names one step4 run -- the part of the classification filename after
    "classification_". Without it, several matches are an error rather than a
    silent pick.
    """
    out = {}
    for i in ids:
        pat = f"scan2_id{i}_*.npz" if tag is None else f"scan2_id{i}_{tag}.npz"
        f = sorted(Path(step04).glob(pat))
        if not f:
            raise SystemExit(f"{pat} not found in {step04}")
        # Several hits mean the directory holds results from several step4 runs.
        # Taking [0] would pick by filename order, and the wrong redshift changes
        # which members belong to the main source, invisibly. Stop and ask.
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

    A single "largest-area" or "brightest" ID does not work: SExtractor's
    deblender splits the main galaxy into a number of pieces that varies with the
    exposure's seeing and dither, so any "pick one ID" rule gets part of it, and
    downstream uses that piece to decide which side to mask and how far to
    exclude around it.

    Two criteria. (1) Direct adjacency, no dilation: deblended siblings are
    carved from one above-threshold region and touch, while a separate object is
    cut off by below-threshold background, a distinction dilation would blur.
    (2) Redshift, because an object superposed on the galaxy is deblended from
    the same parent and touches too: members further than dz_max from the member
    holding the brightest pixel are dropped.

    The redshifts come either as `redshifts`, the {ID: z} mapping step4 returned,
    which is how the pipeline passes them, or by reading step04's files, with
    `tag` naming one run when the directory holds several. Give one or the other.
    With neither, only criterion (1) applies -- a segmentation that has not been
    through step4 has no redshift to offer.

    The returned mask is intersected with the blob, not `isin(seg, ids)`:
    SExtractor's CLEAN merges scattered spurious detections into the bright
    source's ID, and those pixels are not on the main source.
    """
    k = np.unravel_index(np.nanargmax(np.where(np.isfinite(white), white, -np.inf)),
                         white.shape)
    src = seg > 0
    lab, _ = ndimage.label(src)
    # label() leaves the background as 0, so a brightest pixel on no source at all
    # would select the background: `blob & src` comes out empty and the caller is
    # handed a mask of nothing, with the exclusion radius it sets applied to nowhere.
    if lab[k] == 0:
        raise SystemExit(
            f"★ the brightest pixel of the white light image, y={k[0]} x={k[1]} "
            f"(value {white[k]:.6g}), lies on no segmentation source")
    blob = lab == lab[k]
    ids = [int(i) for i in np.unique(seg[blob & src]) if i > 0]

    if step04 is not None or redshifts is not None:
        z = redshifts if redshifts is not None else galaxy_redshifts(step04, ids, tag)
        # galaxy_redshifts stops when a member has no galaxy scan to read; a mapping
        # handed in is held to the same standard, or a member missing from it would
        # drop out of the group without a word.
        missing = [i for i in ids if i not in z]
        if missing:
            raise SystemExit(
                f"★ no galaxy-branch redshift for seg ID(s) {missing}, which are "
                "part of the blob holding the brightest pixel; the main source "
                "group cannot be filtered by redshift without them")
        z0 = z[int(seg[k])]
        # Comparing |dz| and |c dz/(1+z0)| is the same criterion, both sides being
        # scaled by the same positive number; the redshift difference is used
        # directly so the threshold is not tied to a particular z0.
        ids = [i for i in ids if abs(z[i] - z0) <= dz_max]

    return np.isin(seg, ids) & blob, ids, k


def main_source_mask(seg, source_id=None, main_blob=True):
    """Mask of the main source. Returns (boolean mask, seg ID used).

    With source_id omitted the largest-area source is taken -- do not hard-code
    seg == 1: SExtractor numbers sources in detection order, so the main galaxy's
    ID differs between pointings, and hard-coding would silently treat some small
    source as the main one.

    main_blob=True keeps only the largest connected component. One seg ID can be
    several disconnected patches, because SExtractor's `CLEAN Y` merges pixels of
    objects it judges spurious into a nearby bright source; used whole for
    distances, each fragment would produce its own exclusion ring.
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
# Solved freely per spaxel, s is the channel through which the sky model absorbs
# source flux: a spaxel next to a source can explain leaked source light only by
# raising its own s. Built instead from spaxels far from all sources and
# extrapolated inward, no one spaxel's data can budge the field, so source light
# has nowhere to go inside the sky model and stays in the residual.
#
# The form of the field (see rowcol_field) is
#
#     s_hat(x, y) = mu + a(y) + b(x)
#
# which describes axis-aligned striping caused by the instrument -- it extends
# along entire rows and columns, and is neither sky nor source.


def scale(a):
    """Robust spread (p84 - p16) / 2.

    Not rms/std: s has a few spaxels with failed fits whose outlier values are
    extreme enough to dominate either, so neither measures the overall spread.
    """
    a = a[np.isfinite(a)]
    return float((np.percentile(a, 84) - np.percentile(a, 16)) / 2) if a.size else np.nan


def nanmed(a, axis):
    """Median along axis; 0 instead of NaN when an entire row/column is all NaN.

    All-NaN means that row has no training points, so its offset cannot be
    estimated. 0 is "apply no correction" -- an assumption, not a measurement.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nan_to_num(np.nanmedian(a, axis=axis))


FIELD_ITER = 100


def rowcol_field(s, w, n_iter=FIELD_ITER):
    """s ~ mu + a(y) + b(x), solved by alternating medians (Tukey's median
    polish).

    Returns (field, a, b).

    Additive rather than a general f(x, y): a general f is one number per pixel,
    so it cannot predict where there is no data. This form has 1 + ny + nx
    parameters and a(y) is shared by every spaxel in that row, which is how the
    field reaches into a large gap -- a(y) in the middle of the gap is set by the
    training spaxels of the same row, far from the source. It represents stripes
    and any linear gradient; features confined to one spot rather than a whole
    row or column stay in the residual.

    Medians rather than means are robust to bad spaxels and still estimate a
    row's offset when most of that row is covered by a source. a and b are
    coupled -- the column offsets have to be subtracted before a row offset can
    be measured -- so they are solved by alternation, which reaches a fixed point;
    n_iter only has to be past it. Adding c to every a(y) and taking it off every
    b(x) leaves the field unchanged, so only their sum is meaningful.
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

    s is the (ny, nx) map of per-spaxel sky-continuum coefficients from the free
    solve, seg the segmentation (0 = blank, >0 = source) and blank the usable
    blank spaxels. The field has the form mu + a(y) + b(x) (see rowcol_field),
    with a(y) shared along a row and b(x) down a column, which is what lets it
    extrapolate into the source region from training spaxels far outside it.

    The rest decide which spaxels train it.

    Parameters
    ----------
    r_far : float
        Training points must be this far (px) from any source, or they carry
        source flux from its PSF wings. The only cost is fewer samples.
    r_far_haro : float or None
        Extra exclusion radius for the main source alone: its extended halo
        reaches far past the PSF wings of small sources, and training points
        inside it would teach the model the halo as sky. None = no extra.
    clip : float
        Spaxels with |s - median| > clip x robust spread are excluded, which
        rejects failed-fit solutions.
    main_id : int or None
        Segmentation ID of the main source; None takes the largest-area source
        (see main_source_mask).
    exclude : ndarray of bool or None
        Spaxels kept out of training but still sky-subtracted -- mosaic
        sub-fields whose exposure depth is too shallow to do anything but write
        noise into the field.
    main : ndarray of bool or None
        Mask of the main source; when given, main_id is not used.
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
    # answer. Which cut emptied the set is what says which parameter to change.
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

    Linear in the faint parts and logarithmic in the bright ones, which is what
    fits a white-light image's dynamic range into a displayable one. vmin is
    always 0.
    """
    m = np.isfinite(img) & (img != 0)
    v = np.nanpercentile(img[m], 99.5)
    a = img if valid is None else np.where(valid, img, np.nan)
    return np.arcsinh(a / (soft * v)), np.arcsinh(1 / soft)


def plot_main_group(seg, white, main_mask, main_ids, all_ids, peak,
                    out_path, title=""):
    """Two-panel figure: the main source group before and after redshift
    filtering, saved to out_path.

    Left, every seg ID in the adjacent blob (all_ids), each in its own colour and
    labelled. Right, only the IDs that passed (main_ids), with main_mask filled
    and the connected-component boundary drawn as a dashed contour. white is the
    background, peak the brightest pixel as (y, x), and title is usually the
    pointing name.
    """
    valid = white != 0
    stretched, vmax = arcsinh_stretch(white, valid)

    fig, ax = plt.subplots(1, 2, figsize=(15, 7.2))
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    for a in ax:
        a.imshow(stretched, origin="lower", cmap="gray", vmin=0, vmax=vmax)
        a.set_xticks([]); a.set_yticks([])

    # left: one colour per seg ID, before filtering
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
