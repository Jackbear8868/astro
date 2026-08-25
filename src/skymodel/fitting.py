"""Shared linear solvers for the sky subtraction pipeline.

fit_blank and fit_source both minimise the unweighted sum of squared
residuals. The design matrix is the same for every spaxel (within blank or
within a source region), so a single pinv call solves all clean spaxels at
once; only spaxels with bad channels need per-spaxel lstsq, and only those
violating bounds need per-spaxel lsq_linear.

build_templates selects sources that received a model in step4 and
redshifts each model onto the MUSE wavelength grid.

The solves below run under whatever thread limit their caller set; the steps
that call them hold BLAS at one thread (utils.blas_single_thread).
"""
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from utils import blas_single_thread
from templates import (DWARF_DIR, STAR_LIBRARY, load_ascii_template,
                       load_eigen_galaxy, load_eigen_qso, redshift_to_grid)

ROOT      = Path(__file__).resolve().parents[2]
EIGEN_GAL = ROOT / "data/eigen_galaxy_Bolton2012.fits"
EIGEN_QSO = ROOT / "data/qso_eigen_linear_55732.dat"

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
    # of those came from the SDSS stellar library.
    lib   = str(best["star_library"]) if "star_library" in best.files else "sdss"
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
