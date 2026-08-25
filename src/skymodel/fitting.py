"""Shared linear solvers for the sky subtraction pipeline.

fit_blank and fit_source both minimise the unweighted sum of squared
residuals. The design matrix is the same for every spaxel (within blank or
within a source region), so a single pinv call solves all clean spaxels at
once; only spaxels with bad channels need per-spaxel lstsq, and only those
violating bounds need per-spaxel lsq_linear.

build_templates selects sources that received a model in step4 and
redshifts each model onto the MUSE wavelength grid.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from templates import (DWARF_DIR, STAR_LIBRARY, load_ascii_template,
                       load_eigen_galaxy, load_eigen_qso, redshift_to_grid)

ROOT      = Path(__file__).resolve().parents[2]
EIGEN_GAL = ROOT / "data/eigen_galaxy_Bolton2012.fits"
EIGEN_QSO = ROOT / "data/qso_eigen_linear_55732.dat"

N_SRC        = 4
MIN_COVERAGE = 0.9


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
    out   = {}
    for i in np.flatnonzero(np.nansum(np.abs(best["A"]), axis=1) > 0):
        g, name = str(best["group"][i]), str(best["template"][i])
        spline = eigen[g] if g in eigen else load_ascii_template(DWARF_DIR / f"{name}.dat")
        T = redshift_to_grid(spline, float(best["z"][i]), lam_vac)
        out[int(best["id"][i])] = T if T.ndim == 2 else T[:, None]
    return out


def fit_blank(D, sky, fit_mask=None, s_fix=None, _nonneg=True):
    """Coefficients for blank spaxels, with a non-negativity constraint on s.

    Parameters
    ----------
    D : ndarray, shape (nz, n)
    sky : ndarray, shape (K+1, nz)
    fit_mask : ndarray or None, shape (nz,)

    Returns
    -------
    ndarray, shape (K+1, n)
    """
    if s_fix is not None:
        c   = fit_blank(D - s_fix * sky[0][:, None], sky[1:], fit_mask,
                        _nonneg=False)
        out = np.full((sky.shape[0], D.shape[1]), np.nan)
        out[0], out[1:] = s_fix, c
        return out

    K    = sky.shape[0]
    coef = np.full((K, D.shape[1]), np.nan)
    good = np.isfinite(D)
    rows = np.ones(D.shape[0], bool) if fit_mask is None else fit_mask
    if fit_mask is not None:
        good &= fit_mask[:, None]

    clean = good[rows].all(axis=0)
    coef[:, clean] = np.linalg.pinv(sky[:, rows].T) @ D[rows][:, clean]

    for j in np.flatnonzero(~clean):
        g = good[:, j]
        if g.sum() <= K:
            continue
        coef[:, j] = np.linalg.lstsq(sky[:, g].T, D[g, j], rcond=None)[0]

    if not _nonneg:
        return coef
    lb = np.r_[0.0, np.full(K - 1, -np.inf)]
    ub = np.full(K, np.inf)
    for j in np.flatnonzero(coef[0] < 0):
        g = good[:, j]
        coef[:, j] = lsq_linear(sky[:, g].T, D[g, j],
                                bounds=(lb, ub), method="bvls").x
    return coef


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
