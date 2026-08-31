"""Step 3, on its own: learn the sky continuum, the line mask and the sky-line basis.

The sky is learned from the blank spaxels of the sky-included cube: their mean spectrum
gives the continuum C_sky, and what is left after subtracting it is decomposed into K
line-basis vectors.

    python step3_sky_basis.py --work results/skymodel/p01 \
                              --cube data/wsky/DATACUBE_FINAL_1.fits -K 30 \
                              --mask-source-lines 4959.5 4968 5057 5069 \
                                                  5106 5118 6694 6709

The white light and the segmentation are read from <work>/step01, which step 1 wrote.

This is the standalone copy of `Pipeline.learn_sky_basis`, `borrow_line_basis`,
`exclude_source_lines`, `select_faintest_spaxels` and `sky_basis`.
"""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.interpolate import make_interp_spline
from sklearn.decomposition import PCA, TruncatedSVD

import step_io
from step_io import SkyModel
from step1_whitelight import ROOT, repo_path, write_meta
from utils import blas_single_thread, estimate_continuum, wavelength_grid

SEED = 0


def learn_sky_basis(residual, K=10, method="pca", seed=SEED, chunk=200):
    """Learn K sky-line basis vectors from the residuals of blank spaxels.

    Every method returns shape (K, nz), so the design matrix always has exactly K free
    parameters and chi2 values from different methods are comparable.

    Parameters
    ----------
    residual : ndarray, shape (nz, n_blank)
    K : int
    method : {"pca", "svd"}
    seed : int
        random_state for the decomposition.
    chunk : int
        Spaxels converted at a time; memory only.

    Returns
    -------
    basis : ndarray, shape (K, nz)
        In the downstream coefficient order.
    """
    # (n_blank, nz), a block of spaxels at a time so only a block is ever float64.
    # nan_to_num must come before the narrowing cast, never after: narrowing first would
    # turn an infinity into a finite 3.4e38 the decomposition would fit.
    X = np.empty((residual.shape[1], residual.shape[0]), np.float32)
    for i in range(0, X.shape[0], chunk):
        X[i:i+chunk] = np.nan_to_num(residual.T[i:i+chunk])

    # random_state is essential, not a precaution: both TruncatedSVD and PCA default to
    # randomized SVD, so without a fixed seed the basis changes every run.
    if method == "pca":
        p = PCA(n_components=K - 1, random_state=seed).fit(X)
        return np.vstack([p.mean_[None, :], p.components_])

    if method == "svd":
        return TruncatedSVD(n_components=K, random_state=seed).fit(X).components_

    raise ValueError(f"unknown method: {method}")


def borrow_line_basis(run_dir, method, K, wl):
    """Take another finished run's sky-line basis and put it on this grid.

    Returns (basis, record): the (K, nz) basis on the wavelength grid `wl`, and what
    meta.json records about where it came from.

    run_dir is another pointing's output directory; the basis and the grid it was
    learned on are read from its step03. Only the line basis is taken -- the continuum,
    the mean spectrum and the line masks stay this pointing's own.
    """
    src_dir   = Path(run_dir) / "step03"
    src_basis = src_dir / f"sky_line_basis_{method}_K{K}.npy"
    src_wl    = src_dir / "wavelength.npy"
    for f in (src_basis, src_wl):
        if not f.exists():
            raise SystemExit(
                f"★ --borrow-from: {f} does not exist. The run borrowed from has to "
                f"have finished step3 with the same decomposition and width, here "
                f"{method} K={K}")

    B      = np.load(src_basis)
    wl_src = np.load(src_wl)
    if B.shape != (K, wl_src.size):
        raise SystemExit(f"★ {src_basis} is {B.shape}, not ({K}, {wl_src.size}); "
                         "it does not belong to the wavelength grid beside it")

    # Every pointing has its own zero point, so the two grids are offset by a fraction
    # of a channel and the vectors have to be resampled. A resampled vector is
    # interpolated between the samples it was learned on and cannot be continued past
    # the last of them, so a target reaching outside the source range is refused rather
    # than extrapolated.
    if wl[0] < wl_src[0] or wl[-1] > wl_src[-1]:
        raise SystemExit(
            f"★ --borrow-from: {repo_path(run_dir)} learned its basis on "
            f"{wl_src[0]:.4f}-{wl_src[-1]:.4f} A and this pointing needs "
            f"{wl[0]:.4f}-{wl[-1]:.4f} A. The basis is a set of samples and not a "
            "formula, so the part outside cannot be extrapolated")

    # A cubic spline through the samples, not a straight line between them: a sky line
    # is about two channels wide, and linear interpolation is a triangular smoothing,
    # which widens and flattens a feature that narrow.
    out = make_interp_spline(wl_src, np.asarray(B, np.float64), k=3, axis=1)(wl)

    # Interpolation is a linear map and no linear map keeps a basis orthonormal. The
    # downstream fit only spans the basis, so the span is what has to survive and it
    # does; the QR is for conditioning, and it is the same span written in vectors that
    # are orthonormal again.
    before = np.abs(out @ out.T - np.eye(K)).max()
    q, r   = np.linalg.qr(out.T)
    # QR fixes each vector's length and not its sign. Making the diagonal of R positive
    # is what leaves every vector pointing the way it did.
    out    = np.ascontiguousarray((q * np.sign(np.diag(r))).T)
    # Narrowed to what a learned basis is, so a borrowed one is the same product.
    out    = out.astype(np.float32)
    # Measured on the array as it is written, but in float64: what the number is about
    # is the vectors, not the precision of the sums measuring them.
    wide   = out.astype(np.float64)
    after  = np.abs(wide @ wide.T - np.eye(K)).max()

    record = dict(
        run=str(repo_path(run_dir)),
        basis_file=str(repo_path(src_basis)),
        # The file it was taken from can be overwritten by a later run of that
        # pointing; the digest is what still identifies the array that was read.
        basis_md5=hashlib.md5(src_basis.read_bytes()).hexdigest(),
        source_wavelength=[float(wl_src[0]), float(wl_src[-1])],
        target_wavelength=[float(wl[0]), float(wl[-1])],
        # The two grids share a channel width and differ only in zero point, so one
        # number says how far the samples were moved.
        channel_offset=float((wl[0] - wl_src[0]) / (wl_src[1] - wl_src[0])),
        orthonormality_before=float(before),
        orthonormality_after=float(after))

    print(f"borrowed basis {out.shape} <- {record['basis_file']}")
    print(f"  {wl_src[0]:.4f}-{wl_src[-1]:.4f} A resampled onto "
          f"{wl[0]:.4f}-{wl[-1]:.4f} A, offset {record['channel_offset']:.4f} channels")
    print(f"  max|B B^T - I| {before:.3e} -> {after:.3e} after re-orthonormalising")
    return out, record


def exclude_source_lines(residual, wl, windows):
    """Blank the source's own emission-line channels in the decomposition input.

    Returns (mask, record): the (nz,) boolean of the excluded channels, and what
    meta.json records about the windows.

    Where the source fills the field, every "blank" spaxel still carries the source's
    emission lines, the decomposition learns them along with the sky, and step 6 then
    subtracts them from the source as well. The windows named here are the channels
    that is true of, and they are taken out of the SVD input only: the mean spectrum,
    the continuum and the sky-line masks are all built before this and see every
    channel, so step 4 and everything reading the continuum are unaffected. Steps 5 and
    6 need no exclusion of their own -- a basis with no structure at a wavelength
    cannot put flux there whatever they solve.

    The channels are set to 0 and not to the channel's typical residual, which is what
    the clip's rejected positions get. A constant is not nothing: the fit is uncentred,
    so a column that is the same non-zero number in every spaxel is still a direction
    the decomposition can spend a vector on, and the constant it would be filled with
    here is the line's own height. Exactly 0 is what makes every basis vector exactly 0
    at these channels -- a zero column contributes nothing to X^T X, so no singular
    vector with a non-zero singular value has a component on it, and PCA's mean row is
    0 there too.

    This also removes whatever real sky sits inside the windows, which is the price of
    not knowing which part of a blended channel belongs to which. The windows are
    therefore as narrow as the line, not as wide as the neighbourhood.

    Parameters
    ----------
    residual : ndarray, shape (nz, n_blank)
        The decomposition input, modified in place.
    wl : ndarray, shape (nz,)
        The wavelength grid, Angstrom.
    windows : list of [low, high]
        Observed wavelengths in Angstrom. A window is a pair of wavelengths and
        carries no redshift of its own.
    """
    mask = np.zeros(wl.size, bool)
    listed = []
    for lo, hi in windows:
        # Closed at both ends: a window is a range of wavelengths, not a half-open
        # interval, so a channel landing on an edge is inside it.
        m = (wl >= lo) & (wl <= hi)
        mask |= m
        # Per window, so one that fell off the end of the grid is visible as a window
        # of 0 channels instead of being lost in the total.
        listed.append(dict(low=float(lo), high=float(hi), n_channel=int(m.sum())))
    residual[mask] = 0

    record = dict(windows=listed, n_channel=int(mask.sum()),
                  channel_fraction=float(mask.mean()))
    print(f"mask_source_lines: {len(listed)} observed-frame window(s) -> "
          f"{int(mask.sum())} of {wl.size} channels "
          f"({100 * mask.mean():.2f}%) zeroed in the basis input only")
    for w in listed:
        print(f"  {w['low']:9.2f} - {w['high']:9.2f} A  {w['n_channel']:3d} channels")
    return mask, record


def select_faintest_spaxels(field_mean, valid, column_mean, ignore, fraction):
    """Keep only the spaxels the ESO sky rule would have called sky.

    Returns (keep, record): a boolean over the columns `column_mean` indexes, and what
    meta.json records about the cut.

    The ESO pipeline does not use a segmentation to choose its sky spaxels. Its header
    says skymethod = subtract-model, skymodel_ignore = 0.05, skymodel_fraction = 0.10:
    it ranks the spaxels of the field by flux, throws away the faintest `ignore` of
    them, and takes the sky from the next `fraction`. The faintest are thrown away
    because they are the dead and half-covered ones, not sky; the window above them is
    what is left after the sources.

    The two ends are percentiles of the whole valid field, which is what ESO ranks, and
    not of the blank set. The window is then half-open, (low, high]: a spaxel exactly
    at the `ignore` percentile is thrown away with the faintest and one exactly at the
    top is kept, so the two ends cannot both claim the same spaxel.
    """
    ign, frac = float(ignore), float(fraction)
    # A spaxel with no finite mean has no place in the ranking: its spectrum is missing
    # channels, so its mean is over a different part of the spectrum than everyone
    # else's and the two are not the same measurement.
    rank = field_mean[valid & np.isfinite(field_mean)]
    lo, hi = np.percentile(rank, [100 * ign, 100 * (ign + frac)])
    keep = (column_mean > lo) & (column_mean <= hi)

    in_field = int(((field_mean > lo) & (field_mean <= hi)
                    & valid & np.isfinite(field_mean)).sum())
    record = dict(rule="eso_skymodel_ignore_fraction", ignore=ign, fraction=frac,
                  ranked_over="valid field, spectrally complete",
                  n_ranked=int(rank.size),
                  flux_low=float(lo), flux_high=float(hi),
                  n_window_in_field=in_field,
                  n_offered=int(column_mean.size), n_selected=int(keep.sum()))
    print(f"select_faintest: ranked {rank.size:,} spaxels of the valid field by mean "
          f"flux, dropped the faintest {100 * ign:g}% and kept the next {100 * frac:g}%")
    print(f"  flux window ({lo:.4f}, {hi:.4f}] holds {in_field:,} of the field; "
          f"{int(keep.sum()):,} of the {column_mean.size:,} blank spaxels offered are "
          f"inside it ({100 * keep.mean():.1f}%)")
    return keep, record


@blas_single_thread
def sky_basis(white, seg, cube, work, out_dir, K=30, methods=("svd",), seed=SEED,
              continuum_window=300, line_thresholds=(1, 2), max_iter=5,
              clip_sigma=30, min_unmasked_frac=0.16,
              xlim=None, ylim=None, exclude_box=None,
              borrow_from=None, mask_source_lines=None, select_faintest=None,
              keep_intermediate=True):
    """Learn the sky continuum, the line mask and the sky-line basis; return them."""
    work = Path(work)
    out_dir = Path(out_dir)
    if keep_intermediate:
        out_dir.mkdir(parents=True, exist_ok=True)
    WSKY = Path(cube)
    print(f"workdir {work}   cube {WSKY.name}")

    # seg_f is where the segmentation was put, for meta.json below.
    seg_f, seg = seg.path, seg.data
    white = white.data
    print(f"segmentation: {seg_f.name}  source spaxels {int((seg > 0).sum()):,}")

    valid_mask = white != 0
    blank_mask = valid_mask & ~((seg > 0) & valid_mask)
    n_all = int(blank_mask.sum())
    if xlim or ylim:
        yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
        if xlim:
            blank_mask &= (xx >= xlim[0]) & (xx < xlim[1])
        if ylim:
            blank_mask &= (yy >= ylim[0]) & (yy < ylim[1])
        print(f"spatial restriction x={xlim} y={ylim}: "
              f"blank {n_all:,} -> {int(blank_mask.sum()):,}"
              f" ({100 * blank_mask.sum() / max(n_all, 1):.1f}%)")

    if exclude_box:
        y0, y1, x0, x1 = exclude_box
        yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
        box = (yy >= y0) & (yy <= y1) & (xx >= x0) & (xx <= x1)
        n0 = int(blank_mask.sum()); blank_mask &= ~box
        print(f"--exclude-box y {y0}-{y1}, x {x0}-{x1}:"
              f"blank {n0:,} -> {int(blank_mask.sum()):,}"
              f" ({100 * blank_mask.sum() / max(n0, 1):.1f}%)")

    print(f"blank spaxels: {int(blank_mask.sum())}")

    with fits.open(WSKY, memmap=True) as hdul:
        hdr = hdul["DATA"].header
        nz  = hdr["NAXIS3"]
        # The grid every later step reads back from wavelength.npy, so it comes from
        # the shared rule rather than being spelled out again here.
        wl  = wavelength_grid(hdr)

        # The flux every spaxel of the field is ranked by, summed in the pass that
        # reads the blank columns so that ranking the field costs no second pass over
        # the cube. Accumulated only when asked to select on it: nothing else here
        # reads it, and a sum over every block is not free. float64 because it is a
        # running total of thousands of terms.
        field_sum = (np.zeros(seg.shape, np.float64)
                     if select_faintest is not None else None)
        blank = np.empty((nz, int(blank_mask.sum())), np.float32)
        for j in range(0, nz, 200):
            d = np.asarray(hdul["DATA"].data[j:j+200], np.float32)
            blank[j:j+200] = d[:, blank_mask]
            if field_sum is not None:
                field_sum += d.sum(axis=0, dtype=np.float64)

    # Spectrally complete spaxels only: differential atmospheric refraction covers edge
    # spaxels at some wavelengths only, and learn_sky_basis would nan_to_num the rest to
    # 0 -- fabricated data the decomposition would fit.
    complete = np.isfinite(blank).all(axis=0)
    print(f"spectrally complete {int(complete.sum()):,} / {blank.shape[1]:,} "
          f"({100*complete.mean():.1f}%), remainder are partially covered spaxels at "
          f"field edges, excluded")
    blank = blank[:, complete]

    # After the completeness cut and before anything is measured: everything below --
    # the mean spectrum, the continuum, the line masks and the basis -- is built from
    # whatever spaxels are left here. A spaxel missing channels carries a NaN through
    # the sum, so field_sum is NaN exactly where `complete` is False and the two cuts
    # agree on which spaxels they are.
    faintest = None
    if select_faintest is not None:
        field_mean = field_sum / nz
        keep_col, faintest = select_faintest_spaxels(
            field_mean, valid_mask, field_mean[blank_mask][complete],
            select_faintest["ignore"], select_faintest["fraction"])
        if not keep_col.any():
            raise SystemExit(
                "★ --select-faintest left no spaxel to learn the sky from: the flux "
                f"window ({faintest['flux_low']:.4f}, {faintest['flux_high']:.4f}] "
                f"holds {faintest['n_window_in_field']:,} spaxels of the field and "
                f"none of them is among the {int(complete.sum()):,} this step was given")
        blank = blank[:, keep_col]

    # Sigma-clip per channel before averaging: the mean's breakdown point is 0%, so a
    # handful of extreme negatives in one channel pulls its mean down, and
    # estimate_continuum then masks it as a "negative line" -- invisible data loss.
    #
    # The clip runs within one channel across spaxels, never along wavelength: a sky
    # emission line is bright in every spaxel, so its brightness sits inside that
    # channel's median and is never clipped.
    #
    # Centre and spread are robust estimators, but the last step still takes the mean,
    # the cross-spaxel distribution being right-skewed in bright-line channels where
    # the median would be biased low however many samples there are.
    p16, med, p84 = np.percentile(blank, [16, 50, 84], axis=1)
    sg   = np.maximum((p84 - p16) / 2, 1e-6)
    keep = np.abs(blank - med[:, None]) <= clip_sigma * sg[:, None]
    mean_sky = (blank * keep).sum(axis=1, dtype=np.float64) / keep.sum(axis=1)
    print(f"mean_sky: sigma-clip {clip_sigma:g} sigma rejected {int((~keep).sum()):,} / "
          f"{keep.size:,} elements ({100*(~keep).mean():.6f}%)")
    C_sky, _, line_mask, history = estimate_continuum(
        mean_sky, thresholds=tuple(line_thresholds),
        window=continuum_window, max_iter=max_iter,
        min_unmasked_frac=min_unmasked_frac)
    print(f"line_mask: {100*line_mask.mean():.1f}% of channels  "
          f"({len(history)} iterations: "
          f"{' -> '.join(f'{100*h[2].mean():.1f}%' for h in history)})")

    # Per-iteration intermediate results; the masks are not cumulative --
    # utils.load_line_masks is where that is applied.
    iter_line_mask = np.array([h[2] for h in history])

    if keep_intermediate:
        np.save(out_dir / "wavelength.npy",          wl)
        np.save(out_dir / "blank_mean_spectrum.npy", mean_sky)
        np.save(out_dir / "sky_continuum.npy",       C_sky)
        # The final continuum is kept under its own name because it is C_sky, the
        # scientific product; the final threshold and mask are not, being the last row
        # of the bundle below and nothing more. A second name for the same bytes is
        # what makes "which one is authoritative" a question at all.
        #
        # One bundle rather than three arrays: the three are one loop's record, they
        # share their first axis, and separate files leave that a naming convention
        # instead of a fact. npz reads a key at a time, so a script wanting only the
        # mask still does not pay for the other two.
        np.savez(out_dir / "continuum_iterations.npz",
                 continuum=np.array([h[0] for h in history]),
                 threshold=np.array([h[1] for h in history]),
                 line_mask=iter_line_mask)

    # The same keep mask applies: blank - C_sky differs by a per-channel constant only,
    # which shifts x and its median alike, so |x - med| / sg is unchanged.
    #
    # Rejected positions are filled with the channel's typical residual med - C_sky,
    # not 0: a 0 on a sky-line channel claims there is no line there.
    residual = blank - C_sky[:, None]
    np.copyto(residual, (med - C_sky)[:, None], where=~keep)

    # Last, so that what it blanks stays blanked: it is the only thing here that is
    # about the source rather than about the sky, and it has to be the last word on the
    # channels it names.
    source_lines = None
    if mask_source_lines is not None:
        _, source_lines = exclude_source_lines(residual, wl, mask_source_lines)

    bases, borrowed = {}, {}
    for method in methods:
        t0 = time.time()
        if borrow_from is None:
            basis = learn_sky_basis(residual, K=K, method=method, seed=seed)
        else:
            basis, borrowed[method] = borrow_line_basis(borrow_from, method, K, wl)
        bases[method] = basis
        if keep_intermediate:
            # "line" because the continuum was subtracted before the decomposition: the
            # sky cannot be rebuilt from this file alone. K is in the name so different
            # K values can coexist.
            np.save(out_dir / f"sky_line_basis_{method}_K{K}.npy", basis)
        print(f"{method:13s} basis {basis.shape}  {time.time() - t0:6.1f}s", flush=True)

    # Provenance of the products. Only method and K reach the filename, so a re-run
    # with a different spatial range, segmentation or cube overwrites silently; this
    # JSON is the sole record of those choices.
    if keep_intermediate:
        write_meta(
            out_dir, "step3_sky_basis.py",
            cube=str(repo_path(cube)), seg=str(repo_path(seg_f)),
            work=str(repo_path(work)),
            methods=list(methods), K=K, seed=seed,
            continuum_window=continuum_window,
            line_thresholds=list(line_thresholds),
            max_iter=max_iter, clip_sigma=clip_sigma,
            min_unmasked_frac=min_unmasked_frac,
            # max_iter is the cap, not what happened: the loop usually stops on the
            # unmasked-fraction floor well before it, and the pass that triggered the
            # stop is discarded, so the stack's row count is the only record.
            n_iterations=len(history),
            xlim=xlim, ylim=ylim, exclude_box=exclude_box,
            # Absent unless a basis was borrowed. The array beside this file then did
            # not come from the blank spaxels counted below, and nothing else in the
            # products says so.
            **(dict(borrowed_basis=borrowed) if borrowed else {}),
            # Absent unless windows were named. The basis beside this file is 0 at the
            # channels listed here and no other product in the directory says so, the
            # continuum and the masks having been built before the exclusion.
            **(dict(masked_source_lines=source_lines) if source_lines else {}),
            # Absent unless the spaxels were narrowed to a flux window. Every array in
            # the directory then came from the subset counted here rather than from
            # n_blank_complete below, and nothing else in the products says so.
            **(dict(selected_faintest=faintest) if faintest else {}),
            n_blank_all=n_all, n_blank_used=int(blank_mask.sum()),
            # The spaxels that actually made mean_sky: the ones above minus those
            # dropped for incomplete spectral coverage at the field edges -- and minus
            # the flux window's rejects when selected_faintest is present, whose own
            # n_selected is then the count that made mean_sky.
            n_blank_complete=int(complete.sum()))
    return SkyModel(wl, C_sky, bases, iter_line_mask)


def _windows(vals):
    """--mask-source-lines as pairs. `vals` is a flat list, low high low high ..."""
    if vals is None:
        return None
    if len(vals) % 2:
        raise SystemExit("★ --mask-source-lines takes an even number of wavelengths, "
                         "read as [low, high] pairs")
    out = [[float(a), float(b)] for a, b in zip(vals[::2], vals[1::2])]
    for lo, hi in out:
        if hi <= lo:
            raise SystemExit(f"★ --mask-source-lines: window [{lo}, {hi}] is not "
                             "low < high")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="learn the sky continuum, the line mask and the sky-line basis")
    ap.add_argument("--work", type=Path, required=True,
                    help="the run directory; step01's products are read from it and "
                         "step03 is written into it")
    ap.add_argument("--cube", type=Path, required=True,
                    help="the sky-INCLUDED cube; the sky is what is being learned")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory; defaults to <work>/step03")
    ap.add_argument("--methods", nargs="+", default=["svd"], choices=["pca", "svd"])
    ap.add_argument("-K", type=int, default=30, help="number of line-basis vectors")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="random_state; both PCA and TruncatedSVD are randomized, so "
                         "without it the basis changes every run")
    ap.add_argument("--continuum-window", type=int, default=300,
                    help="running-median window for the continuum, in channels")
    ap.add_argument("--line-thresholds", type=float, nargs=2, default=[1, 2],
                    metavar=("POS", "NEG"), help="line detection, in sigma")
    ap.add_argument("--max-iter", type=int, default=5,
                    help="iterations of the continuum / line-mask loop")
    ap.add_argument("--clip-sigma", type=float, default=30,
                    help="sigma clip applied to the mean sky spectrum")
    ap.add_argument("--min-unmasked-frac", type=float, default=0.16,
                    help="stop if the line mask would cover more than 1 - this")
    ap.add_argument("--xlim", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="restrict the blank spaxels to lo <= x < hi")
    ap.add_argument("--ylim", type=int, nargs=2, default=None, metavar=("LO", "HI"))
    ap.add_argument("--exclude-box", type=int, nargs=4, default=None,
                    metavar=("Y0", "Y1", "X0", "X1"),
                    help="drop the blank spaxels inside this box, inclusive")
    ap.add_argument("--borrow-from", type=Path, default=None,
                    help="another run's output directory; its line basis is resampled "
                         "onto this grid and used instead of learning one here")
    ap.add_argument("--mask-source-lines", type=float, nargs="+", default=None,
                    metavar="A",
                    help="observed wavelengths in Angstrom, read as [low, high] "
                         "pairs. The channels inside each window are set to 0 in the "
                         "decomposition input, so every basis vector is 0 there")
    ap.add_argument("--select-faintest", type=float, nargs=2, default=None,
                    metavar=("IGNORE", "FRACTION"),
                    help="rank the valid field by flux, drop the faintest IGNORE and "
                         "learn the sky from the next FRACTION, as ESO's own rule does")
    args = ap.parse_args()

    out = args.out or args.work / "step03"
    sf = (dict(ignore=args.select_faintest[0], fraction=args.select_faintest[1])
          if args.select_faintest else None)
    sky_basis(step_io.white(args.work), step_io.seg(args.work),
              args.cube, args.work, out,
              K=args.K, methods=args.methods, seed=args.seed,
              continuum_window=args.continuum_window,
              line_thresholds=tuple(args.line_thresholds),
              max_iter=args.max_iter, clip_sigma=args.clip_sigma,
              min_unmasked_frac=args.min_unmasked_frac,
              xlim=args.xlim, ylim=args.ylim, exclude_box=args.exclude_box,
              borrow_from=args.borrow_from,
              mask_source_lines=_windows(args.mask_source_lines),
              select_faintest=sf)


if __name__ == "__main__":
    main()
