"""Learn the two components of the sky model from blank spaxels: the sky continuum
C_sky and K sky-line basis vectors.

Output is consumed by step4's template fitting. The decomposition method for
the sky-line basis is interchangeable.
"""
from pathlib import Path
import numpy as np
from astropy.io import fits
from sklearn.decomposition import PCA, TruncatedSVD


from utils import estimate_continuum
import argparse
import json
import sys
import subprocess
import time

SEED       = 0           # random seed shared by all decompositions, ensures reproducibility
K          = 25          # number of basis vectors = degrees of freedom in the sky model
WINDOW     = 300         # running-median window for the continuum (px)
THRESHOLDS = (1, 2)      # line-detection thresholds (positive, negative)
MAX_ITER   = 5           # maximum iterations for estimate_continuum
CLIP_SIGMA = 30          # sigma-clip threshold for mean_sky, in units of robust spread sg.
                         # The goal is only to reject bad-pixel-level outliers, not to trim
                         # the real cross-spaxel variation, so the threshold must be far
                         # above the natural amplitude of the latter.
METHODS    = ["pca", "svd"]


def learn_sky_basis(residual, K=10, method="pca"):
    """Learn K sky-line basis vectors from the residuals of blank spaxels.

    All methods return shape (K, nz), i.e. the design matrix always has
    exactly K free parameters, so chi-square values from different methods
    are directly comparable.

    Parameters
    ----------
    residual : ndarray, shape (nz, n_blank)
    K : int
    method : {"pca", "svd"}

    Returns
    -------
    basis : ndarray, shape (K, nz)
        K sky-line basis vectors. Row order matches downstream coefficient order.
    """
    X = np.nan_to_num(residual.T).astype(np.float32)     # (n_blank, nz)

    # random_state=SEED is essential, not just a precaution: the default
    # algorithm of both TruncatedSVD and PCA is randomized SVD. Without a
    # fixed seed the basis changes on every run and downstream results
    # become irreproducible.
    if method == "pca":
        p = PCA(n_components=K - 1, random_state=SEED).fit(X)
        return np.vstack([p.mean_[None, :], p.components_])

    if method == "svd":
        return TruncatedSVD(n_components=K, random_state=SEED).fit(X).components_

    raise ValueError(f"unknown method: {method}")

ROOT = Path(__file__).resolve().parents[2]   # paths in meta.json are stored relative to this



def main():
    ap = argparse.ArgumentParser(description="Learn sky continuum and sky-line basis from blank spaxels")
    ap.add_argument("--methods", nargs="+", default=["pca", "svd"],
                    choices=METHODS, help="which decomposition methods to run")
    ap.add_argument("-K", type=int, required=True,
                    help="number of sky-line basis vectors; required -- all three steps must use the same K; with separate defaults a missed step silently reads a different basis")
    ap.add_argument("--xlim", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="use only blank spaxels in this x range (pixels, includes LO, excludes HI) "
                         "to learn the sky. The main source's extended halo leaks into nearby "
                         "blank samples; restricting to a strip far from it yields fewer but "
                         "cleaner samples. Trade-off: sky spatial variation is determined by "
                         "only part of the field of view")
    ap.add_argument("--ylim", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="same as --xlim but restricts y")
    ap.add_argument("--exclude-box", type=int, nargs=4, default=None,
                    metavar=("Y0", "Y1", "X0", "X1"),
                    help="exclude blank spaxels inside this box from sky training samples "
                         "(endpoints inclusive). --xlim/--ylim can only trim edges, not cut "
                         "out an interior region. Purpose: high-noise patches in a mosaic "
                         "field with insufficient exposure. Spaxels inside the box are still "
                         "sky-subtracted, just not used for training")
    ap.add_argument("--seg", default=None,
                    help="which segmentation map to use for defining blank. Default is "
                         "SExtractor's step01/seg.fits; pointing to seg_dil{r}.fits from "
                         "experiments/dilate_seg.py uses a version that excludes leaked "
                         "source light from the sky samples")
    ap.add_argument("--work", required=True,
                    help="working directory for this cube (contains step01/step02/...); "
                         "one per pointing, same structure, independent of each other")
    ap.add_argument("--cube", required=True,
                    help="the cube to learn the sky from. **Must be the sky-included wsky** "
                         "-- the nosky cube already has ESO sky subtracted, so there is no "
                         "sky to learn, only noise")
    ap.add_argument("--out", default=None,
                    help="output directory; if omitted writes to {work}/step03 (overwrites); "
                         "specify explicitly for experiments")
    args = ap.parse_args()

    work   = Path(args.work)
    STEP01 = work / "step01"
    out_dir = Path(args.out) if args.out else work / "step03"
    out_dir.mkdir(parents=True, exist_ok=True)
    WSKY = Path(args.cube)
    print(f"workdir {work}   cube {WSKY.name}")

    white  = fits.getdata(STEP01 / "whitelight.fits")
    seg_f  = Path(args.seg) if args.seg else STEP01 / "seg.fits"
    seg    = fits.getdata(seg_f)
    print(f"segmentation: {seg_f.name}  source spaxels {int((seg > 0).sum()):,}")

    valid_mask = white != 0
    blank_mask = valid_mask & ~((seg > 0) & valid_mask)
    n_all = int(blank_mask.sum())
    if args.xlim or args.ylim:
        yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
        if args.xlim:
            blank_mask &= (xx >= args.xlim[0]) & (xx < args.xlim[1])
        if args.ylim:
            blank_mask &= (yy >= args.ylim[0]) & (yy < args.ylim[1])
        print(f"spatial restriction x={args.xlim} y={args.ylim}: "
              f"blank {n_all:,} -> {int(blank_mask.sum()):,}"
              f" ({100 * blank_mask.sum() / max(n_all, 1):.1f}%)")

    if args.exclude_box:
        y0, y1, x0, x1 = args.exclude_box
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
        wl  = hdr["CRVAL3"] + (np.arange(nz) + 1 - hdr["CRPIX3"]) * hdr["CD3_3"]

        blank = np.empty((nz, int(blank_mask.sum())), np.float32)
        for j in range(0, nz, 200):
            d = np.asarray(hdul["DATA"].data[j:j+200], np.float32)
            blank[j:j+200] = d[:, blank_mask]

    # Keep only spectrally complete spaxels. Differential atmospheric refraction
    # shifts the effective field of view with wavelength, so spaxels near the
    # edge are only covered at some wavelengths. learn_sky_basis nan_to_num's
    # missing channels to 0 -- that is fabricated data that the SVD would
    # earnestly fit, so we require 100% coverage instead.
    complete = np.isfinite(blank).all(axis=0)
    print(f"spectrally complete {int(complete.sum()):,} / {blank.shape[1]:,} "
          f"({100*complete.mean():.1f}%), remainder are partially covered spaxels at field edges, excluded")
    blank = blank[:, complete]

    # Sigma-clip per channel before averaging. The breakdown point of the mean
    # is 0% -- a handful of extreme negative values in a single channel is
    # enough to pull the channel mean down, and estimate_continuum would then
    # flag it as a "negative line" and mask it, causing invisible data loss.
    #
    # Clipping is done within one channel across spaxels, not along wavelength:
    # a sky emission line is bright in every spaxel, so its brightness sits
    # inside that channel's median and is never clipped.
    #
    # The centre and spread use robust estimators, but the final step still
    # takes the mean: the cross-spaxel distribution is right-skewed in bright-
    # line channels, so the median would be systematically biased low -- a bias
    # that does not shrink with more samples.
    p16, med, p84 = np.percentile(blank, [16, 50, 84], axis=1)
    sg   = np.maximum((p84 - p16) / 2, 1e-6)
    keep = np.abs(blank - med[:, None]) <= CLIP_SIGMA * sg[:, None]
    # dtype=float64: blank is float32; summing tens of thousands of terms
    # accumulates significant rounding error without the promotion.
    mean_sky = (blank * keep).sum(axis=1, dtype=np.float64) / keep.sum(axis=1)
    print(f"mean_sky: sigma-clip {CLIP_SIGMA} sigma rejected {int((~keep).sum()):,} / "
          f"{keep.size:,} elements ({100*(~keep).mean():.6f}%)")
    C_sky, sigma, line_mask, history = estimate_continuum(
        mean_sky, thresholds=THRESHOLDS, window=WINDOW, max_iter=MAX_ITER)
    print(f"line_mask: {100*line_mask.mean():.1f}% of channels  "
          f"({len(history)} iterations: "
          f"{' -> '.join(f'{100*h[2].mean():.1f}%' for h in history)})")

    np.save(out_dir / "wavelength.npy",    wl)
    np.save(out_dir / "mean_sky.npy",      mean_sky)
    np.save(out_dir / "sky_continuum.npy", C_sky)
    np.save(out_dir / "sky_sigma.npy",     sigma)
    np.save(out_dir / "line_mask.npy",     line_mask)

    # Per-iteration intermediate results. The mask is not cumulative -- each
    # iteration recomputes from the original mean_sky; the previous iteration
    # affects the threshold only indirectly (lines replaced with NaN before
    # re-estimating the continuum), so a small number of marginal channels can
    # drop back out. To understand why the mask grows, the continuum and sigma
    # must be examined alongside it.
    np.save(out_dir / "iter_continuum.npy", np.array([h[0] for h in history]))
    np.save(out_dir / "iter_sigma.npy",     np.array([h[1] for h in history]))
    np.save(out_dir / "iter_line_mask.npy", np.array([h[2] for h in history]))

    # Reuse the same keep mask. R = blank - C_sky differs by only a per-channel
    # constant, and a constant shifts both x and its median by the same amount,
    # so |x - med| / sg is unchanged -- the same inequality applies.
    #
    # Rejected positions are filled with the channel's typical residual
    # med - C_sky, not 0: filling 0 on a sky-line channel amounts to claiming
    # there is no line there; med is the more honest value.
    residual = np.where(keep, blank - C_sky[:, None], (med - C_sky)[:, None])

    for method in args.methods:
        t0 = time.time()
        basis = learn_sky_basis(residual, K=args.K, method=method)
        np.save(out_dir / f"sky_basis_{method}_K{args.K}.npy", basis)   # filename includes K so different K values can coexist
        print(f"{method:13s} basis {basis.shape}  {time.time() - t0:6.1f}s", flush=True)

    # Provenance of the products. Only method and K appear in the filename;
    # the spatial range, the segmentation map, and the cube are not encoded --
    # re-running with a different REGION silently overwrites, and downstream
    # only remembers "sky_dir = .../step03" with no way to tell. This JSON is
    # the sole record of those choices.
    def rel(q):
        q = Path(q)
        try:
            return str(q.resolve().relative_to(ROOT))
        except ValueError:
            return str(q)

    (out_dir / "meta.json").write_text(json.dumps(dict(
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True,
                                  cwd=ROOT).stdout.strip(),
        cube=rel(args.cube), seg=rel(seg_f), work=rel(work),
        methods=list(args.methods), K=args.K,
        xlim=args.xlim, ylim=args.ylim, exclude_box=args.exclude_box,
        n_blank_all=n_all, n_blank_used=int(blank_mask.sum()),
        argv=sys.argv[1:],
    ), indent=2, ensure_ascii=False) + "\n")
    print(f"meta -> {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()