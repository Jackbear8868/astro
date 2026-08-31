"""Step 5, on its own: force the sky-continuum amplitude onto a spatial field.

Every blank spaxel is solved freely for its own s, and the result is then forced onto
an additive row + column field. That field is what step 6 locks s to, including over
the source, where a free solve has no sky-only spaxel to measure.

    python step5_fit_s_field.py --work results/skymodel/p01 \
                                --cube data/wsky/DATACUBE_FINAL_1.fits

step01, step03 and step04 are read from <work>.

This is the standalone copy of `Pipeline.fit_sky_amplitude`.
"""

import argparse
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

import step_io
from step_io import SkyAmplitude
from step1_whitelight import repo_path, write_meta
from utils import (C_KMS, build_amplitude_field, fit_blank, main_source_group,
                   plot_main_group, wavelength_grid)


def fit_sky_amplitude(white, seg, sky, classification, cube, work, out,
                      basis="svd", K=30, blank_channels="all",
                      min_channel_coverage=0.9, min_source_distance=15,
                      min_main_source_distance=50, train_clip_sigma=8,
                      main_source_dz=0.005, n_iter=100,
                      train_xlim=None, train_ylim=None, train_exclude_box=None,
                      fix_blank_s_at=None, keep_intermediate=True):
    """Build the sky-continuum spatial field; return it."""
    work = Path(work)
    CUBE = Path(cube)
    out = Path(out)
    if keep_intermediate:
        out.mkdir(parents=True, exist_ok=True)

    seg_path, seg = seg.path, seg.data
    white = np.asarray(white.data, float)
    print(f"workdir {work}   cube {CUBE.name}")
    print(f"segmentation: {seg_path.name}  source spaxels {int((seg > 0).sum()):,}")

    # The sky model was learned on the grid of whatever cube step 3 read, and configs
    # naming one pointing's cube there and another's here need only agree in channel
    # count to run to the end with the two offset against each other.
    wl_air = sky.wavelength
    wl_cube = wavelength_grid(fits.getheader(CUBE, "DATA"))
    if wl_air.shape != wl_cube.shape:
        raise SystemExit(f"★ step3's sky model has {wl_air.size} channels but "
                         f"{CUBE} has {wl_cube.size}")
    if not np.allclose(wl_air, wl_cube, atol=1e-6):
        raise SystemExit(f"★ step3's sky model was not built from {CUBE}: the two "
                         f"wavelength grids differ by up to "
                         f"{np.abs(wl_air - wl_cube).max():.4g} A")

    fit_mask = sky.iter_line_mask[0] if blank_channels == "line1" else None
    # From here `sky` is the design matrix the spaxel fits use: the continuum as row 0,
    # the K line vectors under it.
    sky = np.vstack([sky.continuum, sky.basis[basis]])
    print(f"sky model {sky.shape}  basis {basis} K{K}")

    with fits.open(CUBE, memmap=True) as hdul:
        D = np.asarray(hdul["DATA"].data, np.float32)
    nz, ny, nx = D.shape
    D = D.reshape(nz, -1)
    seg_f = seg.reshape(-1)
    coverage = np.isfinite(D).sum(axis=0) / nz
    valid = (white != 0).reshape(-1) & (coverage >= min_channel_coverage)
    blank = valid & (seg_f == 0)

    # free blank solve
    print(f"blank {int(blank.sum()):,} spaxels (free solve)...", end="", flush=True)
    t0 = time.time()
    c = fit_blank(D[:, blank], sky, fit_mask=fit_mask, s_fix=fix_blank_s_at)
    print(f" {time.time() - t0:.1f}s", flush=True)

    s_free = np.full(ny * nx, np.nan)
    s_free[blank] = c[0]
    s2d = s_free.reshape(ny, nx)
    ok2d = blank.reshape(ny, nx) & np.isfinite(s2d)

    # spatial exclusion mask
    sf_box = None
    if train_xlim or train_ylim or train_exclude_box:
        yy, xx = np.mgrid[0:ny, 0:nx]
        sf_box = np.zeros((ny, nx), bool)
        if train_xlim:
            sf_box |= ~((xx >= train_xlim[0]) & (xx < train_xlim[1]))
        if train_ylim:
            sf_box |= ~((yy >= train_ylim[0]) & (yy < train_ylim[1]))
        if train_exclude_box:
            by0, by1, bx0, bx1 = train_exclude_box
            sf_box |= (yy >= by0) & (yy <= by1) & (xx >= bx0) & (xx <= bx1)

    # main source group
    # The redshifts come from the same step 4 result the classification does, so the
    # grouping and the source models cannot come from two different fits.
    mg, mids, mk = main_source_group(seg, white, dz_max=main_source_dz,
                                     redshifts=classification.galaxy_z)
    all_ids = main_source_group(seg, white)[1]
    z0 = classification.galaxy_z[int(seg[mk])]
    print(f"  main source (brightest pixel y={mk[0]}, x={mk[1]}): {len(mids)} IDs"
          f", {int(mg.sum()):,} px"
          f" (dz <= {main_source_dz:g},"
          f" i.e. {C_KMS * main_source_dz / (1 + z0):.0f} km/s @ z={z0:.4f})")

    if keep_intermediate:
        plot_main_group(seg, white, mg, mids, all_ids, mk,
                        out / "main_source_group.png", title=Path(work).name)

    # build field
    t0 = time.time()
    s_hat, sf_train = build_amplitude_field(
        s2d, seg, ok2d, min_source_distance, min_main_source_distance or None,
        train_clip_sigma, exclude=sf_box, main=mg, n_iter=n_iter)
    print(f"s spatial field: {int(sf_train.sum()):,} training spaxels"
          f" (dist > {min_source_distance:g} px from sources"
          + (f", Haro 11 > {min_main_source_distance:g} px" if min_main_source_distance else "")
          + f", clip {train_clip_sigma:g} sigma"
          + (f", x {train_xlim}" if train_xlim else "")
          + (f", y {train_ylim}" if train_ylim else "")
          + (", exclude-box" if train_exclude_box else "")
          + f")   {time.time() - t0:.1f}s")
    print(f"  s_hat median {np.nanmedian(s_hat):.5f}   "
          f"s_free median {np.nanmedian(s_free[blank]):.5f}   "
          f"NaN {int((~np.isfinite(s_hat[white != 0])).sum())} spaxels")

    # step 6 locks s to this field, so a field NaN everywhere makes the sky model and
    # the subtracted cube NaN too, which nothing further down separates from a
    # subtraction that worked.
    if not np.isfinite(s_hat).any():
        raise SystemExit("★ s_hat is NaN in every spaxel; the field was not estimated "
                         f"from the {int(sf_train.sum()):,} training spaxels and is "
                         "not written")
    # Narrowed once, here, and step 6 is given these numbers rather than the float64
    # they came from: the file and the fit have to hold the same field, and narrowing
    # afterwards instead would move the last bits of every spaxel.
    s_hat32 = s_hat.astype(np.float32)
    s_hat_path = out / "sky_continuum_amplitude_field.npy"
    if keep_intermediate:
        np.save(s_hat_path, s_hat32)
        np.save(out / "sky_continuum_amplitude_per_spaxel.npy",
                s_free.reshape(ny, nx).astype(np.float32))

        write_meta(
            out, "step5_fit_s_field.py",
            cube=str(repo_path(CUBE)), seg=str(repo_path(seg_path)),
            sky_dir=str(repo_path(work / "step03")),
            classification=str(repo_path(classification.path)),
            basis=basis, K=K,
            blank_channels=blank_channels, fix_blank_s_at=fix_blank_s_at,
            min_channel_coverage=min_channel_coverage,
            sky_amplitude_params=dict(
                min_source_distance=min_source_distance,
                min_main_source_distance=min_main_source_distance,
                train_clip_sigma=train_clip_sigma,
                train_exclude_box=train_exclude_box,
                train_xlim=train_xlim, train_ylim=train_ylim,
                main_source_dz=main_source_dz, n_iter=n_iter),
            main_ids=[int(i) for i in mids],
            n_blank=int(blank.sum()), n_train=int(sf_train.sum()),
            # A row with no training spaxel gets an offset of 0 -- "apply no
            # correction", which utils.nanmed calls an assumption rather than a
            # measurement. The field itself cannot show it: every spaxel comes out
            # finite either way, so these two lists are the only record of where the
            # field is asserting instead of measuring.
            untrained_rows=[int(i) for i in
                            np.flatnonzero(sf_train.sum(axis=1) == 0)],
            untrained_cols=[int(i) for i in
                            np.flatnonzero(sf_train.sum(axis=0) == 0)])
        print(f"saved -> {out}")
    return SkyAmplitude(s_hat32, s_hat_path)


def main():
    ap = argparse.ArgumentParser(
        description="force the sky continuum amplitude onto a spatial field")
    ap.add_argument("--work", type=Path, required=True,
                    help="the run directory; step01, step03 and step04 are read from it")
    ap.add_argument("--cube", type=Path, required=True,
                    help="the sky-INCLUDED cube, the same one step 3 learned from")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory; defaults to <work>/step05")
    ap.add_argument("--basis", default="svd", choices=["pca", "svd"])
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--step04-run", default=None,
                    help="subdirectory under step04, when step 4 wrote more than one")
    ap.add_argument("--blank-channels", default="all", choices=["all", "line1"],
                    help="which channels solve the blank coefficients")
    ap.add_argument("--min-channel-coverage", type=float, default=0.9,
                    help="fraction of channels with data before a spaxel is fitted")
    ap.add_argument("--min-source-distance", type=float, default=15,
                    help="a training spaxel must be this far (px) from any source")
    ap.add_argument("--min-main-source-distance", type=float, default=50,
                    help="extra exclusion radius around the main source; 0 disables it")
    ap.add_argument("--train-clip-sigma", type=float, default=8,
                    help="reject training points beyond this x the robust scatter")
    ap.add_argument("--main-source-dz", type=float, default=0.005,
                    help="largest redshift difference still counted as the same source")
    ap.add_argument("--n-iter", type=int, default=100,
                    help="median-polish iterations for the spatial field")
    ap.add_argument("--train-xlim", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"))
    ap.add_argument("--train-ylim", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"))
    ap.add_argument("--train-exclude-box", type=int, nargs=4, default=None,
                    metavar=("Y0", "Y1", "X0", "X1"))
    ap.add_argument("--fix-blank-s-at", type=float, default=None,
                    help="hold s at this value in the free blank solve instead of "
                         "solving for it")
    args = ap.parse_args()

    out = args.out or args.work / "step05"
    fit_sky_amplitude(
        step_io.white(args.work), step_io.seg(args.work),
        step_io.sky(args.work, args.basis, args.K),
        step_io.classification(args.work, args.step04_run),
        args.cube, args.work, out,
        basis=args.basis, K=args.K, blank_channels=args.blank_channels,
        min_channel_coverage=args.min_channel_coverage,
        min_source_distance=args.min_source_distance,
        min_main_source_distance=args.min_main_source_distance,
        train_clip_sigma=args.train_clip_sigma,
        main_source_dz=args.main_source_dz, n_iter=args.n_iter,
        train_xlim=args.train_xlim, train_ylim=args.train_ylim,
        train_exclude_box=args.train_exclude_box,
        fix_blank_s_at=args.fix_blank_s_at)


if __name__ == "__main__":
    main()
