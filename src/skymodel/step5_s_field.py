"""Build the sky-continuum spatial field s_hat(x, y).

    (1) Solve all blank spaxels freely to get s_free -- each spaxel's own
        sky-continuum coefficient, unconstrained.
    (2) Identify the main source group (brightest-pixel blob + redshift filter).
    (3) From s_free, build a smooth spatial field s_hat using only training
        points far from all sources.

The field is what step6 locks s to when fitting every spaxel. By replacing
per-spaxel freedom with a smooth surface, source light has nowhere to hide
inside the sky model and is preserved in the residual.

fit_s_field() does the work and can be called directly; main() is the same
thing driven from the command line.

    conda run -n astro python src/skymodel/step5_s_field.py \\
        --basis svd -K 30 --work results/skymodel/p01 \\
        --cube data/wsky/DATACUBE_FINAL_1.fits \\
        --classification results/skymodel/p01/step04/classification_*.npz
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

from fitting import MIN_COVERAGE, fit_blank
from plotting import plot_main_group
from templates import air_to_vacuum
from utils import (C_KMS, DZ_MAX, build_s_field, galaxy_redshifts,
                   main_source_group)

ROOT = Path(__file__).resolve().parents[2]


def _rel(p):
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT)
    except ValueError:
        return p


def fit_s_field(work, cube, classification, K, basis="svd", seg=None, sky_dir=None,
        blank_channels="all", min_channel_coverage=MIN_COVERAGE, fix_blank_s_at=None,
        min_source_distance=15.0, min_main_source_distance=50.0, train_exclude_box=None,
        train_xlim=None, train_ylim=None, train_clip_sigma=8.0, main_source_dz=DZ_MAX,
        out=None, argv=None):
    """Write s_hat.npy, s_free.npy, main_group.png and meta.json; return that directory.

    argv is written into meta.json as the record of how the step was launched, and
    only main() can know it: called directly, sys.argv belongs to whatever called
    fit_s_field(), not to this step. It stays null in that case -- everything it would have
    carried is already written out under its own name.
    """
    work = Path(work)
    STEP01 = work / "step01"
    STEP03 = work / "step03"
    CUBE = Path(cube)
    out = Path(out) if out else work / "step05"
    out.mkdir(parents=True, exist_ok=True)

    seg_path = Path(seg) if seg else STEP01 / "seg.fits"
    seg   = fits.getdata(seg_path)
    white = np.asarray(fits.getdata(STEP01 / "whitelight.fits"), float)
    print(f"workdir {work}   cube {CUBE.name}")
    print(f"segmentation: {seg_path.name}  source spaxels {int((seg > 0).sum()):,}")

    wl_vac = air_to_vacuum(np.load(STEP03 / "wavelength.npy"))
    sky_dir = Path(sky_dir) if sky_dir else STEP03
    sky = np.vstack([np.load(sky_dir / "sky_continuum.npy"),
                     np.load(sky_dir / f"sky_basis_{basis}_K{K}.npy")])
    print(f"sky model from {sky_dir.name}")

    classification_file = Path(classification)
    if not classification_file.exists():
        raise SystemExit(f"file not found: {classification_file}")

    fit_mask = None
    if blank_channels == "line1":
        f = STEP03 / "iter_line_mask.npy"
        if not f.exists():
            raise SystemExit(f"{f.name} not found; re-run step3")
        fit_mask = np.load(f)[0]

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
    # The redshifts must come from the fit --classification names; a workspace can hold the
    # results of several step4 runs at once.
    tag = classification_file.stem.removeprefix("classification_")
    mg, mids, mk = main_source_group(seg, white, classification_file.parent,
                                     main_source_dz, tag=tag)
    all_ids = main_source_group(seg, white)[1]
    z0 = galaxy_redshifts(classification_file.parent, [int(seg[mk])], tag)[int(seg[mk])]
    print(f"  main source (brightest pixel y={mk[0]}, x={mk[1]}): {len(mids)} IDs"
          f", {int(mg.sum()):,} px"
          f" (dz <= {main_source_dz:g},"
          f" i.e. {C_KMS * main_source_dz / (1 + z0):.0f} km/s @ z={z0:.4f})")

    plot_main_group(seg, white, mg, mids, all_ids, mk,
                    out / "main_group.png", title=Path(work).name)

    # build field
    t0 = time.time()
    s_hat, sf_train = build_s_field(
        s2d, seg, ok2d, min_source_distance, min_main_source_distance or None,
        train_clip_sigma, exclude=sf_box, main=mg)
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

    # save
    np.save(out / "s_hat.npy", s_hat.astype(np.float32))
    np.save(out / "s_free.npy", s_free.reshape(ny, nx).astype(np.float32))

    meta = dict(
        step="s_field",
        cube=str(_rel(CUBE)), seg=str(_rel(seg_path)), sky_dir=str(_rel(sky_dir)),
        classification=str(_rel(classification_file)), basis=basis, K=K,
        blank_channels=blank_channels, fix_blank_s_at=fix_blank_s_at,
        min_channel_coverage=min_channel_coverage,
        sky_amplitude_params=dict(
            min_source_distance=min_source_distance,
            min_main_source_distance=min_main_source_distance,
            train_clip_sigma=train_clip_sigma,
            train_exclude_box=train_exclude_box,
            train_xlim=train_xlim, train_ylim=train_ylim,
            main_source_dz=main_source_dz),
        main_ids=[int(i) for i in mids],
        n_blank=int(blank.sum()), n_train=int(sf_train.sum()),
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        git_commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True,
                                  cwd=ROOT).stdout.strip(),
        argv=argv)
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"saved -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Build the sky-continuum spatial field from a free blank solve")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, required=True,
                    help="number of sky-line basis vectors")
    ap.add_argument("--classification", required=True,
                    help="step4 classification file (.npz)")
    ap.add_argument("--seg", default=None,
                    help="segmentation map; default step01/seg.fits")
    ap.add_argument("--sky-dir", default=None,
                    help="directory containing sky_continuum.npy and sky_basis_*.npy; default step03")
    ap.add_argument("--blank-channels", choices=["all", "line1"], default="all",
                    help="channels used for the free blank solve")
    ap.add_argument("--min-channel-coverage", type=float, default=MIN_COVERAGE,
                    help="fraction of wavelength channels that must carry data before "
                         "a spaxel is fitted; the field-of-view edge ring falls below it")
    ap.add_argument("--fix-blank-s-at", type=float, default=None,
                    help="fix s in the free blank solve (diagnostic use only)")
    ap.add_argument("--min-source-distance", type=float, default=15.0,
                    help="training points must be >= this far (px) from any source")
    ap.add_argument("--min-main-source-distance", type=float, default=50.0,
                    help="extra exclusion radius around the main source; 0 to disable")
    ap.add_argument("--train-exclude-box", type=int, nargs=4, default=None,
                    metavar=("Y0", "Y1", "X0", "X1"),
                    help="exclude spaxels inside this box from s-field training")
    ap.add_argument("--train-xlim", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="restrict s-field training to this x range (inclusive LO, exclusive HI)")
    ap.add_argument("--train-ylim", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="restrict s-field training to this y range")
    ap.add_argument("--train-clip-sigma", type=float, default=8.0,
                    help="reject training points with |s - median| > clip x robust scatter")
    ap.add_argument("--main-source-dz", type=float, default=DZ_MAX,
                    help="max redshift difference for main-source grouping")
    ap.add_argument("--work", required=True,
                    help="working directory (contains step01/step03/step04)")
    ap.add_argument("--cube", required=True,
                    help="sky-included (wsky) cube")
    ap.add_argument("--out", default=None,
                    help="output directory; default {work}/step05")
    args = ap.parse_args()
    fit_s_field(args.work, args.cube, args.classification, args.K, argv=sys.argv[1:], basis=args.basis,
        seg=args.seg, sky_dir=args.sky_dir, blank_channels=args.blank_channels,
        min_channel_coverage=args.min_channel_coverage, fix_blank_s_at=args.fix_blank_s_at,
        min_source_distance=args.min_source_distance, min_main_source_distance=args.min_main_source_distance,
        train_exclude_box=args.train_exclude_box, train_xlim=args.train_xlim,
        train_ylim=args.train_ylim, train_clip_sigma=args.train_clip_sigma,
        main_source_dz=args.main_source_dz, out=args.out)


if __name__ == "__main__":
    main()
