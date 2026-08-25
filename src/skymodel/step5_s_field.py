"""Build the sky-continuum spatial field s_hat(x, y).

    (1) Solve all blank spaxels freely to get s_free -- each spaxel's own
        sky-continuum coefficient, unconstrained.
    (2) Identify the main source group (brightest-pixel blob + redshift filter).
    (3) From s_free, build a smooth spatial field s_hat using only training
        points far from all sources.

The field is what step6 locks s to when fitting every spaxel. By replacing
per-spaxel freedom with a smooth surface, source light has nowhere to hide
inside the sky model and is preserved in the residual.

run() does the work and can be called directly; main() is the same thing driven from
the command line.

    conda run -n astro python src/skymodel/step5_s_field.py \\
        --basis svd -K 30 --work results/skymodel/p01 \\
        --cube data/wsky/DATACUBE_FINAL_1.fits \\
        --best results/skymodel/p01/step04/classification_*.npz
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


def run(work, cube, best, K, basis="svd", seg=None, sky_dir=None,
        blank_region="all", min_coverage=MIN_COVERAGE, blank_s_fix=None,
        sf_r_far=15.0, sf_r_far_haro=50.0, sf_exclude_box=None,
        sf_xlim=None, sf_ylim=None, sf_clip=8.0, main_dz_max=DZ_MAX,
        out=None):
    """Write s_hat.npy, s_free.npy, main_group.png and meta.json; return that directory."""
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

    best_file = Path(best)
    if not best_file.exists():
        raise SystemExit(f"file not found: {best_file}")

    fit_mask = None
    if blank_region == "line1":
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
    valid = (white != 0).reshape(-1) & (coverage >= min_coverage)
    blank = valid & (seg_f == 0)

    # free blank solve
    print(f"blank {int(blank.sum()):,} spaxels (free solve)...", end="", flush=True)
    t0 = time.time()
    c = fit_blank(D[:, blank], sky, fit_mask=fit_mask, s_fix=blank_s_fix)
    print(f" {time.time() - t0:.1f}s", flush=True)

    s_free = np.full(ny * nx, np.nan)
    s_free[blank] = c[0]
    s2d = s_free.reshape(ny, nx)
    ok2d = blank.reshape(ny, nx) & np.isfinite(s2d)

    # spatial exclusion mask
    sf_box = None
    if sf_xlim or sf_ylim or sf_exclude_box:
        yy, xx = np.mgrid[0:ny, 0:nx]
        sf_box = np.zeros((ny, nx), bool)
        if sf_xlim:
            sf_box |= ~((xx >= sf_xlim[0]) & (xx < sf_xlim[1]))
        if sf_ylim:
            sf_box |= ~((yy >= sf_ylim[0]) & (yy < sf_ylim[1]))
        if sf_exclude_box:
            by0, by1, bx0, bx1 = sf_exclude_box
            sf_box |= (yy >= by0) & (yy <= by1) & (xx >= bx0) & (xx <= bx1)

    # main source group
    # The redshifts must come from the fit --best names; a workspace can hold the
    # results of several step4 runs at once.
    tag = best_file.stem.removeprefix("classification_")
    mg, mids, mk = main_source_group(seg, white, best_file.parent,
                                     main_dz_max, tag=tag)
    all_ids = main_source_group(seg, white)[1]
    z0 = galaxy_redshifts(best_file.parent, [int(seg[mk])], tag)[int(seg[mk])]
    print(f"  main source (brightest pixel y={mk[0]}, x={mk[1]}): {len(mids)} IDs"
          f", {int(mg.sum()):,} px"
          f" (dz <= {main_dz_max:g},"
          f" i.e. {C_KMS * main_dz_max / (1 + z0):.0f} km/s @ z={z0:.4f})")

    plot_main_group(seg, white, mg, mids, all_ids, mk,
                    out / "main_group.png", title=Path(work).name)

    # build field
    t0 = time.time()
    s_hat, sf_train = build_s_field(
        s2d, seg, ok2d, sf_r_far, sf_r_far_haro or None,
        sf_clip, exclude=sf_box, main=mg)
    print(f"s spatial field: {int(sf_train.sum()):,} training spaxels"
          f" (dist > {sf_r_far:g} px from sources"
          + (f", Haro 11 > {sf_r_far_haro:g} px" if sf_r_far_haro else "")
          + f", clip {sf_clip:g} sigma"
          + (f", x {sf_xlim}" if sf_xlim else "")
          + (f", y {sf_ylim}" if sf_ylim else "")
          + (", exclude-box" if sf_exclude_box else "")
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
        best=str(_rel(best_file)), basis=basis, K=K,
        blank_region=blank_region, blank_s_fix=blank_s_fix,
        min_coverage=min_coverage,
        s_field_params=dict(r_far=sf_r_far, r_far_haro=sf_r_far_haro,
                            clip=sf_clip, exclude_box=sf_exclude_box,
                            xlim=sf_xlim, ylim=sf_ylim,
                            main_dz_max=main_dz_max),
        main_ids=[int(i) for i in mids],
        n_blank=int(blank.sum()), n_train=int(sf_train.sum()),
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        git_commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True,
                                  cwd=ROOT).stdout.strip(),
        argv=sys.argv[1:])
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
    ap.add_argument("--best", required=True,
                    help="step4 classification file (.npz)")
    ap.add_argument("--seg", default=None,
                    help="segmentation map; default step01/seg.fits")
    ap.add_argument("--sky-dir", default=None,
                    help="directory containing sky_continuum.npy and sky_basis_*.npy; default step03")
    ap.add_argument("--blank-region", choices=["all", "line1"], default="all",
                    help="channels used for the free blank solve")
    ap.add_argument("--min-coverage", type=float, default=MIN_COVERAGE,
                    help="fraction of wavelength channels that must carry data before "
                         "a spaxel is fitted; the field-of-view edge ring falls below it")
    ap.add_argument("--blank-s-fix", type=float, default=None,
                    help="fix s in the free blank solve (diagnostic use only)")
    ap.add_argument("--sf-r-far", type=float, default=15.0,
                    help="training points must be >= this far (px) from any source")
    ap.add_argument("--sf-r-far-haro", type=float, default=50.0,
                    help="extra exclusion radius around the main source; 0 to disable")
    ap.add_argument("--sf-exclude-box", type=int, nargs=4, default=None,
                    metavar=("Y0", "Y1", "X0", "X1"),
                    help="exclude spaxels inside this box from s-field training")
    ap.add_argument("--sf-xlim", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="restrict s-field training to this x range (inclusive LO, exclusive HI)")
    ap.add_argument("--sf-ylim", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="restrict s-field training to this y range")
    ap.add_argument("--sf-clip", type=float, default=8.0,
                    help="reject training points with |s - median| > clip x robust scatter")
    ap.add_argument("--main-dz-max", type=float, default=DZ_MAX,
                    help="max redshift difference for main-source grouping")
    ap.add_argument("--work", required=True,
                    help="working directory (contains step01/step03/step04)")
    ap.add_argument("--cube", required=True,
                    help="sky-included (wsky) cube")
    ap.add_argument("--out", default=None,
                    help="output directory; default {work}/step05")
    args = ap.parse_args()
    run(args.work, args.cube, args.best, args.K, basis=args.basis,
        seg=args.seg, sky_dir=args.sky_dir, blank_region=args.blank_region,
        min_coverage=args.min_coverage, blank_s_fix=args.blank_s_fix,
        sf_r_far=args.sf_r_far, sf_r_far_haro=args.sf_r_far_haro,
        sf_exclude_box=args.sf_exclude_box, sf_xlim=args.sf_xlim,
        sf_ylim=args.sf_ylim, sf_clip=args.sf_clip,
        main_dz_max=args.main_dz_max, out=args.out)


if __name__ == "__main__":
    main()
