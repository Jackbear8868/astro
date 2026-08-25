"""Final per-spaxel sky subtraction using the s-field from step5.

Reads the spatial field s_hat built in step5 and fits every spaxel with s
locked to s_hat(x, y):

    blank  (seg = 0)   D = s_hat * C_sky + Sum_k c_k L_k
    source (seg > 0)   D = Sum_j a_j T_j + s_hat * C_sky + Sum_k c_k L_k

The output is two cubes: sky_subtracted (= data - sky_model) and sky_model
itself. The source template term is NOT part of sky_model -- only sky is
subtracted; the source is preserved.
"""
import datetime
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

from fitting import MIN_COVERAGE, N_SRC, build_templates, fit_blank, fit_source
from templates import air_to_vacuum
from utils import blas_single_thread, wavelength_grid

ROOT = Path(__file__).resolve().parents[2]


def _rel(p):
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT)
    except ValueError:
        return p


def write_cube(path, data, hdr_pri, hdr_data, stat=None, hdr_stat=None):
    """Write in MUSE structure: data-less primary + DATA [+ STAT]."""
    h = hdr_data.copy()
    if stat is None:
        h.pop("ERRDATA", None)
    hdus = [fits.PrimaryHDU(header=hdr_pri),
            fits.ImageHDU(data, h, name="DATA")]
    if stat is not None:
        hdus.append(fits.ImageHDU(stat, hdr_stat, name="STAT"))
    fits.HDUList(hdus).writeto(path, overwrite=True)


@blas_single_thread
def subtract_sky(work, cube, classification, s_field, K, basis="svd",
        blank_channels="all", min_channel_coverage=MIN_COVERAGE):
    """Write the sky-subtracted and sky-model cubes into step06; return that directory."""
    work = Path(work)
    STEP01 = work / "step01"
    STEP03 = work / "step03"
    CUBE = Path(cube)
    out = work / "step06"
    out.mkdir(parents=True, exist_ok=True)

    seg_path = STEP01 / "seg.fits"
    seg   = fits.getdata(seg_path)
    white = np.asarray(fits.getdata(STEP01 / "whitelight.fits"), float)
    print(f"workdir {work}   cube {CUBE.name}")
    print(f"segmentation: {seg_path.name}  source spaxels {int((seg > 0).sum()):,}")

    # The sky model on disk is sampled on the grid of whatever cube step3 read, and
    # the source templates are about to be redshifted onto that same grid. A work
    # directory and a cube from two pointings need only agree in channel count to run
    # to the end, with model and data offset against each other, so the grid is
    # checked against this cube instead of assumed.
    wl_path = STEP03 / "wavelength.npy"
    wl_air  = np.load(wl_path)
    wl_cube = wavelength_grid(fits.getheader(CUBE, "DATA"))
    if wl_air.shape != wl_cube.shape:
        raise SystemExit(f"★ {wl_path} has {wl_air.size} channels but {CUBE} has "
                         f"{wl_cube.size}")
    if not np.allclose(wl_air, wl_cube, atol=1e-6):
        raise SystemExit(f"★ {wl_path} was not built from {CUBE}: the two wavelength "
                         f"grids differ by up to {np.abs(wl_air - wl_cube).max():.4g} A")

    wl_vac = air_to_vacuum(wl_air)
    sky = np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                     np.load(STEP03 / f"sky_basis_{basis}_K{K}.npy")])
    print(f"sky model from {STEP03.name}")

    classification_file = Path(classification)
    if not classification_file.exists():
        raise SystemExit(f"file not found: {classification_file}")
    classification = np.load(classification_file)
    print(f"source model from {classification_file.name}: {len(classification['id'])} sources")

    templates = build_templates(classification, wl_vac)

    s_field = Path(s_field)
    s_hat_2d = np.load(s_field)
    print(f"s-field from {s_field}  median {np.nanmedian(s_hat_2d):.5f}")

    fit_mask = None
    if blank_channels == "line1":
        f = STEP03 / "iter_line_mask.npy"
        if not f.exists():
            raise SystemExit(f"{f.name} not found; re-run step3")
        fit_mask = np.load(f)[0]

    with fits.open(CUBE, memmap=True) as hdul:
        hdr_pri  = hdul[0].header.copy()
        hdr_data = hdul["DATA"].header
        hdr_stat = hdul["STAT"].header
        hdr_stat["HISTORY"] = ("STAT copied unchanged from the input cube; it does NOT "
                               "include the uncertainty of the sky model itself.")
        D = np.asarray(hdul["DATA"].data, np.float32)

    nz, ny, nx = D.shape
    D = D.reshape(nz, -1)
    seg_f = seg.reshape(-1)
    s_hat = s_hat_2d.ravel()

    if s_hat_2d.shape != (ny, nx):
        raise SystemExit(f"s-field shape {s_hat_2d.shape} != cube spatial shape ({ny}, {nx})")

    coverage = np.isfinite(D).sum(axis=0) / nz
    valid    = (white != 0).reshape(-1) & (coverage >= min_channel_coverage)
    sky_model = np.full((nz, ny * nx), np.nan, np.float32)
    A_map     = np.full((N_SRC, ny * nx), np.nan, np.float32)
    s_map     = np.full(ny * nx, np.nan, np.float32)

    blank = valid & (seg_f == 0)
    rids  = np.unique(seg_f[valid & (seg_f > 0)])
    n_src_tot = int((valid & (seg_f > 0)).sum())

    # blank: re-solve with s locked to s_hat
    print(f"blank {int(blank.sum()):,} spaxels (s locked to field)...",
          end="", flush=True)
    t0 = time.time()
    c = fit_blank(D[:, blank], sky, fit_mask=fit_mask, s_fix=s_hat[blank])
    sky_model[:, blank] = sky.T @ c
    s_map[blank] = c[0]
    print(f" {time.time() - t0:.1f}s", flush=True)

    # source regions
    n_notpl = sum(1 for r in rids if int(r) not in templates)
    print(f"source {n_src_tot:,} spaxels, {len(rids)} regions"
          f" ({len(rids) - n_notpl} with template, {n_notpl} without)",
          flush=True)
    done, t0 = 0, time.time()
    for k, rid in enumerate(rids, 1):
        m = valid & (seg_f == rid)
        T = templates.get(int(rid))
        c = fit_source(D[:, m], sky, T, s_fix=s_hat[m], progress=True)
        A_map[:, m] = c[:N_SRC]
        sky_model[:, m] = sky.T @ c[N_SRC:]
        s_map[m] = c[N_SRC]

        done += int(m.sum())
        el = time.time() - t0
        print(f"  {k:>2}/{len(rids)}  ID {int(rid):>3}  "
              f"{'tpl ' + str(T.shape[1]) + ' col' if T is not None else 'no tpl   '}"
              f"  {int(m.sum()):>6} spaxel   done {done:>6,}/{n_src_tot:,}"
              f" ({100 * done / n_src_tot:5.1f}%)   elapsed {el:6.1f}s"
              f"   ETA {el * (n_src_tot - done) / max(done, 1):6.1f}s",
              flush=True)

    # write output
    # Nothing below reads the data again, so the difference overwrites it.
    sub  = np.subtract(D, sky_model, out=D)
    cube = lambda x: x.reshape(nz, ny, nx)
    # STAT is passed through untouched, so it is handed to the writer straight
    # from the input file rather than held in memory: on disk it is already the
    # big-endian float32 that goes back out.
    with fits.open(CUBE, memmap=True) as hdul:
        write_cube(out / "sky_subtracted.fits", cube(sub),
                   hdr_pri, hdr_data, hdul["STAT"].data, hdr_stat)
    write_cube(out / "sky_model.fits", cube(sky_model), hdr_pri, hdr_data)
    np.save(out / "A_map.npy", A_map.reshape(N_SRC, ny, nx))
    np.save(out / "s_map.npy", s_map.reshape(ny, nx))

    meta = dict(
        step="fit_sky",
        cube=str(_rel(CUBE)), seg=str(_rel(seg_path)), sky_dir=str(_rel(STEP03)),
        classification=str(_rel(classification_file)), basis=basis, K=K,
        s_field=str(_rel(Path(s_field))),
        blank_channels=blank_channels, min_channel_coverage=min_channel_coverage,
        n_blank=int(blank.sum()), n_source=n_src_tot,
        n_source_regions=len(rids), n_template_regions=len(templates),
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        git_commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True,
                                  cwd=ROOT).stdout.strip())
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2))

    region = ("all channels" if fit_mask is None
              else f"line1 {int(fit_mask.sum())}/{fit_mask.size} channels")
    print(f"blank {int(blank.sum()):,} (unweighted, {region})"
          f"  source {n_src_tot:,}"
          f"  source regions {len(rids)} ({len(rids) - n_notpl} with template)")
    print(f"saved -> {out}")
    return out
