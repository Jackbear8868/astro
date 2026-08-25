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

from utils import (MIN_COVERAGE, N_SRC, air_to_vacuum, blas_single_thread,
                   build_templates, fit_blank, fit_source, wavelength_grid)

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
def subtract_sky(work, cube, white, seg, sky, classification, s_field, K,
        basis="svd", blank_channels="all", min_channel_coverage=MIN_COVERAGE):
    """Write the sky-subtracted and sky-model cubes into step06; return that directory.

    white, seg, sky, classification and s_field are what the earlier steps
    returned, in memory. This step's products are the deliverable, so they are
    written whatever keep_intermediate said about the ones before them.
    """
    work = Path(work)
    CUBE = Path(cube)
    out = work / "step06"
    out.mkdir(parents=True, exist_ok=True)

    seg_path, seg = seg.path, seg.data
    white = np.asarray(white.data, float)
    print(f"workdir {work}   cube {CUBE.name}")
    print(f"segmentation: {seg_path.name}  source spaxels {int((seg > 0).sum()):,}")

    # The sky model was learned on the grid of whatever cube step3 read, and the
    # source templates are about to be redshifted onto that same grid. A config
    # naming one pointing's cube in step3 and another's here needs only agree in
    # channel count to run to the end, with model and data offset against each
    # other, so the grid is checked instead of assumed.
    wl_air  = sky.wavelength
    wl_cube = wavelength_grid(fits.getheader(CUBE, "DATA"))
    if wl_air.shape != wl_cube.shape:
        raise SystemExit(f"★ step3's sky model has {wl_air.size} channels but "
                         f"{CUBE} has {wl_cube.size}")
    if not np.allclose(wl_air, wl_cube, atol=1e-6):
        raise SystemExit(f"★ step3's sky model was not built from {CUBE}: the two "
                         f"wavelength grids differ by up to "
                         f"{np.abs(wl_air - wl_cube).max():.4g} A")

    wl_vac = air_to_vacuum(wl_air)
    fit_mask = sky.iter_line_mask[0] if blank_channels == "line1" else None
    # From here `sky` is the design matrix the spaxel fits use: the continuum as
    # row 0, the K line vectors under it.
    sky = np.vstack([sky.continuum, sky.basis[basis]])
    print(f"sky model {sky.shape}  basis {basis} K{K}")

    print(f"source model from {classification.path.name}: "
          f"{len(classification.data['id'])} sources")

    templates = build_templates(classification.data, wl_vac)

    s_hat_2d = s_field.data
    print(f"s-field from {s_field.path}  median {np.nanmedian(s_hat_2d):.5f}")

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
        cube=str(_rel(CUBE)), seg=str(_rel(seg_path)),
        sky_dir=str(_rel(work / "step03")),
        classification=str(_rel(classification.path)), basis=basis, K=K,
        s_field=str(_rel(s_field.path)),
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


# Without this the file would import and exit 0 when run, which reads as having
# done the step. There is one way into the pipeline, and this says where it is.
if __name__ == "__main__":
    raise SystemExit(
        "★ the steps are not run on their own; run the pipeline:\n"
        "      python src/skymodel/run_pipeline.py configs/pNN.yaml")
