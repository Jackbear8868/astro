"""Run the whole pipeline for one pointing, driven by its config file.

    conda run -n astro python src/skymodel/run_pipeline.py configs/p01.yaml
    conda run -n astro python src/skymodel/run_pipeline.py configs/p0[1-4].yaml

Every step runs as a subprocess, so any of them can still be run by hand with
the same arguments; the command line is written at the top of {output}/stepN.log
together with the step's full output, and only the lines listed in KEEP reach
the terminal. The full output is worth keeping: step3's spatial-range statistics
and step4's per-source classification table, margin column included, exist
nowhere else.

The white light is computed from the nosky cube. It is no longer used for
detection, but downstream needs it to locate the main source (the blob holding
the brightest pixel), and the sky continuum of the wsky cube lifts the whole
image, which makes the brightest pixel unreliable.
"""
import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from config import ROOT, load
from step4_fit_source import make_tag

HERE = Path(__file__).resolve().parent

# Which lines of a step reach the terminal while it runs; the rest is in the log.
KEEP = {
    "step3": r"spatial restriction|exclude-box|blank spaxels|svd |pca ",
    "step5": r"main source|s spatial field|saved",
    "step6": r"blank|source |saved",
}

# An upper bound past the edge of the field is harmless -- the comparison is
# against pixel indices, and no pointing is anywhere near this wide. Writing the
# real NAXIS instead would have to be correct for every pointing, and a value
# that is too small drops part of the region without saying so.
BEYOND_EDGE = 9999


def region_argv(reg, prefix=""):
    """sky_region -> the arguments step3 and step5 take.

    prefix is "sf-" for step5, whose options carry that prefix. Config ranges
    are half-open with null for "no bound"; --xlim/--ylim have the same meaning,
    while --exclude-box includes both endpoints, so its upper bound loses one.
    """
    x, y = reg["x"], reg["y"]
    lo = lambda v: 0 if v is None else v

    if reg["include"]:
        argv = []
        if x != [None, None]:
            argv += [f"--{prefix}xlim", lo(x[0]),
                     BEYOND_EDGE if x[1] is None else x[1]]
        if y != [None, None]:
            argv += [f"--{prefix}ylim", lo(y[0]),
                     BEYOND_EDGE if y[1] is None else y[1]]
        return argv

    return [f"--{prefix}exclude-box",
            lo(y[0]), BEYOND_EDGE if y[1] is None else y[1] - 1,
            lo(x[0]), BEYOND_EDGE if x[1] is None else x[1] - 1]


def run_step(label, script, argv, log_path, keep=None, tail=0):
    """Run one step as a subprocess, streaming its output to the log."""
    cmd = [sys.executable, str(HERE / script)] + [str(a) for a in argv]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    buf = []
    with log_path.open("w") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, text=True, bufsize=1,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            log.write(line)
            if keep and re.search(keep, line):
                print("    " + line.rstrip(), flush=True)
            elif tail:
                buf.append(line)
        code = proc.wait()
    for line in buf[-tail:]:
        print("    " + line.rstrip(), flush=True)
    if code:
        raise SystemExit(f"★ {label} failed (exit {code}); full output in {log_path}")


def place_segmentation(seg_src, out):
    """Copy the professor's segmentation next to the white light and confirm the
    two share a pixel grid.

    Equal shapes do not prove the same grid, so the check is "where on the sky
    does this pixel point", not a keyword-by-keyword comparison: the seg carries
    a CD matrix while the cube uses PC + CDELT, and their CRPIX differ by 0.01 px,
    both of which a literal comparison would report as a mismatch.
    """
    dst = out / "step01/seg.fits"
    shutil.copy(seg_src, dst)
    s, hs = fits.getdata(dst, header=True)
    w, hw = fits.getdata(out / "step01/whitelight.fits", header=True)
    if s.shape != w.shape:
        raise SystemExit(f"★ seg {s.shape} and white light {w.shape} differ in shape")
    if "CTYPE1" not in hw:
        raise SystemExit("★ white light carries no WCS -- written by an older step1, re-run it")

    ny, nx = s.shape
    yy = np.array([0, 0, ny - 1, ny - 1, ny // 2])
    xx = np.array([0, nx - 1, 0, nx - 1, nx // 2])
    ws, ww = WCS(hs).celestial, WCS(hw).celestial
    sep = ws.pixel_to_world(xx, yy).separation(ww.pixel_to_world(xx, yy)).arcsec
    off = sep.max() / (proj_plane_pixel_scales(ww)[0] * 3600)
    if off > 0.1:
        raise SystemExit(f"★ seg and white light grids are {off:.2f} px apart "
                         "(largest of the four corners and the centre)")
    print(f"    {len(np.unique(s)) - 1} sources, mask {100 * (s > 0).mean():.1f}%, "
          f"grid offset {off:.3f} px")


def run_pointing(cfg_path):
    cfg = load(cfg_path)
    out = cfg["output"]
    inp = cfg["input"]
    basis, src = cfg["sky_line_basis"], cfg["source_fit"]
    sfield, spx = cfg["s_field"], cfg["spaxel_fit"]

    for key, path in inp.items():
        if not path.exists():
            raise SystemExit(f"★ input.{key} not found: {path}")
    out.mkdir(parents=True, exist_ok=True)

    reg = cfg["sky_region"]
    basis_region  = region_argv(reg) if "basis" in reg["apply_to"] else []
    sfield_region = region_argv(reg, "sf-") if "s_field" in reg["apply_to"] else []

    print("=" * 70)
    print(f"  pointing #{cfg['pointing']}  ->  {out.relative_to(ROOT)}"
          f"   [{Path(cfg_path).name}]")
    print(f"  sky region {reg['x']} x {reg['y']}  "
          f"{'include' if reg['include'] else 'exclude'} -> {reg['apply_to']}")
    print("=" * 70)
    t0 = time.time()

    print("--- [1/7] step1 white light (from the nosky cube)")
    run_step("step1", "step1_whitelight.py",
             [inp["nosky"], "--out", out / "step01"], out / "step1.log")

    print("--- [2/7] the professor's segmentation")
    place_segmentation(inp["seg"], out)

    print("--- [3/7] step2 source spectra (nosky, for classification)")
    run_step("step2", "step2_object_spectra.py",
             ["--work", out, "--cube", inp["nosky"], "--out", out / "step02"],
             out / "step2.log")

    print("--- [4/7] step3 sky basis")
    run_step("step3", "step3_sky_basis.py",
             ["--methods", basis["method"], "-K", basis["K"],
              "--seed", basis["seed"],
              "--continuum-window", basis["continuum_window"],
              "--line-thresholds", *basis["line_thresholds"],
              "--max-iter", basis["max_iter"],
              "--clip-sigma", basis["clip_sigma"],
              "--work", out, "--cube", inp["cube"], *basis_region],
             out / "step3.log", keep=KEEP["step3"])

    print("--- [5/7] step4 template fitting and classification")
    s_fix = ["--s-free"] if src["s_fix"] is None else ["--s-fix", src["s_fix"]]
    run_step("step4", "step4_fit_source.py",
             ["--id", "all", "--basis", basis["method"], "-K", basis["K"], *s_fix,
              "--star-window", *src["fit_window"],
              "--gal-window", *src["fit_window"],
              "--line-mask-iter", *src["line_mask_iter"],
              "--zmin", src["z_min"], "--zmax", src["z_max"],
              "--zstep", src["z_step"], "--star-dz", src["star_dz"],
              "--num-workers", src["num_workers"],
              "--spec-dir", out / "step02", "--work", out],
             out / "step4.log", tail=3)

    # The classification filename is built by step4's own make_tag, so the two
    # cannot drift apart. step4 writes one file per mask iteration; the last one
    # asked for is the one carried downstream.
    line_iter = src["line_mask_iter"][-1]
    best = out / "step04" / f"classification_{make_tag(basis['method'], basis['K'], src['s_fix'], src['fit_window'], src['fit_window'], sky_basis=False, line_iter=line_iter)}.npz"

    print(f"--- [6/7] step5 build the s field   [mask iter {line_iter}]")
    run_step("step5", "step5_s_field.py",
             ["--basis", basis["method"], "-K", basis["K"], "--work", out,
              "--cube", inp["cube"], "--best", best,
              "--blank-region", spx["blank_region"],
              "--min-coverage", spx["min_coverage"],
              "--sf-r-far", sfield["r_far"],
              "--sf-r-far-haro", sfield["r_far_main"],
              "--sf-clip", sfield["clip"],
              "--main-dz-max", sfield["main_dz_max"], *sfield_region],
             out / "step5.log", keep=KEEP["step5"])

    print("--- [7/7] step6 final sky subtraction")
    run_step("step6", "step6_fit_sky.py",
             ["--basis", basis["method"], "-K", basis["K"], "--work", out,
              "--cube", inp["cube"], "--best", best,
              "--s-field", out / "step05/s_hat.npy",
              "--blank-region", spx["blank_region"],
              "--min-coverage", spx["min_coverage"]],
             out / "step6.log", keep=KEEP["step6"])

    free = shutil.disk_usage(ROOT).free / 1024 ** 3
    print(f"*** pointing #{cfg['pointing']} done in {time.time() - t0:.0f} s"
          f"   {free:.0f} GB free")


def main():
    ap = argparse.ArgumentParser(
        description="Run the sky reconstruction pipeline for one or more pointings")
    ap.add_argument("config", nargs="+",
                    help="pointing config file(s), e.g. configs/p01.yaml")
    args = ap.parse_args()
    for path in args.config:
        run_pointing(path)


if __name__ == "__main__":
    main()
