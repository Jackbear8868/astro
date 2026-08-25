"""Run the whole pipeline for one pointing, driven by its config file.

    conda run -n astro python src/skymodel/run_pipeline.py configs/p01.yaml
    conda run -n astro python src/skymodel/run_pipeline.py configs/p0[1-4].yaml

run_pointing() below is the whole method in one place: six named steps in the order
they happen, plus the segmentation check between the first two. Each step is a
function that can equally be called on its own. Reading run_pointing() is meant to be
enough to know what this pipeline does.

Each step's full output goes to {output}/stepN.log with the equivalent Python call
written at the top, so any step can be repeated by hand; only the lines listed in KEEP
reach the terminal. The full output is worth keeping: step3's spatial-range statistics
and step4's per-source classification table, margin column included, exist nowhere
else.

The white light is computed from the nosky cube. It is no longer used for
detection, but downstream needs it to locate the main source (the blob holding
the brightest pixel), and the sky continuum of the wsky cube lifts the whole
image, which makes the brightest pixel unreliable.
"""
import os

# Before numpy is imported, by anything. The steps run in this process now, and step4
# forks a worker per core -- each one spawning its own BLAS threads would oversubscribe
# the machine. Measured on step3, the step with the largest matrices: 16.1 s at one
# thread against 17.0 s at twenty-four, so nothing is given up. fitting.py sets the
# same variables for the same reason, but it is imported too late to be the one that
# decides.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import contextlib
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from config import MAX_GRID_OFFSET, ROOT, load
from step1_whitelight import whitelight
from step2_object_spectra import object_spectra
from step3_sky_basis import sky_basis
from step4_fit_source import classify_sources
from step5_s_field import fit_s_field
from step6_fit_sky import subtract_sky

# Which lines of a step reach the terminal while it runs; the rest is in the log.
KEEP = {
    "step3": r"spatial restriction|exclude-box|blank spaxels|svd |pca ",
    # step5's s_hat median is the one number that shows the field was estimated at
    # all; left in the log only, a field of NaN passes the terminal unremarked.
    "step5": r"main source|s spatial field|s_hat median|saved",
    "step6": r"blank|source |saved",
}

# An upper bound past the edge of the field is harmless -- the comparison is
# against pixel indices, and no pointing is anywhere near this wide. Writing the
# real NAXIS instead would have to be correct for every pointing, and a value
# that is too small drops part of the region without saying so.
BEYOND_EDGE = 9999


def region_kwargs(reg, prefix=""):
    """sky_region -> the keyword arguments step3 and step5 take.

    prefix is "train_" for step5, whose parameters carry that prefix -- there the
    range restricts the spaxels that train the s field, not the ones the sky is
    learned from. Config ranges
    are half-open with null for "no bound"; xlim/ylim have the same meaning,
    while exclude_box includes both endpoints, so its upper bound loses one.
    """
    x, y = reg["x"], reg["y"]
    lo = lambda v: 0 if v is None else v

    if reg["include"]:
        kw = {}
        if x != [None, None]:
            kw[f"{prefix}xlim"] = [lo(x[0]),
                                   BEYOND_EDGE if x[1] is None else x[1]]
        if y != [None, None]:
            kw[f"{prefix}ylim"] = [lo(y[0]),
                                   BEYOND_EDGE if y[1] is None else y[1]]
        return kw

    return {f"{prefix}exclude_box": [
        lo(y[0]), BEYOND_EDGE if y[1] is None else y[1] - 1,
        lo(x[0]), BEYOND_EDGE if x[1] is None else x[1] - 1]}


def call_repr(fn, kwargs):
    """The step call written out as Python, for the head of its log.

    A log that records how the step was reached is what makes the step repeatable
    by hand, and this line can be pasted into an interpreter as it stands. Paths are
    shortened against the repository root: an absolute path from someone else's
    machine is noise in a file another reader is meant to use.
    """
    def show(v):
        if isinstance(v, Path):
            try:
                return repr(str(v.resolve().relative_to(ROOT)))
            except ValueError:
                return repr(str(v))
        return repr(v)
    args = ", ".join(f"{k}={show(v)}" for k, v in kwargs.items())
    return f"{fn.__module__}.{fn.__name__}({args})"


class _Tee:
    """Collect a step's stdout: everything to the log, some of it to the terminal.

    Line-buffered because print() writes the text and the newline separately, and a
    KEEP pattern has to be matched against a whole line. `tail` holds back the last
    few non-matching lines instead, for a step whose summary is at the end.
    """

    def __init__(self, log, keep=None, tail=0):
        self.log, self.keep, self.tail = log, keep, tail
        self.buf, self.held = "", []
        # The real terminal, captured before redirect_stdout puts this object in its
        # place. Calling print() from write() would send the line straight back here.
        self.term = sys.stdout

    def _echo(self, line):
        self.term.write("    " + line.rstrip() + "\n")
        self.term.flush()

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.log.write(line + "\n")
            if self.keep and re.search(self.keep, line):
                self._echo(line)
            elif self.tail:
                self.held.append(line)
        return len(s)

    def flush(self):
        self.log.flush()

    def close(self):
        if self.buf:
            self.log.write(self.buf + "\n")
            self.buf = ""
        for line in self.held[-self.tail:] if self.tail else []:
            self._echo(line)
        self.log.flush()


def run_step(label, fn, kwargs, log_path, keep=None, tail=0):
    """Call one step in this process, sending its output to log_path.

    Whatever the step returns is passed back, so the pipeline uses the paths the step
    itself produced rather than rebuilding them from the same naming rules a second
    time.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write(call_repr(fn, kwargs) + "\n\n")
        log.flush()
        tee = _Tee(log, keep, tail)
        try:
            with contextlib.redirect_stdout(tee):
                result = fn(**kwargs)
        except BaseException:
            # The traceback goes to the log as well: the terminal only ever saw the
            # KEEP lines, so without this the log would end mid-step with no reason.
            tee.close()
            log.write("\n" + traceback.format_exc())
            print(f"★ {label} failed; full output in {log_path}", flush=True)
            raise
        tee.close()
    return result


def place_segmentation(seg_src, out, max_offset=MAX_GRID_OFFSET):
    """Copy the professor's segmentation next to the white light and confirm the
    two share a pixel grid.

    Equal shapes do not prove the same grid, so the check is "where on the sky
    does this pixel point", not a keyword-by-keyword comparison: the seg carries
    a CD matrix while the cube uses PC + CDELT, and their CRPIX differ by 0.01 px,
    both of which a literal comparison would report as a mismatch.

    max_offset above the default is a decision to run anyway on a pointing whose
    headers disagree. It comes from that pointing's config and is printed when it
    is above the default, so the bypass is recorded twice -- in the file and in
    the step log -- rather than living in whoever's shell history raised it.
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
    if off > max_offset:
        raise SystemExit(f"★ seg and white light grids are {off:.2f} px apart "
                         "(largest of the four corners and the centre); "
                         f"the limit is {max_offset:g} px. Raise "
                         "max_grid_offset in this pointing's config to run anyway")
    print(f"    {len(np.unique(s)) - 1} sources, mask {100 * (s > 0).mean():.1f}%, "
          f"grid offset {off:.3f} px")
    if off > MAX_GRID_OFFSET:
        print(f"    ! grid offset {off:.3f} px exceeds the usual limit "
              f"{MAX_GRID_OFFSET:g} px and was allowed by max_grid_offset "
              f"{max_offset:g} in the config. Anything this pointing produces from sky "
              f"coordinates carries that offset.")


def run_pointing(cfg_path):
    """The method, in order: six steps, and the segmentation check between one and two."""
    cfg = load(cfg_path)
    out = cfg["output"]
    inp = cfg["input"]
    basis, src = cfg["sky_line_basis"], cfg["source_fit"]
    amp, spx = cfg["sky_amplitude"], cfg["spaxel_fit"]

    for key, path in inp.items():
        if not path.exists():
            raise SystemExit(f"★ input.{key} not found: {path}")
    out.mkdir(parents=True, exist_ok=True)

    reg = cfg["sky_region"]
    basis_region  = region_kwargs(reg) if "basis" in reg["apply_to"] else {}
    train_region = region_kwargs(reg, "train_") if "sky_amplitude" in reg["apply_to"] else {}

    print("=" * 70)
    print(f"  pointing #{cfg['pointing']}  ->  {out.relative_to(ROOT)}"
          f"   [{Path(cfg_path).name}]")
    print(f"  sky region {reg['x']} x {reg['y']}  "
          f"{'include' if reg['include'] else 'exclude'} -> {reg['apply_to']}")
    print("=" * 70)
    t0 = time.time()

    print("--- [1/7] step1 white light (from the nosky cube)")
    run_step("step1", whitelight,
             dict(cube=inp["nosky"], out=out / "step01"),
             out / "step1.log")

    print("--- [2/7] the professor's segmentation")
    place_segmentation(inp["seg"], out, cfg["max_grid_offset"])

    print("--- [3/7] step2 source spectra (nosky, for classification)")
    run_step("step2", object_spectra,
             dict(work=out, cube=inp["nosky"], out=out / "step02"),
             out / "step2.log")

    print("--- [4/7] step3 sky basis")
    run_step("step3", sky_basis,
             dict(work=out, cube=inp["cube"], K=basis["K"],
                  methods=[basis["method"]], seed=basis["seed"],
                  continuum_window=basis["continuum_window"],
                  line_thresholds=basis["line_thresholds"],
                  max_iter=basis["max_iter"], clip_sigma=basis["clip_sigma"],
                  **basis_region),
             out / "step3.log", keep=KEEP["step3"])

    print("--- [5/7] step4 template fitting and classification")
    # step4 returns the classification file it wrote, for the last mask iteration
    # asked for. Rebuilding that name here from make_tag/make_suffix would be the
    # same naming rule written twice, and the two would drift.
    best = run_step("step4", classify_sources,
                    dict(work=out, K=basis["K"], id="all", basis=basis["method"],
                         fix_s_at=src["fix_s_at"],
                         star_window=src["fit_window"],
                         gal_window=src["fit_window"],
                         line_mask_iter=src["line_mask_iter"],
                         zmin=src["z_min"], zmax=src["z_max"],
                         zstep=src["z_step"], star_dz=src["star_dz"],
                         num_workers=src["num_workers"],
                         spec_dir=out / "step02"),
                    out / "step4.log", tail=3)

    line_iter = src["line_mask_iter"][-1]
    print(f"--- [6/7] step5 build the s field   [mask iter {line_iter}]")
    run_step("step5", fit_s_field,
             dict(work=out, cube=inp["cube"], classification=best, K=basis["K"],
                  basis=basis["method"],
                  blank_channels=spx["blank_channels"],
                  min_channel_coverage=spx["min_channel_coverage"],
                  min_source_distance=amp["min_source_distance"],
                  min_main_source_distance=amp["min_main_source_distance"],
                  train_clip_sigma=amp["train_clip_sigma"],
                  main_source_dz=amp["main_source_dz"],
                  **train_region),
             out / "step5.log", keep=KEEP["step5"])

    print("--- [7/7] step6 final sky subtraction")
    run_step("step6", subtract_sky,
             dict(work=out, cube=inp["cube"], classification=best, K=basis["K"],
                  basis=basis["method"],
                  s_field=out / "step05/s_hat.npy",
                  blank_channels=spx["blank_channels"],
                  min_channel_coverage=spx["min_channel_coverage"]),
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
