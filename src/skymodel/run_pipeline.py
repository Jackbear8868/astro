"""Run the whole pipeline for one pointing, driven by its config file.

    conda run -n astro python src/skymodel/run_pipeline.py configs/p01.yaml
    conda run -n astro python src/skymodel/run_pipeline.py configs/p0[1-4].yaml

This is the pipeline's only entrance. run_pointing() below is the whole method in one
place: six named steps in the order they happen, plus the segmentation check between
the first two. Reading run_pointing() is meant to be enough to know what this pipeline
does -- including the data flow, because each step is handed what the earlier ones
returned rather than reopening the files they wrote. A step that reads its input from
disk can be handed a file some earlier run left there, and nothing says so.

The cube is the exception: every step that needs it opens it itself. It is the one
input large enough that holding it from one step to the next would cost real memory,
and it is memmapped, so opening it again is cheap.

The products are still written, under {output}/stepNN, unless the config turns
keep_intermediate off; step6's are written either way. They are what the evaluation
scripts read and the only record of the middle of a run, but nothing in the pipeline
reads them back.

Each step's full output goes to {output}/stepN.log, headed by the call that produced
it, so the log records which arguments those products came from; only the lines listed
in KEEP reach the terminal. The full output is worth keeping: step3's spatial-range
statistics and step4's per-source classification table, margin column included, exist
nowhere else.

The white light is computed from the nosky cube. It is no longer used for
detection, but downstream needs it to locate the main source (the blob holding
the brightest pixel), and the sky continuum of the wsky cube lifts the whole
image, which makes the brightest pixel unreliable.

Nothing here fixes the BLAS thread count. Each fitting step holds BLAS at one
thread around its own work (utils.blas_single_thread), so the products do not
follow the thread count of the machine the pipeline is run on.
"""
import argparse
import contextlib
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import NamedTuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from config import MAX_GRID_OFFSET, ROOT, load
from step1_whitelight import whitelight
from step2_object_spectra import object_spectra
from step3_sky_basis import sky_basis
from step4_classify_sources import classify_sources
from step5_fit_s_field import fit_s_field
from step6_subtract_sky import subtract_sky

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


# How much of one argument the log prints before it says what the value is instead.
# The point of the head line is the scalars and the paths; a step is also handed
# whole spectra and maps, and printing those would bury the rest.
ARG_WIDTH = 160


def _fit(text, value):
    """text if it is short enough to read on one line, else what the value is."""
    if len(text) <= ARG_WIDTH and "\n" not in text:
        return text
    size = f" of {len(value)}" if hasattr(value, "__len__") else ""
    return f"<{type(value).__name__}{size}>"


def show(v):
    """One argument of a step call, written for the head of its log.

    An array is written as its shape and dtype: what a run was given is answered
    by which array it was, and the values themselves are in the products beside
    the log. The bundles the steps pass each other are opened up so that the
    paths and tags inside them stay visible.

    Paths are shortened against the repository root: an absolute path from
    someone else's machine is noise in a file another reader is meant to use.
    """
    if isinstance(v, np.ndarray):
        return f"<ndarray {v.shape} {v.dtype}>"
    if isinstance(v, Path):
        try:
            return repr(str(v.resolve().relative_to(ROOT)))
        except ValueError:
            return repr(str(v))
    if isinstance(v, tuple) and hasattr(v, "_fields"):          # a step's bundle
        inner = ", ".join(f"{f}={show(x)}" for f, x in zip(v._fields, v))
        return f"{type(v).__name__}({inner})"
    if isinstance(v, dict):
        return _fit("{" + ", ".join(f"{show(k)}: {show(x)}"
                                    for k, x in v.items()) + "}", v)
    if isinstance(v, (list, tuple)):
        body = ", ".join(show(x) for x in v)
        if isinstance(v, tuple):
            body = f"({body},)" if len(v) == 1 else f"({body})"
        else:
            body = f"[{body}]"
        return _fit(body, v)
    return _fit(repr(v), v)


def call_repr(fn, kwargs):
    """The step call written out as Python, for the head of its log.

    It is the record of which arguments produced the products beside it -- a config
    can be edited afterwards, and then nothing else says what this run was given.
    """
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

    Whatever the step returns is passed back, which is how the pipeline hands one
    step's results to the next instead of each of them reopening the files the one
    before it wrote.
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


class Seg(NamedTuple):
    """The segmentation, as steps 2, 3, 5 and 6 are handed it.

    path is where it was put next to the white light. Steps 5 and 6 record that
    in their meta.json, so the products say which map they were made with.
    """
    data: np.ndarray
    path: Path


def place_segmentation(seg_src, white, out, max_offset=MAX_GRID_OFFSET,
                       keep_intermediate=True):
    """Read the professor's segmentation and confirm it shares a pixel grid with
    the white light; return it.

    Equal shapes do not prove the same grid, so the check is "where on the sky
    does this pixel point", not a keyword-by-keyword comparison: the seg carries
    a CD matrix while the cube uses PC + CDELT, and their CRPIX differ by 0.01 px,
    both of which a literal comparison would report as a mismatch.

    With keep_intermediate the map is copied next to the white light, which is
    where the evaluation scripts read the segmentation a run used.

    max_offset above the default is a decision to run anyway on a pointing whose
    headers disagree. It comes from that pointing's config and is printed when it
    is above the default, so the bypass is recorded twice -- in the file and in
    the step log -- rather than living in whoever's shell history raised it.
    """
    dst = out / "step01/seg.fits"
    s, hs = fits.getdata(seg_src, header=True)
    if keep_intermediate:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(seg_src, dst)
    w, hw = white.data, white.header
    if s.shape != w.shape:
        raise SystemExit(f"★ seg {s.shape} and white light {w.shape} differ in shape")
    if "CTYPE1" not in hw:
        raise SystemExit("★ the white light carries no WCS -- the cube's DATA "
                         "header has none to copy")

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
    return Seg(s, dst)


def run_pointing(cfg_path):
    """The method, in order: six steps, and the segmentation check between one and two."""
    cfg = load(cfg_path)
    out = cfg["output"]
    inp = cfg["input"]
    basis, src = cfg["sky_line_basis"], cfg["source_fit"]
    amp, spx = cfg["sky_amplitude"], cfg["spaxel_fit"]
    # Named in full because run_step's own `keep` is a different thing:
    # which of a step's output lines reach the terminal.
    keep_intermediate = cfg["keep_intermediate"]

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
    white = run_step("step1", whitelight,
                     dict(cube=inp["nosky"], out=out / "step01",
                          keep_intermediate=keep_intermediate),
                     out / "step1.log")

    print("--- [2/7] the professor's segmentation")
    seg = place_segmentation(inp["seg"], white, out, cfg["max_grid_offset"],
                             keep_intermediate)

    print("--- [3/7] step2 source spectra (nosky, for classification)")
    spectra = run_step("step2", object_spectra,
                       dict(cube=inp["nosky"], white=white, seg=seg,
                            out=out / "step02", keep_intermediate=keep_intermediate),
                       out / "step2.log")

    print("--- [4/7] step3 sky basis")
    sky = run_step("step3", sky_basis,
                   dict(work=out, cube=inp["cube"], white=white, seg=seg,
                        K=basis["K"],
                        methods=[basis["method"]], seed=basis["seed"],
                        continuum_window=basis["continuum_window"],
                        line_thresholds=basis["line_thresholds"],
                        max_iter=basis["max_iter"], clip_sigma=basis["clip_sigma"],
                        keep_intermediate=keep_intermediate,
                        **basis_region),
                   out / "step3.log", keep=KEEP["step3"])

    print("--- [5/7] step4 template fitting and classification")
    # step4's result is the last mask iteration asked for: the classification
    # fields step6 rebuilds the sources from, the galaxy-branch redshifts step5
    # groups the main source by, and the name of the file all of that was
    # written to.
    classified = run_step("step4", classify_sources,
                          dict(work=out, sky=sky, spectra=spectra,
                               K=basis["K"], id="all", basis=basis["method"],
                               fix_s_at=src["fix_s_at"],
                               star_window=src["fit_window"],
                               gal_window=src["fit_window"],
                               line_mask_iter=src["line_mask_iter"],
                               zmin=src["z_min"], zmax=src["z_max"],
                               zstep=src["z_step"], star_dz=src["star_dz"],
                               num_workers=src["num_workers"],
                               keep_intermediate=keep_intermediate),
                          out / "step4.log", tail=3)

    line_iter = src["line_mask_iter"][-1]
    print(f"--- [6/7] step5 build the s field   [mask iter {line_iter}]")
    s_field = run_step("step5", fit_s_field,
                       dict(work=out, cube=inp["cube"], white=white, seg=seg,
                            sky=sky, classification=classified, K=basis["K"],
                            basis=basis["method"],
                            blank_channels=spx["blank_channels"],
                            min_channel_coverage=spx["min_channel_coverage"],
                            min_source_distance=amp["min_source_distance"],
                            min_main_source_distance=amp["min_main_source_distance"],
                            train_clip_sigma=amp["train_clip_sigma"],
                            main_source_dz=amp["main_source_dz"],
                            keep_intermediate=keep_intermediate,
                            **train_region),
                       out / "step5.log", keep=KEEP["step5"])

    print("--- [7/7] step6 final sky subtraction")
    run_step("step6", subtract_sky,
             dict(work=out, cube=inp["cube"], white=white, seg=seg, sky=sky,
                  classification=classified, s_field=s_field, K=basis["K"],
                  basis=basis["method"],
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
