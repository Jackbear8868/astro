"""The sky reconstruction pipeline for one pointing: six steps, one entrance.

    conda run -n astro python src/skymodel/pipeline.py configs/p01.yaml
    conda run -n astro python src/skymodel/pipeline.py configs/p0[1-4].yaml

Pipeline.run() below is the whole method in one place:

    step 1  whitelight        collapse a cube along wavelength into a white light image
    step 2  source_spectra    sum each source's spectrum over the spaxels its seg ID covers
    step 3  sky_basis         learn the sky continuum and the sky-line basis from blank
    step 4  classify_sources  fit templates to every source, giving it a class and a redshift
    step 5  fit_sky_amplitude force the sky continuum amplitude s onto a spatial field
    step 6  subtract_sky      apply the model to every spaxel and write the subtracted cube

Each step is handed what the earlier ones returned rather than reopening files, so no
step can pick up what an earlier run left on disk. The cube is the exception: large and
memmapped, so every step that needs it opens it again. After the worker pool the file
follows run()'s order, each step's section carrying its own helpers and constants.

The products under {output}/stepNN are written unless keep_intermediate is off, step6's
either way, and nothing reads them back. Each step's output goes to {output}/stepN.log,
the config as it was read to {output}/config.json. The white light comes from the nosky
cube: downstream finds the main source by its brightest pixel, which the wsky cube's sky
continuum would make unreliable. Each fitting step holds BLAS at one thread around its
own work (utils.blas_single_thread).
"""
import argparse
import contextlib
import datetime
import hashlib
import inspect
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.interpolate import make_interp_spline
from scipy.optimize import lsq_linear
from sklearn.decomposition import PCA, TruncatedSVD
from threadpoolctl import threadpool_limits

import matplotlib
matplotlib.use("Agg")              # must be set before importing pyplot: render to file, not screen
import matplotlib.pyplot as plt

# ROOT comes from config rather than being resolved again: that module sits here too.
from config import MAX_GRID_OFFSET, ROOT, load
from utils import (C_KMS, DWARF_DIR, EIGEN_GAL, STAR_LIBRARY,
                   air_to_vacuum, blas_single_thread, build_amplitude_field,
                   build_templates, estimate_continuum, fit_blank, fit_source,
                   load_ascii_template, load_eigen_galaxy, load_line_masks,
                   main_source_group, plot_main_group, redshift_to_grid,
                   wavelength_grid)


# =========================================================================
# the worker pool -- the one part of this file that is not in the class
# =========================================================================
#
# classify_sources maps _scan_one over the sources through a Pool, so these six names
# must be reachable from a worker, which under spawn holds only what it was sent. None
# can be a method: a bound method is pickled with the object it is bound to.


# Step 6 reads this width too, for the A_map it writes.
N_COMPONENTS = 4            # fixed width of the A column: 4 eigenspectra, and a star
                            # uses only column 0


_SHARED = {}


def scan_object(flux, var, sky, jobs, lam_muse, fit, fix_s_at=None,
                allow_partial=False):
    """Scan templates and redshifts for one summed spectrum, over the channel set fit.

    fit is a boolean array as long as the spectrum, the window and the sky-line mask
    already combined into it; fitting and scoring share it, which is what makes the chi2
    values comparable. jobs is a list of (group, name, spline, z grid), the grid
    belonging to each candidate: a star needs only Galactic peculiar velocities.

    A and s are held non-negative, neither being physically able to go negative; the
    sky-line coefficients are free, that basis being signed by construction. fix_s_at
    subtracts s*C_sky first and drops s as a free parameter, breaking the degeneracy in
    which the template absorbs the sky continuum.
    """
    base = (fit & np.isfinite(flux) & np.isfinite(var) & (var > 0)
            & np.all(np.isfinite(sky), axis=0))
    sig  = np.sqrt(np.where(var > 0, var, 1.0))
    n_full = int(base.sum())            # the ceiling every candidate shares
    if not n_full:                      # no candidate could clear n > p anyway, and the
        return []                       # two ends below would have nothing to read
    # The two ends every candidate has to reach across, read off the data itself.
    lam_lo  = float(lam_muse[base].min())
    lam_hi  = float(lam_muse[base].max())
    flux_ok = np.isfinite(flux)         # the z-independent half of the src_min channels

    if fix_s_at is None:
        sky_free, y, s_free = sky, flux, True
    else:
        sky_free, y, s_free = sky[1:], flux - fix_s_at * sky[0], False

    skyw = np.ascontiguousarray((sky_free / sig).T)
    yw   = y / sig

    results = []
    for group, name, spline, z_grid in jobs:
        n_comp = 1 if spline.c.ndim == 1 else spline.c.shape[1]
        p      = sky_free.shape[0] + n_comp

        lb = np.full(p, -np.inf)
        if n_comp == 1:
            lb[0] = 0.0                 # a single template is positive everywhere, so
                                        # A >= 0 is "no negative source light"
        if s_free:
            lb[n_comp] = 0.0
        ub = np.full(p, np.inf)
        # With no finite bound the answer is the plain least-squares one.
        has_bounds = np.any(np.isfinite(lb))

        # The spline's own domain: with extrapolate=False it decides where T is NaN.
        lo_rest = float(spline.t[spline.k])
        hi_rest = float(spline.t[-spline.k - 1])

        for z in z_grid:
            T = redshift_to_grid(spline, z, lam_muse)
            if T.ndim == 1:
                T = T[:, None]
            # Past both ends covers everything between: two comparisons, not a scan.
            covers = lo_rest * (1 + z) <= lam_lo and hi_rest * (1 + z) >= lam_hi
            if covers:
                good, n = base, n_full
            else:
                good = base & np.isfinite(T[:, 0])
                n    = int(good.sum())
            if n <= p:                  # leave at least one degree of freedom, or
                                        # reduced chi2 is undefined
                continue
            # A candidate short of the window is dropped: chi2 is a sum, so fewer is less.
            if not allow_partial and not covers:
                continue

            M = np.empty((n, p))
            M[:, :n_comp] = T[good] / sig[good][:, None]
            M[:, n_comp:] = skyw[good]

            if has_bounds:
                fitres = lsq_linear(M, yw[good], bounds=(lb, ub), method="bvls")
                theta  = fitres.x
                chi2   = 2.0 * fitres.cost
            else:
                # lsq_linear's own cutoff and dot product: both branches feed one chi2.
                theta = np.linalg.lstsq(M, yw[good], rcond=-1)[0]
                r     = M @ theta - yw[good]
                chi2  = float(r @ r)

            # Negative source flux is a model problem, so the whole range is checked;
            # a template is NaN a whole channel at a time, so column 0 answers.
            ok  = flux_ok & np.isfinite(T[:, 0])
            src = T @ theta[:n_comp]

            results.append(dict(group=group, template=name, z=float(z),
                    A=theta[:n_comp], s=theta[n_comp] if s_free else fix_s_at,
                    chi2=chi2, red_chi2=chi2 / (n - p),
                    n_good=n, src_min=float(src[ok].min())))

    return sorted(results, key=lambda r: r["chi2"])


def _pack_scan(results):
    """One source's whole scan as a structured array, one row per candidate fit.

    Returns (rows, templates). One array rather than a column per key, so a worker can
    hand a scan back and the parent write it into one file. The template name becomes an
    index into `templates`, names repeated over thousands of rows being most of the
    array otherwise; `group` is one value for the whole file, so the writer keeps it.
    """
    templates = sorted({x["template"] for x in results})
    code = {t: i for i, t in enumerate(templates)}
    rows = np.zeros(len(results), dtype=[
        ("z", "f8"), ("s", "f8"), ("chi2", "f8"), ("red_chi2", "f8"),
        ("n_good", "i8"), ("src_min", "f8"),
        ("A", "f8", N_COMPONENTS), ("template", "i1")])
    for i, x in enumerate(results):
        rows[i]["A"] = np.nan
        rows[i]["A"][:len(x["A"])] = x["A"]
        for k in ("z", "s", "chi2", "red_chi2", "n_good", "src_min"):
            rows[i][k] = x[k]
        rows[i]["template"] = code[x["template"]]
    return rows, templates


def _init_worker(shared):
    """Give a worker process the one name _scan_one reads. Only a forked worker inherits
    the parent's memory, so it arrives through the initializer instead. The thread limit
    is re-applied for the same reason: a fresh interpreter starts at the machine default,
    and a worker per core would then oversubscribe it.
    """
    global _SHARED
    _SHARED = shared
    threadpool_limits(limits=1)


def _scan_one(t):
    """Fit one source in a single stage: stellar templates and galaxy eigenspectra
    compete on the same channels, the shared data coming from _init_worker. The two
    reduced chi2 are comparable because chi2 / (n_good - n_param) accounts for the
    differing degrees of freedom -- but only over the same channels, which is why both
    branches get one window, and what makes an absolute threshold unnecessary.
    """
    S = _SHARED
    k = int(np.flatnonzero(S["seg_ids"] == t)[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        f = S["flux"][k] / S["nspax"][k]
        v = S["var"][k]  / S["nspax"][k] ** 2

    r1 = scan_object(f, v, S["sky"], S["star_jobs"], S["wl_vac"],
                     S["fit_star"], fix_s_at=S["fix_s_at"],
                     allow_partial=S["allow_partial"])
    r2 = scan_object(f, v, S["sky"], S["gal_jobs"], S["wl_vac"],
                     S["fit_gal"], fix_s_at=S["fix_s_at"],
                     allow_partial=S["allow_partial"])
    if not r1 and not r2:
        return t, None, None
    # The packed scans are a product, not the route home; one writer per branch file.
    scans = None
    if S["keep_scans"]:
        scans = (_pack_scan(r1) if r1 else None, _pack_scan(r2) if r2 else None)

    # The two branch winners face each other; the scans come back sorted by chi2.
    best = min([x[0] for x in (r1, r2) if x], key=lambda d: d["red_chi2"])

    A = np.full(N_COMPONENTS, np.nan)
    A[:len(best["A"])] = best["A"]
    # Both values are kept for the margin; gal_z is the galaxy branch's own, not "z".
    return t, scans, dict(id=t, nspax=int(np.median(S["nspax"][k])),
                   star_red_chi2=r1[0]["red_chi2"] if r1 else np.nan,
                   star_tpl=r1[0]["template"] if r1 else "",
                   gal_red_chi2=r2[0]["red_chi2"] if r2 else np.nan,
                   gal_tpl=r2[0]["template"] if r2 else "",
                   gal_z=(float(r2[int(np.argmin([x["red_chi2"] for x in r2]))]["z"])
                          if r2 else None),
                   **{**best, "A": A})


# =========================================================================
# what one step hands the next
# =========================================================================
#
# These are the arguments of run(): a step's signature names the ones it takes.


class WhiteLight(NamedTuple):
    """What step1 hands the ones after it. The header travels with the image for the
    segmentation check; every other consumer reads only `data`.
    """
    data: np.ndarray          # (ny, nx), the collapsed image, 0 outside the field
    header: fits.Header       # the cube's celestial WCS


class Seg(NamedTuple):
    """The segmentation, as steps 2, 3, 5 and 6 are handed it. path is where it was put
    next to the white light, and steps 5 and 6 record it in their meta.json.
    """
    data: np.ndarray
    path: Path


class SourceSpectra(NamedTuple):
    """What step2 hands step4: one summed spectrum per source. `path` is where the
    arrays were written, carried so step4 can record which spectra it classified.
    """
    ids: np.ndarray           # (n_ids,)   segmentation IDs, ascending
    flux: np.ndarray          # (n_ids, nz)
    var: np.ndarray           # (n_ids, nz)
    nspax: np.ndarray         # (n_ids, nz)
    path: Path


class SkyModel(NamedTuple):
    """What step3 hands steps 4, 5 and 6 -- everything they read of the sky. basis is
    keyed by decomposition method; iter_line_mask is the whole per-iteration stack, of
    which step4 fits one iteration per pass and steps 5 and 6 take the first.
    """
    wavelength: np.ndarray        # (nz,)          air wavelength of each channel
    continuum: np.ndarray         # (nz,)          C_sky
    basis: dict                   # method -> (K, nz) sky-line basis
    iter_line_mask: np.ndarray    # (n_iter, nz)   bool, one row per iteration


class Classification(NamedTuple):
    """What step4 hands steps 5 and 6. data holds the fields of classification.npz, which
    step6 rebuilds each source's model from, and path names the product they came from,
    recorded in steps 5 and 6's meta.json. galaxy_z is the galaxy branch's best redshift,
    which step5 groups the main source by; it is not data["z"], the winning branch's.
    """
    path: Path
    data: dict                # field name -> array, as written to the npz
    galaxy_z: dict            # seg ID -> galaxy-branch redshift


class SkyAmplitude(NamedTuple):
    """What step5 hands step6: the field, and where it was written. data is the float32
    the file holds, not the float64 the fit produced, so the two hold the same field.
    """
    data: np.ndarray          # (ny, nx) float32
    path: Path                # step05/sky_continuum_amplitude_field.npy


# =========================================================================
# the log a step's output goes to
# =========================================================================

class StepLog:
    """Collect a step's stdout: everything to the log, some of it to the terminal.
    Line-buffered because print() writes text and newline separately and a
    TERMINAL_LINES pattern matches whole lines. `tail` instead holds back the last few
    non-matching lines, for a step whose summary is at the end.
    """

    def __init__(self, log, echo=None, tail=0):
        self.log, self.echo, self.tail = log, echo, tail
        self.buf, self.held = "", []
        # The real terminal, captured before redirect_stdout takes this object's place.
        self.term = sys.stdout

    def _echo(self, line):
        self.term.write("    " + line.rstrip() + "\n")
        self.term.flush()

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.log.write(line + "\n")
            if self.echo and re.search(self.echo, line):
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


# =========================================================================
# the pipeline -- the six steps in the order they happen
# =========================================================================

class Pipeline:
    """One pointing, from its config file to its sky-subtracted cube.

        Pipeline("configs/p01.yaml").run()

    The config is read once, in __init__; every step reaches its values through self.
    What one step hands the next stays an argument, so a signature lists the earlier
    steps it consumes and the products stay locals of run(): main() runs several configs
    in turn, and a product left on the object would await the next.
    """

    # =========================================================================
    # the constants the steps read
    # =========================================================================
    #
    # Reached through self, or through Pipeline in a staticmethod: a class body's names
    # go out of scope when that body finishes.

    # Which lines of a step reach the terminal while it runs; the rest is in the log.
    TERMINAL_LINES = {
        # The grid offset decides whether the pointing may run at all.
        "step1": r"sources, mask|grid offset",
        # What the basis was built from -- each of these changes what the basis is.
        "step3": (r"spatial restriction|exclude-box|blank spaxels|borrowed"
                  r"|mask_source_lines|select_faintest|flux window|svd |pca "),
        # The one number showing the field was estimated: NaN everywhere is silent.
        "step5": r"main source|s spatial field|s_hat median|saved",
        "step6": r"blank|source |saved",
    }

    # An upper bound past the edge of the field, compared against pixel indices.
    BEYOND_EDGE = 9999

    # How much of one argument the log prints before it says what the value is instead.
    ARG_WIDTH = 160

    # Defined above learn_sky_basis: a default argument is evaluated as this body runs.
    SEED       = 0           # shared by all decompositions, so a basis is reproducible

    # The whole MUSE range: full_range widens both branches to it, as a control.
    FULL_RANGE  = (4600.0, 9400.0)

    def __init__(self, cfg_path):
        self.cfg_path = Path(cfg_path)
        self.cfg = load(cfg_path)
        self.out = self.cfg["output"]
        self.inp = self.cfg["input"]
        self.sky_line_basis = self.cfg["sky_line_basis"]
        self.source_fit = self.cfg["source_fit"]
        self.sky_amplitude = self.cfg["sky_amplitude"]
        self.spaxel_fit = self.cfg["spaxel_fit"]
        # Named in full: run_step's own `echo` is which output lines reach the terminal.
        self.keep_intermediate = self.cfg["keep_intermediate"]

        # The config's one box, as the xlim / ylim / exclude_box the steps read.
        reg = self.cfg["sky_region"]
        self.basis_region = self.region_kwargs(reg) if "basis" in reg["apply_to"] else {}
        self.train_region = (self.region_kwargs(reg, "train_")
                             if "sky_amplitude" in reg["apply_to"] else {})

    def run(self):
        """The method, in order: six steps, and the segmentation check between one and two."""
        for key, path in self.inp.items():
            if not path.exists():
                raise SystemExit(f"★ input.{key} not found: {path}")
        self.out.mkdir(parents=True, exist_ok=True)
        self.record_config()

        reg = self.cfg["sky_region"]
        print("=" * 70)
        print(f"  pointing #{self.cfg['pointing']}  ->  {self._repo_path(self.out)}"
              f"   [{self.cfg_path.name}]")
        print(f"  sky region {reg['x']} x {reg['y']}  "
              f"{'include' if reg['include'] else 'exclude'} -> {reg['apply_to']}")
        print("=" * 70)
        t0 = time.time()

        # Part of step 1: checked against the white light and written beside it.
        print("--- [1/6] step1 white light (from the nosky cube), and the segmentation")
        white = self.run_step("step1", self.whitelight, {})
        # Same log as the white light, appended: the grid offset is worth the record.
        seg = self.run_step("step1", self.place_segmentation, dict(white=white),
                            echo=self.TERMINAL_LINES["step1"], append=True)

        print("--- [2/6] step2 source spectra (nosky, for classification)")
        spectra = self.run_step("step2", self.source_spectra,
                                dict(white=white, seg=seg))

        print("--- [3/6] step3 sky basis")
        sky = self.run_step("step3", self.sky_basis, dict(white=white, seg=seg),
                            echo=self.TERMINAL_LINES["step3"])

        print("--- [4/6] step4 template fitting and classification")
        # step4's result is the last mask iteration asked for (see Classification).
        classified = self.run_step("step4", self.classify_sources,
                                   dict(sky=sky, spectra=spectra), tail=3)

        line_iter = self.source_fit["line_mask_iter"][-1]
        print(f"--- [5/6] step5 build the s field   [mask iter {line_iter}]")
        s_field = self.run_step("step5", self.fit_sky_amplitude,
                                dict(white=white, seg=seg, sky=sky,
                                     classification=classified),
                                echo=self.TERMINAL_LINES["step5"])

        print("--- [6/6] step6 final sky subtraction")
        self.run_step("step6", self.subtract_sky,
                      dict(white=white, seg=seg, sky=sky,
                           classification=classified, s_field=s_field),
                      echo=self.TERMINAL_LINES["step6"])

        free = shutil.disk_usage(ROOT).free / 1024 ** 3
        print(f"*** pointing #{self.cfg['pointing']} done in {time.time() - t0:.0f} s"
              f"   {free:.0f} GB free")

    def record_config(self):
        """Write the config this run used into the output directory: a step log names the
        products a step was handed, not the values behind them. It is the config as
        load() returned it, not a copy of the file, which can be edited afterwards.
        """
        def plain(v):
            """The config as JSON takes it: paths shortened against the root."""
            if isinstance(v, Path):
                return str(self._repo_path(v))
            if isinstance(v, dict):
                return {k: plain(x) for k, x in v.items()}
            if isinstance(v, list):
                return [plain(x) for x in v]
            return v

        (self.out / "config.json").write_text(
            json.dumps(plain(self.cfg), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")

    def run_step(self, label, fn, kwargs, echo=None, tail=0, append=False):
        """Call one step in this process, sending its output to {output}/{label}.log, and
        pass back whatever it returns -- that is how one step's results reach the next.
        append adds to the log rather than starting it, for the second call sharing a
        step's label, which would otherwise truncate the first call's output.
        """
        log_path = self.out / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a" if append else "w", encoding="utf-8") as log:
            log.write(self.call_repr(fn, kwargs) + "\n\n")
            log.flush()
            step_log = StepLog(log, echo, tail)
            try:
                with contextlib.redirect_stdout(step_log):
                    result = fn(**kwargs)
            except BaseException:
                # The traceback goes to the log, which would otherwise end mid-step.
                step_log.close()
                log.write("\n" + traceback.format_exc())
                print(f"★ {label} failed; full output in {log_path}", flush=True)
                raise
            step_log.close()
        return result

    # =========================================================================
    # what the class runs the steps through: the config translation, and the
    # head of every step's log
    # =========================================================================

    def region_kwargs(self, reg, prefix=""):
        """sky_region -> the xlim / ylim / exclude_box that step3 and step5 read.

        prefix is "train_" for step5, where the range restricts the spaxels that train the
        s field, not the ones the sky is learned from. Config ranges are half-open with
        null for "no bound", as are xlim/ylim; exclude_box includes both its endpoints.
        """
        x, y = reg["x"], reg["y"]
        lo = lambda v: 0 if v is None else v

        if reg["include"]:
            kw = {}
            if x != [None, None]:
                kw[f"{prefix}xlim"] = [lo(x[0]),
                                       self.BEYOND_EDGE if x[1] is None else x[1]]
            if y != [None, None]:
                kw[f"{prefix}ylim"] = [lo(y[0]),
                                       self.BEYOND_EDGE if y[1] is None else y[1]]
            return kw

        return {f"{prefix}exclude_box": [
            lo(y[0]), self.BEYOND_EDGE if y[1] is None else y[1] - 1,
            lo(x[0]), self.BEYOND_EDGE if x[1] is None else x[1] - 1]}

    @staticmethod
    def _shorten(text, value):
        """text if it is short enough to read on one line, else what the value is."""
        if len(text) <= Pipeline.ARG_WIDTH and "\n" not in text:
            return text
        size = f" of {len(value)}" if hasattr(value, "__len__") else ""
        return f"<{type(value).__name__}{size}>"

    @staticmethod
    def _render(v):
        """One argument of a step call, written for the head of its log. An array becomes
        its shape and dtype, a step's bundle is opened up so the paths and tags inside
        stay visible, and a path is shortened against the repository root.
        """
        if isinstance(v, np.ndarray):
            return f"<ndarray {v.shape} {v.dtype}>"
        if isinstance(v, Path):
            try:
                return repr(str(v.resolve().relative_to(ROOT)))
            except ValueError:
                return repr(str(v))
        if isinstance(v, tuple) and hasattr(v, "_fields"):          # a step's bundle
            inner = ", ".join(f"{f}={Pipeline._render(x)}" for f, x in zip(v._fields, v))
            return f"{type(v).__name__}({inner})"
        if isinstance(v, dict):
            return Pipeline._shorten("{" + ", ".join(f"{Pipeline._render(k)}: {Pipeline._render(x)}"
                                                     for k, x in v.items()) + "}", v)
        if isinstance(v, (list, tuple)):
            body = ", ".join(Pipeline._render(x) for x in v)
            if isinstance(v, tuple):
                body = f"({body},)" if len(v) == 1 else f"({body})"
            else:
                body = f"[{body}]"
            return Pipeline._shorten(body, v)
        return Pipeline._shorten(repr(v), v)

    @staticmethod
    def call_repr(fn, kwargs):
        """The step call written out as Python, for the head of its log. It records the
        products this step was handed; the config values are in record_config's file.
        """
        args = ", ".join(f"{k}={Pipeline._render(v)}" for k, v in kwargs.items())
        return f"{fn.__module__}.{fn.__qualname__}({args})"

    @staticmethod
    def _npy_bytes(a):
        """One array as the bytes of a .npy file. np.savez builds a zip of .npy members;
        writing them one at a time gives the same file without holding every array at
        once, so a scan can join as its worker finishes.
        """
        buf = io.BytesIO()
        np.save(buf, a, allow_pickle=False)
        return buf.getvalue()

    @staticmethod
    def write_meta(out, **fields):
        """Write out/meta.json: what the step was given, plus who wrote it and when. The
        step's name is read off the calling method, so renaming the method carries the
        field with it. Nothing reads it, but a record that disagrees with the code is
        worse than no record.
        """
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, cwd=ROOT)
        meta = dict(step=inspect.currentframe().f_back.f_code.co_name,
                    created=datetime.datetime.now().isoformat(timespec="seconds"),
                    git_commit=head.stdout.strip(), **fields)
        (out / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"meta -> {out / 'meta.json'}")

    # A path against the repository root, so a record does not depend on the cwd.
    @staticmethod
    def _repo_path(p):
        p = Path(p)
        try:
            return p.resolve().relative_to(ROOT)
        except ValueError:
            return p

    # =========================================================================
    # step 1 -- white light
    # =========================================================================
    #
    # Everything downstream asking "where is the source" works on this image.

    def whitelight(self, rows=32):
        """Collapse `cube` along wavelength; return the image and its WCS. With
        keep_intermediate it also goes to `out` as whitelight_nosky.fits plus a preview
        png; the filename names the cube because the pointing has two and an evaluation
        script collapses the other. rows is how many image rows are collapsed at a time.
        """
        cube = self.inp["nosky"]
        out = self.out / "step01"
        keep_intermediate = self.keep_intermediate

        cube, out = Path(cube), Path(out)
        if keep_intermediate:
            out.mkdir(parents=True, exist_ok=True)
        white_fits = out / "whitelight_nosky.fits"
        white_png = out / "whitelight_preview.png"

        with fits.open(cube, memmap=True) as hdul:
            data = hdul["DATA"].data
            # A band of rows at a time, so nanmean copies one band and not the cube. The
            # split must stay spatial: along wavelength it changes the summation order.
            white = np.concatenate([np.nanmean(data[:, y:y + rows, :], axis=0)
                                    for y in range(0, data.shape[1], rows)])
            white = np.nan_to_num(white, nan=0.0)
            # The cube's celestial WCS; without it the check could compare only shapes.
            hdr = WCS(hdul["DATA"].header).celestial.to_header()

        if keep_intermediate:
            fits.writeto(white_fits, white, hdr, overwrite=True)

            fig = plt.figure(figsize=(6, 6))
            plt.imshow(white, origin="lower", cmap="gray",
                       vmin=np.nanpercentile(white, 5),
                       vmax=np.nanpercentile(white, 99))
            plt.colorbar()
            fig.savefig(white_png, dpi=130)
            # Closed explicitly: this runs in-process, so open figures accumulate.
            plt.close(fig)
            print(f"saved -> {white_fits}")

        print(f"white light {white.shape} {white.dtype}")
        return WhiteLight(white, hdr)

    # =========================================================================
    # the segmentation check, between step 1 and step 2
    # =========================================================================

    def place_segmentation(self, white):
        """Read the segmentation this pointing was given, check it shares a pixel grid
        with the white light, and return it.

        The pipeline does not detect sources: which spaxels hold one is an input named by
        the config. Equal shapes do not prove the same grid, so the check asks where on
        the sky each pixel points rather than comparing keywords -- the seg carries a CD
        matrix while the cube uses PC + CDELT, and their CRPIX differ.

        With keep_intermediate the map is copied next to the white light, at a fixed path
        for the evaluation scripts. The copy is byte-identical and so cannot show that it
        was checked; meta.json beside it carries the offset and the source's checksum.
        max_offset above the default runs a pointing whose headers disagree, and prints.
        """
        seg_src = self.inp["seg"]
        out = self.out
        max_offset = self.cfg["max_grid_offset"]
        keep_intermediate = self.keep_intermediate

        dst = out / "step01/segmentation_input.fits"
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
        if keep_intermediate:
            self.write_meta(
                out / "step01",
                cube=str(self._repo_path(self.inp["nosky"])),
                seg_source=str(self._repo_path(seg_src)),
                # The source can be replaced; the digest still identifies what was read.
                seg_md5=hashlib.md5(Path(seg_src).read_bytes()).hexdigest(),
                grid_offset_px=round(float(off), 4),
                max_grid_offset=max_offset,
                n_sources=int(len(np.unique(s)) - 1),
                mask_fraction=round(float((s > 0).mean()), 4))
        return Seg(s, dst)

    # =========================================================================
    # step 2 -- source spectra
    # =========================================================================
    #
    # One summed spectrum per source, with its variance and valid spaxel count per
    # channel. They come from a sky-subtracted cube: a spectrum still holding the sky
    # fits normally, with every template and redshift wrong.

    @staticmethod
    def sum_spectra_by_id(cube_path, seg, ids, chunk=200, var_path=None):
        """Sum the spectra of all spaxels belonging to the same segmentation ID. All
        three outputs count the same spaxels, the ok mask zeroing unusable positions
        first; np.nansum per array would let each skip different ones.

        Parameters
        ----------
        cube_path : path-like
            MUSE cube; requires a DATA extension.
        var_path : path-like or None
            Where STAT is read from; defaults to cube_path. A sky-subtracted cube has
            only DATA, and subtracting a deterministic model leaves the pixel variance
            alone, so the original cube's STAT is correct.
        seg : ndarray, shape (ny, nx)
            Segmentation map, 0 meaning no source. Pixels outside the field of view must
            be set to 0 before calling or they enter the summation.
        ids : ndarray, shape (n_ids,)
        chunk : int
            Wavelength planes read at once; memory and speed only.

        Returns
        -------
        flux, var, nspax : ndarray, shape (n_ids, nz)
            The summed spectra, the summed variance (variance is what adds for
            independent pixels, not sigma), and the count of valid spaxels per ID per
            channel. nspax varies with wavelength, so a sum becomes a mean by dividing
            by it and never by the total: mean_var = var / nspax**2.
        """
        seg_flat = seg.ravel()
        members  = [np.flatnonzero(seg_flat == i) for i in ids]

        with fits.open(cube_path, memmap=True) as hdul, \
             fits.open(var_path or cube_path, memmap=True) as vdul:
            nz   = hdul["DATA"].header["NAXIS3"]
            flux = np.zeros((len(ids), nz))
            var  = np.zeros((len(ids), nz))
            nspax = np.zeros((len(ids), nz))

            for j in range(0, nz, chunk):
                # Left at the cube's float32: the widening happens per source below.
                d = np.asarray(hdul["DATA"].data[j:j+chunk], np.float32).reshape(-1, seg_flat.size)
                v = np.asarray(vdul["STAT"].data[j:j+chunk], np.float32).reshape(-1, seg_flat.size)

                ok = np.isfinite(d) & np.isfinite(v) & (v > 0)
                d  = np.where(ok, d, 0.0)
                v  = np.where(ok, v, 0.0)

                for k, idx in enumerate(members):
                    # Widened first: float32 loses digits over thousands of spaxels.
                    flux[k,  j:j+chunk] = d[:, idx].astype(np.float64).sum(axis=1)
                    var[k,   j:j+chunk] = v[:, idx].astype(np.float64).sum(axis=1)
                    nspax[k, j:j+chunk] = ok[:, idx].sum(axis=1)

        return flux, var, nspax

    def source_spectra(self, white, seg, var_cube=None, top=20):
        """Sum every source's spectrum over the spaxels its segmentation ID covers. white
        and seg come from step1 and the segmentation check, in memory; with
        keep_intermediate the summed arrays go into `out` as one npz too. top is how many
        rows of the SNR table are printed, which is there to notice a far weaker source.
        """
        cube = self.inp["nosky"]
        out = self.out / "step02"
        keep_intermediate = self.keep_intermediate

        out = Path(out)
        if keep_intermediate:
            out.mkdir(parents=True, exist_ok=True)
        print(f"spectra -> {out}   cube {Path(cube).name}")

        white, seg = white.data, seg.data

        valid_mask  = white != 0
        source_mask = (seg > 0) & valid_mask
        seg_valid   = np.where(valid_mask, seg, 0)      # outside FoV -> 0, excluded from sum

        ids, counts = np.unique(seg_valid[source_mask], return_counts=True)
        print(f"{len(ids)} sources, {counts.sum()} source spaxels")

        print(f"DATA <- {Path(cube).name}   STAT <- {Path(var_cube or cube).name}")
        flux, var, nspax = self.sum_spectra_by_id(cube, seg_valid, ids, var_path=var_cube)

        with np.errstate(invalid="ignore", divide="ignore"):
            snr = np.nanmedian(flux / np.sqrt(var), axis=1)

        order = np.argsort(snr)[::-1]
        print(f"{'ID':>5} {'N':>7} {'sqrt(N)':>9} {'median SNR':>12}")
        for k in order[:top]:
            print(f"{ids[k]:>5d} {counts[k]:>7d} {np.sqrt(counts[k]):>9.1f} {snr[k]:>12.2f}")

        if keep_intermediate:
            # One bundle: ids is the row order, flux needs spaxel_count to be a mean.
            np.savez(out / "source_spectra.npz",
                     ids=ids, flux_sum=flux, variance_sum=var, spaxel_count=nspax,
                     wavelength=wavelength_grid(fits.getheader(cube, "DATA")))
            print("saved ->", out / "source_spectra.npz")
        return SourceSpectra(ids, flux, var, nspax, out)

    # =========================================================================
    # step 3 -- the sky model
    # =========================================================================
    #
    # The two things learned from blank spaxels: the sky continuum C_sky and K sky-line
    # basis vectors, by an interchangeable decomposition.

    @staticmethod
    def learn_sky_basis(residual, K=10, method="pca", seed=SEED, chunk=200):
        """Learn K sky-line basis vectors from the residuals of blank spaxels. Every
        method returns shape (K, nz), so the design matrix always has K free parameters
        and chi2 from different methods stays comparable.

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
        basis : ndarray, shape (K, nz), in the downstream coefficient order.
        """
        # A block at a time, so only a block is float64. nan_to_num comes before the
        # cast: narrowing an infinity gives a finite number that gets fitted.
        X = np.empty((residual.shape[1], residual.shape[0]), np.float32)
        for i in range(0, X.shape[0], chunk):
            X[i:i+chunk] = np.nan_to_num(residual.T[i:i+chunk])

        # random_state is essential: PCA and TruncatedSVD default to randomized SVD.
        if method == "pca":
            p = PCA(n_components=K - 1, random_state=seed).fit(X)
            return np.vstack([p.mean_[None, :], p.components_])

        if method == "svd":
            return TruncatedSVD(n_components=K, random_state=seed).fit(X).components_

        raise ValueError(f"unknown method: {method}")

    @staticmethod
    def borrow_line_basis(run_dir, method, K, wl):
        """Take another finished run's sky-line basis and put it on this grid. Returns
        (basis, record): the (K, nz) basis on the wavelength grid `wl`, and what meta.json
        records about where it came from. Only the line basis is taken -- the continuum,
        the mean spectrum and the line masks stay this pointing's own.

        Parameters
        ----------
        run_dir : path-like
            Another pointing's output directory, as sky_line_basis.borrow_from names it;
            the basis and its grid come from that run's step03.
        method, K : str, int
            This pointing's decomposition and basis width, naming the file to read.
        wl : ndarray, shape (nz,)
            This pointing's grid, the one the basis has to come out on.
        """
        src_dir   = Path(run_dir) / "step03"
        src_basis = src_dir / f"sky_line_basis_{method}_K{K}.npy"
        src_wl    = src_dir / "wavelength.npy"
        for f in (src_basis, src_wl):
            if not f.exists():
                raise SystemExit(
                    f"★ sky_line_basis.borrow_from: {f} does not exist. The run "
                    f"borrowed from has to have finished step3 with the same "
                    f"decomposition and width, here {method} K={K}")

        B      = np.load(src_basis)
        wl_src = np.load(src_wl)
        if B.shape != (K, wl_src.size):
            raise SystemExit(f"★ {src_basis} is {B.shape}, not ({K}, {wl_src.size}); "
                             "it does not belong to the wavelength grid beside it")

        # The grids differ by a fraction of a channel; resampling cannot extrapolate.
        if wl[0] < wl_src[0] or wl[-1] > wl_src[-1]:
            raise SystemExit(
                f"★ sky_line_basis.borrow_from: {Pipeline._repo_path(run_dir)} learned "
                f"its basis on {wl_src[0]:.4f}-{wl_src[-1]:.4f} A and this pointing "
                f"needs {wl[0]:.4f}-{wl[-1]:.4f} A. The basis is a set of samples and "
                "not a formula, so the part outside cannot be extrapolated")

        # A cubic spline: a two-channel-wide line is flattened by linear interpolation.
        out = make_interp_spline(wl_src, np.asarray(B, np.float64), k=3, axis=1)(wl)

        # No linear map keeps a basis orthonormal; the QR restores it on the same span.
        before = np.abs(out @ out.T - np.eye(K)).max()
        q, r   = np.linalg.qr(out.T)
        # QR fixes each vector's length, not its sign; a positive R diagonal keeps it.
        out    = np.ascontiguousarray((q * np.sign(np.diag(r))).T)
        # Narrowed to what a learned basis is, so a borrowed one is the same product.
        out    = out.astype(np.float32)
        # Measured in float64: the number is about the vectors, not about the sums.
        wide   = out.astype(np.float64)
        after  = np.abs(wide @ wide.T - np.eye(K)).max()

        record = dict(
            run=str(Pipeline._repo_path(run_dir)),
            basis_file=str(Pipeline._repo_path(src_basis)),
            # The source file can be overwritten; the digest identifies what was read.
            basis_md5=hashlib.md5(src_basis.read_bytes()).hexdigest(),
            source_wavelength=[float(wl_src[0]), float(wl_src[-1])],
            target_wavelength=[float(wl[0]), float(wl[-1])],
            # The grids share a channel width, so one number says how far samples moved.
            channel_offset=float((wl[0] - wl_src[0]) / (wl_src[1] - wl_src[0])),
            orthonormality_before=float(before),
            orthonormality_after=float(after))

        print(f"borrowed basis {out.shape} <- {record['basis_file']}")
        print(f"  {wl_src[0]:.4f}-{wl_src[-1]:.4f} A resampled onto "
              f"{wl[0]:.4f}-{wl[-1]:.4f} A, offset {record['channel_offset']:.4f} channels")
        print(f"  max|B B^T - I| {before:.3e} -> {after:.3e} after re-orthonormalising")
        return out, record

    @staticmethod
    def exclude_source_lines(residual, wl, windows):
        """Blank the source's own emission-line channels in the decomposition input.

        Returns (mask, record): the (nz,) boolean of the excluded channels, and what
        meta.json records about the windows. Where the source fills the field, every
        "blank" spaxel still carries its emission lines, so the decomposition learns
        them with the sky and step 6 subtracts them from the source too. Only the
        decomposition input is touched: the mean spectrum, the continuum and the masks
        are built before this, and a basis flat at a wavelength cannot put flux there,
        so steps 5 and 6 need no exclusion of their own. Real sky inside a window goes
        with it, so the windows are as narrow as the line.

        The channels go to 0, not the typical residual the clip's rejects get: the fit is
        uncentred, so a column holding one non-zero number everywhere is still a direction
        worth a vector, and that constant is the line's own height. A zero column
        contributes nothing to X^T X and leaves PCA's mean row 0 there.

        Parameters
        ----------
        residual : ndarray, shape (nz, n_blank)
            The decomposition input, modified in place.
        wl : ndarray, shape (nz,)
            The wavelength grid, Angstrom.
        windows : list of [low, high]
            sky_line_basis.mask_source_lines, in observed Angstrom. The config
            normalises the forms it accepts to this one, so a window carries no
            redshift of its own.
        """
        mask = np.zeros(wl.size, bool)
        listed = []
        for lo, hi in windows:
            # Closed at both ends: a channel landing on an edge is inside the range.
            m = (wl >= lo) & (wl <= hi)
            mask |= m
            # Per window, so one that fell off the grid shows as 0 channels.
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

    @staticmethod
    def select_faintest_spaxels(field_mean, valid, column_mean, spec):
        """Keep only the spaxels the ESO sky rule would have called sky.

        Returns (keep, record): a boolean over the columns `column_mean` indexes, and
        what meta.json records about the cut. The ESO pipeline chooses its sky spaxels
        without a segmentation: it ranks the field by flux, drops the faintest `ignore`
        fraction as the dead and half-covered spaxels, and takes the sky from the next
        `fraction` above them. Both ends are percentiles of the whole valid field, which
        is what ESO ranks, not of the blank set -- see the config README. The window is
        half-open, (low, high], so the two ends cannot claim the same spaxel, and what it
        can select is whatever the caller's own cuts have already left.

        Parameters
        ----------
        field_mean : ndarray, shape (ny, nx)
            Per-spaxel mean over wavelength of the sky-included cube, NaN where the
            spectrum is incomplete. The flux the ranking is on.
        valid : ndarray of bool, shape (ny, nx)
            The field that is ranked, before any of the caller's restrictions.
        column_mean : ndarray, shape (n_col,)
            field_mean at the spaxels the caller is holding, in their column order.
        spec : dict
            sky_line_basis.select_faintest: ignore and fraction.
        """
        ign  = float(spec["ignore"])
        frac = float(spec["fraction"])
        # A spaxel with no finite mean cannot be ranked: its mean covers other channels.
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
        print(f"select_faintest: ranked {rank.size:,} spaxels of the valid field by "
              f"mean flux, dropped the faintest {100 * ign:g}% and kept the next "
              f"{100 * frac:g}%")
        print(f"  flux window ({lo:.4f}, {hi:.4f}] holds {in_field:,} of the field; "
              f"{int(keep.sum()):,} of the {column_mean.size:,} blank spaxels offered "
              f"are inside it ({100 * keep.mean():.1f}%)")
        return keep, record

    @blas_single_thread
    def sky_basis(self, white, seg):
        """Learn the sky continuum, the line mask and the sky-line basis; return them.

        white and seg come from step1 and the segmentation check, in memory. With
        keep_intermediate everything learned here goes into step03 too, with a meta.json
        recording which spatial range it came from. Steps 3, 4 and 6 read K and the
        decomposition method from the one config section, so they cannot come apart.

        Three config keys change what is learned. borrow_from takes the basis from the run
        it names (see borrow_line_basis) and mask_source_lines drops the source's own
        emission-line channels from the input (see exclude_source_lines); both touch the
        basis alone. select_faintest narrows the blank spaxels to a flux window (see
        select_faintest_spaxels), which moves every product of this step. Steps 5 and 6
        are untouched by all three.
        """
        b = self.sky_line_basis
        work = self.out
        cube = self.inp["cube"]
        K = b["K"]
        methods = [b["method"]]
        seed = b["seed"]
        continuum_window = b["continuum_window"]
        line_thresholds = b["line_thresholds"]
        max_iter = b["max_iter"]
        # A clip on mean_sky in sg units: bad pixels only, far above real variation.
        clip_sigma = b["clip_sigma"]
        min_unmasked_frac = b["min_unmasked_frac"]
        # None unless the config names a run to borrow the line basis from.
        borrow_from = b.get("borrow_from")
        # None unless the config names source emission-line windows to keep out.
        mask_source_lines = b.get("mask_source_lines")
        # None unless a flux window is named; this one moves every product of this step.
        select_faintest = b.get("select_faintest")
        # The spatial restriction, empty unless sky_region applies to the basis.
        xlim = self.basis_region.get("xlim")
        ylim = self.basis_region.get("ylim")
        exclude_box = self.basis_region.get("exclude_box")
        keep_intermediate = self.keep_intermediate

        work   = Path(work)
        out_dir = work / "step03"
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
            # The grid later steps read back from wavelength.npy, from the shared rule.
            wl  = wavelength_grid(hdr)

            # The flux the field is ranked by, summed in the pass reading blank columns.
            field_sum = (np.zeros(seg.shape, np.float64)
                         if select_faintest is not None else None)
            blank = np.empty((nz, int(blank_mask.sum())), np.float32)
            for j in range(0, nz, 200):
                d = np.asarray(hdul["DATA"].data[j:j+200], np.float32)
                blank[j:j+200] = d[:, blank_mask]
                if field_sum is not None:
                    field_sum += d.sum(axis=0, dtype=np.float64)

        # Spectrally complete spaxels only: nan_to_num would fabricate 0s to fit.
        complete = np.isfinite(blank).all(axis=0)
        print(f"spectrally complete {int(complete.sum()):,} / {blank.shape[1]:,} "
              f"({100*complete.mean():.1f}%), remainder are partially covered spaxels at field edges, excluded")
        blank = blank[:, complete]

        # Before anything is measured: every product below comes from these spaxels.
        faintest = None
        if select_faintest is not None:
            field_mean = field_sum / nz
            keep_col, faintest = self.select_faintest_spaxels(
                field_mean, valid_mask, field_mean[blank_mask][complete],
                select_faintest)
            if not keep_col.any():
                raise SystemExit(
                    "★ sky_line_basis.select_faintest left no spaxel to learn the sky "
                    f"from: the flux window ({faintest['flux_low']:.4f}, "
                    f"{faintest['flux_high']:.4f}] holds {faintest['n_window_in_field']:,} "
                    f"spaxels of the field and none of them is among the "
                    f"{int(complete.sum()):,} this step was given")
            blank = blank[:, keep_col]

        # Sigma-clip per channel before averaging: the mean's breakdown point is 0%, so
        # a few extreme negatives pull a channel's mean down and estimate_continuum then
        # masks it as a "negative line". The clip runs across spaxels within a channel,
        # never along wavelength, so a sky line, bright everywhere, survives it. The mean
        # still ends it: bright-line channels are right-skewed, which biases a median.
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

        # Per-iteration results; utils.load_line_masks is what makes them cumulative.
        iter_line_mask = np.array([h[2] for h in history])

        if keep_intermediate:
            np.save(out_dir / "wavelength.npy",         wl)
            np.save(out_dir / "blank_mean_spectrum.npy", mean_sky)
            np.save(out_dir / "sky_continuum.npy",      C_sky)
            # The continuum keeps its own name because it is C_sky, the product; the
            # threshold and mask are only the last row of the bundle below.
            np.savez(out_dir / "continuum_iterations.npz",
                     continuum=np.array([h[0] for h in history]),
                     threshold=np.array([h[1] for h in history]),
                     line_mask=iter_line_mask)

        # The same keep mask applies: subtracting C_sky shifts x and its median alike.
        # Rejected positions get med - C_sky and not 0, which would deny the line.
        residual = blank - C_sky[:, None]
        np.copyto(residual, (med - C_sky)[:, None], where=~keep)

        # Last, so what it blanks stays blanked: the only step here about the source.
        source_lines = None
        if mask_source_lines is not None:
            _, source_lines = self.exclude_source_lines(residual, wl, mask_source_lines)

        bases, borrowed = {}, {}
        for method in methods:
            t0 = time.time()
            if borrow_from is None:
                basis = self.learn_sky_basis(residual, K=K, method=method, seed=seed)
            else:
                basis, borrowed[method] = self.borrow_line_basis(
                    borrow_from, method, K, wl)
            bases[method] = basis
            if keep_intermediate:
                # "line" because the continuum was subtracted first; K is in the name.
                np.save(out_dir / f"sky_line_basis_{method}_K{K}.npy", basis)
            print(f"{method:13s} basis {basis.shape}  {time.time() - t0:6.1f}s", flush=True)

        # Provenance: only method and K reach the filename, so a re-run overwrites.
        def rel(q):
            q = Path(q)
            try:
                return str(q.resolve().relative_to(ROOT))
            except ValueError:
                return str(q)

        if keep_intermediate:
            self.write_meta(
                out_dir,
                cube=rel(cube), seg=rel(seg_f), work=rel(work),
                methods=list(methods), K=K, seed=seed,
                continuum_window=continuum_window,
                line_thresholds=list(line_thresholds),
                max_iter=max_iter, clip_sigma=clip_sigma,
                min_unmasked_frac=min_unmasked_frac,
                # max_iter is the cap, not what happened; the row count is the record.
                n_iterations=len(history),
                xlim=xlim, ylim=ylim, exclude_box=exclude_box,
                # Absent unless borrowed; the basis then is not from the spaxels below.
                **(dict(borrowed_basis=borrowed) if borrowed else {}),
                # Absent unless windows were named; the basis is 0 at those channels.
                **(dict(masked_source_lines=source_lines) if source_lines else {}),
                # Absent unless narrowed; every array here then came from that subset.
                **(dict(selected_faintest=faintest) if faintest else {}),
                n_blank_all=n_all, n_blank_used=int(blank_mask.sum()),
                # The spaxels that made mean_sky: those above minus everything dropped.
                n_blank_complete=int(complete.sum()))
        return SkyModel(wl, C_sky, bases, iter_line_mask)

    # =========================================================================
    # step 4 -- template fitting and classification
    # =========================================================================
    #
    # One stage, a fixed wavelength window (source_fit.fit_window), sky-line channels
    # kept out of chi2 because their residual is the sky subtraction's error, not the
    # source. Stellar templates and galaxy eigenspectra are fitted on the same channels;
    # the lower reduced chi2 wins and fixes the redshift with it. The window is fixed
    # rather than following z, which would put pure channel-count steps into chi2(z).
    # Sky residuals lift every source's reduced chi2 together, so there is no absolute
    # threshold on "star-like enough": both values go out, the gap saying how firm.

    @staticmethod
    def write_classification(out_dir, best, ids=None, over=None,
                             keep_intermediate=True):
        """Reduce the fit results to the list step6 rebuilds the sources from.

        Returns (path, fields): the path of classification.npz and the fields that went
        into it, written with keep_intermediate and returned either way, because that is
        how step6 receives them. The stellar library's name is stored alongside (see
        utils.build_templates). The classification was decided by the scan above and is
        not recomputed -- one decision in two places drifts apart invisibly.

        ids  keep only these seg IDs; None means every ID in the best file. Leaving one
             out only means step5 has no template to subtract for it.
        over {id: z} overrides one source's redshift, for sensitivity tests only; the
             amplitude is re-solved at that z.
        """
        over = over or {}
        idx  = {int(i): k for k, i in enumerate(best["id"])}
        ids  = ids if ids else [int(i) for i in best["id"]]

        rows = []
        print(f"\n{'ID':>4}{'class':>8}{'template':>10}{'z':>10}"
              f"{'star X2':>10}{'gal X2':>10}{'margin':>9}")
        print("-" * 61)
        for t in ids:
            if t not in idx:
                print(f"{t:>4}   source not found in best file, skipping")
                continue
            k = idx[t]
            group, tpl = str(best["group"][k]), str(best["template"][k])
            z, A = float(best["z"][k]), np.asarray(best["A"][k], float)
            r1, r2 = float(best["star_red_chi2"][k]), float(best["gal_red_chi2"][k])

            if t in over:
                # The override re-solves from the galaxy scan: keep_scans had to be on.
                s2 = np.load(out_dir / "scans" / f"galaxy_id{t}.npz")
                j  = int(np.argmin(np.abs(s2["z"] - over[t])))
                group, tpl = "galaxy", str(s2["template"][j])
                z, A = float(s2["z"][j]), np.asarray(s2["A"][j], float)

            a = np.full(N_COMPONENTS, np.nan)
            a[:len(A)] = A
            rows.append(dict(id=t, group=group, template=tpl, z=z, A=a))
            mark = "  <- overridden" if t in over else ""
            print(f"{t:>4}{group:>8}{tpl:>10}{z:>10.4f}{r1:>10.2f}{r2:>10.2f}"
                  f"{max(r1, r2) / min(r1, r2):>8.2f}x{mark}")

        if not rows:
            raise SystemExit("no sources found; classification file not written")

        out = out_dir / "classification.npz"
        fields = dict(
            id=np.array([r["id"] for r in rows]),
            group=np.array([r["group"] for r in rows]),
            template=np.array([r["template"] for r in rows]),
            z=np.array([r["z"] for r in rows]),
            A=np.vstack([r["A"] for r in rows]),
            star_library=np.array(STAR_LIBRARY))
        if keep_intermediate:
            np.savez(out, **fields)
        ns = sum(1 for r in rows if r["group"] == "star")
        print(f"\n{len(rows)} sources: {ns} stars / {len(rows) - ns} galaxies")
        print("margin = ratio of the two models' reduced chi2; closer to 1 means less classification confidence")
        if keep_intermediate:
            print(f"saved -> {out}")
        return out, fields

    @staticmethod
    def _visible_cpus():
        """How many CPUs this process is allowed to run on. cpu_count() answers for the
        machine, wrong under an affinity mask or inside a cpuset; sched_getaffinity is
        right but Linux-only. process_cpu_count() is both; the rest are fallbacks.
        """
        if hasattr(os, "process_cpu_count"):            # 3.13+
            n = os.process_cpu_count()
        elif hasattr(os, "sched_getaffinity"):          # Linux
            n = len(os.sched_getaffinity(0))
        else:
            n = os.cpu_count()
        return n or 1

    @blas_single_thread
    def classify_sources(self, sky, spectra, id="all", full_range=False,
            sky_basis=False, allow_partial=False, raw_mask=False,
            ids=None, z_override=[]):
        """Fit every source of `spectra`; return the last mask iteration's classification.
        sky is step3's model and spectra is step2's, both in memory. With keep_intermediate
        source_fits.npz, classification.npz and meta.json go into step04, or into
        step04/mask_iter{N} when several mask iterations are asked for, each being a
        separate run; the whole scans follow only with source_fit.keep_scans.
        """
        s = self.source_fit
        work = self.out
        K = self.sky_line_basis["K"]
        basis = self.sky_line_basis["method"]
        fix_s_at = s["fix_s_at"]
        # One window for both branches: chi2 compares only over the same channels.
        star_window = gal_window = s["fit_window"]
        line_mask_iter = s["line_mask_iter"]
        zmin, zmax, zstep = s["z_min"], s["z_max"], s["z_step"]
        star_dz = s["star_dz"]
        num_workers = s["num_workers"]
        keep_scans = s["keep_scans"]
        keep_intermediate = self.keep_intermediate

        over = {int(k): float(v) for k, v in (x.split("=") for x in z_override)}
        work    = Path(work)
        STEP04 = work / "step04"
        print(f"workspace {work}")
        if full_range:
            star_window = gal_window = self.FULL_RANGE

        # z_override needs the galaxy scan on disk: only the winning row comes back.
        if over and not (keep_intermediate and keep_scans):
            raise SystemExit("★ z_override re-solves from the galaxy scans, which are "
                             "written only with keep_intermediate and "
                             "source_fit.keep_scans both on")
        if keep_intermediate:
            STEP04.mkdir(parents=True, exist_ok=True)

        # The source spectra have to be a sky-subtracted set (see the step 2 section).

        # Row i of iter_line_mask is step3's iteration i+1, so its rows are the choices.
        line_masks = load_line_masks(sky.iter_line_mask, cumulative=not raw_mask)
        for it in line_mask_iter:
            if not isinstance(it, (int, np.integer)) or not 1 <= it <= len(line_masks):
                raise SystemExit(f"★ line_mask_iter {it!r}: step3 produced "
                                 f"{len(line_masks)} mask iterations, so the iterations "
                                 f"available are 1-{len(line_masks)}")

        seg_ids, flux, var, nspax = (spectra.ids, spectra.flux, spectra.var,
                                     spectra.nspax)

        wl_air = sky.wavelength
        wl_vac = air_to_vacuum(wl_air)
        C_sky  = sky.continuum
        sky    = (np.vstack([C_sky, sky.basis[basis]]) if sky_basis
                  else C_sky[None, :])

        # The mask is on air wavelengths, so the fitting window is cut there too.
        win_star = (wl_air >= star_window[0]) & (wl_air < star_window[1])
        win_gal  = (wl_air >= gal_window[0])  & (wl_air < gal_window[1])

        z_exg  = np.arange(zmin, zmax + zstep / 2, zstep)
        z_star = np.arange(-star_dz, star_dz + zstep / 2, zstep)
        files = sorted(DWARF_DIR.glob("*.dat"))
        if not files:
            raise SystemExit(f"★ no .dat templates under {DWARF_DIR}")
        # A template must cover the whole MUSE band: a NaN channel is never solved.
        need_lo = wl_vac.min() / (1 + z_star.max())
        need_hi = wl_vac.max() / (1 + z_star.min())
        star_jobs = []
        for f in files:
            sp = load_ascii_template(f)
            lo, hi = float(sp.t[3]), float(sp.t[-4])
            if lo > need_lo or hi < need_hi:
                print(f"  skipping {f.stem}: rest range {lo:.0f}-{hi:.0f} A does not "
                      f"cover the {need_lo:.0f}-{need_hi:.0f} A needed")
                continue
            # Coverage is read off the spline's domain: a hole inside it passes unseen.
            if not np.all(np.isfinite(sp.c)):
                print(f"  skipping {f.stem}: the spline has a hole inside its own "
                      f"{lo:.0f}-{hi:.0f} A range")
                continue
            star_jobs.append(("star", f.stem, sp, z_star))
        if not star_jobs:
            raise SystemExit(f"★ no template under {DWARF_DIR} covers the MUSE band")
        print(f"{len(star_jobs)} stellar candidates ({STAR_LIBRARY}): "
              + ", ".join(n for _, n, _, _ in star_jobs))
        # One job covers every galaxy: the four eigenspectra interpolate between types.
        gal_jobs = [("galaxy", "eigen", load_eigen_galaxy(EIGEN_GAL), z_exg)]
        # The same check, but fatal: without the one galaxy job that branch is empty.
        if not np.all(np.isfinite(gal_jobs[0][2].c)):
            raise SystemExit(f"★ {EIGEN_GAL.name} has a hole inside its own rest range")

        targets = seg_ids.tolist() if id == "all" else [int(id)]

        n_workers = num_workers or max(1, self._visible_cpus() // 3)
        n_workers = min(n_workers, len(targets))

        print(f"star  {star_window[0]:.0f}-{star_window[1]:.0f} A  "
              f"window {int(win_star.sum())} channels   {len(star_jobs)} stellar templates x "
              f"{z_star.size} z values")
        print(f"galaxy  {gal_window[0]:.0f}-{gal_window[1]:.0f} A  "
              f"window {int(win_gal.sum())} channels   galaxy eigenspectra x {z_exg.size} z values")
        print("classification = lower reduced chi2 on the same channel set (no absolute threshold)")
        print("s is a free parameter" if fix_s_at is None else
              f"sky continuum fixed to {fix_s_at} x C_sky, subtracted first")
        print("source model = A x template" + ("  + sky-line basis" if sky_basis
                                              else "   (1 free parameter)"))
        print(f"spectra from {spectra.path.name}")
        print(f"{len(targets)} object(s)   {n_workers} workers   "
              f"mask iterations {line_mask_iter}")

        # gal_z is the galaxy branch's own best redshift, not `z`; scans are optional.
        KEYS = ("id", "nspax", "group", "template", "z", "A", "s", "chi2",
                "red_chi2", "n_good", "src_min", "star_red_chi2", "star_tpl",
                "gal_red_chi2", "gal_tpl", "gal_z")
        outs = []
        classified = None

        # Each mask iteration is a separate result set; only the mask changes here.
        for it in line_mask_iter:
            line = line_masks[it - 1]
            fit_star, fit_gal = win_star & ~line, win_gal & ~line
            # A step04 directory holds one run, its settings in the meta.json beside it.
            run_dir = STEP04 if len(line_mask_iter) == 1 else STEP04 / f"mask_iter{it}"
            if keep_intermediate:
                run_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{'=' * 112}")
            print(f"mask iter{it}{'(cumulative)' if not raw_mask else '(independent)'}: flagged {int(line.sum()):,} / {line.size} channels"
                  f" ({100 * line.mean():.1f}%)   "
                  f"clean channels for fitting {int(fit_star.sum())}")
            print(f"{'=' * 112}")
            print(f"{'ID':>5}{'nspax':>8}{'group':>8}{'tpl':>7}{'z':>10}{'A':>12}"
                  f"{'n':>7}{'chi2':>14}{'chi2/dof':>10}{'star chi2/dof':>15}"
                  f"{'gal chi2/dof':>14}"
                  f"{'src_min':>10}")
            print("-" * 112)

            _SHARED.update(seg_ids=seg_ids, flux=flux, var=var, nspax=nspax, sky=sky,
                           star_jobs=star_jobs, gal_jobs=gal_jobs, wl_vac=wl_vac,
                           fit_star=fit_star, fit_gal=fit_gal,
                           fix_s_at=fix_s_at,
                           allow_partial=allow_partial,
                           keep_scans=keep_intermediate and keep_scans)

            summary = []
            # One file per branch, written as results arrive, not collected first.
            with contextlib.ExitStack() as stack:
                scan_out = {}
                if keep_intermediate and keep_scans:
                    for branch in ("star", "galaxy"):
                        f = run_dir / f"scans_{branch}.npz"
                        scan_out[branch] = stack.enter_context(
                            zipfile.ZipFile(f, "w", zipfile.ZIP_STORED))
                        scan_out[branch].writestr(
                            "group.npy", self._npy_bytes(np.array(branch)))
                pool = stack.enter_context(Pool(n_workers, initializer=_init_worker,
                                                initargs=(_SHARED,)))
                for t, scans, row in pool.imap(_scan_one, targets):
                    if scans and scan_out:
                        for branch, packed in zip(("star", "galaxy"), scans):
                            if packed is None:
                                continue
                            rows, templates = packed
                            zf = scan_out[branch]
                            zf.writestr(f"id{t}.npy", self._npy_bytes(rows))
                            zf.writestr(f"id{t}_templates.npy",
                                        self._npy_bytes(np.array(templates)))
                    if row is None:
                        print(f"{t:>5}   (all fits failed, skipping)")
                        continue
                    summary.append(row)
                    print(f"{t:>5}{row['nspax']:>8}{row['group']:>8}{row['template']:>7}"
                          f"{row['z']:>10.5f}{row['A'][0]:>12.4g}{row['n_good']:>7}"
                          f"{row['chi2']:>14,.0f}{row['red_chi2']:>10.2f}"
                          f"{row['star_red_chi2']:>15.2f}{row['gal_red_chi2']:>14.2f}"
                          f"{row['src_min']:>10.2f}",
                          flush=True)

            # gal_z is None where the branch failed; as a column that has to be NaN.
            new = {k: np.array([np.nan if x[k] is None else x[k] for x in summary],
                               dtype=float) if k == "gal_z"
                      else np.array([x[k] for x in summary])
                   for k in KEYS}

            out = run_dir / "source_fits.npz"
            # Merge rather than overwrite: re-running one ID updates that row only.
            if keep_intermediate and out.exists():
                old = np.load(out, allow_pickle=False)
                if set(old.files) != set(KEYS):
                    print(f"  * {out.name} fields differ from current format, discarding entire file."
                          f"\n    extra {sorted(set(old.files) - set(KEYS))}"
                          f"  missing {sorted(set(KEYS) - set(old.files))}")
                if set(old.files) == set(KEYS):
                    keep = ~np.isin(old["id"], new["id"])
                    if keep.any():
                        new = {k: np.concatenate([old[k][keep], new[k]]) for k in KEYS}
                        print(f"merged {int(keep.sum())} existing sources")
            o = np.argsort(new["id"])
            # Every value is already the dtype np.savez stores and np.load returns.
            best = {k: v[o] for k, v in new.items()}
            if keep_intermediate:
                np.savez(out, **best)
            cls_path, fields = self.write_classification(run_dir, best, ids, over,
                                                         keep_intermediate)
            # Rebuilt each iteration, so this belongs to the same one as cls_path.
            galaxy_z = {int(x["id"]): x["gal_z"] for x in summary
                        if x["gal_z"] is not None}
            classified = Classification(cls_path, fields, galaxy_z)
            outs.append((it, out, summary))

        print(f"\n{'=' * 60}\ncross-iteration comparison")
        print(f"{'iter':>6}{'clean ch':>10}{'stars':>7}{'galaxies':>9}"
              f"{'star chi2/dof med':>20}{'neg-flux src':>13}")
        print("-" * 65)
        for it, out, summary in outs:
            ns = sum(1 for r in summary if r["group"] == "star")
            med = float(np.median([r["star_red_chi2"] for r in summary]))
            neg = sum(1 for r in summary if r["src_min"] < 0)
            print(f"{it:>6}{int((win_star & ~line_masks[it-1]).sum()):>10}"
                  f"{ns:>7}{len(summary) - ns:>9}{med:>20.2f}{neg:>13}")
        if keep_intermediate:
            print("\n" + "\n".join(f"saved -> {o}" for _, o, _ in outs))
            # Every setting the products were made with; the filenames carry none of it.
            for it, o, summary in outs:
                self.write_meta(
                    o.parent,
                    cube=str(self._repo_path(self.inp["nosky"])),
                    seg=str(self._repo_path(self.out / "step01/segmentation_input.fits")),
                    spectra=str(self._repo_path(spectra.path)),
                    basis=basis, K=K, sky_basis=sky_basis,
                    fix_s_at=fix_s_at, fit_window=list(star_window),
                    line_mask_iter=it, cumulative=not raw_mask,
                    z_min=zmin, z_max=zmax, z_step=zstep, star_dz=star_dz,
                    star_library=STAR_LIBRARY, keep_scans=keep_scans,
                    n_sources=len(summary))
        return classified

    # =========================================================================
    # step 5 -- the sky continuum's spatial field
    # =========================================================================
    #
    #     (1) solve every blank spaxel freely for its own sky-continuum coefficient
    #     (2) identify the main source group (brightest-pixel blob + redshift filter)
    #     (3) fit a smooth field s_hat to those, on points far from every source
    #
    # step6 locks s to that field, leaving source light nowhere to hide inside the sky
    # model (see utils.py).

    @blas_single_thread
    def fit_sky_amplitude(self, white, seg, sky, classification, fix_blank_s_at=None):
        """Build the sky-continuum spatial field; return it. white, seg, sky and
        classification are what steps 1, 3 and 4 returned, in memory. With
        keep_intermediate the field, the per-spaxel solve, main_source_group.png and
        meta.json go into step05 too.
        """
        a = self.sky_amplitude
        work = self.out
        cube = self.inp["cube"]
        K = self.sky_line_basis["K"]
        basis = self.sky_line_basis["method"]
        blank_channels = self.spaxel_fit["blank_channels"]
        min_channel_coverage = self.spaxel_fit["min_channel_coverage"]
        min_source_distance = a["min_source_distance"]
        min_main_source_distance = a["min_main_source_distance"]
        train_clip_sigma = a["train_clip_sigma"]
        main_source_dz = a["main_source_dz"]
        n_iter = a["n_iter"]
        # Empty unless this pointing's sky_region applies to the s field.
        train_xlim = self.train_region.get("train_xlim")
        train_ylim = self.train_region.get("train_ylim")
        train_exclude_box = self.train_region.get("train_exclude_box")
        keep_intermediate = self.keep_intermediate

        work = Path(work)
        CUBE = Path(cube)
        out = work / "step05"
        if keep_intermediate:
            out.mkdir(parents=True, exist_ok=True)

        seg_path, seg = seg.path, seg.data
        white = np.asarray(white.data, float)
        print(f"workdir {work}   cube {CUBE.name}")
        print(f"segmentation: {seg_path.name}  source spaxels {int((seg > 0).sum()):,}")

        # Two configs naming different cubes need only agree in channel count to run.
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
        # From here `sky` is the design matrix: continuum row 0, K line vectors under.
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
        # The redshifts come from the same step4 result the classification does.
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

        # save
        # A field of all NaN would make the sky model and the subtracted cube NaN too.
        if not np.isfinite(s_hat).any():
            raise SystemExit("★ s_hat is NaN in every spaxel; the field was not estimated "
                             f"from the {int(sf_train.sum()):,} training spaxels and is not "
                             "written")
        # Narrowed once, here, so the file and the fit hold the same field to the bit.
        s_hat32 = s_hat.astype(np.float32)
        s_hat_path = out / "sky_continuum_amplitude_field.npy"
        if keep_intermediate:
            np.save(s_hat_path, s_hat32)
            np.save(out / "sky_continuum_amplitude_per_spaxel.npy",
                    s_free.reshape(ny, nx).astype(np.float32))

        if keep_intermediate:
            self.write_meta(
                out,
                cube=str(self._repo_path(CUBE)), seg=str(self._repo_path(seg_path)),
                sky_dir=str(self._repo_path(work / "step03")),
                classification=str(self._repo_path(classification.path)),
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
                # A row with no training spaxel gets 0 -- an assumption, not a measure.
                untrained_rows=[int(i) for i in
                                np.flatnonzero(sf_train.sum(axis=1) == 0)],
                untrained_cols=[int(i) for i in
                                np.flatnonzero(sf_train.sum(axis=0) == 0)])
            print(f"saved -> {out}")
        return SkyAmplitude(s_hat32, s_hat_path)

    # =========================================================================
    # step 6 -- the sky subtraction
    # =========================================================================
    #
    # Every spaxel fitted with s locked to the field s_hat(x, y) step5 built:
    #
    #     blank  (seg = 0)   D = s_hat * C_sky + Sum_k c_k L_k
    #     source (seg > 0)   D = Sum_j a_j T_j + s_hat * C_sky + Sum_k c_k L_k
    #
    # Two cubes come out: sky_subtracted (= data - sky_model) and sky_model, which holds
    # no source term -- only sky is subtracted.

    @staticmethod
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
    def subtract_sky(self, white, seg, sky, classification, s_field):
        """Write the sky-subtracted and sky-model cubes into step06; return that
        directory. white, seg, sky, classification and s_field are what the earlier steps
        returned, in memory. These are the deliverable, so they are written whatever
        keep_intermediate said about the ones before them.
        """
        work = self.out
        cube = self.inp["cube"]
        K = self.sky_line_basis["K"]
        basis = self.sky_line_basis["method"]
        blank_channels = self.spaxel_fit["blank_channels"]
        min_channel_coverage = self.spaxel_fit["min_channel_coverage"]

        work = Path(work)
        CUBE = Path(cube)
        out = work / "step06"
        out.mkdir(parents=True, exist_ok=True)

        seg_path, seg = seg.path, seg.data
        white = np.asarray(white.data, float)
        print(f"workdir {work}   cube {CUBE.name}")
        print(f"segmentation: {seg_path.name}  source spaxels {int((seg > 0).sum()):,}")

        # The templates are redshifted onto this grid, so it is checked, not assumed.
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
        # From here `sky` is the design matrix: continuum row 0, K line vectors under.
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
        A_map     = np.full((N_COMPONENTS, ny * nx), np.nan, np.float32)

        blank = valid & (seg_f == 0)
        rids  = np.unique(seg_f[valid & (seg_f > 0)])
        n_src_tot = int((valid & (seg_f > 0)).sum())

        # blank: re-solve with s locked to s_hat
        print(f"blank {int(blank.sum()):,} spaxels (s locked to field)...",
              end="", flush=True)
        t0 = time.time()
        c = fit_blank(D[:, blank], sky, fit_mask=fit_mask, s_fix=s_hat[blank])
        sky_model[:, blank] = sky.T @ c
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
            A_map[:, m] = c[:N_COMPONENTS]
            sky_model[:, m] = sky.T @ c[N_COMPONENTS:]

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
        # STAT passes through untouched, already the big-endian float32 that goes out.
        with fits.open(CUBE, memmap=True) as hdul:
            self.write_cube(out / "sky_subtracted.fits", cube(sub),
                            hdr_pri, hdr_data, hdul["STAT"].data, hdr_stat)
        self.write_cube(out / "sky_model.fits", cube(sky_model), hdr_pri, hdr_data)
        np.save(out / "source_template_amplitude_map.npy",
                A_map.reshape(N_COMPONENTS, ny, nx))
        # The s applied is not written: step5's field, and sky_model.fits shows the mask.

        self.write_meta(
            out,
            cube=str(self._repo_path(CUBE)), seg=str(self._repo_path(seg_path)),
            sky_dir=str(self._repo_path(work / "step03")),
            classification=str(self._repo_path(classification.path)), basis=basis, K=K,
            s_field=str(self._repo_path(s_field.path)),
            blank_channels=blank_channels, min_channel_coverage=min_channel_coverage,
            n_blank=int(blank.sum()), n_source=n_src_tot,
            n_source_regions=len(rids), n_template_regions=len(templates))

        region = ("all channels" if fit_mask is None
                  else f"line1 {int(fit_mask.sum())}/{fit_mask.size} channels")
        print(f"blank {int(blank.sum()):,} (unweighted, {region})"
              f"  source {n_src_tot:,}"
              f"  source regions {len(rids)} ({len(rids) - n_notpl} with template)")
        print(f"saved -> {out}")
        return out


# =========================================================================
# the entrance and the command line
# =========================================================================

def run_pointing(cfg_path):
    """Run one pointing's config through the pipeline. The public entrance: __init__.py
    exports it, and the command line below calls it once per config file.
    """
    Pipeline(cfg_path).run()


def main():
    """The command line: run each config given, in the order given. Pointings do not
    depend on each other, so a failure stops the run rather than being collected: the
    ones already finished keep their products.
    """
    ap = argparse.ArgumentParser(
        description="Run the sky reconstruction pipeline for one or more pointings")
    ap.add_argument("config", nargs="+",
                    help="pointing config file(s), e.g. configs/p01.yaml")
    args = ap.parse_args()
    for path in args.config:
        run_pointing(path)


if __name__ == "__main__":
    main()
