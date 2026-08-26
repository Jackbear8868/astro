"""The sky reconstruction pipeline for one pointing: six steps, one entrance.

    conda run -n astro python src/skymodel/pipeline.py configs/p01.yaml
    conda run -n astro python src/skymodel/pipeline.py configs/p0[1-4].yaml

Pipeline.run() below is the whole method in one place: six named steps in the order
they happen, plus the segmentation check between the first two.

    step 1  whitelight        collapse a cube along wavelength into a white light image
    step 2  object_spectra    sum each source's spectrum over the spaxels its seg ID covers
    step 3  sky_basis         learn the sky continuum and the sky-line basis from blank
    step 4  classify_sources  fit templates to every source, giving it a class and a redshift
    step 5  fit_s_field       force the sky continuum amplitude s onto a spatial field
    step 6  subtract_sky      apply the model to every spaxel and write the subtracted cube

Each step is handed what the earlier ones returned rather than reopening the files
they wrote, because a step that reads its input from disk can be handed a file some
earlier run left there, and nothing says so. The cube is the exception: it is the one
input large enough that holding it across steps would cost real memory, and it is
memmapped, so every step that needs it opens it again.

The worker pool step4 fits through comes first: it is the one part of the pipeline
that cannot sit in the class, and the comment heading it says why. After it the file
follows run()'s order -- the products the steps hand each other, then the class, each
step's section carrying the helpers and constants that step uses.

The products under {output}/stepNN are written unless the config turns
keep_intermediate off; step6's are written either way. They are the only record of
the middle of a run, but nothing in the pipeline reads them back. Each step's full
output goes to {output}/stepN.log, headed by the
call that produced it, and only the lines listed in Pipeline.KEEP reach the terminal. The
config as it was read is in {output}/config.json, because the file itself can be
edited afterwards.

The white light is computed from the nosky cube: downstream locates the main source by
its brightest pixel, and the sky continuum of the wsky cube lifts the whole image,
which makes that pixel unreliable.

Nothing here fixes the BLAS thread count. Each fitting step holds BLAS at one thread
around its own work (utils.blas_single_thread), so the products do not follow the
thread count of the machine the pipeline is run on.
"""
import argparse
import contextlib
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.optimize import lsq_linear
from sklearn.decomposition import PCA, TruncatedSVD
from threadpoolctl import threadpool_limits

import matplotlib
matplotlib.use("Agg")              # must be set before importing pyplot: render to file, not screen
import matplotlib.pyplot as plt

# ROOT comes from config rather than being resolved again here: that module sits in
# this directory too, so its parents[2] is the same root.
from config import MAX_GRID_OFFSET, ROOT, load
from utils import (C_KMS, DWARF_DIR, EIGEN_GAL, STAR_LIBRARY,
                   air_to_vacuum, blas_single_thread, build_s_field,
                   build_templates, estimate_continuum, fit_blank, fit_source,
                   load_ascii_template, load_eigen_galaxy, load_line_masks,
                   main_source_group, plot_main_group, redshift_to_grid,
                   wavelength_grid)


# =========================================================================
# the worker pool -- the one part of this file that is not in the class
# =========================================================================
#
# classify_sources runs Pool(n_workers, initializer=_init_worker,
# initargs=(_SHARED, STEP04)) and maps _scan_one over the sources, so all seven
# names have to be reachable from a worker process. None of them can be a
# method: a bound method is pickled together with the object it is bound to, so
# every worker would be sent the whole Pipeline, its config and its paths with
# it. Under the spawn start method -- the default on macOS and Windows -- a
# worker is a fresh interpreter that holds nothing but what it was sent, which
# is why what crosses into one is kept to these seven. N_SRC is one of them
# because _save_scan and _scan_one read it; steps 4 and 6 read the same
# module-level name.

# Where the scans are written. This and _SHARED below are the two names _scan_one
# reads from inside a worker process, where the locals of the call that started the
# Pool do not exist; _init_worker fills both in.
STEP04 = None


# Step 6 reads this width too, for the A_map it writes.
N_SRC    = 4                # fixed width of the A column: 4 eigenspectra, and a star
                            # uses only column 0


_SHARED = {}


def scan_object(flux, var, sky, jobs, lam_muse, fit, fix_s_at=None,
                allow_partial=False):
    """Scan templates and redshifts for one summed spectrum, over the channel set fit.

    fit is a boolean array as long as the spectrum, the wavelength window and the
    sky-line mask already combined into it. Fitting and scoring use the same set,
    which is what makes the chi2 values comparable.

    jobs is a list of (group, name, spline, z grid). The z grid belongs to each
    candidate rather than being shared: a star needs only the peculiar velocities
    inside the Galaxy scanned, a galaxy needs the cosmological range.

    A and s are constrained non-negative, because a source's amplitude and the sky
    continuum's coefficient cannot physically be negative. The sky-line coefficients
    are left free, that basis being learned from residuals and signed by construction.

    When fix_s_at is given, s*C_sky is subtracted from the data first and s stops
    being a free parameter. That breaks the degeneracy between A*T and s*C_sky: the
    data cannot separate the two, and left free the template absorbs the sky
    continuum.
    """
    base = (fit & np.isfinite(flux) & np.isfinite(var) & (var > 0)
            & np.all(np.isfinite(sky), axis=0))
    sig  = np.sqrt(np.where(var > 0, var, 1.0))
    n_full = int(base.sum())            # the ceiling every candidate shares
    if not n_full:                      # no candidate could clear n > p anyway, and the
        return []                       # two ends below would have nothing to read
    # The two ends every candidate has to reach across, taken from the data rather than
    # from the window values, so a mask or a shorter cube shortens them with it.
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

        # The spline's own domain, which is the only thing deciding where the
        # redshifted template comes back NaN -- it is evaluated with extrapolate=False.
        lo_rest = float(spline.t[spline.k])
        hi_rest = float(spline.t[-spline.k - 1])

        for z in z_grid:
            T = redshift_to_grid(spline, z, lam_muse)
            if T.ndim == 1:
                T = T[:, None]
            # Reaching past both ends means reaching across everything between them, so
            # coverage is two comparisons and not a pass over T.
            covers = lo_rest * (1 + z) <= lam_lo and hi_rest * (1 + z) >= lam_hi
            if covers:
                good, n = base, n_full
            else:
                good = base & np.isfinite(T[:, 0])
                n    = int(good.sum())
            if n <= p:                  # leave at least one degree of freedom, or
                                        # reduced chi2 is undefined
                continue
            # A candidate not covering the whole window is dropped: chi2 is a sum, so
            # fewer channels is smaller for free, and the scan would run to whatever z
            # leaves the template on a handful of them and call that perfect.
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
                # The singular-value cutoff and the dot product are lsq_linear's own:
                # both branches feed one comparison, so they have to agree to the last
                # bit and not merely to rounding.
                theta = np.linalg.lstsq(M, yw[good], rcond=-1)[0]
                r     = M @ theta - yw[good]
                chi2  = float(r @ r)

            # Negative source flux is checked over the whole range, not just inside
            # the window: it is a physical problem with the model, and it does not
            # stop existing because those channels were left out of chi2.
            #
            # A template is NaN for a whole channel at once -- the components share
            # one spline domain -- so column 0 answers for all of them.
            ok  = flux_ok & np.isfinite(T[:, 0])
            src = T @ theta[:n_comp]

            results.append(dict(group=group, template=name, z=float(z),
                    A=theta[:n_comp], s=theta[n_comp] if s_free else fix_s_at,
                    chi2=chi2, red_chi2=chi2 / (n - p),
                    n_good=n, src_min=float(src[ok].min())))

    return sorted(results, key=lambda r: r["chi2"])


def _save_scan(path, results):
    """Write a whole scan to an npz the diagnostic scripts can read directly."""
    A = np.full((len(results), N_SRC), np.nan)
    for i, x in enumerate(results):
        A[i, :len(x["A"])] = x["A"]
    np.savez(path, A=A,
             group=np.array([x["group"] for x in results]),
             template=np.array([x["template"] for x in results]),
             z=np.array([x["z"] for x in results]),
             s=np.array([x["s"] for x in results]),
             chi2=np.array([x["chi2"] for x in results]),
             red_chi2=np.array([x["red_chi2"] for x in results]),
             n_good=np.array([x["n_good"] for x in results]),
             src_min=np.array([x["src_min"] for x in results]))


def _init_worker(shared, step04):
    """Give a worker process the two names _scan_one reads.

    Only a forked worker inherits the parent's memory. Under spawn and forkserver a
    worker is a fresh interpreter where both names are still empty and every fit
    would fail; passed through the initializer they arrive whichever way the worker
    was started.

    The thread limit is re-applied for the same reason: a fresh interpreter starts at
    the machine default, so a spawned worker would fit with more threads than the
    parent and return different last bits, and a worker per core would oversubscribe
    the machine.
    """
    global _SHARED, STEP04
    _SHARED, STEP04 = shared, step04
    threadpool_limits(limits=1)


def _scan_one(t):
    """Fit one source in a single stage: the stellar templates and the galaxy
    eigenspectra compete on the same channels.

    The shared data comes from _init_worker, once per worker.

    The two reduced chi2 can be compared because reduced chi2 = chi2 /
    (n_good - n_param) already accounts for the difference in degrees of freedom, 4
    components for the galaxy eigenspectra against 1 for a stellar template. That
    holds only if both used the same channels, which is why the two branches are
    given the same window, and it is what makes an absolute threshold unnecessary.
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
        return t, None
    # The whole scan of each branch is a product, not how the result travels: the
    # row below comes back through the Pool either way.
    if S["keep_intermediate"]:
        if r1:
            _save_scan(STEP04 / f"scan1_id{t}_{S['tag']}.npz", r1)
        if r2:
            _save_scan(STEP04 / f"scan2_id{t}_{S['tag']}.npz", r2)

    # The two branch winners face each other. The scans come back sorted by chi2, so
    # [0] is each branch's best.
    best = min([x[0] for x in (r1, r2) if x], key=lambda d: d["red_chi2"])

    A = np.full(N_SRC, np.nan)
    A[:len(best["A"])] = best["A"]
    # Both winning values are kept: the classification is decided by which of the two
    # is smaller, and without them nothing downstream can ask by how much it won.
    #
    # gal_z is the galaxy branch's own redshift, the lowest reduced chi2 over that
    # branch's whole scan. It is not "z" above, which belongs to whichever branch won;
    # step5 needs the galaxy value even for a source classified as a star.
    return t, dict(id=t, nspax=int(np.median(S["nspax"][k])),
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
# These are the arguments of run(): a step's signature names the ones it takes,
# and they stay locals of run() so nothing of one pointing is left for the next.


class WhiteLight(NamedTuple):
    """What step1 hands the ones after it.

    The header travels with the image because the segmentation check asks where
    each pixel points on the sky, which the array alone cannot answer; every other
    consumer reads only `data`.
    """
    data: np.ndarray          # (ny, nx), the collapsed image, 0 outside the field
    header: fits.Header       # the cube's celestial WCS


class Seg(NamedTuple):
    """The segmentation, as steps 2, 3, 5 and 6 are handed it.

    path is where it was put next to the white light; steps 5 and 6 record that in
    their meta.json, so the products say which map they were made with.
    """
    data: np.ndarray
    path: Path


class SourceSpectra(NamedTuple):
    """What step2 hands step4: one summed spectrum per source.

    `path` is the directory the four arrays were written to. It is carried because
    step4 encodes that directory's name into its output tag -- a different spectrum
    source is a different scientific product -- not because anything reads the
    files back.
    """
    ids: np.ndarray           # (n_ids,)   segmentation IDs, ascending
    flux: np.ndarray          # (n_ids, nz)
    var: np.ndarray           # (n_ids, nz)
    nspax: np.ndarray         # (n_ids, nz)
    path: Path


class SkyModel(NamedTuple):
    """What step3 hands steps 4, 5 and 6 -- everything they read of the sky.

    basis is keyed by decomposition method, because `methods` may ask for several
    in one run and the later steps name the one they fit with. iter_line_mask is
    the whole per-iteration stack: step4 fits one iteration per pass, steps 5 and 6
    take the first.
    """
    wavelength: np.ndarray        # (nz,)          air wavelength of each channel
    continuum: np.ndarray         # (nz,)          C_sky
    basis: dict                   # method -> (K, nz) sky-line basis
    iter_line_mask: np.ndarray    # (n_iter, nz)   bool, one row per iteration


class Classification(NamedTuple):
    """What step4 hands steps 5 and 6.

    data holds the fields of classification_{tag}.npz -- step6 rebuilds each
    source's model from them. galaxy_z is the galaxy branch's best redshift for
    every source it could fit, a different number from data["z"], which belongs to
    the winning branch and for a star is a radial velocity; step5 groups the main
    source by redshift and needs the galaxy branch's.

    path and tag name the product these came from: steps 5 and 6 record the path in
    their meta.json, and step5 reads the tag to name the step4 run.
    """
    path: Path
    tag: str
    data: dict                # field name -> array, as written to the npz
    galaxy_z: dict            # seg ID -> galaxy-branch redshift


class SField(NamedTuple):
    """What step5 hands step6: the field, and where it was written.

    data is the float32 the file holds, not the float64 the fit produced: step6
    locks s to these numbers, and narrowing them afterwards instead would move the
    last bits of every spaxel it fits.
    """
    data: np.ndarray          # (ny, nx) float32
    path: Path                # step05/s_hat.npy


# =========================================================================
# the log a step's output goes to
# =========================================================================

class _Tee:
    """Collect a step's stdout: everything to the log, some of it to the terminal.

    Line-buffered because print() writes the text and the newline separately, and a
    KEEP pattern has to be matched against a whole line. `tail` instead holds back
    the last few non-matching lines, for a step whose summary is at the end.
    """

    def __init__(self, log, keep=None, tail=0):
        self.log, self.keep, self.tail = log, keep, tail
        self.buf, self.held = "", []
        # The real terminal, captured before redirect_stdout puts this object in its
        # place; calling print() from write() would send the line straight back here.
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


# =========================================================================
# the pipeline -- the six steps in the order they happen
# =========================================================================

class Pipeline:
    """One pointing, from its config file to its sky-subtracted cube.

        Pipeline("configs/p01.yaml").run()

    The config is read once, in __init__, and every step reaches the values it
    needs through self.

    What one step hands the next stays an argument, so a method's signature is the
    list of earlier steps it consumes and the products stay locals of run() --
    main() runs several configs one after another, and a product left on the object
    would be there for the next one to pick up.
    """

    # =========================================================================
    # the constants the steps read
    # =========================================================================
    #
    # A method reaches these through self, or through Pipeline when it is a
    # staticmethod: the names of a class body are in scope while that body runs,
    # not inside the methods defined in it.

    # Which lines of a step reach the terminal while it runs; the rest is in the log.
    KEEP = {
        "step3": r"spatial restriction|exclude-box|blank spaxels|svd |pca ",
        # step5's s_hat median is the one number that shows the field was estimated at
        # all; left in the log only, a field of NaN passes the terminal unremarked.
        "step5": r"main source|s spatial field|s_hat median|saved",
        "step6": r"blank|source |saved",
    }

    # An upper bound past the edge of the field: the comparison is against pixel
    # indices, and writing the real NAXIS instead would have to be correct for every
    # pointing, a value too small dropping part of the region without saying so.
    BEYOND_EDGE = 9999

    # How much of one argument the log prints before it says what the value is instead.
    # The point of the head line is the paths and the tags, and a step is also handed
    # whole spectra and maps.
    ARG_WIDTH = 160

    # Defined above learn_sky_basis, which takes it as a default argument: a
    # default is evaluated while this class body runs.
    SEED       = 0           # shared by all decompositions, so a basis is reproducible

    # The whole MUSE range, kept as a control: full_range widens both branches to it in
    # place of the config's window. Both branches always get the same window, because
    # reduced chi2 = chi2 / (n_good - n_param) is comparable only over the same channels.
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
        # Named in full because run_step's own `keep` is a different thing: which
        # of a step's output lines reach the terminal.
        self.keep_intermediate = self.cfg["keep_intermediate"]

        # The one box in the config, translated into the xlim / ylim / exclude_box
        # the steps read, and only for the steps apply_to names. A step the box
        # does not apply to finds nothing here and restricts nothing.
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
        print(f"  pointing #{self.cfg['pointing']}  ->  {self._rel(self.out)}"
              f"   [{self.cfg_path.name}]")
        print(f"  sky region {reg['x']} x {reg['y']}  "
              f"{'include' if reg['include'] else 'exclude'} -> {reg['apply_to']}")
        print("=" * 70)
        t0 = time.time()

        # The segmentation belongs to step 1 rather than counting as a step of its
        # own: it is checked against the white light and written beside it, into the
        # same step01 directory.
        print("--- [1/6] step1 white light (from the nosky cube), and the segmentation")
        white = self.run_step("step1", self.whitelight, {})
        seg = self.place_segmentation(white)

        print("--- [2/6] step2 source spectra (nosky, for classification)")
        spectra = self.run_step("step2", self.object_spectra,
                                dict(white=white, seg=seg))

        print("--- [3/6] step3 sky basis")
        sky = self.run_step("step3", self.sky_basis, dict(white=white, seg=seg),
                            keep=self.KEEP["step3"])

        print("--- [4/6] step4 template fitting and classification")
        # step4's result is the last mask iteration asked for: the classification
        # fields step6 rebuilds the sources from, the galaxy-branch redshifts step5
        # groups the main source by, and the file all of that was written to.
        classified = self.run_step("step4", self.classify_sources,
                                   dict(sky=sky, spectra=spectra), tail=3)

        line_iter = self.source_fit["line_mask_iter"][-1]
        print(f"--- [5/6] step5 build the s field   [mask iter {line_iter}]")
        s_field = self.run_step("step5", self.fit_s_field,
                                dict(white=white, seg=seg, sky=sky,
                                     classification=classified),
                                keep=self.KEEP["step5"])

        print("--- [6/6] step6 final sky subtraction")
        self.run_step("step6", self.subtract_sky,
                      dict(white=white, seg=seg, sky=sky,
                           classification=classified, s_field=s_field),
                      keep=self.KEEP["step6"])

        free = shutil.disk_usage(ROOT).free / 1024 ** 3
        print(f"*** pointing #{self.cfg['pointing']} done in {time.time() - t0:.0f} s"
              f"   {free:.0f} GB free")

    def record_config(self):
        """Write the config this run used into the output directory.

        A step log says which products the step was handed, not the values behind
        them, so this file is where an output directory answers what it was run
        with. It is the config as load() returned it -- optional keys filled in,
        paths resolved -- not a copy of the file, which can be edited afterwards
        and does not carry the defaults at all.
        """
        def plain(v):
            """The config as JSON takes it: paths shortened against the root."""
            if isinstance(v, Path):
                return str(self._rel(v))
            if isinstance(v, dict):
                return {k: plain(x) for k, x in v.items()}
            if isinstance(v, list):
                return [plain(x) for x in v]
            return v

        (self.out / "config.json").write_text(
            json.dumps(plain(self.cfg), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")

    def run_step(self, label, fn, kwargs, keep=None, tail=0):
        """Call one step in this process, sending its output to {output}/{label}.log.

        Whatever the step returns is passed back, which is how the pipeline hands
        one step's results to the next.
        """
        log_path = self.out / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(self.call_repr(fn, kwargs) + "\n\n")
            log.flush()
            tee = _Tee(log, keep, tail)
            try:
                with contextlib.redirect_stdout(tee):
                    result = fn(**kwargs)
            except BaseException:
                # The traceback goes to the log too: the terminal only ever saw the
                # KEEP lines, so the log would otherwise end mid-step with no reason.
                tee.close()
                log.write("\n" + traceback.format_exc())
                print(f"★ {label} failed; full output in {log_path}", flush=True)
                raise
            tee.close()
        return result

    # =========================================================================
    # what the class runs the steps through: the config translation, and the
    # head of every step's log
    # =========================================================================

    def region_kwargs(self, reg, prefix=""):
        """sky_region -> the xlim / ylim / exclude_box that step3 and step5 read.

        prefix is "train_" for step5, where the range restricts the spaxels that train
        the s field rather than the ones the sky is learned from.

        Config ranges are half-open with null for "no bound", and xlim/ylim have the
        same meaning; exclude_box includes both endpoints, so its upper bound loses one.
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
    def _fit(text, value):
        """text if it is short enough to read on one line, else what the value is."""
        if len(text) <= Pipeline.ARG_WIDTH and "\n" not in text:
            return text
        size = f" of {len(value)}" if hasattr(value, "__len__") else ""
        return f"<{type(value).__name__}{size}>"

    @staticmethod
    def show(v):
        """One argument of a step call, written for the head of its log.

        An array is written as its shape and dtype, the values themselves being in the
        products beside the log, and the bundles the steps pass each other are opened up
        so the paths and tags inside them stay visible. Paths are shortened against the
        repository root, an absolute one from another machine being noise here.
        """
        if isinstance(v, np.ndarray):
            return f"<ndarray {v.shape} {v.dtype}>"
        if isinstance(v, Path):
            try:
                return repr(str(v.resolve().relative_to(ROOT)))
            except ValueError:
                return repr(str(v))
        if isinstance(v, tuple) and hasattr(v, "_fields"):          # a step's bundle
            inner = ", ".join(f"{f}={Pipeline.show(x)}" for f, x in zip(v._fields, v))
            return f"{type(v).__name__}({inner})"
        if isinstance(v, dict):
            return Pipeline._fit("{" + ", ".join(f"{Pipeline.show(k)}: {Pipeline.show(x)}"
                                                 for k, x in v.items()) + "}", v)
        if isinstance(v, (list, tuple)):
            body = ", ".join(Pipeline.show(x) for x in v)
            if isinstance(v, tuple):
                body = f"({body},)" if len(v) == 1 else f"({body})"
            else:
                body = f"[{body}]"
            return Pipeline._fit(body, v)
        return Pipeline._fit(repr(v), v)

    @staticmethod
    def call_repr(fn, kwargs):
        """The step call written out as Python, for the head of its log.

        It records which products this step was handed; what the config gave it is in
        the output directory, written by record_config.
        """
        args = ", ".join(f"{k}={Pipeline.show(v)}" for k, v in kwargs.items())
        return f"{fn.__module__}.{fn.__qualname__}({args})"

    # =========================================================================
    # step 1 -- white light
    # =========================================================================
    #
    # Collapse a cube along wavelength into a white light image.
    #
    # Everything downstream that has to say "where is the source" works on this image
    # rather than on the cube: the segmentation is checked against it, and the main
    # source is the blob holding its brightest pixel.

    def whitelight(self, rows=32):
        """Collapse `cube` along wavelength; return the image and its WCS.

        With keep_intermediate the same image is written to `out` as whitelight.fits
        plus a preview png. rows is the number of image rows collapsed at a time --
        memory and speed only.
        """
        cube = self.inp["nosky"]
        out = self.out / "step01"
        keep_intermediate = self.keep_intermediate

        cube, out = Path(cube), Path(out)
        if keep_intermediate:
            out.mkdir(parents=True, exist_ok=True)
        white_fits = out / "whitelight.fits"
        white_png = out / "whitelight.png"

        with fits.open(cube, memmap=True) as hdul:
            data = hdul["DATA"].data
            # A band of image rows at a time, so nanmean's copy of its input is one
            # band's worth, not the cube's. The split must stay spatial: splitting
            # along wavelength would change each pixel's summation order.
            white = np.concatenate([np.nanmean(data[:, y:y + rows, :], axis=0)
                                    for y in range(0, data.shape[1], rows)])
            white = np.nan_to_num(white, nan=0.0)
            # The cube's celestial WCS, the two sky axes without the wavelength one.
            # Without it the white light is a bare array, and the segmentation check
            # could compare only shapes, which do not guarantee alignment.
            hdr = WCS(hdul["DATA"].header).celestial.to_header()

        if keep_intermediate:
            fits.writeto(white_fits, white, hdr, overwrite=True)

            fig = plt.figure(figsize=(6, 6))
            plt.imshow(white, origin="lower", cmap="gray",
                       vmin=np.nanpercentile(white, 5),
                       vmax=np.nanpercentile(white, 99))
            plt.colorbar()
            fig.savefig(white_png, dpi=130)
            # Closed explicitly: this runs in-process, so figures left open would
            # accumulate for the whole run.
            plt.close(fig)
            print(f"saved -> {white_fits}")

        print(f"white light {white.shape} {white.dtype}")
        return WhiteLight(white, hdr)

    # =========================================================================
    # the segmentation check, between step 1 and step 2
    # =========================================================================

    def place_segmentation(self, white):
        """Read the segmentation this pointing was given and confirm it shares a
        pixel grid with the white light; return it.

        The pipeline does not detect sources. Which spaxels hold one is an input,
        named by the config, and the only thing checked here is that it describes
        the same sky as the cube.

        Equal shapes do not prove the same grid, so the check is "where on the sky
        does this pixel point", not a keyword-by-keyword comparison: the seg carries
        a CD matrix while the cube uses PC + CDELT, and their CRPIX differ, both of
        which a literal comparison would report as a mismatch.

        With keep_intermediate the map is copied next to the white light, so the
        output directory records which segmentation the run was given.

        max_offset above the default is a decision to run anyway on a pointing whose
        headers disagree. It comes from that pointing's config and is printed when it
        is above the default, so the bypass is recorded in the file and in the log.
        """
        seg_src = self.inp["seg"]
        out = self.out
        max_offset = self.cfg["max_grid_offset"]
        keep_intermediate = self.keep_intermediate

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

    # =========================================================================
    # step 2 -- source spectra
    # =========================================================================
    #
    # Sum every source's spectrum over the spaxels its segmentation ID covers.
    #
    # These summed spectra are what step4 classifies: one spectrum per source, with its
    # variance and the number of contributing spaxels per channel. They come from a
    # sky-subtracted cube -- classifying a spectrum that still holds the sky gives
    # output that looks entirely normal with every template and redshift wrong.

    def object_spectra(self, white, seg, var_cube=None, top=20):
        """Sum every source's spectrum over the spaxels its segmentation ID covers.

        white and seg come from step1 and from the segmentation check, in memory.
        With keep_intermediate the four summed arrays are written into `out` as well.

        top sets how many rows of the SNR table are printed and changes nothing that
        is saved. The table is there to notice a source far weaker than the rest,
        which no saved array announces on its own.
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
            np.save(out / "object_ids.npy",   ids)
            np.save(out / "object_flux.npy",  flux)
            np.save(out / "object_var.npy",   var)
            np.save(out / "object_nspax.npy", nspax)
            print("saved ->", out)
        return SourceSpectra(ids, flux, var, nspax, out)

    # =========================================================================
    # step 2's helper
    # =========================================================================

    @staticmethod
    def sum_spectra_by_id(cube_path, seg, ids, chunk=200, var_path=None):
        """Sum the spectra of all spaxels belonging to the same segmentation ID.

        Parameters
        ----------
        cube_path : path-like
            MUSE cube; requires a DATA extension.
        var_path : path-like or None
            Where the STAT (variance) is read from; defaults to cube_path. A
            sky-subtracted cube has only DATA -- subtracting a deterministic sky model
            does not change the pixel variance, so the original cube's STAT is correct.
        seg : ndarray, shape (ny, nx)
            Segmentation map, 0 meaning no source. Pixels outside the field of view
            must be set to 0 before calling or they enter the summation.
        ids : ndarray, shape (n_ids,)
            IDs to process.
        chunk : int
            Wavelength planes read at once; memory and speed only.

        Returns
        -------
        flux, var, nspax : ndarray, shape (n_ids, nz)
            The summed spectra, the summed variance (for independent pixels variance is
            what adds, not sigma), and the number of spaxels with valid data per ID per
            channel. nspax is below an ID's total spaxel count wherever spaxels are bad
            or the band ends, and it varies with wavelength, so a sum becomes a mean by
            dividing by it and never by the total: mean_flux = flux / nspax, and
            mean_var = var / nspax**2 because the variance of the mean is 1/n^2 times
            the variance of the sum.

        All three count the same set of spaxels, because the ok mask zeros unusable
        positions before summing. np.nansum on flux and var separately would let each
        skip different positions -- a pixel with valid flux but NaN variance would enter
        flux and not var -- making the two inconsistent.
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
                # Left at the cube's own float32: the widening the sums need happens
                # per source below, so doing it here would double the chunk for nothing.
                d = np.asarray(hdul["DATA"].data[j:j+chunk], np.float32).reshape(-1, seg_flat.size)
                v = np.asarray(vdul["STAT"].data[j:j+chunk], np.float32).reshape(-1, seg_flat.size)

                ok = np.isfinite(d) & np.isfinite(v) & (v > 0)
                d  = np.where(ok, d, 0.0)
                v  = np.where(ok, v, 0.0)

                for k, idx in enumerate(members):
                    # Widened before the sum, not during it: a source covers thousands of
                    # spaxels and float32 accumulation loses digits, while sum()'s float64
                    # accumulator depends on a blocking it does not promise to keep.
                    flux[k,  j:j+chunk] = d[:, idx].astype(np.float64).sum(axis=1)
                    var[k,   j:j+chunk] = v[:, idx].astype(np.float64).sum(axis=1)
                    nspax[k, j:j+chunk] = ok[:, idx].sum(axis=1)

        return flux, var, nspax

    # =========================================================================
    # step 3 -- the sky model
    # =========================================================================
    #
    # Learn the two components of the sky model from blank spaxels: the sky continuum
    # C_sky and K sky-line basis vectors.
    #
    # Output is consumed by step4's template fitting. The decomposition method for
    # the sky-line basis is interchangeable.

    @blas_single_thread
    def sky_basis(self, white, seg):
        """Learn the sky continuum, the line mask and the sky-line basis; return them.

        white and seg come from step1 and from the segmentation check, in memory. With
        keep_intermediate everything learned here is written into step03 as well,
        together with the meta.json recording which spatial range it came from.

        Steps 3, 4 and 6 must all be given the same K and the same decomposition
        method; they read both from the one config section, so they cannot come apart.
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
        # A sigma clip on mean_sky, in units of the robust spread sg. It rejects
        # bad-pixel-level outliers and must not trim the real cross-spaxel variation,
        # so it has to sit far above that variation's natural amplitude.
        clip_sigma = b["clip_sigma"]
        min_unmasked_frac = b["min_unmasked_frac"]
        # The spatial restriction, empty unless this pointing's sky_region applies
        # to the basis.
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
            # The grid every later step reads back from wavelength.npy, so it comes
            # from the shared rule rather than being spelled out again here.
            wl  = wavelength_grid(hdr)

            blank = np.empty((nz, int(blank_mask.sum())), np.float32)
            for j in range(0, nz, 200):
                d = np.asarray(hdul["DATA"].data[j:j+200], np.float32)
                blank[j:j+200] = d[:, blank_mask]

        # Spectrally complete spaxels only: differential atmospheric refraction leaves
        # edge spaxels covered at some wavelengths only, and learn_sky_basis would
        # nan_to_num the rest to 0 -- fabricated data the decomposition would fit.
        complete = np.isfinite(blank).all(axis=0)
        print(f"spectrally complete {int(complete.sum()):,} / {blank.shape[1]:,} "
              f"({100*complete.mean():.1f}%), remainder are partially covered spaxels at field edges, excluded")
        blank = blank[:, complete]

        # Sigma-clip per channel before averaging: the mean's breakdown point is 0%,
        # so a handful of extreme negatives in one channel pulls its mean down, and
        # estimate_continuum then masks it as a "negative line" -- invisible data loss.
        #
        # The clip runs within one channel across spaxels, never along wavelength: a
        # sky emission line is bright in every spaxel, so its brightness sits inside
        # that channel's median and is never clipped.
        #
        # Centre and spread are robust estimators, but the last step still takes the
        # mean, because the cross-spaxel distribution is right-skewed in bright-line
        # channels and the median would be biased low however many samples there are.
        p16, med, p84 = np.percentile(blank, [16, 50, 84], axis=1)
        sg   = np.maximum((p84 - p16) / 2, 1e-6)
        keep = np.abs(blank - med[:, None]) <= clip_sigma * sg[:, None]
        mean_sky = (blank * keep).sum(axis=1, dtype=np.float64) / keep.sum(axis=1)
        print(f"mean_sky: sigma-clip {clip_sigma:g} sigma rejected {int((~keep).sum()):,} / "
              f"{keep.size:,} elements ({100*(~keep).mean():.6f}%)")
        C_sky, sigma, line_mask, history = estimate_continuum(
            mean_sky, thresholds=tuple(line_thresholds),
            window=continuum_window, max_iter=max_iter,
            min_unmasked_frac=min_unmasked_frac)
        print(f"line_mask: {100*line_mask.mean():.1f}% of channels  "
              f"({len(history)} iterations: "
              f"{' -> '.join(f'{100*h[2].mean():.1f}%' for h in history)})")

        # Per-iteration intermediate results. The mask is not cumulative -- see
        # utils.load_line_masks, which is where that is applied.
        iter_line_mask = np.array([h[2] for h in history])

        if keep_intermediate:
            np.save(out_dir / "wavelength.npy",    wl)
            np.save(out_dir / "mean_sky.npy",      mean_sky)
            np.save(out_dir / "sky_continuum.npy", C_sky)
            np.save(out_dir / "sky_sigma.npy",     sigma)
            np.save(out_dir / "line_mask.npy",     line_mask)
            np.save(out_dir / "iter_continuum.npy", np.array([h[0] for h in history]))
            np.save(out_dir / "iter_sigma.npy",     np.array([h[1] for h in history]))
            np.save(out_dir / "iter_line_mask.npy", iter_line_mask)

        # The same keep mask applies: blank - C_sky differs by a per-channel constant
        # only, which shifts x and its median alike, so |x - med| / sg is unchanged.
        #
        # Rejected positions are filled with the channel's typical residual
        # med - C_sky, not 0: a 0 on a sky-line channel claims there is no line
        # there, and med is the more honest value.
        residual = blank - C_sky[:, None]
        np.copyto(residual, (med - C_sky)[:, None], where=~keep)

        bases = {}
        for method in methods:
            t0 = time.time()
            bases[method] = basis = self.learn_sky_basis(residual, K=K, method=method, seed=seed)
            if keep_intermediate:
                np.save(out_dir / f"sky_basis_{method}_K{K}.npy", basis)   # filename includes K so different K values can coexist
            print(f"{method:13s} basis {basis.shape}  {time.time() - t0:6.1f}s", flush=True)

        # Provenance of the products. Only method and K reach the filename, so a
        # re-run with a different spatial range, segmentation or cube overwrites
        # silently; this JSON is the sole record of those choices.
        def rel(q):
            q = Path(q)
            try:
                return str(q.resolve().relative_to(ROOT))
            except ValueError:
                return str(q)

        if keep_intermediate:
            (out_dir / "meta.json").write_text(json.dumps(dict(
                created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                git_commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                          capture_output=True, text=True,
                                          cwd=ROOT).stdout.strip(),
                cube=rel(cube), seg=rel(seg_f), work=rel(work),
                methods=list(methods), K=K, seed=seed,
                continuum_window=continuum_window,
                line_thresholds=list(line_thresholds),
                max_iter=max_iter, clip_sigma=clip_sigma,
                xlim=xlim, ylim=ylim, exclude_box=exclude_box,
                n_blank_all=n_all, n_blank_used=int(blank_mask.sum()),
            ), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"meta -> {out_dir / 'meta.json'}")
        return SkyModel(wl, C_sky, bases, iter_line_mask)

    # =========================================================================
    # step 3's helper
    # =========================================================================

    @staticmethod
    def learn_sky_basis(residual, K=10, method="pca", seed=SEED, chunk=200):
        """Learn K sky-line basis vectors from the residuals of blank spaxels.

        Every method returns shape (K, nz), so the design matrix always has exactly K
        free parameters and chi2 values from different methods are comparable.

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
        basis : ndarray, shape (K, nz)
            K sky-line basis vectors, in the downstream coefficient order.
        """
        # (n_blank, nz), a block of spaxels at a time so only a block is ever float64.
        # nan_to_num must come before the narrowing cast, never after: narrowing first
        # would turn an infinity into a finite 3.4e38 the decomposition would fit.
        X = np.empty((residual.shape[1], residual.shape[0]), np.float32)
        for i in range(0, X.shape[0], chunk):
            X[i:i+chunk] = np.nan_to_num(residual.T[i:i+chunk])

        # random_state is essential, not a precaution: both TruncatedSVD and PCA default
        # to randomized SVD, so without a fixed seed the basis changes on every run.
        if method == "pca":
            p = PCA(n_components=K - 1, random_state=seed).fit(X)
            return np.vstack([p.mean_[None, :], p.components_])

        if method == "svd":
            return TruncatedSVD(n_components=K, random_state=seed).fit(X).components_

        raise ValueError(f"unknown method: {method}")

    # =========================================================================
    # step 4 -- template fitting and classification
    # =========================================================================
    #
    # Template fitting for the sources -- one stage, a fixed wavelength window, sky-line
    # channels kept out of chi2.
    #
    # The stellar templates and the galaxy eigenspectra are fitted separately on the
    # same set of channels; whichever branch reaches the lower reduced chi2 wins, and
    # fixes the redshift at the same time. The window comes from the pointing's config
    # (source_fit.fit_window) and is not repeated here.
    #
    # The sky-line channels are excluded because their residual is dominated by the
    # error of the sky subtraction rather than by the source, and counting them would
    # let "which template absorbs the sky residual better" decide the classification.
    # The rule for blank spaxels is the opposite -- there only the line channels are
    # used, because the sky is what is being learned.
    #
    # The window is fixed, and the same for both branches, because reduced chi2 =
    # chi2 / (n_good - n_param): channels that come and go with z would put steps into
    # chi2(z) that are pure channel count, and two different windows are not the same
    # statistic at all.
    #
    # There is no absolute threshold on "star-like enough": sky-line residuals and
    # flux-scale errors lift every source's reduced chi2 together, so any such
    # threshold is either too loose or rejects everything. The two branch winners are
    # compared directly instead, and both star_red_chi2 and gal_red_chi2 are written
    # out, because the gap between them is what says whether the answer is firm.

    @blas_single_thread
    def classify_sources(self, sky, spectra, id="all", full_range=False,
            sky_basis=False, allow_partial=False, raw_mask=False,
            ids=None, z_override=[]):
        """Fit every source of `spectra`; return the last mask iteration's classification.

        sky is step3's model and spectra is step2's, both in memory. With
        keep_intermediate one best_*.npz and one classification_*.npz per mask
        iteration are written into step04, alongside each source's full scan.
        """
        s = self.source_fit
        work = self.out
        K = self.sky_line_basis["K"]
        basis = self.sky_line_basis["method"]
        fix_s_at = s["fix_s_at"]
        # One window from the config, handed to both branches: the two reduced chi2
        # are comparable only if both were fitted over the same channels.
        star_window = gal_window = s["fit_window"]
        line_mask_iter = s["line_mask_iter"]
        zmin, zmax, zstep = s["z_min"], s["z_max"], s["z_step"]
        star_dz = s["star_dz"]
        num_workers = s["num_workers"]
        keep_intermediate = self.keep_intermediate

        over = {int(k): float(v) for k, v in (x.split("=") for x in z_override)}
        work    = Path(work)
        # A module global because that is where _scan_one reads it from inside a
        # worker process.
        global STEP04
        STEP04 = work / "step04"
        print(f"workspace {work}")
        if full_range:
            star_window = gal_window = self.FULL_RANGE

        # z_override re-solves one source at a redshift taken from its galaxy scan,
        # and that scan is on disk or nowhere -- only its winning row comes back
        # through the Pool.
        if over and not keep_intermediate:
            raise SystemExit("★ z_override reads the scan files, which "
                             "keep_intermediate false does not write")
        if keep_intermediate:
            STEP04.mkdir(parents=True, exist_ok=True)

        # Where the source spectra came from. It has to be a sky-subtracted set:
        # classifying from spectra that still contain the sky produces output that
        # looks entirely normal, with every source's template and redshift wrong.
        suffix = self.make_suffix(spectra.path.name)

        # Row i of iter_line_mask is step3's iteration i+1, so the number of rows is
        # the number of iterations there are to ask for. That count is known only
        # here, a config being written before step3 has run.
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

        # The mask is defined on air wavelengths, so the fitting window is cut on air
        # wavelengths too, and the two agree.
        win_star = (wl_air >= star_window[0]) & (wl_air < star_window[1])
        win_gal  = (wl_air >= gal_window[0])  & (wl_air < gal_window[1])

        z_exg  = np.arange(zmin, zmax + zstep / 2, zstep)
        z_star = np.arange(-star_dz, star_dz + zstep / 2, zstep)
        files = sorted(DWARF_DIR.glob("*.dat"))
        if not files:
            raise SystemExit(f"★ no .dat templates under {DWARF_DIR}")
        # A template's rest range has to cover the whole MUSE band: steps 5 and 6
        # evaluate templates across all of it, and a channel that is NaN in the design
        # matrix is dropped for every spaxel and never solved again.
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
            # The scan reads a candidate's coverage off the spline's domain, so a hole
            # inside that domain would pass unseen. It is a property of the file, not
            # of any one redshift, so it is settled here.
            if not np.all(np.isfinite(sp.c)):
                print(f"  skipping {f.stem}: the spline has a hole inside its own "
                      f"{lo:.0f}-{hi:.0f} A range")
                continue
            star_jobs.append(("star", f.stem, sp, z_star))
        if not star_jobs:
            raise SystemExit(f"★ no template under {DWARF_DIR} covers the MUSE band")
        print(f"{len(star_jobs)} stellar candidates ({STAR_LIBRARY}): "
              + ", ".join(n for _, n, _, _ in star_jobs))
        # One job covers the whole galaxy population: the eigenspectra are a single
        # four-component model whose linear combinations interpolate continuously
        # between types, so no list of discrete representative spectra is needed.
        gal_jobs = [("galaxy", "eigen", load_eigen_galaxy(EIGEN_GAL), z_exg)]
        # The same check, but fatal: there is only one galaxy job, and without it the
        # branch would be empty and nothing could be classified.
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

        KEYS = ("id", "nspax", "group", "template", "z", "A", "s", "chi2",
                "red_chi2", "n_good", "src_min", "star_red_chi2", "star_tpl",
                "gal_red_chi2", "gal_tpl")
        outs = []
        classified = None

        # Each mask iteration is a separate set of results: a different channel set
        # gives different chi2, and the two cannot be mixed. Only the mask changes
        # inside the loop; the templates, spectra and z grids are prepared once.
        for it in line_mask_iter:
            line = line_masks[it - 1]
            fit_star, fit_gal = win_star & ~line, win_gal & ~line
            tag = self.make_tag(basis, K, fix_s_at, star_window,
                                gal_window, sky_basis, it, not raw_mask, suffix)

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
                           tag=tag, fix_s_at=fix_s_at,
                           allow_partial=allow_partial,
                           keep_intermediate=keep_intermediate)

            summary = []
            with Pool(n_workers, initializer=_init_worker,
                      initargs=(_SHARED, STEP04)) as pool:
                for t, row in pool.imap(_scan_one, targets):
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

            new = {k: np.array([x[k] for x in summary]) for k in KEYS}

            out = STEP04 / f"best_{tag}.npz"
            # Merge rather than overwrite: re-running a single ID should update that
            # row and nothing else.
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
            # The rows in the order they are written. Every value is already an array
            # of the dtype np.savez stores and np.load returns, so writing the file and
            # reading it back would hand on exactly this dict.
            best = {k: v[o] for k, v in new.items()}
            if keep_intermediate:
                np.savez(out, **best)
            cls_path, fields = self.write_classification(STEP04, tag, best, ids, over,
                                                         keep_intermediate)
            # The galaxy branch's redshift for every source it could fit. Rebuilt each
            # iteration, so what is returned belongs to the same iteration as cls_path.
            galaxy_z = {int(x["id"]): x["gal_z"] for x in summary
                        if x["gal_z"] is not None}
            classified = Classification(cls_path, tag, fields, galaxy_z)
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
        return classified

    # =========================================================================
    # step 4's helpers
    # =========================================================================

    @staticmethod
    def make_tag(basis, K, fix_s_at, star_window, gal_window, sky_basis, line_iter,
                 cumulative=True, suffix=""):
        """The output filename. Every setting that changes the result is encoded into it,
        so a re-run cannot quietly overwrite the previous one.

        The windows and the mask iteration are in there because they decide which channels
        enter chi2, and results from different channel sets are different scientific
        products that have to coexist. The diagnostic scripts call this same function, so
        a naming written out twice cannot drift into "reading the wrong file".
        """
        base = f"{basis}_K{K}" if sky_basis else "nobasis"
        return (f"{base}_s{'free' if fix_s_at is None else fix_s_at}"
                f"_{star_window[0]:.0f}-{star_window[1]:.0f}"
                f"_{gal_window[0]:.0f}-{gal_window[1]:.0f}"
                f"_L{line_iter}{'cum' if cumulative else 'raw'}" + suffix)

    @staticmethod
    def make_suffix(spec_dir_name):
        """The tag's suffix: whatever changes the result but is not encoded by make_tag.

        A different spectrum source is a different scientific product, and one workspace
        can hold several. The default source, step02, gets no suffix -- the suffix marks a
        departure from the default.

        _{STAR_LIBRARY}star names the stellar library, so that which library produced a
        product is in the filename rather than only inside the file.
        """
        return (("" if spec_dir_name == "step02"
                 else f"_{spec_dir_name.replace('step02', '')}")
                + f"_{STAR_LIBRARY}star")

    @staticmethod
    def write_classification(out_dir, tag, best, ids=None, over=None,
                             keep_intermediate=True):
        """Reduce the fit results to the list step6 rebuilds the sources from.

        Returns (path, fields): the path of classification_{tag}.npz, and the fields that
        went into it. With keep_intermediate the file is written; the fields are returned
        either way, because that is how step6 receives them.

        The classification was already decided by the scan above and is not recomputed
        here -- the same decision written in two places drifts apart the moment one is
        edited, and that error is invisible in the output.

        ids  keep only these seg IDs; None means every ID in the best file. Leaving a
             source out only means step5 has no template to subtract for it.
        over {id: z} overrides one source's redshift, for sensitivity tests only. The
             amplitude is re-solved at that z, the template's shape changing with z.

        The stellar library's name is stored alongside, because a template name alone
        does not say which library it came from and the wrong one would rebuild the
        source from the wrong spectrum, silently.
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
                s2 = np.load(out_dir / f"scan2_id{t}_{tag}.npz")
                j  = int(np.argmin(np.abs(s2["z"] - over[t])))
                group, tpl = "galaxy", str(s2["template"][j])
                z, A = float(s2["z"][j]), np.asarray(s2["A"][j], float)

            a = np.full(N_SRC, np.nan)
            a[:len(A)] = A
            rows.append(dict(id=t, group=group, template=tpl, z=z, A=a))
            mark = "  <- overridden" if t in over else ""
            print(f"{t:>4}{group:>8}{tpl:>10}{z:>10.4f}{r1:>10.2f}{r2:>10.2f}"
                  f"{max(r1, r2) / min(r1, r2):>8.2f}x{mark}")

        if not rows:
            raise SystemExit("no sources found; classification file not written")

        out = out_dir / f"classification_{tag}.npz"
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
        """How many CPUs this process is allowed to run on.

        cpu_count() answers for the machine, which is the wrong number under an affinity
        mask or inside a cpuset, and sched_getaffinity is right but Linux-only.
        process_cpu_count() is both; the fallbacks are for interpreters predating it.
        """
        if hasattr(os, "process_cpu_count"):            # 3.13+
            n = os.process_cpu_count()
        elif hasattr(os, "sched_getaffinity"):          # Linux
            n = len(os.sched_getaffinity(0))
        else:
            n = os.cpu_count()
        return n or 1

    # =========================================================================
    # step 5 -- the sky continuum's spatial field
    # =========================================================================
    #
    # Build the sky-continuum spatial field s_hat(x, y).
    #
    #     (1) Solve all blank spaxels freely to get s_free -- each spaxel's own
    #         sky-continuum coefficient, unconstrained.
    #     (2) Identify the main source group (brightest-pixel blob + redshift filter).
    #     (3) From s_free, build a smooth spatial field s_hat using only training
    #         points far from all sources.
    #
    # step6 locks s to that field. Replacing per-spaxel freedom with a smooth surface
    # leaves source light nowhere to hide inside the sky model, so it is preserved in
    # the residual.

    @blas_single_thread
    def fit_s_field(self, white, seg, sky, classification, fix_blank_s_at=None):
        """Build the sky-continuum spatial field; return it.

        white, seg, sky and classification are what steps 1, 3 and 4 returned, in
        memory. With keep_intermediate s_hat.npy, s_free.npy, main_group.png and
        meta.json are written into step05 as well.
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

        # The sky model was learned on the grid of whatever cube step3 read, and a
        # config naming one pointing's cube there and another's here need only agree
        # in channel count to run to the end with the two offset against each other.
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
        # From here `sky` is the design matrix the spaxel fits use: the continuum as
        # row 0, the K line vectors under it.
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
        # The redshifts come from the same step4 result the classification does, so
        # the grouping and the source models cannot come from two different fits.
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
                            out / "main_group.png", title=Path(work).name)

        # build field
        t0 = time.time()
        s_hat, sf_train = build_s_field(
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
        # step6 locks s to this field, so a field that is NaN everywhere makes the sky
        # model and the subtracted cube NaN too, and nothing further down separates
        # that from a subtraction that worked.
        if not np.isfinite(s_hat).any():
            raise SystemExit("★ s_hat is NaN in every spaxel; the field was not estimated "
                             f"from the {int(sf_train.sum()):,} training spaxels and is not "
                             "written")
        # Narrowed once, here, and step6 is given these numbers rather than the float64
        # they came from: the file and the fit have to hold the same field.
        s_hat32 = s_hat.astype(np.float32)
        s_hat_path = out / "s_hat.npy"
        if keep_intermediate:
            np.save(s_hat_path, s_hat32)
            np.save(out / "s_free.npy", s_free.reshape(ny, nx).astype(np.float32))

        meta = dict(
            step="s_field",
            cube=str(self._rel(CUBE)), seg=str(self._rel(seg_path)),
            sky_dir=str(self._rel(work / "step03")),
            classification=str(self._rel(classification.path)), basis=basis, K=K,
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
            created=datetime.datetime.now().isoformat(timespec="seconds"),
            git_commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True,
                                      cwd=ROOT).stdout.strip())
        if keep_intermediate:
            (out / "meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"saved -> {out}")
        return SField(s_hat32, s_hat_path)

    # =========================================================================
    # step 6 -- the sky subtraction
    # =========================================================================
    #
    # Final per-spaxel sky subtraction using the s-field from step5.
    #
    # Reads the spatial field s_hat built in step5 and fits every spaxel with s
    # locked to s_hat(x, y):
    #
    #     blank  (seg = 0)   D = s_hat * C_sky + Sum_k c_k L_k
    #     source (seg > 0)   D = Sum_j a_j T_j + s_hat * C_sky + Sum_k c_k L_k
    #
    # The output is two cubes: sky_subtracted (= data - sky_model) and sky_model
    # itself. The source template term is NOT part of sky_model -- only sky is
    # subtracted; the source is preserved.

    @blas_single_thread
    def subtract_sky(self, white, seg, sky, classification, s_field):
        """Write the sky-subtracted and sky-model cubes into step06; return that directory.

        white, seg, sky, classification and s_field are what the earlier steps
        returned, in memory. This step's products are the deliverable, so they are
        written whatever keep_intermediate said about the ones before them.
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

        # The sky model was learned on the grid of whatever cube step3 read, and the
        # source templates are about to be redshifted onto that same grid, so the grid
        # is checked instead of assumed (see the same check in fit_s_field).
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
        # STAT is passed through untouched, so it goes to the writer straight from the
        # input file: on disk it is already the big-endian float32 that goes back out.
        with fits.open(CUBE, memmap=True) as hdul:
            self.write_cube(out / "sky_subtracted.fits", cube(sub),
                            hdr_pri, hdr_data, hdul["STAT"].data, hdr_stat)
        self.write_cube(out / "sky_model.fits", cube(sky_model), hdr_pri, hdr_data)
        np.save(out / "A_map.npy", A_map.reshape(N_SRC, ny, nx))
        np.save(out / "s_map.npy", s_map.reshape(ny, nx))

        meta = dict(
            step="fit_sky",
            cube=str(self._rel(CUBE)), seg=str(self._rel(seg_path)),
            sky_dir=str(self._rel(work / "step03")),
            classification=str(self._rel(classification.path)), basis=basis, K=K,
            s_field=str(self._rel(s_field.path)),
            blank_channels=blank_channels, min_channel_coverage=min_channel_coverage,
            n_blank=int(blank.sum()), n_source=n_src_tot,
            n_source_regions=len(rids), n_template_regions=len(templates),
            created=datetime.datetime.now().isoformat(timespec="seconds"),
            git_commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True,
                                      cwd=ROOT).stdout.strip())
        (out / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        region = ("all channels" if fit_mask is None
                  else f"line1 {int(fit_mask.sum())}/{fit_mask.size} channels")
        print(f"blank {int(blank.sum()):,} (unweighted, {region})"
              f"  source {n_src_tot:,}"
              f"  source regions {len(rids)} ({len(rids) - n_notpl} with template)")
        print(f"saved -> {out}")
        return out

    # =========================================================================
    # steps 5 and 6's helpers
    # =========================================================================

    # Shortens a path against the repository root, for the meta.json steps 5 and 6 write.
    @staticmethod
    def _rel(p):
        p = Path(p)
        try:
            return p.resolve().relative_to(ROOT)
        except ValueError:
            return p

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


# =========================================================================
# the entrance and the command line
# =========================================================================

def run_pointing(cfg_path):
    """Run one pointing's config through the pipeline.

    The public entrance: __init__.py exports it, and the command line below calls it
    once per config file it was given.
    """
    Pipeline(cfg_path).run()


def main():
    """The command line: run each config given, in the order given.

    Pointings do not depend on each other, so a failure stops the run rather than
    being collected -- the ones already finished keep their products, and the
    command can be given the rest.
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
