"""Template fitting for the sources -- one stage, a fixed wavelength window, sky-line
channels kept out of chi2.

How a source is classified

    The stellar templates and the galaxy eigenspectra are fitted separately on the
    **same set of channels**; whichever branch reaches the lower reduced chi2 wins,
    and fixes the redshift at the same time. chi2 is summed only over channels that
    are inside the window and are not sky lines.

    The alternative is to throw all 33 templates (stars + galaxies + QSO) into one
    chi2 comparison over all 3801 channels. The next two paragraphs are why that is
    not what happens here.

Three things are fixed by the method rather than chosen here: the sky-line mask is
applied, the line channels are left out of chi2, and stars and galaxies are each
fitted inside a fixed wavelength window. The window values themselves are the
STAR_WINDOW / GAL_WINDOW defaults below and are not repeated here -- the same
numbers written in two places drift apart eventually.

Why the sky-line channels are excluded: their residual is dominated by the error of
the sky subtraction, not by the source. Counting them in chi2 lets "which template
absorbs the sky residual better" decide the classification and the redshift. (The
rule for blank spaxels is the opposite -- there only the line channels are used,
because the sky is exactly what is being learned.)

Why a fixed window: if every candidate were allowed the channels it happens to
cover, n_good would change with z and chi2(z) would grow steps that come purely
from the channel count. With a fixed window -- as long as it lies where the galaxy
eigenspectra (rest 1183-9840 A) reach across the whole scanned z range -- every
candidate sees exactly the same channels, the steps disappear, and chi2 values can
be subtracted from each other.

Why the classification uses no absolute threshold: a threshold on "is this
star-like enough" only works if the absolute value of reduced chi2 means something,
and sky-line residuals and flux-scale errors lift every source's reduced chi2
together, so the threshold is either too loose or rejects everything. The two
branch winners are compared directly instead, and how much to trust the answer is
carried by **the gap between them** -- star_red_chi2 and gal_red_chi2 are both
written out, and a small gap says the classification is not firm.

The two windows have to be equal: reduced chi2 = chi2 / (n_good - n_param), and
n_good in that denominator is set by the channel set. Different windows are not the
same statistic, so comparing them means nothing. A pointing's config holds one
fit_window and hands it to both branches, so the pair cannot come apart.
"""
import os
from multiprocessing import Pool
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits

from templates import (DWARF_DIR, STAR_LIBRARY,
                       load_ascii_template, load_eigen_galaxy, redshift_to_grid,
                       air_to_vacuum)
from utils import blas_single_thread, load_line_masks

ROOT      = Path(__file__).resolve().parents[2]
# Where the scans are written. This and _SHARED below are the two names _scan_one
# reads, and it reads them from inside a worker process, where the locals of the
# call that started the Pool do not exist; _init_worker fills both in.
STEP04 = None
EIGEN_GAL = ROOT / "data/eigen_galaxy_Bolton2012.fits"

N_SRC    = 4                # fixed width of the A column: 4 eigenspectra, and a star
                            # uses only column 0

# The wavelength window of each branch (A, air). These are the defaults of the
# star_window / gal_window parameters; a pointing's config sets them per run. The
# window is encoded into the output tag, so results from different windows sit side
# by side instead of overwriting each other.
#
# All three share a lower bound, so the only thing separating the windows is how far
# right they reach. 4600 is not "from the beginning" -- 13 pointings start between
# 4599.6 and 4600.3 A, but p14 starts at 4749.83 A with only 3681 channels. The real
# start is whatever each cube's first channel is; comparing reduced chi2 within one
# pointing is unaffected, because both branches use the same channels.
#
# The two windows have to be equal (see the paragraph in the module docstring), so the
# 4600-6000 / 4600-7000 pair the method was first written with cannot be used
# literally -- the n_good in the denominator would differ and the two reduced chi2
# would not be comparable. The upper bound taken is the union of the two, 8000; the
# lower stays 4600.
STAR_WINDOW = (4600.0, 8000.0)      # fitting window for the stellar templates
GAL_WINDOW  = (4600.0, 8000.0)      # for the galaxy eigenspectra; must equal the above
FULL_RANGE  = (4600.0, 9400.0)      # the whole MUSE range, kept as a control

_SHARED = {}


class Classification(NamedTuple):
    """What this step hands steps 5 and 6.

    data holds the fields of classification_{tag}.npz -- step6 rebuilds each
    source's model from them. galaxy_z is the galaxy branch's best redshift for
    every source it could fit, which is a different number from data["z"]: that
    one belongs to the winning branch, and for a star it is a radial velocity.
    Step5 groups the main source by redshift and needs the galaxy branch's.

    path and tag name the product these came from. Steps 5 and 6 record the path
    in their meta.json, and step5 reads the tag to name the step4 run.
    """
    path: Path
    tag: str
    data: dict                # field name -> array, as written to the npz
    galaxy_z: dict            # seg ID -> galaxy-branch redshift


def make_tag(basis, K, fix_s_at, star_window, gal_window, sky_basis, line_iter,
             cumulative=True, suffix=""):
    """The output filename. Every setting that changes the result is encoded into it,
    so a re-run cannot quietly overwrite the previous one.

    The windows and the mask iteration are in there because they decide which channels
    enter chi2: results from different settings are different scientific products and
    have to coexist. The diagnostic scripts call this same function, so the two always
    agree on the naming; written out twice, changing one and forgetting the other
    turns into "reading the wrong file".
    """
    base = f"{basis}_K{K}" if sky_basis else "nobasis"
    return (f"{base}_s{'free' if fix_s_at is None else fix_s_at}"
            f"_{star_window[0]:.0f}-{star_window[1]:.0f}"
            f"_{gal_window[0]:.0f}-{gal_window[1]:.0f}"
            f"_L{line_iter}{'cum' if cumulative else 'raw'}" + suffix)


def make_suffix(spec_dir_name):
    """The tag's suffix: whatever changes the result but is not encoded by make_tag.

    A different spectrum source is a different scientific product. The default source,
    step02, gets no suffix -- the suffix marks a departure from the default, and the
    default itself does not need marking.

    _{STAR_LIBRARY}star names the stellar library. There is only one library at the
    moment, so it is a constant; it stays because every existing product is already
    named this way and dropping it would rename all of them, and because it puts
    "which library produced these" in the filename instead of inside the file.
    """
    return (("" if spec_dir_name == "step02"
             else f"_{spec_dir_name.replace('step02', '')}")
            + f"_{STAR_LIBRARY}star")


def scan_object(flux, var, sky, jobs, lam_muse, fit, fix_s_at=None,
                allow_partial=False):
    """Scan templates and redshifts for one summed spectrum, over the channel set fit.

    fit is a boolean array as long as the spectrum; the wavelength window and the
    sky-line mask are already combined into it. Fitting and scoring use the same set,
    which is what makes the chi2 values comparable.

    jobs is a list of (group, name, spline, z grid). The z grid belongs to each
    candidate rather than being shared -- a star only needs +/-0.005 scanned (peculiar
    velocities inside the Galaxy), a galaxy needs 0 to 1.5.

    A and s are constrained non-negative: a source's amplitude and the sky continuum's
    coefficient cannot physically be negative. The sky-line coefficients are left free,
    because that basis is learned from residuals and is signed by construction.

    When fix_s_at is given, s*C_sky is subtracted from the data first and s stops being a
    free parameter (one fewer). The point is to break the degeneracy between A*T and
    s*C_sky -- the data cannot separate them, and left free the template absorbs the
    sky continuum.
    """
    base = (fit & np.isfinite(flux) & np.isfinite(var) & (var > 0)
            & np.all(np.isfinite(sky), axis=0))
    sig  = np.sqrt(np.where(var > 0, var, 1.0))
    n_full = int(base.sum())            # channels the data offers -- the ceiling every
                                        # candidate shares
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
                                        # A >= 0 is "the source does not emit negative light"
        if s_free:
            lb[n_comp] = 0.0
        ub = np.full(p, np.inf)
        # With no finite bound there is nothing for an active-set method to look for,
        # and the answer is the plain least-squares one.
        has_bounds = np.any(np.isfinite(lb))

        # The spline's own domain. It is the only thing that decides where the
        # redshifted template comes back NaN, because it is evaluated with
        # extrapolate=False.
        lo_rest = float(spline.t[spline.k])
        hi_rest = float(spline.t[-spline.k - 1])

        for z in z_grid:
            T = redshift_to_grid(spline, z, lam_muse)
            if T.ndim == 1:
                T = T[:, None]
            # Reaching past both ends means reaching across everything between them, so
            # whether this candidate covers the channel set is two comparisons and not a
            # pass over T.
            covers = lo_rest * (1 + z) <= lam_lo and hi_rest * (1 + z) >= lam_hi
            if covers:
                good, n = base, n_full
            else:
                good = base & np.isfinite(T[:, 0])
                n    = int(good.sum())
            if n <= p:                  # leave at least one degree of freedom, or
                                        # reduced chi2 is undefined
                continue
            # A candidate that does not cover the whole window is dropped. chi2 is a
            # sum, so a candidate with fewer channels is smaller for free -- without
            # this the scan would run to whatever z leaves the template covering a
            # handful of channels and return a solution that looks perfect because it
            # has almost no data in it. Templates have a finite rest range, so past
            # some z they no longer reach across the window; those candidates have to
            # be excluded rather than allowed to win.
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
                # The singular-value cutoff and the dot product are the ones
                # lsq_linear itself uses. Both branches feed one comparison, so they
                # have to agree to the last bit and not merely to rounding.
                theta = np.linalg.lstsq(M, yw[good], rcond=-1)[0]
                r     = M @ theta - yw[good]
                chi2  = float(r @ r)

            # Whether the source spectrum goes negative is checked over the whole
            # range, not just inside the window: negative flux is a physical problem
            # with the model, and it does not stop existing because those channels
            # were left out of chi2.
            #
            # A template is NaN for a whole channel at once -- the rest wavelength is
            # either inside the spline's domain or it is not, and the components share
            # that domain -- so column 0 answers for all of them. Those channels carry
            # the NaN through the product and ok drops them afterwards, so nothing has
            # to be substituted for them first.
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

    Only a forked worker inherits the parent's memory. Under spawn (the default on
    macOS and Windows) and under forkserver a worker is a fresh interpreter, where
    both are still empty and every fit would fail; passed through the initializer
    they arrive whichever way the worker was started, for one pickle of the shared
    arrays per worker.

    The thread limit is re-applied for the same reason: a fresh interpreter starts
    at the machine default, so a spawned worker would fit with more threads than
    the parent and return different last bits, and a worker per core each taking
    the whole machine would oversubscribe it. It is not scoped to a block -- the
    worker exists to run this and nothing else.
    """
    global _SHARED, STEP04
    _SHARED, STEP04 = shared, step04
    threadpool_limits(limits=1)


def _scan_one(t):
    """Fit one source in a single stage: the stellar templates and the galaxy
    eigenspectra compete on the same channels.

    The shared data comes from _init_worker, once per worker.

    Both branches are scanned and compared directly, so no absolute threshold is
    needed anywhere.

    Why the two reduced chi2 can be compared
        reduced chi2 = chi2 / (n_good - n_param), and that denominator already
        accounts for the difference in degrees of freedom -- 4 components for the
        galaxy eigenspectra against 1 for a stellar template. It only works if both
        used the **same channels** (the same n_good), which is why the two branches
        are given the same window.
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
    # Both winning values are kept: the classification is decided by which of these
    # two numbers is smaller, and without recording them nothing downstream can ask
    # by how much it won, i.e. whether the classification is firm.
    #
    # gal_z is the galaxy branch's own redshift, picked by the lowest reduced chi2
    # over that branch's whole scan. It is not "z" above, which belongs to whichever
    # branch won; step5 needs the galaxy value even for a source classified as a star.
    return t, dict(id=t, nspax=int(np.median(S["nspax"][k])),
                   star_red_chi2=r1[0]["red_chi2"] if r1 else np.nan,
                   star_tpl=r1[0]["template"] if r1 else "",
                   gal_red_chi2=r2[0]["red_chi2"] if r2 else np.nan,
                   gal_tpl=r2[0]["template"] if r2 else "",
                   gal_z=(float(r2[int(np.argmin([x["red_chi2"] for x in r2]))]["z"])
                          if r2 else None),
                   **{**best, "A": A})


def write_classification(out_dir, tag, best, ids=None, over=None,
                         keep_intermediate=True):
    """Reduce the fit results to the list step6 rebuilds the sources from.

    Returns (path, fields): the path of classification_{tag}.npz, and the fields
    that went into it. With keep_intermediate the file is written; the fields are
    returned either way, because that is how step6 receives them.

    The classification was already decided by the scan above -- stellar templates and
    galaxy eigenspectra competing on the same channels, lower reduced chi2 wins. It is
    not recomputed here: the same decision written in two places drifts apart silently
    the moment one of them is edited, and that kind of error is invisible in the output.

    ids  keep only these seg IDs; None means every ID in the best file. Leaving a
         source out only means step5 has no template to subtract for it, with nothing
         gained, so nothing is filtered by default.
    over {id: z} overrides one source's redshift, for sensitivity tests only. The
         amplitude is re-solved at that z -- the template's shape changes with z, so
         an amplitude does not carry over.

    The stellar library's name is stored alongside: downstream has to rebuild the
    source from the same template, and a template name alone does not say which
    library it came from -- the wrong library rebuilds the source from the wrong
    spectrum, silently.
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


def _visible_cpus():
    """How many CPUs this process is allowed to run on.

    cpu_count() answers for the machine, which is the wrong number under an
    affinity mask or inside a cpuset, and sched_getaffinity is the right number
    but exists on Linux alone. process_cpu_count() is both; the two fallbacks are
    for interpreters that predate it.
    """
    if hasattr(os, "process_cpu_count"):            # 3.13+
        n = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):          # Linux
        n = len(os.sched_getaffinity(0))
    else:
        n = os.cpu_count()
    return n or 1


@blas_single_thread
def classify_sources(work, sky, spectra, K, id="all", basis="svd",
        star_window=STAR_WINDOW, gal_window=GAL_WINDOW,
        full_range=False, line_mask_iter=[1, 2, 3, 4], sky_basis=False,
        zmin=0.0, zmax=1.5, zstep=1e-4, star_dz=0.005,
        allow_partial=False, raw_mask=False, fix_s_at=1.0,
        ids=None, z_override=[], num_workers=0, keep_intermediate=True):
    """Fit every source of `spectra`; return the last mask iteration's classification.

    sky is step3's model and spectra is step2's, both in memory. With
    keep_intermediate one best_*.npz and one classification_*.npz per mask
    iteration are written into step04, alongside each source's full scan.

    The result comes back rather than being read from those files by step5 and
    step6, which would make them read one that an earlier run happened to leave
    under the same name.
    """
    over = {int(k): float(v) for k, v in (x.split("=") for x in z_override)}
    work    = Path(work)
    # The scan directory is the module global, because that is where _scan_one
    # reads it from inside a worker; the other one is read here and nowhere else.
    global STEP04
    STEP04 = work / "step04"
    print(f"workspace {work}")
    if full_range:
        star_window = gal_window = FULL_RANGE

    # z_override re-solves one source at a redshift taken from its galaxy scan, and
    # that scan is on disk or nowhere -- only the winning row of it comes back
    # through the Pool.
    if over and not keep_intermediate:
        raise SystemExit("★ z_override reads the scan files, which "
                         "keep_intermediate false does not write")
    if keep_intermediate:
        STEP04.mkdir(parents=True, exist_ok=True)

    # Where the source spectra came from. It has to be a sky-subtracted set:
    # classifying from spectra that still contain the sky produces output that looks
    # entirely normal, with every source's template and redshift wrong.
    #
    # A different spectrum source is a different scientific product, so the tag has to
    # separate them; one workspace can hold several sources (step02_eso and
    # step02_ours, say), and a name that does not encode the source silently
    # overwrites the previous run. The default source, step02, gets no suffix -- the
    # suffix marks a departure from the default, and the default needs no marking.
    suffix = make_suffix(spectra.path.name)

    # The sky-line mask. Row i of iter_line_mask is step3's iteration i+1, so the
    # number of rows is the number of iterations there are to ask for. That count is
    # known only here -- a config is written before step3 has run -- so the requested
    # iterations are checked against it now, ahead of the templates and the workers.
    # An iteration below 1 would index backwards from the end of the array at
    # line_masks[it - 1] and fit a mask nobody asked for.
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
    # A template's rest range has to cover the whole MUSE band. step5 and step6
    # evaluate templates across the whole band, and a channel that is NaN in the
    # design matrix is dropped for every spaxel -- those channels never take part in
    # the solve again. Candidates that cannot cover it are excluded here.
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
        # inside that domain would pass it unseen. That is a property of the file, not
        # of any one redshift, so it is settled here rather than asked per candidate.
        if not np.all(np.isfinite(sp.c)):
            print(f"  skipping {f.stem}: the spline has a hole inside its own "
                  f"{lo:.0f}-{hi:.0f} A range")
            continue
        star_jobs.append(("star", f.stem, sp, z_star))
    if not star_jobs:
        raise SystemExit(f"★ no template under {DWARF_DIR} covers the MUSE band")
    print(f"{len(star_jobs)} stellar candidates ({STAR_LIBRARY}): "
          + ", ".join(n for _, n, _, _ in star_jobs))
    # The galaxy side: the eigenspectra are one four-component model, so a single job
    # scans the whole galaxy population -- linear combinations of the components
    # interpolate continuously between types, and no list of discrete representative
    # spectra is needed.
    gal_jobs = [("galaxy", "eigen", load_eigen_galaxy(EIGEN_GAL), z_exg)]
    # Same check as above, and there is only one galaxy job: with it dropped the branch
    # would be empty and nothing could be classified, so this one has to be fatal.
    if not np.all(np.isfinite(gal_jobs[0][2].c)):
        raise SystemExit(f"★ {EIGEN_GAL.name} has a hole inside its own rest range")

    targets = seg_ids.tolist() if id == "all" else [int(id)]

    n_workers = num_workers or max(1, _visible_cpus() // 3)
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

    # Each mask iteration is a separate set of results: a different channel set gives
    # different chi2, and the two cannot be mixed. The static data (templates, spectra,
    # z grids) is prepared once, and only the mask changes inside the loop.
    for it in line_mask_iter:
        line = line_masks[it - 1]
        fit_star, fit_gal = win_star & ~line, win_gal & ~line
        tag = make_tag(basis, K, fix_s_at, star_window,
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
        # Merge into what is already there rather than overwriting: re-running a
        # single ID should update that row and nothing else. Only when the file is
        # being written -- with nothing to write to, there is nothing to merge into.
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
        # The rows in the order they are written, which is the order everything
        # below reads them in. Every value here is already an array of the dtype
        # np.savez stores and np.load returns, so writing the file and reading it
        # back would hand on exactly this dict.
        best = {k: v[o] for k, v in new.items()}
        if keep_intermediate:
            np.savez(out, **best)
        cls_path, fields = write_classification(STEP04, tag, best, ids, over,
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


# Without this the file would import and exit 0 when run, which reads as having
# done the step. There is one way into the pipeline, and this says where it is.
if __name__ == "__main__":
    raise SystemExit(
        "★ the steps are not run on their own; run the pipeline:\n"
        "      python src/skymodel/run_pipeline.py configs/pNN.yaml")
