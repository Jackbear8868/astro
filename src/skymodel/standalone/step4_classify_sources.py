"""Step 4, on its own: fit templates to every source and classify it.

One stage, a fixed wavelength window, sky-line channels kept out of chi2.

The stellar templates and the galaxy eigenspectra are fitted separately on the same set
of channels; whichever branch reaches the lower reduced chi2 wins, and fixes the
redshift at the same time.

The sky-line channels are excluded because their residual is dominated by the error of
the sky subtraction rather than by the source, and counting them would let "which
template absorbs the sky residual better" decide the classification. The rule for blank
spaxels is the opposite -- there only the line channels are used, because the sky is
what is being learned.

The window is fixed rather than following z, because reduced chi2 = chi2 /
(n_good - n_param): channels that came and went with z would put steps into chi2(z)
that are pure channel count.

There is no absolute threshold on "star-like enough": sky-line residuals and flux-scale
errors lift every source's reduced chi2 together, so any such threshold is either too
loose or rejects everything. The two branch winners are compared directly instead, and
both star_red_chi2 and gal_red_chi2 are written out, because the gap between them is
what says whether the answer is firm.

    python step4_classify_sources.py --work results/skymodel/p01

step01, step02 and step03 are read from <work>.

This is the standalone copy of `scan_object`, `_pack_scan`, `_init_worker`,
`_scan_one`, `Pipeline.write_classification`, `Pipeline._visible_cpus` and
`Pipeline.classify_sources`.
"""

import argparse
import contextlib
import io
import os
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits

import step_io
from step_io import Classification
from step1_whitelight import repo_path, write_meta
from utils import (DWARF_DIR, EIGEN_GAL, N_COMPONENTS, STAR_LIBRARY, air_to_vacuum,
                   blas_single_thread, load_ascii_template, load_eigen_galaxy,
                   load_line_masks, redshift_to_grid)

FULL_RANGE = (4600.0, 9400.0)


# =========================================================================
# the worker pool
# =========================================================================
#
# classify_sources runs Pool(n_workers, initializer=_init_worker, initargs=(_SHARED,))
# and maps _scan_one over the sources, so all of these names have to be reachable from
# a worker process. Under the spawn start method -- the default on macOS and Windows --
# a worker is a fresh interpreter that holds nothing but what it was sent, which is why
# what crosses into one is kept to these.

_SHARED = {}


def npy_bytes(a):
    """One array as the bytes of a .npy file.

    np.savez builds a zip of .npy members; writing the members one at a time is the
    same file, arrived at without holding every array at once, which is what lets a
    scan join the file as its worker finishes.
    """
    buf = io.BytesIO()
    np.save(buf, a, allow_pickle=False)
    return buf.getvalue()


def scan_object(flux, var, sky, jobs, lam_muse, fit, fix_s_at=None,
                allow_partial=False):
    """Scan templates and redshifts for one summed spectrum, over the channel set fit.

    fit is a boolean array as long as the spectrum, the wavelength window and the
    sky-line mask already combined into it. Fitting and scoring use the same set, which
    is what makes the chi2 values comparable.

    jobs is a list of (group, name, spline, z grid). The z grid belongs to each
    candidate rather than being shared: a star needs only the peculiar velocities
    inside the Galaxy scanned, a galaxy the cosmological range.

    A and s are constrained non-negative, a source's amplitude and the sky continuum's
    coefficient not being physically able to go negative. The sky-line coefficients are
    left free, that basis being learned from residuals and signed by construction.

    When fix_s_at is given, s*C_sky is subtracted from the data first and s stops being
    a free parameter, which breaks the degeneracy between A*T and s*C_sky: the data
    cannot separate the two, and left free the template absorbs the sky continuum.
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
            # leaves the template on a handful of them.
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

            # Negative source flux is a physical problem with the model, so it is
            # checked over the whole range and not only inside the window.
            #
            # A template is NaN for a whole channel at once -- the components share one
            # spline domain -- so column 0 answers for all of them.
            ok  = flux_ok & np.isfinite(T[:, 0])
            src = T @ theta[:n_comp]

            results.append(dict(group=group, template=name, z=float(z),
                    A=theta[:n_comp], s=theta[n_comp] if s_free else fix_s_at,
                    chi2=chi2, red_chi2=chi2 / (n - p),
                    n_good=n, src_min=float(src[ok].min())))

    return sorted(results, key=lambda r: r["chi2"])


def _pack_scan(results):
    """One source's whole scan as a structured array, one row per candidate fit.

    Structured rather than a column per key so a whole scan is one array, which is what
    lets a worker hand it back and the parent write every source into one file. The
    template name becomes an index into `templates`, returned alongside: a branch scans
    a handful of templates over thousands of redshifts, so the name repeated once per
    row is a third of what the scan would otherwise weigh.

    Returns (rows, templates). `group` is not a column: a scan is one branch, so it is
    one value for the whole file and the writer keeps it there.
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
    """Give a worker process the one name _scan_one reads.

    Only a forked worker inherits the parent's memory; passed through the initializer
    the names arrive whichever way the worker was started.

    The thread limit is re-applied for the same reason: a fresh interpreter starts at
    the machine default, so a spawned worker would fit with more threads than the
    parent and return different last bits, and a worker per core would oversubscribe
    the machine.
    """
    global _SHARED
    _SHARED = shared
    threadpool_limits(limits=1)


def _scan_one(t):
    """Fit one source in a single stage: the stellar templates and the galaxy
    eigenspectra compete on the same channels.

    The shared data comes from _init_worker, once per worker.

    The two reduced chi2 can be compared because reduced chi2 = chi2 /
    (n_good - n_param) already accounts for the difference in degrees of freedom, 4
    components for the galaxy eigenspectra against 1 for a stellar template. That holds
    only if both used the same channels, which is why the two branches are given the
    same window, and it is what makes an absolute threshold unnecessary.
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
    # The scans go back to the parent rather than to disk from here: one file per
    # branch is one writer, and a worker cannot be that. Packed first, so what crosses
    # between processes is an array and not a list of thousands of dicts.
    scans = None
    if S["keep_scans"]:
        scans = (_pack_scan(r1) if r1 else None, _pack_scan(r2) if r2 else None)

    # The two branch winners face each other. The scans come back sorted by chi2, so
    # [0] is each branch's best.
    best = min([x[0] for x in (r1, r2) if x], key=lambda d: d["red_chi2"])

    A = np.full(N_COMPONENTS, np.nan)
    A[:len(best["A"])] = best["A"]
    # Both winning values are kept: the classification is decided by which of the two
    # is smaller, and without them nothing downstream can ask by how much it won.
    #
    # gal_z is the galaxy branch's own redshift, the lowest reduced chi2 over that
    # branch's whole scan, and not "z" above, which belongs to whichever branch won.
    # step 5 needs the galaxy value even for a source classified as a star.
    return t, scans, dict(id=t, nspax=int(np.median(S["nspax"][k])),
                   star_red_chi2=r1[0]["red_chi2"] if r1 else np.nan,
                   star_tpl=r1[0]["template"] if r1 else "",
                   gal_red_chi2=r2[0]["red_chi2"] if r2 else np.nan,
                   gal_tpl=r2[0]["template"] if r2 else "",
                   gal_z=(float(r2[int(np.argmin([x["red_chi2"] for x in r2]))]["z"])
                          if r2 else None),
                   **{**best, "A": A})


# =========================================================================
# the step
# =========================================================================

def write_classification(out_dir, best, ids=None, over=None, keep_intermediate=True):
    """Reduce the fit results to the list step 6 rebuilds the sources from.

    Returns (path, fields): the path of classification.npz, and the fields that went
    into it. With keep_intermediate the file is written; the fields are returned either
    way, because that is how step 6 receives them.

    The classification was already decided by the scan above and is not recomputed here
    -- the same decision written in two places drifts apart the moment one is edited,
    and that error is invisible in the output.

    ids  keep only these seg IDs; None means every ID in the best file. Leaving a
         source out only means step 5 has no template to subtract for it.
    over {id: z} overrides one source's redshift, for sensitivity tests only. The
         amplitude is re-solved at that z, the template's shape changing with z.

    The stellar library's name is stored alongside (see utils.build_templates).
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
            # The override re-solves from the galaxy scan, so it needs one --
            # --keep-scans has to have been on for the run being overridden.
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
    print("margin = ratio of the two models' reduced chi2; closer to 1 means less "
          "classification confidence")
    if keep_intermediate:
        print(f"saved -> {out}")
    return out, fields


def visible_cpus():
    """How many CPUs this process is allowed to run on.

    cpu_count() answers for the machine, the wrong number under an affinity mask or
    inside a cpuset; sched_getaffinity is right but Linux-only. process_cpu_count() is
    both; the fallbacks are for interpreters predating it.
    """
    if hasattr(os, "process_cpu_count"):            # 3.13+
        n = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):          # Linux
        n = len(os.sched_getaffinity(0))
    else:
        n = os.cpu_count()
    return n or 1


@blas_single_thread
def classify_sources(sky, spectra, work, cube, seg_path, out=None,
                     K=30, basis="svd", fit_window=(4600, 8000),
                     line_mask_iter=(1,), fix_s_at=0.0,
                     z_min=0.0, z_max=1.5, z_step=0.0001, star_dz=0.005,
                     num_workers=0, keep_scans=True,
                     id="all", full_range=False, sky_basis=False,
                     allow_partial=False, raw_mask=False, ids=None, z_override=(),
                     keep_intermediate=True):
    """Fit every source of `spectra`; return the last mask iteration's classification."""
    # One window from the config, handed to both branches: the two reduced chi2 are
    # comparable only over the same channels (see _scan_one).
    star_window = gal_window = tuple(fit_window)
    zmin, zmax, zstep = z_min, z_max, z_step

    over = {int(k): float(v) for k, v in (x.split("=") for x in z_override)}
    work = Path(work)
    STEP04 = Path(out) if out else work / "step04"
    print(f"workspace {work}")
    if full_range:
        star_window = gal_window = FULL_RANGE

    # z_override re-solves one source at a redshift taken from its galaxy scan, and
    # that scan is on disk or nowhere -- only its winning row comes back from a worker.
    if over and not (keep_intermediate and keep_scans):
        raise SystemExit("★ --z-override re-solves from the galaxy scans, which are "
                         "written only with --keep-scans on")
    if keep_intermediate:
        STEP04.mkdir(parents=True, exist_ok=True)

    # Row i of iter_line_mask is step 3's iteration i+1, so the number of rows is the
    # number of iterations there are to ask for -- a count known only here, a config
    # being written before step 3 has run.
    line_masks = load_line_masks(sky.iter_line_mask, cumulative=not raw_mask)
    for it in line_mask_iter:
        if not isinstance(it, (int, np.integer)) or not 1 <= it <= len(line_masks):
            raise SystemExit(f"★ line_mask_iter {it!r}: step3 produced "
                             f"{len(line_masks)} mask iterations, so the iterations "
                             f"available are 1-{len(line_masks)}")

    seg_ids, flux, var, nspax = (spectra.ids, spectra.flux, spectra.var, spectra.nspax)

    wl_air = sky.wavelength
    wl_vac = air_to_vacuum(wl_air)
    C_sky  = sky.continuum
    sky    = (np.vstack([C_sky, sky.basis[basis]]) if sky_basis else C_sky[None, :])

    # The mask is defined on air wavelengths, so the fitting window is cut on air
    # wavelengths too, and the two agree.
    win_star = (wl_air >= star_window[0]) & (wl_air < star_window[1])
    win_gal  = (wl_air >= gal_window[0])  & (wl_air < gal_window[1])

    z_exg  = np.arange(zmin, zmax + zstep / 2, zstep)
    z_star = np.arange(-star_dz, star_dz + zstep / 2, zstep)
    files = sorted(DWARF_DIR.glob("*.dat"))
    if not files:
        raise SystemExit(f"★ no .dat templates under {DWARF_DIR}")
    # A template's rest range has to cover the whole MUSE band: steps 5 and 6 evaluate
    # templates across all of it, and a channel NaN in the design matrix is dropped for
    # every spaxel and never solved again.
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
        # inside that domain would pass unseen. It belongs to the file, not to any one
        # redshift, so it is settled here.
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
    # four-component model whose linear combinations interpolate continuously between
    # types, so no list of discrete representative spectra is needed.
    gal_jobs = [("galaxy", "eigen", load_eigen_galaxy(EIGEN_GAL), z_exg)]
    # The same check, but fatal: there is only one galaxy job, and without it that
    # branch is empty and nothing can be classified.
    if not np.all(np.isfinite(gal_jobs[0][2].c)):
        raise SystemExit(f"★ {EIGEN_GAL.name} has a hole inside its own rest range")

    targets = seg_ids.tolist() if id == "all" else [int(id)]

    n_workers = num_workers or max(1, visible_cpus() // 3)
    n_workers = min(n_workers, len(targets))

    print(f"star  {star_window[0]:.0f}-{star_window[1]:.0f} A  "
          f"window {int(win_star.sum())} channels   {len(star_jobs)} stellar templates "
          f"x {z_star.size} z values")
    print(f"galaxy  {gal_window[0]:.0f}-{gal_window[1]:.0f} A  "
          f"window {int(win_gal.sum())} channels   galaxy eigenspectra x {z_exg.size} "
          f"z values")
    print("classification = lower reduced chi2 on the same channel set (no absolute "
          "threshold)")
    print("s is a free parameter" if fix_s_at is None else
          f"sky continuum fixed to {fix_s_at} x C_sky, subtracted first")
    print("source model = A x template" + ("  + sky-line basis" if sky_basis
                                           else "   (1 free parameter)"))
    print(f"spectra from {spectra.path.name}")
    print(f"{len(targets)} object(s)   {n_workers} workers   "
          f"mask iterations {list(line_mask_iter)}")

    # gal_z is the galaxy branch's own best redshift, which is not `z` -- that belongs
    # to whichever branch won. It is what decides which seg IDs join the main source
    # group, and it has to be a column here: recovering it from the galaxy scan means
    # opening a 15001-row file for one number, and the scans are not written by default.
    KEYS = ("id", "nspax", "group", "template", "z", "A", "s", "chi2",
            "red_chi2", "n_good", "src_min", "star_red_chi2", "star_tpl",
            "gal_red_chi2", "gal_tpl", "gal_z")
    outs = []
    classified = None

    # Each mask iteration is a separate set of results, a different channel set giving
    # chi2 that cannot be mixed with the others. Only the mask changes inside the loop;
    # the templates, spectra and z grids are prepared once.
    for it in line_mask_iter:
        line = line_masks[it - 1]
        fit_star, fit_gal = win_star & ~line, win_gal & ~line
        # A step04 directory holds one run, and its settings are in the meta.json beside
        # the products rather than encoded in their names. Several mask iterations are
        # several runs, so they get a directory each; the single iteration every config
        # asks for stays flat.
        run_dir = STEP04 if len(line_mask_iter) == 1 else STEP04 / f"mask_iter{it}"
        if keep_intermediate:
            run_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 112}")
        print(f"mask iter{it}{'(cumulative)' if not raw_mask else '(independent)'}: "
              f"flagged {int(line.sum()):,} / {line.size} channels"
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
        # One file per branch, written as the results arrive rather than collected
        # first: a whole pointing's galaxy scans do not have to be in memory at once,
        # and a source becomes a member of the file the moment it is done.
        with contextlib.ExitStack() as stack:
            scan_out = {}
            if keep_intermediate and keep_scans:
                for branch in ("star", "galaxy"):
                    f = run_dir / f"scans_{branch}.npz"
                    scan_out[branch] = stack.enter_context(
                        zipfile.ZipFile(f, "w", zipfile.ZIP_STORED))
                    scan_out[branch].writestr(
                        "group.npy", npy_bytes(np.array(branch)))
            pool = stack.enter_context(Pool(n_workers, initializer=_init_worker,
                                            initargs=(_SHARED,)))
            for t, scans, row in pool.imap(_scan_one, targets):
                if scans and scan_out:
                    for branch, packed in zip(("star", "galaxy"), scans):
                        if packed is None:
                            continue
                        rows, templates = packed
                        zf = scan_out[branch]
                        zf.writestr(f"id{t}.npy", npy_bytes(rows))
                        zf.writestr(f"id{t}_templates.npy", npy_bytes(np.array(templates)))
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

        # gal_z is None for a source the galaxy branch could not fit at all; as a column
        # that has to be NaN, or numpy stores the whole thing as objects.
        new = {k: np.array([np.nan if x[k] is None else x[k] for x in summary],
                           dtype=float) if k == "gal_z"
                  else np.array([x[k] for x in summary])
               for k in KEYS}

        out_f = run_dir / "source_fits.npz"
        # Merge rather than overwrite: re-running a single ID should update that row and
        # nothing else.
        if keep_intermediate and out_f.exists():
            old = np.load(out_f, allow_pickle=False)
            if set(old.files) != set(KEYS):
                print(f"  * {out_f.name} fields differ from current format, discarding "
                      f"entire file.\n    extra {sorted(set(old.files) - set(KEYS))}"
                      f"  missing {sorted(set(KEYS) - set(old.files))}")
            if set(old.files) == set(KEYS):
                keep = ~np.isin(old["id"], new["id"])
                if keep.any():
                    new = {k: np.concatenate([old[k][keep], new[k]]) for k in KEYS}
                    print(f"merged {int(keep.sum())} existing sources")
        o = np.argsort(new["id"])
        # The rows in the order they are written. Every value is already an array of the
        # dtype np.savez stores and np.load returns, so writing the file and reading it
        # back would hand on exactly this dict.
        best = {k: v[o] for k, v in new.items()}
        if keep_intermediate:
            np.savez(out_f, **best)
        cls_path, fields = write_classification(run_dir, best, ids, over,
                                                keep_intermediate)
        # The galaxy branch's redshift for every source it could fit, rebuilt each
        # iteration, so what is returned belongs to the same one as cls_path.
        galaxy_z = {int(x["id"]): x["gal_z"] for x in summary
                    if x["gal_z"] is not None}
        classified = Classification(cls_path, fields, galaxy_z)
        outs.append((it, out_f, summary))

    print(f"\n{'=' * 60}\ncross-iteration comparison")
    print(f"{'iter':>6}{'clean ch':>10}{'stars':>7}{'galaxies':>9}"
          f"{'star chi2/dof med':>20}{'neg-flux src':>13}")
    print("-" * 65)
    for it, out_f, summary in outs:
        ns = sum(1 for r in summary if r["group"] == "star")
        med = float(np.median([r["star_red_chi2"] for r in summary]))
        neg = sum(1 for r in summary if r["src_min"] < 0)
        print(f"{it:>6}{int((win_star & ~line_masks[it-1]).sum()):>10}"
              f"{ns:>7}{len(summary) - ns:>9}{med:>20.2f}{neg:>13}")
    if keep_intermediate:
        print("\n" + "\n".join(f"saved -> {o}" for _, o, _ in outs))
        # Every setting the products were made with, in one machine-readable place. The
        # filenames carry none of it -- a directory holds one run, and this is what says
        # which run that is.
        for it, o, summary in outs:
            write_meta(
                o.parent, "step4_classify_sources.py",
                cube=str(repo_path(cube)),
                seg=str(repo_path(seg_path)),
                spectra=str(repo_path(spectra.path)),
                basis=basis, K=K, sky_basis=sky_basis,
                fix_s_at=fix_s_at, fit_window=list(star_window),
                line_mask_iter=it, cumulative=not raw_mask,
                z_min=zmin, z_max=zmax, z_step=zstep, star_dz=star_dz,
                star_library=STAR_LIBRARY, keep_scans=keep_scans,
                n_sources=len(summary))
    return classified


def main():
    ap = argparse.ArgumentParser(
        description="fit templates to every source and classify it")
    ap.add_argument("--work", type=Path, required=True,
                    help="the run directory; step01, step02 and step03 are read from it")
    ap.add_argument("--cube", type=Path, default=None,
                    help="the sky-subtracted cube the spectra came from; recorded in "
                         "meta.json only")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory; defaults to <work>/step04")
    ap.add_argument("--basis", default="svd", choices=["pca", "svd"])
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--fit-window", type=float, nargs=2, default=[4600, 8000],
                    metavar=("LO", "HI"),
                    help="air wavelengths; the same window for both branches, which is "
                         "what makes their reduced chi2 comparable")
    ap.add_argument("--line-mask-iter", type=int, nargs="+", default=[1],
                    help="which sky-line mask iterations to fit; more than one puts "
                         "each in its own step04/mask_iter{N}")
    ap.add_argument("--fix-s-at", type=float, default=0.0,
                    help="hold the sky continuum at this multiple of C_sky and "
                         "subtract it first; omit to leave s free")
    ap.add_argument("--z-min", type=float, default=0.0)
    ap.add_argument("--z-max", type=float, default=1.5)
    ap.add_argument("--z-step", type=float, default=0.0001)
    ap.add_argument("--star-dz", type=float, default=0.005,
                    help="half-width of the redshift scan for stars")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="0 = one third of the visible CPUs")
    ap.add_argument("--keep-scans", action="store_true", default=True)
    ap.add_argument("--no-keep-scans", dest="keep_scans", action="store_false")
    ap.add_argument("--id", default="all", help="one seg ID, or all")
    ap.add_argument("--full-range", action="store_true",
                    help=f"fit over {FULL_RANGE[0]:.0f}-{FULL_RANGE[1]:.0f} A instead "
                         f"of --fit-window")
    ap.add_argument("--sky-basis", action="store_true",
                    help="give the source fit the sky-line basis as well as C_sky")
    ap.add_argument("--allow-partial", action="store_true",
                    help="keep candidates that do not cover the whole window; chi2 is "
                         "a sum, so this lets the scan run to wherever the template "
                         "covers fewest channels")
    ap.add_argument("--raw-mask", action="store_true",
                    help="use each iteration's own mask instead of the cumulative one")
    ap.add_argument("--ids", type=int, nargs="+", default=None,
                    help="keep only these seg IDs in classification.npz")
    ap.add_argument("--z-override", nargs="+", default=[], metavar="ID=Z",
                    help="re-solve one source at this redshift; needs --keep-scans")
    args = ap.parse_args()

    seg_path = args.work / "step01/segmentation_input.fits"
    classify_sources(
        step_io.sky(args.work, args.basis, args.K),
        step_io.spectra(args.work),
        args.work, args.cube or "(not given)", seg_path, out=args.out,
        K=args.K, basis=args.basis, fit_window=tuple(args.fit_window),
        line_mask_iter=tuple(args.line_mask_iter), fix_s_at=args.fix_s_at,
        z_min=args.z_min, z_max=args.z_max, z_step=args.z_step,
        star_dz=args.star_dz, num_workers=args.num_workers,
        keep_scans=args.keep_scans, id=args.id, full_range=args.full_range,
        sky_basis=args.sky_basis, allow_partial=args.allow_partial,
        raw_mask=args.raw_mask, ids=args.ids, z_override=tuple(args.z_override))


if __name__ == "__main__":
    main()
