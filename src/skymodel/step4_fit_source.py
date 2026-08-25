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

The specification comes from reminder.txt: use the line mask, leave out the line
channels, and fit stars and galaxies each inside a fixed wavelength window. The
window values themselves are the defaults of --star-window / --gal-window and are
not repeated here -- the same numbers written in two places drift apart eventually.

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
same statistic, so comparing them means nothing. main() refuses a mismatched pair.

    conda run -n astro python src/skymodel/step4_fit_source.py --id all -K 54 \\
        --spec-dir results/skymodel/ne_pointing/step02_eso --s-fix 0.0 \\
        --star-window 4700 8000 --gal-window 4700 8000 --line-mask-iter 1
"""
import os

# The BLAS thread count has to be set before numpy is imported -- the library reads
# it once, when it loads.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from templates import (DWARF_DIR, STAR_LIBRARY,
                       load_ascii_template, load_eigen_galaxy, redshift_to_grid,
                       air_to_vacuum)
from utils import load_line_masks

ROOT      = Path(__file__).resolve().parents[2]
# These three have to be module-level globals: multiprocessing forks its workers, and
# a worker sees the module globals as they were at the fork, not main()'s locals.
# main() assigns them from --work, before the Pool is opened.
STEP02B = STEP03 = STEP04 = None
EIGEN_GAL = ROOT / "data/eigen_galaxy_Bolton2012.fits"

N_SRC    = 4                # fixed width of the A column: 4 eigenspectra, and a star
                            # uses only column 0

# The wavelength window of each branch (A, air). Editing here changes the default;
# --star-window / --gal-window override it without touching the code. The window is
# encoded into the output tag, so results from different windows sit side by side
# instead of overwriting each other.
#
# All three share a lower bound, so the only thing separating the windows is how far
# right they reach. 4600 is not "from the beginning" -- 13 pointings start between
# 4599.6 and 4600.3 A, but p14 starts at 4749.83 A with only 3681 channels. The real
# start is whatever each cube's first channel is; comparing reduced chi2 within one
# pointing is unaffected, because both branches use the same channels.
#
# The two windows have to be equal (see the paragraph in the module docstring), so
# reminder.txt's 4600-6000 / 4600-7000 pair cannot be used literally -- the n_good in
# the denominator would differ and the two reduced chi2 would not be comparable. The
# upper bound taken is the union of the two, 8000; the lower stays reminder.txt's 4600.
STAR_WINDOW = (4600.0, 8000.0)      # fitting window for the stellar templates
GAL_WINDOW  = (4600.0, 8000.0)      # for the galaxy eigenspectra; must equal the above
FULL_RANGE  = (4600.0, 9400.0)      # the whole MUSE range, kept as a control

_SHARED = {}


def make_tag(basis, K, s_fix, star_window, gal_window, sky_basis, line_iter,
             cumulative=True, aperture=False, suffix=""):
    """The output filename. Every setting that changes the result is encoded into it,
    so a re-run cannot quietly overwrite the previous one.

    The windows and the mask iteration are in there because they decide which channels
    enter chi2: results from different settings are different scientific products and
    have to coexist. The diagnostic scripts call this same function, so the two always
    agree on the naming; written out twice, changing one and forgetting the other
    turns into "reading the wrong file".
    """
    base = f"{basis}_K{K}" if sky_basis else "nobasis"
    return (f"{base}_s{'free' if s_fix is None else s_fix}"
            f"_{star_window[0]:.0f}-{star_window[1]:.0f}"
            f"_{gal_window[0]:.0f}-{gal_window[1]:.0f}"
            f"_L{line_iter}{'cum' if cumulative else 'raw'}"
            + ("_ap" if aperture else "") + suffix)


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


def scan_object(flux, var, sky, jobs, lam_muse, fit, s_fix=None,
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

    When s_fix is given, s*C_sky is subtracted from the data first and s stops being a
    free parameter (one fewer). The point is to break the degeneracy between A*T and
    s*C_sky -- the data cannot separate them, and left free the template absorbs the
    sky continuum.
    """
    base = (fit & np.isfinite(flux) & np.isfinite(var) & (var > 0)
            & np.all(np.isfinite(sky), axis=0))
    sig  = np.sqrt(np.where(var > 0, var, 1.0))
    n_full = int(base.sum())            # channels the data offers -- the ceiling every
                                        # candidate shares

    if s_fix is None:
        sky_free, y, s_free = sky, flux, True
    else:
        sky_free, y, s_free = sky[1:], flux - s_fix * sky[0], False

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

        for z in z_grid:
            T = redshift_to_grid(spline, z, lam_muse)
            if T.ndim == 1:
                T = T[:, None]
            good = base & np.all(np.isfinite(T), axis=1)
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
            if not allow_partial and n < n_full:
                continue

            M = np.empty((n, p))
            M[:, :n_comp] = T[good] / sig[good][:, None]
            M[:, n_comp:] = skyw[good]

            fitres = lsq_linear(M, yw[good], bounds=(lb, ub), method="bvls")
            theta  = fitres.x
            chi2   = 2.0 * fitres.cost

            # Whether the source spectrum goes negative is checked over the whole
            # range, not just inside the window: negative flux is a physical problem
            # with the model, and it does not stop existing because those channels
            # were left out of chi2.
            src      = np.nan_to_num(T, nan=0.0) @ theta[:n_comp]
            m_all    = src + theta[n_comp:] @ sky_free
            chi2_all = float((((y - m_all) / sig) ** 2)[base].sum())
            ok       = np.isfinite(flux) & np.all(np.isfinite(T), axis=1)

            results.append(dict(group=group, template=name, z=float(z),
                    A=theta[:n_comp], s=theta[n_comp] if s_free else s_fix,
                    chi2=chi2, chi2_all=chi2_all, red_chi2=chi2 / (n - p),
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
             chi2_all=np.array([x["chi2_all"] for x in results]),
             red_chi2=np.array([x["red_chi2"] for x in results]),
             n_good=np.array([x["n_good"] for x in results]),
             src_min=np.array([x["src_min"] for x in results]))


def _scan_one(t):
    """Fit one source in a single stage: the stellar templates and the galaxy
    eigenspectra compete on the same channels.

    The shared data is inherited through the fork; nothing is pickled.

    Both branches are scanned and compared directly, so no absolute threshold is
    needed anywhere.

    Why the two reduced chi2 can be compared
        reduced chi2 = chi2 / (n_good - n_param), and that denominator already
        accounts for the difference in degrees of freedom -- 4 components for the
        galaxy eigenspectra against 1 for a stellar template. It only works if both
        used the **same channels** (the same n_good), which is why main() checks that
        the two windows agree.
    """
    S = _SHARED
    k = int(np.flatnonzero(S["ids"] == t)[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        f = S["flux"][k] / S["nspax"][k]
        v = S["var"][k]  / S["nspax"][k] ** 2

    r1 = scan_object(f, v, S["sky"], S["star_jobs"], S["wl_vac"],
                     S["fit_star"], s_fix=S["s_fix"],
                     allow_partial=S["allow_partial"])
    r2 = scan_object(f, v, S["sky"], S["gal_jobs"], S["wl_vac"],
                     S["fit_gal"], s_fix=S["s_fix"],
                     allow_partial=S["allow_partial"])
    if not r1 and not r2:
        return t, None
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
    return t, dict(id=t, nspax=int(np.median(S["nspax"][k])),
                   star_red_chi2=r1[0]["red_chi2"] if r1 else np.nan,
                   star_tpl=r1[0]["template"] if r1 else "",
                   gal_red_chi2=r2[0]["red_chi2"] if r2 else np.nan,
                   gal_tpl=r2[0]["template"] if r2 else "",
                   **{**best, "A": A})


def write_classification(out_dir, tag, best, ids=None, over=None):
    """Reduce the fit results to the list step5 reads.

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
    np.savez(out,
             id=np.array([r["id"] for r in rows]),
             group=np.array([r["group"] for r in rows]),
             template=np.array([r["template"] for r in rows]),
             z=np.array([r["z"] for r in rows]),
             A=np.vstack([r["A"] for r in rows]),
             star_library=np.array(STAR_LIBRARY))
    ns = sum(1 for r in rows if r["group"] == "star")
    print(f"\n{len(rows)} sources: {ns} stars / {len(rows) - ns} galaxies")
    print("margin = ratio of the two models' reduced chi2; closer to 1 means less classification confidence")
    print(f"saved -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="single-stage source template fitting (fixed window + sky-line mask)")
    ap.add_argument("--id",    default="all",           help="segmentation ID, or all")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, required=True,
                    help="number of sky-line basis vectors. Required -- all three steps must use the same K; separate defaults would silently read a different basis set")
    ap.add_argument("--star-window", type=float, nargs=2, default=STAR_WINDOW,
                    metavar=("LO", "HI"), help="stellar template fitting window (A, air); must match --gal-window")
    ap.add_argument("--gal-window", type=float, nargs=2, default=GAL_WINDOW,
                    metavar=("LO", "HI"), help="galaxy eigenspectrum fitting window (A, air); must match --star-window")
    ap.add_argument("--full-range", action="store_true",
                    help=f"use full MUSE range {FULL_RANGE[0]:.0f}-{FULL_RANGE[1]:.0f} A "
                         "as a control. Note: stellar templates extend to ~9200 A at rest; "
                         "under full range different templates cover different channel counts, "
                         "so chi2 comparisons are affected by channel count.")
    ap.add_argument("--line-mask-iter", type=int, nargs="+", default=[1, 2, 3, 4],
                    help="which step3 sky-line mask iteration(s) to use; can specify "
                         "multiple, each produces a separate result. Iter 1 is the loosest "
                         "(only the strongest lines); higher iterations mask more channels. "
                         "Default: all four.")
    ap.add_argument("--sky-basis", action="store_true",
                    help="include sky-line basis in the source fit. Off by default: "
                         "sky-line channels are already excluded from chi2, the basis has "
                         "almost no power in the remaining channels, and those K weakly "
                         "constrained parameters only absorb source signal. Without it the "
                         "source model has only 1 free parameter A.")
    ap.add_argument("--zmin",  type=float, default=0.0)
    ap.add_argument("--zmax",  type=float, default=1.5)
    ap.add_argument("--zstep", type=float, default=1e-4)
    ap.add_argument("--star-dz", type=float, default=0.005,
                    help="half-width of z scan for stars (+-1500 km/s). Resolved point "
                         "sources must be Milky Way foreground stars with no Hubble flow, "
                         "only peculiar velocity.")
    ap.add_argument("--aperture", action="store_true",
                    help="read circular aperture spectra from step02b/ (produced by "
                         "experiments/step2b_aperture.py) instead of segmentation footprint "
                         "from step02/")
    ap.add_argument("--allow-partial", action="store_true",
                    help="allow candidates where the template covers only part of the "
                         "window. Off by default -- chi2 is a sum, so candidates with fewer "
                         "channels are inherently smaller, biasing the scan toward z values "
                         "where the template barely covers the window. Enable only when you "
                         "know what you are doing.")
    ap.add_argument("--spec-dir", default=None,
                    help="directory of source spectra, overrides --aperture. Classification "
                         "requires sky-subtracted spectra, e.g. .../step02_ours (our subtraction) "
                         "or step02_eso (ESO subtraction). Directory name is encoded in the "
                         "output tag so different sources are stored separately")
    ap.add_argument("--raw-mask", action="store_true",
                    help="use each mask iteration independently without accumulation. "
                         "Default is cumulative -- step3's raw iterations are not strictly "
                         "nested (some channels drop out); cumulative mode produces a clean "
                         "'progressively more masked' sequence.")
    ap.add_argument("--s-fix", type=float, default=1.0)
    ap.add_argument("--s-free", action="store_true")
    ap.add_argument("--ids", type=int, nargs="+", default=None,
                    help="classification file includes only these seg IDs. Omit = all fitted sources")
    ap.add_argument("--z-override", nargs="*", default=[], metavar="ID=Z",
                    help="override a source's redshift to a specified value, for sensitivity "
                         "testing only -- production records should not contain manually set "
                         "values. Amplitude is re-solved at the overridden z, not carried over")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--work", required=True,
                    help="working directory for this cube (contains step02/step03/step04)")
    args = ap.parse_args()
    over = {int(k): float(v) for k, v in (x.split("=") for x in args.z_override)}
    work    = Path(args.work)
    # These three have to be module-level globals: _scan_one runs inside a
    # multiprocessing worker and cannot see main()'s locals -- a forked worker sees
    # the module globals as they stood at the fork.
    global STEP02B, STEP03, STEP04
    STEP02B = work / "step02b"
    STEP03  = work / "step03"
    STEP04 = work / "step04"
    print(f"workspace {work}")
    s_fix = None if args.s_free else args.s_fix
    if args.full_range:
        args.star_window = args.gal_window = FULL_RANGE

    STEP04.mkdir(parents=True, exist_ok=True)

    # Where the source spectra come from has to be said out loud. There is no usable
    # default: classifying from spectra that still contain the sky produces output
    # that looks entirely normal, with every source's template and redshift wrong.
    if not args.spec_dir and not args.aperture:
        raise SystemExit(f"requires --spec-dir (e.g. {work}/step02_eso) or --aperture")
    src = Path(args.spec_dir) if args.spec_dir else STEP02B
    # A different spectrum source is a different scientific product, so the tag has to
    # separate them; one workspace can hold several sources (step02_eso and
    # step02_ours, say), and a name that does not encode the source silently
    # overwrites the previous run. The default source, step02, gets no suffix -- the
    # suffix marks a departure from the default, and the default needs no marking.
    suffix = make_suffix(src.name)
    ids   = np.load(src / "object_ids.npy")
    flux  = np.load(src / "object_flux.npy")
    var   = np.load(src / "object_var.npy")
    nspax = np.load(src / "object_nspax.npy")

    wl_air = np.load(STEP03 / "wavelength.npy")
    wl_vac = air_to_vacuum(wl_air)
    C_sky  = np.load(STEP03 / "sky_continuum.npy")
    B      = np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")
    sky    = np.vstack([C_sky, B]) if args.sky_basis else C_sky[None, :]

    # The sky-line mask. Row i of iter_line_mask is step3's iteration i+1. The mask is
    # defined on air wavelengths, so the window is cut on air wavelengths too, and the
    # two agree.
    line_masks = load_line_masks(STEP03 / "iter_line_mask.npy",
                                 cumulative=not args.raw_mask)
    win_star = (wl_air >= args.star_window[0]) & (wl_air < args.star_window[1])
    win_gal  = (wl_air >= args.gal_window[0])  & (wl_air < args.gal_window[1])

    z_exg  = np.arange(args.zmin, args.zmax + args.zstep / 2, args.zstep)
    z_star = np.arange(-args.star_dz, args.star_dz + args.zstep / 2, args.zstep)
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

    targets = ids.tolist() if args.id == "all" else [int(args.id)]
    if args.id != "all" and targets[0] not in ids:
        raise SystemExit(f"ID {targets[0]} does not exist. Available: {ids.min()}-{ids.max()}")

    n_workers = args.num_workers or max(1, len(os.sched_getaffinity(0)) // 3)
    n_workers = min(n_workers, len(targets))

    # For the two branches' reduced chi2 to be comparable they must be computed on the
    # same channels. Different windows give different n_good, hence different
    # denominators, and comparing them means nothing.
    if tuple(args.star_window) != tuple(args.gal_window):
        raise SystemExit(
            f"star window {args.star_window} and galaxy window {args.gal_window} differ. "
            "Single-stage fitting directly compares their reduced chi2, so the channel set "
            "must be identical -- set --star-window and --gal-window to the same range.")

    print(f"star  {args.star_window[0]:.0f}-{args.star_window[1]:.0f} A  "
          f"window {int(win_star.sum())} channels   {len(star_jobs)} stellar templates x "
          f"{z_star.size} z values")
    print(f"galaxy  {args.gal_window[0]:.0f}-{args.gal_window[1]:.0f} A  "
          f"window {int(win_gal.sum())} channels   galaxy eigenspectra x {z_exg.size} z values")
    print("classification = lower reduced chi2 on the same channel set (no absolute threshold)")
    print("s is a free parameter" if s_fix is None else f"sky continuum fixed to {s_fix} x C_sky, subtracted first")
    print("source model = A x template" + ("  + sky-line basis" if args.sky_basis
                                          else "   (1 free parameter)"))
    print(f"spectra from {src.name}"
          + ("  (circular aperture r=6 px)" if args.aperture else "  (segmentation footprint)"))
    print(f"{len(targets)} object(s)   {n_workers} workers   "
          f"mask iterations {args.line_mask_iter}")

    KEYS = ("id", "nspax", "group", "template", "z", "A", "s", "chi2", "chi2_all",
            "red_chi2", "n_good", "src_min", "star_red_chi2", "star_tpl",
            "gal_red_chi2", "gal_tpl")
    outs = []

    # Each mask iteration is a separate set of results: a different channel set gives
    # different chi2, and the two cannot be mixed. The static data (templates, spectra,
    # z grids) is prepared once, and only the mask changes inside the loop.
    for it in args.line_mask_iter:
        line = line_masks[it - 1]
        fit_star, fit_gal = win_star & ~line, win_gal & ~line
        tag = make_tag(args.basis, args.K, s_fix, args.star_window,
                       args.gal_window, args.sky_basis, it, not args.raw_mask,
                       args.aperture, suffix)

        print(f"\n{'=' * 112}")
        print(f"mask iter{it}{'(cumulative)' if not args.raw_mask else '(independent)'}: flagged {int(line.sum()):,} / {line.size} channels"
              f" ({100 * line.mean():.1f}%)   "
              f"clean channels for fitting {int(fit_star.sum())}")
        print(f"{'=' * 112}")
        print(f"{'ID':>5}{'nspax':>8}{'group':>8}{'tpl':>7}{'z':>10}{'A':>12}"
              f"{'n':>7}{'chi2':>14}{'chi2/dof':>10}{'star chi2/dof':>15}"
              f"{'gal chi2/dof':>14}"
              f"{'src_min':>10}")
        print("-" * 112)

        _SHARED.update(ids=ids, flux=flux, var=var, nspax=nspax, sky=sky,
                       star_jobs=star_jobs, gal_jobs=gal_jobs, wl_vac=wl_vac,
                       fit_star=fit_star, fit_gal=fit_gal,
                       tag=tag, s_fix=s_fix,
                       allow_partial=args.allow_partial)

        summary = []
        with Pool(n_workers) as pool:
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

        # Merge into what is already there rather than overwriting: re-running a
        # single ID should update that row and nothing else.
        out = STEP04 / f"best_{tag}.npz"
        if out.exists():
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
        np.savez(out, **{k: v[o] for k, v in new.items()})
        write_classification(STEP04, tag, np.load(out), args.ids, over)
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
    print("\n" + "\n".join(f"saved -> {o}" for _, o, _ in outs))


if __name__ == "__main__":
    main()
