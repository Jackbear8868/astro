"""One curve per pointing, overlaid: the sky the model learned, or the galaxy's halo.

    pointing_curves.py --curve continuum --pointings p01 p02 p03
    pointing_curves.py --curve halo --frac 0.5 --xlim 6500 6900 --normalise

--curve continuum draws the final sky continuum of every pointing. The sky model is
D = s * C_sky(lambda) + sum_k c_k * L_k(lambda): one continuum shape per pointing,
scaled per spaxel by s. Drawing them together shows how far the pointings agree on
that shape and how much they differ in level.

--curve halo draws Haro 11's extended light instead. Each pointing sees the same
galaxy through a different dither, exposure depth and part of the field of view, so
"is the extended light the same thing in all of them" is a question only a figure like
this can answer. The halo is the faintest fraction of the main source group by
white-light surface brightness -- an isophotal zone, the same construction
zone_spectra uses for its outermost layer. Not a box at a fixed position, since the
galaxy sits somewhere different in every pointing; not a ring of fixed radius either,
since a merger is neither round nor centred and a ring would cross knots and outskirts
at one radius.

What neither curve may do is compare wavelengths that are not the same wavelength. The
grids differ between pointings in length and in start, so stacking the arrays by index
would silently equate different channels: each curve is drawn against its own array
and nothing is resampled. The halo zones are not interchangeable either, differing in
area and in which part of the galaxy they cover, so the printed table gives each one's
spaxel count and median surface brightness.

This is what continuum_compare.py and halo_compare.py were. One figure of one curve
per pointing, one legend, one colour rule; they differed only in where the curve comes
from -- a continuum step 3 already wrote, or an average over a cube that takes minutes
and is cached.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL, ROOT, data_hdu  # noqa: E402
from products import Run, latest_run  # noqa: E402
from spectra import LINES, Z_HARO, panel_ylim  # noqa: E402
from utils import DZ_MAX, main_source_group  # noqa: E402
from zones import CHUNK, EDGE_MARGIN  # noqa: E402

# Each curve lands with the figures of the thing it measures: the sky continuum beside
# the rest of the sky basis, the halo beside the other zone figures.
FIGURES = {"continuum": EVAL / "sky_basis", "halo": EVAL / "halo"}
CACHE = FIGURES["halo"] / "cache"

# A qualitative map: the pointing number labels a mosaic tile and a dither, not a
# quantity, so a sequential map would imply an order that is not there.
COLOURS = plt.get_cmap("tab20").colors


def load(work):
    """(wavelength, continuum) for one pointing, or None if step3 has not run."""
    if not (Path(work) / "step03/sky_continuum.npy").exists():
        return None
    run = Run(work)
    return run.wl, run.continuum


def halo_mask(seg, white, valid, main, frac, band=None):
    """Which spaxels count as halo, away from the field edge.

    Two constructions answering different questions. frac takes the faintest `frac` of
    this pointing's main source group: self-scaling, so it always finds the outskirts
    of whatever the group is, but the zones then sit at different surface brightnesses
    between pointings. band takes an absolute surface-brightness window (lo, hi) in
    white-light units, the same numbers everywhere, so the zones are genuinely one
    isophote; the price is that a pointing may contribute few spaxels, and the count
    has to be read alongside the curve.
    """
    ok = valid & (ndimage.distance_transform_edt(valid) > EDGE_MARGIN)
    inside = main & ok
    if band is not None:
        return inside & (white >= band[0]) & (white <= band[1])
    v = white[inside]
    if v.size == 0:
        return inside
    cut = np.percentile(v, 100 * frac)
    return inside & (white <= cut)


def mean_spectrum(cube, mask, nz):
    """Mean over the masked spaxels, read in wavelength chunks."""
    idx = np.flatnonzero(mask.ravel())
    out = np.full(nz, np.nan)
    with fits.open(cube, memmap=True) as h:
        hdu = data_hdu(h)
        if hdu.data.shape[0] != nz:
            raise SystemExit(f"{cube.name}: {hdu.data.shape[0]} channels, "
                             f"wavelength.npy has {nz}")
        for c0 in range(0, nz, CHUNK):
            c1 = min(c0 + CHUNK, nz)
            block = np.asarray(hdu.data[c0:c1], np.float32).reshape(c1 - c0, -1)
            with np.errstate(invalid="ignore"):
                out[c0:c1] = np.nanmean(block[:, idx], axis=1)
    return out


def cont_level(wl, s, band):
    """A curve's continuum level: the median over a line-free band.

    The table reports it and --normalise divides by it, so the two must be the one
    number -- a level measured twice over different channels is a level nobody can
    check the figure against.
    """
    b = (wl > band[0]) & (wl < band[1])
    return float(np.median(s[b])) if b.any() else np.nan


def continuum_curves(args):
    """(name, wavelength, continuum) per pointing, and the table that goes with it."""
    got = []
    for name in args.pointings:
        r = load(ROOT / args.root / name)
        if r is None:
            print(f"  skip {name}: no step03/sky_continuum.npy")
            continue
        got.append((name, *r))
    if not got:
        raise SystemExit("no pointing has a continuum to plot")

    print(f"{'':>5}{'median':>9}{'nz':>6}{'range [A]':>20}")
    print("-" * 40)
    for name, w, c in got:
        print(f"{name:>5}{np.median(c):>9.2f}{c.size:>6}{f'{w.min():.1f}-{w.max():.1f}':>20}")
    return got


def halo_curves(args):
    """(name, wavelength, spectrum, spaxels, median white, run) per pointing.

    The zones differ in area and in which part of the galaxy they cover, so the table
    reports the spaxel count and the median surface brightness beside each curve.
    """
    got = []
    for name in args.pointings:
        W = ROOT / args.root / name
        run = latest_run(W, "sky_subtracted.fits", "step06", args.run)
        if run is None:
            print(f"  skip {name}: no sky_subtracted.fits under step05/step06")
            continue
        pointing = Run(W)
        seg, white, valid = pointing.seg, pointing.white, pointing.valid
        main_, ids, _ = main_source_group(seg, np.where(valid, white, np.nan),
                                          W / "step04" if args.step04 else None, DZ_MAX)
        m = halo_mask(seg, white, valid, main_, args.frac, args.band)
        wl = pointing.wl
        if int(m.sum()) == 0:
            print(f"  skip {name}: the selection contains no spaxel")
            continue
        # A cube read takes minutes and the figure is redrawn often, so spectra are
        # cached. The key carries every input that changes the numbers -- a cache read
        # back for a different selection is worse than no cache.
        key = (f"{name}_{run.name}_"
               + (f"band{args.band[0]:g}-{args.band[1]:g}" if args.band
                  else f"frac{args.frac:g}")
               + ("_z" if args.step04 else ""))
        cf = CACHE / f"{key}.npz"
        if cf.exists() and not args.refresh:
            d = np.load(cf)
            spec, npx = d["spec"], int(d["npx"])
            print(f"  {name}: halo {npx:,} px   cached")
        else:
            print(f"  {name}: main group {int(main_.sum()):,} px -> "
                  f"halo {int(m.sum()):,} px   reading {run.name} ...", flush=True)
            spec = mean_spectrum(run / "sky_subtracted.fits", m, wl.size)
            npx = int(m.sum())
            CACHE.mkdir(parents=True, exist_ok=True)
            np.savez(cf, spec=spec, npx=npx, wl=wl)
        got.append((name, wl, spec, npx, float(np.median(white[m])), run.name))
    if not got:
        raise SystemExit("no pointing has a halo spectrum to draw")

    print(f"\n    {'':>5}{'spaxels':>9}{'median white':>14}{'continuum':>11}"
          f"{'Ha peak':>10}{'channels':>10}   run")
    for name, wl, s, n, mw, run in got:
        obs = 6562.8 * (1 + Z_HARO)
        w = np.abs(wl - obs) < 6
        loc = np.median(s[(np.abs(wl - obs) > 12) & (np.abs(wl - obs) < 30)])
        ha = float(s[w].max() - loc) if w.any() else np.nan
        print(f"    {name:>5}{n:>9,}{mw:>14.3f}{cont_level(wl, s, args.cont):>11.3f}{ha:>10.2f}"
              f"{wl.size:>10}   {run}")
    return got


def draw_continuum(got, figsize):
    """Every pointing's sky continuum on one axis, and where the figure goes."""
    fig, a = plt.subplots(figsize=figsize)
    for i, (name, w, c) in enumerate(got):
        a.plot(w, c, lw=1.0, color=COLOURS[i % len(COLOURS)], label=name)
    a.set_xlim(min(w.min() for _, w, _ in got), max(w.max() for _, w, _ in got))
    a.set_xlabel("wavelength [$\\AA$]")
    a.set_ylabel("flux")
    a.legend(fontsize=11, loc="upper left", bbox_to_anchor=(1.005, 1.0),
             borderaxespad=0, frameon=False)
    return fig, FIGURES["continuum"] / "continuum_compare.png"


def draw_halo(args, got, figsize):
    """Every pointing's halo spectrum on one axis, and where the figure goes."""
    fig, ax = plt.subplots(figsize=figsize)
    if not args.no_lines:
        for _, lam in LINES:
            ax.axvline(lam * (1 + Z_HARO), lw=0.6, color="0.85", zorder=0)
    ax.axhline(0, lw=0.8, color="0.55")
    drawn = []
    for i, (name, wl, s, n, mw, _) in enumerate(got):
        y = s / cont_level(wl, s, args.cont) if args.normalise else s
        if args.smooth > 1:
            y = np.convolve(y, np.ones(args.smooth) / args.smooth, mode="same")
        ax.plot(wl, y, lw=0.7, color=COLOURS[i % len(COLOURS)], label=f"{name}  ({n:,} px)")
        drawn.append(y)
    ax.set_xlim(*(args.xlim if args.xlim else
                  (min(w.min() for _, w, _, _, _, _ in got),
                   max(w.max() for _, w, _, _, _, _ in got))))
    if args.ylim:
        ax.set_ylim(*args.ylim)
    else:
        ax.set_ylim(*panel_ylim(np.concatenate(drawn)))
    ax.set_xlabel("wavelength [$\\AA$]")
    ax.set_ylabel("flux / continuum" if args.normalise else "flux")
    ax.legend(fontsize=10, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              borderaxespad=0, frameon=False)
    ax.grid(alpha=0.2)

    sel = (f"band{args.band[0]:g}-{args.band[1]:g}" if args.band else f"f{args.frac:g}")
    stem = (f"halo_compare_{sel}" + ("_z" if args.step04 else "")
            + ("_norm" if args.normalise else "")
            + (f"_{args.xlim[0]:.0f}-{args.xlim[1]:.0f}" if args.xlim else ""))
    return fig, FIGURES["halo"] / f"{stem}.png"


def main():
    ap = argparse.ArgumentParser(
        description="One curve per pointing, overlaid: the sky continuum, or Haro 11's halo")
    ap.add_argument("--curve", choices=["continuum", "halo"], default="continuum",
                    help="the sky continuum step 3 fitted, or the mean spectrum of the "
                         "galaxy's faintest layer in the sky-subtracted cube")
    ap.add_argument("--pointings", nargs="+", default=[f"p{i:02d}" for i in range(1, 15)],
                    help="pointing directory names under results/skymodel")
    ap.add_argument("--root", default="results/skymodel")

    ap.add_argument("--run", default=None,
                    help="halo only: glob naming the run under step05 that holds "
                         "sky_subtracted.fits")
    ap.add_argument("--frac", type=float, default=0.25,
                    help="halo only: fraction of the main source group counted as "
                         "halo, faintest first")
    ap.add_argument("--band", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="halo only: absolute white-light surface brightness window "
                         "used instead of --frac. The same isophote in every pointing, "
                         "which is what makes the curves comparable; a pointing may "
                         "contribute few spaxels, so read the count with the curve")
    ap.add_argument("--refresh", action="store_true",
                    help="halo only: re-read the cubes even if a cache from an "
                         "identical selection exists")
    ap.add_argument("--step04", action="store_true",
                    help="halo only: require members of the main group to share the "
                         "main source's redshift; without it adjacency alone decides")
    ap.add_argument("--normalise", action="store_true",
                    help="halo only: divide each curve by its own continuum median, "
                         "for comparing shape when the pointings differ in depth")
    ap.add_argument("--cont", type=float, nargs=2, metavar=("LO", "HI"), default=(5700, 5750),
                    help="halo only: line-free band used for the continuum level")
    ap.add_argument("--xlim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="halo only")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="halo only")
    ap.add_argument("--smooth", type=int, default=1, help="halo only")
    ap.add_argument("--no-lines", action="store_true",
                    help="halo only: drop the redshifted line markers")

    ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=None,
                    help="default 22 7 for continuum, 22 8 for halo")
    ap.add_argument("--dpi", type=int, default=None,
                    help="default 200 for continuum, 180 for halo")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # Each curve keeps the size and resolution it was read at. One shared default
    # would be a change to whichever figure did not have it, made for no reason to do
    # with the figure.
    continuum = args.curve == "continuum"
    figsize = tuple(args.figsize) if args.figsize else ((22, 7) if continuum else (22, 8))
    dpi = args.dpi if args.dpi else (200 if continuum else 180)

    if continuum:
        fig, default = draw_continuum(continuum_curves(args), figsize)
    else:
        fig, default = draw_halo(args, halo_curves(args), figsize)

    # The filenames are the ones the two programs wrote before they became one: run.py
    # dates a figure set by the name it lands under, and the poster versions of these
    # figures carry the same names.
    out = Path(args.out) if args.out else default
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
