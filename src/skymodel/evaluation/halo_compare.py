"""Haro 11's extended light, one spectrum per pointing, on one figure.

Each pointing sees the same galaxy through a different dither, a different exposure
depth and a different part of the field of view, so "is the extended light the same
thing in all of them" is a question only a figure like this can answer.

What counts as the halo
-----------------------
The faintest fraction of the main source group, by white-light surface brightness --
an isophotal zone, the same construction halo_spectra uses for its outermost layer.
Not a box at a fixed position: the galaxy sits somewhere different in every pointing,
and a box would sample a different part of it each time. Not a ring of fixed radius
either: Haro 11 is a merger and is neither round nor centred on anything, so a ring
crosses the bright knots and the faint outskirts at the same radius.

What is not comparable
----------------------
The wavelength grids. Pointings have 3801 or 3802 channels, their starts differ by up
to 0.7 A, and p14 begins about 150 A redward of the rest with 120 fewer channels.
Nothing is resampled -- each curve is drawn against its own wavelength array, so no
two channels are silently equated.

The zone sizes. Each pointing contributes its own faintest quartile, and how much of
the galaxy is inside the field differs (in some pointings Haro 11 runs off the edge),
so the zones are neither the same area nor the same part of the galaxy. The printed
table gives the spaxel count and the median surface brightness of each, which is what
makes the differences readable rather than invisible.

    conda run -n astro python src/skymodel/evaluation/halo_compare.py
    conda run -n astro python src/skymodel/evaluation/halo_compare.py \\
        --frac 0.5 --xlim 6500 6900 --normalise
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL, ROOT, load_field  # noqa: E402
from blank_compare import data_hdu, our_cube  # noqa: E402
from halo_spectra import EDGE_MARGIN, LINES, Z_HARO, CHUNK, panel_ylim  # noqa: E402
from utils import DZ_MAX, main_source_group  # noqa: E402
from scipy import ndimage  # noqa: E402

FIGURES = EVAL / "halo"
CACHE = FIGURES / "cache"


def halo_mask(seg, white, valid, main, frac, band=None):
    """Which spaxels count as halo, away from the field edge.

    Two constructions, and they answer different questions.

    frac -- the faintest `frac` of this pointing's main source group. Self-scaling, so
    it always finds the outskirts of whatever the group is; but how far out the group
    reaches differs between pointings, so the zones sit at different surface
    brightnesses and are not the same part of the galaxy.

    band -- an absolute surface-brightness window (lo, hi) in white-light units, the
    same numbers in every pointing. The zones are then genuinely the same isophote and
    can be compared directly; the price is that a pointing may contribute very few
    spaxels, or none, and the count has to be read alongside the curve.
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


def main():
    ap = argparse.ArgumentParser(
        description="Haro 11's halo spectrum in every pointing, on one figure")
    ap.add_argument("--pointings", nargs="+", default=[f"p{i:02d}" for i in range(1, 15)])
    ap.add_argument("--root", default="results/skymodel")
    ap.add_argument("--run", default=None,
                    help="glob naming the run under step05 that holds sky_subtracted.fits")
    ap.add_argument("--frac", type=float, default=0.25,
                    help="fraction of the main source group counted as halo, faintest first")
    ap.add_argument("--band", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="absolute white-light surface brightness window used instead "
                         "of --frac. The same isophote in every pointing, which is what "
                         "makes the curves comparable; a pointing may contribute few "
                         "spaxels, so read the count with the curve")
    ap.add_argument("--refresh", action="store_true",
                    help="re-read the cubes even if a cache from an identical selection exists")
    ap.add_argument("--step04", action="store_true",
                    help="require members of the main group to share the main source's "
                         "redshift; without it adjacency alone decides")
    ap.add_argument("--normalise", action="store_true",
                    help="divide each curve by its own continuum median, for comparing "
                         "shape when the pointings differ in depth")
    ap.add_argument("--cont", type=float, nargs=2, metavar=("LO", "HI"), default=(5700, 5750),
                    help="line-free band used for the continuum level")
    ap.add_argument("--xlim", type=float, nargs=2, metavar=("LO", "HI"), default=None)
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None)
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--no-lines", action="store_true")
    ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(22, 8))
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    got = []
    for name in args.pointings:
        W = ROOT / args.root / name
        run = our_cube(W, args.run)
        if run is None:
            print(f"  skip {name}: no sky_subtracted.fits under step05/step06")
            continue
        seg, white, valid = load_field(W)
        main_, ids, _ = main_source_group(seg, np.where(valid, white, np.nan),
                                          W / "step04" if args.step04 else None, DZ_MAX)
        m = halo_mask(seg, white, valid, main_, args.frac, args.band)
        wl = np.load(W / "step03/wavelength.npy")
        if int(m.sum()) == 0:
            print(f"  skip {name}: the selection contains no spaxel")
            continue
        # One cube read takes minutes and the figure gets re-drawn many times while its
        # scales are settled, so the spectra are cached on the selection that produced
        # them. The key carries every input that changes the numbers -- a cache that can
        # be read back for a different selection is worse than no cache.
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

    def cont(wl, s):
        b = (wl > args.cont[0]) & (wl < args.cont[1])
        return float(np.median(s[b])) if b.any() else np.nan

    print(f"\n    {'':>5}{'spaxels':>9}{'median white':>14}{'continuum':>11}"
          f"{'Ha peak':>10}{'channels':>10}   run")
    for name, wl, s, n, mw, run in got:
        obs = 6562.8 * (1 + Z_HARO)
        w = np.abs(wl - obs) < 6
        loc = np.median(s[(np.abs(wl - obs) > 12) & (np.abs(wl - obs) < 30)])
        ha = float(s[w].max() - loc) if w.any() else np.nan
        print(f"    {name:>5}{n:>9,}{mw:>14.3f}{cont(wl, s):>11.3f}{ha:>10.2f}"
              f"{wl.size:>10}   {run}")

    # tab20: the pointing number labels a dither, not a quantity, so a sequential map
    # would imply an order that is not there.
    cols = plt.get_cmap("tab20").colors

    fig, ax = plt.subplots(figsize=args.figsize)
    if not args.no_lines:
        for _, lam in LINES:
            ax.axvline(lam * (1 + Z_HARO), lw=0.6, color="0.85", zorder=0)
    ax.axhline(0, lw=0.8, color="0.55")
    drawn = []
    for i, (name, wl, s, n, mw, _) in enumerate(got):
        y = s / cont(wl, s) if args.normalise else s
        if args.smooth > 1:
            y = np.convolve(y, np.ones(args.smooth) / args.smooth, mode="same")
        ax.plot(wl, y, lw=0.7, color=cols[i % len(cols)], label=f"{name}  ({n:,} px)")
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
    out = Path(args.out) if args.out else FIGURES / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
