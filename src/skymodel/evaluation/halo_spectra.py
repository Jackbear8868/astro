"""The spectrum of Haro 11's extended light, layer by layer.

The white-light map says where the extended light is, not what it is. This cuts the
main source group into surface-brightness layers from the core outwards, adds a few
rings of sky beyond its boundary, and draws the mean spectrum of each.

Layers by brightness, not rings by radius: the galaxy is a merger, so a ring around the
peak averages bright knots with faint outskirts at the same radius. A brightness layer
is an isophotal zone, which is what "how far out" means for such a shape, and
equal-count layers give comparable noise -- an equal-width split in flux would put
almost every spaxel in the faintest one. Outside the group the seg edge is the only
reference, so those zones are rings of distance from it: light continuing past the
boundary was treated as sky, the check that matters here.

Each layer gets its own panel and y scale: the core is orders of magnitude brighter
than the outskirts, and a shared scale would flatten every other layer onto zero.

The cube read is one that has had the sky taken out: this pointing's own
step06/sky_subtracted.fits, or ESO's with --cube. A wsky cube would show the sky.

    conda run -n astro python src/skymodel/evaluation/halo_spectra.py --work results/skymodel/p01
    conda run -n astro python src/skymodel/evaluation/halo_spectra.py --work results/skymodel/p07 \\
        --cube data/nosky/DATACUBE_FINAL_ESOSKY_7.fits
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy.signal import medfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, arcsinh_stretch  # noqa: E402
from products import Run  # noqa: E402
from utils import DZ_MAX, main_source_group  # noqa: E402

# Haro 11's redshift, and the lines bright enough to mark; the markers are guides for
# reading the panels. Both halves of the [O III] doublet are marked because they share
# an upper level, so the transition probabilities alone fix the ratio at
# 5007/4959 = 2.98 -- a zone where it is not about 3 has a problem in the subtraction
# or the fit, not in the physics.
Z_HARO = 0.0204
LINES = [("Hb", 4861.3), ("[O III] 4959", 4958.9), ("[O III] 5007", 5006.8),
         ("Ha", 6562.8), ("[S II]", 6716.4)]
# Pale red for the line markers -- a colour neither the grid nor the zero line uses.
# Grey would match the grid, whose 5000 A line falls beside the redshifted Hb marker.
C_LINE = "#f4a3a3"
# Transparency for the full-height marker (--marker line): it sits under the peak it
# names, and at full strength the two are one stroke. A paler colour would lose the hue.
A_LINE = 0.45

# Spaxels this close to the field-of-view edge are dropped: exposures fall off there
# and step5 writes NaN below 90% coverage, so a layer there measures the mosaic.
EDGE_MARGIN = 6
# Channels read at a time: small enough to keep a chunk in memory, large enough to
# average in vectorised blocks.
CHUNK = 256


def zone_labels(seg, white, valid, main, n_layers, rings):
    """Integer zone map (0 = unused) plus the zone names, ordered inner to outer."""
    d_edge = ndimage.distance_transform_edt(valid)
    ok = valid & (d_edge > EDGE_MARGIN)

    zones = np.zeros(seg.shape, int)
    names = []

    # --- inside the main group: equal-count layers of white-light brightness ---
    inside = main & ok
    v = white[inside]
    # Quantile edges, brightest first, so zone 1 is the core.
    edges = np.percentile(v, np.linspace(0, 100, n_layers + 1))
    for k in range(n_layers):
        lo, hi = edges[n_layers - 1 - k], edges[n_layers - k]
        m = inside & (white >= lo) & (white <= hi if k == 0 else white < hi)
        zones[m] = len(names) + 1
        # L1 is the brightest, and the panels are stacked in that order.
        names.append(f"galaxy L{k + 1}")

    # --- outside it: rings of distance from the boundary ---
    d_main = ndimage.distance_transform_edt(~main)
    outside = ok & ~main & (seg == 0)      # other sources excluded: their light is not Haro 11's
    for lo, hi in zip(rings[:-1], rings[1:]):
        m = outside & (d_main > lo) & (d_main <= hi)
        zones[m] = len(names) + 1
        names.append(f"outside {lo}-{hi} px")
    return zones, names


def zone_means(cube_path, zones, n_zones, nz):
    """Mean spectrum of every zone, read in wavelength chunks."""
    idx = [np.flatnonzero((zones == k + 1).ravel()) for k in range(n_zones)]
    out = np.full((n_zones, nz), np.nan)
    with fits.open(cube_path, memmap=True) as h:
        hdu = h["DATA"] if "DATA" in h else h[0]
        if hdu.data.shape[0] != nz:
            raise SystemExit(f"cube has {hdu.data.shape[0]} channels, "
                             f"wavelength.npy has {nz}")
        for c0 in range(0, nz, CHUNK):
            c1 = min(c0 + CHUNK, nz)
            block = np.asarray(hdu.data[c0:c1], np.float32).reshape(c1 - c0, -1)
            with np.errstate(invalid="ignore"):
                for k, ix in enumerate(idx):
                    if ix.size:
                        out[k, c0:c1] = np.nanmean(block[:, ix], axis=1)
            print(f"    {c1}/{nz} channels", end="\r", flush=True)
    print(" " * 30, end="\r")
    return out


def panel_ylim(spec):
    """y range for one panel: what the spectrum does, not what one bad voxel does.

    A single dead or hot channel can be many times the whole range of a zone, and
    autoscaling to it flattens the real spectrum onto zero. The range therefore comes
    from the 3-channel median-filtered spectrum, which a one-channel excursion cannot
    survive but a resolved line can, MUSE lines being several channels wide. The raw
    curve is still what gets drawn.

    The range is then widened to cover any raw value within one panel height of that,
    so the tips of real lines are not cut off by a rule aimed at single channels.
    Anything beyond runs off the panel, and the caller reports it.
    """
    m = medfilt(spec, 3)
    lo, hi = float(np.nanmin(m)), float(np.nanmax(m))
    span = max(hi - lo, 1e-9)
    near = spec[(spec >= lo - span) & (spec <= hi + span)]
    if near.size:
        lo, hi = min(lo, float(near.min())), max(hi, float(near.max()))
    pad = 0.06 * max(hi - lo, 1e-9)
    return lo - pad, hi + pad


def main():
    ap = argparse.ArgumentParser(
        description="Mean spectrum of Haro 11's extended light, in surface-brightness layers")
    ap.add_argument("--work", required=True,
                    help="pointing work directory, e.g. results/skymodel/p01")
    ap.add_argument("--cube", default=None,
                    help="sky-subtracted cube; defaults to the pointing's own "
                         "step06/sky_subtracted.fits")
    ap.add_argument("--layers", type=int, default=4,
                    help="equal-count brightness layers inside the main source group")
    ap.add_argument("--rings", type=int, nargs="+", default=[0, 10, 25, 50],
                    help="ring edges in px outside the group's boundary; N edges give N-1 rings")
    ap.add_argument("--step04", default=None,
                    help="step04 directory; given, the main group keeps only members "
                         "matching the main source's redshift")
    ap.add_argument("--dz-max", type=float, default=DZ_MAX)
    ap.add_argument("--smooth", type=int, default=1,
                    help="running-mean width in channels; 1 leaves the spectrum untouched")
    ap.add_argument("--xlim", type=float, nargs=2, metavar=("LO", "HI"), default=None)
    ap.add_argument("--no-lines", action="store_true",
                    help="drop the redshifted line markers")
    ap.add_argument("--marker", choices=["tick", "line"], default="tick",
                    help="tick: a short stroke at the top of the panel, which names "
                         "the wavelength without crossing the spectrum at all. "
                         "line: a full-height rule, which puts the wavelength next to "
                         "the data at the cost of drawing over it")
    ap.add_argument("--width", type=float, default=20)
    ap.add_argument("--panel-height", type=float, default=1.9)
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Run(args.work)
    name = run.name
    seg, white, valid = run.seg, run.white, run.valid
    main_, ids, peak = main_source_group(seg, np.where(valid, white, np.nan),
                                         Path(args.step04) if args.step04 else None,
                                         args.dz_max)

    cube = Path(args.cube) if args.cube else run.cube
    if not cube.is_absolute():
        cube = ROOT / cube
    if not cube.exists():
        raise SystemExit(f"{cube} does not exist -- this pointing has no step06 yet. "
                         f"Pass --cube data/nosky/DATACUBE_FINAL_ESOSKY_N.fits to use "
                         f"ESO's sky subtraction instead.")
    wl = run.wl

    zones, names = zone_labels(seg, white, valid, main_, args.layers, args.rings)
    n = len(names)
    print(f"{name}: main group {len(ids)} ids {ids}, {int(main_.sum()):,} px")
    print(f"  cube {cube.relative_to(ROOT) if cube.is_relative_to(ROOT) else cube}")
    print(f"  {wl.size} channels {wl.min():.1f}-{wl.max():.1f} A\n")

    print(f"    {'zone':<22}{'spaxels':>9}{'median white':>14}")
    for k, nm in enumerate(names):
        m = zones == k + 1
        wv = white[m]
        print(f"    {nm:<22}{int(m.sum()):>9}"
              f"{(np.median(wv) if wv.size else np.nan):>14.3f}")

    print("\n  averaging the cube ...")
    spec = zone_means(cube, zones, n, wl.size)

    if args.smooth > 1:
        k = np.ones(args.smooth) / args.smooth
        spec = np.array([np.convolve(s, k, mode="same") for s in spec])

    # Sequential colours: the zones run inner to outer, and a qualitative palette would
    # hide that order. The pale end of viridis is left out; it does not read on white.
    cols = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, n))

    fig, axes = plt.subplots(n, 1, sharex=True,
                             figsize=(args.width, args.panel_height * n + 1.2),
                             gridspec_kw={"hspace": 0.10})
    axes = np.atleast_1d(axes)
    clipped = []
    for k, (nm, ax) in enumerate(zip(names, axes)):
        ax.axhline(0, lw=0.7, color="0.6")
        if not args.no_lines:
            for _, lam in LINES:
                if args.marker == "tick":
                    # ymin/ymax are axes fractions, so the stroke keeps to the top of
                    # the panel and never crosses the spectrum, whatever the range.
                    ax.axvline(lam * (1 + Z_HARO), ymin=0.92, ymax=1.0,
                               lw=1.4, color=C_LINE, zorder=3)
                else:
                    ax.axvline(lam * (1 + Z_HARO), lw=0.8, color=C_LINE,
                               alpha=A_LINE, zorder=0)
        ax.plot(wl, spec[k], lw=0.6, color=cols[k])
        lo, hi = panel_ylim(spec[k])
        ax.set_ylim(lo, hi)
        off = np.flatnonzero((spec[k] < lo) | (spec[k] > hi))
        if off.size:
            # Named, not silently cropped: a channel drawn off the panel is a
            # measurement about the cube, and the reader cannot see it in the figure.
            worst = off[np.argmax(np.abs(spec[k][off]))]
            clipped.append(f"    {nm:<22}{off.size:>4} channel(s) outside the panel; "
                           f"the largest is {spec[k][worst]:.1f} at {wl[worst]:.1f} A")
        ax.set_ylabel("flux", fontsize=9)
        # The zone name sits inside the panel: a legend outside would be a column of
        # text to match back by colour. The spaxel count stays in the printed table,
        # where it can be read against the other zones.
        ax.text(0.004, 0.93, nm,
                transform=ax.transAxes, fontsize=10, va="top", ha="left",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.6))
        # No axis grid: its vertical lines sit at tick positions, mean nothing
        # physical, and would compete with the redshift markers. Anything vertical
        # here should be a wavelength worth knowing.
    if clipped:
        print("\n  drawn off the panel (the y range follows the median-filtered "
              "spectrum, so single bad channels do not set it):")
        print("\n".join(clipped))
    axes[-1].set_xlabel("wavelength [$\\AA$]")
    axes[-1].set_xlim(*(args.xlim if args.xlim else (wl.min(), wl.max())))

    # A cube given on the command line goes into the filename, or this pointing's own
    # sky subtraction and ESO's would write the same figure name.
    stem = "halo_spectra" if args.cube is None else f"halo_spectra_{cube.stem}"
    out = Path(args.out) if args.out else run.figdir("halo") / f"{stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")

    # The map is not optional decoration: a mean spectrum with no picture of where it
    # came from cannot be checked against the field.
    stretched, vmax = arcsinh_stretch(white, valid, soft=0.004)
    fig, ax = plt.subplots(figsize=(9, 9 * seg.shape[0] / seg.shape[1]))
    ax.imshow(stretched, origin="lower", cmap="gray", vmin=0, vmax=vmax)
    rgba = np.zeros(seg.shape + (4,))
    for k in range(n):
        rgba[zones == k + 1] = list(cols[k][:3]) + [0.55]
    ax.imshow(rgba, origin="lower")
    ax.contour(main_, levels=[0.5], colors="#ff7f0e", linewidths=1.2)
    ax.legend(handles=[mpatches.Patch(color=cols[k], label=names[k]) for k in range(n)],
              fontsize=9, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              borderaxespad=0, frameon=False)
    ax.set_axis_off()
    out_map = out.with_name(out.stem + "_map.png")
    fig.savefig(out_map, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_map}")


if __name__ == "__main__":
    main()
