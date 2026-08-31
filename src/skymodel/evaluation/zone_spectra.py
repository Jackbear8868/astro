"""The mean spectrum of each zone of the field, from one cube or from several.

Was the sky removed cleanly, and did the source survive? Residual size alone cannot
tell one from the other -- subtracting the source away flattens the residual the same
way -- so both readings have to land in the same figure. A zone at a time is how: in
blank the curve should sit on zero, and on the galaxy it should be a galaxy spectrum
with its lines standing up rather than pushed down.

    zone_spectra.py --work results/skymodel/p01
    zone_spectra.py --work results/skymodel/p01 --zones outside --cubes ours eso
    zone_spectra.py --work results/skymodel/p01 --zones galaxy --map

Two switches carry it. `--zones` picks which of the field's zones to draw -- the
brightness layers of the galaxy, the rings outside its boundary, or both -- and
`--cubes` says which cubes to draw in each. Everything else is how it looks.

Colour carries whichever of the two is varying. With one cube the panels differ by
zone, so the colour runs inner to outer through viridis and the zone map is drawn in
the same colours. With several, the colour names the cube and is the same in every
panel, since what is being read is the difference between them and it must mean one
thing from panel to panel.

This is what halo_spectra.py and outside_compare.py were. They were the same program:
the same zones from the same construction, the same panel, the same markers, the same
report of what fell off it. One drew every zone with one curve, the other drew the
outer zones with two.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, arcsinh_stretch, slug  # noqa: E402
from products import Run, spectrum_stats  # noqa: E402
from spectra import (A_LINE, C_ESO, C_LINE, C_OURS, C_ZERO, LINES,  # noqa: E402
                     Z_HARO, despiked_range, panel_ylim)
from utils import DZ_MAX, main_source_group  # noqa: E402
from zones import zone_labels, zone_means  # noqa: E402

# Opacity of every curve after the first. The first is the one being examined and is
# drawn solid on top; the rest are references underneath, and at full strength a pair
# that agrees reads as one stroke rather than as agreement.
A_REF = 0.75
# Colours after the first two. Reached only by --cubes with three or more entries.
C_MORE = ["#2ca02c", "#9467bd", "#8c564b"]


def select(names, which):
    """The zone numbers to draw, and their names.

    zone_labels numbers every zone it builds, and the numbers are what the spectra are
    keyed by, so a subset keeps its original numbers rather than being renumbered --
    otherwise the rings drawn here and the rings in the full figure would be different
    zones with the same names.
    """
    if which == "all":
        keep = list(range(len(names)))
    elif which == "galaxy":
        keep = [i for i, nm in enumerate(names) if nm.startswith("galaxy")]
    else:
        keep = [i for i, nm in enumerate(names) if nm.startswith("outside")]
    if not keep:
        raise SystemExit(f"the zone construction produced nothing matching --zones {which}")
    return [i + 1 for i in keep], [names[i] for i in keep]


def curve_colours(n_cubes, n_zones):
    """One colour per curve, carrying whichever of cube and zone is varying.

    Returns a function of (cube index, zone index). With one cube the zones are what
    differ between panels, and a sequential map says which way is outwards; a
    qualitative palette would hide that order. With several, the colour is the cube's
    and is the same in every panel, or a difference between two panels would be
    unreadable.
    """
    if n_cubes == 1:
        # The pale end of viridis is left out; it does not read on white.
        cols = plt.get_cmap("viridis")(np.linspace(0.05, 0.85, n_zones))
        return lambda c, z: cols[z]
    fixed = [C_OURS, C_ESO] + C_MORE
    return lambda c, z: fixed[c % len(fixed)]


def panel_range(curves, rule, pct):
    """The y range of one panel, given every curve that goes in it.

    One range for all of them, or the larger residual is squeezed until it looks like
    the smaller. They can differ by orders of magnitude, so keeping every channel of
    every curve spends the height on one and draws the rest flat.

    medfilt   follows the median-filtered extremes of all the curves. Everything stays
              on the panel, at the cost of letting one curve's outliers flatten the
              others.
    first-pct the first curve is never allowed off the panel and the rest reach only to
              their percentiles, so a reference's deepest sky-line residuals may run
              off. The first curve is the one being examined, which is what makes this
              the default where there is more than one.
    """
    if rule == "medfilt" or len(curves) == 1:
        return panel_ylim(np.concatenate(curves))
    lo, hi = despiked_range(curves[0])
    for y in curves[1:]:
        lo = min(lo, float(np.nanpercentile(y, pct)))
        hi = max(hi, float(np.nanpercentile(y, 100 - pct)))
    m = 0.08 * max(hi - lo, 1e-9)
    return lo - m, hi + m


def mark_lines(ax, marker):
    """Haro 11's lines, at the top of the panel or across it.

    tick: a short stroke at the top, which names the wavelength without crossing the
    spectrum at all. line: a full-height rule, which puts the wavelength next to the
    data at the cost of drawing over it.
    """
    for _, lam in LINES:
        if marker == "tick":
            # ymin/ymax are axes fractions, so the stroke keeps to the top of the panel
            # whatever the range is.
            ax.axvline(lam * (1 + Z_HARO), ymin=0.92, ymax=1.0, lw=1.4,
                       color=C_LINE, zorder=3)
        else:
            ax.axvline(lam * (1 + Z_HARO), lw=0.8, color=C_LINE, alpha=A_LINE, zorder=0)


def draw_map(run, zones, keys, names, colour, main, out, dpi):
    """Where the zones are, in the colours the curves were drawn in.

    Not optional decoration: a mean spectrum with no picture of where it came from
    cannot be checked against the field.
    """
    seg, white, valid = run.seg, run.white, run.valid
    stretched, vmax = arcsinh_stretch(white, valid, soft=0.004)
    fig, ax = plt.subplots(figsize=(9, 9 * seg.shape[0] / seg.shape[1]))
    ax.imshow(stretched, origin="lower", cmap="gray", vmin=0, vmax=vmax)
    rgba = np.zeros(seg.shape + (4,))
    for z, k in enumerate(keys):
        rgba[zones == k] = list(colour(0, z)[:3]) + [0.55]
    ax.imshow(rgba, origin="lower")
    # The boundary the rings are measured from, in a colour none of the zones uses.
    ax.contour(main, levels=[0.5], colors="#ff7f0e", linewidths=1.2)
    ax.legend(handles=[mpatches.Patch(color=colour(0, z), label=names[z])
                       for z in range(len(keys))],
              fontsize=9, loc="upper left", bbox_to_anchor=(1.005, 1.0),
              borderaxespad=0, frameon=False)
    ax.set_axis_off()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def main():
    ap = argparse.ArgumentParser(
        description="Mean spectrum of each zone of the field, from one cube or several")
    ap.add_argument("--work", required=True,
                    help="pointing work directory, e.g. results/skymodel/p01")
    ap.add_argument("--zones", choices=["all", "galaxy", "outside"], default="all",
                    help="which zones to draw: the galaxy's brightness layers, the "
                         "rings outside its boundary, or both")
    ap.add_argument("--cubes", nargs="+", default=["ours"], metavar="CUBE",
                    help="ours (step06), eso (the nosky the config names), wsky (the "
                         "input), model (the sky taken out), run:GLOB (a run under "
                         "step05), or a path. The first is the one being examined")
    ap.add_argument("--labels", nargs="+", default=None, metavar="NAME",
                    help="names for the curves; defaults to what --cubes says")

    ap.add_argument("--layers", type=int, default=4,
                    help="equal-count brightness layers inside the main source group. "
                         "Given even with --zones outside, so the rings match the "
                         "layer figure exactly")
    ap.add_argument("--rings", type=int, nargs="+", default=[0, 10, 25, 50],
                    help="ring edges in px outside the boundary; N edges give N-1 rings")
    ap.add_argument("--step04", default=None,
                    help="step04 directory; given, the main group keeps only members "
                         "matching the main source's redshift")
    ap.add_argument("--dz-max", type=float, default=DZ_MAX)

    ap.add_argument("--exclude-source-lines", type=float, nargs="?", const=12.0,
                    default=None, metavar="HW",
                    help="also report the statistics with Haro 11's own lines removed, "
                         "+-HW Angstrom around each line the figure marks (default 12). "
                         "Outside the boundary is not outside the galaxy's line "
                         "emission, and those channels make the residual columns "
                         "favour whichever cube keeps less source light")
    ap.add_argument("--smooth", type=int, default=1,
                    help="running-mean width in channels; 1 leaves the spectrum alone")
    ap.add_argument("--ylim-rule", choices=["first-pct", "medfilt"], default=None,
                    help="see panel_range. Defaults to medfilt for one cube and "
                         "first-pct for more")
    ap.add_argument("--pct", type=float, default=5.0,
                    help="percentile the reference curves reach to, with first-pct")
    ap.add_argument("--xlim", type=float, nargs=2, metavar=("LO", "HI"), default=None)
    ap.add_argument("--no-lines", action="store_true",
                    help="drop the redshifted line markers")
    ap.add_argument("--marker", choices=["tick", "line"], default="tick")
    ap.add_argument("--separate", action="store_true",
                    help="one file per zone instead of one stacked figure")
    ap.add_argument("--map", action="store_true",
                    help="also draw where the zones are. On by default with one cube, "
                         "which is the figure that is read against the field")
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--width", type=float, default=20)
    ap.add_argument("--panel-height", type=float, default=None,
                    help="default 1.9 for one cube, 2.4 for more -- two curves in a "
                         "panel need the height to stay apart")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Run(args.work)
    labels = args.labels or args.cubes
    if len(labels) != len(args.cubes):
        raise SystemExit(f"{len(args.cubes)} cubes but {len(labels)} labels")
    paths = [run.named_cube(c) for c in args.cubes]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"{p} does not exist")

    rule = args.ylim_rule or ("medfilt" if len(paths) == 1 else "first-pct")
    ph = args.panel_height or (1.9 if len(paths) == 1 else 2.4)
    want_map = (len(paths) == 1 or args.map) and not args.no_map

    seg, white, valid = run.seg, run.white, run.valid
    main_, ids, _ = main_source_group(seg, np.where(valid, white, np.nan),
                                      Path(args.step04) if args.step04 else None,
                                      args.dz_max)
    zones, all_names = zone_labels(seg, white, valid, main_, args.layers, args.rings)
    keys, names = select(all_names, args.zones)
    n = len(keys)
    wl = run.wl

    print(f"{run.name}: main group {len(ids)} ids {ids}, {int(main_.sum()):,} px")
    for lab, p in zip(labels, paths):
        print(f"  {lab:<10}{p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
    print(f"  {wl.size} channels {wl.min():.1f}-{wl.max():.1f} A\n")
    print(f"    {'zone':<22}{'spaxels':>9}{'median white':>14}")
    for z, k in enumerate(keys):
        m = zones == k
        wv = white[m]
        print(f"    {names[z]:<22}{int(m.sum()):>9,}"
              f"{(np.median(wv) if wv.size else np.nan):>14.3f}")

    print("\n  averaging the cubes ...")
    spec = [zone_means(p, zones, keys, wl.size) for p in paths]

    # The same lines the figure marks, so the dropped channels are exactly what a
    # reader sees marked. All False unless --exclude-source-lines was given.
    src = np.zeros(wl.size, bool)
    if args.exclude_source_lines:
        for _lab, lam in LINES:
            src |= np.abs(wl - lam * (1 + Z_HARO)) <= args.exclude_source_lines
        print(f"\n  source lines removed from the second set of columns: "
              f"+-{args.exclude_source_lines:g} A around {len(LINES)} lines at "
              f"z={Z_HARO:g}, {int(src.sum())} of {wl.size} channels "
              f"({100 * src.mean():.1f}%)")

    wid = max(6, max(len(l) for l in labels) + 2)
    head = f"\n    {'':<22}{'':<{wid}}{'mean':>10}{'sigma':>10}{'rms_from_zero':>16}"
    if src.any():
        head += f"{'mean':>12}{'sigma':>10}{'rms_from_zero':>16}    source lines out"
    print(head)
    for z in range(n):
        for c, lab in enumerate(labels):
            st = spectrum_stats(spec[c][z])
            row = (f"    {names[z] if c == 0 else '':<22}{lab:<{wid}}"
                   f"{st['mean']:>10.4f}{st['sigma']:>10.4f}{st['rms_from_zero']:>16.4f}")
            if src.any():
                sk = spectrum_stats(spec[c][z][~src])
                row += (f"{sk['mean']:>12.4f}{sk['sigma']:>10.4f}"
                        f"{sk['rms_from_zero']:>16.4f}")
            print(row)

    if args.smooth > 1:
        kk = np.ones(args.smooth) / args.smooth
        spec = [np.array([np.convolve(y, kk, mode="same") for y in s]) for s in spec]

    colour = curve_colours(len(paths), n)
    if args.separate:
        # A zone per file. The full canvas height is what makes small residuals
        # readable; the price is independent y ranges, so zones cannot be compared.
        figs = [plt.subplots(figsize=(args.width, ph * 2.2)) for _ in keys]
        axes = [a for _, a in figs]
    else:
        fig, axes = plt.subplots(n, 1, sharex=True,
                                 figsize=(args.width, ph * n + 1.2),
                                 gridspec_kw={"hspace": 0.10})
        axes = list(np.atleast_1d(axes))

    clipped = []
    for z, ax in enumerate(axes):
        if not args.no_lines:
            mark_lines(ax, args.marker)
        ax.axhline(0, lw=0.8, color=C_ZERO)
        # Drawn last to first: the references go underneath and thicker, the curve
        # being examined on top and thinner, so an orange rim around the blue reads as
        # agreement rather than as one line.
        # Alone, the curve is as fine as it reads; over a reference it has to be
        # thick enough to stay visible on top of it, and the reference thicker still
        # so an orange rim around the blue reads as agreement rather than one line.
        for c in range(len(paths) - 1, -1, -1):
            lw = 0.6 if len(paths) == 1 else (0.7 if c == 0 else 1.3)
            ax.plot(wl, spec[c][z], lw=lw, color=colour(c, z),
                    label=labels[c], alpha=1.0 if c == 0 else A_REF,
                    zorder=4 if c == 0 else 2)
        lo, hi = panel_range([spec[c][z] for c in range(len(paths))], rule, args.pct)
        ax.set_ylim(lo, hi)
        for c, lab in enumerate(labels):
            y = spec[c][z]
            off = np.flatnonzero((y < lo) | (y > hi))
            if off.size:
                # Named, not silently cropped: a channel drawn off the panel is a
                # measurement about the cube, and the reader cannot see it in the figure.
                worst = off[np.argmax(np.abs(y[off]))]
                clipped.append(f"    {names[z]:<22}{lab:<{wid}}{off.size:>4} channel(s) "
                               f"off the panel; the largest is {y[worst]:.1f} at "
                               f"{wl[worst]:.1f} A")
        ax.set_ylabel("flux", fontsize=9)
        if not args.separate:
            # Stacked, the corner text is the only thing telling the panels apart. One
            # zone per file names itself in the filename, so the text is dropped there.
            ax.text(0.004, 0.93, names[z], transform=ax.transAxes, fontsize=10,
                    va="top", ha="left",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.6))
        if len(paths) > 1:
            if args.separate:
                # Above the axes, so it never lands on a residual the panel exists to show.
                ax.legend(fontsize=11, loc="lower left", frameon=False, ncol=len(paths),
                          bbox_to_anchor=(0, 1.005), borderaxespad=0)
            elif z == 0:
                ax.legend(fontsize=10, loc="upper right", frameon=False, ncol=len(paths))
        if args.separate:
            ax.set_xlabel("wavelength [$\\AA$]")
        ax.set_xlim(*(args.xlim if args.xlim else (wl.min(), wl.max())))
    if not args.separate:
        axes[-1].set_xlabel("wavelength [$\\AA$]")
        axes[-1].set_xlim(*(args.xlim if args.xlim else (wl.min(), wl.max())))
    if clipped:
        print(f"\n  drawn off the panel (y range rule {rule}"
              + (f", pct {args.pct:g}" if rule == "first-pct" else "") + "):")
        print("\n".join(clipped))

    # The name says what is in the figure: which zones, and which cubes against which.
    span = f"_{args.xlim[0]:.0f}-{args.xlim[1]:.0f}" if args.xlim else ""
    stem = f"{args.zones}_" + "_vs_".join(slug(l) for l in labels)
    # --out may name either a directory or the file itself, so the directory to make
    # is not known until that is decided; each branch below makes its own.
    d = Path(args.out) if args.out else run.figdir("halo")
    if args.separate:
        for (f, _), nm in zip(figs, names):
            o = d / f"{stem}_{slug(nm)}{span}.png"
            o.parent.mkdir(parents=True, exist_ok=True)
            f.savefig(o, dpi=args.dpi, bbox_inches="tight")
            plt.close(f)
            print(f"saved -> {o}")
    else:
        o = d if d.suffix == ".png" else d / f"{stem}{span}.png"
        o.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(o, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"\nsaved -> {o}")

    if want_map:
        draw_map(run, zones, keys, names, colour, main_,
                 (o if not args.separate else d / f"{stem}{span}.png")
                 .with_name(f"{stem}{span}_map.png"), args.dpi)


if __name__ == "__main__":
    main()
