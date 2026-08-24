"""Outside the source boundary: what our sky subtraction leaves, and what ESO's does.

The rings just outside the segmentation are where the two pipelines can be told apart.
They are blank by the mask's own definition, so a sky model that worked leaves nothing
there; but they are also where Haro 11's extended light still is, so a sky model that
over-subtracted takes that light with it. Both readings are in the same panel, which
is the point -- residual size alone cannot separate "clean" from "the galaxy was
eaten", because both make the line flatter.

The zones are halo_spectra's, imported rather than redefined: the same distance rings
from the same main source group, other sources excluded. A comparison drawn over a
second definition of "outside" would differ from the layer figure for reasons that
have nothing to do with the pipelines.

    ours   this pointing's step06/sky_subtracted.fits
    ESO    data/nosky/DATACUBE_FINAL_ESOSKY_N.fits

Both are averaged over identical spaxels, with no clipping and nothing resampled.

    conda run -n astro python src/skymodel/evaluation/outside_compare.py --work results/skymodel/p01
    conda run -n astro python src/skymodel/evaluation/outside_compare.py --work results/skymodel/p07 \\
        --rings 0 10 25 50 100 --xlim 6500 6900
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.signal import medfilt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, load_field, pointing_dir, slug  # noqa: E402
from blank_compare import data_hdu, our_cube  # noqa: E402
from halo_spectra import (C_LINE, CHUNK, LINES, Z_HARO, panel_ylim,  # noqa: E402
                          zone_labels)
from utils import DZ_MAX, main_source_group, spectrum_stats  # noqa: E402

C_OURS, C_ESO, C_ZERO = "#1f77b4", "#e8710a", "0.55"
# Transparency for the reference curve only. It is the lower of the two, so fading it
# lets our curve read cleanly where they overlap. Putting alpha on the upper curve
# instead would blend it with whatever is behind -- our blue would turn muddy exactly
# on top of ESO's spikes, which is where it most needs to be legible.
A_ESO = 0.75


def despiked_range(y):
    """The range a curve occupies, with single-channel excursions left out.

    Our residual carries dead or hot channels -- one channel near 4790 A reaches -122
    where its neighbours are near -2 -- and a 3-channel median removes those while
    keeping a spectrally resolved emission line, which is several channels wide.

    The range is then extended back to any raw value within one full span of it, so the
    tip of a real line is not cut off by a rule aimed at single channels.
    """
    m = medfilt(y, 3)
    lo, hi = float(np.nanmin(m)), float(np.nanmax(m))
    span = max(hi - lo, 1e-9)
    near = y[(y >= lo - span) & (y <= hi + span)]
    if near.size:
        lo, hi = min(lo, float(near.min())), max(hi, float(near.max()))
    return lo, hi


def zone_means(cube, zones, keys, nz):
    """Mean spectrum of each requested zone, read in wavelength chunks."""
    idx = [np.flatnonzero((zones == k).ravel()) for k in keys]
    out = np.full((len(keys), nz), np.nan)
    with fits.open(cube, memmap=True) as h:
        hdu = data_hdu(h)
        if hdu.data.shape[0] != nz:
            raise SystemExit(f"{cube.name} has {hdu.data.shape[0]} channels, "
                             f"wavelength.npy has {nz}")
        for c0 in range(0, nz, CHUNK):
            c1 = min(c0 + CHUNK, nz)
            block = np.asarray(hdu.data[c0:c1], np.float32).reshape(c1 - c0, -1)
            with np.errstate(invalid="ignore"):
                for j, ix in enumerate(idx):
                    if ix.size:
                        out[j, c0:c1] = np.nanmean(block[:, ix], axis=1)
            print(f"    {cube.name[:28]:<28} {c1}/{nz}", end="\r", flush=True)
    print(" " * 46, end="\r")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="The rings outside the source boundary, ours against ESO's")
    ap.add_argument("--work", required=True)
    ap.add_argument("--run", default=None,
                    help="glob naming the run under step05 that holds sky_subtracted.fits")
    ap.add_argument("--nosky", default=None,
                    help="ESO cube; by default derived from the wsky name in step03/meta.json")
    ap.add_argument("--layers", type=int, default=4,
                    help="passed to the zone construction so the rings match the layer "
                         "figure exactly; the layers themselves are not drawn here")
    ap.add_argument("--rings", type=int, nargs="+", default=[0, 10, 25, 50],
                    help="ring edges in px outside the boundary; N edges give N-1 rings")
    ap.add_argument("--step04", action="store_true",
                    help="require main-group members to share the main source's redshift")
    ap.add_argument("--smooth", type=int, default=1)
    ap.add_argument("--xlim", type=float, nargs=2, metavar=("LO", "HI"), default=None)
    ap.add_argument("--no-lines", action="store_true")
    ap.add_argument("--marker", choices=["tick", "line"], default="tick")
    ap.add_argument("--ylim-rule", choices=["ours-eso-pct", "medfilt"],
                    default="ours-eso-pct",
                    help="ours-eso-pct: the range holds our curve minus its single-"
                         "channel spikes, and ESO only "
                         "between its --eso-pct percentiles, so ESO's deepest sky-line "
                         "residuals may run off the panel. medfilt: the range follows "
                         "the median-filtered extremes of both, which keeps everything "
                         "on the panel but lets one pipeline's outliers flatten the "
                         "other into a line")
    ap.add_argument("--eso-pct", type=float, default=5.0,
                    help="percentile of the ESO curve the range reaches to, with "
                         "ours-eso-pct; smaller keeps more of ESO on the panel")
    ap.add_argument("--alpha", type=float, default=A_ESO,
                    help="opacity of the ESO curve; ours is always drawn solid")
    ap.add_argument("--separate", action="store_true",
                    help="one file per ring instead of one stacked figure")
    ap.add_argument("--width", type=float, default=20)
    ap.add_argument("--panel-height", type=float, default=2.4)
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    W = ROOT / args.work
    name = Path(args.work).name
    meta = json.loads((W / "step03/meta.json").read_text())
    run = our_cube(W, args.run)
    if run is None:
        raise SystemExit(f"no sky_subtracted.fits under {W}/step05 or {W}/step06")
    if args.nosky:
        nosky = Path(args.nosky)
        nosky = nosky if nosky.is_absolute() else ROOT / nosky
    else:
        mo = re.fullmatch(r"DATACUBE_FINAL_(\d+)\.fits", Path(meta["cube"]).name)
        if not mo:
            raise SystemExit(f"cannot derive the ESO cube from {meta['cube']}; pass --nosky")
        nosky = ROOT / "data/nosky" / f"DATACUBE_FINAL_ESOSKY_{mo.group(1)}.fits"
    if not nosky.exists():
        raise SystemExit(f"{nosky} does not exist")

    seg, white, valid = load_field(W)
    main_, ids, _ = main_source_group(seg, np.where(valid, white, np.nan),
                                      W / "step04" if args.step04 else None, DZ_MAX)
    zones, names = zone_labels(seg, white, valid, main_, args.layers, args.rings)
    # The layers keep their zone numbers; only the rings are drawn, and they are the
    # entries the construction named "outside ...".
    keys = [i + 1 for i, nm in enumerate(names) if nm.startswith("outside")]
    keep = [names[k - 1] for k in keys]
    if not keys:
        raise SystemExit("the ring edges produced no zone outside the boundary")

    wl = np.load(W / "step03/wavelength.npy")
    print(f"{name}:  ours {run.relative_to(ROOT)}   ESO {nosky.name}")
    print(f"  main group {len(ids)} ids, {int(main_.sum()):,} px")
    for k, nm in zip(keys, keep):
        print(f"    {nm:<20}{int((zones == k).sum()):>9,} spaxels")

    ours = zone_means(run / "sky_subtracted.fits", zones, keys, wl.size)
    eso = zone_means(nosky, zones, keys, wl.size)

    print(f"\n    {'':<20}{'':<6}{'mean':>10}{'sigma':>10}{'rms_from_zero':>16}")
    for j, nm in enumerate(keep):
        for lab, y in (("ours", ours[j]), ("ESO", eso[j])):
            st = spectrum_stats(y)
            print(f"    {nm if lab == 'ours' else '':<20}{lab:<6}"
                  f"{st['mean']:>10.4f}{st['sigma']:>10.4f}{st['rms_from_zero']:>16.4f}")

    if args.smooth > 1:
        kk = np.ones(args.smooth) / args.smooth
        ours = np.array([np.convolve(y, kk, mode="same") for y in ours])
        eso = np.array([np.convolve(y, kk, mode="same") for y in eso])

    n = len(keys)
    if args.separate:
        # A ring per file. Each gets the full canvas height instead of a third of it,
        # which is what makes the small residuals readable; the price is that the three
        # cannot be compared at a glance, and their y ranges are independent.
        figs = [plt.subplots(figsize=(args.width, args.panel_height * 2.2))
                for _ in keys]
        axes = [a for _, a in figs]
    else:
        fig, axes = plt.subplots(n, 1, sharex=True,
                                 figsize=(args.width, args.panel_height * n + 1.2),
                                 gridspec_kw={"hspace": 0.10})
        axes = list(np.atleast_1d(axes))
    clipped = []
    for j, (nm, ax) in enumerate(zip(keep, axes)):
        if not args.no_lines:
            for _, lam in LINES:
                if args.marker == "tick":
                    ax.axvline(lam * (1 + Z_HARO), ymin=0.92, ymax=1.0,
                               lw=1.4, color=C_LINE, zorder=3)
                else:
                    ax.axvline(lam * (1 + Z_HARO), lw=0.8, color=C_LINE,
                               alpha=0.45, zorder=0)
        ax.axhline(0, lw=0.8, color=C_ZERO)
        # ESO underneath and thicker, ours on top and thinner. Whichever is drawn last
        # wins where they overlap, and our curve is the one being examined -- putting
        # ESO on top of it hides exactly what the figure is for. The widths still
        # differ so that "the two agree here" (an orange rim around the blue) cannot be
        # confused with "only one line was drawn".
        # The legend names the ESO curve "pipeline" -- the label is what a reader outside
        # this repo calls it; the console table below keeps "ESO", which names the file.
        ax.plot(wl, eso[j], lw=1.3, color=C_ESO, label="pipeline", zorder=2,
                alpha=args.alpha)
        ax.plot(wl, ours[j], lw=0.7, color=C_OURS, label="ours", zorder=4)
        # One y range for both curves. Given their own, the larger residual would be
        # squeezed to look like the smaller one.
        #
        # Which range is not a cosmetic choice. The two pipelines can differ by two
        # orders of magnitude in a panel, so a rule that keeps every channel of both
        # visible spends the whole height on one of them and draws the other as a flat
        # line -- which is a statement about the axis, not about the data. Our curve is
        # the one being examined, so it is the one that is never allowed off the panel;
        # ESO is shown to a percentile and its tail is reported instead of drawn.
        if args.ylim_rule == "medfilt":
            lo, hi = panel_ylim(np.concatenate([ours[j], eso[j]]))
        else:
            lo, hi = despiked_range(ours[j])
            lo = min(lo, float(np.nanpercentile(eso[j], args.eso_pct)))
            hi = max(hi, float(np.nanpercentile(eso[j], 100 - args.eso_pct)))
            m = 0.08 * max(hi - lo, 1e-9)
            lo, hi = lo - m, hi + m
        ax.set_ylim(lo, hi)
        for lab, y in (("ours", ours[j]), ("ESO", eso[j])):
            off = np.flatnonzero((y < lo) | (y > hi))
            if off.size:
                worst = off[np.argmax(np.abs(y[off]))]
                clipped.append(f"    {nm:<18}{lab:<5}{off.size:>4} channel(s) off the "
                               f"panel; the largest is {y[worst]:.1f} at "
                               f"{wl[worst]:.1f} A")
        ax.set_ylabel("flux", fontsize=9)
        if not args.separate:
            # Stacked, the corner text is the only thing telling the panels apart. One
            # ring per file names itself in the filename, so the text is dropped there
            # rather than sitting on top of the curve.
            ax.text(0.004, 0.93, nm, transform=ax.transAxes, fontsize=10, va="top",
                    ha="left", bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.6))
        # The legend is left in drawing order -- ESO, then ours. The order the two
        # are drawn in is the thing that decides which one is legible, so having
        # the legend read the same way means the figure says out loud what it did.
        if args.separate:
            # Above the axes, so it never lands on a residual the panel exists to show.
            ax.legend(fontsize=11, loc="lower left", frameon=False, ncol=2,
                      bbox_to_anchor=(0, 1.005), borderaxespad=0)
        elif j == 0:
            ax.legend(fontsize=10, loc="upper right", frameon=False, ncol=2)
        if args.separate:
            ax.set_xlabel("wavelength [$\\AA$]")
        ax.set_xlim(*(args.xlim if args.xlim else (wl.min(), wl.max())))
    if not args.separate:
        axes[-1].set_xlabel("wavelength [$\\AA$]")
    if clipped:
        # Named, not silently cropped. A channel drawn off the panel is a measurement
        # about the cube that the reader cannot recover from the figure.
        print(f"\n  drawn off the panel (rule {args.ylim_rule}"
              + (f", eso-pct {args.eso_pct:g}" if args.ylim_rule != "medfilt" else "") + "):")
        print("\n".join(clipped))

    span = f"_{args.xlim[0]:.0f}-{args.xlim[1]:.0f}" if args.xlim else ""
    d = Path(args.out) if args.out else pointing_dir(name, "halo")
    if args.separate:
        for (f, _), nm in zip(figs, keep):
            o = d / f"outside_vs_eso_{slug(nm)}{span}.png"
            o.parent.mkdir(parents=True, exist_ok=True)
            f.savefig(o, dpi=args.dpi, bbox_inches="tight")
            plt.close(f)
            print(f"saved -> {o}")
    else:
        o = d if d.suffix == ".png" else d / f"outside_vs_eso{span}.png"
        o.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(o, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"\nsaved -> {o}")


if __name__ == "__main__":
    main()
