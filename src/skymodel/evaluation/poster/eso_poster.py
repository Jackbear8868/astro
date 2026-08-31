"""Poster version of the ours-vs-pipeline ring figure.

Same treatment as halo_poster.py -- doubled line widths, 30 pt axis text, axis names
just outside the row of numbers, legend written separately -- applied to the comparison
against the ESO cube instead of against the raw one.

The y range follows zone_spectra.py's first-pct rule so the panel keeps saying what
that figure said; only the tick spacing is chosen here, to land on round numbers.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # evaluation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # skymodel
from common import EVAL, ROOT, slug
from products import Run
from spectra import C_ESO, C_LINE, C_OURS, LINES, Z_HARO, despiked_range
from zones import zone_labels, zone_means
from utils import DZ_MAX, main_source_group

RINGS = [0, 10, 25, 50]
LAYERS = 4
FS = 30
XTICKS = [5000, 6000, 7000, 8000, 9000]
NAME_PAD = 6
ESO_PCT = 5.0
A_ESO = 0.75
WANT = "outside 0-10 px"

ap = argparse.ArgumentParser()
ap.add_argument("--pointing", default="p01")
ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(20, 10))
ap.add_argument("--suffix", default="")
args = ap.parse_args()

name = args.pointing
run = Run(ROOT / "results/skymodel" / name)
work = run.work
# One cache per pointing: the zone means are what the two cube reads produce, and they
# are not interchangeable between fields.
CACHE = EVAL / "poster_cache" / f"eso_{name}.npz"
wl = run.wl

seg, white, valid = run.seg, run.white, run.valid
main_, ids, _ = main_source_group(seg, np.where(valid, white, np.nan), None, DZ_MAX)
zones, names = zone_labels(seg, white, valid, main_, LAYERS, RINGS)
keys = [i + 1 for i, nm in enumerate(names) if nm.startswith("outside")]
keep = [names[k - 1] for k in keys]

if CACHE.exists():
    z = np.load(CACHE)
    ours, eso = z["ours"], z["eso"]
else:
    ours = zone_means(run.cube, zones, keys, wl.size)
    eso = zone_means(run.nosky, zones, keys, wl.size)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, ours=ours, eso=eso)
print(f"cached -> {CACHE}")

j = keep.index(WANT)
outdir = run.figdir("halo", poster=True)

fig, ax = plt.subplots(figsize=tuple(args.figsize))
for _, lam in LINES:
    ax.axvline(lam * (1 + Z_HARO), ymin=0.92, ymax=1.0, lw=2.8, color=C_LINE, zorder=3)
ax.axhline(0, lw=1.6, color="0.55")
# The pair of widths is zone_spectra's: the reference curve underneath and thicker,
# ours on top and thinner, both doubled from the screen version.
ax.plot(wl, eso[j], lw=2.6, color=C_ESO, alpha=A_ESO, zorder=2)
ax.plot(wl, ours[j], lw=1.4, color=C_OURS, zorder=4)

lo, hi = despiked_range(ours[j])
lo = min(lo, float(np.nanpercentile(eso[j], ESO_PCT)))
hi = max(hi, float(np.nanpercentile(eso[j], 100 - ESO_PCT)))
m = 0.08 * (hi - lo)
ax.set_ylim(lo - m, hi + m)
ax.set_xlim(wl.min(), wl.max())
# A round tick every step, with step the smallest that keeps the count under six.
step = next(s for s in (1, 2, 5, 10, 20, 50, 100) if (hi - lo) / s <= 6)
ax.yaxis.set_major_locator(MultipleLocator(step))
ax.xaxis.set_major_locator(FixedLocator(XTICKS))
ax.tick_params(labelsize=FS, length=8, width=1.6, pad=8)
ax.set_xlabel("wavelength [$\\AA$]", fontsize=FS, labelpad=NAME_PAD)
ax.set_ylabel("flux", fontsize=FS, labelpad=NAME_PAD)

for lab, y in (("ours", ours[j]), ("pipeline", eso[j])):
    off = np.flatnonzero((y < lo - m) | (y > hi + m))
    if off.size:
        w = off[np.argmax(np.abs(y[off]))]
        print(f"  {lab:<9}{off.size:>4} channel(s) off the panel; "
              f"largest {y[w]:.1f} at {wl[w]:.1f} A")

out = outdir / f"outside_vs_eso_{slug(WANT)}{args.suffix}.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"saved -> {out}")

lfig = plt.figure(figsize=(6, 1.0))
handles = [Line2D([], [], color=C_ESO, lw=2.6, alpha=A_ESO),
           Line2D([], [], color=C_OURS, lw=2.6)]
lfig.legend(handles, ["pipeline", "ours"], loc="center", ncol=2, frameon=False,
            fontsize=FS * 1.4, handlelength=2.4, columnspacing=2.5)
lout = outdir / "outside_vs_eso_legend.png"
lfig.savefig(lout, dpi=300, bbox_inches="tight", transparent=True)
plt.close(lfig)
print(f"saved -> {lout}")
