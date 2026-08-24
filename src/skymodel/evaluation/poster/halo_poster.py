"""Poster version of the outside-ring raw/signal figure.

Thicker curves, doubled axis text, only the extreme ticks kept, the axis names sitting
next to the number they name, and the legend written as its own transparent image.

The zone means are cached to an .npy next to this script: the two cubes are ~3 GB each
and re-running only to nudge a font size should not read them again.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # evaluation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # skymodel
from common import EVAL, ROOT, load_field, pointing_dir, slug
from halo_spectra import C_LINE, CHUNK, LINES, Z_HARO, zone_labels
from utils import DZ_MAX, main_source_group

C_RAW = "0.55"
C_SIGNAL = "#d62728"
RINGS = [0, 10, 25, 50]
LAYERS = 4

LW = 1.4            # twice the 0.7 the screen version used
FS = 30
YTICKS = [0, 200, 400, 600, 800, 1000]
XTICKS = [5000, 6000, 7000, 8000, 9000]
NAME_PAD = 6        # gap between the row of numbers and the axis name, in points
WANT = "outside 0-10 px"

ap = argparse.ArgumentParser()
ap.add_argument("--pointing", default="p01")
ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(20, 5))
ap.add_argument("--suffix", default="", help="appended to the output name, so a trial "
                                             "aspect ratio does not replace the kept one")
ap.add_argument("--zoom", action="store_true",
                help="the signal-level cut instead of the full raw range")
args = ap.parse_args()

if args.zoom:
    YLIM, YTICKS = (-10, 50), [0, 10, 20, 30, 40, 50]
else:
    YLIM, YTICKS = None, YTICKS      # None: the bottom follows the data, the top is 1000


def zone_means(cube_path, zones, keys, nz):
    idx = [np.flatnonzero((zones == k).ravel()) for k in keys]
    out = np.full((len(keys), nz), np.nan)
    with fits.open(cube_path, memmap=True) as h:
        hdu = h["DATA"] if "DATA" in h else h[0]
        for c0 in range(0, nz, CHUNK):
            c1 = min(c0 + CHUNK, nz)
            block = np.asarray(hdu.data[c0:c1], np.float32).reshape(c1 - c0, -1)
            with np.errstate(invalid="ignore"):
                for j, ix in enumerate(idx):
                    if ix.size:
                        out[j, c0:c1] = np.nanmean(block[:, ix], axis=1)
            print(f"    {cube_path.name[:28]:<28} {c1}/{nz}", end="\r", flush=True)
    print(" " * 50, end="\r")
    return out


name = args.pointing
work = ROOT / "results/skymodel" / name
# The cache goes under results, not next to this file: everything a script writes
# belongs in results/skymodel/evaluation, and one cache per pointing because the zone
# means are tied to that field.
CACHE = EVAL / "poster_cache" / f"halo_{name}.npz"
wl = np.load(work / "step03/wavelength.npy")

seg, white, valid = load_field(work)
main_, ids, _ = main_source_group(seg, np.where(valid, white, np.nan), None, DZ_MAX)
zones, names = zone_labels(seg, white, valid, main_, LAYERS, RINGS)
keys = [i + 1 for i, nm in enumerate(names) if nm.startswith("outside")]
keep = [names[k - 1] for k in keys]

if CACHE.exists():
    z = np.load(CACHE)
    raw, signal = z["raw"], z["signal"]
    print(f"cached -> {CACHE}")
else:
    meta = json.loads((work / "step03/meta.json").read_text())
    raw = zone_means(ROOT / meta["cube"], zones, keys, wl.size)
    signal = zone_means(work / "step06/sky_subtracted.fits", zones, keys, wl.size)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, raw=raw, signal=signal)
    print(f"cached -> {CACHE}")

j = keep.index(WANT)
outdir = pointing_dir(name, "halo")

fig, ax = plt.subplots(figsize=tuple(args.figsize))
for _, lam in LINES:
    ax.axvline(lam * (1 + Z_HARO), ymin=0.92, ymax=1.0, lw=2.8, color=C_LINE, zorder=3)
ax.axhline(0, lw=1.6, color="0.55")
ax.plot(wl, raw[j], lw=LW, color=C_RAW, zorder=2)
ax.plot(wl, signal[j], lw=LW, color=C_SIGNAL, zorder=4)

if YLIM:
    ax.set_ylim(*YLIM)
else:
    ylo = min(float(np.nanmin(signal[j])), float(np.nanmin(raw[j])))
    ax.set_ylim(ylo - 0.05 * (1000 - ylo), 1000)
ax.set_xlim(wl.min(), wl.max())
ax.yaxis.set_major_locator(FixedLocator(YTICKS))
ax.xaxis.set_major_locator(FixedLocator(XTICKS))
ax.tick_params(labelsize=FS, length=8, width=1.6, pad=8)
# Just outside the row of numbers rather than inside it, so no tick has to be dropped
# to make room. labelpad is measured from the numbers, not from the axis, so it stays
# correct if the tick font size changes.
ax.set_xlabel("wavelength [$\\AA$]", fontsize=FS, labelpad=NAME_PAD)
ax.set_ylabel("flux", fontsize=FS, labelpad=NAME_PAD)

tag = "zoom_" if args.zoom else ""
out = outdir / f"outside_raw_vs_signal_{tag}{slug(WANT)}{args.suffix}.png"
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"saved -> {out}")

# The legend on its own, transparent, so it can be placed anywhere on the poster.
lfig = plt.figure(figsize=(6, 1.0))
handles = [Line2D([], [], color=C_RAW, lw=LW * 2),
           Line2D([], [], color=C_SIGNAL, lw=LW * 2)]
lfig.legend(handles, ["raw", "signal"], loc="center", ncol=2, frameon=False,
            fontsize=FS * 1.4, handlelength=2.4, columnspacing=2.5)
lout = outdir / "outside_raw_vs_signal_legend.png"
lfig.savefig(lout, dpi=300, bbox_inches="tight", transparent=True)
plt.close(lfig)
print(f"saved -> {lout}")
