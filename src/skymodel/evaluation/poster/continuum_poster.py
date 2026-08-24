"""Poster version of the all-pointings sky continuum figure.

Same treatment as the ring figures: doubled line width, 30 pt axis text, axis names
just outside the row of numbers, legend written as its own transparent image.

The wavelength grids differ between pointings, so as in continuum_compare.py each
curve is drawn against its own axis and nothing is resampled.
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
from common import EVAL, ROOT
from continuum_compare import FIGURES, load

FS = 30
LW = 2.0
XTICKS = [5000, 6000, 7000, 8000, 9000]
NAME_PAD = 6

ap = argparse.ArgumentParser()
ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(22, 10))
ap.add_argument("--suffix", default="")
args = ap.parse_args()

names = [f"p{i:02d}" for i in range(1, 15)]
got = []
for name in names:
    r = load(ROOT / "results/skymodel" / name)
    if r is None:
        print(f"  skip {name}: no step03/sky_continuum.npy")
        continue
    got.append((name, *r))

cols = plt.get_cmap("tab20").colors

fig, ax = plt.subplots(figsize=tuple(args.figsize))
for i, (name, w, c) in enumerate(got):
    ax.plot(w, c, lw=LW, color=cols[i % len(cols)])
ax.set_xlim(min(w.min() for _, w, _ in got), max(w.max() for _, w, _ in got))

lo, hi = ax.get_ylim()
step = next(s for s in (1, 2, 5, 10, 20, 50, 100, 200, 500) if (hi - lo) / s <= 6)
ax.yaxis.set_major_locator(MultipleLocator(step))
ax.xaxis.set_major_locator(FixedLocator(XTICKS))
ax.tick_params(labelsize=FS, length=8, width=1.6, pad=8)
ax.set_xlabel("wavelength [$\\AA$]", fontsize=FS, labelpad=NAME_PAD)
ax.set_ylabel("flux", fontsize=FS, labelpad=NAME_PAD)

out = FIGURES / f"continuum_compare{args.suffix}.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"saved -> {out}   y step {step}")

# Two rows of seven, so the strip is wide rather than tall and can sit under the panel.
lfig = plt.figure(figsize=(16, 2.0))
handles = [Line2D([], [], color=cols[i % len(cols)], lw=LW * 1.5)
           for i in range(len(got))]
lfig.legend(handles, [n for n, _, _ in got], loc="center", ncol=7, frameon=False,
            fontsize=FS, handlelength=2.0, columnspacing=1.8)
lout = FIGURES / "continuum_compare_legend.png"
lfig.savefig(lout, dpi=300, bbox_inches="tight", transparent=True)
plt.close(lfig)
print(f"saved -> {lout}")
