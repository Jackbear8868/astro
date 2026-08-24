"""Poster version of plot_basis.py's stacked top-N figure.

Thicker traces and larger axis text, at whatever aspect ratio is asked for. The lane
construction is plot_basis.py's and is not changed: a constant offset taken from a
percentile of the whole basis, traces clipped to half a lane, e0 on top.

    conda run -n astro python basis_poster.py --work results/skymodel/p01 --figsize 10 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # evaluation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # skymodel
from common import ROOT, pointing_dir
from plot_basis import SPAN_PCT, basis_colour

XTICKS = [5000, 6000, 7000, 8000, 9000]
NAME_PAD = 6

ap = argparse.ArgumentParser()
ap.add_argument("--work", required=True)
ap.add_argument("--basis", default="svd")
ap.add_argument("-K", type=int, default=30)
ap.add_argument("--top", type=int, default=5)
ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(10, 5))
ap.add_argument("--fs", type=float, default=15)
ap.add_argument("--lw", type=float, default=1.4)
ap.add_argument("--gain", type=float, default=1.0)
ap.add_argument("--suffix", default="")
args = ap.parse_args()

W = ROOT / args.work
wl = np.load(W / "step03/wavelength.npy")
B = np.load(W / "step03" / f"sky_basis_{args.basis}_K{args.K}.npy")
n = min(args.top, B.shape[0])

# The offset is set from the whole basis, not from the vectors drawn, so a trace looks
# the same here as in the full overview figure.
step = 2.2 * np.percentile(np.abs(B), SPAN_PCT)
half = 0.5 * step

fig, ax = plt.subplots(figsize=tuple(args.figsize))
for row in range(n):
    y0 = (n - 1 - row) * step
    ax.axhline(y0, color="0.85", lw=0.8, zorder=1)
    trace = np.clip(args.gain * B[row].astype(np.float64), -half, half)
    ax.plot(wl, trace + y0, lw=args.lw, color=basis_colour(row), zorder=2)
ax.set_xlim(wl.min(), wl.max())
ax.set_ylim(-step, n * step)
# No ticks: the vertical direction is an offset between lanes, not a quantity, and the
# stacking order already says which vector comes first. The axis name carries what the
# direction means.
ax.set_yticks([])
ax.xaxis.set_major_locator(FixedLocator(XTICKS))
ax.tick_params(labelsize=args.fs, length=8, width=1.6, pad=8)
ax.set_xlabel("wavelength [$\\AA$]", fontsize=args.fs, labelpad=NAME_PAD)
# "sky" earns its place: the galaxy template figure next to this one on a poster calls
# its curves components, and these are not those.
ax.set_ylabel("sky line basis", fontsize=args.fs, labelpad=NAME_PAD)

out = pointing_dir(W.name, "basis") / f"top{n}{args.suffix}.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"saved -> {out}")
