"""Poster version of the template panel figures.

Same treatment as the spectrum figures -- doubled line width, enlarged axis text, axis
names just outside the row of numbers -- for plot_eigen.py's --mode panels layout. The
corner label naming each curve is kept.

--figsize is the size of the whole figure here, not the per-panel size plot_eigen.py
takes, so the aspect ratio asked for is the aspect ratio produced.

    conda run -n astro python eigen_poster.py --kind galaxy --figsize 8 6
    conda run -n astro python eigen_poster.py --kind star --figsize 9 11 \\
        --only-full --class-labels
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # evaluation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # skymodel
from common import POSTER
from plot_eigen import LABEL_TOP, MUSE_RANGE, N_SAMPLE, Z_HARO, load_curves

FIGURES = POSTER / "templates"

XTICKS = [5000, 6000, 7000, 8000, 9000]
NAME_PAD = 6

ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="galaxy")
ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(8, 6))
ap.add_argument("--fs", type=float, default=15, help="axis text size in points")
ap.add_argument("--hspace", type=float, default=0.25,
                help="vertical gap between panels, as a fraction of panel height")
ap.add_argument("--lw", type=float, default=1.8)
ap.add_argument("--muse", action="store_true", default=True)
ap.add_argument("--only-full", action="store_true")
ap.add_argument("--class-labels", action="store_true")
ap.add_argument("--only", nargs="+", default=None,
                help="draw only these curves, matched against the label or the file "
                     "name it came from (e.g. G K M); order follows the library, not "
                     "the order given here")
ap.add_argument("--suffix", default="")
args = ap.parse_args()
FS = args.fs

lo, hi, sample, labels = load_curves(args.kind, args.class_labels)
x0, x1 = (w / (1.0 + Z_HARO) for w in MUSE_RANGE)
x0, x1 = max(x0, lo), min(x1, hi)

lam = np.linspace(x0, x1, N_SAMPLE)
F = sample(lam)
keep = [j for j in range(F.shape[1])
        if not args.only_full or np.isfinite(F[:, j]).mean() > 0.99999]
F, labels = F[:, keep], [labels[j] for j in keep]
# The colour index is fixed before --only runs, so a curve keeps the colour it has in
# the full figure. Re-indexing after the cut would repaint G, K, M as the first three
# of the palette and stop the two figures agreeing.
cidx = list(range(len(labels)))
if args.only:
    want = {s.upper() for s in args.only}
    # Three ways to name a curve: its whole label ("component 1"), the class letter a
    # stellar file name starts with ("G"), or the last word of the label ("1").
    keep = [j for j, nm in enumerate(labels)
            if want & {nm.upper(), nm[0].upper(), nm.split()[-1].upper()}]
    if not keep:
        raise SystemExit(f"--only {args.only} matched none of {labels}")
    F, labels, cidx = F[:, keep], [labels[j] for j in keep], [cidx[j] for j in keep]
n = F.shape[1]
print(f"{args.kind}: {n} curves over {x0:.1f}-{x1:.1f} A rest")

tab10 = plt.get_cmap("tab10").colors
colours = [tab10[c % len(tab10)] for c in cidx]

fig, axes = plt.subplots(n, 1, sharex=True, figsize=tuple(args.figsize))
axes = list(np.atleast_1d(axes))
for j, ax in enumerate(axes):
    y = F[:, j]
    ax.axhline(0, lw=1.2, color="0.7", zorder=1)
    ax.plot(lam, y, lw=args.lw, color=colours[j], zorder=2)
    ax.set_xlim(x0, x1)
    # Room for the corner label, made by stretching the top until the curve's own
    # right-hand end clears LABEL_TOP -- plot_eigen.py's rule, kept so the panels
    # scale the same way they did before.
    y0, y1 = ax.get_ylim()
    e = max(1, F.shape[0] // 5)
    top = float(np.nanmax(y[-e:]))
    room = y0 + (top - y0) / LABEL_TOP if top > y0 else y1
    ax.set_ylim(y0, max(y1 + 0.10 * (y1 - y0), room))
    # Three ticks at most: the panels are short, and a full default tick set would
    # collide with itself once the numbers are this size.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.xaxis.set_major_locator(FixedLocator(XTICKS))
    ax.tick_params(labelsize=FS, length=6, width=1.2, pad=6)
    ax.grid(alpha=0.2)
    ax.text(0.996, 0.94, labels[j], transform=ax.transAxes, ha="right", va="top",
            fontsize=FS + 2, color=colours[j])
axes[-1].set_xlabel("wavelength [$\\AA$]", fontsize=FS, labelpad=NAME_PAD)
# One name for the whole column instead of one per panel: every panel measures the
# same quantity, and repeating it costs width that the numbers need.
fig.supylabel("flux", fontsize=FS)
fig.subplots_adjust(hspace=args.hspace)

stem = f"eigen_{args.kind}_panels_muse" + ("_onlyfull" if args.only_full else "")
out = FIGURES / f"{stem}{args.suffix}.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"saved -> {out}")
