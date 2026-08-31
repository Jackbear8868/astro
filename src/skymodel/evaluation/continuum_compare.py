"""The final sky continuum of every pointing, on one figure.

The sky model is D = s * C_sky(lambda) + sum_k c_k * L_k(lambda): one continuum shape
per pointing, scaled per spaxel by s. Drawing them together shows how far the pointings
agree on that shape and how much they differ in level.

The wavelength grids are not interchangeable -- they differ in length and in start --
so stacking the arrays by index would silently compare different wavelengths. Each
curve is drawn against its own wavelength axis and nothing is resampled.

    conda run -n astro python src/skymodel/evaluation/continuum_compare.py \\
        --pointings p01 p02 p03 --out /tmp/c.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL, ROOT  # noqa: E402
from products import Run  # noqa: E402

FIGURES = EVAL / "sky_basis"


def load(work):
    """(wavelength, continuum) for one pointing, or None if step3 has not run."""
    if not (Path(work) / "step03/sky_continuum.npy").exists():
        return None
    run = Run(work)
    return run.wl, run.continuum


def main():
    ap = argparse.ArgumentParser(description="Compare the final sky continuum across pointings")
    ap.add_argument("--pointings", nargs="+", default=[f"p{i:02d}" for i in range(1, 15)],
                    help="pointing directory names under results/skymodel")
    ap.add_argument("--root", default="results/skymodel")
    ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(22, 7))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

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

    # A qualitative map: the pointing number labels a mosaic tile, not a quantity, so
    # a sequential map would imply an order that is not there.
    cols = plt.get_cmap("tab20").colors

    fig, a = plt.subplots(figsize=args.figsize)
    for i, (name, w, c) in enumerate(got):
        a.plot(w, c, lw=1.0, color=cols[i % len(cols)], label=name)
    a.set_xlim(min(w.min() for _, w, _ in got), max(w.max() for _, w, _ in got))
    a.set_xlabel("wavelength [$\\AA$]")
    a.set_ylabel("flux")
    a.legend(fontsize=11, loc="upper left", bbox_to_anchor=(1.005, 1.0),
             borderaxespad=0, frameon=False)

    out = Path(args.out) if args.out else FIGURES / "continuum_compare.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
