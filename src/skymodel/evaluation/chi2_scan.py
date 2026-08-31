"""step4's redshift scan for one source: reduced chi2 against z, stars and galaxy
side by side.

step4 classifies by scanning both model families over redshift and keeping whichever
reaches the lower reduced chi2 on the same channel set. There is no absolute threshold,
so the decision is the two minima of this figure and their separation is the confidence
in it. The families get separate panels because their scan ranges differ by orders of
magnitude -- a stellar radial velocity against a galaxy redshift -- and on one x axis
the stellar scan would be a single vertical line. The shared y axis is the axis the
comparison is made on.

    conda run -n astro python src/skymodel/evaluation/chi2_scan.py --work results/skymodel/p01 --id all
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
from products import Run  # noqa: E402
from utils import load_scan  # noqa: E402
# Imported rather than repeated: the stellar library is drawn in two places, and the
# order and colours have to agree or one template gets two colours.
from plot_eigen import SPECTRAL_ORDER  # noqa: E402

C_GAL = "#e8710a"
C_MIN = "0.35"
LOG_RATIO = 20.0        # span above which the shared y axis switches to log


def scan_files(step04, sid):
    """(stars, galaxy) scan arrays for one source, either possibly None.

    One file per branch, named after it, because z means a radial velocity on the star
    side and a redshift on the galaxy side.
    """
    def one(branch):
        try:
            return load_scan(step04, branch, sid)
        except SystemExit:
            return None
    return one("star"), one("galaxy")


def curves(d, spectral=False):
    """(label, z, reduced chi2, colour index) per template, each sorted by z.

    The saved rows are ordered by chi2, not by z, so file order would draw a curve
    jumping back and forth. spectral=True puts the stellar library in Harvard order and
    ties each colour to a place in that sequence, so a template keeps its colour.
    """
    out = []
    for tpl in dict.fromkeys(d["template"].tolist()):
        m = d["template"] == tpl
        z, r = d["z"][m], d["red_chi2"][m]
        o = np.argsort(z)
        out.append([tpl, z[o], r[o], 0])
    if spectral:
        out.sort(key=lambda c: (SPECTRAL_ORDER.find(c[0][0].upper()), c[0]))
        # Spectral class alone: subtype and luminosity class are shared across the
        # library, so printing them repeats one fact in every legend entry.
        for c in out:
            c[0] = c[0][0].upper()
    else:
        out.sort(key=lambda c: np.nanmin(c[2]))
    for i, c in enumerate(out):
        c[3] = i
    return out


def draw(ax, cs, colours, side="left"):
    """One family's scan, with each curve's own minimum marked.

    The legend sits above the axes: any curve can be the lowest, so no corner is
    reliably free and a box inside would cover one of the curves being compared. The
    two panels anchor to opposite ends, or the wider legend lands on the other panel.
    """
    for tpl, z, r, ci in cs:
        col = colours[ci % len(colours)]
        ax.plot(z, r, lw=1.0, color=col, label=tpl)
        k = int(np.nanargmin(r))
        ax.plot(z[k], r[k], "o", ms=4, color=col, zorder=5)
    ax.set_xlabel("z")
    ax.legend(fontsize=9, frameon=False, ncol=len(cs), borderaxespad=0,
              handlelength=1.2, columnspacing=0.9,
              loc=f"lower {side}", bbox_to_anchor=(0 if side == "left" else 1, 1.005))
    ax.grid(alpha=0.2)


def main():
    ap = argparse.ArgumentParser(description="step4's reduced chi2 against redshift")
    ap.add_argument("--work", required=True)
    ap.add_argument("--id", default="all", help="source id, or all")
    ap.add_argument("--step04", default=None,
                    help="the step4 directory to draw, e.g. "
                         "results/skymodel/p01/step04/mask_iter2; default is the "
                         "work directory's own step04")
    ap.add_argument("--logy", choices=["auto", "on", "off"], default="auto",
                    help="auto uses log whenever the values span more than 20x")
    ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(16, 6))
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    run = Run(args.work)
    name = Path(args.work).name
    step04 = Path(args.step04) if args.step04 else run.work / "step04"
    fit_file = step04 / "source_fits.npz"
    if not fit_file.exists():
        raise SystemExit(f"no source_fits.npz under {step04}")
    source_fits = np.load(fit_file)
    # step4 writes the whole chi2 curve only when asked, so its absence is a config
    # to change, not a lost file.
    if not (step04 / "scans_galaxy.npz").exists():
        raise SystemExit(
            f"no scans_*.npz under {step04} -- this figure is the redshift scans "
            "themselves, and step4 keeps them only when source_fit.keep_scans is "
            "true in the config; set it and rerun step4")
    print(f"{name}: {fit_file}")

    ids = source_fits["id"].tolist() if args.id == "all" else [int(args.id)]
    out_dir = Path(args.out_dir) if args.out_dir else run.figdir("template_fit")
    out_dir.mkdir(parents=True, exist_ok=True)
    tab10 = plt.get_cmap("tab10").colors

    for sid in ids:
        d1, d2 = scan_files(step04, sid)
        if d1 is None and d2 is None:
            print(f"  id {sid}: no scan files")
            continue
        k = source_fits["id"].tolist().index(sid)
        won, tpl, z_best = (str(source_fits["group"][k]), str(source_fits["template"][k]),
                            float(source_fits["z"][k]))
        rs, rg = float(source_fits["star_red_chi2"][k]), float(source_fits["gal_red_chi2"][k])
        margin = max(rs, rg) / min(rs, rg)

        fig, (axs, axg) = plt.subplots(1, 2, sharey=True, figsize=args.figsize,
                                       gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.03})
        lo = hi = None
        if d1 is not None:
            cs = curves(d1, spectral=True)
            draw(axs, cs, tab10)
            lo, hi = (np.nanmin([np.nanmin(c[2]) for c in cs]),
                      np.nanmax([np.nanmax(c[2]) for c in cs]))
        if d2 is not None:
            cg = curves(d2)
            draw(axg, cg, [C_GAL], side="right")
            g = (np.nanmin([np.nanmin(c[2]) for c in cg]),
                 np.nanmax([np.nanmax(c[2]) for c in cg]))
            lo = g[0] if lo is None else min(lo, g[0])
            hi = g[1] if hi is None else max(hi, g[1])
        axs.set_ylabel("reduced $\\chi^2$")

        # The lower minimum is the classification, so the line crosses both panels.
        for ax in (axs, axg):
            ax.axhline(min(rs, rg), lw=0.9, ls="--", color=C_MIN, zorder=1)
        if args.logy == "on" or (args.logy == "auto" and lo and hi / lo > LOG_RATIO):
            axs.set_yscale("log")
        else:
            pad = 0.06 * (hi - lo)
            axs.set_ylim(lo - pad, hi + pad)

        out = out_dir / f"chi2_scan_id{sid}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  id {sid:>3}  {won:<7} {tpl:<7} z={z_best:>7.4f}  "
              f"star {rs:>10.3f}  galaxy {rg:>10.3f}  margin {margin:.2f}x  -> {out.name}")


if __name__ == "__main__":
    main()
