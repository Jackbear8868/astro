"""step4's redshift scan for one source: reduced chi2 against z, stars and galaxy
side by side.

step4 classifies by scanning both model families over redshift and keeping whichever
reaches the lower reduced chi2 on the same channel set. There is no absolute
threshold, so the whole decision is the two minima of this figure -- and how far
apart they are is the confidence in it.

The two families are drawn in separate panels because they are scanned over ranges
that differ by two orders of magnitude: stars over +-0.005 (a velocity, the star is
in our own Galaxy), the galaxy eigenspectra over 0 to 1.5. On one x axis the stellar
scan would be a single vertical line. The y axis is shared, which is the axis the
comparison is actually made on.

    conda run -n astro python src/skymodel/evaluation/chi2_scan.py --work results/skymodel/p01 --id 10
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
from common import ROOT, pointing_dir  # noqa: E402
# Imported rather than repeated: the stellar library is drawn in two places and the
# order and colours have to agree between them, or the same template is a different
# colour on two figures of the same talk.
from plot_eigen import SPECTRAL_ORDER  # noqa: E402

C_GAL = "#e8710a"
C_MIN = "0.35"
LOG_RATIO = 20.0        # span above which the shared y axis switches to log


def scan_files(step04, sid, tag):
    """(stars, galaxy) scan arrays for one source, either possibly None."""
    def one(n):
        hits = sorted(step04.glob(f"scan{n}_id{sid}_{tag}.npz"))
        if not hits:
            return None
        if len(hits) > 1:
            raise SystemExit(f"{len(hits)} files match scan{n}_id{sid}_{tag}.npz; "
                             "narrow it with --tag")
        return np.load(hits[0])
    return one(1), one(2)


def curves(d, spectral=False):
    """(label, z, reduced chi2, colour index) per template, each sorted by z.

    The saved rows are ordered by chi2, not by z -- plotting them in file order
    would draw a curve that jumps back and forth across the axis.

    spectral=True puts the stellar library in Harvard order and ties each
    template's colour to its place in that sequence, so a template keeps its
    colour whichever source is being looked at.
    """
    out = []
    for tpl in dict.fromkeys(d["template"].tolist()):
        m = d["template"] == tpl
        z, r = d["z"][m], d["red_chi2"][m]
        o = np.argsort(z)
        out.append([tpl, z[o], r[o], 0])
    if spectral:
        out.sort(key=lambda c: (SPECTRAL_ORDER.find(c[0][0].upper()), c[0]))
        # Label with the spectral class alone. The subtype and luminosity class are
        # the same for every template in the library, so printing them repeats one
        # fact seven times in a legend that has to fit above half a panel.
        for c in out:
            c[0] = c[0][0].upper()
    else:
        out.sort(key=lambda c: np.nanmin(c[2]))
    for i, c in enumerate(out):
        c[3] = i
    return out


def draw(ax, cs, colours, side="left"):
    """One family's scan, with each curve's own minimum marked.

    The legend sits above the axes rather than inside: with seven curves that can
    each be the lowest, no corner is reliably free, and a box placed inside covers
    one of the curves being compared.

    The two panels anchor their legends to opposite ends of their own axes. Both
    anchored left, the wider one runs past its panel and lands on top of the other.
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
    ap.add_argument("--tag", default="*",
                    help="glob for the run tag in the scan filenames; with several "
                         "runs in one step04 this picks between them")
    ap.add_argument("--logy", choices=["auto", "on", "off"], default="auto",
                    help="auto uses log whenever the values span more than 20x")
    ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=(16, 6))
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    W = ROOT / args.work
    name = Path(args.work).name
    step04 = W / "step04"
    best_hits = sorted(step04.glob(f"best_{args.tag}.npz"))
    if not best_hits:
        raise SystemExit(f"no best_{args.tag}.npz under {step04}")
    if len(best_hits) > 1:
        raise SystemExit(f"{len(best_hits)} files match best_{args.tag}.npz; "
                         "narrow it with --tag")
    best = np.load(best_hits[0])
    run = best_hits[0].stem[len("best_"):]
    print(f"{name}: {best_hits[0].name}")

    ids = best["id"].tolist() if args.id == "all" else [int(args.id)]
    out_dir = Path(args.out_dir) if args.out_dir else pointing_dir(name, "template_fit")
    out_dir.mkdir(parents=True, exist_ok=True)
    tab10 = plt.get_cmap("tab10").colors

    for sid in ids:
        d1, d2 = scan_files(step04, sid, run)
        if d1 is None and d2 is None:
            print(f"  id {sid}: no scan files")
            continue
        k = best["id"].tolist().index(sid)
        won, tpl, z_best = str(best["group"][k]), str(best["template"][k]), float(best["z"][k])
        rs, rg = float(best["star_red_chi2"][k]), float(best["gal_red_chi2"][k])
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

        # The lower of the two minima is the classification, so it is drawn across
        # both panels: the decision is which curve dips below this line.
        for ax in (axs, axg):
            ax.axhline(min(rs, rg), lw=0.9, ls="--", color=C_MIN, zorder=1)
        if args.logy == "on" or (args.logy == "auto" and lo and hi / lo > LOG_RATIO):
            axs.set_yscale("log")
        else:
            pad = 0.06 * (hi - lo)
            axs.set_ylim(lo - pad, hi + pad)

        out = out_dir / f"chi2_scan_id{sid}_{run}.png"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  id {sid:>3}  {won:<7} {tpl:<7} z={z_best:>7.4f}  "
              f"star {rs:>10.3f}  galaxy {rg:>10.3f}  margin {margin:.2f}x  -> {out.name}")


if __name__ == "__main__":
    main()
