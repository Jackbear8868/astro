"""How the main source group is put back together from the seg IDs it was split
into -- a before/after comparison.

SExtractor's deblender splits Haro 11 into several pieces (it is a merging galaxy, so
it has several bright knots to begin with), and how many pieces it is split into, and
how, differs from one observation to the next. Any rule of the form "pick one seg ID"
will only get hold of one of those pieces, and when the downstream steps use it to
decide "how many px around it to exclude" and "which half to mask", they will mask the
wrong place.

What utils.main_source_group does: take the connected component containing the
brightest pixel (with no dilation), then require that a member's galaxy-branch
redshift differ from that of the main source group by no more than dz_max --
adjacency only says that they are siblings from the same deblend, and another object
superimposed on the galaxy would be adjacent just as well, so the redshift is what
separates it out.

    left   before: every seg ID inside the adjacent blob gets its own colour,
           labelled with its number
    right  after: those that passed the redshift criterion and were kept

    conda run -n astro python src/skymodel/evaluation/main_group_map.py -n 12 5 1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, arcsinh_stretch, load_field, pointing_dir  # noqa: E402
from utils import DZ_MAX, main_source_group  # noqa: E402



def main():
    ap = argparse.ArgumentParser(description="Main source group merging: before vs after")
    ap.add_argument("-n", type=int, nargs="+", default=[12, 5, 1])
    ap.add_argument("--dz-max", type=float, default=DZ_MAX,
                    help="maximum redshift difference from the main source to accept a member")
    ap.add_argument("--out-suffix", default="",
                    help="filename suffix so different settings do not overwrite each other")
    args = ap.parse_args()

    for n in args.n:
        W = ROOT / f"results/skymodel/p{n:02d}"
        seg, white, valid = load_field(W)
        wn = np.where(valid, white, np.nan)
        mg, ids, pk = main_source_group(seg, wn, W / "step04", args.dz_max)
        # without a redshift only the adjacency criterion is applied -- and the blob
        # "before filtering" is exactly what the left panel is meant to show
        all_ids = main_source_group(seg, wn)[1]

        # draw the whole field of view -- cropping would hide where the rejected
        # members fall, and "which things did not get taken in" is precisely what this
        # figure is there to answer
        y0, x0 = 0, 0
        sub = np.s_[:, :]

        stretched, vmax = arcsinh_stretch(white, valid)
        bg = stretched[sub]

        fig, ax = plt.subplots(1, 2, figsize=(15, 7.2))
        cmap = plt.cm.tab20(np.linspace(0, 1, 20))
        for a in ax:
            a.imshow(bg, origin="lower", cmap="gray", vmin=0, vmax=vmax)
            a.set_xticks([]); a.set_yticks([])

        # left: one colour per seg ID inside the adjacent blob (before filtering)
        for k, i in enumerate(all_ids):
            m = (seg == i)[sub]
            rgba = np.zeros(m.shape + (4,))
            rgba[m] = list(cmap[k % 20][:3]) + [0.55]
            ax[0].imshow(rgba, origin="lower")
            yy, xx = np.nonzero(m)
            if yy.size > 40:
                ax[0].text(xx.mean(), yy.mean(), str(i), color="w", fontsize=11,
                           fontweight="bold", ha="center", va="center",
                           path_effects=[pe.withStroke(linewidth=2.5,
                                                       foreground="k")])
        # everything on the figure is in English -- matplotlib's default font has no
        # Chinese glyphs
        ax[0].set_title(f"before ({len(all_ids)} sources in total)", fontsize=12)

        # right: after merging
        m = mg[sub]
        rgba = np.zeros(m.shape + (4,))
        rgba[m] = [1.0, 0.5, 0.05, 0.5]
        ax[1].imshow(rgba, origin="lower")
        ax[1].contour(m, levels=[0.5], colors="#ff7f0e", linewidths=1.6)
        # the connected component used by the adjacency criterion -- the members
        # rejected by the redshift are inside this component too
        lab, _ = ndimage.label(seg > 0)
        ax[1].contour((lab == lab[pk])[sub], levels=[0.5],
                      colors="#00e5ff", linewidths=0.9, linestyles="--")
        ax[1].plot(pk[1] - x0, pk[0] - y0, "w+", ms=14, mew=2)
        ax[1].set_title(f"after ({len(ids)} sources in total)", fontsize=12)

        f = {int(i): float(np.nansum(np.where(seg == i, white, 0)))
             for i in np.unique(seg) if i > 0}
        tot = sum(f.values()); mf = sum(f[i] for i in ids)
        rest = sorted((v_ for k_, v_ in f.items() if k_ not in ids), reverse=True)
        fig.suptitle(f"p{n:02d}", fontsize=14)
        fig.tight_layout()
        out = pointing_dir(f"p{n:02d}") / f"main_group{args.out_suffix}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"p{n:02d}: adjacent {len(all_ids)} sources -> redshift keeps "
              f"{len(ids)} {ids} -> {int(mg.sum()):,} px   source flux fraction "
              f"{100*mf/tot:.1f}%   rejected {sorted(set(all_ids) - set(ids))}"
              f"   saved -> {out}")


if __name__ == "__main__":
    main()
