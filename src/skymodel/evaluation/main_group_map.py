"""How the main source group is put back together from the seg IDs it was split
into -- a before/after comparison, for any number of pointings at once.

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

step5 already draws this for the pointing it is running, into
step05/main_source_group.png,
by calling the same utils.plot_main_group. This script exists for the two things
that one cannot do: several pointings in one command, and the numbers underneath --
which IDs were rejected, and how much of the field's flux the group holds.

    conda run -n astro python src/skymodel/evaluation/main_group_map.py -n 12 5 1
    conda run -n astro python src/skymodel/evaluation/main_group_map.py -n 1 --step04 ...
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, load_field, pointing_dir, step04_dir  # noqa: E402
from utils import DZ_MAX, main_source_group, plot_main_group  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Main source group merging: before vs after")
    ap.add_argument("-n", type=int, nargs="+", default=[12, 5, 1])
    ap.add_argument("--dz-max", type=float, default=DZ_MAX,
                    help="maximum redshift difference from the main source to accept a member")
    ap.add_argument("--step04", default=None,
                    help="the step4 directory to take the redshifts from, e.g. "
                         "results/skymodel/p01/step04; by default it is read from "
                         "step5's meta.json, which records the run step5 itself used")
    ap.add_argument("--run", default=None,
                    help="a run directory under step05 to read that meta.json from")
    ap.add_argument("--out-suffix", default="",
                    help="filename suffix so different settings do not overwrite each other")
    args = ap.parse_args()

    for n in args.n:
        name = f"p{n:02d}"
        W = ROOT / "results/skymodel" / name
        seg, white, valid = load_field(W)
        wn = np.where(valid, white, np.nan)
        step04 = (Path(args.step04) if args.step04
                  else step04_dir(W, args.run) or W / "step04")
        mg, ids, pk = main_source_group(seg, wn, step04, args.dz_max)
        # without a redshift only the adjacency criterion is applied -- and the blob
        # "before filtering" is exactly what the left panel is meant to show
        all_ids = main_source_group(seg, wn)[1]

        out = pointing_dir(name) / f"main_group{args.out_suffix}.png"
        plot_main_group(seg, white, mg, ids, all_ids, pk, out, title=name)

        f = {int(i): float(np.nansum(np.where(seg == i, white, 0)))
             for i in np.unique(seg) if i > 0}
        tot = sum(f.values())
        mf = sum(f[i] for i in ids)
        print(f"{name}: adjacent {len(all_ids)} sources -> redshift keeps "
              f"{len(ids)} {ids} -> {int(mg.sum()):,} px   source flux fraction "
              f"{100 * mf / tot:.1f}%   rejected {sorted(set(all_ids) - set(ids))}"
              f"   step04 {step04}   saved -> {out}")


if __name__ == "__main__":
    main()
