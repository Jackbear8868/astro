"""The source ID map of any segmentation -- drawn the same way as products.id_map.

products.id_map()'s rows normally carry the group, redshift and similar fit fields, but
sometimes the question is only which things a mask circles as sources, with nothing to
do with the fitting -- comparing two segmentations of the same field, say.

So this assembles the rows id_map() needs (id and centroid) from one seg image and
leaves the drawing to products.id_map, with no second implementation: two figures of one
kind drawn by different rules would look like different data because the stretch,
colours or labelling differ.

    conda run -n astro python src/skymodel/evaluation/seg_id_map.py \\
        --work results/skymodel/p01 --seg data/wsky_seg/DATACUBE_FINAL_1_seg.fits
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL, map_name, seg_and_background
from products import id_map

FIGURES = EVAL / "masking"


def main():
    ap = argparse.ArgumentParser(description="Source ID map for any segmentation")
    ap.add_argument("--work", required=True,
                    help="pointing work directory, e.g. results/skymodel/p01. "
                         "seg and background default to its step01/")
    ap.add_argument("--seg", default=None,
                    help="segmentation image to plot; defaults to "
                         "step01/segmentation_input.fits in the work directory")
    ap.add_argument("--white", default=None,
                    help="background image; defaults to step01/whitelight_nosky.fits. "
                         "when using a different seg, the background must still be "
                         "that pointing's own whitelight -- a mismatched background "
                         "creates a false impression of misalignment")
    ap.add_argument("--min-area", type=int, default=1,
                    help="only plot sources with area >= this value; "
                         "tiny sources overlap when labeled")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seg, white, seg_path = seg_and_background(args.work, args.seg, args.white)

    ids, cnt = np.unique(seg[seg > 0], return_counts=True)
    rows = []
    for i, c in zip(ids, cnt):
        if c < args.min_area:
            continue
        y, x = np.nonzero(seg == i)
        # id_map reads group only when by_group=True, but the field still has to be
        # present or GROUP_COLOR over in products raises KeyError.
        rows.append(dict(id=int(i), x=float(x.mean()), y=float(y.mean()),
                         group="galaxy"))

    name = map_name(args.work, seg_path)
    out = Path(args.out) if args.out else FIGURES / f"id_map_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    id_map(seg, white, rows, out, by_group=False)
    print(f"{name}: {len(ids)} sources, plotted {len(rows)} (area >= {args.min_area} px)")
    print(f"  source spaxels {int((seg > 0).sum()):,}  ({100 * (seg > 0).mean():.1f}% of field)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
