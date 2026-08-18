"""The source ID map of any segmentation -- drawn the same way as utils.id_map.

The rows of utils.id_map() need to carry the group,
redshift and similar fields). But sometimes all we want to see is "which things does
this mask circle as sources", with nothing to do with the fitting -- for instance when
comparing the professor's 1 sigma and 2 sigma segmentations.

This script does one thing only: assemble the rows id_map() needs (id and centroid)
from one seg image, and leave the rest of the drawing entirely to utils.id_map, with
no second implementation. One figure can only have one way of being drawn in this
project, otherwise putting two of them side by side would make them look like
different data merely because the stretch, the colours, or the labelling rules
differ.

    conda run -n astro python src/skymodel/evaluation/seg_id_map.py \\
        --work results/skymodel/p01
    conda run -n astro python src/skymodel/evaluation/seg_id_map.py \\
        --work results/skymodel/p01 --seg data/wsky_seg/DATACUBE_FINAL_1_seg.fits
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL, ROOT
from utils import id_map

FIGURES = EVAL / "masking"


def main():
    ap = argparse.ArgumentParser(description="Source ID map for any segmentation")
    ap.add_argument("--work", required=True,
                    help="pointing work directory, e.g. results/skymodel/p01. "
                         "seg and background default to its step01/")
    ap.add_argument("--seg", default=None,
                    help="segmentation image to plot; defaults to step01/seg.fits in the work directory")
    ap.add_argument("--white", default=None,
                    help="background image; defaults to step01/whitelight.fits. "
                         "when using a different seg, the background must still be "
                         "that pointing's own whitelight -- a mismatched background "
                         "creates a false impression of misalignment")
    ap.add_argument("--min-area", type=int, default=1,
                    help="only plot sources with area >= this value; "
                         "tiny sources overlap when labeled")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    STEP01 = ROOT / args.work / "step01"
    white = fits.getdata(Path(args.white) if args.white
                         else STEP01 / "whitelight.fits")
    seg_path = Path(args.seg) if args.seg else STEP01 / "seg.fits"
    seg   = fits.getdata(seg_path)
    if seg.shape != white.shape:
        raise SystemExit(f"seg {seg.shape} and whitelight {white.shape} have different dimensions")

    ids, cnt = np.unique(seg[seg > 0], return_counts=True)
    rows = []
    for i, c in zip(ids, cnt):
        if c < args.min_area:
            continue
        y, x = np.nonzero(seg == i)
        # id_map only uses the group field when by_group=True, and here it is always
        # by_group=False, but the fields of rows have to be complete, otherwise
        # GROUP_COLOR over in utils raises KeyError.
        rows.append(dict(id=int(i), x=float(x.mean()), y=float(y.mean()),
                         group="galaxy"))

    # the filename carries the working directory name: step01/seg.fits has the same
    # name for all 14 pointings, so using only the seg filename would overwrite.
    name = f"{Path(args.work).name}_{seg_path.stem}"
    out = Path(args.out) if args.out else FIGURES / f"id_map_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    id_map(seg, white, rows, out, by_group=False)
    print(f"{name}: {len(ids)} sources, plotted {len(rows)} (area >= {args.min_area} px)")
    print(f"  source spaxels {int((seg > 0).sum()):,}  ({100 * (seg > 0).mean():.1f}% of field)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
