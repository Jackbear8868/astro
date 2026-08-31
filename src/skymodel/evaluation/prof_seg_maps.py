"""What the 14 segmentations given by the professor look like -- one ID map per
pointing, plus one overview.

Each pointing comes with two files besides the cube:

    DATACUBE_FINAL_{N}_pseudo_r.fits    the background image used for detection, the
                                        equally weighted average of the nosky cube
                                        over 5625-6825 A
    DATACUBE_FINAL_{N}_seg.fits         the segmentation produced on that image

This script overlays the two and draws them, on the professor's own pseudo_r rather
than our whitelight -- the mask grew out of pseudo_r, and a different background would
give the illusion of a misalignment. The drawing is left to products.id_map, with no
second implementation: two figures of one kind drawn by different rules would look
like different data because the stretch, colours or labelling differ.

    conda run -n astro python src/skymodel/evaluation/prof_seg_maps.py -n 4 8 12
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL, ROOT, arcsinh_stretch, pointing_dir  # noqa: E402
from products import id_map  # noqa: E402

SEGDIR  = ROOT / "data/wsky_seg"
FIGURES = EVAL / "masking/prof_seg"    # overview.png belongs to no single pointing


def load(n):
    seg = fits.getdata(SEGDIR / f"DATACUBE_FINAL_{n}_seg.fits").astype(int)
    img = np.asarray(fits.getdata(SEGDIR / f"DATACUBE_FINAL_{n}_pseudo_r.fits"), float)
    return seg, img


def rows_from(seg, min_area):
    """The rows id_map needs: the number and the centroid of each source."""
    ids, cnt = np.unique(seg[seg > 0], return_counts=True)
    out = []
    for i, c in zip(ids, cnt):
        if c < min_area:
            continue
        y, x = np.nonzero(seg == i)
        # id_map only uses the group field when by_group=True, but the fields of rows
        # have to be complete, otherwise GROUP_COLOR in products raises KeyError.
        out.append(dict(id=int(i), x=float(x.mean()), y=float(y.mean()),
                        group="galaxy"))
    return out, len(ids)


def overview(ns, out_path):
    """All 14 pointings seen together -- a single figure cannot answer "is this set of
    masks consistent across the whole mosaic"."""
    ncol = 5
    nrow = int(np.ceil(len(ns) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.5 * nrow))
    for a in np.ravel(ax):
        a.axis("off")
    for k, n in enumerate(ns):
        seg, img = load(n)
        a = np.ravel(ax)[k]
        a.axis("on"); a.set_xticks([]); a.set_yticks([])
        # the stretch matches id_map, so the overview and the single figures look
        # like the same data.
        bg, vmax = arcsinh_stretch(img)
        a.imshow(bg, origin="lower", cmap="gray", vmin=0, vmax=vmax)
        a.contour(seg > 0, levels=[0.5], colors="#1f77b4", linewidths=0.6)
        n_src = int(seg.max())
        a.set_title(f"#{n}   {n_src} src   "
                    f"{100 * (seg > 0).mean():.1f}% masked", fontsize=9)
    fig.suptitle("professor's segmentation on pseudo-r  (blue = source)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Professor's segmentation ID maps")
    ap.add_argument("-n", type=int, nargs="+", default=list(range(1, 15)),
                    help="which pointings to plot")
    ap.add_argument("--min-area", type=int, default=1,
                    help="only label sources with area >= this value; tiny sources overlap when labeled")
    ap.add_argument("--no-overview", action="store_true")
    args = ap.parse_args()

    print(f"{'':>4}{'n_src':>6}{'src_px':>10}{'field%':>8}"
          f"{'top-3 flux share':>18}")
    for n in args.n:
        seg, img = load(n)
        rows, n_all = rows_from(seg, args.min_area)
        out = pointing_dir(f"p{n:02d}") / "segmentation_map.png"
        id_map(seg, img, rows, out, by_group=False)

        # A main source split into several pieces shows up as a divided flux share,
        # so measure it here rather than in a separate script. The flux is integrated
        # over pseudo_r, the quantity detection used.
        flux = {int(i): float(np.nansum(np.where(seg == i, img, 0)))
                for i in np.unique(seg) if i > 0}
        tot  = sum(flux.values()) or 1.0
        top  = sorted(flux.values(), reverse=True)[:3]
        print(f"#{n:<3}{n_all:>6}{int((seg > 0).sum()):>10,}"
              f"{100 * (seg > 0).mean():>7.1f}%"
              + "".join(f"{100 * f / tot:>6.1f}%" for f in top))
    print("\none map per pointing -> results/skymodel/evaluation/pNN/segmentation_map.png")
    if not args.no_overview:
        overview(args.n, FIGURES / "overview.png")


if __name__ == "__main__":
    main()
