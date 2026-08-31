"""The regions the sky is learned from -- which spaxels each of the two stages used.

The two sets differ, and that difference is the figure. One panel each, over the
white light:

    step3  sky basis   blank spaxels in the field of view, inside the sky_region box
                       when its apply_to names `basis`, and spectrally complete -- one
                       missing channel would hand the decomposition a fabricated zero.
    step5  s field     blank spaxels with a finite free solve, min_source_distance
                       from any source and min_main_source_distance from the main
                       group, inside the box when apply_to names `sky_amplitude`, and
                       within train_clip_sigma robust spreads of the median s.

Why the two differ
------------------
The basis learns the shape of the sky lines, the same in every blank spaxel, so PSF
wings from the source barely move it and the main source's halo is enough to avoid. s
learns the amplitude of the sky continuum, too close in shape to the galaxy's: source
light props s up, the model grows with it, and subtracting it eats the source.

Where the numbers come from
---------------------------
Every parameter comes from the meta.json the step wrote beside its products, never the
config, which can be edited after the run and leave the figure disagreeing with the
products. Spectral completeness no product records, so the wsky cube is counted here.

    conda run -n astro python src/skymodel/evaluation/sky_region_map.py --work results/skymodel/p01
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (ROOT, SEG_COLOR, arcsinh_stretch, load_field,  # noqa: E402
                    pointing_dir, step04_dir)
from products import fit_dirs, sky_amplitude_params  # noqa: E402
from utils import build_amplitude_field, main_source_group  # noqa: E402


# One colour per panel. Each has to stand off the grey background and the green
# segmentation contour, and the two are read side by side, so also off each other.
STEP3_COLOR = "#ff7f0e"
STEP5_COLOR = "#e377c2"
FILL_ALPHA = 0.72
HALO_ALPHA = 0.30
HALO_RADIUS = 2
# A set gets a halo only when growing it by HALO_RADIUS multiplies its area by more
# than this: a solid region only gains a rim, scattered spaxels grow many times over.
HALO_WHEN_AREA_GROWS_BY = 2.0
# The box crosses the fill, the sources and the dark background, so no single colour
# stays visible along its whole length; white outlined in black does.
BOX_COLOR = "#ffffff"
BOX_STROKE = [pe.withStroke(linewidth=4.0, foreground="black")]


def region_of(meta, prefix=""):
    """The spatial restriction a step recorded, as {xlim, ylim, exclude_box}.

    The pipeline already translated the config's single box into these keys in the
    step's meta.json, so the rule is not copied here. prefix is "train_" for step5.
    """
    keys = ("xlim", "ylim", "exclude_box")
    return {k: meta[prefix + k] for k in keys if meta.get(prefix + k)}


def region_mask(region, shape):
    """The spaxels a region keeps -- (ny, nx) boolean, all True if it is empty.

    xlim/ylim are half-open and keep what is inside; exclude_box includes both endpoints
    and drops what is inside, so which form was written says which the config box was.
    """
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    keep = np.ones(shape, bool)
    if "xlim" in region:
        keep &= (xx >= region["xlim"][0]) & (xx < region["xlim"][1])
    if "ylim" in region:
        keep &= (yy >= region["ylim"][0]) & (yy < region["ylim"][1])
    if "exclude_box" in region:
        y0, y1, x0, x1 = region["exclude_box"]
        keep &= ~((yy >= y0) & (yy <= y1) & (xx >= x0) & (xx <= x1))
    return keep


def region_rect(region, shape):
    """The rectangle to draw for a region -- ((x, y), width, height) or None.

    An unbounded side is recorded as a coordinate past the edge of the image, so the
    bounds are clipped first, or the axes would rescale to fit the rectangle.
    """
    ny, nx = shape
    if "exclude_box" in region:
        y0, y1, x0, x1 = region["exclude_box"]
        x0, x1 = max(x0, 0), min(x1, nx - 1)
        y0, y1 = max(y0, 0), min(y1, ny - 1)
        return (x0 - .5, y0 - .5), x1 - x0 + 1, y1 - y0 + 1
    if "xlim" in region or "ylim" in region:
        x0, x1 = region.get("xlim", [0, nx])
        y0, y1 = region.get("ylim", [0, ny])
        x0, x1 = max(x0, 0), min(x1, nx)
        y0, y1 = max(y0, 0), min(y1, ny)
        return (x0 - .5, y0 - .5), x1 - x0, y1 - y0
    return None


def channel_coverage(cube, chunk=200):
    """Fraction of the channels holding data in each spaxel -- (ny, nx) float.

    Neither step wrote this down, and step3 keeps the spaxels where it is exactly 1.
    Chunked in wavelength, because counting does not need the whole cube in memory.
    """
    with fits.open(cube, memmap=True) as hdul:
        data = hdul["DATA"].data
        nz = data.shape[0]
        n = np.zeros(data.shape[1:], np.int32)
        for j in range(0, nz, chunk):
            n += np.isfinite(np.asarray(data[j:j + chunk], np.float32)).sum(
                axis=0, dtype=np.int32)
    return n / nz


def layers_for(mask):
    """The (mask, alpha) layers one set is painted with, bottom first.

    A spaxel is a couple of screen pixels wide, so a set of scattered single spaxels
    would look like an empty panel. Such a set gets a halo -- itself grown by
    HALO_RADIUS, drawn faintly underneath -- to lend every spaxel some surround. The
    element is a disk, not the default cross, so the surround adds no direction.
    """
    yy, xx = np.mgrid[-HALO_RADIUS:HALO_RADIUS + 1, -HALO_RADIUS:HALO_RADIUS + 1]
    halo = ndimage.binary_dilation(mask, structure=yy**2 + xx**2 <= HALO_RADIUS**2)
    if halo.sum() > HALO_WHEN_AREA_GROWS_BY * mask.sum():
        return [(halo, HALO_ALPHA), (mask, FILL_ALPHA)]
    return [(mask, FILL_ALPHA)]


def draw_panel(ax, mask, background, vmax, seg, colour, rect, title):
    """One step's panel: its spaxel set over the white light, with the boundaries.

    The set is an RGBA layer per pass rather than a recolouring of the background, so
    the white light stays readable through it and the contour stays on top of both.
    """
    ax.imshow(background, origin="lower", cmap="gray", vmin=0, vmax=vmax)
    for m, alpha in layers_for(mask):
        rgba = np.zeros(mask.shape + (4,))
        rgba[m] = list(mcolors.to_rgb(colour)) + [alpha]
        ax.imshow(rgba, origin="lower")
    ax.contour(seg > 0, levels=[0.5], colors=SEG_COLOR, linewidths=0.5, alpha=0.85)
    # The box is drawn even where it coincides with the edge of the fill, so that
    # "this line was set by hand" can be told apart from a boundary the data drew.
    if rect is not None:
        (x, y), w, h = rect
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False, ec=BOX_COLOR,
                                        lw=2.0, ls="--", path_effects=BOX_STROKE))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12, family="monospace")


def check(label, recorded, computed):
    """Warn when a set rebuilt here does not have the size the step recorded.

    The two are separate paths to the same set, and a replaced cube or stale products
    part them with nothing in the figure looking wrong.
    """
    if recorded is not None and int(recorded) != int(computed):
        print(f"★ {label}: the step recorded {int(recorded):,} spaxels, "
              f"this figure rebuilds {int(computed):,}")


def main():
    ap = argparse.ArgumentParser(
        description="Where the sky is learned from: step3's spaxel set and step5's, one panel each")
    ap.add_argument("--work", required=True, help="pointing work directory, e.g. results/skymodel/p01")
    ap.add_argument("--run", default=None,
                    help="alternative run directory under step05; default is the pipeline's own step05/step06")
    ap.add_argument("--step04", default=None,
                    help="the step4 directory to take the redshifts from, e.g. "
                         "results/skymodel/p01/step04; by default it is read from "
                         "step5's meta.json, which records the run step5 itself used")
    args = ap.parse_args()

    W = ROOT / args.work
    run, _ = fit_dirs(W, args.run)
    m3 = json.loads((W / "step03" / "meta.json").read_text())
    m5 = json.loads((run / "meta.json").read_text())
    p = sky_amplitude_params(m5)

    seg, white, valid = load_field(W)
    coverage = channel_coverage(ROOT / m3["cube"])

    # --- step3: blank, the sky_region box, and spectrally complete spaxels only ---
    basis_region = region_of(m3)
    blank3 = valid & (seg == 0)
    basis = blank3 & region_mask(basis_region, seg.shape) & (coverage == 1)
    check("step3 sky basis", m3.get("n_blank_complete"), basis.sum())

    # --- step5: further off the sources, and no failed fits. The free solve is NaN
    #     wherever step5 did not fit, so its finite spaxels start the cuts below ---
    s = np.load(run / "sky_continuum_amplitude_per_spaxel.npy").astype(float)
    ok = valid & (seg == 0) & np.isfinite(s)
    check("step5 free solve", m5.get("n_blank"), ok.sum())
    train_region = region_of(p, "train_")
    step04 = (Path(args.step04) if args.step04
              else step04_dir(W, args.run) or W / "step04")
    main_group, _, _ = main_source_group(
        seg, np.where(valid, white, np.nan), step04, dz_max=p["main_source_dz"])
    _, sfield = build_amplitude_field(
        s, seg, ok, p["min_source_distance"], p["min_main_source_distance"] or None,
        p["train_clip_sigma"],
        exclude=~region_mask(train_region, seg.shape) if train_region else None,
        main=main_group, n_iter=p["n_iter"])
    check("step5 s field training set", m5.get("n_train"), sfield.sum())

    n_valid = int(valid.sum())
    n3, n5 = int(basis.sum()), int(sfield.sum())

    # --- the figure: one panel per step, nothing in either but its own set ---
    fig, (ax3, ax5) = plt.subplots(1, 2, figsize=(13.5, 7.0),
                                   sharex=True, sharey=True)
    img, vmax = arcsinh_stretch(white, valid)
    rect3 = region_rect(basis_region, seg.shape)
    rect5 = region_rect(train_region, seg.shape)
    draw_panel(ax3, basis, img, vmax, seg, STEP3_COLOR, rect3,
               f"step3  -  C_sky and line basis\n"
               f"{n3:,}   ({100 * n3 / n_valid:.1f}% of the field)")
    draw_panel(ax5, sfield, img, vmax, seg, STEP5_COLOR, rect5,
               f"step5  -  s(x,y) training set\n"
               f"{n5:,}   ({100 * n5 / n_valid:.1f}% of the field)")
    fig.suptitle(f"{W.name}    where the sky is learned from", fontsize=14, y=0.99)

    # The only prose in the figure: the two line styles are the boundaries the spaxel
    # set did not draw itself.
    boxed = [name for name, rect in (("step3", rect3), ("step5", rect5)) if rect]
    caption = "green: segmentation"
    if boxed:
        caption += ("    dashed white: the sky_region box, applied to "
                    + " and ".join(boxed))
    fig.text(0.5, 0.02, caption, ha="center", fontsize=10, color="0.25")

    # The panels are laid out between the caption and the suptitle: both belong to the
    # figure, so the axes do not know they are there and a wide field runs into them.
    fig.tight_layout(rect=[0, 0.035, 1, 0.93])
    # The run goes into the filename: a pointing can hold several step05 runs, all
    # writing the same amplitude files, so otherwise one figure replaces the other.
    tag = "" if args.run in (None, "default") else f"_{args.run}"
    out = pointing_dir(W) / f"sky_region{tag}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"{W.name}  valid {n_valid:,}   "
          f"step3 {n3:,} ({100 * n3 / n_valid:.1f}%)   "
          f"step5 {n5:,} ({100 * n5 / n_valid:.1f}%)")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
