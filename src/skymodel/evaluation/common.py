"""Things shared by the scripts under evaluation -- display stretch, colour scales,
and the one cube read more than one of them does.

One condition for putting something here: two or more scripts do the same thing.
Separate copies drift apart, and a stretch or colour centre that differs between two
figures looks like a change in the data.

Reading a run's products is not here: that is `products.Run`, which serves the
standalone steps as well and so cannot live under evaluation.

The file must not be named utils.py: the scripts point sys.path at the level above to
import `src/skymodel/utils.py`, and a same-named file would shadow it.
"""
import os
import re
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
# Where figures that belong to no single run go -- the ones comparing several
# pointings, which have no one run directory to sit beside. SKYMODEL_EVAL moves them,
# so a checkout kept read-only, or results kept on another disk, still has somewhere
# to write. Figures that do belong to one run use pointing_dir instead and follow it
# without being told.
EVAL = Path(os.environ.get(
    "SKYMODEL_EVAL", ROOT / "results/skymodel/evaluation")).expanduser()

# Where the print versions go. They carry the same filenames as the screen figures
# they are versions of, so they need a level of their own or they overwrite them.
# Run.figdir(poster=True) is the same rule for a figure that belongs to one pointing.
POSTER = EVAL / "poster"

# Threshold for rejecting bad voxels across spaxels, following mean_sky in
# pipeline.py's sky_basis.
CLIP_SIGMA = 30

# One colour scale for every spatial map of s. RdBu_r is diverging, so "above or
# below the typical value" is visible at a glance; a sequential scale is not.
S_CMAP = "RdBu_r"


# Contour colour over a magma background. magma runs black -> purple -> orange ->
# pale yellow, so only a colour outside that range stays visible on top.
SEG_COLOR = "#39ff14"


def qualitative(n):
    """n colours that neighbouring sources will not be confused between.

    The three tab20 families give 60 colours before the cycle restarts, and a shared
    colour only matters for adjacent sources -- ID order is not spatial.
    """
    cols = (list(plt.get_cmap("tab20").colors)
            + list(plt.get_cmap("tab20b").colors)
            + list(plt.get_cmap("tab20c").colors))
    return [cols[i % len(cols)] for i in range(n)]


def asinh_bar(fig, im, ax, label, lo, hi):
    """Colour bar for an asinh stretch, ticked with the original physical values.

    The image shows arcsinh(z), whose default ticks cannot be read as brightness.
    Ticks at arcsinh(z) labelled with z read as z while keeping the asinh dynamic
    range. lo/hi are the range of z, and decide which ticks fall inside.
    """
    ticks = [-3, -1, 0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
    t = [v for v in ticks if lo <= v <= hi]
    cb = fig.colorbar(im, ax=ax, fraction=0.046,
                      ticks=[np.arcsinh(v) for v in t])
    cb.ax.set_yticklabels([f"{v:g}" for v in t])
    cb.set_label(label, fontsize=9)
    return cb


def slug(name):
    """Region name -> filename. "src edge d=7px" -> "src_edge_d_7px".

    Everything non-alphanumeric becomes an underscore, and repeats collapse. Spaces,
    '=' and '#' all need escaping in the shell and in paths.
    """
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", name)).strip("_")


def diverging_range(a, centre=None, pct=2.0):
    """(centre, vmin, vmax) for a diverging colour scale -- symmetric and robust
    against outlier pixels.

    Percentiles rather than min and max: a handful of bad pixels drags min/max out
    until everything else is one colour. The default centre is the median, which
    gives the structure the largest contrast.
    """
    v = a[np.isfinite(a)]
    c = float(np.median(v)) if centre is None else float(centre)
    r = float(max(abs(np.percentile(v, pct) - c), abs(np.percentile(v, 100 - pct) - c)))
    return c, c - r, c + r


from utils import arcsinh_stretch  # noqa: E402, F401 — canonical def in utils
from products import Run  # noqa: E402


def seg_and_background(work, seg=None, white=None):
    """A segmentation and the image to draw it over -- the run's own by default.

    Both maps of a field take the same three arguments and have to read them the same
    way. A background from one pointing under another's segmentation looks like a
    misalignment rather than a mistake, so the shape check belongs here too.

    Returns (seg, background, the segmentation's path), the path being what the figure
    is named after.
    """
    step01 = Run(work).work / "step01"
    bg = fits.getdata(Path(white) if white else step01 / "whitelight_nosky.fits")
    seg_path = Path(seg) if seg else step01 / "segmentation_input.fits"
    s = fits.getdata(seg_path)
    if s.shape != bg.shape:
        raise SystemExit(f"seg {s.shape} and whitelight {bg.shape} "
                         "have different dimensions")
    return s, bg, seg_path


def map_name(work, seg_path):
    """What a figure of this segmentation is called.

    The pointing and the seg's own directory both go in: every pointing's seg has the
    same basename, and so do two segmentations of one pointing, so either part alone
    lets one figure silently overwrite another.
    """
    return f"{Path(work).name}_{seg_path.parent.name}_{seg_path.stem}"


def band_tag(band, default=(4600.0, 9350.0)):
    """The wavelength band as a filename suffix, empty for the default one.

    A non-default band is a different figure and must not overwrite the usual one; the
    default carries no suffix so the everyday filename stays short.
    """
    return "" if tuple(band) == tuple(default) else f"_{band[0]:.0f}-{band[1]:.0f}"


def sigma_image(ax, z, lo, hi):
    """A field in units of the blank spread, asinh-stretched -- the project's one
    stretch for such an image, so two figures of one field cannot differ for a reason
    that is about the drawing.

    `z` is already in sigma; lo and hi are the range of z to show, and are passed
    through arcsinh here rather than by the caller. Give `hi` as it was computed and
    not through float(): arcsinh of a float32 differs from arcsinh of the same number
    widened to float64, and the colour scale is set from it.
    """
    return ax.imshow(np.arcsinh(z), origin="lower", cmap="magma",
                     vmin=np.arcsinh(lo), vmax=np.arcsinh(hi))


def seg_outline(ax, seg, lw=0.6, alpha=0.8):
    """Outline the sources and take the axes off.

    The two go together: these images are read as pictures of the field, where a tick
    label is noise, and the outline is what says which part is source.
    """
    ax.contour(seg > 0, levels=[0.5], colors=SEG_COLOR, linewidths=lw, alpha=alpha)
    ax.set_xticks([]); ax.set_yticks([])


def s_panel(ax, a, seg, vmin, vmax, color="black", width=1.2, alpha=1.0,
            halo=None, main=None):
    """One map of the sky continuum amplitude s, on the given scale.

    RdBu_r runs dark blue -> white -> dark red, so no single outline colour is legible
    against all of it: black weakens in the saturated corners, white weakens near
    s = 1. `halo` draws a wider line of the opposite tone underneath, which removes
    that dependence at the cost of a heavier line, so it is off unless a field
    saturates. `main` outlines the main source group, for the figures that show where
    the galaxy sits in the field.
    """
    im = ax.imshow(a, origin="lower", cmap=S_CMAP, vmin=vmin, vmax=vmax)
    if width > 0:
        if halo:
            ax.contour(seg > 0, levels=[0.5], colors=halo, linewidths=width * 3.0,
                       alpha=0.9)
        ax.contour(seg > 0, levels=[0.5], colors=color, linewidths=width, alpha=alpha)
    if main is not None:
        ax.contour(main, levels=[0.5], colors=color, linewidths=1.6)
    ax.set_axis_off()
    return im


def data_hdu(h):
    """The image extension of an open MUSE cube.

    A pipeline product names it DATA; a cube written by hand often has it in the
    primary. Every reader here has to make the same choice, so it is made once.
    """
    return h["DATA"] if "DATA" in h else h[0]


def collapse(path, band, wl, seg):
    """Collapse the cube over the given band -- returns (image, number rejected,
    number of elements taking part in the rejection).

    Bad voxels are rejected before averaging, following mean_sky in pipeline.py's
    sky_basis: within a channel, across spaxels, centre is the median and spread is
    (p84 - p16) / 2, so no absolute threshold is needed. Only blank is clipped -- a
    sky line is bright in every spaxel and sits inside the channel median, but a
    source is bright in a few spaxels and is an outlier by construction, so the same
    ruler would clip its whole spectrum to NaN.
    """
    m = (wl >= band[0]) & (wl < band[1])
    with fits.open(path, memmap=True) as h:
        d = np.asarray(h[0].data[m] if h[0].data is not None
                       else h["DATA"].data[m], np.float32).copy()
    blank = seg == 0
    bl  = d[:, blank]
    p16, med, p84 = np.nanpercentile(bl, [16, 50, 84], axis=1)
    sg  = np.maximum((p84 - p16) / 2, 1e-6)
    bad = np.abs(bl - med[:, None]) > CLIP_SIGMA * sg[:, None]
    d[:, blank] = np.where(bad, np.nan, bl)
    return np.nanmean(d, axis=0), int(bad.sum()), int(bad.size)
