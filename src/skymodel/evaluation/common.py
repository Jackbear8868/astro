"""Things shared by the scripts under evaluation -- paths, file loading, display
stretch, colour scales.

There is only one condition for putting something here: **two or more scripts are
doing the same thing**. If the same thing is kept as a separate copy in each script,
then editing one copy without editing the other makes the two figures inconsistent in
a place nobody can see -- e.g. one background image uses a different stretch, or one
colour scale is centred differently, while the figure merely looks as if the data had
changed.

Anything only one script needs stays in that script. Moving it here means whoever
reads that script has to cross two files to see what it is doing.

The file must not be named utils.py: the scripts here point sys.path at the level
above in order to import `src/skymodel/utils.py`, and a file with the same name would
shadow it.
"""
import re
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[3]
# Figures and measured values always go to one central place, with the pointing in
# the filename -- if they lived in each working directory, comparing N pointings would
# mean opening N directories, and identical filenames would never sort together.
EVAL = ROOT / "results/skymodel/evaluation"

# Threshold for rejecting bad voxels across spaxels, following mean_sky in
# step3_sky_basis.py.
CLIP_SIGMA = 30

# The spatial maps of s always use the same colour scale. RdBu_r is a diverging scale,
# splitting the two sides of the centre into two colours -- "higher or lower than the
# typical value" is visible at a glance, whereas a sequential scale only separates
# "light or dark".
S_CMAP = "RdBu_r"


def pointing_dir(name, *sub):
    """Where one pointing's figures go -- evaluation/pNN/[subdir...], creating the
    directory if needed.

    One directory per pointing, rather than mixing the figures of all 14 into one
    level and telling them apart by filename: when looking at one pointing, what you
    want is everything about that one, not picking the ones carrying pNN out of
    several hundred filenames.
    """
    d = EVAL.joinpath(name, *sub)
    d.mkdir(parents=True, exist_ok=True)
    return d


# Colour of the contour drawn over a magma background image. magma runs
# "black -> purple -> orange -> pale yellow", so the contour must not be black,
# orange or yellow -- those are colours of the colour map itself, and the line could
# not be told apart from the data in the dark/bright regions. Bright green falls
# outside the colour map, so it is always visible on top.
SEG_COLOR = "#39ff14"


def asinh_bar(fig, im, ax, label, lo, hi):
    """Colour bar for an asinh stretch -- ticks labelled with the original physical
    values, not with the stretched numbers.

    The image shows arcsinh(z), so the colour bar's default ticks are 0, 2, 4... which
    are compressed numbers, from which "how much brighter than the sky is this" cannot
    be read. Putting the ticks at the arcsinh(z) positions and labelling them with z
    itself lets the colour bar be read directly as z, while the image still enjoys the
    dynamic range of asinh.

    lo/hi are the range of z (not of arcsinh(z)), used to decide which ticks fall
    inside.
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

    Keep only alphanumerics, turn everything else into an underscore and then collapse
    repeated underscores -- spaces, '=' and '#' all have to be escaped in the shell and
    in paths, and since these names are assembled automatically they will show up
    sooner or later.
    """
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", name)).strip("_")


def diverging_range(a, centre=None, pct=2.0):
    """(centre, vmin, vmax) for a diverging colour scale -- symmetric, and robust
    against outlier pixels.

    The range is taken from percentiles rather than from the min and max: a handful of
    bad pixels is enough to drag min/max out until everything else is squeezed into a
    single colour. If no centre is given the median is used, which gives the structure
    the largest contrast.
    """
    v = a[np.isfinite(a)]
    c = float(np.median(v)) if centre is None else float(centre)
    r = float(max(abs(np.percentile(v, pct) - c), abs(np.percentile(v, 100 - pct) - c)))
    return c, c - r, c + r


def load_field(work):
    """One pointing's step01 products -- returns (seg, white light image, valid field
    of view mask).

    The white light image is converted to float so that computing percentiles and
    nanmean from it later is unaffected by the original dtype. Spaxels outside the
    field of view are 0 in the white light image, so valid means "this pixel has
    data".
    """
    seg   = fits.getdata(work / "step01/seg.fits").astype(int)
    white = np.asarray(fits.getdata(work / "step01/whitelight.fits"), float)
    return seg, white, white != 0


from utils import arcsinh_stretch  # noqa: E402, F401 — canonical def in utils


def collapse(path, band, wl, seg):
    """Collapse the cube over the given band into a single image -- returns (image,
    number rejected, number of elements taking part in the rejection).

    Bad voxels are rejected before averaging, with the same procedure and threshold as
    mean_sky in step3_sky_basis.py: within one channel, across spaxels, the centre is
    the median and the spread is (p84 - p16) / 2 (robust against outliers), and
    anything deviating by more than CLIP_SIGMA times that is rejected. This way no
    separate absolute threshold has to be chosen.

    But **only blank is clipped**. A sky emission line is bright in every spaxel, so
    its brightness is inside that channel's median and it does not get clipped; a
    source is not like that -- a source is bright in only a few spaxels, so seen
    across spaxels it is an outlier by construction, and using the same ruler would
    clip the whole spectrum of the source spaxels away, leaving NaN after nanmean.
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
