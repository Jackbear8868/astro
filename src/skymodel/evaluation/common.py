"""Things shared by the scripts under evaluation -- paths, file loading, display
stretch, colour scales.

One condition for putting something here: two or more scripts do the same thing.
Separate copies drift apart, and a stretch or colour centre that differs between two
figures looks like a change in the data.

The file must not be named utils.py: the scripts point sys.path at the level above to
import `src/skymodel/utils.py`, and a same-named file would shadow it.
"""
import json
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

# Threshold for rejecting bad voxels across spaxels, following mean_sky in
# pipeline.py's sky_basis.
CLIP_SIGMA = 30

# One colour scale for every spatial map of s. RdBu_r is diverging, so "above or
# below the typical value" is visible at a glance; a sequential scale is not.
S_CMAP = "RdBu_r"


def pointing_dir(work, *sub):
    """Where one pointing's figures go. `Run.figdir` under its old name.

    Kept so that the scripts which have not moved to Run yet read as they did.
    """
    return Run(work).figdir(*sub)


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


def load_field(work):
    """One pointing's step01 products -- (seg, white light image, valid field).

    Three attributes of a Run under their old name, for the reason `pointing_dir`
    is kept.
    """
    r = Run(work)
    return r.seg, r.white, r.valid


from utils import arcsinh_stretch  # noqa: E402, F401 — canonical def in utils
from products import Run, fit_dirs  # noqa: E402


def step04_dir(W, run=None):
    """Which step4 directory this pointing's redshifts come from, or None.

    A work directory can hold several step4 runs, and the redshift decides which
    members belong to the main source, so picking by directory order would be an
    invisible error. Step5's meta.json records the classification file it was given.
    """
    meta = fit_dirs(W, run)[0] / "meta.json"
    if not meta.exists():
        return None
    m = json.loads(meta.read_text())
    # "classification" is the key; older products on disk spell it "best".
    c = m.get("classification") or m.get("best")
    # The path is recorded against the repository root, and a step4 directory holds
    # one run, so the file's parent directory is that run.
    return ROOT / Path(c).parent if c else None


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
