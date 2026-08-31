"""Which spaxels a measurement is made over.

A zone is a set of spaxels with a name. The scripts that compare spectra all start by
choosing one -- the blank a run learned from, a brightness layer of the galaxy, a ring
outside its boundary -- and a second definition of any of them would make two figures
disagree for a reason that has nothing to do with the pipeline. So the definitions live
here and the figures import them.

`zone_means` is the other half: the mean spectrum of each zone, read a band of channels
at a time so a cube is never held whole.
"""
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import data_hdu  # noqa: E402
from config import resolve_path  # noqa: E402

# Spaxels this close to the field-of-view edge are dropped: exposures fall off there
# and step5 writes NaN below 90% coverage, so a layer there measures the mosaic.
EDGE_MARGIN = 6
# Channels read at a time: small enough to keep a chunk in memory, large enough to
# average in vectorised blocks.
CHUNK = 256


def zone_labels(seg, white, valid, main, n_layers, rings):
    """Integer zone map (0 = unused) plus the zone names, ordered inner to outer."""
    d_edge = ndimage.distance_transform_edt(valid)
    ok = valid & (d_edge > EDGE_MARGIN)

    zones = np.zeros(seg.shape, int)
    names = []

    # --- inside the main group: equal-count layers of white-light brightness ---
    inside = main & ok
    v = white[inside]
    # Quantile edges, brightest first, so zone 1 is the core.
    edges = np.percentile(v, np.linspace(0, 100, n_layers + 1))
    for k in range(n_layers):
        lo, hi = edges[n_layers - 1 - k], edges[n_layers - k]
        m = inside & (white >= lo) & (white <= hi if k == 0 else white < hi)
        zones[m] = len(names) + 1
        # L1 is the brightest, and the panels are stacked in that order.
        names.append(f"galaxy L{k + 1}")

    # --- outside it: rings of distance from the boundary ---
    d_main = ndimage.distance_transform_edt(~main)
    outside = ok & ~main & (seg == 0)      # other sources excluded: their light is not Haro 11's
    for lo, hi in zip(rings[:-1], rings[1:]):
        m = outside & (d_main > lo) & (d_main <= hi)
        zones[m] = len(names) + 1
        names.append(f"outside {lo}-{hi} px")
    return zones, names


def zone_means(cube, zones, keys, nz):
    """Mean spectrum of each requested zone, read in wavelength chunks."""
    idx = [np.flatnonzero((zones == k).ravel()) for k in keys]
    out = np.full((len(keys), nz), np.nan)
    with fits.open(cube, memmap=True) as h:
        hdu = data_hdu(h)
        if hdu.data.shape[0] != nz:
            raise SystemExit(f"{cube.name} has {hdu.data.shape[0]} channels, "
                             f"wavelength.npy has {nz}")
        for c0 in range(0, nz, CHUNK):
            c1 = min(c0 + CHUNK, nz)
            block = np.asarray(hdu.data[c0:c1], np.float32).reshape(c1 - c0, -1)
            with np.errstate(invalid="ignore"):
                for j, ix in enumerate(idx):
                    if ix.size:
                        out[j, c0:c1] = np.nanmean(block[:, ix], axis=1)
            print(f"    {cube.name[:28]:<28} {c1}/{nz}", end="\r", flush=True)
    print(" " * 46, end="\r")
    return out


def blank_mask(work, meta):
    """The blank spaxels step3 used, rebuilt from its meta -- that run's blank, not
    blank in general, since two numbers over different spaxels are not a comparison."""
    white = fits.getdata(work / "step01/whitelight_nosky.fits")
    # The same rule the config was read with: relative against the repository, an
    # absolute path as written.
    seg_p = resolve_path(meta["seg"])
    seg = fits.getdata(seg_p)
    valid = white != 0
    m = valid & ~((seg > 0) & valid)
    n_all = int(m.sum())
    yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
    if meta.get("xlim"):
        m &= (xx >= meta["xlim"][0]) & (xx < meta["xlim"][1])
    if meta.get("ylim"):
        m &= (yy >= meta["ylim"][0]) & (yy < meta["ylim"][1])
    if meta.get("exclude_box"):
        y0, y1, x0, x1 = meta["exclude_box"]
        m &= ~((yy >= y0) & (yy <= y1) & (xx >= x0) & (xx <= x1))
    return m, n_all, seg_p
