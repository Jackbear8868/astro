"""Pipeline diagnostic figures -- auto-generated during a pipeline run.

Each function draws one self-contained figure and saves it to disk.
The pipeline steps call these after the corresponding computation; the
figures are not essential to the pipeline's data flow but let the user
verify intermediate results without running evaluation scripts separately.
"""
from pathlib import Path

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from utils import arcsinh_stretch


def plot_main_group(seg, white, main_mask, main_ids, all_ids, peak,
                    out_path, title=""):
    """Two-panel figure: before and after redshift filtering of the main
    source group.

    Left panel: every seg ID inside the adjacent blob, each in its own
    colour and labelled with the ID number.  Right panel: only the IDs
    that passed the redshift criterion, with the connected-component
    boundary drawn as a dashed contour.

    Parameters
    ----------
    seg : 2-d int array
        Segmentation map.
    white : 2-d float array
        White-light image (used as background).
    main_mask : 2-d bool array
        Footprint of the final main source group (after filtering).
    main_ids : list of int
        Seg IDs kept after redshift filtering.
    all_ids : list of int
        Seg IDs in the adjacent blob before filtering.
    peak : tuple (y, x)
        Coordinates of the brightest pixel.
    out_path : path-like
        Where to save the figure.
    title : str
        Figure title (typically the pointing name).
    """
    valid = white != 0
    stretched, vmax = arcsinh_stretch(white, valid)

    fig, ax = plt.subplots(1, 2, figsize=(15, 7.2))
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    for a in ax:
        a.imshow(stretched, origin="lower", cmap="gray", vmin=0, vmax=vmax)
        a.set_xticks([]); a.set_yticks([])

    # left: one colour per seg ID inside the adjacent blob (before filtering)
    for k, i in enumerate(all_ids):
        m = seg == i
        rgba = np.zeros(m.shape + (4,))
        rgba[m] = list(cmap[k % 20][:3]) + [0.55]
        ax[0].imshow(rgba, origin="lower")
        yy, xx = np.nonzero(m)
        if yy.size > 40:
            ax[0].text(xx.mean(), yy.mean(), str(i), color="w", fontsize=11,
                       fontweight="bold", ha="center", va="center",
                       path_effects=[pe.withStroke(linewidth=2.5,
                                                   foreground="k")])
    ax[0].set_title(f"before ({len(all_ids)} sources)", fontsize=12)

    # right: after redshift filtering
    rgba = np.zeros(main_mask.shape + (4,))
    rgba[main_mask] = [1.0, 0.5, 0.05, 0.5]
    ax[1].imshow(rgba, origin="lower")
    ax[1].contour(main_mask, levels=[0.5], colors="#ff7f0e", linewidths=1.6)
    lab, _ = ndimage.label(seg > 0)
    ax[1].contour(lab == lab[peak], levels=[0.5],
                  colors="#00e5ff", linewidths=0.9, linestyles="--")
    ax[1].plot(peak[1], peak[0], "w+", ms=14, mew=2)
    ax[1].set_title(f"after ({len(main_ids)} sources)", fontsize=12)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
