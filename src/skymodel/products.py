"""Reading back and displaying what the pipeline wrote.

No pipeline step imports this module. It holds what the scripts under
`evaluation/` and `experiments/` share when they work from a finished run --
locating a pointing's fitted products, reading the settings recorded beside
them, condensing a spectrum into numbers, and the figures more than one script
draws. It sits next to `utils.py` rather than inside `evaluation/` so that
`experiments/` scripts do not have to import from `evaluation/`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")              # must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import skew, kurtosis


OLD_AMP_KEYS = {"r_far": "min_source_distance",
                "r_far_haro": "min_main_source_distance",
                "clip": "train_clip_sigma",
                "exclude_box": "train_exclude_box",
                "xlim": "train_xlim", "ylim": "train_ylim",
                "main_dz_max": "main_source_dz"}


def sky_amplitude_params(meta):
    """The s-field settings out of a step5 meta.json, under their current names.

    Products written before the parameters were renamed carry the old spelling;
    translating here keeps that compatibility in one place instead of six.
    """
    p = meta.get("sky_amplitude_params") or meta.get("s_field_params") or {}
    return {OLD_AMP_KEYS.get(k, k): v for k, v in p.items()}


def fit_dirs(work, run=None):
    """Where one pointing's fitted products are -- (s-field dir, cube dir).

        s-field dir   sky_continuum_amplitude_field.npy and
                      sky_continuum_amplitude_per_spaxel.npy      by step5
        cube dir      sky_subtracted.fits, sky_model.fits and
                      source_template_amplitude_map.npy           by step6

    With no `run`, or with run="default", the two are step05 and step06. Any
    other `run` names a single directory under step05 holding both kinds of
    product, which is how an alternative run is kept beside the pipeline's own.
    """
    work = Path(work)
    if run is None or run == "default":
        return work / "step05", work / "step06"
    d = work / "step05" / run
    if not d.is_dir():
        raise SystemExit(f"★ run directory not found: {d}")
    return d, d


def spectrum_stats(spec):
    """Condense a spectrum into summary statistics."""
    spec = spec[np.isfinite(spec)]
    return {
        "mean":          np.mean(spec),
        "sigma":         np.std(spec),
        "skewness":      skew(spec),
        "kurtosis":      kurtosis(spec),
        "rms_from_zero": np.sqrt(np.mean(spec**2)),
    }


def plot_compare(wl, spec, spec_compare, out_path, label="ours", label_compare="nosky", ylim=(-20, 20), title=None):
    """Two panels: the two spectra (blue = spec, dashed orange = spec_compare) on
    the left, their summary statistics on the right."""
    fig, (ax, stat_ax) = plt.subplots(1, 2, figsize=(15.5, 4.5), gridspec_kw={"width_ratios": [5, 1]})
    ax.axhline(0, color="0.5", lw=0.5)
    ax.plot(wl, spec, lw=0.9, color="#1f77b4", label=label)
    ax.plot(wl, spec_compare, lw=0.9, ls="--", alpha=0.7, color="#e8710a", label=label_compare)
    ax.set_ylim(*ylim)
    ax.set_xlabel("wavelength [A]"); ax.set_ylabel("flux")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)

    stat_ax.axis("off")
    def fmt(name, s):
        st = spectrum_stats(s)
        return f"[{name}]\n" + "\n".join(f"{k:<13} = {v:.4g}" for k, v in st.items())
    stat_ax.text(0, 0.95, fmt(label, spec), color="#1f77b4", va="top", family="monospace", fontsize=8, transform=stat_ax.transAxes)
    stat_ax.text(0, 0.45, fmt(label_compare, spec_compare), color="#e8710a", va="top", family="monospace", fontsize=8, transform=stat_ax.transAxes)

    fig.tight_layout()
    fig.savefig(out_path, dpi=145)
    plt.close()


def per_spaxel_continuum(
    spectra,
    line_mask,
    window=300,
    chunk=8000,
):
    """Estimate each spaxel's own smooth continuum by a running median.

    Parameters
    ----------
    spectra : ndarray, shape (nz, n_spaxels)
    line_mask : ndarray of bool, shape (nz,)
        Sky-line channels, left out of the running median.
    window : int
        Running-median window, in spectral pixels.
    chunk : int
        Spaxels processed at once; memory and speed only.

    Returns
    -------
    ndarray, shape (nz, n_spaxels), float32
        The continuum of each spaxel.
    """
    nz, n_spaxels = spectra.shape

    continuum_own = np.empty(
        (nz, n_spaxels),
        dtype=np.float32,
    )

    for low in range(0, n_spaxels, chunk):
        high = min(low + chunk, n_spaxels)

        chunk_spectra = spectra[:, low:high].astype(
            np.float64,
            copy=True,
        )

        chunk_spectra[line_mask, :] = np.nan

        chunk_continuum = (
            pd.DataFrame(chunk_spectra)
            .rolling(
                window=window,
                center=True,
                min_periods=1,
            )
            .median()
            .to_numpy()
            .astype(np.float32)
        )

        continuum_own[:, low:high] = chunk_continuum

    return continuum_own


# Colours for the source locator map.
GROUP_COLOR = {"star": "#2ca02c", "galaxy": "#1f77b4", "qso": "#d62728"}
PLAIN_COLOR = "#ff7b7b"     # not grouping; pale red stays visible on greyscale


def id_map(seg, white, rows, out, by_group=False):
    """White light background + source outlines + ID labels.

    The background is asinh-stretched: white light spans several orders of
    magnitude, and a linear display makes everything outside the galaxy body
    black.

    by_group=False (the default) draws no group colours. Classification is
    step4's conclusion, and putting it on a locator map invites reading it as
    established fact; this map only answers "which spot is which source".
    """
    fig, ax = plt.subplots(figsize=(13, 12.5))
    v = np.nanpercentile(white[np.isfinite(white) & (white != 0)], 99.5)
    ax.imshow(np.arcsinh(white / (0.02 * v)), origin="lower", cmap="gray",
              vmin=0, vmax=np.arcsinh(1 / 0.02))

    # Fill plus contour: the fill shows extent, and the contour stays visible on
    # sources too small to show one.
    for r in rows:
        m = seg == r["id"]
        c = GROUP_COLOR[r["group"]] if by_group else PLAIN_COLOR
        rgba = np.zeros(seg.shape + (4,))
        rgba[m] = list(matplotlib.colors.to_rgb(c)) + [0.45]
        ax.imshow(rgba, origin="lower")
        ax.contour(m, levels=[0.5], colors=c, linewidths=0.9)
        ax.text(r["x"], r["y"], str(r["id"]), color="white", fontsize=11,
                fontweight="bold", ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=2.6, foreground="black")])

    ax.set_title("source ID map" + ("\nwhitelight (asinh) + SExtractor "
                 "segmentation, colour = the group step4b assigned"
                 if by_group else ""), fontsize=14)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    if by_group:
        # Outside the axes: sources sit in the upper right, where an in-axes
        # legend would cover them.
        ax.legend(handles=[plt.Line2D([], [], color=c, lw=6, label=g)
                           for g, c in GROUP_COLOR.items()],
                  loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3,
                  frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
