"""Reading back and displaying what the pipeline wrote.

This module sits beside the six steps but is not part of them: no step imports
it, and a pipeline run never loads it. It holds what the scripts under
`evaluation/` and `experiments/` share when they work from a finished run --
locating a pointing's fitted products, reading the settings recorded beside
them, condensing a spectrum into numbers, and the figures more than one script
draws.

It lives next to `utils.py` rather than inside `evaluation/` because both
`evaluation/` and `experiments/` already put `src/skymodel` on sys.path and can
import it by name from either side; under `evaluation/` it would make every
`experiments/` script that uses it import from `evaluation/`.
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

    Products written before the parameters were renamed carry the old spelling, and
    they are the results now on disk. Translating here means the readers ask for one
    set of names and the compatibility lives in one place instead of six.
    """
    p = meta.get("sky_amplitude_params") or meta.get("s_field_params") or {}
    return {OLD_AMP_KEYS.get(k, k): v for k, v in p.items()}


def fit_dirs(work, run=None):
    """Where one pointing's fitted products are -- (s-field dir, cube dir).

        s-field dir   s_hat.npy, s_free.npy       written by step5
        cube dir      sky_subtracted.fits, sky_model.fits, A_map.npy, s_map.npy
                                                  written by step6

    With no `run`, the two are step05 and step06. Giving `run` names a single
    directory under step05 that holds both kinds of product, which is how an
    alternative run is kept side by side with the pipeline's own output.
    The name "default" also means step05 and step06, so that a comparison can
    name the pipeline's own output explicitly instead of only by omission.
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
    spec = spec[np.isfinite(spec)]                     # drop NaN first
    return {
        "mean":          np.mean(spec),
        "sigma":         np.std(spec),
        "skewness":      skew(spec),
        "kurtosis":      kurtosis(spec),
        "rms_from_zero": np.sqrt(np.mean(spec**2)),    # sqrt(mean of squares) = RMS from zero
    }


def plot_compare(wl, spec, spec_compare, out_path, label="ours", label_compare="nosky", ylim=(-20, 20), title=None):
    """Comparison plot: left panel shows spectra (blue=spec, dashed orange=spec_compare), right panel shows summary stats for both."""
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
    """Estimate each spaxel's own smooth continuum.

    Parameters
    ----------
    spectra : ndarray, shape (nz, n_spaxels)
        Spectra of multiple spaxels; each column is one spaxel.
    line_mask : ndarray of bool, shape (nz,)
        Global sky-line mask detected from the mean blank-sky spectrum.
        True means that wavelength channel is excluded from the running
        median.
    window : int
        Wavelength window for the running median, in spectral pixels.
    chunk : int
        Number of spaxels processed at once; controls only memory and
        speed, not the scientific definition of the continuum.

    Returns
    -------
    continuum_own : ndarray, shape (nz, n_spaxels)
        Smooth continuum estimated independently for each spaxel.
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


# Colours for the source locator map. GROUP_COLOR is only used when by_group=True.
GROUP_COLOR = {"star": "#2ca02c", "galaxy": "#1f77b4", "qso": "#d62728"}
PLAIN_COLOR = "#ff7b7b"     # uniform colour when not grouping; pale red is visible on greyscale


def id_map(seg, white, rows, out, by_group=False):
    """White light background + source outlines + ID labels.

    The background uses an asinh stretch: the dynamic range of white light
    spans several orders of magnitude (Haro 11 body vs faint sources), and
    a linear display makes everything outside the body black. asinh is
    logarithmic in the bright regime and linear in the faint regime -- the
    standard practice for image display.

    by_group=False (default) draws only outlines and labels, without group
    colours. Classification is step4's "conclusion"; drawing it on the
    locator map would lead viewers to unconsciously treat it as established
    fact. The locator map's job is only to answer "which spot is which source".
    """
    fig, ax = plt.subplots(figsize=(13, 12.5))
    v = np.nanpercentile(white[np.isfinite(white) & (white != 0)], 99.5)
    ax.imshow(np.arcsinh(white / (0.02 * v)), origin="lower", cmap="gray",
              vmin=0, vmax=np.arcsinh(1 / 0.02))

    # Each source gets a semi-transparent colour fill plus a contour outline --
    # the fill shows extent, and the contour remains visible on small sources.
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
        # Legend placed outside the axes -- when sources sit in the upper right,
        # an in-axes legend would cover them.
        ax.legend(handles=[plt.Line2D([], [], color=c, lw=6, label=g)
                           for g, c in GROUP_COLOR.items()],
                  loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3,
                  frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
