"""Shared utilities.

Two groups:
  * spectral direction -- running median, iterative sky-line detection
    (estimate_continuum etc.)
  * spatial direction -- building a smooth field from per-spaxel coefficients
    (the s-field section in the second half of this file)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy import ndimage
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


def running_median(spectrum, window=300):
    half = window // 2
    n = len(spectrum)
    result = np.empty(n)
    for j in range(n):
        result[j] = np.nanmedian(spectrum[max(0, j - half):j + half])
    return result


def detect_lines(mean_sky, exclude=None, thresholds = (1, 2), window=300):
    m = mean_sky.copy()
    if exclude is not None:
        m[exclude] = np.nan                        # lines found in previous iteration -> excluded from continuum
    continuum = running_median(m, window)
    
    x = np.arange(len(m))
    good = np.isfinite(m) & np.isfinite(continuum)
    xg, yg = x[good], continuum[good]
    spl = UnivariateSpline(xg, yg, k=3, s=len(xg) * 0.05**2, ext=3)
    continuum = spl(x)

    abs_diff = np.abs(m - continuum)
    sigma = running_median(abs_diff, window)       # running median along wavelength

    line_mask = (mean_sky > continuum + thresholds[0] * sigma) | (mean_sky < continuum - thresholds[1] * sigma)
    return continuum, sigma, line_mask


def estimate_continuum(mean_sky, thresholds=(1, 2), window=300, max_iter=5, min_unmasked_frac=0.16):
    line_mask = None
    history = []

    for i in range(max_iter):
        continuum, sigma, new_mask = detect_lines(mean_sky, exclude=line_mask, thresholds=thresholds, window=window)
        
        unmasked_frac = 1.0 - new_mask.sum() / new_mask.size
        if unmasked_frac < min_unmasked_frac:
            print(f"Iteration {i+1}: unmasked fraction {unmasked_frac:.1%} < floor {min_unmasked_frac:.0%}. Stop iteration.")
            break

        if line_mask is not None and np.array_equal(new_mask, line_mask):
            break

        line_mask = new_mask
        history.append((continuum, sigma, line_mask))
    
    if not history:
        raise ValueError(f"First iteration already masked more than {1 - min_unmasked_frac:.0%} of the spectrum.\n"
        "Check the input spectrum and parameters.")

    return history[-1][0], history[-1][1], history[-1][2], history


def load_line_masks(path, cumulative=True):
    """Load the per-iteration sky-line masks saved by step3; returns the
    cumulative version by default.

    The saved file stores the mask each iteration actually used -- inside
    estimate_continuum's loop, new_mask is recomputed from scratch by
    detect_lines and wholly replaces the old one, not accumulated
    (`line_mask = new_mask`). The convergence condition
    `np.array_equal(new_mask, line_mask)` relies on this: a mask that can
    only grow monotonically would never equal its predecessor. So the saved
    file must remain non-cumulative; it is a faithful record of the process.

    When used as a "progressively wider" sequence, however, the non-cumulative
    version has exceptions -- after the continuum is re-estimated, individual
    channels can drop back below threshold, so iteration i does not strictly
    contain iteration i-1. cumulative=True applies a logical_or prefix
    accumulation to enforce monotonicity.

    Iteration 1 is the same either way, so call sites that read only [0] are
    unaffected.
    """
    m = np.load(path)
    return np.logical_or.accumulate(m, axis=0) if cumulative else m


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


# ---------------------------------------------------------------------------
# s field -- build a spatial field from the per-spaxel sky-continuum
# coefficient s
# ---------------------------------------------------------------------------
# Why this is needed: if s is solved freely per spaxel, a spaxel adjacent to
# a source that sees leaked source light can only explain it by raising its
# own s -- that per-spaxel degree of freedom is the channel through which
# the sky model absorbs source flux. Replacing it with "build a field from
# spaxels far from all sources, then extrapolate to the source vicinity"
# means one spaxel's data cannot budge the field, so source light has nowhere
# to go inside the sky model and stays in the residual (= is preserved).
#
# The functional form of the field (see rowcol_field):
#
#     s_hat(x, y) = mu + a(y) + b(x)
#
# describes axis-aligned striping caused by the instrument (extending along
# entire rows/columns, neither sky nor source).


def arcsinh_stretch(img, valid=None, soft=0.02):
    """asinh stretch for display -- returns (stretched image, vmax).

    Maps the dynamic range of a white-light image into a displayable
    range: linear in the faint parts, logarithmic in the bright parts.
    vmin is always 0; vmax = arcsinh(1 / soft).
    """
    m = np.isfinite(img) & (img != 0)
    v = np.nanpercentile(img[m], 99.5)
    a = img if valid is None else np.where(valid, img, np.nan)
    return np.arcsinh(a / (soft * v)), np.arcsinh(1 / soft)


def scale(a):
    """Robust spread (p84 - p16) / 2.

    rms/std cannot be used: s has a few spaxels with failed fits whose
    outlier values are extreme, and rms/std would be dominated by those few,
    no longer measuring the overall spread.
    """
    a = a[np.isfinite(a)]
    return float((np.percentile(a, 84) - np.percentile(a, 16)) / 2) if a.size else np.nan


def nanmed(a, axis):
    """Median along axis; returns 0 instead of NaN when an entire row/column
    is all NaN.

    All-NaN means that row has no training points, so the offset cannot be
    estimated. Setting 0 is "apply no correction" -- the only honest choice
    in that situation, though it is an assumption, not a measurement.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nan_to_num(np.nanmedian(a, axis=axis))


C_KMS = 299792.458

# How close in redshift a member must be to the main source to count as part
# of the same galaxy. The galaxy has internal rotation and outflows, so the
# criterion is "within the galaxy's velocity range", not "identical redshift".
#
# The threshold must be loose enough not to exclude the galaxy's own bright
# knots and tight enough to reject background galaxies. The data make this
# easy: genuine members differ by only tens of km/s, while a superposed
# background galaxy differs by ~350,000 km/s -- four orders of magnitude
# apart, so any threshold between ~300 and ~3000 km/s gives the same result
# (see the scan table in evaluation/main_group_spec.py).
#
# 0.005 is chosen because step4's stellar-redshift scan half-width (--star-dz)
# uses the same value, and reusing the same number is easier to remember.
# Converted to velocity: c dz / (1+z) = 1469 km/s at z = 0.0205, which falls
# in the safe interval.
DZ_MAX = 0.005


def galaxy_redshifts(step04, ids, tag=None):
    """Best galaxy-branch redshift for each seg ID. Returns {id: z}.

    step4 stores the two branches separately; scan2 is the galaxy branch.
    The z from the classification file is not used -- that is the winning
    branch's value, and when the source is classified as a star it is a
    radial velocity, not a redshift.

    tag names one step4 run, and is the part of the classification filename
    after "classification_". Callers that hold a --best file know it and
    should pass it; without it several matches are an error rather than a
    silent pick.
    """
    out = {}
    for i in ids:
        pat = f"scan2_id{i}_*.npz" if tag is None else f"scan2_id{i}_{tag}.npz"
        f = sorted(Path(step04).glob(pat))
        if not f:
            raise SystemExit(f"{pat} not found in {step04}")
        # Multiple hits mean the directory contains results from several step4
        # runs (different windows, different mask iterations). Taking [0] picks
        # one by filename order, and since the redshift determines which members
        # belong to the main source, picking the wrong one is invisible
        # downstream. Better to stop and ask.
        if len(f) > 1:
            raise SystemExit(
                f"id{i} has {len(f)} scan2 files in {step04}:\n  "
                + "\n  ".join(x.name for x in f)
                + "\n  redshift must come from the same fit as --best; pass tag= to pick one.")
        d = np.load(f[0], allow_pickle=True)
        out[int(i)] = float(d["z"][np.argmin(d["red_chi2"])])
    return out


def main_source_group(seg, white, step04=None, dz_max=DZ_MAX, tag=None):
    """Full footprint of the main galaxy -- the connected blob containing the
    brightest pixel, keeping only members with matching redshifts.
    Returns (mask, ID list, peak coordinates).

    Why a single "largest-area" or "brightest" ID does not work: SExtractor's
    deblender splits the main galaxy. A merging galaxy naturally has several
    bright knots, and whether they are split and into how many pieces depends
    on that exposure's seeing and dither -- it varies between exposures. When
    split, any "pick one ID" rule gets only part of the galaxy, and downstream
    uses it to decide "which side to mask" and "how many pixels to exclude
    around it" -- picking the wrong piece misplaces both.

    Two criteria:

    (1) **Directly adjacent** (no dilation). Deblended siblings are carved from
        the same above-threshold connected region, so they touch; a different
        object is separated by below-threshold background. Dilation would blur
        this distinction, pulling in unrelated nearby sources.
    (2) **Redshift match**. Adjacency alone is not enough -- another object
        superposed on the galaxy is also deblended as a child of the same
        parent detection and therefore also touches. The galaxy-branch
        redshift from step4's fit discriminates: members differing from the
        main source by more than dz_max are not part of this galaxy. The main
        source's redshift is taken from the member containing the brightest
        pixel.

    tag is passed through to galaxy_redshifts to name one step4 run when the
    directory holds several.

    When step04 is omitted, only criterion (1) is applied. The professor's
    delivered seg has no corresponding template fit, so no redshift is
    available; returning the entire adjacent blob is the only honest choice.

    The returned mask is intersected with the connected blob, not
    `isin(seg, ids)` -- SExtractor's CLEAN merges scattered spurious
    detections into the bright source's ID, and those pixels carry the main
    source's number but are not on the main source.
    """
    k = np.unravel_index(np.nanargmax(np.where(np.isfinite(white), white, -np.inf)),
                         white.shape)
    src = seg > 0
    lab, _ = ndimage.label(src)
    blob = lab == lab[k]
    ids = [int(i) for i in np.unique(seg[blob & src]) if i > 0]

    if step04 is not None:
        z = galaxy_redshifts(step04, ids, tag)
        z0 = z[int(seg[k])]
        # Comparing |dz| and |c dz/(1+z0)| is the same criterion -- both sides
        # are multiplied by the same positive number. Using the redshift
        # difference directly avoids tying the threshold to a particular z0.
        ids = [i for i in ids if abs(z[i] - z0) <= dz_max]

    return np.isin(seg, ids) & blob, ids, k


def main_source_mask(seg, source_id=None, main_blob=True):
    """Mask of the main source. Returns (boolean mask, seg ID used).

    When source_id is omitted, the largest-area source is selected -- do not
    hard-code seg == 1. SExtractor numbers sources in detection order, so a
    different pointing may give the main galaxy a different ID, and hard-coding
    would silently use some small source as the main, excluding dozens of
    pixels in the wrong place.

    main_blob=True keeps only the largest connected component. A single seg ID
    can consist of **several disconnected patches** -- that is caused by
    SExtractor's `CLEAN Y` merging pixels of objects it judges to be spurious
    into the nearby bright source. Using the entire ID for distance
    calculations would let those scattered fragments each produce an exclusion
    ring, inflating the exclusion area far beyond the main source itself.
    """
    if source_id is None:
        ids, cnt = np.unique(seg[seg > 0], return_counts=True)
        source_id = int(ids[np.argmax(cnt)])
    m = seg == source_id
    if main_blob:
        lab, n = ndimage.label(m)
        if n > 1:
            sz = ndimage.sum(m, lab, range(1, n + 1))
            m = lab == (int(np.argmax(sz)) + 1)
    return m, source_id


def rowcol_field(s, w, n_iter=4):
    """s ~ mu + a(y) + b(x), solved by alternating medians (Tukey's median
    polish).

    Returns (field, a, b).

    Why additive rather than a general 2D function f(x, y): a general f is
    one number per pixel -- no compression, and therefore no ability to
    predict where there is no data. The additive form has only 1 + ny + nx
    parameters, and **a(y) is shared by all spaxels in that row** -- this is
    exactly why information can flow sideways into large gaps: a(y) for a row
    in the centre of the gap is determined by the training spaxels in the same
    row far from the source.

    Can represent: horizontal stripes, vertical stripes, any linear diagonal
    gradient (alpha*x + beta*y is separable). Cannot represent: features that
    appear only at a specific location and do not extend along an entire row
    or column (statistically: interaction terms). Those stay in the residual.

    Why median rather than mean: (1) robust to bad spaxels; (2) **natural
    for large gaps** -- even when most of a row is occupied by a source, the
    median of the remaining spaxels is still a good estimate of that row's
    offset.

    Why alternate: a and b are coupled. To measure "how much higher is this
    row", the column offsets must be subtracted first, otherwise the
    measurement reflects "which columns happen to remain in this row". n_iter
    defaults to 4, providing margin above the number of iterations needed for
    convergence.

    Note the degeneracy: adding c to every a(y) and subtracting c from every
    b(x) leaves the field unchanged. So (mu, a, b) are individually
    non-unique; only their sum is meaningful. The alternating medians
    naturally keep median(a) and median(b) near 0, with mu absorbing the
    overall level. This must be kept in mind when interpreting a(y) alone.
    """
    S  = np.where(w, s, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = float(np.nanmedian(S))
    a = np.zeros(s.shape[0])
    b = np.zeros(s.shape[1])
    for _ in range(n_iter):
        a = nanmed(S - mu - b[None, :], 1)
        b = nanmed(S - mu - a[:, None], 0)
    return mu + a[:, None] + b[None, :], a, b


def build_s_field(s, seg, blank, r_far, r_far_haro, clip,
                  main_id=None, exclude=None, main=None):
    """Build a spatial field from the per-spaxel s map. Returns (s_hat,
    training mask).

    The field has the form mu + a(y) + b(x) (see rowcol_field). It has only
    1 + ny + nx parameters, and a(y) is shared across an entire row while
    b(x) is shared across an entire column -- this is exactly why it can
    extrapolate into the source region: rows that pass through the source
    still have training spaxels far away, and the parameters are determined
    from those, then applied to the centre of the gap.

    Parameters
    ----------
    s : ndarray, shape (ny, nx)
        Sky-continuum coefficients from the per-spaxel free solve. Values in
        the source region are not used.
    seg : ndarray, shape (ny, nx)
        Segmentation; 0 = blank, >0 = source.
    blank : ndarray of bool, shape (ny, nx)
        Usable blank spaxels (inside FoV, not source, spectrally complete,
        s successfully solved).
    r_far : float
        Training points must be at least this far (px) from **any** source,
        to avoid the source PSF wings; otherwise the training points
        themselves carry source flux. The only cost is fewer samples.
    r_far_haro : float or None
        Extra exclusion radius applied only to the main source. The main
        source's extended halo reaches much farther than the PSF wings of
        small sources; using the same r_far would leave training points
        inside the halo, and the model would learn the halo as sky.
        None = no extra exclusion.
    clip : float
        Spaxels with |s - median| > clip x robust spread are excluded
        (rejects failed-fit solutions).
    main_id : int or None
        Segmentation ID of the main source. None = automatically select the
        largest-area source (see main_source_mask).
    exclude : ndarray of bool or None
        Additional spaxels to exclude from training (they are still sky-
        subtracted, just not used for building the field). Purpose: mosaic
        sub-fields with insufficient exposure depth where noise is far
        higher than the outer ring -- using them to learn the sky writes
        noise into the field.
    main : ndarray of bool or None
        Mask of the main source; when given, main_id is not used. Callers
        typically have already computed this via main_source_group.
    """
    train = blank & (ndimage.distance_transform_edt(seg == 0) > r_far)
    if exclude is not None:
        train &= ~exclude
    if r_far_haro:
        m = main if main is not None else main_source_mask(seg, main_id)[0]
        train &= ndimage.distance_transform_edt(~m) > r_far_haro
    med = float(np.median(s[train]))
    train &= np.abs(s - med) <= clip * scale(s[train])

    M, _, _ = rowcol_field(s, train)
    return M, train


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
