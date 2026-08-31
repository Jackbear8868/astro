"""Reading back and displaying what the pipeline wrote.

No pipeline step imports this module. It holds what the scripts under `evaluation/`
and `experiments/` share when working from a finished run -- locating a pointing's
fitted products, reading the settings recorded beside them, condensing a spectrum into
numbers, and the figures more than one script draws. It sits next to `utils.py` so
`experiments/` need not import from `evaluation/`.

`Run` is the way in: a run's products under one object, read as they are asked for.
The two programs that make a run write the same tree, so it serves either without
being told which one it is looking at.
"""

import json
from functools import cached_property
from pathlib import Path

import numpy as np
from astropy.io import fits
import pandas as pd
import matplotlib
matplotlib.use("Agg")              # must be set before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import skew, kurtosis

from config import resolve_path


OLD_AMP_KEYS = {"r_far": "min_source_distance",
                "r_far_haro": "min_main_source_distance",
                "clip": "train_clip_sigma",
                "exclude_box": "train_exclude_box",
                "xlim": "train_xlim", "ylim": "train_ylim",
                "main_dz_max": "main_source_dz"}


def sky_amplitude_params(meta):
    """The s-field settings out of a step5 meta.json, under their current names.

    Older products carry the old spelling; translating here keeps that compatibility
    in one place instead of six.
    """
    p = meta.get("sky_amplitude_params") or meta.get("s_field_params") or {}
    return {OLD_AMP_KEYS.get(k, k): v for k, v in p.items()}


def fit_dirs(work, run=None):
    """Where one pointing's fitted products are -- (s-field dir, cube dir).

        s-field dir   sky_continuum_amplitude_field.npy and
                      sky_continuum_amplitude_per_spaxel.npy      by step5
        cube dir      sky_subtracted.fits, sky_model.fits and
                      source_template_amplitude_map.npy           by step6

    With no `run`, or run="default", the two are step05 and step06. Any other `run`
    names one directory under step05 holding both kinds -- an alternative run kept
    beside the pipeline's own.
    """
    work = Path(work)
    if run is None or run == "default":
        return work / "step05", work / "step06"
    d = work / "step05" / run
    if not d.is_dir():
        raise SystemExit(f"★ run directory not found: {d}")
    return d, d


def _need(path, step):
    """The path, or an error naming the step that should have written it.

    A product that is not there means that step was not run, or was run with
    keep_intermediate off. Both are the caller's to fix, and neither is visible in
    the exception numpy or astropy raises three frames further on.
    """
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"★ {path} does not exist -- run {step} for this pointing first.")
    return path


class Run:
    """One finished run's products, read on demand.

    A run is a directory: whatever a config's `output` names. The two ways of making
    one write the same tree -- `pipeline.py` in a single process, and the six scripts
    under `standalone/` one at a time, which `standalone/check_mirror.py` exists to
    keep true. So this asks only where the products are, never which program wrote
    them, and one instance serves either.

        run = Run("results/skymodel/p01")
        run.wl            the wavelength grid
        run.seg           the segmentation
        run.cube          the path to the sky-subtracted cube

    Each attribute is read the first time it is touched and kept after that, so a
    script wanting the wavelength grid does not pay for the segmentation. The cubes
    are the exception: they are the size of the input cube, so `cube`, `sky_model`,
    `wsky` and `nosky` give a path and leave the opening to the caller, which is what
    lets `collapse` memmap one instead of holding it.

    `run` names an alternative directory under step05 the way `fit_dirs` does.
    `step04` overrides which classification the figures are drawn against, which is
    otherwise taken from what step 5 recorded.
    """

    def __init__(self, work, run=None, step04=None):
        self.work = resolve_path(work)
        self.name = self.work.name
        self._run = run
        self._step04 = resolve_path(step04) if step04 else None

    def __repr__(self):
        return f"Run({self.work})"

    # ---- the config the run was given ---------------------------------------

    @cached_property
    def config(self):
        """The config the run was made with, as `config.load` returned it.

        The pipeline writes it before the first step whatever keep_intermediate says,
        so it is there even for a run that kept nothing else.
        """
        f = _need(self.work / "config.json", "the pipeline")
        return json.loads(f.read_text())

    @cached_property
    def pointing(self):
        """Which of the 14 pointings this is, from the run's own config.

        Not parsed out of the directory name: an experiment directory is named for
        what it varies, and `p14_borrow_p03` is not a fourteenth of anything.
        """
        return int(self.config["pointing"])

    @cached_property
    def wsky(self):
        """The input cube, the sky still in it -- what the sky was learned from."""
        return resolve_path(self.config["input"]["cube"])

    @cached_property
    def nosky(self):
        """ESO's own subtraction of the same field, the independent reference.

        Read from the run's config rather than rebuilt from the pointing number, so a
        run whose data sits outside the checkout is looked for where it actually is.
        """
        return resolve_path(self.config["input"]["nosky"])

    # ---- step 1: the field --------------------------------------------------

    @cached_property
    def seg(self):
        """The segmentation as step 1 placed it beside the white light: one ID per
        source, 0 for background."""
        f = _need(self.work / "step01/segmentation_input.fits", "step 1")
        return fits.getdata(f).astype(int)

    @cached_property
    def white(self):
        """The white light image, cast to float so that later percentiles and nanmean
        do not depend on the dtype it was written with. It is 0 outside the field."""
        f = _need(self.work / "step01/whitelight_nosky.fits", "step 1")
        return np.asarray(fits.getdata(f), float)

    @cached_property
    def valid(self):
        """The field of view: where the white light is not the 0 it is padded with."""
        return self.white != 0

    # ---- step 3: the sky model ----------------------------------------------

    @cached_property
    def wl(self):
        """The air wavelength of each channel, (nz,)."""
        return np.load(_need(self.work / "step03/wavelength.npy", "step 3"))

    @cached_property
    def continuum(self):
        """C_sky, the one sky continuum shape, before s scales it per spaxel."""
        return np.load(_need(self.work / "step03/sky_continuum.npy", "step 3"))

    @cached_property
    def mean_sky(self):
        """The sigma-clipped mean of the blank spaxels -- the sky as measured, before
        it was split into a continuum and a line basis."""
        return np.load(_need(self.work / "step03/blank_mean_spectrum.npy", "step 3"))

    @cached_property
    def iterations(self):
        """Step 3's continuum iterations: the line mask and threshold of each pass.

        Copied into a dict rather than kept as the NpzFile, which would hold the file
        open for as long as the Run lives.
        """
        f = _need(self.work / "step03/continuum_iterations.npz", "step 3")
        with np.load(f) as z:
            return {k: z[k] for k in z.files}

    @cached_property
    def line_mask(self):
        """The sky-line mask step 3 finished on -- the last iteration's."""
        return self.iterations["line_mask"][-1]

    def basis(self, method="svd", K=30):
        """The (K, nz) sky-line basis.

        A run may have written more than one, so the method and K name the file rather
        than being read from meta.json -- a figure is often drawn against a basis
        other than the one that config asked for.
        """
        f = _need(self.work / f"step03/sky_line_basis_{method}_K{K}.npy", "step 3")
        return np.load(f)

    # ---- step 4: the sources ------------------------------------------------

    @cached_property
    def step04(self):
        """The step 4 directory the later steps were given.

        A work directory can hold several step 4 runs, and the redshift decides which
        seg IDs belong to the main source, so picking by directory order would be an
        invisible error. Step 5 records the classification file it was handed, and
        that file's directory is the run. The plain step04 is the fallback, for a
        pointing whose step 5 has not been made yet.
        """
        if self._step04:
            return self._step04
        meta = self.fit_dir / "meta.json"
        if meta.exists():
            m = json.loads(meta.read_text())
            # "classification" is the key; older products spell it "best".
            c = m.get("classification") or m.get("best")
            if c:
                return resolve_path(c).parent
        return self.work / "step04"

    @cached_property
    def classification(self):
        """Step 4's conclusion for every source: its class, redshift and amplitudes."""
        f = _need(self.step04 / "classification.npz", "step 4")
        with np.load(f, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    @cached_property
    def source_fits(self):
        """Both branches' fit of every source, the galaxy branch's redshift among
        them -- which is not classification's `z`, the winning branch's."""
        f = _need(self.step04 / "source_fits.npz", "step 4")
        with np.load(f, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}

    # ---- steps 5 and 6: the fit, and the subtracted cube --------------------

    @cached_property
    def _dirs(self):
        return fit_dirs(self.work, self._run)

    @property
    def fit_dir(self):
        """Where step 5's products are: step05, or the alternative run under it."""
        return self._dirs[0]

    @property
    def cube_dir(self):
        """Where step 6's products are."""
        return self._dirs[1]

    @cached_property
    def s_field(self):
        """s forced onto a smooth spatial field, (ny, nx) -- what step 6 applied."""
        f = _need(self.fit_dir / "sky_continuum_amplitude_field.npy", "step 5")
        return np.load(f).astype(float)

    @cached_property
    def s_per_spaxel(self):
        """s solved freely per blank spaxel, before the field was fitted to it. The
        sources are holes in it, and it carries the solving noise the field smooths."""
        f = _need(self.fit_dir / "sky_continuum_amplitude_per_spaxel.npy", "step 5")
        return np.load(f).astype(float)

    @property
    def cube(self):
        """The sky-subtracted cube -- a path, not the array: it is the size of the
        input cube, and its readers take a band at a time out of a memmap."""
        return self.cube_dir / "sky_subtracted.fits"

    @property
    def sky_model(self):
        """The sky that was subtracted, the cube's shape. A path, as `cube` is."""
        return self.cube_dir / "sky_model.fits"

    # ---- what a step recorded, and where the figures go ---------------------

    def meta(self, step):
        """One step's meta.json: what it was given, and what it did.

        Steps 5 and 6 are looked up in the fitted directories, so an alternative run's
        settings are reached rather than the pipeline's own.
        """
        d = {5: self.fit_dir, 6: self.cube_dir}.get(step, self.work / f"step{step:02d}")
        return json.loads(_need(d / "meta.json", f"step {step}").read_text())

    def figdir(self, *sub, poster=False):
        """Where this run's figures go -- an evaluation directory beside the run.

        evaluation/<run>/[subdir...] next to the output directory, so results kept
        outside the checkout carry their figures with them instead of writing back
        into the repository, and a run under experiments/ keeps its figures there
        rather than among the real pointings.

        One directory per run rather than one flat level: reading a pointing is then
        not a filename filter over several hundred entries.

        poster=True inserts that level under evaluation/poster/. A print version of a
        figure carries the same name as the screen version it is a version of, so
        without a directory of its own it overwrites the one it was made from.
        """
        parts = ("evaluation", "poster", self.name) if poster else ("evaluation", self.name)
        d = self.work.parent.joinpath(*parts, *sub)
        d.mkdir(parents=True, exist_ok=True)
        return d


def latest_run(work, product, flat=None, pattern=None):
    """The newest run directory under step05 that holds `product`, or None.

    Steps 5 and 6 write straight into step05 and step06, but a run kept beside them
    sits in a named subdirectory of step05 whose name carries what it varied, so no
    single literal name fits every pointing and `pattern` is a glob. With no pattern
    the newest wins, by the `created` each meta.json records. `flat` names where the
    pipeline's own run put that product, and is weighed against the named ones.

    The directory comes back rather than the file, so a caller can say which run the
    figure is about: a run picked by date is a choice, and an invisible choice makes
    two figures silently incomparable.
    """
    work = Path(work)
    runs = [x for x in (work / "step05").glob(pattern or "*")
            if x.is_dir() and (x / product).exists()]
    if pattern is None and flat is not None and (work / flat / product).exists():
        runs.append(work / flat)
    if not runs:
        return None
    if len(runs) > 1:
        runs.sort(key=lambda x: json.loads((x / "meta.json").read_text()).get("created", ""))
    return runs[-1]


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
    """The two spectra on the left (blue = spec, dashed orange), their stats on the
    right."""
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
        Sky-line channels, left out of the median.
    window : int
        Running-median window, in spectral pixels.
    chunk : int
        Spaxels per pass; memory and speed only.

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

    The background is asinh-stretched: white light spans several orders of magnitude,
    and a linear display makes everything outside the galaxy body black. by_group=False
    draws no group colours -- classification is step4's conclusion, and putting it on a
    locator map invites reading it as fact; this map only says which spot is which.
    """
    fig, ax = plt.subplots(figsize=(13, 12.5))
    v = np.nanpercentile(white[np.isfinite(white) & (white != 0)], 99.5)
    ax.imshow(np.arcsinh(white / (0.02 * v)), origin="lower", cmap="gray",
              vmin=0, vmax=np.arcsinh(1 / 0.02))

    # Fill plus contour: the fill shows extent, the contour stays visible on sources
    # too small to show one.
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
        # Outside the axes, where it cannot cover a source.
        ax.legend(handles=[plt.Line2D([], [], color=c, lw=6, label=g)
                           for g, c in GROUP_COLOR.items()],
                  loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3,
                  frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
