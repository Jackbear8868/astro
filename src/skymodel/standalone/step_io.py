"""The values the steps hand each other, and how to rebuild them from a finished run.

The live pipeline runs the six steps in one process and passes these between them in
memory: step 1 returns a WhiteLight, step 3 takes it as an argument. A step run on its
own has no predecessor to hand it anything, so it reads what it needs out of an earlier
step's products instead. That is the whole difference between running a step here and
running it inside `pipeline.py`, and it is why these loaders live in this folder.

Every loader takes `work`, a run's output directory -- the `output` its config names --
and reads fixed paths beneath it. A predecessor run with `keep_intermediate: false`
wrote nothing to read, so each loader names the file it wanted rather than returning an
empty array that fails somewhere further on.

The tuples are copies of the ones in `pipeline.py`. They are copied and not imported:
importing would put the live pipeline on this folder's import path, and the point of
the folder is that a step here runs without it.
"""

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
from astropy.io import fits


# =========================================================================
# what a step hands the next one
# =========================================================================

class WhiteLight(NamedTuple):
    """What step 1 hands the ones after it.

    The header travels with the image for the segmentation check; every other
    consumer reads only `data`.
    """
    data: np.ndarray          # (ny, nx), the collapsed image, 0 outside the field
    header: fits.Header       # the cube's celestial WCS


class Seg(NamedTuple):
    """The segmentation, as steps 2, 3, 5 and 6 are handed it.

    path is where it was put next to the white light; steps 5 and 6 record that in
    their meta.json, so the products say which map they were made with.
    """
    data: np.ndarray
    path: Path


class SourceSpectra(NamedTuple):
    """What step 2 hands step 4: one summed spectrum per source.

    `path` is the directory the arrays were written to. It is carried so step 4 can
    record which spectra it classified, not because anything reads the file back.
    """
    ids: np.ndarray           # (n_ids,)   segmentation IDs, ascending
    flux: np.ndarray          # (n_ids, nz)
    var: np.ndarray           # (n_ids, nz)
    nspax: np.ndarray         # (n_ids, nz)
    path: Path


class SkyModel(NamedTuple):
    """What step 3 hands steps 4, 5 and 6 -- everything they read of the sky.

    basis is keyed by decomposition method, because `methods` may ask for several in
    one run and the later steps name the one they fit with. iter_line_mask is the whole
    per-iteration stack: step 4 fits one iteration per pass, steps 5 and 6 take the
    first.
    """
    wavelength: np.ndarray        # (nz,)          air wavelength of each channel
    continuum: np.ndarray         # (nz,)          C_sky
    basis: dict                   # method -> (K, nz) sky-line basis
    iter_line_mask: np.ndarray    # (n_iter, nz)   bool, one row per iteration


class Classification(NamedTuple):
    """What step 4 hands steps 5 and 6.

    data holds the fields of classification.npz -- step 6 rebuilds each source's model
    from them. galaxy_z is the galaxy branch's best redshift for every source it could
    fit, which step 5 groups the main source by; it is not data["z"], the winning
    branch's.

    path names the product these came from: steps 5 and 6 record it in their meta.json,
    which is how a script reading the products finds the step 4 run they used.
    """
    path: Path
    data: dict                # field name -> array, as written to the npz
    galaxy_z: dict            # seg ID -> galaxy-branch redshift


class SkyAmplitude(NamedTuple):
    """What step 5 hands step 6: the field, and where it was written.

    data is the float32 the file holds, not the float64 the fit produced, so that the
    file and the fit hold the same field.
    """
    data: np.ndarray          # (ny, nx) float32
    path: Path                # step05/sky_continuum_amplitude_field.npy


# =========================================================================
# reading them back
# =========================================================================

def _need(p, step, flag="keep_intermediate"):
    """The path, or an error naming the step that should have written it.

    A missing product here means the earlier step was not run, or was run with
    keep_intermediate off. Both are the caller's to fix, and neither is visible from
    the exception numpy or astropy would raise three frames later.
    """
    p = Path(p)
    if not p.exists():
        raise SystemExit(
            f"★ {p} does not exist. Run {step} for this pointing first"
            f"{'' if flag is None else f', with {flag} on'}.")
    return p


def white(work):
    """step 1's white light, with the celestial WCS it was written with."""
    f = _need(Path(work) / "step01/whitelight_nosky.fits", "step1_whitelight.py")
    data, hdr = fits.getdata(f, header=True)
    return WhiteLight(np.asarray(data), hdr)


def seg(work):
    """step 1's copy of the segmentation.

    The copy, not the config's original: it is byte-identical, and reading it here is
    what makes a step depend on the run directory alone.
    """
    f = _need(Path(work) / "step01/segmentation_input.fits", "step1_whitelight.py")
    return Seg(np.asarray(fits.getdata(f)), f)


def spectra(work):
    """step 2's summed spectrum per source."""
    d = Path(work) / "step02"
    f = _need(d / "source_spectra.npz", "step2_object_spectra.py")
    z = np.load(f)
    return SourceSpectra(z["ids"], z["flux_sum"], z["variance_sum"],
                         z["spaxel_count"], d)


def sky(work, method="svd", K=30):
    """step 3's sky model: the grid, the continuum, the basis and the mask stack.

    method and K name the basis file, a run having possibly written more than one.
    They are arguments rather than read from meta.json because a step being tested on
    its own is often being pointed at a basis other than the one its config asked for.
    """
    d = Path(work) / "step03"
    wl = np.load(_need(d / "wavelength.npy", "step3_sky_basis.py"))
    cont = np.load(_need(d / "sky_continuum.npy", "step3_sky_basis.py"))
    it = np.load(_need(d / "continuum_iterations.npz", "step3_sky_basis.py"))
    b = np.load(_need(d / f"sky_line_basis_{method}_K{K}.npy", "step3_sky_basis.py"))
    return SkyModel(wl, cont, {method: b}, it["line_mask"])


def classification(work, run=None):
    """step 4's fits, and the galaxy branch's redshift for every source it could fit.

    `run` names a directory under step04 when step 4 wrote more than one; None is the
    flat case, which is what a single-iteration config produces.
    """
    d = Path(work) / "step04"
    if run:
        d = d / run
    f = _need(d / "classification.npz", "step4_classify_sources.py")
    z = np.load(f, allow_pickle=True)
    data = {k: z[k] for k in z.files}
    # gal_z is a column of source_fits.npz, not of classification.npz: the winning
    # branch's z is in the latter, and for a star that is a radial velocity.
    sf = _need(d / "source_fits.npz", "step4_classify_sources.py")
    s = np.load(sf, allow_pickle=True)
    gal = {int(i): float(z_) for i, z_ in zip(s["id"], s["gal_z"])
           if np.isfinite(z_)}
    return Classification(f, data, gal)


def s_field(work, run=None):
    """step 5's spatial field of the sky continuum amplitude."""
    d = Path(work) / "step05"
    if run:
        d = d / run
    f = _need(d / "sky_continuum_amplitude_field.npy", "step5_fit_s_field.py")
    return SkyAmplitude(np.load(f), f)


def config_of(work):
    """The config the run was made with, as config.load() returned it.

    run() writes it before the first step whatever keep_intermediate says, so it is
    there even for a run that kept nothing else. A step run on its own can take its
    parameters from the command line instead, which is what the arguments are for --
    this is for the ones that want the run's own settings without being told them
    twice.
    """
    f = _need(Path(work) / "config.json", "the pipeline", flag=None)
    return json.loads(f.read_text())
