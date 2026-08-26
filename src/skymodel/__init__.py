"""Sky reconstruction for the Haro 11 MUSE mosaic.

The method learns the sky from the blank spaxels of a sky-included cube and subtracts
it, in six steps:

    whitelight         collapse a cube along wavelength into a white light image
    object_spectra     sum each source's spectrum over the spaxels its seg ID covers
    sky_basis          learn the sky continuum and the sky-line basis from blank
    classify_sources   fit templates to every source, giving it a class and a redshift
    fit_s_field        force the sky continuum amplitude s onto a spatial field
    subtract_sky       apply the model to every spaxel and write the subtracted cube

`run_pointing` is the only way in: those six steps in order, driven by one
pointing's config file, which is where every value they take comes from.

    from skymodel import run_pointing
    run_pointing("configs/p01.yaml")

Each step is handed what the earlier ones returned; only the cube is opened again by
every step that needs it. The products under {output}/step01 ... step06 are the record
of the middle of a run, and nothing in the pipeline reads them back;
`keep_intermediate: false` writes only step06's, which are the deliverable.

The four fitting steps hold BLAS at one thread while they work, which is what makes
the products reproducible: a threaded BLAS adds a sum up in as many pieces as it has
threads, and the last bits follow the thread count. Importing this package leaves the
threading of the importing process alone.
"""
# The modules address each other by bare name (`from utils import ...`), which needs
# this directory on the path.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from config import load as load_config          # noqa: E402
from pipeline import run_pointing               # noqa: E402

__all__ = ["run_pointing", "load_config"]
