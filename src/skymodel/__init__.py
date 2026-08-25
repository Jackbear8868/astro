"""Sky reconstruction for the Haro 11 MUSE mosaic.

The method learns the sky from the blank spaxels of a sky-included cube and subtracts
it, in six steps:

    whitelight         collapse a cube along wavelength into a white light image
    object_spectra     sum each source's spectrum over the spaxels its seg ID covers
    sky_basis          learn the sky continuum and the sky-line basis from blank
    classify_sources   fit templates to every source, giving it a class and a redshift
    fit_s_field        force the sky continuum amplitude s onto a spatial field
    subtract_sky       apply the model to every spaxel and write the subtracted cube

`run_pointing` is the way in, and the only one: it is those six in the order they
happen, driven by one pointing's config file, which is where every value they take
comes from.

    from skymodel import run_pointing
    run_pointing("configs/p01.yaml")

Each step is handed what the earlier ones returned; only the cube is opened again by
every step that needs it. The products still go to disk, under {output}/step01 ...
step06, because they are the interface the evaluation scripts read -- but nothing in
the pipeline reads them back. A config may set `keep_intermediate: false` to write
only step06's, which are the deliverable.

The four steps that fit -- sky_basis, classify_sources, fit_s_field, subtract_sky --
hold BLAS at one thread while they work and lift the limit again when they return.
That is what makes the products reproducible: a threaded BLAS adds a sum up in as many
pieces as it has threads, and the last bits of the answer follow the thread count.
Importing this package changes nothing about the threading of the process that imports
it.
"""
# The modules address each other by bare name (`from utils import ...`). Putting this
# directory on the path first is what makes those names resolve when the package is
# imported.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from config import load as load_config          # noqa: E402
from pipeline import run_pointing               # noqa: E402

__all__ = ["run_pointing", "load_config"]
