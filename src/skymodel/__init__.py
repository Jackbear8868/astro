"""Sky reconstruction for the Haro 11 MUSE mosaic.

The method learns the sky from the blank spaxels of a sky-included cube and subtracts
it, in six steps. Each is a function here, and `run_pointing` is all six in the order
they happen, driven by one pointing's config file:

    from skymodel import run_pointing
    run_pointing("configs/p01.yaml")

Any step can equally be used on its own, against a work directory that holds what the
earlier steps produced:

    from skymodel import sky_basis
    sky_basis(work="results/skymodel/p01", cube="data/wsky/DATACUBE_FINAL_1.fits", K=30)

    whitelight         collapse a cube along wavelength into a white light image
    object_spectra     sum each source's spectrum over the spaxels its seg ID covers
    sky_basis          learn the sky continuum and the sky-line basis from blank
    classify_sources   fit templates to every source, giving it a class and a redshift
    fit_s_field        force the sky continuum amplitude s onto a spatial field
    subtract_sky       apply the model to every spaxel and write the subtracted cube

Each step also has a command line of its own -- `python src/skymodel/stepN_*.py --help`
-- and the products of every step go to disk, under {output}/step01 ... step06. That is
deliberate: the products are the interface the evaluation scripts read, a step can be
repeated on its own without redoing the ones before it, and memory stays at the cost of
the largest single step rather than the sum of all six.
"""
# The modules address each other by bare name (`from utils import ...`), which is what
# lets each one run as a script. Putting this directory on the path first means the
# same names resolve when the package is imported instead, so both ways work without
# two sets of import statements.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))

from config import load as load_config          # noqa: E402
from run_pipeline import run_pointing           # noqa: E402
from step1_whitelight import whitelight         # noqa: E402
from step2_object_spectra import object_spectra  # noqa: E402
from step3_sky_basis import sky_basis           # noqa: E402
from step4_fit_source import classify_sources   # noqa: E402
from step5_s_field import fit_s_field           # noqa: E402
from step6_fit_sky import subtract_sky          # noqa: E402

__all__ = [
    "run_pointing", "load_config",
    "whitelight", "object_spectra", "sky_basis",
    "classify_sources", "fit_s_field", "subtract_sky",
]
