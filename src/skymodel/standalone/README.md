# Standalone steps

The same six steps as `src/skymodel/pipeline.py`, one script each, so a single step can
be run and tested on its own.

    python step3_sky_basis.py --help

Every step takes its inputs from the command line and writes its products to a
directory, so a step can be pointed at the products of an earlier run instead of
re-deriving them. `run_pipeline.py` is the six chained from a config file, the same
config `pipeline.py` reads.

    step1_whitelight.py         cube -> white light, and the segmentation placed on it
    step2_object_spectra.py     sum each source's spectrum over its seg ID
    step3_sky_basis.py          learn the sky continuum and the sky-line basis
    step4_classify_sources.py   fit templates to every source
    step5_fit_s_field.py        force the sky-continuum amplitude onto a spatial field
    step6_subtract_sky.py       apply the model to every spaxel, write the cubes

Three files are not steps:

    step_io.py       the values the steps hand each other, and how to rebuild them
                     from a finished run
    sync_shared.py   copy utils.py and config.py in from the live pipeline
    check_mirror.py  run a pointing both ways and compare every product

## What is different from the live pipeline, and what is not

The method is the same and the products are the same, byte for byte. What differs is
how a step gets its inputs.

`pipeline.py` runs the six in one process and passes results between them in memory:
step 1 returns a `WhiteLight`, step 3 takes it as an argument. A step run on its own
has no predecessor to hand it anything, so it reads what it needs out of an earlier
step's products instead -- that is what `step_io.py` is, and it is the only thing here
the live pipeline has no counterpart for.

A step therefore needs `--work`, the run directory to read from, and its parameters on
the command line rather than from a config section. `run_pipeline.py` is where a config
is turned into those arguments.

## This is a second implementation

The step files are a copy of the live pipeline's logic, not a wrapper around it. Two
implementations of one method drift: a change made to one and not the other leaves both
running, both writing plausible products, and only the science different.

`check_mirror.py` is what catches that. It runs a config through both programs and
compares every product:

    python check_mirror.py ../../../configs/p01.yaml

Exit status is 0 when every product matches and 1 when any does not. Run it after
changing either side. FITS cubes are compared on their data and `meta.json` on the
fields that say what was computed, so a different timestamp is not reported as a
difference in the answer.

`utils.py` and `config.py` are the exception: they are libraries, copied in verbatim by
`sync_shared.py` with the repository-root depth rewritten for this folder, so there is
no second copy of them to maintain.

    python sync_shared.py            copy
    python sync_shared.py --check    exit 1 if a copy is behind

## Why there is no `__init__.py`

The modules address each other by bare name -- `from utils import ...`,
`from step_io import ...`. That resolves because Python puts a script's own directory
first on `sys.path` when the script is run, which is exactly what makes each step
runnable on its own -- and it is the property this folder exists to keep.

An `__init__.py` doing `sys.path.insert(0, <its own dir>)` would make the same bare
names resolve when the folder is imported as a package, but that insert is global to
the process: importing this folder would then put its `utils`/`config` ahead of the
live ones for everything else in that process. Without an `__init__.py` the folder
stays a plain directory of scripts, which is all it needs to be.
