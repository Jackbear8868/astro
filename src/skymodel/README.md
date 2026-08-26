# The pipeline

Six steps, run in order by `pipeline.py` from one pointing's config file. Reading
`Pipeline.run()` is meant to be enough to know what the pipeline does; this file says
what each step reads and writes, and why the shape is what it is.

    conda run -n astro python src/skymodel/pipeline.py configs/p01.yaml

## The model

    D(p, λ) = s(p) · C_sky(λ) + Σₖ lₖ(p) · Lₖ(λ)

One sky continuum shape scaled per spaxel, plus K sky-line basis vectors. Both are
learned from the blank spaxels of the sky-included cube. On a source spaxel a template
`Σⱼ Aⱼ(p) · Tⱼ(λ/(1+z))` is fitted alongside, so the sky model cannot absorb the source.

## The steps

| | function | reads | writes |
|---|---|---|---|
| 1 | `whitelight` | the ESO nosky cube | `step01/whitelight.fits` + preview |
| — | `place_segmentation` | the segmentation it is given | `step01/seg.fits`, after checking the two share a pixel grid |
| 2 | `object_spectra` | nosky cube, seg | `step02/object_{ids,flux,var,nspax}.npy` |
| 3 | `sky_basis` | wsky cube, seg | `step03/` continuum, line mask, `sky_basis_{method}_K{K}.npy` |
| 4 | `classify_sources` | `step02/`, `step03/` | `step04/scan{1,2}_id*.npz`, `best_*.npz`, `classification_*.npz` |
| 5 | `fit_s_field` | wsky cube, `step03/`, `step04/` | `step05/s_free.npy`, `s_hat.npy`, `main_group.png` |
| 6 | `subtract_sky` | wsky cube, `step03/`–`step05/` | `step06/sky_subtracted.fits`, `sky_model.fits`, `s_map.npy`, `A_map.npy` |

The white light comes from the **nosky** cube, not the wsky one: downstream needs it to
find the main source (the blob holding the brightest pixel), and the sky continuum of
the wsky cube lifts the whole image, which makes the brightest pixel unreliable.

Step 2 also runs on the nosky cube. Classifying a spectrum that still holds the sky
produces output that looks entirely normal, with every template and redshift wrong.

## Why the products still go to disk

A step hands its results to the next one in memory -- `Pipeline.run()` is where you can
see what each produces and who reads it -- and writes them to `{output}/stepNN/` as
well. The writing is not how the steps communicate; it is there because:

- the products are the interface the 23 scripts under `evaluation/` read
- every intermediate result stays there to be looked at after the run, instead of
  existing only inside one process
- `sky_subtracted.fits` is 4 GB and is the deliverable anyway

`keep_intermediate: false` in a config turns off everything except step 6, which
always writes. It changes what is left on disk and nothing else; a run with it off
produces the same step 6 as a run with it on.

Peak memory is the cost of the largest single step (about 6 GB, step 6), not the sum
of all six: what passes between steps is under 9 MB, and the cube each step opens is
memmapped and released when that step returns.

## One entrance

`pipeline.py` is the only supported way in, from a shell or from Python:

```bash
conda run -n astro python src/skymodel/pipeline.py configs/p01.yaml
```

```python
from skymodel import run_pointing
run_pointing("configs/p01.yaml")
```

The six steps are methods of `Pipeline`, but they are not exported and have no command
lines of their own. Each reads the values it needs from the config the object was built
from, which is what makes them agree with each other -- the same K in steps 3, 4 and 6,
an s-field solved against the sky basis step 6 reads, one output directory holding what
the earlier steps wrote.

What that run was given is recorded twice, because a config file can be edited
afterwards and then nothing else says:

- `{output}/config.json` is the config as `config.load()` returned it -- every value
  the run used, optional keys filled in, paths resolved. `run()` writes it before the
  first step, whatever `keep_intermediate` says.
- the head of each `stepN.log` is the call that produced the products beside it,
  written out as Python: which earlier products that step was handed. It is not a
  command to re-run.

## The modules

Four of them, plus an `__init__.py` that exports `run_pointing` and `load_config`.

| file | holds |
|---|---|
| `pipeline.py` | `Pipeline`, whose methods are the six steps and the segmentation check between the first two, in that order; the products they hand each other are above the class and each step's helpers below it; `run_pointing` and the command line are at the end |
| `config.py` | reads and checks a pointing config; every value the pipeline takes comes from there |
| `utils.py` | everything the six steps share: the wavelength axis and the air-to-vacuum conversion, continuum estimation and line detection, the source templates (eigenspectra and the stellar library, read as splines), the per-spaxel solves steps 5 and 6 share, the main source group, the s-field construction, the figures the pipeline itself produces, and the one-thread BLAS limit the fitting steps run under |
| `products.py` | reading a finished run back: where its products are, the settings recorded beside them, and the figures `evaluation/` and `experiments/` share. No step imports it |

## The segmentation

The pipeline does not detect sources -- it is given a segmentation map. `SExtractor/`
holds the configuration used to produce the one shipped with this project, and its
README covers running it. A user supplying their own needs a 2-d integer map on the
white light image's pixel grid, one ID per source and 0 for background.

## Neighbours

`evaluation/` asks how the current results look; `experiments/` asks whether something
should be done differently; `evaluation/poster/` is the same figures with print
typography. Each has its own README.
