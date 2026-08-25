# The pipeline

Six steps, run in order by `run_pipeline.py` from one pointing's config file. Reading
`run_pointing()` is meant to be enough to know what the pipeline does; this file says
what each step reads and writes, and why the shape is what it is.

    conda run -n astro python src/skymodel/run_pipeline.py configs/p01.yaml

## The model

    D(p, λ) = s(p) · C_sky(λ) + Σₖ lₖ(p) · Lₖ(λ)

One sky continuum shape scaled per spaxel, plus K sky-line basis vectors. Both are
learned from the blank spaxels of the sky-included cube. On a source spaxel a template
`Σⱼ Aⱼ(p) · Tⱼ(λ/(1+z))` is fitted alongside, so the sky model cannot absorb the source.

## The steps

| | function | reads | writes |
|---|---|---|---|
| 1 | `whitelight` | the ESO nosky cube | `step01/whitelight.fits` + preview |
| — | `place_segmentation` | the delivered seg map | `step01/seg.fits`, after checking the two share a pixel grid |
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

## Why the products go to disk

Each step writes to `{output}/stepNN/` and the next reads from there, rather than
passing arrays in memory. That is deliberate:

- the products are the interface the 23 scripts under `evaluation/` read
- a step can be repeated on its own without redoing the ones before it
- peak memory stays at the cost of the largest single step (about 7 GB, step 6)
  instead of the sum of all six
- `sky_subtracted.fits` is 4 GB and is the deliverable anyway

## Calling a step directly

Every step is a function with named parameters, and `main()` is that function driven
from a command line. Both do the same work:

```python
from skymodel import sky_basis
sky_basis(work="results/skymodel/p01",
          cube="data/wsky/DATACUBE_FINAL_1.fits", K=30, ylim=[170, 9999])
```

```bash
conda run -n astro python src/skymodel/step3_sky_basis.py \
    --work results/skymodel/p01 --cube data/wsky/DATACUBE_FINAL_1.fits \
    -K 30 --ylim 170 9999
```

The head of each `stepN.log` is the equivalent Python call, so a step is repeatable
from its own log.

## Shared modules

| file | holds |
|---|---|
| `config.py` | reads and checks a pointing config; every value the pipeline takes comes from there |
| `utils.py` | continuum estimation and line detection, the main source group, the s-field construction |
| `fitting.py` | the per-spaxel solves shared by steps 5 and 6 |
| `templates.py` | the source templates: eigenspectra and the stellar library, read as splines |
| `plotting.py` | figures the pipeline itself produces |

## K, and why it is required everywhere

`-K` has no default in any step. All three of steps 3, 4 and 6 must use the same K; with
separate defaults, one step missed means silently reading a different basis.

## The segmentation

The pipeline does not detect sources -- it is given a segmentation map. `SExtractor/`
holds the configuration used to produce the one shipped with this project, and its
README covers running it. A user supplying their own needs a 2-d integer map on the
white light image's pixel grid, one ID per source and 0 for background.

## Neighbours

`evaluation/` asks how the current results look; `experiments/` asks whether something
should be done differently; `evaluation/poster/` is the same figures with print
typography. Each has its own README.
