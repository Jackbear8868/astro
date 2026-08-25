# Sky reconstruction for the Haro 11 MUSE mosaic

MUSE spectra of a faint, extended source sit on top of a sky that is far brighter than
the source itself. This code learns the sky from the parts of the field where there is
no source, and subtracts it everywhere.

The model for one spaxel p at wavelength λ is

    D(p, λ) = s(p) · C_sky(λ) + Σₖ lₖ(p) · Lₖ(λ)

one sky continuum shape `C_sky` scaled per spaxel by an amplitude `s`, plus K sky-line
basis vectors `Lₖ` with their own per-spaxel coefficients. Both are learned from the
blank spaxels of a cube that still contains the sky. `s` is then forced onto a smooth
spatial field before the model is applied to every spaxel, source spaxels included --
where a source template is fitted alongside the sky, so that the sky is not allowed to
absorb the galaxy.

The target is Haro 11, a merging dwarf galaxy whose ionised gas reaches across a large
fraction of the field, which is what makes "where is there no source" the hard part.

---

## Install

```bash
git clone --recursive <url> astro     # --recursive: libs/zap is a submodule
cd astro
conda env create -f environment.yml
conda activate astro
```

If you already cloned without `--recursive`:

```bash
git submodule update --init
```

The submodule is only needed for the ZAP comparison arm under `src/zap/`. The pipeline
itself does not import it; `pip install -e .` is enough to use `skymodel` alone.

## Input

One pointing needs three files:

| | what | used for |
|---|---|---|
| `data/wsky/DATACUBE_FINAL_N.fits` | the cube with the sky still in it | the sky is learned and subtracted here |
| `data/nosky/DATACUBE_FINAL_ESOSKY_N.fits` | the same field, sky-subtracted by the ESO pipeline | the white light image, and the source spectra that get classified |
| `data/wsky_seg/DATACUBE_FINAL_N_seg.fits` | a segmentation map on the same pixel grid | which spaxels hold a source, and which are blank |

The cubes are MUSE data products with `DATA` and `STAT` extensions; the mosaic used
here is 14 pointings of about 4 GB each. The segmentation is a 2-d integer map, one ID
per source and 0 for background, sharing the white light image's WCS -- the pipeline
checks that agreement and refuses a pointing whose grids disagree.

Also required, and small enough to keep beside the code: the galaxy eigenspectra
(`data/eigen_galaxy_Bolton2012.fits`), the QSO eigenspectra
(`data/qso_eigen_linear_55732.dat`) and the stellar library (`data/stellar_templates/`).

## Run

Everything a pointing needs is in one config file; nothing else is passed on the
command line.

```bash
conda run -n astro python src/skymodel/run_pipeline.py configs/p01.yaml
conda run -n astro python src/skymodel/run_pipeline.py configs/*.yaml   # all 14
```

About 80 seconds and 6 GB of memory per pointing, and about 8 GB of output.

The products land under the config's `output` directory:

```
results/skymodel/p01/
  step01/  white light image, and the segmentation placed beside it
  step02/  every source's summed spectrum, its variance, its spaxel count
  step03/  the sky continuum, the sky-line mask, the K basis vectors
  step04/  the template fit: each source's class, redshift and amplitudes
  step05/  s solved per blank spaxel, and the smooth field it is forced onto
  step06/  sky_subtracted.fits, sky_model.fits, and the per-spaxel coefficients
  stepN.log  each step's full output, headed by the call that produced it
```

`configs/README.md` documents every field of a config.

## The six steps

The pipeline is six steps in order:

    whitelight         collapse a cube along wavelength into a white light image
    object_spectra     sum each source's spectrum over the spaxels its seg ID covers
    sky_basis          learn the sky continuum and the sky-line basis from blank
    classify_sources   fit templates to every source, giving it a class and a redshift
    fit_s_field        force the sky continuum amplitude s onto a spatial field
    subtract_sky       apply the model to every spaxel and write the subtracted cube

`run_pointing` is those six, driven by one pointing's config file, and it is the only
way in -- the same thing `run_pipeline.py` does from a shell:

```python
from skymodel import run_pointing
run_pointing("configs/p01.yaml")
```

Each step writes its products to disk, so every intermediate result is there to be
looked at after the run. `src/skymodel/README.md` says what each step reads and writes.

## Reproducibility

Runs are bit-reproducible: the same config gives the same products, byte for byte. Each
fitting step holds BLAS at one thread while it works, which is what makes it so -- the
randomized SVD behind the sky-line basis follows the thread count, and at 24 threads
the basis moves by about 1 part in 10⁴. Steps 3, 5 and 6 stamp the git commit into
their `meta.json`, and each `stepN.log` opens with the call that produced the products
beside it.

## Repository

```
src/skymodel/              the pipeline
  evaluation/              figures and numbers from the products (see its README)
  evaluation/poster/       the same figures, laid out for print
  experiments/             one-off "should we do it differently" tests
src/zap/                   the ZAP comparison arm -- a different method, same data
configs/                   one file per pointing
docs/                      method notes, parameter references, what was rejected
```

## The ZAP comparison

`src/zap/` runs [ZAP](https://github.com/ktsoto/zap) on the same cubes as an
independent check. Its own README has the commands. One result worth carrying over:
the source mask decides everything. Haro 11's ionised gas covers 30-44% of the field,
and a mask that misses it lets ZAP learn Hα as if it were sky and remove most of the
source.

## Credit and license

This work was carried out in Hsiao-Wen Chen's group at the University of Chicago.

MIT, see `LICENSE`. Copyright (c) 2026 Yu-Jung Lin and Hsiao-Wen Chen.
