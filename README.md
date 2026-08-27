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

Python 3.12 or newer, on Linux, macOS or Windows. Pick either tool; both install
the same package from the same `pyproject.toml`.

```bash
git clone <url> astro
cd astro
```

**conda**

```bash
conda env create -f environment.yml
conda activate astro
```

**uv**

```bash
uv venv --python 3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e .
```

Either way, `skymodel` becomes a command:

```bash
skymodel configs/p01.yaml
```

### Checking it took

```bash
python -c "from skymodel import run_pointing, load_config; print('ok')"
```

That imports the pipeline without touching any data, so it fails on a broken
install and not on a missing cube.

### On Windows

Run under UTF-8:

```
set PYTHONUTF8=1
```

The pipeline writes its logs as UTF-8 and its messages contain non-ASCII
characters. Without this, redirecting the output to a file on a system whose
locale is not UTF-8 fails on the first such message. Python 3.15 makes UTF-8 mode
the default and the setting stops being necessary.

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

The templates a source is fitted against ship with the code, since they are small
and nothing runs without them: the four galaxy eigenspectra of Bolton et al. 2012
(`data/eigen_galaxy_Bolton2012.fits`) and seven stellar templates, O through M
(`data/stellar_templates/`). The cubes and the segmentation are yours to supply.

## Run

Everything a pointing needs is in one config file; nothing else is passed on the
command line.

```bash
conda run -n astro python src/skymodel/pipeline.py configs/p01.yaml
conda run -n astro python src/skymodel/pipeline.py configs/*.yaml   # all 14
```

About 80 seconds and 6 GB of memory per pointing, and about 8 GB of output.

The products land under the config's `output` directory:

```
results/skymodel/p01/
  step01/  white light image, and the segmentation placed beside it
  step02/  source_spectra.npz: every source's summed spectrum, variance, spaxel count
  step03/  the sky continuum, the sky-line mask of every continuum iteration,
           the K basis vectors
  step04/  source_fits.npz and classification.npz: each source's class, redshift
           and amplitudes, scanned once against the stellar library and once
           against the galaxies, with meta.json saying how they were fitted; the
           scans themselves go to scans/ and only with source_fit.keep_scans
  step05/  s solved per blank spaxel, and the smooth field it is forced onto
  step06/  sky_subtracted.fits, sky_model.fits, and the per-spaxel coefficients
  stepN.log  each step's full output, headed by the call that produced it
```

`configs/README.md` documents every field of a config.

## The six steps

The pipeline is six steps in order:

    whitelight         collapse a cube along wavelength into a white light image
    source_spectra     sum each source's spectrum over the spaxels its seg ID covers
    sky_basis          learn the sky continuum and the sky-line basis from blank
    classify_sources   fit templates to every source, giving it a class and a redshift
    fit_sky_amplitude  force the sky continuum amplitude s onto a spatial field
    subtract_sky       apply the model to every spaxel and write the subtracted cube

`run_pointing` is those six, driven by one pointing's config file, and it is the only
way in -- the same thing `pipeline.py` does from a shell:

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
the basis moves by about 1 part in 10⁴. Steps 1, 3, 4, 5 and 6 stamp the git commit
into their `meta.json`, each `stepN.log` opens with the call that produced the products
beside it, and `{output}/config.json` records the config the run was given.

```bash
python scripts/verify.py p05 p14
```

re-runs those pointings and compares every product against the stored ones, which
is how a change meant to leave the answer alone is held to it. `scripts/README.md`
covers that and two faster checks beside it.

## Repository

```
src/skymodel/              the pipeline
  evaluation/              figures and numbers from the products (see its README)
  evaluation/poster/       the same figures, laid out for print
  experiments/             one-off "should we do it differently" tests
src/zap/                   the ZAP comparison arm -- a different method, same data
configs/                   one file per pointing
scripts/                   checking a change did what it says (see its README)
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
