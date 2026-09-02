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
It reaches far enough that the blank spaxels carry the galaxy's own emission lines, so
`Lₖ` is held at zero over the wavelengths those lines land on, which every pointing's
config names as `sky_line_basis.mask_source_lines`. A basis with no structure at a
wavelength cannot subtract anything there, and that is what keeps the model from
taking the galaxy's lines along with the sky.

---

## Repository

```
src/skymodel/              the pipeline
  standalone/              the same six steps as separate scripts, one runnable
                           alone (see its README)
  evaluation/              figures and numbers from the products (see its README)
  evaluation/poster/       the same figures, laid out for print
  experiments/             one-off "should we do it differently" tests
src/zap/                   a second method run on the same cubes (see its README)
configs/                   one file per pointing (see its README: every field,
                           what it does, and what to change to try a variation)
scripts/                   checking a change did what it says (see its README)
docs/                      method notes, parameter references, what was rejected
```

## Install

Python 3.12 or newer. Pick either tool; both install the same package from the
same `pyproject.toml`.

```bash
git clone git@github.com:Jackbear8868/sky-subtraction.git
cd sky-subtraction
```

**conda**

```bash
conda env create -f environment.yml
conda activate astro
```

**uv**

```bash
uv venv --python 3.12
source .venv/bin/activate
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

## Where the files live

`input` and `output` in a config take a relative path, an absolute one, or one
starting with `~`. Relative is taken against this directory, so a config reads the same
wherever it is run from; absolute is what lets the cubes and the results sit outside
the checkout entirely, which is the usual case once the data is on another disk. The
pipeline reads nothing else from the repository, so such a run writes nothing back
into it.

Figures follow the run: every script under `evaluation/` takes `--work`, the run's
output directory, and writes beside it into `evaluation/<run>/`. The few that compare
several pointings belong to no single run and go to `results/skymodel/evaluation`,
which `SKYMODEL_EVAL` moves.

## Run

Everything a pointing needs is in one config file; nothing else is passed on the
command line.

```bash
skymodel configs/p01.yaml
skymodel configs/*.yaml                                  # all 14
```

The installed command and the file are the same entry point, so a checkout that was
never installed runs it the same way:

```bash
conda run -n astro python src/skymodel/pipeline.py configs/p01.yaml
```

A pointing takes about a minute and holds one cube in memory while it works. Nearly
all of the output is step 6's two cubes, which are the size of the input cube each.

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
           scans themselves go to scans_star.npz and scans_galaxy.npz, one file
           per branch, and only with source_fit.keep_scans
  step05/  s solved per blank spaxel, and the smooth field it is forced onto
  step06/  sky_subtracted.fits, sky_model.fits, and source_template_amplitude_map.npy
           -- the source templates' amplitudes. The sky coefficients are not written:
           wherever the model was solved, s is step 5's field to the bit
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
randomized SVD behind the sky-line basis sums in an order that follows the thread
count, so the same data on a machine of a different width would otherwise give a
basis differing in its last digits. Steps 1, 3, 4, 5 and 6 stamp the git commit
into their `meta.json`, each `stepN.log` opens with the call that produced the products
beside it, and `{output}/config.json` records the config the run was given.

```bash
python scripts/verify.py p05 p14
```

re-runs those pointings and compares every product against the stored ones, which
is how a change meant to leave the answer alone is held to it. `scripts/README.md`
covers that and two faster checks beside it.

## Credit and license

This work was carried out in Hsiao-Wen Chen's group at the University of Chicago.

MIT, see `LICENSE`. Copyright (c) 2026 Yu-Jung Lin and Hsiao-Wen Chen.
