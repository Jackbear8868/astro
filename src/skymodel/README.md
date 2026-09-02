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

Six, in order, by what each one does:

    1  whitelight          collapse the cube along wavelength into a white light image
    2  source_spectra      sum each source's spectrum over the spaxels its seg ID covers
    3  sky_basis           learn the sky continuum and the K sky-line basis vectors
    4  classify_sources    fit templates to every source, giving it a class and a redshift
    5  fit_sky_amplitude   force the sky continuum amplitude s onto a smooth spatial field
    6  subtract_sky        apply the model to every spaxel and write the subtracted cube

| | function | reads | writes |
|---|---|---|---|
| 1 | `whitelight` | the ESO nosky cube | `step01/whitelight_nosky.fits` + preview |
| — | `place_segmentation` | the segmentation it is given | `step01/segmentation_input.fits`, after checking the two share a pixel grid |
| 2 | `source_spectra` | nosky cube, seg | `step02/source_spectra.npz` |
| 3 | `sky_basis` | wsky cube, seg | `step03/sky_continuum.npy`, `continuum_iterations.npz`, `sky_line_basis_{method}_K{K}.npy` |
| 4 | `classify_sources` | `step02/`, `step03/` | `step04/source_fits.npz`, `classification.npz`, `meta.json`, and `scans_star.npz`, `scans_galaxy.npz` with `source_fit.keep_scans` |
| 5 | `fit_sky_amplitude` | wsky cube, `step03/`, `step04/` | `step05/sky_continuum_amplitude_per_spaxel.npy`, `sky_continuum_amplitude_field.npy`, `main_source_group.png` |
| 6 | `subtract_sky` | wsky cube, `step03/`–`step05/` | `step06/sky_subtracted.fits`, `sky_model.fits`, `source_template_amplitude_map.npy` |

Step 3 writes no single sky-line mask: the three stacks hold the continuum, the detection
threshold and the mask of every iteration, one row each, and steps 4 to 6 use the row
`source_fit.line_mask_iter` names. It reads one thing more when
`sky_line_basis.borrow_from` names another run: that run's
`step03/sky_line_basis_{method}_K{K}.npy`, resampled onto this pointing's wavelength
grid and put in place of the basis this step would have learned. Only the basis is
taken; the mean spectrum, the continuum and the masks are still learned here.
`sky_line_basis.mask_source_lines` changes the same one thing from the other end: the
channels the source's own emission lines land on are zeroed in the decomposition
input, so the basis is exactly 0 there and step 6 cannot subtract those lines from
anything. The two scans step 4 can write per source are
its two branches, the stellar library and the galaxy eigenspectra, not two passes over
one of them. Step 6 does not write the s it applied: wherever that has a value it is
step 5's field to the bit, and which spaxels step 6 solved is `np.isfinite` of any
channel of `sky_model.fits`.

The white light comes from the **nosky** cube, not the wsky one: downstream needs it to
find the main source (the blob holding the brightest pixel), and the sky continuum of
the wsky cube lifts the whole image, which makes the brightest pixel unreliable.

Step 2 also runs on the nosky cube. Classifying a spectrum that still holds the sky
produces output that looks entirely normal, with every template and redshift wrong.

## Fitting a template to every source -- what step 4 writes

One run, in files whose names say what they are rather than how they were fitted:

    step04/
      source_fits.npz      one row per source: class, redshift, template amplitudes,
                           and the galaxy branch's own best redshift beside the winner's
      classification.npz   what steps 5 and 6 read back
      meta.json            every setting the run was made with, machine-readable
      scans_star.npz       every star scan, one member per source, with keep_scans
      scans_galaxy.npz     the same for the galaxy branch

Every setting that decides a number in there -- the fit window, the redshift grid, the
stellar library, which sky-line mask iteration was excluded -- is a key of `meta.json`,
so a directory is read by opening one JSON file and not by parsing a file name. The
price is that the settings no longer separate two runs written to the same place: a
second run overwrites the first, and `output` in the config is what keeps them apart.

`source_fit.line_mask_iter` is the one thing that can put several fits in one step04.
It is a list, and with more than one entry each iteration gets `step04/mask_iter{N}/`
holding the same files one level down, the way step5 keeps an alternative run beside
its own. All 14 configs name a single iteration, so their step04 is flat.

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

`source_fit.keep_scans` is the same kind of switch one level down: it decides whether
step 4 writes each source's redshift scan as well as the row it picked out of it.
Those scans are nearly all of what step04 weighs, and `configs/README.md` says what is
in one and what still reads it.

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
| `pipeline.py` | `Pipeline`, whose methods are the six steps and the segmentation check between the first two, in that order; the products they hand each other are above the class and each step's own helpers immediately above that step; `run_pointing` and the command line are at the end |
| `config.py` | reads and checks a pointing config; every value the pipeline takes comes from there |
| `utils.py` | everything the six steps share: the wavelength axis and the air-to-vacuum conversion, continuum estimation and line detection, the source templates (eigenspectra and the stellar library, read as splines), the per-spaxel solves steps 5 and 6 share, the main source group, the s-field construction, the figures the pipeline itself produces, and the one-thread BLAS limit the fitting steps run under |
| `products.py` | reading a finished run back: where its products are, the settings recorded beside them, and the figures `evaluation/` and `experiments/` share. No step imports it |

## The main source group

Step 5 needs the whole footprint of the main galaxy, and `utils.main_source_group`
assembles it from several segmentation IDs rather than picking one. A single
"largest-area" or "brightest" ID does not work: SExtractor's deblender splits the main
galaxy into a number of pieces that varies with the exposure's seeing and dither, so
any "pick one ID" rule gets part of it, and downstream uses that piece to decide which
side to mask and how far to exclude around it.

## The segmentation

The pipeline does not detect sources -- it is given a segmentation map. `SExtractor/`
holds the configuration used to produce the one shipped with this project, and its
README covers running it. A user supplying their own needs a 2-d integer map on the
white light image's pixel grid, one ID per source and 0 for background.

## Neighbours

`evaluation/` asks how the current results look; `experiments/` asks whether something
should be done differently; `evaluation/poster/` is the same figures with print
typography. Each has its own README.
