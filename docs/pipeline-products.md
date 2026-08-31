# What each step writes, and what is inside it

> Every file a pointing leaves under `results/skymodel/pNN/`: what it holds, what
> the numbers mean, and which script draws it. Measured from p14 -- 20 sources,
> 3681 channels, 323 x 343 spaxels.
>
> `src/skymodel/README.md` has the table of which step writes what. This document
> is about the contents.

---

## 0. One pointing

```
results/skymodel/p14/
  step01/     959 KB    white light and segmentation
  step02/       2 MB    one spectrum per source
  step03/     764 KB    the sky model
  step04/      24 MB    the template fit
  step05/       1 MB    the sky amplitude field
  step06/       5 GB    the subtracted cube
  config.json           the config this run was given, resolved
  step1.log … step6.log each step's full output
```

Step 6 is 99% of the total. The first five together are 28 MB, which is why
`keep_intermediate` is a tidiness switch and not a storage decision.

`config.json` is written before step 1 runs, whatever `keep_intermediate` says.
It is the only answer to "what values produced this directory", since the YAML it
came from can be edited afterwards.

## 1. step01 -- white light and segmentation

```
whitelight_nosky.fits      436 KB   PRIMARY (323, 343) float32
segmentation_input.fits    439 KB   DATA    (323, 343) int32
whitelight_preview.png      84 KB
meta.json                  361 B
```

**`whitelight_nosky.fits`** is the **nanmean** of the nosky cube along wavelength
-- a mean, not a sum, because the number of usable channels differs from spaxel to
spaxel and dividing by the count is what makes two spaxels comparable. The header
carries the cube's celestial WCS, which is what lets the segmentation be checked
against it rather than merely compared for shape.

`0` in this image does not mean "measured zero". It means the spaxel had no valid
channel at all: `nanmean` over all-NaN gives NaN, and the next line turns NaN into
0. `white != 0` is how the whole repository defines the field of view. About half
the pixels are negative, which is ESO subtraction residual and not an error.

The filename names the cube because the pointing has two -- one with the sky still
in it and this one with the sky removed -- and an evaluation script collapses the
other. Everything downstream that asks "where is the source" needs this one: the
sky continuum of the wsky cube lifts the whole image, which makes the brightest
pixel unreliable, and the main source is found as the blob holding it.

**`segmentation_input.fits`** is a byte-identical copy of the segmentation the run
was given. The pipeline does not detect sources; which spaxels hold one is an
input named by the config. The copy exists so that the 36 scripts that read it
address one fixed path and need to know nothing about where the inputs live.

**`meta.json`** records the grid check. The check is "where on the sky does this
pixel point", not a keyword comparison -- the segmentation carries a CD matrix
while the cube uses PC + CDELT, which a literal comparison would report as a
mismatch -- and it takes the largest separation over the four corners and the
centre.

```json
{ "grid_offset_px": 0.3302, "max_grid_offset": 0.5,
  "seg_source": "data/wsky_seg/DATACUBE_FINAL_14_seg.fits",
  "seg_md5": "271ec76ad8f7d9f271ca3e6a00122dcc" }
```

The checksum is there because the copy beside it is byte-identical to its source
and the source can be replaced; the digest is what still identifies it then.

**Drawing it**

```bash
conda run -n astro python src/skymodel/evaluation/seg_id_map.py --work results/skymodel/p14
conda run -n astro python src/skymodel/evaluation/whitelight_compare.py --work results/skymodel/p14
```

## 2. step02 -- one spectrum per source

```
source_spectra.npz    2 MB
    ids            (20,)        int32     the segmentation labels
    flux_sum       (20, 3681)   float64
    variance_sum   (20, 3681)   float64
    spaxel_count   (20, 3681)   float64
    wavelength     (3681,)      float64   air, Angstrom
```

Each source's spectrum summed over the spaxels its segmentation ID covers, from
the **nosky** cube -- classifying a spectrum that still holds the sky produces
output that looks entirely normal with every template and redshift wrong.

Three things about this file are easy to read wrongly.

**`flux_sum` is a sum.** Comparing two rows measures area far more than
brightness: in p14 the largest source's median is 9291 times the smallest's, while
their areas differ by a factor of 378. The mean spectrum is
`flux_sum / spaxel_count`.

**`spaxel_count` is per channel, not per source.** Its shape is `(20, 3681)`. A
voxel is counted only when the flux is finite, the variance is finite and the
variance is positive, so the count follows wavelength: source 13 has 14 spaxels
and its worst channel keeps 7 of them. Two channels of one source are not on the
same footing until each is divided by its own count.

**`variance_sum` is a variance, not a sigma, and it is the variance of the sum.**
The variance of the mean is `variance_sum / spaxel_count**2` -- squared, because
the variance of a mean is `1/n^2` times the variance of the sum.

Flux and variance share one validity mask rather than each using `nansum`. This is
a correctness property and not an accident: a voxel with finite flux and NaN
variance would otherwise enter one array and not the other, and the two would no
longer describe the same spaxels.

The row order is `ids`, not `0..n-1`. In p14 the two happen to coincide, which only
means no source fell entirely outside the field of view.

**Nothing reads this file.** Step 4 is handed the arrays in memory. It is on disk
so the spectra that were classified can be looked at afterwards, which is the
reason the intermediate products exist at all.

## 3. step03 -- the sky model

```
wavelength.npy                 29 KB    (3681,)     air, Angstrom
blank_mean_spectrum.npy        29 KB    (3681,)     the raw material
sky_continuum.npy              29 KB    (3681,)     C_sky
sky_line_basis_svd_K30.npy    431 KB    (30, 3681)  the K basis vectors
continuum_iterations.npz      245 KB    how C_sky was arrived at
meta.json                     609 B
```

The two halves of the model:

```
D(p, λ) = s(p)·C_sky(λ) + Σₖ lₖ(p)·Lₖ(λ)
          └ sky_continuum ┘  └ sky_line_basis ┘
```

Both are learned from the blank spaxels of the **wsky** cube. A cube whose sky has
already been removed has no sky to learn from.

### Which spaxels count as blank

```
n_blank_all        82340    white light non-zero, and seg == 0
n_blank_used       58090    minus the config's sky_region box
n_blank_complete   50316    minus the spaxels missing any channel
```

The last cut is worth knowing about: differential atmospheric refraction moves a
spaxel's footprint with wavelength, so spaxels at the field edge are covered in
part of the band and not the rest. 7774 of them are dropped, and
`n_blank_complete` is the count that actually produced the mean spectrum.

### The iteration

`blank_mean_spectrum.npy` is the sigma-clipped mean of those spaxels. The
continuum is estimated from it by masking the lines and re-fitting, and that loop
is self-reinforcing: masking emission lowers the continuum, which shrinks the
threshold scale, which lowers the bar for the next pass.

| pass | masked | continuum median | threshold scale median |
|---|---|---|---|
| 1 | 35.5% | 31.791 | 5.0832 |
| 2 | 56.1% | 27.526 | 1.9691 |
| 3 | 69.9% | 26.259 | 1.0570 |
| 4 | 79.1% | 25.875 | 0.7375 |

`continuum_iterations.npz` holds all three quantities, one row per pass, under the
keys `continuum`, `threshold` and `line_mask`. They are one loop's record and share
one axis, which is why they are one file.

**The loop did not converge; it hit a floor.** A fifth pass would have left 14.1%
of the channels unmasked, below `min_unmasked_frac` -- past that there is not
enough continuum left to measure one from. The pass that triggered the stop is
discarded, so the stack's row count is the only record of how far it got, and
`meta.json` records `n_iterations` alongside the `max_iter` that was never reached.

**The masks are not cumulative.** Every pass re-judges against the untouched mean
spectrum, so a channel flagged early can be unflagged later: in p14 the union of
the four rows is 2933 channels and the last row is 2913.
`utils.load_line_masks` applies the running `logical_or` at read time, which is
what steps 4 to 6 use.

**`threshold` is not a noise sigma.** It is
`running_median(|mean − continuum|, 300)`, a local roughness scale with one job:
setting the line threshold at `+1` and `-2` of itself. It is about 40 times the
actual uncertainty on the mean spectrum, and it falls by a factor of 6.9 across
the four passes purely because the estimator masks more of its own input each time.

**Steps 4 to 6 use row 0, not the last row** -- iteration 1, 35.5% of the channels,
named by `source_fit.line_mask_iter` in the config. Reading the last row as "the
line mask" would conclude that three quarters of MUSE is sky line.

### The basis

30 orthonormal vectors, learned by truncated SVD of the blank residuals after
`C_sky` has been subtracted. Every one of them puts at least 99.0% of its energy
inside iteration 1's mask, which is why the filename says `line`: **the sky cannot
be rebuilt from this file alone.** It needs `sky_continuum.npy` and a per-spaxel
amplitude from step 5.

The method and K are in the filename so two values of K can sit side by side. The
seed is not -- it is in `meta.json`, and without it the randomized SVD would not
repeat.

With `sky_line_basis.borrow_from` the vectors in this file were not learned here at
all: they are another run's, resampled onto this pointing's grid with a cubic spline
and re-orthonormalised, since the two pointings share a channel width but not a zero
point. The filename is the same, so `meta.json` is the only thing that says so:

```json
{ "borrowed_basis": { "svd": {
    "run": "results/skymodel/p11",
    "basis_file": "results/skymodel/p11/step03/sky_line_basis_svd_K30.npy",
    "basis_md5": "b78735c584f7f5f081f5455212795154",
    "source_wavelength": [4599.8472, 9349.8472],
    "target_wavelength": [4749.8296, 9349.8296],
    "channel_offset": 119.9859375,
    "orthonormality_before": 1.27e-3, "orthonormality_after": 4.70e-8 } } }
```

The key is absent from a run that learned its own basis. Nothing else in the
directory changes: the mean spectrum, the continuum and the line masks are this
pointing's own either way, which is what keeps step 4 comparable with a run that
borrowed nothing.

With `sky_line_basis.mask_source_lines` the vectors were learned here, but from an
input whose source emission-line channels were set to 0, so every one of them is
exactly 0 at those channels and the model cannot put flux there. Again the filename
is the same and `meta.json` is the only thing that says so:

```json
{ "masked_source_lines": {
    "n_channel": 39, "channel_fraction": 0.0106,
    "windows": [ { "low": 5106.0, "high": 5118.0, "n_channel": 10 } ] } }
```

The windows are observed wavelengths and carry no redshift; the config names them as
bounds and step 3 records the same bounds. The per-window channel count is what the grid gave, so it moves
by one between pointings whose grids are offset by less than a channel.

The key is absent from a run that masked nothing, and the products beside it are
again unchanged: the exclusion happens after the mean spectrum, the continuum and
the line masks are built, so only the basis file differs.

### Learning from a flux window instead

With `sky_line_basis.select_faintest` the spaxels counted above are narrowed once
more, to the flux window the ESO pipeline would have called sky: the valid field is
ranked by its mean over wavelength, the faintest `ignore` of it is thrown away and
the sky comes from the next `fraction`. On p14 with ESO's own 0.05 / 0.10 that is

```
n_blank_complete    50316    the count above, before the window
n_selected           8225    inside the flux window as well
```

**This one moves every file in the directory**, unlike the two keys above: the mean
spectrum, `sky_continuum.npy`, the masks and the basis are all built from the
selected spaxels, so step 4's fitting channels move with them. `meta.json` is again
what says so:

```json
{ "selected_faintest": {
    "rule": "eso_skymodel_ignore_fraction", "ignore": 0.05, "fraction": 0.10,
    "ranked_over": "valid field, spectrally complete", "n_ranked": 95599,
    "flux_low": 67.5358, "flux_high": 82.0066,
    "n_window_in_field": 9560, "n_offered": 50316, "n_selected": 8225 } }
```

`n_window_in_field` is how many spaxels of the ranked field the window holds, and
`n_selected` how many of those survived the segmentation, the box and the
completeness cut -- the two together are what say how much of ESO's own sample this
run could use. The key is absent from a run that selected nothing.

**Drawing it**

```bash
conda run -n astro python src/skymodel/experiments/plot_linemask_iters.py --work results/skymodel/p14 --with-rejected
conda run -n astro python src/skymodel/evaluation/plot_basis.py --work results/skymodel/p14 --basis svd -K 30
conda run -n astro python src/skymodel/evaluation/continuum_compare.py
```

## 4. step04 -- the template fit

```
source_fits.npz        8 KB    one row per source
classification.npz     3 KB    what steps 5 and 6 consume
meta.json            555 B     every setting the fit was made with
scans_star.npz         1 MB    the whole chi2 surface, star branch
scans_galaxy.npz      23 MB    the whole chi2 surface, galaxy branch
```

Every source is scanned against two model families independently, and the two
winners then face each other:

```
star branch      7 templates x 101 radial velocities  =    707 candidate fits
galaxy branch    1 eigen model x 15001 redshifts      =  15001 candidate fits
```

The comparison is on reduced chi2, which already accounts for the two branches
having different degrees of freedom -- four galaxy components against one stellar
template -- and that is what makes an absolute threshold unnecessary. It holds only
because both branches are given the same channels.

### `source_fits.npz` -- one row per source

The winning row, plus **both** branches' scores, so "by how much did it win" stays
answerable. Two of its columns are not obvious:

- `nspax` is the median over wavelength of step 2's per-channel spaxel count, not
  the segmentation area. For source 13 those are 11 and 14.
- **`gal_z` is the galaxy branch's own best redshift, which is not `z`.** `z`
  belongs to whichever branch won, and for a source classified as a star it is a
  radial velocity. Step 5 needs the galaxy value for every source to decide which
  segmentation IDs are one galaxy.

`chi2` and `red_chi2` are **not comparable between rows**. Sigma comes from the
cube's STAT extension at face value, so a bright source has small sigma and a
slight imperfection of the model becomes an enormous chi2. Only the star-against-
galaxy comparison within one row means anything.

### `classification.npz` -- the contract with step 6

A strict column subset of the file above -- `id`, `group`, `template`, `z`, `A` --
plus `star_library`. Step 6 rebuilds each source's model from exactly these. The
library name is stored so that rebuilding with a different library than the fit
used is a hard error rather than a silent wrong answer.

### The scans

One file per branch, keyed by source. Each member is a structured array with one
row per candidate fit, ordered by chi2 rather than by redshift.

```python
from utils import load_scan, scan_ids

scan_ids(step04, "galaxy")            # [1, 2, ..., 20]
r = load_scan(step04, "galaxy", 1)    # 15001 rows
r["z"], r["red_chi2"], r["template"]
```

`group` is one value for a whole file and the template name is one of a handful,
so both are stored once and indexed; `load_scan` gives the name back, so a reader
does not see the compaction.

The scans are the search, not the result, and `source_fit.keep_scans` turns them
off. Two scripts need the whole curve -- `evaluation/chi2_scan.py` and
`evaluation/main_group_spec.py` -- and both say so when the files are absent.

### Where the settings are

Not in the filenames. A step04 directory holds one run, and `meta.json` beside the
products records the cube, the segmentation, the spectra, the fit window, the mask
iteration, the whole redshift grid and the stellar library. Several mask iterations
are several runs and get a directory each, `step04/mask_iter{N}/`, the way step05
already treats alternative runs.

**Drawing it**

```bash
conda run -n astro python src/skymodel/evaluation/chi2_scan.py --work results/skymodel/p14 --id all
conda run -n astro python src/skymodel/evaluation/main_group_spec.py -n 14
```

## 5. step05 -- the sky amplitude field

```
sky_continuum_amplitude_per_spaxel.npy   433 KB   (323, 343)  80,295 of 110,789 finite
sky_continuum_amplitude_field.npy        433 KB   (323, 343)  all finite
main_source_group.png                    473 KB
meta.json                                  1 KB
```

`C_sky` is one spectrum for the whole pointing, but the sky is not uniform across
the field, so every spaxel needs its own scaling. `s` is dimensionless: `s = 1`
means "this spaxel has exactly the sky continuum of this pointing's own blank
spaxels".

**`..._per_spaxel.npy`** is the free solve, one least-squares fit per blank spaxel.
Its NaN carries three unrelated meanings that the file cannot separate: a source
sits here, this is outside the field of view, or the channel coverage is below the
threshold.

**`..._field.npy`** is that solve forced onto a smooth field, defined everywhere
including over the sources and outside the field of view.

### Why the free solve is not used directly

Next to a source it is propped up by the source's own light:

| | spaxels | median of free − field |
|---|---|---|
| within 15 px of a source | 20,198 | +0.01253 |
| further than 15 px | 60,097 | +0.00325 |

The PSF wings leak into blank spaxels, the free solve raises `s` to absorb them,
and a larger `s` means a larger sky model -- which then subtracts the source's own
light as if it were sky. This is the entry point of over-subtraction, and closing
it is what the field is for.

### The training set

```
n_blank   80,295     the free solve has a value
                     more than 15 px from any source
                     more than 50 px from the main galaxy
                     within 8 robust spreads of the median
n_train   38,290     what actually decides the field
```

The main galaxy gets its own radius because its extended halo reaches well past
the PSF wings of a small source and would otherwise be learned as sky. Which
segmentation IDs form it is decided by step 4's `gal_z`, recorded as `main_ids`.

### The shape of the field

`mu + a(y) + b(x)`, solved by Tukey's median polish -- alternating row and column
medians until they stop moving. It is exactly additive to float32 precision.

Additive rather than a general `f(x, y)` because a general `f` is one number per
pixel and therefore has nothing to say where there is no data -- and under a source
there is none. This form has `1 + ny + nx` parameters, and `a(y)` is shared by a
whole row, so the offset in the middle of a source is set by training spaxels on
the same row far away from it. **That is what lets it reach into the holes.** The
cost is that it represents stripes and linear gradients and nothing else;
everything local stays in the residual.

It halves the scatter: the free solve's robust half-width is 0.02060 and the
field's is 0.01049.

### Where the field asserts rather than measures

`a(y)` is the median of that row's training spaxels. For **41 of 323 rows and 10
of 343 columns there are none**, and `utils.nanmed` supplies 0 -- "apply no
correction", which its own docstring calls an assumption rather than a
measurement. 11,353 spaxels that step 6 went on to fit sit in one of them.

The value there is not 1; it is `mu + b(x)`, so those spaxels still have whatever
their column says. But every one of the untrained rows and columns is at the edge
of the field, which is where "assume this row is average" is least likely to hold.
The field itself cannot show this -- it has no NaN anywhere -- so `meta.json`
records `untrained_rows` and `untrained_cols`.

**`main_source_group.png`** is the check that a deblended galaxy was put back
together correctly: every seg ID in the adjacent blob on the left, the ones the
redshift criterion kept on the right. No script reads it. A wrong grouping is
invisible in the numbers and has to be seen.

**Drawing it**

```bash
conda run -n astro python src/skymodel/evaluation/s_shape_map.py --work results/skymodel/p14
conda run -n astro python src/skymodel/evaluation/s_compare.py --which both --separate
conda run -n astro python src/skymodel/evaluation/sky_region_map.py --work results/skymodel/p14
```

## 6. step06 -- the subtracted cube

```
sky_subtracted.fits                 3 GB   PRIMARY + DATA + STAT
sky_model.fits                      2 GB   PRIMARY + DATA
source_template_amplitude_map.npy   2 MB   (4, 323, 343)
meta.json                         576 B
```

The model is applied to every spaxel that passes `white != 0` **and**
`coverage >= min_channel_coverage`. The field having values outside the field of
view is therefore harmless: those spaxels are never fitted.

```
blank   (seg == 0)    D(λ) = s·C_sky + Σₖ cₖ·Lₖ                    80,295 spaxels
source  (seg > 0)     D(λ) = s·C_sky + Σₖ cₖ·Lₖ + Σⱼ Aⱼ·Tⱼ(λ/(1+z))  22,337
```

`s` is locked to step 5's field on both sides. Letting it float on a source would
let it rise to absorb the source's light, and the sky model would grow with it.

**The source template is fitted but not subtracted.** `sky_model.fits` holds
`s·C_sky + Σₖ cₖ·Lₖ` and nothing else. The template's job is to occupy the source's
place while the sky coefficients are solved, so that they do not absorb it; once
solved, the `A·T` term is dropped. The deliverable is a cube with the sky removed,
not one with only the sky left, and a source's spectrum after subtraction is
non-zero on purpose.

**`source_template_amplitude_map.npy`** holds those amplitudes anyway, per spaxel:
four planes for a galaxy, one plane and three NaN for a star. Note that this `A` is
not the `A` in `classification.npz` -- step 4 fits one per region on the summed
nosky spectrum, step 6 fits one per spaxel on the wsky cube.

### Three traps

**`sky_model.fits` is finite in channels where the data is NaN.** It is a model
evaluation, not a measurement: once a spaxel passes the coverage test the whole
spectrum can be evaluated, including channels the data is missing. Computing
`D - M` yourself and comparing NaN counts against the delivered cube will not
match.

**STAT is the input's, unchanged.** Two HISTORY cards in the file say so: it does
**not** carry the uncertainty of the sky model itself.

**The primary header is the ESO original, verbatim** -- `PIPEFILE`,
`ESO PRO CATG`, its own `DATE`, 1212 HIERARCH cards. A FITS viewer will describe
this file as an ESO MUSE DATACUBE_FINAL. The commit and the timestamp of the run
that produced it are in `meta.json`, not in the FITS.

### What NaN means

| file | NaN means |
|---|---|
| `sky_model.fits` | this spaxel was not fitted -- outside the field of view, or coverage below the threshold. The same at every channel. |
| `sky_subtracted.fits` | the above, or this voxel's input was NaN. Channel-dependent. |
| `source_template_amplitude_map.npy` | not a fitted source spaxel, or this plane is beyond this source's template width |

`np.isfinite(sky_model[DATA][c])` at any channel `c` is the mask of which spaxels
step 6 solved.

**Drawing it**

```bash
conda run -n astro python src/skymodel/evaluation/blank_compare.py --work results/skymodel/p14
conda run -n astro python src/skymodel/evaluation/outside_compare.py --work results/skymodel/p14
conda run -n astro python src/skymodel/evaluation/whitelight_compare.py --work results/skymodel/p14
```

## 7. Reading the record

Steps 1, 3, 4, 5 and 6 each write a `meta.json` carrying the git commit, the
timestamp and every parameter that step was given. The `step` field is read off
the calling method at write time, so it cannot fall behind a rename.

Five facts live only there and nowhere in the arrays: step 1's measured grid
offset, step 5's list of the rows and columns whose offset was defaulted rather than
measured, step 3's `borrowed_basis`, which is where a line basis that came from
another pointing says so, step 3's `masked_source_lines`, which is where a basis
made blind at some wavelengths says so, and step 3's `selected_faintest`, which is
where a run that learned the sky from a flux window rather than from every blank
spaxel says so. The first two decide how much of a pointing's answer rests on an
assumption; the last three say what the basis beside them was built from, which no
filename carries and no array shows.

Each `stepN.log` opens with the call that produced the products beside it. It is a
record of the call, not a command to run -- the pipeline's one entrance is
`run_pointing` with a config file.
