# Pointing configs

One file per pointing of the Haro 11 mosaic. Everything the pipeline needs for
that pointing lives in it; nothing else is passed on the command line.

    conda run -n astro python src/skymodel/pipeline.py configs/p01.yaml
    conda run -n astro python src/skymodel/pipeline.py configs/p0[1-4].yaml

Paths are written relative to the repository root. `src/skymodel/config.py`
reads a file, checks the values and hands them out with the paths resolved.

## Sections

| key | what it settles |
|---|---|
| `input` | the three files: the wsky cube the sky is learned from, the ESO nosky cube used for the white light and the source spectra, and the segmentation saying which spaxels hold a source |
| `output` | where this pointing's `step01` … `step06` directories go |
| `sky_region` | which part of the field the sky is learned from |
| `sky_line_basis` | the sky model: continuum estimation and the K line-basis vectors (step3) |
| `source_fit` | the template fit that gives every source a class and a redshift (step4) |
| `sky_amplitude` | the spatial field the sky continuum amplitude s is forced onto (step5) |
| `spaxel_fit` | applying the model to every spaxel (step5 and step6) |
| `max_grid_offset` | optional, default 0.1. How far apart the seg and white-light grids may be, in pixels, before the pointing is refused. Write it only to raise it, which is a decision to run on headers that disagree; the run then prints the offset and the limit that allowed it |
| `keep_intermediate` | optional, default true. False makes steps 1 to 5 skip writing their products; step 6 always writes, being the deliverable. It changes what is left on disk and nothing else -- the step 6 of a run with it off is the step 6 of a run with it on |

## Two values that are easy to misread

`sky_line_basis.min_unmasked_frac` is a floor on how much of the spectrum
survives the sky-line mask, not a target. The continuum is estimated by masking
lines and re-fitting; if an iteration would leave less than this fraction of the
channels unmasked, there is not enough continuum left to fit and the loop stops
with the previous iteration's answer. Lowering it lets the mask grow further.

`sky_amplitude.n_iter` is how many alternating median passes build the spatial
field of s. The alternation converges -- the row and column offsets stop moving
altogether -- so this only has to be past that point, and past is free: an
iteration costs a few milliseconds against a step that takes seconds. Too few is
the failure that matters, because the field is then still drifting, and it
drifts most where it is extrapolating over the source, which is the part the
field exists to supply.

## sky_region

One box, in pixel coordinates:

    x: [0, 165]        half-open: 0 <= x < 165
    y: [null, null]    null = no bound on that side
    include: true      true keeps what is inside the box, false what is outside
    apply_to: [basis]  which steps the box restricts

`include: false` is how a pointing with the galaxy in the middle of the field
keeps its outer ring: the same single box, excluded rather than kept, so there
is no second mechanism to reason about.

`apply_to` may name `basis` (step3, which spaxels the sky basis is learned from)
and `sky_amplitude` (step5, which spaxels train the spatial field of s). All 14
configs currently restrict `basis` only.

These boxes are read off the pseudo-r isophotes of each pointing by eye, not
derived from a rule, so they are not something to recompute.
`src/skymodel/experiments/sky_region_visual.py` redraws the figures they are
read from.

## sky_line_basis.borrow_from

Optional, and written by no pointing. It names another run's output directory, and
step 3 then takes the K line-basis vectors from that run instead of learning them
from this pointing's own blank spaxels:

    sky_line_basis:
      borrow_from: results/skymodel/p11

The run named has to have finished step 3 with the same `method` and `K`, that
being the file step 3 reads: `step03/sky_line_basis_{method}_K{K}.npy`.

**Only the basis is borrowed.** The mean blank spectrum, the continuum and the line
masks in `continuum_iterations.npz` are still learned here, so step 4's fitting
channels and everything downstream that reads the continuum are this pointing's own
whatever this key says.

Every pointing has its own wavelength zero point and the offsets between them are
not whole channels, so the vectors are resampled onto this pointing's grid -- with a
cubic spline, a sky line being about two channels wide -- and re-orthonormalised
afterwards. A pointing reaching outside the range the basis was learned on is
refused rather than extrapolated: p14 runs to 9349.83 A, which p01's grid
(4599.66-9349.66 A) does not reach, so p14 cannot borrow from p01.

`step03/meta.json` records a `borrowed_basis` key naming the run, the file, its md5
and the offset between the two grids, so a run that borrowed is recognisable from
its own products and not only from this file.

## sky_line_basis.mask_source_lines

Written by all fourteen pointings. It names the source's own emission lines, and
step 3 excludes the channels they land on from the decomposition:

    sky_line_basis:
      mask_source_lines:
        - [4959.5, 4968]        # A, H-beta
        - [5057,   5069]        # A, [O III] 4959
        - [5106,   5118]        # A, [O III] 5007
        - [6694,   6709]        # A, H-alpha

Where the source fills the field, every blank spaxel still carries its emission
lines; the basis learns them along with the sky, and step 6 then subtracts them
wherever it applies the model, the source included. The channels inside each window
are set to 0 in the decomposition input.

The windows are **observed** wavelengths, in Angstrom, `[low, high]`, closed at both
ends. Nothing here is redshifted: a window is a stretch of the grid, and the redshift
that put the line there has already been applied by whoever wrote the numbers. The
same four windows serve every pointing because it is the same galaxy in all fourteen.

A mapping is accepted as well, and gives the same thing by a different route -- rest
wavelengths and one symmetric half width:

    sky_line_basis:
      mask_source_lines:
        redshift: 0.0206
        rest_wavelengths: [4861.33, 4958.91, 5006.84, 6548.05, 6562.80, 6583.45]
        half_width: 5.0                 # A, observed frame, each side of the line

`config.load()` turns it into the same list of observed `[low, high]` pairs before
step 3 sees it, so the redshift stops there and the decomposition only ever handles
wavelengths. Use the mapping when the lines are a list and one width fits them all;
use the list when the windows differ from line to line or are not centred on it,
which is what the four above are -- each reaches further to the red than to the blue.

**Only the basis is affected.** The mean blank spectrum, the continuum and the line
masks in `continuum_iterations.npz` are all built before the exclusion and see every
channel, so step 4's fitting channels and everything reading the continuum are the
same as without the key. Steps 5 and 6 need nothing added: a basis with no structure
at a wavelength cannot put flux there.

The channels are set to exactly 0, not to the channel's typical residual that the
sigma clip's rejected positions get. A constant is not nothing -- the decomposition
is uncentred, so a column holding the same non-zero number in every spaxel is still
a direction it can spend a vector on, and the constant in question is the line's own
height. At exactly 0 every basis vector is exactly 0 at those channels.

Whatever real sky falls inside the windows is removed with the line, there being no
way to tell the two apart within one channel, so the windows should be as narrow as
the line rather than as wide as its neighbourhood.

`step03/meta.json` records a `masked_source_lines` key listing every window, its
bounds and its channel count, so a run that excluded them is recognisable from its
own products and not only from this file. The count can differ by a channel
between pointings, their wavelength grids being offset from each other by less than a
channel, so a window's edge falls on either side of one.

## sky_line_basis.select_faintest

Optional, and written by no pointing. It narrows the spaxels step 3 learns from to a
flux window instead of taking every blank spaxel:

    sky_line_basis:
      select_faintest:
        ignore: 0.05                    # fraction of the field thrown away, faintest first
        fraction: 0.10                  # fraction taken as sky, immediately above it

This is the ESO pipeline's own rule, read off the cube headers: `skymethod =
subtract-model`, `skymodel_ignore = 0.05`, `skymodel_fraction = 0.10`. ESO uses no
segmentation at all -- it ranks the spaxels of the field by flux, throws away the
faintest `ignore` of them (the dead and the half-covered, which are not sky), and
learns the sky from the next `fraction`. What that also rejects, and a segmentation
does not, is low-surface-brightness light no detection ever found.

**The percentiles are taken over the whole valid field**, which is what ESO ranks, and
not over the blank set. The two are not the same question: on p14 the window over the
field is the 5th to 15th percentile of 95,599 spaxels, and the same spaxels sit between
the 9th and 25th percentile of the 50,316 blank ones, so `fraction: 0.10` read against
the blank set would be a different and smaller sample. Ranking over the field is also
what makes the cut values comparable to ESO's own.

The flux is the mean over wavelength of the sky-included cube, accumulated in the pass
that reads the blank spaxels, so ranking the field costs no second read. A spaxel whose
spectrum is not complete has no place in the ranking -- its mean is over a different
part of the spectrum than everyone else's -- so the ranked field is the valid,
spectrally complete one. The window is half-open, `(low, high]`.

**Selecting is narrowing, not replacing.** The segmentation, `sky_region` and the
completeness cut all still apply; the window then chooses among what they leave. On p14
the window holds 9,560 spaxels of the field, of which 219 are on a detected source and
1,116 are inside this pointing's excluded box, leaving 8,225.

**Unlike `borrow_from` and `mask_source_lines`, this moves every product of step 3.**
The mean blank spectrum, the continuum `C_sky`, the line masks and the basis are all
built from the selected spaxels, so step 4's fitting channels move with them. Steps 5
and 6 do not: their spaxels come from the segmentation and from
`sky_amplitude.min_source_distance`, never from what step 3 learned on, so the
amplitude field is trained on the same set with or without this key.

`step03/meta.json` records a `selected_faintest` key with the rule, the two flux cut
values, how many spaxels were ranked, how many the window holds in the field and
`n_selected`, so a run that selected is recognisable from its own products and not only
from this file. `n_blank_complete` beside it stays the count *before* the window, which
is what makes the size of the cut readable.

## source_fit.keep_scans

Step 4 scans a redshift grid for every source twice, once against the stellar
library and once against the galaxy eigenspectra, and writes the winning row of
each scan into `step04/source_fits.npz`. `keep_scans` decides whether the scans
themselves are written too, one file per source and branch:

    step04/scans/star_id7.npz
    step04/scans/galaxy_id7.npz

A scan is the surface the winner was picked out of -- the reduced chi2 of that
branch at every redshift of the grid, so it is as long as the grid is. Steps 5
and 6 take what they need from the columns of `source_fits.npz`, the galaxy
branch's redshift included, and the one program left that opens a scan is
`src/skymodel/evaluation/chi2_scan.py`, which draws the curve for a single
source.

Hence the default of false. The galaxy branch's scans alone are about 96% of the
bytes step4 writes, and they are kept for a diagnostic figure that is drawn for
one source at a time. Turn them on for the pointing whose scan you want to look
at, and the rest of the run is unchanged.
