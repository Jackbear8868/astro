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
