# Pointing configs

One file per pointing of the Haro 11 mosaic. Everything the pipeline needs for
that pointing lives in it; nothing else is passed on the command line.

    conda run -n astro python src/skymodel/run_pipeline.py configs/p01.yaml
    conda run -n astro python src/skymodel/run_pipeline.py configs/p0[1-4].yaml

Paths are written relative to the repository root. `src/skymodel/config.py`
reads a file, checks the values and hands them out with the paths resolved.

## Sections

| key | what it settles |
|---|---|
| `input` | the three files: the wsky cube the sky is learned from, the ESO nosky cube used for the white light and the source spectra, and the segmentation delivered by the professor |
| `output` | where this pointing's `step01` … `step06` directories go |
| `sky_region` | which part of the field the sky is learned from |
| `sky_line_basis` | the sky model: continuum estimation and the K line-basis vectors (step3) |
| `source_fit` | the template fit that gives every source a class and a redshift (step4) |
| `s_field` | the spatial field the sky continuum amplitude s is forced onto (step5) |
| `spaxel_fit` | applying the model to every spaxel (step5 and step6) |

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
and `s_field` (step5, which spaxels train the spatial field of s). All 14
configs currently restrict `basis` only.

These boxes are read off the pseudo-r isophotes of each pointing by eye, not
derived from a rule, so they are not something to recompute.
`src/skymodel/experiments/sky_region_visual.py` redraws the figures they are
read from.
