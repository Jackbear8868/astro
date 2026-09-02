# Pointing configs

One file per pointing. Everything a run needs is in it, and nothing else is passed
on the command line:

    skymodel configs/p01.yaml
    skymodel configs/p0[1-4].yaml
    conda run -n astro python src/skymodel/pipeline.py configs/p01.yaml   # same thing, uninstalled

`src/skymodel/config.py` reads the file, checks the values, and hands the pipeline a
dict with the paths resolved. A pointing takes about a minute.

To try a variation, copy a config, change `output` and the one key you are testing,
and run it. Nothing is shared between runs except the files a config names, so a
variation cannot disturb the run it was copied from:

    cp configs/p01.yaml configs/experiments/p01_wider_basis.yaml
    # edit two lines: output -> a new directory, and the key being tested
    skymodel configs/experiments/p01_wider_basis.yaml

## The six steps a config drives

Every section below says which step reads it, so here is what those steps do. They
always run in this order, and each writes into `{output}/stepNN/`:

    1  whitelight          collapse the cube along wavelength into a white light image
    2  source_spectra      sum each source's spectrum over the spaxels its seg ID covers
    3  sky_basis           learn the sky continuum and the K sky-line basis vectors
    4  classify_sources    fit templates to every source, giving it a class and a redshift
    5  fit_sky_amplitude   force the sky continuum amplitude s onto a smooth spatial field
    6  subtract_sky        apply the model to every spaxel and write the subtracted cube

---

## What a run needs on disk

**The three cubes named in `input`.** All three are required, and all three are
checked for existence before step 1 runs, so a wrong path fails in a second rather
than in a minute.

| key | what it is | which steps read it |
|---|---|---|
| `cube` | the cube with the sky still in it | step 3 learns the sky here; step 6 subtracts it here |
| `nosky` | the same field, already subtracted by the ESO pipeline | step 1's white light, step 2's source spectra (so step 4 classifies against ESO's reduction, not ours) |
| `seg` | a 2-d integer segmentation on the same pixel grid, one ID per source, 0 for background | step 1 checks and copies it; steps 2, 3, 5 and 6 read that copy |

The pipeline does not detect sources. `seg` is supplied, and step 1 refuses the
pointing if its WCS and the white light's disagree by more than `max_grid_offset`.

**What ships with the code**, used by step 4 and named by no config:

    data/eigen_galaxy_Bolton2012.fits     four galaxy eigenspectra (Bolton et al. 2012)
    data/stellar_templates/               seven class-V stellar templates, O through M

These paths are fixed relative to the repository root. To fit against a different
library you edit `DWARF_DIR` / `EIGEN_GAL` in `src/skymodel/utils.py`; there is no
config key for it.

**Only if you use `sky_line_basis.borrow_from`:** another finished run's
`step03/sky_line_basis_{method}_K{K}.npy` and `step03/wavelength.npy`. That run must
have used the same `method` and `K` as this one, and its wavelength range must cover
this pointing's.

---

## Every field

Required means the pipeline will not run without it. `load()` checks every one of
them before step 1 starts, so a missing or mistyped key costs a second, not a run.

### Top level

| key | type | required | what it does |
|---|---|---|---|
| `pointing` | integer | yes | **A label only.** It appears in the two lines the run prints and nowhere else — it does not pick a file, and nothing checks it against `input`. A config named `p14.yaml` with `pointing: 3` runs p14's data quite happily. |
| `input` | mapping | yes | the three cubes above |
| `output` | path | yes | where `step01` … `step06` and `config.json` go |
| `sky_region` | mapping | yes | which part of the field the sky is learned from |
| `sky_line_basis` | mapping | yes | the sky model (step 3) |
| `source_fit` | mapping | yes | template fitting and classification (step 4) |
| `sky_amplitude` | mapping | yes | the spatial field of s (step 5) |
| `spaxel_fit` | mapping | yes | applying the model (steps 5 and 6) |
| `max_grid_offset` | number > 0 | no, default **0.1** | how far apart the seg and white-light grids may sit, in pixels, before step 1 refuses the pointing. Write it only to raise it — that is a decision to run on headers that disagree, and the run prints the offset and the limit that allowed it. |
| `keep_intermediate` | bool | no, default **true** | false makes steps 1–5 skip writing their products. Step 6 always writes, being the deliverable. It changes what is left on disk and nothing else: the step 6 of a run with it off is the step 6 of a run with it on. |

### `input`

| key | type | required |
|---|---|---|
| `cube` | path | yes |
| `nosky` | path | yes |
| `seg` | path | yes |

Each takes a relative path, an absolute one, or one starting with `~`. Relative is
resolved against the **repository root**, not the working directory, so a config reads
the same wherever it is run from.

### `sky_region` — which part of the field counts as sky

One box, in pixel coordinates. Every key is required.

| key | type | what it does |
|---|---|---|
| `x` | `[lo, hi]`, integers or `null` | half-open: `lo <= x < hi`. `null` means no bound on that side. |
| `y` | `[lo, hi]`, integers or `null` | the same |
| `include` | bool | `true` keeps what is **inside** the box, `false` keeps what is **outside** it |
| `apply_to` | list | which steps the box restricts: `basis` (step 3) and/or `sky_amplitude` (step 5). An empty list is accepted and applies the box to nothing. |

`include: false` is how a pointing with the galaxy in the middle of the field keeps
its outer ring — the same single box, excluded rather than kept, so there is no second
mechanism to reason about.

A box is a judgement about where the source stops, read off the field by eye rather
than derived from a rule. `src/skymodel/experiments/sky_region_visual.py` draws the
field it is read from.

### `sky_line_basis` — learning the sky from blank spaxels (step 3)

The model for one spaxel is `D(p, λ) = s(p)·C_sky(λ) + Σₖ lₖ(p)·Lₖ(λ)`. This section
builds `C_sky` and the K vectors `Lₖ`.

| key | type | required | what it does |
|---|---|---|---|
| `method` | `svd` or `pca` | yes | which decomposition produces the basis. Both come out `(K, nz)`: `svd` takes K components, `pca` takes K-1 and prepends the mean spectrum as the leading row. |
| `K` | integer ≥ 1 | yes | how many line-basis vectors the sky-line part of the model has — see below |
| `seed` | integer | yes | `random_state` for the decomposition. Both PCA and TruncatedSVD are randomized, so this is what makes a basis reproducible. |
| `continuum_window` | integer | yes | width in channels of the running median that estimates the continuum |
| `line_thresholds` | `[pos, neg]` | yes | sigma above / below the continuum at which a channel is called a sky line |
| `max_iter` | integer | yes | how many times the continuum / line-mask loop may repeat |
| `clip_sigma` | number | yes | sigma clip applied **across spaxels within each channel**, before the channel is averaged. It never runs along wavelength, so a sky line — bright in every spaxel — survives it. Its purpose is to stop a few extreme negatives dragging a channel's mean down, which the continuum step would then read as a negative line. |
| `min_unmasked_frac` | number in [0, 1] | yes | a **floor on how much spectrum survives the mask**, not a target. If an iteration would leave less than this fraction of channels unmasked there is not enough continuum left to fit, and the loop stops with the previous iteration's answer. Lowering it lets the mask grow further. |
| `borrow_from` | path or `null` | no | take the basis from another run instead of learning it — see below |
| `mask_source_lines` | list of `[low, high]` or `null` | no | keep the source's own emission lines out of the basis — see below |
| `select_faintest` | mapping or `null` | no | pick sky spaxels by flux instead of by segmentation — see below |

**On `K`.** More vectors let the basis follow more of the sky's line structure, fewer
leave more of it behind. How wide a basis a pointing can carry depends on how many
blank spaxels it has to learn from, so a field with a lot of sky supports a wider one
than a field with little. Raising it costs step 3 a little time and steps 5 and 6 a
slightly larger solve, and changes no other setting.

### `source_fit` — fitting a template to every source (step 4)

Every source gets fitted twice, once against the stellar library and once against the
galaxy eigenspectra, and the lower reduced chi² on the same channel set wins. There is
no absolute threshold.

| key | type | required | what it does |
|---|---|---|---|
| `fit_window` | `[lo, hi]`, increasing | yes | the wavelength range, in air Angstrom, both branches are fitted over |
| `line_mask_iter` | non-empty list of integers ≥ 1 | yes | which of step 3's sky-line mask iterations is **excluded** from the fit. Counts from 1. Naming more than one produces one step 4 run per entry, in `step04/mask_iter<N>/`; naming one keeps step 4's products flat. Step 5 uses the last entry. |
| `fix_s_at` | number or `null` | yes | the sky continuum amplitude held fixed while classifying. `null` leaves s a free parameter of the fit. Fixing it at 0 is what suits spectra whose sky has already been removed, which is what step 4 classifies. |
| `z_min`, `z_max`, `z_step` | numbers | yes | the redshift grid the galaxy branch scans |
| `star_dz` | number | yes | half-width of the redshift scan for stars, which are searched around 0 rather than over the whole grid |
| `num_workers` | integer | yes | processes for the scan. **0 means one third of the visible CPUs** — a conservative default for a shared machine. On a machine of your own, raise it. |
| `keep_scans` | bool | no, default **true** | whether the whole chi² surface is written (`step04/scans_star.npz`, `scans_galaxy.npz`) as well as the winning row in `source_fits.npz`. The scans are much the larger part of what step 4 writes, and the only program that opens one is `evaluation/source_fit.py --which scan`, which draws a single source's curve. Turn it off for a run whose scans you will not look at; nothing else changes. |

### `sky_amplitude` — smoothing the sky brightness across the field (step 5)

s is solved freely per blank spaxel, then forced onto a smooth field so it can be
extrapolated over the source, where it cannot be measured.

| key | type | required | what it does |
|---|---|---|---|
| `min_source_distance` | number ≥ 0 | yes | a training spaxel must be at least this far, in pixels, from any detected source |
| `min_main_source_distance` | number ≥ 0 | yes | an extra exclusion radius around the **main source group**, the target galaxy, whose faint outskirts can reach past what the segmentation caught. `0` disables it. |
| `main_source_dz` | number ≥ 0 | yes | how close in redshift two segmentation IDs must be to count as the same object, which is what assembles the main source group |
| `train_clip_sigma` | number > 0 | yes | reject training points beyond this many times the robust scatter |
| `n_iter` | integer ≥ 1 | yes | alternating median passes ("median polish") that build the field. The alternation converges — the row and column offsets stop moving — so this only has to be past that point, and past is free at a few milliseconds an iteration. **Too few is the failure that matters**, because the field is then still drifting, and it drifts most where it is extrapolating over the source, which is the part the field exists to supply. |

### `spaxel_fit` — applying the model to every spaxel (steps 5 and 6)

| key | type | required | what it does |
|---|---|---|---|
| `blank_channels` | `all` or `line1` | yes | which channels solve a blank spaxel's coefficients. `all` uses every channel; `line1` uses **only** the channels step 3's *first* line-mask iteration flagged as sky lines — the line coefficients being best constrained where the lines are. |
| `min_channel_coverage` | number in [0, 1] | yes | the fraction of channels that must hold finite data before a spaxel is fitted at all. Below it the spaxel is left alone, which is what keeps half-covered edge spaxels out. |

---

## Which step reads what

Change a key and everything from its step onward has to run again. The pipeline
always runs all six, so this is really a table of what a change can possibly move.

| section | first step affected | also read by |
|---|---|---|
| `input.nosky`, `max_grid_offset` | step 1 | step 2. Step 4 records its path but classifies step 2's spectra, not the cube |
| `input.seg` | step 1 | steps 2, 3, 5, 6 |
| `input.cube` | step 3 | steps 5 and 6, which both re-open it to solve spaxels |
| `sky_region` (`apply_to: basis`) | step 3 | — |
| `sky_line_basis.*` | step 3 | — |
| `sky_line_basis.K`, `.method` | step 3 | **steps 4, 5 and 6**, which use them to name the basis file they read |
| `source_fit.*` | step 4 | step 5 (`line_mask_iter`, last entry) |
| `sky_region` (`apply_to: sky_amplitude`) | step 5 | — |
| `sky_amplitude.*` | step 5 | — |
| `spaxel_fit.*` | step 5 | step 6 |

---

## Conflicts and precedence

**The three optional keys touch different things, and they compose.**

| key | what it changes in step 3 | what it leaves alone |
|---|---|---|
| `mask_source_lines` | the decomposition input only, so only the **basis** | the mean spectrum, `C_sky` and the line masks all still see every channel |
| `borrow_from` | the **basis** only, taken from elsewhere | the mean spectrum, `C_sky` and the line masks are still learned here |
| `select_faintest` | **every product of step 3** — mean spectrum, `C_sky`, line masks and basis | steps 5 and 6, whose spaxels come from the segmentation and `min_source_distance`, never from what step 3 learned on |

Because `select_faintest` moves `C_sky` and the line masks, it also moves step 4's
fitting channels. The other two do not.

**`borrow_from` wins over the local decomposition.** When it is set, `method` and `K`
stop describing a decomposition performed here and become the name of the file to
read. `seed`, `continuum_window`, `line_thresholds`, `max_iter`, `clip_sigma` and
`min_unmasked_frac` are still used, because the continuum and the masks are still
learned locally — only the basis is not.

**`mask_source_lines` is applied last**, after the sigma-clip replacement, so what it
blanks stays blanked.

**Spaxel selection is a chain, and every link narrows.** `select_faintest` does not
replace the others; it chooses among what they leave:

    valid field (white light != 0)
      -> minus detected sources (seg > 0)
      -> minus / restricted to sky_region, if apply_to names basis
      -> minus spectrally incomplete spaxels
      -> then, if set, select_faintest's flux window

**`sky_region.include` picks the code path.** `true` becomes an `xlim` / `ylim`
restriction, and an axis written `[null, null]` is dropped entirely. `false` becomes a
single `exclude_box`. They are not two ways of writing one thing — see the endpoint
trap below.

---

## Traps

**1. A malformed config fails at once, but a wrong value does not.** Every required
key is checked by `load()` before step 1 runs, so a missing or mistyped one costs a
second and writes nothing. What no check can catch is a value that is well-formed and
physically wrong, which is the failure worth worrying about — so copy an existing
config rather than writing one from scratch.

`source_fit.fix_s_at` is the one key where absent and `null` mean different things:
`null` is legal and leaves s free in the fit, while leaving the key out is an error.

**2. `include: false` endpoints are inclusive; `include: true` endpoints are not.**
Both are written the same way in the config, and both are documented as half-open, but
the exclusion path converts to an inclusive box internally. `x: [10, 30]` with
`include: false` excludes columns 10 to **29**. The written form is consistent — this
only matters if you are comparing a config against `step03/meta.json`, which records
the converted `exclude_box` as `[y0, y1, x0, x1]` with both ends inclusive.

**3. `K` and `method` name a file that four steps open.** Steps 4, 5 and 6 build
`sky_line_basis_{method}_K{K}.npy` from these two values and read whatever is there.
They are one setting shared by four steps, not a step-3 setting.

**4. `apply_to: []` is legal and silently applies the box to nothing.** So is naming
only `sky_amplitude` while leaving a box written for the basis.

**5. `keep_intermediate: false` leaves the run unable to be inspected.** Every
`evaluation/` script and every `standalone/` step reads products from `step01`–`step05`.
Step 6's cubes are still written, but almost nothing can be said about them afterwards.

**6. `pointing` is a label.** It selects nothing. Only `input` decides which data is
read, and only `output` decides where results land.

---

## The three optional keys

### `sky_line_basis.borrow_from`

Names another run's output directory. Step 3 then takes the K line-basis vectors from
that run instead of learning them here:

    sky_line_basis:
      borrow_from: results/skymodel/p11

Every pointing has its own wavelength zero point, and the offsets between them are not
whole channels, so the vectors are resampled onto this pointing's grid with a cubic
spline — a sky line being about two channels wide, linear interpolation would flatten
it — and re-orthonormalised afterwards. A pointing reaching outside the range the
basis was learned on is **refused rather than extrapolated**, a basis being a set of
samples and not a formula. Pointings do not all cover the same range, so a pair that
can borrow one way cannot always borrow the other.

`step03/meta.json` records a `borrowed_basis` key naming the run, the file, its md5,
the two wavelength ranges, the channel offset and the orthonormality before and after,
so a run that borrowed is recognisable from its own products.

**What to expect.** A borrowed basis describes the sky of the night it was learned
on. Airglow changes over minutes to hours, so a basis from a pointing observed close in
time fits better than one from a different observing run, and neither fits as well as a
basis the pointing learns for itself. The key is there to ask what a shared basis
costs, not to skip step 3.

### `sky_line_basis.mask_source_lines`

The source's own emission lines, kept out of the decomposition input:

    sky_line_basis:
      mask_source_lines:
        - [4959.5, 4968]        # A, H-beta
        - [5057,   5069]        # A, [O III] 4959
        - [5106,   5118]        # A, [O III] 5007
        - [6694,   6709]        # A, H-alpha

Where the source fills the field, every blank spaxel still carries its emission lines;
the basis learns them along with the sky, and step 6 then subtracts them wherever it
applies the model, the source included. The channels inside each window are set to 0
in the decomposition input, and a basis with no structure at a wavelength cannot
subtract anything there.

The windows are **observed** wavelengths in Angstrom, `[low, high]`, closed at both
ends. Nothing here is redshifted: a window is a stretch of the grid, and the redshift
that put the line there has already been applied by whoever wrote the numbers. A
mapping of rest wavelengths, a redshift and a half width is no longer accepted.

**A window protects the sky inside it too.** Within one channel there is no telling
source light from sky, so whatever airglow falls in a window stays in the cube along
with the line it was written for. A window is therefore a trade and not a free gain,
which is why one should be as narrow as the line rather than as wide as its
neighbourhood. Where an airglow line sits on top of a source line, no choice of window
separates them.

`step03/meta.json` records a `masked_source_lines` key (past tense) listing every
window, its bounds and its channel count. The count can differ by a channel between
pointings, their grids being offset by less than a channel.

### `sky_line_basis.select_faintest`

Narrows the spaxels step 3 learns from to a flux window instead of taking every blank
spaxel:

    sky_line_basis:
      select_faintest:
        ignore: 0.05                    # fraction of the field thrown away, faintest first
        fraction: 0.10                  # fraction taken as sky, immediately above it

Both are fractions in [0, 1]; `fraction` must be above 0, and `ignore + fraction` may
not exceed 1. This is the ESO pipeline's own rule, the two keys mirroring its
`skymodel_ignore` and `skymodel_fraction`. ESO uses no segmentation at all — it ranks
the spaxels of the field by flux, throws away the faintest `ignore` of them (the dead
and the half-covered, which are not sky), and learns the sky from the next `fraction`.
What that also rejects, and a segmentation does not, is low-surface-brightness light
no detection ever found.

**The percentiles are taken over the whole valid field**, which is what ESO ranks, and
not over the blank set. The two are different questions and select different spaxels,
so the same fraction read against the blank set would not mean the same thing. Ranking
over the field is also what makes the cut values comparable to ESO's own.

The flux is the mean over wavelength of the sky-included cube, accumulated in the pass
that already reads the blank spaxels, so ranking the field costs no second read. A
spectrally incomplete spaxel has no place in the ranking — its mean is over a
different part of the spectrum than everyone else's. The window is half-open,
`(low, high]`.

`step03/meta.json` records a `selected_faintest` key with the rule, the two flux cut
values, how many spaxels were ranked, how many the window holds in the field, and
`n_selected`. `n_blank_complete` beside it stays the count *before* the window, which
is what makes the size of the cut readable.

---

## Where the results go

`output` takes the same three path forms as `input`. The pipeline reads nothing else
from the repository, so a config naming external inputs and an external `output`
writes nothing back into the checkout.

Figures follow the run. Every `evaluation/` script takes `--work`, the run's output
directory, and writes into an `evaluation` directory beside it:

    --work results/skymodel/p01          ->  results/skymodel/evaluation/p01/...
    --work /mnt/runs/p01                 ->  /mnt/runs/evaluation/p01/...

The few figures that compare several pointings belong to no single run and go to
`results/skymodel/evaluation`. `SKYMODEL_EVAL` moves those, which is what a read-only
checkout needs:

    SKYMODEL_EVAL=/mnt/runs/evaluation python src/skymodel/evaluation/pointing_curves.py --curve halo

`{output}/config.json` records the config the run was given, as `load()` returned it —
not a copy of the file, which can be edited afterwards. Steps 1, 3, 4, 5 and 6 stamp
the git commit into their `meta.json`, and each `stepN.log` opens with the call that
produced the products beside it.
