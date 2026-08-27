> **Archived. This is not the current specification.**
> What this document defines is the five metrics for evaluating ZAP's sky subtraction.
> M1 was built; M2, M3 and M5 have a specification and no implementation; M4 was marked
> deferred from the start. The project's main line is sky reconstruction
> (`src/skymodel/`), not a comparison against ZAP -- come back to this document if that
> comparison is ever resumed.
>
> **Before resuming it, you have to know that the ZAP products it rests on are no longer
> on disk.** The four full-field runs named in section 3,
> `results/zap/cubes/{target}_maskfrom-{masksrc}/{zap,sky,var}.fits`, have been deleted.
> The masks (`results/zap/masks/`) and the input cubes are both still there, so a run can
> be redone with `src/zap/zap.py`: about 65 minutes and a peak of 43.7 GB per run. ZAP
> cubes for the NE pointing are a separate matter and are still on disk.
>
> **Three more things would have to be settled before any of this could be run again.**
> Section 3 multiplies the STAT noise floor by 1.5, and M2's pass line is built on that
> factor; the project now takes STAT at face value, with an ideal chi of 1, and treats
> any such correction as an open question rather than a settled one -- so that pass line
> would have to be decided before it means anything. M4 states in bold that it does not
> use STAT at all, for the same reason M2 scales it, and the document never reconciles
> the two. And the code it names has moved on: `eval_common.py` no longer exists (the
> nearest surviving code is `src/zap/eval_spectrum.py`), and none of
> `HALPHA_LINE_WINDOW`, `HALPHA_FLUX_CONTINUUM` or `halpha_narrowband_image` -- which
> M3's equivalent width and M5's narrow-band image both depend on -- exists in the tree.

# Metric Spec — evaluation metrics for ZAP sky subtraction (Haro11 / MUSE)

This file is the **specification** of the evaluation metrics. Each metric is written down
in its final form as soon as it is settled; the file keeps no drafts, no before-and-after
comparisons and no shortlists of options.

---

## 1. Evaluation principle

Evaluating a sky subtraction has to prove two things at once, and neither one on its own
will do:

- **Removal**: the sky has been cleanly subtracted.
- **Preservation**: the source has not been eaten away by over-subtraction.

Looking at the residual alone is not enough to decide: over-subtraction flattens the
residual **and** cuts into the source signal at the same time, so every removal metric
has to be shown beside the preservation metric that corresponds to it.

---

## 2. References

| Key | Reference | arXiv |
|---|---|---|
| ZAP | Soto et al. 2016, MNRAS 458, 3210 | 1602.08037 |
| W20 | Weilbacher et al. 2020, A&A 641, A28 (MUSE pipeline) | 2006.08638 |
| Wis16 | Wisotzki et al. 2016, A&A 587, A98 | 1509.05143 |
| Lec17 | Leclercq et al. 2017, A&A 608, A8 | 1710.10271 |
| WH05 | Wild & Hewett 2005, MNRAS | astro-ph/0501460 |
| SP10 | Sharp & Parkinson 2010 | 1007.0648 |

---

## 3. Data and the baseline to compare against

- **Raw cubes** (read-only): `data/Haro11_{nosky,wsky}.fits`, holding `DATA` + `STAT`
  (STAT = the per-voxel variance, as propagated by the MUSE pipeline).
- **ZAP products**: `results/zap/cubes/{target}_maskfrom-{masksrc}/{zap,sky,var}.fits`
  (the STAT of `zap` is copied over from the raw cube unchanged, not recomputed).
- **The baseline to compare against (the truth)**: `nosky` raw, which is the result of
  the official MUSE sky subtraction.
- **Sky lines** (used for removal; they have nothing to do with the source):
  [OI] 5577.339 Å, [OI] 6300.304 Å, and the OH bands (the W20 zoom is on the OH 7-3 band
  at ~8760 Å).
- **Source lines** (used for preservation; observed frame, with Haro11 at z=0.0206):
  Hα 6698 Å, [NII] 6683 / 6719 Å.
- **STAT correction factor**: the MUSE STAT underestimates the true noise of an aperture
  measurement, measured/expected ≈ **1.5** (the range is 1.2–2.5), because resampling the
  cube correlates neighbouring pixels (Wis16 §3.2.5). Wherever √STAT is used as the noise
  floor, it is multiplied by 1.5.

---

## 4. Metrics

### M1 — Sky-subtraction residuals (the residual sky spectrum)

**Quantity**: the residual flux left in blank (source-free) spaxels after sky
subtraction, wavelength by wavelength, plotted against wavelength.

**Physical meaning**: once the sky is subtracted, a blank region should hold nothing but
noise and should approach 0. A residual that departs from 0 says the sky was not fully
removed; a residual that turns **negative** says it was over-subtracted.

**The definition of blank is valid & ~source**: take only the spaxels that are outside
the mask (that is, not a source) **and** inside the field of view (FoV). Outside the FoV
a spaxel is NaN in the raw cube, but ZAP fills it with a finite value (≈0), so it has to
be excluded by a validity test (nansum over all wavelengths of the raw cube ≠ 0);
otherwise about 28,000 edge spaxels pull the statistics towards 0. **Use every valid
blank spaxel, without sampling**, and collapse over the spaxels at each wavelength into a
single value → one residual spectrum, which is both more precise than sampling and
reproducible.

**Statistic**: the MUSE convention is the **median** (W20; robust against outliers). The
ZAP convention is **one figure each for the mean and the median** (the mean is the
statistic of Soto's paper, but over a large mask it is dominated by outlying spaxels and
spikes at the sky lines, while the median stays clean).

**Produce a figure in each of the two conventions**, both drawing the same quantity, each
overlaid only with the criteria of its own convention and not the other's:

#### Figure A — the MUSE convention (after W20 Fig 15)
- x axis: wavelength [Å], with zoom insets on 5577 and on the OH band (~8760 Å).
- y axis: residual flux [10⁻²⁰ erg s⁻¹ cm⁻² Å⁻¹], **linear, centred on 0, on a shared ±5
  scale**.
- Curves: `{target}+ZAP` (median) and `nosky` raw (the MUSE standard, for comparison).
- Criterion bands: envelopes at **±1% (black) / ±5% / ±10%** of the original sky (the
  mean of `wsky` raw).
- Pass: the continuum residual falls inside **±1%**, and away from the strong sky lines
  the residual falls inside **±5%** (the target is 2%).

#### Figure B — the ZAP convention (after ZAP Fig 1)
- Layout: **two panels** -- on the left `Standard processing (MUSE pipeline) = nosky raw`,
  on the right `ZAP = {target}+ZAP`. This is the "standard vs ZAP" comparison of Soto
  Fig 1.
- x axis: wavelength [Å].
- y axis: **twin axes** -- residual flux on the left axis, the original sky flux on the
  right (a grey line, `wsky` raw).
- **No error band**: neither ZAP Fig 1 nor W20 Fig 15 draws any noise or Poisson band;
  the grey line on the figure is **the original sky spectrum itself**, not an error
  envelope.
- Pass: the residual is far below the sky on the right axis, and it does not turn
  negative (no over-subtraction).

**Scale rules**: residuals are always **linear**, because a residual is signed and log or
symlog cannot show a negative value, which would hide over-subtraction. The y axis has a
**fixed range**, never autoscaled, so that the four runs can be compared against each
other directly: ±5 on the MUSE figure, ±40 on the ZAP-mean figure, ±8 on the ZAP-median
figure, and 0–1500 on the right-hand sky axis. Log is used only for drawing **the
original sky spectrum itself**, which is always positive and spans orders of magnitude.

**Computation** (`eval_common.py`, with the results cached in `results/zap/blankstats/`):
take the mean and the median over all valid blank spaxels, wavelength by wavelength.
- The residual spectrum is the `zap` cube; the standard, or truth, spectrum is `nosky`
  raw; the original sky spectrum is `wsky` raw -- that is, the sky of this field, which
  keeps the baseline of the % envelopes the same for every run rather than letting it
  degrade with the target.
- Validity is decided by nansum ≠ 0 over all wavelengths of the raw cube the mask came
  from.
- (The `(by,bx)` sample in `blanks.npz` is no longer used; `blanks.npz` now keeps only the
  bright-source coordinates `(sy,sx)`, for M3.)

**Naming** (the folder carries the run, the file name carries the metric and the
convention):
- `results/zap/cubes/{target}_maskfrom-{masksrc}/fig_M1_muse.png`
- `.../fig_M1_zap_mean.png` and `.../fig_M1_zap_median.png`

**Scope**: a figure of its own for each run; all four runs can be drawn.

---

### M2 — Noise spectrum

**Quantity**: the **scatter (robust rms)** of the residual in the blank spaxels,
wavelength by wavelength, against wavelength. It catches the failure that M1, being a
median and so an offset, cannot see: ZAP pouring noise in or drawing it out.

**Physical meaning**: subtracting the sky should not change the noise; the scatter of the
residual should sit at the statistical noise floor.
- rms ≈ floor → ideal, nothing but statistical noise is left.
- rms > floor → under-subtraction, with sky structure left behind.
- rms < floor, and especially so at the wavelengths of the strong sky lines →
  over-subtraction or denoising, with source signal drawn out (ZAP §5). This is a
  preservation warning, and only looking at the scatter catches it.

**Compared against**: `{target}` raw (before subtraction) vs `{target}+ZAP` (after).

Reproduces **WH05 Fig 4** (the only figure that plots rms against wavelength with a
before/after comparison; it is SDSS fibres, and the idea is carried over to MUSE blank
spaxels. Neither ZAP nor W20 has such a figure). **Two stacked panels**:

#### Upper panel
- x axis: wavelength [Å], the full MUSE range 4750–9350, linear.
- y axis: robust rms [10⁻²⁰ erg s⁻¹ cm⁻² Å⁻¹], linear.
- Curves: `{target}` raw and `{target}+ZAP`.

#### Lower panel
- x axis: as above.
- y axis: rms ÷ (√STAT × 1.5), dimensionless, linear.
- Pass: **≈ 1 and flat**. Above 1 is under-subtraction, with OH bumps left behind; below 1
  is over-subtraction, showing as troughs, or denoising.

**The definition of rms** (the robust one of WH05 §2.2.1): at each wavelength take the
**67th percentile** of `|flux − median|` over the blank spaxels (≈ 1σ, and robust against
outliers).

**Scale rules**: rms is always positive, so y is **linear**, following WH05.

**Computation**: over all valid blank spaxels (= valid & ~source, as in M1, using
`eval_common`).
- rms_raw(λ) = 67pct( |`{target}` raw − median| ) across blanks.
- rms_zap(λ) = 67pct( |`{target}+ZAP` − median| ) across blanks.
- noise floor(λ) = √( the per-wavelength median of the `{target}` STAT over the blanks )
  × 1.5.
- Lower panel = rms_raw / floor and rms_zap / floor.

**Naming** (as for M1):
- File: `results/zap/cubes/{target}_maskfrom-{masksrc}/fig_noise-spectrum.png`.
- Title, in two lines: `Noise spectrum` as the main line, and
  `target = {target} · mask from {masksrc}` as the subtitle, in small type.

**Scope**: a single run to begin with.

**Basis**: WH05 Fig 4 (astro-ph/0501460); the idea of a noise floor comes from ZAP Fig 1
and SP10; the STAT×1.5 factor is in §3.

### M3 — Source Hα fidelity (fidelity of the source's bright core)

**Quantity**: the **spectrum integrated over a 1″ aperture** on the source's bright core,
drawn before and after sky subtraction on the same axes, together with a measurement of
whether the strength of the Hα line (its EW) is preserved. This measures the **bright
core**; the extended, faint emission is M4's business.

**Physical meaning**: subtracting the sky should not eat the source's emission lines. The
spectrum of `{target}+ZAP` should lie on top of the truth, and the Hα EW should be ≈ the
truth. An EW clearly below the truth means the source has been over-subtracted.

**Compared against**: `nosky` raw (the MUSE truth) vs `{target}+ZAP` vs the original sky
(which only `wsky` has).

**Reproduces ZAP Fig 6** (the accepted figure for source fidelity; the original is a
gallery of 5 sources, and we have a single galaxy → a single aperture):
- Layout: one spectrum panel.
- x axis: wavelength [Å], **the full 4750–9350 range with no zoom**, linear.
- y axis: flux [10⁻²⁰ erg s⁻¹ cm⁻² Å⁻¹], linear.
- Aperture: a **circle 1″ in diameter** (radius 2.5 px), placed on the brightest pixel,
  (237, 315), which is Haro11's bright core.
- Curves: `nosky` raw, `{target}+ZAP`, and the original sky (`wsky` only).

**The quantitative part, following the EW invariance of WH05 Fig 12 + Table 1**: measure
the equivalent width of Hα and compare `{target}+ZAP` against the truth.
- EW(Hα) = Σ_line ( F − F_cont ) / F_cont · Δλ, with the line window 6692–6708 Å
  (`HALPHA_LINE_WINDOW`, Hα only, keeping clear of [NII]).
- The continuum F_cont: one window on each side, **both keeping clear of [NII]6548=6682.9
  and [NII]6583=6719.1**: **6655–6678** on the left, **6730–6758** on the right.
  (This corrects the old `HALPHA_FLUX_CONTINUUM`, whose left window 6660–6688 cut into
  [NII]6548.)
- Report **EW_zap / EW_truth** (≈ 1 means it was kept): annotated in a corner of the
  figure, and entered into the scalar summary table.

**Computation**: for the `zap` cube and for each raw cube, sum the flux of the spaxels
lying within 2.5 px of (237,315) → the aperture spectrum; measure EW(Hα) on each of them.

**Naming** (as for M1 and M2):
- File: `results/zap/cubes/{target}_maskfrom-{masksrc}/fig_source-halpha.png`.
- Title, in two lines: `Source Hα fidelity` as the main line, and
  `target = {target} · mask from {masksrc}` as the subtitle, in small type.

**Scope**: a single run to begin with.

**Basis**: ZAP Fig 6 (arXiv:1602.08037: a 1″ aperture, the full range, linear, with three
curves for pipeline, ZAP and sky); the quantitative part is WH05 Fig 12 + Table 1 (EW
invariance).

### M4 — Extended Hα surface-brightness profile (radial profile of the extended halo) [⏸ deferred · to be discussed later]

> **Status: deferred, not yet settled.** What follows is the record of an investigation
> that has been checked out (how Wisotzki Fig 4 is drawn, plus the literature on PSF
> parameters), to be used as it stands when the discussion resumes. How the PSF is to be
> obtained -- fitting a foreground star versus fixing β=2.8 -- and whether M4 is included
> at all are still undecided.

**Quantity**: the **azimuthally averaged Hα surface brightness** about the source,
against radius. It measures whether ZAP keeps the **faint extended halo** -- the place
where PCA sky subtraction is most prone to over-subtracting faint extended signal (M3
covers the bright core, M4 the extended emission).

**Physical meaning**: the SB(r) of `{target}+ZAP` should lie on the `nosky` truth, and at
large radius should still be above the PSF (so the emission is genuinely extended and not
the wings of a point source) and above the 1σ limit (so it is a genuine detection). A ZAP
SB(r) that drops below the truth means the extended halo has been over-subtracted.

**Compared against**: `nosky` raw (the truth), `{target}+ZAP`, the PSF profile, and the 1σ
detection limit.

**Reproduces Wisotzki 2016 Fig 4** (the azimuthally averaged radial SB profile; the
method does not depend on which line is used, and there is precedent for low-redshift Hα
CGM work in Dutta 2024 and Chung/Dey 2019):
- x axis: radius [arcsec], linear.
- y axis: Hα surface brightness [erg s⁻¹ cm⁻² arcsec⁻²], **log**, with negative values
  marked as triangles.
- Annuli: concentric, **0.2″ = 1 spaxel wide**, averaged azimuthally with masked and bad
  pixels excluded, centred on Haro11's core at (237,315).
- Curves: `nosky` raw, `{target}+ZAP`, the PSF profile, and the 1σ limit.

**The 1σ detection limit** (Wisotzki's empirical method): run the same annular extraction
at ~100 blank positions and take (Q3−Q1)/1.35 = σ_eff in each radius bin. **The STAT cube
is not used**, because the MUSE STAT underestimates correlated noise.

**PSF** (a circular Moffat):
- **β**: the field holds an isolated, unsaturated foreground star → fit it, with β free;
  otherwise **fix β=2.8**, the WFM-NOAO standard (B17/Leclercq).
- **FWHM(λ) = a + b·λ**, linear, and shrinking towards the red: fit it if there is a star,
  otherwise anchor it on the header QC seeing (`EXPCOMB FWHM MEDIAN`=1.24″ in `wsky`;
  ⚠️ in `nosky` that value is unpopulated, =0, and the wsky value ranges over 0.70–1.92″
  across exposures, so it serves only as an anchor and a single value is not applied
  across the whole range).
- **Sensitivity**: any conclusion about the extended halo versus the PSF has to be run as
  a sensitivity test over **β ∈ [2.5, 3.0]** and reported, because β controls the wings of
  the PSF and so bears directly on the "extended emission versus the wings of a point
  source" verdict.

> **PSF parameters (a record of the literature, for reference)**: for seeing-limited MUSE
> WFM the Moffat β clusters at **2.5–2.8** (B17 fixes it at 2.8, HDFS Bacon+2015 fits 2.6,
> seeing-limited work generally sits near 2.5; the AO case, Fusco+2020, fits 2.3–2.7). β
> is treated as constant with wavelength, while FWHM falls linearly with it (0.71″→0.57″
> in the UDF). B17's 2.8 itself comes from **seeing-limited WFM-NOAO** (the MUSE UDF,
> 2014–2016, before AO came online), which is the same regime as Haro11, so it can be used
> here. The standard way of getting the PSF: fit a Moffat if there is a star
> (PampelMuse/mpdaf), and if there is not, fix β=2.8 and fit only FWHM(λ) (B17).

**Computation**: build a continuum-subtracted Hα narrow-band image from the `zap` and
`nosky raw` cubes → average azimuthally about the core → SB(r); estimate 1σ the same way
over blank regions; build the Moffat profile for the PSF from the header or from a fit.

**Naming** (as for M1–M3):
- File: `results/zap/cubes/{target}_maskfrom-{masksrc}/fig_radial-halpha.png`.
- Title, in two lines: `Extended Hα surface-brightness profile` as the main line, and
  `target = {target} · mask from {masksrc}` as the subtitle, in small type.

**Scope**: a single run to begin with. It corresponds to fig7 of the existing CGM Hα
analysis, of which this is the formal specification.

**Basis**: Wisotzki 2016 Fig 4 (arXiv:1509.05143), Leclercq 2017 (1710.10271); for the
PSF, Bacon+2017 (1710.03002, β=2.8) and Bacon+2015 (1411.7667, β=2.6); for precedent on
Hα CGM, Dutta 2024 (2410.05392) and Chung/Dey 2019 (1904.07874).

### M5 — Source mask diagnostic

**Role**: this is a **diagnostic, and documentation** -- not a validation metric, and it
has **no pass line**. It sets out which spaxels are masked, and how much difference
building the mask from a different cube makes.

**Compared against**: `sep_from-nosky/mask.fits` (≈41%) vs `sep_from-wsky/mask.fits`
(≈36%), overlaid on the same image so the two can be compared.

**Physical meaning**: the mask is what decides whether ZAP succeeds or fails (mask too
little → the source is learned as sky, and 70% of it is lost). This shows how far the
mask reaches and how the two mask sources differ.

**Layout (one figure, two panels side by side, sharing a `nosky` background image)**:
- Left panel: an **Hα narrow-band image** (nosky) underneath -- this shows whether the
  mask follows the Hα emission, which is what the mask was detected on in the first place.
- Right panel: a **white-light image** (nosky, the nansum over the whole range) underneath
  -- this shows the mask against the stellar continuum light; the Hα halo reaches beyond
  the starlight, so the mask extends a ring further out.
- Both panels carry **two mask contours**: the `nosky`-built one in one colour and the
  `wsky`-built one in another. The background image is greyscale and the contours are in
  contrasting colours.

**Annotations**: the coverage of each mask (nosky 41% / wsky 36%), and the largest radius
it reaches.

**Naming** (it belongs to the masks, so it goes in masks/):
- File: `results/zap/masks/fig_source-mask.png`.
- Title, in a single line: `Source mask (sep) — built from nosky vs wsky`.

**Computation**: the Hα narrow band is the in-line flux − the continuum (using the shared
`halpha_narrowband_image` settings); the white light is the nansum over the whole range;
the contours are the 0.5 level of the two masks; both background images are taken from
`nosky`, which is the cleanest and can be shared fairly.

**Basis**: there is no figure in the literature to follow -- ZAP has no mask figure. This
exists purely to account for reproducibility; it is a diagnostic, and it carries no
removal or preservation criterion.

### Scalar summary table

**Purpose**: to condense the key numbers of M1–M3 into a single table for the terminal or
a report, so that "the sky came off cleanly and the source was not eaten" can be judged
without looking at the figures. **Removal and preservation stand side by side**
(principle 1).

**Layout**: the columns are the four runs (2×2, `{target}_maskfrom-{masksrc}`); the
baseline is `nosky` raw.

| Row | Definition | From | Pass |
|---|---|---|---|
| sky 5577 residual (% of sky) | median residual in the blank region ÷ the original sky @5577.339 Å | M1 | < 5% (target 2%) |
| sky 6300 residual (% of sky) | the same @6300.304 Å | M1 | < 5% |
| sky OH residual (% of sky) | the same @ the OH band (for the wavelength see M1; the current settings use 8400) | M1 | < 5% |
| continuum residual (% of sky) | the median over the line-free wavelengths | M1 | < 1% |
| residual RMS ÷ (√STAT×1.5) | the median over the line-free wavelengths | M2 | ≈ 1 |
| Hα EW retention | EW_zap ÷ EW_truth (1″ aperture, bright core) | M3 | ≈ 1 (100%) |

**M4 (the extended halo) is deferred** → the numbers that go with the radial profile (the
radius out to which the halo is detected, the SB retention at large radius, and so on)
**will be added once M4 is settled**.

**Reading it**: `wsky+ZAP` should pass on removal (the first 5 rows) and preserve ≈ 100%
(the last row) at the same time.
