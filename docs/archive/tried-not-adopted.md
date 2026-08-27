> **Archived on 2026-08-27. A record of what was tried, not a description of the
> pipeline.**
> It names scripts that no longer exist -- that is the point of it, since the questions
> they asked are worth not reinventing -- and it points into `step3_sky_basis.py`, which
> was replaced by `src/skymodel/pipeline.py`, with line numbers that no longer resolve.
> `experiments/` has since been trimmed further, from the 24 scripts stated here to 20.

# Tried and not adopted (written up 2026-08-18)

These are the experiments that were deleted when `src/skymodel/experiments/` was trimmed
from 60 scripts down to 24. What is kept here is **what question each one asked and how
it was designed to answer that question**, so that the same subject does not get
reinvented from scratch.

**First, be clear about what this document cannot do.** Those scripts' docstrings set out
the question and the method in full, but **they record no results** -- the numbers were
only ever printed to the terminal, and most of the figures lived in the `evaluation/`
subdirectory, which has since been deleted. So every entry below can tell you what was
asked and what the criterion was, but unless it says otherwise it **cannot tell you what
the answer turned out to be**. If one of them is to be redone, the code is in git:

```
git show 665d7a9:src/skymodel/experiments/<name>.py
```

Conclusions that already made it into the pipeline are not here. They are in
`docs/rejected-approaches.md` and in the comments of the individual steps.

---

## 1 How the sky basis is learned

Current practice: `step3_sky_basis.py` runs an SVD on the **unweighted** residuals, one
set of K = 30 vectors covering the whole wavelength range and shared by the whole field.
The four scripts below each challenge one of those choices.

### Whitened SVD (`whitened_basis.py`)

An unweighted SVD picks out the directions that explain the most variance in **flux**,
which is why the few brightest sky lines swallow most of the basis. After whitening it
picks out the directions that explain the most **significance** instead. **This is what
ZAP does** (the "variance normalisation" of `docs/zap-parameters-reference.md` §4).

The three variants differ only in how the channels are scaled before the SVD: no
whitening at all (current practice), scaling by STAT's σ, or scaling by the residuals'
own standard deviation across spaxels (which is the same as running PCA on the
correlation matrix rather than the covariance matrix).

The design deliberately whitens only in the step that learns the basis, and **leaves the
fit unweighted throughout** -- keeping the two apart is what tells you which of the two
steps a difference came from.

### Noise-corrected PCA (`noise_corrected_pca.py`)

This is not the same thing as whitening:

```
  whitening         eigendecompose D^(-1/2) M D^(-1/2)   rescales the metric, and distorts the signal with it
  noise correction  eigendecompose M − alpha·diag(v)     subtracts the noise floor only, leaves the signal directions alone
```

`alpha` is "the fraction of the true noise that STAT accounts for". Rather than being
pinned at 1 it is scanned, and held-out data decides it; `alpha = 0` is current practice.
The number of degrees of freedom is unchanged (still K vectors), so there is no risk of
eating the source.

### Learning the basis in wavelength segments (`segmented_basis.py`)

ZAP cuts the spectrum into 11 segments and runs a separate SVD in each, with the
boundaries following the families of OH vibrational bands. A fair comparison has to
**hold the total number of parameters fixed**: K vectors over the whole wavelength range
against K/11 vectors in each segment gives the same degrees of freedom either way, so the
only thing that differs is whether the basis is allowed to couple across segments.

### Learning the basis in spatial regions (`regional_basis.py`)

The question is whether the blank residuals contain spatial structure that is not shared
by the whole field -- MUSE's slices are laid out along a particular direction, and
slice-level systematics produce exactly that kind of pattern.

**All three of these are validated out of sample**: the spaxels are split at random into
train and test, the basis is learned on train alone, and the score is computed on test
alone. The reason is written down in `regional_basis.py`: dividing the field into regions
is bound to shrink the residuals of the very spaxels the basis was learned from, and that
is nothing but extra degrees of freedom. This part of the design is worth keeping.

---

## 2 How the blank region is fitted

Current practice: `fit_blank` and `fit_source` both minimise the **unweighted squared
error**, and there is no other option.

For a while the two regions were treated differently: the source region was always
chi2-weighted, blank defaulted to unweighted with `--blank-chi2` to switch it over, and
the reason for choosing unweighted was speed (the design matrix does not change from
spaxel to spaxel, so `pinv` is computed once). On 2026-08-18, after comparing weighted
against unweighted on p01/p02/p03 -- measuring the step in the sky model across the mask
boundary, the sky-line residual in the source region, and the source's total continuum
flux -- both regions were changed to unweighted and the weighted code path was removed
altogether.

The reason for removing it is no longer speed alone: the chi2 weight is 1/σ, and σ is
particularly large in the bright sky-line channels, so weighting amounts to systematically
suppressing exactly those channels -- and almost all of the sky-line basis's energy is
concentrated there.

### Two-stage fitting (`blank_two_stage.py`)

The two components of the sky are each measurable in their own region, and current
practice solves for both of them mixed together:

```
  outside line1  continuum dominates, sky-line basis nearly indistinguishable (design matrix ill-conditioned there)  -> best place to measure s
  inside line1   sky lines dominate, almost all of the basis's energy is concentrated here                           -> best place to measure cₖ
```

Two-stage means using both regions but solving each one separately: outside the lines it
solves `d ≈ s·C_sky` (1 parameter, extremely well-conditioned), inside them it solves
`d − s·C_sky ≈ Σcₖ·Lₖ`. Neither design matrix changes from spaxel to spaxel, so the speed
advantage is still there.

### A chi2-style metric (`chi2_metric.py`)

**What this one records is a methodological trap, and it is worth remembering on its
own**: every earlier comparison used unweighted quantities (rms, channel-by-channel
scatter), but the objective function of an unweighted fit is precisely "minimise the
unweighted squared error" -- take that same thing as your metric and of course it wins.
That cannot be used to show that a weighted fit is worse.

`chi = R/σ` asks a different question: how large the residual is relative to the noise in
that channel -- and that is the quantity which decides whether a faint source can be
detected in that channel at all.

**The comparison made on 2026-08-18 steps around this trap**: none of the three
quantities used there is the objective function of either side -- the step in the sky
model across the two sides of the mask boundary (the real sky is continuous there, so the
whole step is produced by the fit), the same ratio of sky-line residual in the source
region against the blank side, and the source's total flux in a window free of sky lines.
The weighted code path has not existed since then, so the comparison this section
describes can no longer be re-run directly.

### Spatial smoothing of the coefficient maps (`blank_smooth_coef.py`, `coef_spatial_stats.py`)

`coef_spatial_stats` first measures two things in order to decide whether smoothing is
worth doing at all: the difference between two neighbouring cells, which gives the noise
floor, and the autocorrelation length, which gives the upper bound on the smoothing
kernel, both measured separately in x (along the slice) and y (across the slice).

The leave-one-cell-out validation in `blank_smooth_coef` has an explicit yardstick:
exclude the centre cell, predict it from its neighbours alone, and if the truth is
locally close to constant the error should be

```
  err ≈ σ_n · sqrt(1 + 1/k) ≈ 1.06 · σ_n     (k ≈ 7 effective neighbours)
```

An error close to 1.06 means the neighbours predicted everything except the noise, so
smoothing is safe; an error well above it means there is real structure the neighbours
cannot predict, and smoothing would wipe it out.

It also records a trap in the criterion itself: **the blank residual cannot be used as
the criterion** -- not smoothing is by definition the solution that makes that residual
smallest, any smoothing at all necessarily makes it larger, and so that criterion's
conclusion is bound to be "smoothing is harmful" and carries no information whatsoever.
Metrics against the ESO nosky cube were used instead.

---

## 3 A non-linear sky model (`nonlinear_sky.py`)

The intensities in the current model are linear. This script tried putting the wavelength
shift and the LSF width in as well:

```
  S(p, λ) = Σᵢ Aᵢ(p) · Lᵢ(λ − δ(p) ; w(p))
```

**It records one design decision that cannot be avoided**: the basis has to be brought
into the iteration and relearned along with everything else. If you take a basis learned
from data that was never aligned and then add δ on top of it, then supposing the shift
really is there, that basis will long since have absorbed it into extra components
(describing "some spaxels sit to the left, some to the right"), and adding δ after the
fact is bound to find nothing -- what it is looking for is already in the basis.

---

## 4 What is left in the error

### How the in-line error is distributed (`channel_error.py`)

Outside the lines the residual sits on the noise floor; inside the lines is where the
difference from ESO is. The question asked is whether the error that remains inside the
lines is spread evenly over all the in-line channels -- in which case it is noise and
there is nothing to be done about it -- or concentrated in a few lines, in which case it
is something that can be acted on.

The decomposition does not rely on STAT: `mean(c)` is the coherent error (noise averages
away, so a value significantly different from zero can only be systematic model error)
and `std(c)` is the noise plus the spaxel-to-spaxel variation.

### The shape of the lines (`line_shape.py`)

This tests the claim that "a linear intensity basis cannot express changes in the shape
of a line". Within the window of each bright line it takes that line's mean profile and
its derivatives as the basis: `P` is an intensity error, `P'` is a wavelength shift (a
first-order expansion, which looks like a double peak, one positive and one negative),
`P''` is a change in LSF width, and the constant is leftover local continuum. QR
orthogonalisation first, then the projection.

### The step at the mask boundary (`boundary_step.py`)

**This one's conclusion is written in the docstring, and it is usable**:

> **Inside** the mask, the source's light is taken up by the template and never enters
> sky_model, so it survives; **outside** the mask, the entire fit result is subtracted as
> sky. Real source light is continuous across the boundary, but the treatment of it jumps
> at the boundary, so a step in the residual is inevitable. **Dilation cannot remove it,
> only push it out to where the source is faint enough** -- the height of the step is
> always equal to how much source light is left at that radius.

Which means that "how far should the dilation go" is not a question of taste but a
question of how far out it takes for the source to fall below the noise, and that is
something that can be measured.

---

## 5 The ones that landed

The conclusions below are **already in the pipeline**. Deleting the code does not
threaten the conclusions themselves, but the scan numbers behind "why this value" were
not kept.

| experiment | what it became |
|---|---|
| `clip_threshold.py` | `CLIP_SIGMA = 30` in `step3_sky_basis.py` |
| `outlier_reject.py` / `outlier_fill.py` / `outlier_chi2_jump.py` | rejecting outliers before the basis is learned (`step3_sky_basis.py:246-260`) |
| `mean_estimator.py` | `mean_sky` uses a **sigma-clipped mean** (`step3_sky_basis.py:195`) |
| `measure_seg_thresh.py` | see `docs/rejected-approaches.md`: no longer measuring the threshold ourselves, using the segmentation the professor delivered instead |

The docstring of `clip_threshold` records a point about the criterion that is worth
keeping: a threshold is not chosen on the grounds that "setting it higher is safer", but
by looking for a gap in the data -- how many σ the real spaxel-to-spaxel variation
extends to, and from how many σ the bad values begin. If the gap is wide enough, the
threshold gives the same result wherever inside it you put it, and **then it is not a
parameter that needs tuning at all**.

`mean_estimator` records what to look at when the estimator is changed: not "does
mean_sky look right", but whether the three things it decides downstream (`C_sky`, the σ
used for line detection, and `line_mask`) change.

---

## 6 A one-off answer

`sky_vs_source.py` answers the professor's question, "how much of that is sky continuum?",
by splitting one source's spectrum into three curves -- observed, C_sky, and
observed − C_sky -- and plotting them together. The motivation is that the figure step4
draws is the already sky-subtracted `observed − 1.0·C_sky`, and the height of that curve
is easily misread as "the source was this bright all along".
