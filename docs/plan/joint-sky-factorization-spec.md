# Joint Sky Factorization -- the method specification

> This file is the **authoritative method specification for Joint Sky Factorization**: the
> definition of the model, the objective function, the preprocessing, the solver, the
> procedure by which the hyper-parameters are decided, the mechanisms against
> over-subtraction, the validation protocol and the upgrade path. The implementation
> follows this file.
>
> **Nothing here is built yet.** None of the nine modules in section 9 exists, and the
> repository holds no PyTorch, no graph Laplacian and no NMF sky basis. What runs today
> is a different, deliberately simpler pipeline -- six steps, an SVD line basis, and an
> additive spatial field for the sky continuum amplitude (`src/skymodel/`,
> `docs/pipeline-products.md`). This file is what the method is meant to become, and it
> is written in the settled voice of a specification because that is what a
> specification is for; it is not a description of running code.
>
> Three of its premises have moved since it was written. It describes the single
> 499x559x3679 field rather than the 14-pointing mosaic the pipeline now runs; it
> assumes the mask is detected here, with a matched filter and dilation, whereas the
> segmentation is now an input the run is given; and the nine module names it reserves
> under `src/skymodel/` would land in a directory that is already occupied.
>
> The method belongs to the sky reconstruction family of `CLAUDE.md` Principle 1 (strategy
> one: predict the sky): a joint factorization of a low-rank NMF sky basis with a spatial
> model, and the main line of research contribution (no supervised DL).
> The data and metric framework is in `docs/plan/data-and-metrics-overview.md`; the metric
> definitions are in `docs/archive/metric_spec.md` (archived); the parameters of the ZAP
> comparison arm are in `docs/zap-parameters-reference.md`.
>
> **The method contains no neural network of any kind.** Every parameter that is fitted --
> the sky basis, the coefficient maps, the source term -- is itself a physical quantity,
> and every one of them can be plotted and inspected. The hyper-parameters are decided by
> held-out data or by a physical scale, without exception (Principle 2).

---

## 0. In one line, and where it sits

The whole cube is taken apart, in one go, into three parts:

```
observation(line spectrum) = sky(low rank × spatially smooth) + source(inside the mask only, sparse) + noise
L[p,λ]   ≈ Σₖ A[p,k]·B[k,λ]      + S[p,λ]
```

The three are **fitted jointly by one objective function**, and the source is an explicit
component of the model rather than dirty data to be thrown out.

How it stands structurally against the existing methods:

| | Where the basis comes from | The spatial assumption on the coefficients | What is done with the source |
|---|---|---|---|
| Super Sky | The mean of the blanks (1 component) | The same over the whole field (the strongest assumption) | Nothing |
| ZAP | PCA on the blank spaxels | Free per spaxel (no assumption) | Masked out of the basis; at the projection the source can leak into the coefficients → a risk of over-subtraction |
| **This method** | NMF on the blank spaxels (fixed once settled) | **A spatial smoothness prior** (a graph Laplacian) | **An explicit component S**, sparse and local |

This method's defence against over-subtraction is **structural**: for source signal to enter
the sky term it has to violate the shape of the basis (B holds no source-line feature), the
spatial smoothness (the coefficient map would have to swell into the shape of the galaxy)
and the cost accounting (S takes it on more cheaply) all at once -- rather than resting on a
fine adjustment of the mask size or the number of components.

---

## 1. The data and the notation

| Symbol | What it is | Its value in this project |
|---|---|---|
| `D[p,λ]` | The input cube (**wsky**, the sky not yet subtracted), flattened spatially | 499×559×3679; `ESO PRO DATANCOM = 3` (3 exposures combined) |
| `V[p,λ]` | The STAT extension (the per-voxel variance) | **Taken at face value** (σ=√V, an ideal χ=1). The ~1.8× underestimate correction recorded in the older document (`data-and-metrics-overview.md` §3.1) has not been confirmed by the professor; it is listed as an open question and is not used |
| `M` | The source mask (2D, source=1) | Produced by the current matched-filter procedure (kernel = the seeing, 4 px) plus 2σ plus dilation, masking ~44% |
| blank | The valid spaxels outside `M` | The sky samples; the mainstay that trains the basis and pins the coefficient field down |
| P, Nλ, K | The number of spaxels, of wavelengths and of basis vectors | P ≈ 2×10⁵; Nλ = 3679; K is chosen on held-out data, and is expected to be O(10) |

The input must be wsky. Running this method on a nosky whose sky is already subtracted is a
null test, since there is no sky left to learn from, which is the settled conclusion for ZAP
as well (`docs/zap-conclusions.md` §1).

The second dataset, `Haro11_NEpointing_wsky.fits` (3 exposures combined as well), is what
the method is re-run on independently to test its robustness.

---

## 2. Preprocessing

### 2.1 Throughput normalisation (with a diagnostic gate)

**The purpose**: the spatial smoothness of the coefficients is this method's central prior,
and MUSE is pieced together from 24 IFU × slicer units, so flat-field residual jumps at the
slice boundaries and breaks that prior outright. ZAP's free per-spaxel fit is immune to it;
this method has to handle it explicitly.

**What is actually available at cube level**: the slice labels exist only in the pixel
table and cannot be had in the final cube, which is 3 exposures resampled and combined, and
the combination has already smeared the striping in part. The calibration is therefore
**purely empirical and self-contained**, and does not rely on the slice geometry:

1. **The diagnostic gate (mandatory, and done first)**: build the 5577 Å [O I] amplitude
   map of the wsky -- a narrow window integrated at the line centre − a window of the same
   width integrated in the line-free region beside it, spaxel by spaxel. Airglow is physically
   uniform over a 1′ field, so any spatial structure on that map rising above the noise is
   throughput residual.
   **A flat map (no structure above the noise) → `t≡1`, and the whole of this section is skipped.**
2. **Building the t map**: take N strong sky lines (5577, 6300, and a number of strong OH
   lines at the red end), divide each line's amplitude map by its own whole-field median,
   and take the median over the lines spaxel by spaxel → `t[p]` (the noise ↓√N);
   then apply a 3×3 median filter, on a scale far smaller than that of the striping.
   **The greyness test**: compare the 5577 map against the red-end OH map; agreement → one t
   serves the whole spectrum, disagreement → t is built per wavelength band.
3. **Applying it**: before the fit, `L[p,λ] /= t[p]`; at subtraction, `t[p]·sky_model` is
   taken off the original data.
   The flux calibration is never touched, and t exists only inside the model.

`t[p]` is a property of that exposure group, that combination and that grid: **it is
measured for each cube on its own and is never carried across cubes** (the method calibrates
itself, and every wsky cube brings its own yardstick; cubes derived from the same exposure
group share one t). The diagnostic figures, from step 1 and from the re-check after it is
applied, go into the standing QC figure set.

### 2.2 Separating the continuum from the emission lines

Each spectrum is median filtered with a large window, carrying over ZAP's `cfwidth` concept
and its settled range, to give the continuum `C[p,λ]`;
`L = D − C` is the line spectrum, and the low-rank model acts on `L` alone.

The sky continuum -- moonlight and zodiacal light -- is handled separately: `C` is coarsely
binned along wavelength (~50 Å per bin), a low-order 2D surface is fitted to each bin over
the blank spaxels, extended smoothly into the source region and then subtracted; the
source's stellar continuum is what is left over in each bin, and it is kept untouched.
(The same "learn on the blanks and push it out smoothly" logic, with the basis replaced by a
smooth function along wavelength.)

### 2.3 The weights and the bad data

`w[p,λ] = 1/V[p,λ]`. **All bad data is thrown out with `w=0`, without exception**, and
nothing is interpolated in its place: NaN, the edge of the field, and (on AO data) the
5800–6000 Å sodium notch. The form of the objective function supports missing data
natively, which is a simplification of the implementation relative to ZAP, where NaNs have
to be interpolated.

### 2.4 The wavelength axis: one model over the whole spectrum

**The decision: one model over the whole spectrum (4750–9348 Å), with no segmentation.**
The reasons:

1. It matches the single-segment default of the vendored ZAP 2.1
   (`zap-parameters-reference.md` §4.1; CHANGELOG 2.0 records that several segments make
   the continuum oscillate at the boundaries and the per-segment component choice fragile),
   so the comparison between the methods is not confounded by a segmentation strategy.
2. NMF is non-negative and forbids cancellation, so the components localise on the sky-line
   families of their own accord: the block structure segmentation tries to impose by hand
   is learned from the data itself, and without a hard boundary; and the genuine
   correlation across line families, driven by the same geophysics, can be exploited.
3. This method's component count K is a single global value, chosen on the held-out error,
   so the fragile per-segment choice does not arise at all.

**Segmentation is a rescue measure and nothing more.** It is triggered when the residual
diagnostics show one sky-line family systematically failing to come off cleanly while the
others are fine, and raising K does not help. At that point 2–3 physical segments are tried,
and they may only be adopted once they have been validated against the two indicators of §7
(a scientific decision, following `CLAUDE.md`'s rule for SKYSEG).

---

## 3. The model and the objective function

### 3.1 The unknowns and the constraints

| Quantity | Shape | What it means | Constraint |
|---|---|---|---|
| `B` | K×Nλ | The sky spectral basis (≈ the shape of each sky-line family) | Non-negative; **settled by a blank-only NMF and then held fixed** (the first line of defence, §6) |
| `A` | P×K | The spatial amplitude map of each basis vector | Non-negative; spatially smooth |
| `S` | P×Nλ | The source line spectrum (v2 only) | Identically 0 outside `M`; sparse inside `M` |

### 3.2 The objective function

```
min  Σ_{p,λ} w[p,λ]·( L[p,λ] − (A·B)[p,λ] − S[p,λ] )²          ── ① data fidelity
   + λ_sp · Σₖ Σ_{(p,q)∈E} ( A[p,k] − A[q,k] )²                 ── ② spatial smoothness
   + λ_S  · Σ_{p∈M,λ} |S[p,λ]| / σ[p,λ]                          ── ③ source sparsity (v2)
s.t. A ≥ 0,  B ≥ 0 (fixed),  S[p,·] = 0 for p ∉ M
```

- **①** is the weighted sum of squared residuals. On a blank spaxel S≡0, and the ~10⁵ blank
  spectra are the mainstay that pins A and B down.
- **②**: E is the 4-neighbour graph of the spaxels, and the term is the discrete ∫|∇a|² of
  each coefficient map (the discrete form of a Gaussian MRF or GP prior). The physical
  grounds: every known source of spatial variation in the sky -- the moonlight gradient,
  airglow at ~90 km, atmospheric transmission -- is smooth on a large scale, and the one
  thing that violates it, the slicer throughput, has already been removed by §2.1.
- **③** is an L1 penalty, and its closed-form solution voxel by voxel is a **soft
  threshold**, `S = sign(r)·max(|r|−τ, 0)` with τ ∝ λ_S·σ. The effect is that a voxel whose
  residual falls below the threshold has S **exactly 0**, which is the sparsity; the
  threshold is in units of σ, so what λ_S means is how many σ it takes to count as signal.
  Note that the soft threshold pushes S systematically low, by −τ, so **S must not be used
  to measure the source's flux**; the final science product is always `D − t·sky_model`.

**Why the decomposition holds (the cost accounting)**: the sky is everywhere in the field
and its shape lies within the span of B, so explaining it with A·B is almost free; the
source is local and its shape is not in B, so for A·B to absorb it costs both ② (the
coefficient map swelling) and ① (B's shape not matching), while S pays only ③. Each of the
two signals takes the cheapest road open to it, and that is the separation.

### 3.3 v1 (the first release): no S, and the source line windows thrown out

Haro11's redshift is known (z ≈ 0.0206), so the observed wavelength of every source emission
line is entirely predictable. v1 introduces no S variable:

```
For a spaxel p ∈ M, the voxels inside a source line window are simply set to w = 0; the model degenerates to L ≈ A·B (① + ② only).
```

The line windows are generated programmatically from the known redshift (the line centre ±
a half-width, the half-width taken as a conservative value of the line width plus the LSF).
The main lines, at their observed wavelengths, are:
Hβ 4961, [O III] 5061/5110, He I 5997, [O I] 6430, [N II] 6683/6719, Hα 6698,
[S II] 6854/6870, and the Ca II triplet absorption band 8675–8845 (a feature of the source
continuum, thrown out along with them).
The remaining wavelengths inside `M` -- the great majority of channels, which hold no source
line -- constrain the local sky coefficients as usual.
The fidelity term inside `M` uses a Huber loss, which cheaply absorbs an unexpected source
feature outside the line list.

### 3.4 v2 (the upgrade): S added

**What triggers the upgrade**: the injection test of §7 showing that v1 carries a systematic
offset at the source's position, because source features outside the line windows -- broad
line wings, continuum residual -- contaminate A. v2 lays ③ and the variable block S on top
of the same loss, and the code adds to it strictly. The two extremes of λ_S each degenerate
into a known method (λ_S→∞ forces the whole spectrum of a source spaxel to be explained by
the sky, which is the ZAP-style risk; λ_S→0 leaves the data inside M out of it entirely, and
degenerates into pure spatial interpolation), and an intermediate value is what makes the
source-free channels contribute sky information while the channels that do hold a source are
taken on by S.

---

## 4. Solving it

### 4.1 Initialisation (a complete conventional method in itself, and the baseline besides)

1. Run `sklearn.decomposition.NMF` on the `L` of the blank spaxels → `B⁰`, fixed from then on.
2. Solve a weighted NNLS against `B⁰` for every spaxel → `A⁰` (a spaxel in the source region
   uses only the wavelengths outside the line windows).
3. `S⁰ = 0`.

This solution, written **NMF-only**, goes into the comparison set of §7.

### 4.2 The optimisation

**The main implementation (PyTorch / Adam, on the GPU)**: `A = softplus(θ_A)` and
`S = mask_M · θ_S`, with the whole objective function written straight out as the loss
(~15 lines) and solved by autodiff. Changing the model -- adding a term, swapping a penalty
-- means changing the loss and nothing else.

**The reference implementation (block coordinate descent)**: the three blocks take turns,
and each is a convex subproblem -- A is a sparse linear system followed by a projection onto
non-negativity; S is a soft threshold voxel by voxel, in closed form; and B, if it is
experimentally let loose, is a weighted NNLS at each λ.
It is there to check the main implementation and to debug it.

### 4.3 Convergence and what it costs to compute

- The convergence criterion is a relative change in the objective function of < 10⁻⁶, or no
  improvement in the held-out blank error over 20 consecutive iterations.
- Memory: `L`, `w` and the residual are P×Nλ×float32 ≈ 3 GB each, with a GPU peak of ~12 GB,
  below the 43.7 GB of RAM measured for ZAP over the whole field.
- Time: a few hundred–10³ Adam iterations, expected to take minutes on a GPU; the NE
  pointing re-run is of the same order.

---

## 5. The hyper-parameters (the full table, and how each is decided)

**The held-out blank block protocol** is the yardstick every hyper-parameter shares: a
number of **contiguous spatial blocks** are cut out of the blank region (of order 20×20 px,
~5% of the blank in total) and take no part in any fit whatsoever; the model's weighted
reconstruction RMSE(λ) is measured on those blocks. The split has to be made by block --
splitting point by point at random leaks, because the noise of neighbouring spaxels is
correlated, and it makes the error look smaller than it is
(`data-and-metrics-overview.md` §2.4).

| Parameter | What it does | How it is decided (the Principle 2 defence) |
|---|---|---|
| K | The sky's degrees of freedom | The elbow of the held-out RMSE, stopping where it no longer falls. Choosing it by the flattest residual is **forbidden** -- over-subtraction flattens the residual just as well (the Principle 1 trap) |
| λ_sp | The effective correlation length of the coefficient maps | Chosen on the held-out RMSE, then checked that the effective length falls in the physical range: longer than the ~tens of px scale of the slicer striping, shorter than the ~field scale of the moonlight gradient |
| λ_S (v2) | The threshold at which something counts as source | Set directly at 1–2σ in units of the corrected σ, on the same logic as the mask procedure's 2σ threshold |
| cfwidth | The line/continuum divide | ZAP's settled range, carried over (`zap-parameters-reference.md` §4.2) |
| The source line window half-width (v1) | How much is thrown out | The known redshift plus the measured line width plus the LSF, taken conservatively; not something to be tuned |
| M | The range S is allowed | The current matched-filter procedure. **Too large rather than too small**: in this method M only delimits the domain of S, and it is the blank constraint that does the tightening; enlarging M costs no more than a few S=0 hard constraints (an asymmetry with ZAP, where masking too much starves the basis -- the sensitivity to the mask is expected to be flat, and the difference itself goes into the §7 report) |

Any value that departs from the procedure in the table above is a scientific decision, and
under `CLAUDE.md`'s rule it is validated first and adopted afterwards.

---

## 6. The mechanisms against over-subtraction (four lines of defence), and the standing diagnostics

1. **B is settled on the blanks**: the basis structurally holds no source line (features
   such as Hα 6698 at z=0.0206 are not present in a blank spectrum), so the precondition for
   the sky term absorbing the source -- the basis having power at the source lines'
   wavelengths -- does not hold. For an extended, smooth CGM halo, which is where the third
   line of defence is weakest, this is the main defence.
2. **The source line windows are thrown out** (standing from v1 onwards): the coefficients
   of a spaxel in the source region are decided by the source-free channels alone. It rests
   on the known redshift, and costs nothing to tune.
3. **Spatial smoothness**: for a coefficient map to absorb the source it would have to swell
   into the shape of the galaxy, which ② resists directly. Against a large-scale CGM this
   defence is the weaker one, which is why 1 and 2 are the mainstays.
4. **The standing diagnostic figures**, produced after every run:
   (a) the spectrum of the reconstructed sky at the source's position, with the source line
   windows overlaid -- the sky model may hold no feature at a source line's wavelength;
   (b) each coefficient map `A[·,k]` -- no galaxy morphology may appear in it;
   (c) the throughput diagnostic figure (§2.1);
   (d) the held-out block residual vs λ.

---

## 7. The validation protocol

The comparison set: `super_sky` (the Phase 1 baseline), `NMF-only` (the initial solution of
§4.1), `ZAP` (the current 0706 pipeline) and `nosky` (the MUSE pipeline's model method).
The metric definitions are carried over from `docs/archive/metric_spec.md` (archived).

1. **The held-out blank RMSE(λ)** (source A): a direct quantification of how accurate the
   sky reconstruction is, with the whole comparison set listed in the same table.
2. **The injection test** (source B, and the only cheap truth for underneath a source): take
   the real sky and noise of a blank region of the wsky, lay on it a known source spectrum
   taken from a source region of the nosky, and plant it at the original blank position;
   then re-run the whole pipeline, where the mask has to detect the injected source.
   Measured: (a) the recovery rate of the source line flux, which is a quantitative upper
   bound on over-subtraction; (b) the sky reconstruction error at the injected position
   (the original blank observation there is already the true sky with noise on it, and the
   addition counts the noise twice over, which is noted when the result is read).
3. **The full truth-free set**: the blank residual std(λ), the line residual at 5577/6300/OH
   8400, pure noise in a line-free region (which may not rise appreciably), a 2D image of
   the residual (with no spatial structure in it), and the source's Hα before and after.
4. **Ablations** (one set of quantitative evidence for each design choice): v1 vs v2;
   with and without the second line of defence; a scan over λ_sp; the sensitivity to the
   mask size (against ZAP's sensitivity curve); the whole spectrum vs segments (only
   if §2.4 is triggered).
5. **An independent re-run**: the whole procedure re-run on the NE pointing cube to test the
   method's robustness, the t map re-measured with it.

**The iron rule of the two indicators (Principle 1)**: every table of results must present
how cleanly the sky came off and how faithfully the source was preserved at the same time,
and no conclusion may be drawn from the residual as a single indicator.

---

## 8. The upgrade path (the slots defined; outside the scope of this specification)

The framework is fixed as basis × spatial coefficients + source. What follows are the
settled upgrade slots for each component; each may replace what is there only once it has
been compared against the NN-free version under the protocol of §7:

| Slot | The current component | The upgrade option | The motivation, or what triggers it |
|---|---|---|---|
| The spatial coefficient field | Graph-Laplacian smoothing | **A two-stage GP regression** (a sparse GP; trained on the blanks, predicting over the source region) | When a point-by-point map of the predicted uncertainty is needed (how far the sky over the source region can be trusted, where extrapolation is dangerous); it is also the other end of the "interpolation vs interpolation plus local evidence" ablation |
| The spatial coefficient field | As above | **A neural field** (a coordinate MLP with Fourier features; Rhea 2024's route) | When the smoothness prior is not expressive enough; the upper bound on the Fourier bandwidth has to be matched to the known spatial scales |
| The spectral basis | The span of a linear NMF | **A small autoencoder** (a non-linear spectral manifold) | When the diagnostics show the sky spectrum varying non-linearly (if, say, the change of line shape with position through the LSF takes too many linear components to approximate) |
| The locality of the basis | NMF's natural tendency | An L1 sparsity penalty on B | When the components need explicit encouragement to localise on the line families; validated on held-out data |
| Updating the basis | B held fixed (the first line of defence) | Updating B jointly (with the diagnostics of §6(4) made mandatory alongside) | When evidence appears that the blank samples do not span the sky's variation over the source region; a scientific decision |
| More data | Built from a single cube | One B shared across exposures or across cubes (like ZAP's extSVD) | When several exposures or an offset sky become available |
| The errors | STAT copied verbatim (ZAP's limitation, shared) | Propagating the sky model's uncertainty through to the output from the fit covariance of ① | When downstream needs correct errors; the GP slot supplies it directly |

The settled discipline for the DL slots: the NN-free version is the baseline any NN version
has to beat, and an NN version's hyper-parameters go through the same held-out procedure of
§5.

---

## 9. The module structure

```
src/skymodel/
  preprocess.py        # throughput diagnostics and normalisation, continuum separation, weights and bad data
  linewindows.py       # the source line windows, generated from the redshift
  nmf_basis.py         # blank-only NMF (B settled) + NNLS initialisation (= the NMF-only baseline)
  joint.py             # the objective function (PyTorch), the v1/v2 switch, the optimisation loop
  bcd_reference.py     # the BCD reference implementation (for validation)
  holdout.py           # the held-out block split, and the RMSE
  inject.py            # the injection test
  diagnostics.py       # the standing diagnostic figures of §6(4)
  metrics.py           # the interface to docs/archive/metric_spec.md
```

---

## 10. Cross-references

- `CLAUDE.md` -- Principle 1 (the two indicators), Principle 2 (parameters have to be defensible).
- `docs/plan/data-and-metrics-overview.md` -- sources A/B/C and the metric framework.
- `docs/archive/metric_spec.md` (archived) -- the metric definitions.
- `docs/zap-parameters-reference.md` -- the ZAP comparison arm's settings (a single segment, automatic nevals, the mask).
- `docs/zap-conclusions.md` -- the conclusion about the wsky/nosky input, and the lesson about the mask.
- The literature anchors: Kolganov 2023 (a low-rank NMF sky), Rhea 2024 (spatial
  interpolation of the IFU coefficients), Zhang 2025 SMI (a sky per position, 5577
  efficiency normalisation), Soto 2016 (ZAP).
