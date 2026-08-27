# The authoritative ZAP parameter reference (Haro11 MUSE cube)

> This file is this project's **authoritative parameter handbook** for ZAP (Zurich Atmosphere Purge, Soto+2016):
> it explains the algorithm, defines every keyword of `zap.process` / `zap.SVDoutput` one by one, lists the defaults actually in force in this project,
> marks — following `CLAUDE.md` Principle 2 — which of those defaults still need scientific validation, and finally gives a reproducible procedure for "tuning the parameters on any new cube".
>
> **Source of truth**: the ZAP source vendored into this project, `libs/zap/`.
> Version **2.1** (`zap.__version__ == '2.1'`; `git describe` = `2.1-6-g974231e`, commit `974231e`, and the newest CHANGELOG entry is `2.2.dev`).
> Every "default value / behaviour" stated in this file has been checked one by one against `libs/zap/zap/zap.py`; where it disagrees with readthedocs or with the Soto+2016 paper, **the vendored 2.1 is what counts**, and the divergence is noted.
>
> Related documents: the settled conclusions are in `docs/zap-conclusions.md`; the physical derivation of the source mask parameters is in `docs/archive/segmentation-parameters-explained.md`; the core principles are in `CLAUDE.md`.
> Where it is actually called: `src/zap/zap.py`; the nevals scan procedure is in §4.3 (the historical script `tune_nevals.py` has been removed and survives in git history).

---

## 0. Where this sits, in one line

ZAP subtracts the sky by **reconstructing the sky**: first a per-wavelength median removes about 99% of the sky, then a PCA/SVD is run on what is left of it — on the **pure-sky spaxels outside the source mask** — to learn eigenspectra of the residual sky, and an appropriate number of components is chosen to reconstruct it and subtract it. ZAP is the engine of `CLAUDE.md` Principle 1, "sky reconstruction": **the sky basis is learned only from source-free spaxels**, and bright sources take no part in it. Evaluation must therefore always look at two things at once — **the sky-line residual coming down, and the source flux being preserved** (Principle 1).

---

## 1. The ZAP algorithm (so that every parameter has something to stand on)

ZAP does the following to each spectrum (spaxel), in order (`Zap._run` → `_prepare` → `_msvd` → `optimize`/`chooseevals` → `reconstruct` → `remold`):

1. **NaN cleaning (`clean`)** — each spaxel is checked for NaNs. A spaxel whose NaN fraction is >25% is thrown out whole and not processed; the remaining NaNs are interpolated from a 3×3×3 neighbourhood, and once the processing is done the NaNs are put back into the output cube (`_nanclean`, with `boxsz=1` and `rejectratio=0.25`).

2. **Zeroth-order sky subtraction, zlevel (`zlevel`)** — for each monochromatic layer the **median** is taken (or a sigma-clipped mean) and subtracted. This step alone removes about 99% of the sky signal; what is left is instrument-related residual (the LSF, discontinuities in the wavelength calibration, flat-field errors), which shows up spatially as variation from one spaxel to the next.

3. **Continuum filtering (`cftype` / `cfwidth`)** — on the stack that is left after zlevel subtraction, each spectrum is **median filtered** along wavelength (with a window `cfwidth` wavelength pixels wide; implemented as `uniform_filter(width=3)` followed by `median_filter(cfwidth)`). The purpose is to flatten "the astronomical continuum" and leave only the narrow sky residual, so that what the PCA afterwards learns is **the sky** and not **the source's continuum**. The continuum that was filtered out is kept in `contarray`, and the residual in `normstack`.

4. **Variance normalisation** — within each segment, `normstack` is divided by its per-spaxel variance, so that the SVD is not dominated by individual bright spaxels.

5. **Cutting the sky into segments (`SKYSEG`)** — the wavelength axis is cut into several segments, and each segment gets its own PCA. The physical reason for segmenting: sky emission lines are produced by **families of transitions** of OH, O₂, O I, Na I and so on, so within one wavelength segment the sky residual is highly correlated in time and in space; and artefacts of the data reduction are likewise "coherent over a short wavelength interval". **The default in the vendored 2.1 is a single segment** (see §4.1).

6. **PCA / SVD (per segment)** — that segment's `normstack` (of dimensions ≈ n wavelength pixels × m spaxels) is put through a singular value decomposition. The rows (spaxels) are the samples and the columns (wavelengths) are the features, and what comes out is a set of **sky eigenspectra** (shapes along wavelength). Because the masked source spaxels have already been set to NaN and are dropped in `_extract`, **this basis is spanned by pure-sky spaxels alone** (sklearn `PCA`).

7. **Choosing the number of components (`nevals` / `optimize`)** — this decides how many of the leading eigenspectra go into the reconstruction. The variance curve (explained variance vs number of components) has two parts: **a steep drop at the front** (the first ~10 components are subtracting genuine residual sky) and **a nearly linear tail** (the higher-order components start eating into the source and the noise). ZAP's automatic method finds the turn between the two (`optimize`: it takes the first derivative over the leading 25% of the components and finds where that falls back to `mean − 5σ`; this is the implementation of the paper's "the inflection point where the second derivative of the variance curve goes to zero"). With `nevals=[]` the automatic method is used; giving a value fixes it instead.

8. **Reconstruct and subtract (`reconstruct` / `remold`)** — the chosen components are used to project and back-project the residual sky into `recon`, which is subtracted from the stack and put back into the cube as `cleancube`.

**Why the number of components makes or breaks it**: the sky eigenspectra are constrained only at the wavelengths of sky lines, and are not constrained anywhere else (at the source's emission lines and continuum). Take too many components and the higher-order eigenspectra begin to fit and subtract **genuine source flux and line shapes** → over-subtraction, eating the source. Take too few and the sky residual is not cleanly removed. This is exactly the root of `CLAUDE.md` Principle 1, "you cannot look at the residual alone": over-subtraction lowers the residual **and** eats the source at the same time.

**The role of the mask (the heart of sky reconstruction)**: source spaxels flagged ≥1 in `mask` are set to NaN before the SVD is built and are thereby kept out of the sky basis (`_applymask`). Too little masking → the source (above all Haro11's extended Hα halo) mixes into the sky basis → and is subtracted as if it were sky. Measured in this project: a 2σ white-light cut masks only 8% and produces a **70% loss of source flux**; detection plus dilation masks 44%, after which the source is preserved at 124% (see `docs/zap-conclusions.md` §1). The derivation of the mask's own parameters is in `docs/archive/segmentation-parameters-explained.md`.

---

## 2. The complete `zap.process` parameter reference

The actual signature (checked word for word against `libs/zap/zap/zap.py`, ZAP 2.1):

```python
zap.process(cubefits, outcubefits='DATACUBE_ZAP.fits', clean=True,
            zlevel='median', cftype='median', cfwidthSVD=300, cfwidthSP=300,
            nevals=[], extSVD=None, skycubefits=None, mask=None,
            interactive=False, ncpu=None, pca_class=None, n_components=None,
            overwrite=False, varcurvefits=None)
```

| Parameter | Type / unit | Default (2.1) | What it means, and what changing it does | Physical / practical guidance |
|---|---|---|---|---|
| `cubefits` | str | (required) | The input cube's filename. For MUSE it reads extension 1 (DATA). | **A divergence to note**: the paper and the old readthedocs call this `incubefits`, whereas the positional argument in the vendored 2.1 is named `cubefits`. It takes a single filename string only. |
| `outcubefits` | str | `'DATACUBE_ZAP.fits'` | The sky-subtracted output cube. It is written back into the original file's structure with `mergefits`, **keeping every extension, STAT included**. | STAT is **copied across verbatim** and is not propagated through ZAP (see §5). This project sets it to `.../zap.fits`. |
| `clean` | bool | `True` | NaN cleaning (what the paper calls nanclean). Spaxels with >25% NaN are thrown out and the rest are interpolated; on output the NaNs are put back. | Keep it `True`. MUSE cubes have a great many NaNs at the edges from atmospheric refraction, and turning this off leaves any spaxel holding even one NaN entirely unprocessed. |
| `zlevel` | str | `'median'` | The zeroth-order sky subtraction method: `'none'` / `'sigclip'` (a 3σ clipped mean) / `'median'`. | `median` is robust and is the standard choice. `sigclip` costs more and is the safer bet against a strong source, but is generally not necessary. `none` is only for when the sky has already been subtracted externally. |
| `cftype` | str | `'median'` | The continuum filtering method: `'median'` / `'fit'` (a degree-5 polynomial) / `'none'`. | `median` is the 2.1 default and the most stable. `'fit'` is MUSE-specific (it excludes the red end beyond pixel>3600 and the notch region) and **readily goes out of control at the red end**, so do not use it without a reason. `'none'` does not filter the continuum at all, and suits only a field with no continuum in it. |
| `cfwidthSVD` | int (**wavelength pixels**) | `300` | The continuum filtering window used when building the **SVD basis**. | 300 px. For this cube `CD3_3=1.25 Å/px` → **300 px = 375 Å**. A large value, and a stable one for the sky continuum. |
| `cfwidthSP` | int (**wavelength pixels**) | `300` | The continuum filtering window used when computing **each individual spectrum's eigenvalues/projection**. | **⚠ Scientifically critical, and named in `CLAUDE.md`**. ZAP's own docstring says the best range is **20–50 px** (=25–62.5 Å) because it "tracks the source better"; the default of 300 px is on the large side. See §3 and §4.2. |
| `nevals` | list / int | `[]` | How many eigenspectra each segment is reconstructed with. `[]`→ automatic `optimize()`; a single value → the same for every segment; a list as long as the number of segments → one value per segment. | **⚠ Scientifically critical**. Too many → over-subtraction that eats the source; too few → sky residual left behind. The automatic method is usually reasonable, but **it must be validated against both indicators** (§4.3). The docstring's "a list of 11 values" is a leftover from the old 11-segment design; with 2.1's single segment a single value is enough. |
| `extSVD` | Zap object | `None` | Use instead an SVD basis computed by `zap.SVDoutput(...)` on **a different cube** (an offset sky, or another exposure). | For multiple exposures / a filled field. **Mutually exclusive with `mask`** (giving both raises `ValueError`: if you want a mask, the SVD has to be recomputed). See §4.4. |
| `skycubefits` | str | `None` | Additionally writes out "the sky that was subtracted" = input − output cube (`writeskycube`). | For diagnostics. This project sets it to `.../sky.fits`. |
| `mask` | str | `None` | A 2D FITS mask: source is **≥1**, sky is **0**. Source spaxels are set to NaN and kept out of the sky basis. | The heart of sky reconstruction. The quality of the mask decides the outcome directly (§1, `docs/archive/segmentation-parameters-explained.md`). |
| `interactive` | bool | `False` | With `True` it returns the `Zap` object and **writes no file**, so that `reprocess(nevals=...)` can scan the component count quickly. | For the nevals scan (procedure in §4.3). |
| `ncpu` | int | `None` | The number of parallel processes. `None`→`cpu_count()` (every core). | Set it to the cores available. This project sets 16; over the whole field the measured peak memory is ~43.7 GB (`zap-conclusions.md` §4). |
| `pca_class` | class | `None` | Replaces the PCA implementation class. `None`→sklearn `PCA`. | Advanced; normally left alone. |
| `n_components` | float | `None` | How many PCA components to compute (not the nevals used for the reconstruction). When given, `ncomp = max(nwave_seg × n_components, 60)`. | Advanced; `None`→ compute the full PCA. Changing it affects only the ceiling on how many components are computed, and the speed — it is not the number of components used for the reconstruction. |
| `overwrite` | bool | `False` | Whether to overwrite an output file that already exists. | This project sets `True`. |
| `varcurvefits` | str | `None` | Additionally writes each segment's `explained_variance_` curve out as a FITS table (`writevarcurve`). | **Strongly recommended**: together with the `ZAPNEV*` header keywords it lets you see whether the automatically chosen component count is reasonable. This project sets it to `.../var.fits`. |

**The ZAP keywords written into the output header** (`_newheader`): `ZAPvers`, `ZAPzlvl`, `ZAPclean`, `ZAPcftyp`, `ZAPcfwid`, `ZAPnseg`, and, per segment, `ZAPseg{i}` (the segment's pixel range) / `ZAPnev{i}` (the number of components used for that segment). **After a run, always check `ZAPnev*` and `ZAPnseg`.**

### 2.1 Divergences from readthedocs / the paper (the vendored 2.1 is what counts)

- **No `pevals`, no `optimizeType`**: ZAP 1.0 once had these two keywords (choosing components by percentage, and the selection mode). 2.x has removed them; automatic component selection is done instead by the `optimize()` method (the derivative-of-the-variance-curve method). If an external tutorial mentions `pevals`/`optimizeType`, it **does not apply** to this vendored version.
- **No `Zap.getzcube`**: `getzcube` does not exist in 2.1 either, so an external source naming it **does not apply** to this vendored version. The related methods are `make_cube_from_stack`, `make_contcube`, `writecube`, `writeskycube`, `writevarcurve`, `mergefits`, `reprocess` and `optimize`.
- **SKYSEG defaults to a single segment**: the paper (1.0) describes 11 segments; from 2.0 onwards the default is **a single segment** (see §4.1).
- **`cfwidth` defaults to 300**: the paper describes 100 px (for building the basis) + 20–50 px (for computing the eigenvalues); from 2.0 onwards the defaults were merged into 300 (the old 100/50 was too small and caused the background to oscillate at the red end, see CHANGELOG 2.0).

---

## 3. The related top-level functions

| Function | Signature (2.1) | Purpose |
|---|---|---|
| `zap.SVDoutput` | `(cubefits, clean=True, zlevel='median', cftype='median', cfwidth=300, mask=None, ncpu=None, pca_class=None, n_components=None)` | Computes an SVD basis on some cube and returns a `Zap` object that can be fed to `process(extSVD=...)`. For multiple exposures / an offset sky. Note that there is only a single `cfwidth` here (corresponding to `process`'s `cfwidthSVD`). |
| `zap.nancleanfits` | `(cubefits, outfn='NANCLEAN_CUBE.fits', rejectratio=0.25, boxsz=1, overwrite=False)` | Runs the NaN interpolation on its own and writes the result to a file. |
| `zap.contsubfits` | `(cubefits, outfits='CONTSUB_CUBE.fits', ncpu=None, cftype='median', cfwidth=300, clean_nan=True, zlevel='median', overwrite=False)` | Writes out a continuum-subtracted cube on its own (for diagnosing the continuum filtering). |
| `zap.mask_nan_edges` | `(cube, outfile=None, plot=False, threshold=50, extname='DATA')` | Masks out the edge spaxels that hold too many NaNs (>threshold %), so that these spaxels, which never had the sky subtracted, do not leave a high residual in the output. |

What happens inside `process`: if a `mask` was given (or if `cfwidthSVD != cfwidthSP`), it first calls `SVDoutput(cfwidth=cfwidthSVD, mask=mask)` to build the basis from **the masked cube**; then it builds the `Zap` object, does the per-spectrum continuum filtering with `cfwidthSP`, and reuses that same basis for the projection and reconstruction. **The `mask` therefore acts only at the SVD-basis step** (zlevel is computed under the mask as well), while `cfwidthSP` decides how finely the source is tracked in each spectrum when the sky is actually subtracted.

---

## 4. The defaults currently in force in this project, and whether they stand up

`src/zap/zap.py` passes `mask`, `ncpu=16`, `cfwidthSP` (at the default 300 = ZAP's default) and `overwrite=True` (plus the three output paths); everything else takes ZAP 2.1's defaults:

| Knob | Value in force | Does it stand up physically (Principle 2) |
|---|---|---|
| `zlevel` | `'median'` (the default) | ✅ Standard and robust, no need to change it. |
| `cftype` | `'median'` (the default) | ✅ The 2.1 default and the most stable. Changing it to `'fit'` is not advisable (it readily goes out of control at the red end). |
| `cfwidthSVD` | `300 px = 375 Å` (the default) | ✅ A large window for building the basis is reasonable. |
| `cfwidthSP` | `300 px = 375 Å` (the default) | ⚠ **Needs scientific validation**. ZAP's own documentation says 20–50 px is best; 300 px is coarse for a "bright compact core + extended Hα halo". See §4.2. |
| `SKYSEG` | `[]` → **a single segment**, 4750–9348 Å (the default) | ✅ This is precisely what 2.x recommends as the default (§4.1). Changing to several segments is a scientific decision and needs validating. |
| `nevals` | `[]` → **automatic** `optimize()` (over the whole field it chose 53 in practice) | ⚠ **Needs validating on both indicators**. The automatic method works properly on this data (the source is preserved at 124%), but source fidelity still has to be looked at, not the residual alone. See §4.3. |
| `extSVD` | `None` → builds its own SVD from the same cube (a single-exposure reconstruction) | ✅ At the moment there is only 1 sky-included cube; with multiple exposures, switch to extSVD (§4.4). |
| `clean` | `True` (the default) | ✅ Keep it. |
| `ncpu` | `16` (given explicitly) | ✅ Follows the machine's cores; changing it does not affect the scientific result, only the speed and the memory. |

> **The `CLAUDE.md` Principle 2 warning (this is settled guidance, not a draft)**:
> `cfwidthSP` and `SKYSEG` are named as scientifically critical in `CLAUDE.md`'s "Other" section. Of the two, the current value of `SKYSEG` (a single segment) is itself the recommended default and stands up on its own; **`cfwidthSP=300` (the default) is on the large side and may under-track the source's continuum**, so evaluating 20–50 px as described in §4.2 is advisable. Any change away from the current values — of `cfwidthSP`, of `SKYSEG`, or to a fixed `nevals` — **changes the scientific result**, and must first be checked for whether it is result-preserving; if it is not, treat it as a scientific decision and adopt it only after validating it against the two indicators of §4.3.

---

### 4.1 SKYSEG (the sky segment boundaries)

- **The vendored 2.1 default = a single segment**: `SKYSEG = []`, and `Zap` takes a single interval from the cube's λ min/max (for this cube the whole of 4750–9348 Å, `ZAPnseg=1`).
- **The physical reason**: a single segment lets the PCA make use of **the correlation of the sky lines across the entire wavelength range**, which subtracts the sky more cleanly; and it greatly **reduces the risk of killing emission lines** (several segments make the continuum oscillate, and the per-segment component count becomes extremely sensitive). This is why 2.0 changed to a single segment (CHANGELOG 2.0).
- **The old 11-segment boundaries (for reference only)**: `[0, 5400, 5850, 6440, 6750, 7200, 7700, 8265, 8602, 8731, 9275, 10000]` Å — grouped by the OH / O₂ / Na I / [O I] sky-line families and by the breaks in the instrument response.
- **When to change it**: only when a single segment is plainly not subtracting cleanly, and the diagnostics point to the sky residual behaving very differently in different wavelength regions. **How to change it**: `from zap.zap import SKYSEG; SKYSEG[:] = [...]` (edit the list in place; it cannot be reassigned).
- **Validation**: changing the number of segments changes both the component selection in each segment and the continuum behaviour, so it is **not result-preserving**. A single segment has to be compared against several using the two indicators of §4.3: the sky-line residual has to come down, the source's Hα flux and line shape have to be preserved, and no noise may be poured into the line-free regions.

### 4.2 cfwidth / cfwidthSP (the continuum filtering window)

- **The unit is wavelength pixels, not Å.** For this cube `CD3_3 = 1.25 Å/px`, so the conversion is `300 px = 375 Å` and `20–50 px = 25–62.5 Å`.
- **The trade-off**:
  - Too small → the filter chases the source's own continuum and emission-line structure, and **subtracts the genuine source continuum as though it were residual** (eating the source).
  - Too large → the filter is too smooth, it **under-tracks the source**, and the source's continuum leaks into `normstack`, where the PCA may learn it into the sky basis.
- **The division of labour between the two windows**: `cfwidthSVD` (building the basis) can use a large window for stability (the default 300 is reasonable); for `cfwidthSP` (the per-spectrum projection) ZAP's own documentation says **20–50 px** is best, because this step has to track the source finely and avoid taking the source for sky.
- **Where this project stands, and what is advised**: `cfwidthSP` currently takes the default of 300 px (=375 Å), which is coarse for something like Haro11 with its "bright compact core + low surface brightness Hα halo". **Measuring across the 20–50 px range (≈25–62.5 Å) is advisable**; but because `CLAUDE.md` names this parameter as scientifically critical, **changing it is a scientific decision**, and it may only be adopted once the two indicators of §4.3 have confirmed that source preservation does not get worse and the sky residual does not get worse either.
- If all you want is to see the effect of the continuum filtering on its own, `zap.contsubfits(cfwidth=...)` writes out a continuum-subtracted cube to inspect.

### 4.3 nevals / the number of components — validate on both indicators, no magic number

- `nevals=[]` uses the automatic `optimize()`; over the whole field it automatically chose **53**, the source was preserved at 124% and the sky lines were pushed down to ~0–1.3 (`zap-conclusions.md` §2), so the automatic method **works properly** on this data.
- **But the component count is a direct trade of source fidelity against a clean sky**: too many → over-subtraction that eats the source; too few → sky residual left behind. **`CLAUDE.md` Principle 1: never look at the residual alone** — over-subtraction lowers the residual and eats the source at the same time.
- **A settled conclusion, not to be repeated**: when ZAP was run on the already sky-subtracted `nosky`, sweeping the component count from 3 up to 55 **rescued nothing** of the over-subtraction — because the problem was **the wrong input cube** (nosky has no sky left to learn from), not the tuning of nevals (`zap-conclusions.md` §1). **First make sure that what you are feeding it is the sky-included `wsky`**, and only then talk about the component count.
- **The validation procedure (reproducible)** — compute the SVD once with `interactive` and scan quickly with `reprocess()` (the historical template `tune_nevals.py` survives in git history):

  ```python
  import zap
  zobj = zap.process("<wsky_cube>.fits", mask="<source_mask>.fits",
                     interactive=True, overwrite=True)   # compute the SVD only once
  print("auto nevals =", zobj.nevals)
  for N in [3,5,8,10,12,15,20,25,30,40, int(zobj.nevals[0])]:
      zobj.reprocess(nevals=[N])
      # measure the three indicators on zobj.cleancube (see below)
  ```

  On source-free blank spaxels and on source spaxels alike, measure **three indicators**:
  1. **The spatial std of the sky lines** (at 5577/6300/8400 Å, say): it has to **come down** to close to the MUSE `nosky` truth.
  2. **Pure noise in a line-free region** (the per-spaxel RMS over 7000–7120 Å, say): it **must not rise appreciably** (an empirical threshold of ≤ 1.5× raw).
  3. **The source's integrated Hα flux**: **preserved at ≥ ~98%** (relative to raw), with the line shape unchanged.

  Pick the N that satisfies all three at once and is as high as possible (the cleanest sky subtraction). The definitions of the three and their thresholds are in `docs/archive/metric_spec.md` (archived).

### 4.4 ncpu and extSVD (multiple exposures)

- **`ncpu`**: set it to the number of cores available (it does not affect the scientific result). The full 499×559×3679 cube peaks at about 43.7 GB of memory, so make sure there is enough RAM.
- **`extSVD` (multiple exposures / a filled field)**: at present this project has only a single sky-included cube (a single-exposure reconstruction). **If there are ever multiple exposures or an offset sky frame**, compute the SVD on one frame (with the sources well masked) and apply it to each of the frames:

  ```python
  extSVD = zap.SVDoutput("offset_or_expo1.fits", mask="mask.fits", cfwidth=300)
  zap.process("science_expo.fits", outcubefits="out.fits", extSVD=extSVD)
  ```

  Note that **`extSVD` and `mask` cannot both be given** (if you want a mask, the SVD has to be recomputed); an offset frame only needs a short exposure of 2–3 minutes.

---

## 5. The output, and how STAT is handled (important)

- `outcubefits` is written by `mergefits`: **it opens the original input file, replaces only DATA (extension 1) with `cleancube`, and leaves every other extension, STAT included, exactly as it was**.
- **STAT (the per-voxel variance) is therefore copied verbatim, and is not propagated through ZAP.** ZAP's sky subtraction does not update the variance; if later analysis needs correct error propagation, that has to be dealt with separately (this is a known limitation, see `zap-conclusions.md` §5).
- `skycubefits` = input − output (the sky that was subtracted); `varcurvefits` = each segment's explained variance curve (for diagnostics).

---

## 6. Cross-references

- `CLAUDE.md` — Principle 1 (sky reconstruction, the two indicators), Principle 2 (parameters have to be physically defensible), and the "Other" section, which names `SKYSEG`/`cfwidthSP` as scientifically critical.
- `docs/zap-conclusions.md` — the settled conclusions: feeding it the right cube (wsky) is what matters, too little masking eats the source, sweeping the component count cannot rescue a wrong input, and the numbers over the whole field.
- `docs/archive/segmentation-parameters-explained.md` — the full physical derivation of the `mask`'s detection parameters (threshold 2σ, matched filter kernel = the seeing, bw > the halo, minarea, dilation).
- `docs/archive/metric_spec.md` (archived) — the definitions of the evaluation metrics.
- `src/zap/zap.py` — where it is actually called.
- The ZAP paper: Soto, Lilly, Bacon, Richard & Conseil (2016), MNRAS 458, 3210. The vendored source: `libs/zap/` (2.1).

---

## 7. What could not be fully verified, and what remains open

- The details of the ZAP paper PDF (the original recommendation of 100 px vs 20–50 px, and which sky-line families the 11 segment boundaries belong to) were taken from the paper's HTML (ar5iv) and from the readthedocs summary; **the step-by-step behaviour of the algorithm follows the vendored 2.1 source**, and the paper's numbers serve only as background. Wherever the two disagree (the number of segments, the cfwidth defaults, `pevals`/`optimizeType`, `getzcube`), this file has noted it in §2.1 and taken 2.1 as authoritative.
