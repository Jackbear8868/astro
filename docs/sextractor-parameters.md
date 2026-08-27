# SExtractor detection parameters (this project's masking workflow)

> How this project detects sources and builds its mask: **SExtractor run on the white light
> image**, which is the workflow the project was handed rather than one worked out here.
> Working directory: `src/skymodel/SExtractor/` (its README covers running it); downstream:
> `src/skymodel/pipeline.py` — step3 uses the result to decide what counts as blank, and
> step5 and step6 use it to decide what counts as source.
>
> **Division of labour**:
> - This document = the **general physical meaning** of each detection parameter, and how to
>   derive a reference value for it from any cube's data or header.
> - The physical reasoning behind "why the Haro11 data is set up this way" is in
>   [`docs/archive/segmentation-parameters-explained.md`](./archive/segmentation-parameters-explained.md).
> - The core principles are [`CLAUDE.md`](../CLAUDE.md) Principle 2 and its "Operational
>   Checklist" table.
>
> **How the baseline and the reference values rank (Principle 2)**: `default.sex` is
> **the authoritative baseline the project was given, and it is used exactly as it came**;
> trying a different value for any parameter means overriding it on the command line
> (for example `-DETECT_THRESH 2.0`), and an override of that kind is an experiment.
> The "self-derived reference" column below is **there to explain the physics and to give a
> discussion something to stand on, not to correct the baseline**.

---

## 1. The detection parameters and what they mean physically

How the background is estimated (SExtractor's `BACK_*` family): the image is cut into square
cells of side `BACK_SIZE`; inside each cell the sources are sigma-clipped away and the local
background and RMS are estimated from what remains; the grid of cell values is then smoothed
with a median filter `BACK_FILTERSIZE` cells wide and interpolated back onto every pixel.
**The sources are rejected automatically before the noise is measured**, so the RMS that comes
out is the σ of clean sky — which is exactly the σ a detection threshold should be measured
against. (Estimate σ over the whole image with the sources still in it and the sources pull σ
up, which distorts the threshold along with it.)

| keyword | physical meaning | baseline (`default.sex`) | effect of raising / lowering it | self-derived reference (for discussion) |
|---|---|---|---|---|
| `DETECT_THRESH` | detection threshold, in units of the background RMS (σ); a pixel takes part in a detection only if it exceeds `thresh × σ` | **1.5** | high → only bright sources, faint ones missed; low → reaches fainter, but false positives rise (see §3) | 2σ (2.3% false positives) |
| `DETECT_MINAREA` | how many connected above-threshold pixels a detection needs before it counts as a source | **10** | high → filters out small noise specks but can miss small sources; low → picks up noise fragments | ≈1 PSF area = π(FWHM/2)² ≈ 13 px |
| `FILTER` / `FILTER_NAME` | the smoothing kernel applied before detection (matched filter, see §2); `default.conv` = Gaussian with FWHM≈2 px | **Y / default.conv** | wide kernel → suppresses noise and favours faint extended sources, but blurs small structure; narrow kernel → keeps detail but does not suppress enough noise | Gaussian FWHM ≈ seeing ≈ 4 px |
| `BACK_SIZE` | side length of a background cell (px) | **64** | large → smooth background that does not eat extended sources, but responds sluggishly to background gradients; **small → an extended source larger than the cell is estimated as background and subtracted away** (see §4) | > the largest object to be kept (halo Ø≈226 → ≥256, or global) |
| `BACK_FILTERSIZE` | width of the median filter over the background cells (in cells) | **3** | large → suppresses cells contaminated by a bright source, at the cost of a blurrier background | 3 (standard) |
| `DEBLEND_NTHRESH` | how many levels deblending cuts between [threshold, peak] while looking for sub-peaks | **64** | more → better at separating sources that lie close together | 32 (standard) |
| `DEBLEND_MINCONT` | the smallest fraction of the parent source's flux a sub-peak must hold to count as an independent source; 1.0 = deblending off | **0.0005** | high → inclined not to split; low → splits aggressively | 0.005 (standard) |
| `CLEAN` / `CLEAN_PARAM` | removes spurious detections caused by the wings of bright sources and by noise | **Y / 1.0** | large → cleans more aggressively | the default |

Dilating the mask afterwards, as a safety margin, is not part of SExtractor itself: downstream
code does it with `scipy.ndimage.binary_dilation`, at a reference scale of ≈ 1×seeing (≈4 px).

---

## 2. The matched filter: why the kernel FWHM ≈ seeing, and its limit on the extended halo

- **The principle**: when what is being detected is a signal of known shape sitting in white
  noise, the filter that maximises S/N is **a kernel of the same shape as the signal**
  (the matched-filter theorem). An astronomical point source is smeared by the atmosphere into
  the PSF, so the best kernel for point-source detection is ≈ the **PSF (a Gaussian with
  FWHM = seeing)**.
- **Why that particular scale**: the seeing FWHM is the smallest scale at which real structure
  can exist in the image — variation finer than that cannot be a real object. Smoothing at that
  scale suppresses as much noise as possible while blurring away no real structure. Too small a
  kernel does not suppress enough noise; too large a kernel blurs position and shape.
- **The limit on the extended halo**: a matched filter is optimal for **point sources**;
  Haro11's Hα halo is extended, low surface brightness structure on a scale far larger than the
  seeing. A kernel of one seeing does raise the halo's per-pixel S/N, but it is **not the
  strictly optimal kernel for large-scale structure**. The right way to catch a faint halo is
  "raise S/N with the matched filter, plus a large enough background box (§4)", not to push the
  threshold down below the noise (§3).

---

## 3. What a threshold means statistically

A threshold corresponds to a false positive rate — the chance a pure-noise pixel is mistaken
for a source (Gaussian, one-tailed):

| threshold | chance pure noise exceeds it |
|---|---|
| 0.75σ | ≈ 23% |
| 1σ | ≈ 16% |
| 1.5σ | ≈ 6.7% |
| 2σ | ≈ 2.3% (2–5σ is the usual astronomical range) |

A low threshold buys a fainter detection limit and pays for it in false positives. For catching
a faint extended halo, lowering the threshold is not the first lever to reach for — raising S/N
with the matched filter and pairing it with a large enough background box is the established
approach (§2, §4). The current 1.5σ baseline and the two-threshold experiment (1σ/2σ) are
**working values the project was given and is still exploring**; this table only supplies the
statistical background, and passes no judgement on those working values.

---

## 4. How `BACK_SIZE` fails: the background box must be larger than the object being kept

- **The failure mode**: if `BACK_SIZE` is **smaller** than an extended object, the background
  cells sit entirely inside the object, so the **object itself is estimated as background** and
  subtracted → the object disappears, and **no threshold, however low, will detect it**.
- **The rule**: `BACK_SIZE` must be larger than the diameter of the largest object you want to
  keep; for a huge extended halo, use a global background or `≥ the object diameter`.
  Haro11's halo is Ø≈226 px → reference value 256. (This agrees with the CLAUDE.md checklist,
  and the difference between the baseline 64 and the reference value is one of the items that
  checklist already files as a question for discussion.)

---

## 5. Deriving values file by file (how to get reference values for any new cube from its header and data)

| parameter | derived from | formula | value for Haro11 |
|---|---|---|---|
| pixel scale | header `CD1_1` | `√(CD1_1²+CD2_1²)×3600` (deg→arcsec) | 0.20″/px |
| seeing FWHM (px) | **the PSF of stars measured in the cube itself** (see below) | `median(2.3548·√(a·b))` over the stars | **≈4.06 px ≈ 0.81″** |
| smoothing kernel FWHM | = seeing FWHM | — | ≈4 px |
| detection threshold | the statistical standard (§3) | — | 2σ (reference) |
| MINAREA | the area of 1 PSF | `π(FWHM/2)²` | ≈13 px |
| dilation (downstream) | 1×seeing | `round(FWHM)` | ≈4 px |
| BACK_SIZE | > the largest object (§4) | take the power of 2 that is ≥ the object diameter | 256 (or global) |

**Where the seeing comes from, in order of preference**:

1. **First choice — measure the PSF FWHM directly from stars in the cube**: build a continuum
   white light image with the emission lines removed, extract the sources, and for each one
   compute `FWHM = 2.3548·√(a·b)` from its second moments; keep only stars that are compact
   (FWHM < 8 px), round (b/a > 0.6) and bright enough, and take the median. Measured on this
   Haro11 data: **≈4.06 px = 0.81″** (10 stars, 16–84% range 3.58–4.77 px).
2. **Fallback when no star is usable — header proxies**: `ESO OCS SGS AG FWHMX/Y MED`
   (autoguider) ≈0.89″≈4.4 px, and `ESO TEL AMBI FWHM` (DIMM) ≈0.94–0.96″≈4.7 px; both agree
   with the measurement in pointing to ≈4 px.
3. ⚠️ **Not to be used blindly**: `ESO QC EXPCOMB FWHM MEDIAN` is **0.0 (unpopulated)** in this
   data (`Haro11_nosky.fits`, `Haro11_NEpointing_esonosky.fits`), so reading it straight gives a
   0-px kernel; `ESO OCS SGS FWHM *` is likewise 0.0. Any automatic read needs a `fwhm > 0`
   guard.

---

## 6. Cross-references and bibliography

- [`docs/archive/segmentation-parameters-explained.md`](./archive/segmentation-parameters-explained.md) — the
  physical reasoning behind the parameters for this Haro11 data.
- [`CLAUDE.md`](../CLAUDE.md) — Principle 2 and the Operational Checklist (how parameters rank:
  the baseline the project was given outranks a self-derived reference).
- `src/skymodel/SExtractor/` — the current working directory (`default.sex` baseline, the det
  image rescaling formula, the batch script).

**Authoritative sources**:

- Bertin, E. & Arnouts, S. (1996). *SExtractor: Software for source extraction.* A&AS, 117, 393–404.
  doi:10.1051/aas:1996164 — the original algorithm (background, segmentation, deblending, CLEAN).
- The SExtractor parameter manual: https://sextractor.readthedocs.io/ .
- The older document (the full sep API comparison; the sep workflow is no longer used) is kept
  in git history (`docs/sep-sextractor-parameters.md`).
