# CLAUDE.md — Core Principles

This file records the **core principles** for working with Claude on this project.
Read this before starting any task.

---

## Project Goal

Study **sky subtraction / sky reconstruction** on the Haro11 MUSE cube:
learn a sky model from "sky-only" regions (blank spaxels), then reconstruct and
subtract the sky while preserving the source.
(Data and workflow: see `README.md`, `docs/`.)

---

## Big Principles

### Principle 1 — Use **sky reconstruction**
- The core idea is to **reconstruct the sky**: learn it from sky samples
  (blank spaxels / sky-only regions), then subtract it.
- Always evaluate **two things together**: how cleanly the sky is removed **and**
  whether the source is preserved. (Never judge on residuals alone — over-subtraction
  also flattens residuals and can silently eat the source.)
- Background: `docs/plan/data-and-metrics-overview.md` (data sources A/B/C, metrics).

### Principle 2 — Parameters must be **physically defensible**; if a value may NOT be, **warn me**
- When setting any hyperparameter, first ask **"is this value physically justified?"** —
  do not pick it arbitrarily, or just to make a number look good / make it run faster.
- **If a value might not be physically defensible, proactively warn me**, and explain:
  - why it is questionable;
  - whether there is a more physically-grounded alternative.
- Example: pushing the detection threshold down to **0.75σ** (below the noise floor,
  ~23% false-positive rate) is NOT defensible → warn me. The correct fix is to
  **boost the signal S/N with a matched filter, then use a normal threshold (≥2σ)**,
  not to lower the threshold below the noise.

---

## Operational Checklist for Principle 2 (masking / detection parameters)

Derive parameters from the data itself (header / measurements), not from guesses:

| Parameter | Should match | This Haro11 data |
|---|---|---|
| pixel scale | header `CD1_1` | 0.20″/px |
| seeing / PSF FWHM | measure PSF from a star in the cube (the QC keyword is empty) | ≈0.81″ ≈ 4 px (measured; `ESO QC EXPCOMB FWHM MEDIAN` = 0.0, do not use) |
| matched-filter kernel FWHM | **≈ seeing FWHM** | gauss FWHM ≈ 4 px |
| detection threshold | **≥ 2σ** (2.3% false positives, standard) | 2σ (on matched-filtered image) |
| minarea | **≈ 1 PSF area** | ≈ 13 px |
| dilation (safety margin) | **≈ 1 × seeing FWHM** | ≈ 4 px |
| background box `bw` | **> largest object**; use global if the object is huge | halo Ø≈226 px → global / bw ≥ 256 |

> **Seeing note:** `ESO QC EXPCOMB FWHM MEDIAN` is **unpopulated (0.0)** in this dataset (both
> `Haro11_nosky.fits` and `Haro11_NEpointing_esonosky.fits`), so the seeing must be **measured from a
> star in the cube** (PSF FWHM = 2.3548·√(a·b) of compact, round sources), not read from that keyword.
> The measured PSF is **≈4.06 px ≈ 0.81″**; header proxies (`ESO OCS SGS AG FWHM{X,Y} MED` ≈0.89″,
> `ESO TEL AMBI FWHM` ≈0.94–0.96″) agree at ≈4–4.7 px. The earlier 6 px / 1.24″ was an unverified assumption.

**Two common traps to avoid:**
1. **Estimate the noise σ from source-free regions** (sigma-clip out the sources).
   Estimating MAD over the whole image (sources included) inflates σ and distorts the
   threshold. `sep`'s `Background.rms()` rejects sources automatically.
2. **A too-small background box (`bw`) absorbs the extended halo as "background"** →
   the halo is never detected, no matter how low the threshold. The box must be larger
   than the object (or global).

---

## Other

- Established ZAP conclusions, speed experiments, and masking experiments live in
  `docs/`, the scripts under `src/`, and `docs/progress/`.
- Before changing parameters that affect the science result (e.g. ZAP segmentation
  `SKYSEG`, `cfwidthSP`), first confirm whether the change is result-preserving. If it
  is not, treat it as a scientific decision — validate it and warn me.
