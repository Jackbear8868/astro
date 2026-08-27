# Summaries of the key predict-sky papers, with the focus on ML and data-driven work

> Read closely: `ml/Zhang2025_SMI` (deep learning), `ml/Rhea2024_IFU-background-ML` (★ IFU with PCA and a neural field, the closest of all of them to this project), and `data-driven/Kolganov2023_NMF` (non-negative matrix factorisation).
> The remaining empirical and physical-model entries are summarised for context only. The classification is in [`sky-subtraction-papers.md`](sky-subtraction-papers.md).

---

## A. Machine learning and data-driven (read closely)

### A1. Zhang et al. 2025 — SMI: building the sky with a mutual-information network (`ml/`)
**Journal and source**: RAA, `arXiv:2508.19875`. Data: LAMOST multi-object fiber spectra. **The only deep-learning "predict the sky" paper.**

**The problem it sets out to solve**
- Current practice **averages** the sky fiber spectra into one "Super Sky" and subtracts that. The trouble is that an averaged spectrum **carries no information about the local environment around each target**, so it cannot capture the **spatial gradient** of the sky — and moonlight in particular is a gradient that shades across the field from one side to the other.

**The core idea**
- Use **every fiber across the whole field**, not only the sky fibers, together with deep learning and **mutual information (MI)**, to estimate a sky background **that belongs to each target's own position**.
- Split the sky into `Sl` (the continuum, taken as the median of the neighbourhood) + `Ssm` (the emission lines **shared** by the whole region) + `So` (the emission lines **unique** to that position). The model concentrates on the **emission lines**, since the continuum is easy to handle and varies little.
  - The observed spectrum: `O(i,λ) = [Oo + Sl + Ssm + So]·H` (H is the efficiency difference between fibers, normalised on the 5577Å airglow line, following Han 2023).

**The network architecture (two stages)**
1. **The feature extraction block**: a 1D convolution with a small kernel, plus a **calibration module** — anomaly detection realigns the emission-line features that the convolution has shifted (correcting the feature shift, without which the MI estimate goes wrong) — plus an activation function.
2. **The pretrained model**: trained on the data of the central 6 of the 16 spectrographs; the sky label is a spectrum with only the emission lines left (the continuum removed by the median of the neighbourhood, following Bai 2007); the loss is KL divergence (DeepInfoMax).
3. **Mutual information in two stages** (the MI is estimated by a MINE-like method, citing Belghazi 2018 / Bachman 2019):
   - **The first stage** **maximises** the MI between the representations of different spectra, which gives the **shared** common sky `Ssm`.
   - **The second stage** **minimises** the MI between shared and unique, which gives the **unique** emission lines `So` of that position.
   - The data is **split into segments adaptively by emission-line density**, and each segment has its own pretrained model.

**Results**
- Against LAMOST's "Super Sky", the SMI residual is **more concentrated on 0, with smaller MAE and RMSE and fewer outliers**, and **the improvement at the blue end is especially clear**; on some spectrographs the RMSE is close to **halved**.
- A few cases come out worse (spec04 of testplanid-2, for instance, which is put down to the MI extraction being incomplete or the sky fiber positions being off), but the RMSE is still smaller and steadier there.
- The residual at the emission lines is larger than across the spectrum as a whole. That is true of both methods; lines are hard to subtract.

**Limitations and future work**: no downstream task has been done yet (measuring stellar parameters after the subtraction); variation in the time domain is not included.

**What it means for this project**
- ✅ It belongs to **strategy 1 (predicting the sky)**, which is the direction the professor wants.
- The ideas worth borrowing: **"use every spaxel to estimate a sky specific to each position"** and **"separate the shared emission lines from the position-specific ones"**.
- ⚠️ It is **multi-fiber (LAMOST)** rather than an IFU; but a whole field of IFU spaxels is much like a great many fibers, so the concept carries over.

---

### A2. Kolganov, Chilingarian & Grishin 2023 — NMF sky subtraction (`data-driven/`)
**Source**: an ADASS 2023 conference note (4 pages, a "first version"), `arXiv:2312.06761`. Data: MagE/Magellan Echelle.

**Lineage**: it extends **Kurtz & Mink 2000**, which took hundreds of sky spectra, ran SVD/PCA on them and built that exposure's own sky **without needing an offset sky**.

**Two things that are new**
1. **NMF (non-negative matrix factorisation) in place of PCA**: sky flux is physically non-negative, which yields about **10 times** as many useful eigenspectra.
2. Extension to **2D (long-slit) spectra**: the extra spatial dimension makes it possible to **separate the sky continuum (flat along the slit) from the galaxy continuum (which peaks)** — something 1D cannot do, where the two are degenerate.

**The algorithm**
- First compute the NMF components from a set of sky spectra: `A ≈ W·C`, everything non-negative.
- Three assumptions: the sky is flat along the slit; the source's non-continuum features (emission, motion) occupy only a small part of the wavelength range; and the galaxy profile is approximately Moffat.
- **Step 1, remove the galaxy continuum**: bin in wavelength, where `F(y)=SKY+GAL(y)`; shift along the slit and self-subtract to cancel the flat sky, which leaves `GAL(y+Δy)−GAL(y)`; fit that with the difference of two Moffats; then rebuild and subtract the galaxy continuum (the bins holding emission lines are masked out).
- **Step 2, build the sky model**: collapse along the slit to raise the SNR, fit the NMF components by least squares as `s = Cᵀ·x + r`, and subtract the sky that reconstructs.
- The test: ~200 sky spectra give 20 NMF components; **it works even when the source fills the whole slit and there is no offset sky**.

**Future work**: enlarge the library of sky spectra; apply it to ESI/MagE/X-Shooter; and combine it with Kelson 2003 to reduce the interpolation noise on the airglow lines.

**What it means for this project**
- ✅ It belongs to **strategy 1**: a data-driven low-rank **model of the sky** rather than of the residual, and it needs no dedicated sky exposure.
- It sits between the PCA family (Kurtz, ZAP) and the physical models. Its **"separate the flat sky from the peaked source in 2D"** is exactly the "the sky is uniform across the field and the source is local" that we verified earlier.

---

### A3. Rhea et al. 2024 — IFU background reconstruction (PCA plus a neural field) (`ml/`) ★ the closest to this project
**Source**: `arXiv:2404.01175`. Data: **SITELLE** (the imaging FTS on CFHT, an IFU); tested on NGC 4449 (full of DIG, with Hα the main line) and NGC 1275 (the Perseus BCG). **The only paper that uses ML for background reconstruction on an IFU.**

**The problem it sets out to solve**
- Every spaxel of an IFU has the background sitting underneath it, so measuring a source's true flux means modelling that background and subtracting it first. The traditional global background (one average across the whole field) and local background (an annulus around the source) both go wrong when the source covers a large fraction of the field of view and the clean background is left only in scraps.
- ⚠️ "Background" here means **airglow lines plus astrophysical background and foreground plus noise**, not the sky alone; but the skeleton of the method is exactly the one that reconstructing the sky alone would use.

**The core idea**: **segment → PCA (which denoises) → interpolate the coefficients with a neural field → rebuild and subtract**. It uses the spatial dimension to interpolate the background beneath the spaxels the source covers out of the clean background around them.

**The chain**
1. **Segmentation** (photutils): bin the deep image (box 50×50) → a 3×3 Gaussian plus a sigma-clipped median estimates the background → subtract the background → convolve again → `detect_sources` (threshold 0.01× the background RMS) → background and source spaxels are separated. How accurate this step is decides how reliable the whole thing is.
2. **Normalisation**: normalise first on the maximum over 670–675 nm (no strong emission lines there, noise-dominated, and it carries the continuum level), then on the maximum of the whole spectrum; the same rule can be applied to the source spaxels so that the flux can be restored afterwards.
3. **PCA on the background spaxels** (sklearn incremental PCA): `s_r = μ + Σ α_i p_i`; k is kept at the turn of the scree (3 for NGC4449, 2 for NGC1275); the components carry physical meaning (Hα, [NII], [SII], negative DIG flux); and throwing the higher-order components away is what denoises.
4. **Neural field** (TensorFlow): **input (x,y) → output the k PCA coefficients**; 2 layers of 200/300 nodes with tanh, Huber loss, Adam at lr=1e-2 (lr×0.75 after 5ep without improvement on validation, early stop after 10ep without improvement, capped at 100); **99%/1% train/validation**, to maximise the spatial coverage, because the goal is only to learn the coefficient map of this one cube rather than to generalise; hyperparameters tuned with Optuna. What comes out is a smooth mapping from coordinate to coefficient that gives the source spaxels coefficients with no discontinuity, which beats linear or nearest-neighbour interpolation.
5. **Rebuild and subtract**: predict the coefficients at the source spaxels → restore the background as `μ+Σα·p` → rescale → subtract → then in LUCI fit 5 emission lines with tied velocities using a sinc, which is the FTS line shape. The background model moves only amplitude and flux; it does not touch velocity or dispersion.

**Results**: presented mostly as maps of the amplitude difference, **with no quantitative numbers**. On NGC4449 the traditional method overestimates in the centre while the new one recovers more flux out in the DIG region; on NGC1275 the traditional method underestimates at the outer edge; and the reconstructed background is visibly less noisy, especially outside the transmission region at 6300–6450 Å.

**Limitations**: segmentation error propagates (DIG leaks into the background); the linearity assumption is questionable (the explained variance is low, so most of it is noise, and linear may not be the right choice); and the neural field is smooth, which makes high frequencies hard to capture. **Future work**: add Fourier features to the input to catch the high frequencies, and model the stellar continuum instead.

**What it means for this project**
- ✅ **The most direct springboard**: segment → PCA denoising → neural-field interpolation → rebuild and subtract has already been made to work on an IFU, and the authors say explicitly that it generalises to **MUSE**, needs no library of sky templates and is independent of wavelength.
- Where we could differ: narrow "background" down to **the sky alone** and evaluate on **held-out blank** instead; replace linear PCA with something **non-linear, non-negative or mutual-information based**; and take up the future work Rhea lists (Fourier features to catch the high-frequency OH).
- It complements SMI and NMF: Kolganov and Rhea decouple using the **spatial dimension**; SMI separates shared from unique using **mutual information**; Zhang2016 writes the physics in as constraints, **non-negativity, sparsity and homogeneity**.

---

## B. Context summaries of the remaining predict-sky work (empirical and physical)

### empirical (traditional empirical and observational)
- **Kelson 2003**: subtracts the sky on the 2D image **before extraction and resampling**, making good use of what is known about the distortion and the LSF to sample the sky at sub-pixel scale. The founding work for subtracting the sky before extraction, and followed by much of what came after.
- **Noll 2014 — skycorr**: takes the airglow lines of one **reference sky** and **scales them group by group, grouped by physics**, to fit the airglow lines of the target spectrum (the continuum is separated out beforehand). The instrument-independent standard tool for scaling a sky model.
- **Law 2016 — MaNGA DRP**: ~92 sky fibers are combined into a **supersampled sky**, estimated on each fiber's own wavelength grid, scaled and subtracted (near the Poisson limit below 8500Å).
- **Sánchez 2016 — CALIFA**: builds the sky from the mean of the faintest sky fibers and subtracts it.
- **Glazebrook & Bland-Hawthorn 2001 — nod-and-shuffle**: nodding the telescope while shuffling charge on the CCD **measures the sky directly and along the same path** (~0.04%).
- **Rodrigues 2010 / accurate-sky-continuum (1302.3620)**: spatial reconstruction of the sky from many fibers, and precise subtraction of the sky continuum on a fibre.

### physical-model (physical and synthetic sky models, which generate a sky that can be subtracted)
- **Noll 2012 — Cerro Paranal sky model**: builds the whole sky by radiative transfer (scattered moonlight and starlight, zodiacal light, airglow lines and continuum); what ESO SkyCalc runs on underneath.
- **Jones 2013**: the scattered-moonlight component. **PALACE 2025 (Noll)**: a dedicated airglow model (26541 lines plus a climatology). **Patat 2008**: an empirical characterisation of how airglow varies.

---

## C. What all of this suggests for this project
1. **Two templates for an ML or data-driven "predict the sky"**:
   - **SMI** (deep learning, position by position, using every fiber, separating the shared from the unique emission lines with MI).
   - **NMF** (low-rank, separating the flat sky from the peaked source in 2D, with no offset sky needed).
2. **Both carry over to a MUSE IFU**: a whole field of IFU spaxels is much like a great many "fibers", so SMI's "estimate the sky position by position" and NMF's "separate in 2D" both apply naturally (which echoes the "the sky is uniform across the field and the source is local" that we verified).
3. **The common thread**: all of them handle the **continuum (smooth, shared)** separately from the **emission lines (variable, position-dependent)**, and **the emission lines are the hard part**, where the residual is largest. That agrees with the professor's "make the sky model itself accurate": what matters is the airglow lines and how they change with position.
