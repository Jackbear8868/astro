# Sky-subtraction literature, classified: predicting the sky against predicting the residual

> Compiled from 5 parallel search agents (physical sky models / empirical sky / PCA-residual / ML / a broad sweep).
> What the classes mean:
> - **Predicting the sky (strategy 1)**: model the sky itself, continuum plus airglow lines, either physically or empirically and data-driven, and then take `data − sky`.
> - **Predicting the residual (strategy 2)**: subtract roughly first, then model and remove what that subtraction left behind. Mostly PCA and data-driven.
> - **Mixed**: the pipeline uses both strategies at once. **Other**: related, but none of the above (line lists, telluric, ML denoising, observing strategy, reviews).
> - Every entry was confirmed by an agent to exist in an actual search result; the ones still to be verified are listed at the end.

---

## Category 1: predicting the SKY (strategy 1)

### 1a. Empirical and observational (traditional)
- **Kelson 2003** — Optimal Techniques in 2D Spectroscopy: Background Subtraction. `astro-ph/0303507` · PASP 115,688. Builds a 2D sky model directly and subtracts it before extraction; the founding work for subtracting the sky before extraction.
- **Glazebrook & Bland-Hawthorn 2001** — Microslit Nod-Shuffle Spectroscopy. `astro-ph/0011104` · PASP 113,197. Nodding the telescope while shuffling charge on the CCD **measures the sky directly** (~0.04%); it belongs to the family that obtains a sky and then subtracts it.
- **Davies 2007** — A method to remove residual OH emission from NIR spectra. `astro-ph/0612257` · MNRAS 375,1099. Groups the OH lines of a reference sky by their physics, rescales them onto the science frame and subtracts (the title says residual, but what it does at heart is **rescale a sky**). Adopted by SINFONI and KMOS.
- **Noll et al. 2014 — Skycorr**. `1405.3679` · A&A 567,A25. Davies generalised: separate lines from continuum, group OH and O₂ by variability, and fit a scaled reference sky to the science spectrum. The instrument-independent standard tool.
- **Law et al. 2016 — MaNGA DRP**. `1607.08619` · AJ 152,83. ~92 sky fibers are combined into one supersampled sky, then estimated on each fiber's own wavelength grid, scaled and subtracted (near the Poisson limit below 8500Å).
- **Sánchez et al. 2016 — CALIFA DR3**. `1604.02289` · A&A 594,A36. Builds the sky from the mean of the faintest PPak sky fibers and subtracts it, leaving a residual of ~1–5%.
- **Streicher et al. 2011 — MUSE pipeline sky subtraction**. ADS 2011ASPC..442..257S (no arXiv). MUSE estimates the sky from its faintest spaxels, with a sky-line LSF and supersampling.
- **Rodrigues et al. 2010** — new algorithm for sky extraction for multi-fiber. `1009.0554`. Reconstructs the spatial variation of the sky from the sky fibers, then subtracts it.
- **Han, Song & Zhao 2023** — Sky subtraction of LAMOST at bright night. MNRAS 526,5520 (no arXiv). Reconstructs the sky fiber by fiber with a weighted trend surface, which is what lets it handle the colour gradient of moonlight.
- (an extension) **"Accurate Sky Continuum Subtraction with Fibre-fed Spectrographs"** `1302.3620`.

### 1b. Data-driven and low-rank (traditional; NMF or low-rank models of the sky)
- **Zhang, Zhang & Ye 2016** — NMF with Sparsity sky model (LAMOST). PASA 33,e058. Models the sky with NMF plus sparsity, and compares it against B-spline and PCA.
- **Kolganov, Chilingarian & Grishin 2023** — NMF approach to sky subtraction. `2312.06761`. Replaces PCA with NMF so that the sky basis is non-negative, which yields ~10× as many effective components, and needs no offset sky.

### 1c. Machine learning (ML)
- **Zhang et al. 2025 — Sky Background Building via Mutual Information Network (SMI)**. `2508.19875` · RAA. Two networks, one for wavelength calibration and one that maximises mutual information, use **every** fiber to predict the sky at each object's own position and subtract it. LAMOST, with a clear improvement at the blue end. **The only pure deep-learning "predict the sky" paper, and it is fiber-fed.**
- **Rhea et al. 2024 — Reconstructing Robust Background IFU spectra using Machine Learning**. `2404.01175`. ★ **The only ML background reconstruction done on an IFU**, and the closest of all of these to this project. SITELLE FTS (NGC4449, which is full of DIG, and NGC1275). The chain is: photutils segments background spaxels from source spaxels → incremental PCA on the background spaxels (2–3 components kept by the scree turn, which denoises them at the same time) → **a neural field (input (x,y) → output the PCA coefficients; 2 layers of 200/300 tanh, Huber, Adam) interpolates those coefficients smoothly into the spaxels the source covers** → the background is rebuilt and subtracted, and the emission lines are then fitted with a sinc. Note that "background" here means airglow lines plus astrophysical background and foreground plus noise, not the sky alone; the method is independent of wavelength and needs no library of sky templates, and the authors state explicitly that it generalises to **MUSE**.

### 1d. Physical and synthetic sky models (traditional; they "generate" a sky spectrum outright, which makes them predict-sky)
> Note: these are sky models and databases rather than subtraction algorithms, but what they generate is a sky that can be subtracted directly.
- **Noll et al. 2012 — Cerro Paranal Advanced Sky Model (optical)**. `1205.2003` · A&A 543,A92. Models scattered moonlight and starlight, zodiacal light, and airglow lines plus continuum from the physics (LBLRTM radiative transfer).
- **Jones et al. 2013 — advanced scattered moonlight model**. `1310.7030` · A&A 560,A91. The scattered-moonlight component.
- **Noll et al. 2025 — PALACE v1.0 (airglow model)**. `2504.10683` · GMD 18,4353. 9 species, 26541 airglow lines and 3 continuum components, with a solar-cycle and seasonal climatology.
- **Patat 2008** — The dancing sky: 6 yr at Cerro Paranal. `0801.2270` · A&A 481,575. An empirical characterisation of how airglow varies, and the foundation Noll 2012 was built on.
- **Krisciunas & Schaefer 1991** — Model of the Brightness of Moonlight. PASP 103,1033. The classic analytic model of moonlight brightness.
- **Yoachim et al. 2016** — optical–IR sky brightness model for LSST. SPIE 9910. Builds a library of Rubin sky spectra from SkyCalc templates.
- **ESO SkyCalc** (a tool) — the web and CLI implementation of the Cerro Paranal model (citing Noll 2012 / Jones 2013).

---

## Category 2: predicting the RESIDUAL (strategy 2)

### 2a. PCA and SVD (traditional) — the mainstay
- **Kurtz & Mink 2000 — Eigenvector Sky Subtraction**. `astro-ph/0003112` · ApJ 533,L183. Iteratively subtracts an eigen sky and residual model derived by SVD; the source of the whole PCA-residual lineage.
- **Wild & Hewett 2005 — Peering through the OH-forest**. `astro-ph/0501460` · MNRAS 358,1083. Runs PCA on the OH residual of spectra whose sky has already been subtracted to build eigenspectra, then subtracts those spectrum by spectrum. The founding predict-residual work for fiber surveys.
  - Data release: **Wild & Hewett 2010** (SDSS DR7 with the residual removed) `1010.2500`.
- **Sharp & Parkinson 2010 — Sky subtraction at the Poisson limit**. `1007.0648` · MNRAS 408,2495. Points out that subtracting the sky from fibers leaves a systematic residual, and proposes a PCA residual procedure (long exposures beat nod-and-shuffle). It also compares strategies.
- **Soto et al. 2016 — ZAP (Zurich Atmosphere Purge)**. `1602.08037` · MNRAS 458,3210 · github.com/musevlt/zap. ★ **The standard residual-removal tool for MUSE and other IFUs**: after a first subtraction it filters the cube and runs segmented PCA, building a "sanitised" set of eigenspectra that catch only the sky residual and leave the source intact.
- **Marchetti et al. 2017 — VIPERS PCA cleaning/reconstruction**. `1612.01825` · A&A 600,A54. PCA in observed coordinates flags the sky-line residual, and PCA in rest coordinates then reconstructs and repairs it, across ~90,000 spectra.
- **Hart 2019 — Sky Residual Correction**. AJ 157,213 · DOI 10.3847/1538-3881/ab1a35 (no arXiv). Takes already sky-subtracted sky fibers as a training set, builds ~20 PCA components from them and subtracts the residual (SDSS/BOSS/APOGEE).
- **Husemann et al. 2022 — CARS / CubePCA**. `2111.10417` · A&A 659,A124. A simplified version of ZAP's PCA sky-line residual suppressor, with fewer parameters and more robust to the source content, used on MUSE.

### 2b. Bayesian and other data-driven approaches (borderline ML)
- **Uzsoy et al. 2025 — Bayesian Component Separation for DESI LAE**. `2504.06870`. With a data-driven prior it **infers the sky residual components and the LAE signal jointly**, marginalising over the residual rather than subtracting it outright. Conceptually predict-residual, but Bayesian rather than neural.

### 2c. Machine learning (ML)
- **(a gap)** — none of the five agents found a paper that uses deep learning **specifically to predict the residual left over after sky subtraction**. The closest work there is is classical PCA (ZAP / Wild&Hewett) plus the Bayesian method above (2b). **This is a clear gap in the literature**: an ML predict-residual method would be working in almost empty territory.

---

## Category 3: mixed (the pipeline uses both strategies)
- **Weilbacher et al. 2020 — MUSE DRP**. `2006.08638` · A&A 641,A28. The pipeline has its own sky model (continuum plus airglow lines including the LSF) that it subtracts (predicting the sky), and it recommends following that with **ZAP** (predicting the residual).
- **Guy et al. 2023 — DESI spectro pipeline**. `2209.14482` · AJ 165,144. Forward-models the sky fibers per petal first (spectro-perfectionism, predicting the sky), then applies a PCA residual correction to the strong airglow lines (predicting the residual).
- **Croom et al. 2021 — SAMI DR3**. `2101.12224` · MNRAS 505,991. Subtracts a master sky first, then takes a few principal components from the faintest ~10% of the fibers to minimise the sky-line residual.
- **Bai et al. 2017 — Sky Subtraction for LAMOST**. `1705.02079` · RAA 17,91. A B-spline supersampled master sky (predicting the sky), and then a PCA pass that corrects a further ~25% of the red-end OH residual (predicting the residual).

---

## Category 4: other (related, but neither predicting sky nor residual)

### 4a. Airglow emission line lists and atlases (the input data for the modelling)
- **Hanuschik 2003** — a UVES optical airglow emission atlas. A&A 407,1157 (2808 lines, R~45000).
- **Rousselot et al. 2000** — a NIR OH line list. A&A 354,1134 (4732 OH lines, 1.0–2.25µm).
- **Cosby et al. 2006** — nightglow line identification with UVES/VLT. JGR 111,A12307.
- **Oliva et al. 2015** — NIR airglow lines and continuum from GIANO-TNG. `1506.09004` · A&A 581,A47 (which revealed "hot-OH").
- **Viuho, Fynbo & Andersen 2025** — the NIR airglow continuum conundrum. `2506.02102` (FeO dominates the continuum).

### 4b. Telluric absorption correction (the sister problem, not sky emission)
- **Smette et al. 2015 — Molecfit I**. `1501.07239` · A&A 576,A77 (models atmospheric transmission by radiative transfer). (II: Kausch et al. 2015 `1501.07237`)
- **Sedaghat et al. 2023 — Stellar Karaoke**. `2301.00313` · MNRAS. A deep autoencoder blindly separates the atmospheric component out of ~250,000 HARPS spectra; the closest ML analogue to "learn the atmospheric component from the data itself".
- **Telluric autoencoder 2021** — Unsupervised spectral unmixing for telluric correction. `2111.09081` (authors still to be confirmed).

### 4c. ML spectral denoising, robustness to residuals, classification (ML, and all of it belongs under other)
- **Melchior et al. 2023 — SPENDER (Autoencoding Galaxy Spectra I)**. `2211.07890` · AJ 166,74. A convolutional autoencoder, deliberately designed to be **robust to skyline residuals**. (II: `2302.02496`)
- **Camilleri et al. 2025** — Emergent Denoising of SDSS Galaxy Spectra (unsupervised AE). `2510.08411`.
- **Denoising medium-res stellar spectra with U-Net 2025**. `2504.02523`.
- **Mukae et al. 2026** — CNN for Lyα Emitter ID in HETDEX. `2604.12414`. Tells a real LAE apart from "artifacts and sky residuals" rather than removing them.
- **MaNGA anomaly-detection autoencoder 2026**. `2603.03734`.
- **CNN–Transformer denoiser for low-S/N galaxy spectra 2026**. `2605.04434`.
- **Physics-informed super-resolution of galaxy spectra 2026**. `2603.18357`.
- **Vision Transformers for spectral analysis 2025**. `2506.00294`.
- **DESI DR2 pipeline QA with AI 2026**. `2606.21035` (it assesses how good a sky subtraction was, rather than performing one).

### 4d. Comparisons of observing strategy, and reviews
- **Rodrigues et al. 2012/2016** — On-sky tests of sky-subtraction methods (FLAMES). `1609.06142` · SPIE 8450 (cross-beam-switching and dual-stare, <1%).
- (Sharp & Parkinson 2010 also compares strategies; see 2a.)

---

## To be verified, or doubtful (look again before citing)
- **Subaru PFS sky subtraction** — SPIE 13096,130962M (2024), no arXiv; 2D-PSF forward modelling (predicting the sky). DOI 10.1117/12.3015628 still to be checked.
- **HETDEX/VIRUS** — a local amplifier-level sky model (predicting the sky); the instrument paper is `2110.03843` and the data release `2606.04208` (2026, provisional).
- **The 2026 ML arXiv entries (several of them in 4c)** — the titles and IDs were checked and do exist, but **the author lists are provisional**; confirm them before citing.
- **Bai 2008 — PCA sky-subtraction**. ChA&A 32,109 (2008ChA&A..32..109B). The author's full name is unconfirmed (probably the same Z.-R. Bai).
- **LAMOST 2D sky-background (PASA AS11071)** and **KICA-based LAMOST (IEEE 8564351)** — the articles exist, but their authors and dates are not fully confirmed.

---

## What this means for this project
1. **The predict-sky camp is mature and varied**: empirical (sky fibers, nod-shuffle, Kelson), data-driven (NMF), physical models (Cerro Paranal, skycorr), and 2 recent ML papers (SMI 2025 on fibers, and **Rhea 2024 on an IFU**, which is the closest to this project).
2. **Predicting the residual is all but synonymous with the PCA family**: it starts at Kurtz&Mink 2000 → Wild&Hewett 2005 → **ZAP 2016, the IFU standard**; the variants are CubePCA, Hart and VIPERS; DESI, LAMOST and SAMI all use it as a second pass inside the pipeline.
3. **A clear gap**: there is no paper on deep learning aimed specifically at predicting the residual (the nearest thing is the Bayesian Uzsoy 2025).
4. The professor wants to take the predict-sky route, so the main references are **skycorr (Noll 2014) + MUSE DRP (Weilbacher 2020) + MaNGA (Law 2016) + Kelson 2003**, with **Noll 2012 / SkyCalc** available as a physical sky prior.
