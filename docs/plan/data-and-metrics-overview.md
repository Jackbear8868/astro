# Sky subtraction / sky reconstruction: datasets, data sources, evaluation metrics (a conceptual overview)

> A conceptual reference, and nothing more. It explains three things:
> 1. **Data sources** -- where the sky samples and the labels come from;
> 2. **Datasets** -- what train / val / test each are, and which kind of source can serve as which;
> 3. **Evaluation metrics** -- what yardstick measures how good a result is, with a ground truth and without one.
>
> This document holds nothing about what to do, in what order, which method to pick or what is still outstanding. It sets out the concepts and how they stand to one another.

---

## 0. The common premise: what a ground truth actually is in sky subtraction

### 0.1 The physical reality: there is no true sky underneath a source
The fundamental difficulty of sky subtraction is that **the true sky directly beneath a given source, at that moment, can never be measured**, because the source itself sits in the way along that line of sight. What is observed is the sum of source + sky + noise, and a single real observation **cannot** be taken apart into the pure sky at that position.

### 0.2 The three roles of a ground truth (GT), which are not to be confused with one another
| Role | What it is | Where it comes from |
|---|---|---|
| **The training GT** | The supervision signal while the parameters are learned | Most methods (PCA / NMF / autoencoder) are **unsupervised** and need no GT at all -- a pile of sky-only spectra is enough to learn a basis from; the few supervised ones take their labels from **sky-only regions** as well |
| **The evaluation GT** | The reference for comparing the reconstructed sky vs the true sky | Sky-only regions (a blank spaxel, a sky fibre): there the observation **is** the sky |
| **The underneath-the-source GT** | The true sky at the position the source occupies | Real data cannot supply it; **only simulation or injection can** (a known sky synthesised for the purpose) |

### 0.3 Two orthogonal axes: the material vs the yardstick
- **Axis one | where the data comes from (the material) = sources A / B / C**. This is what "data source" genuinely means.
- **Axis two | what criterion the scoring goes by (the yardstick) = the metrics**. **A metric is a set of scoring criteria, not data.**

> One yardstick measures any material: a real blank, an injected sample, a mock, even the raw observation -- the same set of metrics applies to all of them. Metrics are therefore not an option standing alongside A/B/C, but something laid **on top of** them. Section 1 of this document is about the material, section 2 about the roles within a dataset, and section 3 about the yardstick.

---

## 1. Data sources (the material: where the sky samples and labels come from)

### 1.1 The kind of data decides, before anything else, what a sky sample looks like
| Aspect | **IFU (datacube)** | Multi-fibre | Long slit |
|---|---|---|---|
| Structure | **(x, y, λ)**, a complete 2D spatial field × wavelength | N **discrete** 1D spectra, scattered over the field | A 2D image = **1D space along the slit** × λ |
| The sky sample | Blank spaxels, **contiguous and numerous** | Dedicated sky fibres, **sparse** (a few, to a few tens) | The rows of the slit that hold no source |
| Spatial relation | Neighbouring spaxels are highly correlated → the sky can be interpolated spatially | The positions are sparse and discontinuous | The sky along the slit varies ≈ slowly |
| One datum | One cube = one exposure, holding tens of thousands of spaxels | One plate holds thousands of fibres, and there can be many plates | One exposure on one source |
| Common extra asset | The per-voxel variance (variance / STAT) | Each fibre's throughput has to be normalised | — |

> The point is not which of them is better. It is that **the kind of data fixes how many sky samples there are, how dense they are, and how continuous they are in space** -- which in turn decides whether interpolating the sky from its surroundings to somewhere else is possible at all.

### 1.2 Source A | a sky-dominated region is the sky (a blank spaxel, a sky fibre)
- **How it works**: the observed spectrum of a sky-only region is taken directly as a sky sample.
- **What it gives**: it is free, it is **real** (a real LSF, real OH lines, real noise), and there is a great deal of it.
- **Limits**: it holds only where there is no source → **it gives no reference whatsoever for underneath a source**; and the sample is sky with noise on it, not clean sky.
- **Can it test underneath a source**: ✗.
- **Relative effort**: low, since the data is already there.

### 1.3 Source B | injection, or semi-synthetic data (a real sky on a real clean source)
- **How it works**: a **known true sky**, taken from a blank, is laid on top of a **known clean source region**, giving a sample whose answer is known.
- **What it gives**: it has a **known answer** and it uses a **real sky**, which is more realistic than a full simulation; and the sky can be injected at a position that does hold a source → **it reaches underneath the source**.
- **Limits**: the clean source underneath it is not itself perfect and may carry residual; the addition **counts the noise once over**; and if the injected sky does not vary with position, nothing is measured about the ability to follow a spatial gradient.
- **Can it test underneath a source**: ✓.
- **Relative effort**: medium.

### 1.4 Source C | a full mock (synthesised from scratch, underneath the source included)
- **How it works**: a synthetic cube = a known source + a known sky (a physical sky model plus the instrument's LSF) + realistic noise. **Every spaxel, the ones underneath the source included, has a known true sky.**
- **What it gives**: it is **the only thing that supplies a point-by-point truth underneath a source**, and it is entirely under control -- the moonlight, the gradient and the OH strength can all be set at will.
- **Limits**: it costs the most work; it has a realism gap, in that too simple a simulated sky makes the result less convincing; and it needs a forward model of the instrument (the LSF, the throughput, the correlated noise).
- **Can it test underneath a source**: ✓.
- **Relative effort**: high.

### 1.5 The three sources at a glance
| Source | Realism | A known answer? | Tests underneath a source? | Relative effort |
|---|---|---|---|---|
| **A** sky region = sky | The most real | No, it is only an observation | ✗ | Low |
| **B** injection / semi-synthetic | High, since the sky is real | ✓ | ✓ | Medium |
| **C** a full mock | It depends on the model | ✓, point by point | ✓ | High |

---

## 2. Datasets (the roles of train, validation and test)

### 2.1 What the three roles are
- **train**: used to learn the parameters, the basis and the network weights.
- **validation**: used to tune the hyper-parameters and to decide where to stop -- how many components to keep, when to stop early -- and it takes no part in the final report.
- **test**: entirely held out, and used for the final, unbiased report of the score.

### 2.2 Which roles each source can fill
| Source | train | validation | test | Tests underneath a source? |
|---|---|---|---|---|
| **A** a real blank | ✓, learning the basis or the network | ✓, a held-out blank | ✓, another blank → a genuine reconstruction error | ✗ |
| **B** injection / semi-synthetic | Optional, and rarely used as train | ✓ | ✓, a controlled probe | ✓ |
| **C** a mock | ✓, the only one that gives a label underneath a source | ✓ | ✓, a point-by-point truth | ✓ |

- **A** can fill all three roles, but **only over the blank region**.
- **B** is in practice mostly test or validation, as a controlled probe, and rarely train, because what lies underneath it is not clean enough.
- **C** can fill all three roles, and it is **the only train/val/test that covers underneath the source**.

### 2.3 Why only B and C can evaluate underneath a source
Evaluating underneath a source needs a position that **holds a source and a known sky at the same time**. Real data cannot supply one, since a blank has no source in it by definition. Only two things can:
- **B**, by laying a known sky onto a real source and so manufacturing such a position;
- **C**, where even underneath the source there is a synthesised, known sky.

So **A cannot test underneath a source on any kind of data**. This is also where a method is most liable to over-subtract or under-subtract, and the place the literature quantifies least.

### 2.4 One property of the data to watch: spatial correlation
The noise of neighbouring IFU spaxels is highly correlated. Splitting the spaxels into train and test **at random** can leave a test sample's neighbour sitting in the train set, which leaks information and makes the error look smaller than it is. This correlation is therefore a property of the data that has to be borne in mind when the split is made; the usual treatment is to split into spatial blocks rather than point by point.

---

## 3. Evaluation metrics (the yardstick: what measures how good a result is)

The metrics are independent of the data source -- **one set of metrics applies to A, B, C or the raw observation alike**. A metric can also serve as an unsupervised training loss: "the smallest blank residual with the source flux left alone" is a natural loss in itself. They fall into three layers, according to whether a ground truth is needed:

### 3.1 (a) Truth-free metrics (no GT needed; they can be computed on real data)
| Metric | What it measures | Better is | How it is usually shown |
|---|---|---|---|
| **The blank residual std(λ)** | Whether the blank region is flat once the sky is subtracted | Low | A line plot of std vs λ, with several methods overlaid |
| **The sky-line residual** | The residual at 5577, at 6300, at OH 8400 and the like | Low | A zoomed line plot, or a table |
| **Source flux fidelity** | Whether the emission lines (Hα, [OIII]) are unchanged before and after | They overlap | A before/after overlay |
| **The distribution of the residual** | Whether the residual is concentrated on 0 | mean≈0, with little scatter | A histogram |
| **The ratio to the noise floor** | The residual std vs the theoretical noise floor (Poisson / variance) | Close to 1 | A line plot of the ratio vs λ |
| **A 2D image of the residual** | Whether the residual has spatial structure in it | No coherent structure | The residual image at one sky line |

> What they are like: always available, at no cost in ground truth, and computed on real data. But **they measure only how clean the residual is, and not directly whether the true sky was recovered**, and they can be misled by over-subtraction flattening the residual, or by over-fitting eating the source flux.
>
> ⚠️ **How the noise floor is to be taken -- an open question, not a settled correction.** The standard deviation of DATA over a blank region divided by √STAT was measured at about 1.8, which would say that MUSE's per-pixel variance underestimates the true noise by that factor because the data reduction correlates the noise of neighbouring pixels. Other work on the same instrument puts the factor at about 1.5. **Neither figure is confirmed for this data, and the project takes STAT at face value, with an ideal chi of 1.** A ratio that puts bare √STAT in the denominator therefore has whatever distortion that correlation causes still in it; using the measured blank scatter instead avoids the question entirely, and is the safer of the two when the choice is available. Applying a correction factor is a scientific decision that has not been made, not a step to be taken silently.

### 3.2 (b) GT-based metrics (they need a held-out blank (A), or B or C)
| Metric | What it measures | Better is | How it is usually shown |
|---|---|---|---|
| **The reconstruction RMSE / MAE** | The predicted sky vs the true sky | Low | A comparison table, or a line plot |
| **The reconstruction error underneath a source** | The reconstruction error at the source's position (it needs B or C) | Low | A comparison table |
| **bias (a systematic offset)** | mean(pred − true), which is the average over- or under-subtraction | ≈0 | A table, split by blank and by source position |
| **The error in the lines vs the continuum, separately** | Whether the error comes from the sky lines or from the continuum | Low | A table, or a line plot |

### 3.3 (c) Downstream, or scientific, metrics (the ultimate acceptance test)
| Metric | What it measures | Better is | How it is usually shown |
|---|---|---|---|
| **The recovered source quantity** | Whether the Hα flux and the kinematics are right once the sky is subtracted | Close to the truth, or to the reference | A flux map, a kinematics figure |
| **The improvement in S/N** | The gain relative to the baseline | High | A table, or a line plot |

---

## 4. A comparison: where earlier work falls within this framework

### 4.1 Kind of data × source / metric, cross-tabulated
| | A (sky region = sky) | B (injection) | C (mock) | Metrics (the residual yardstick) |
|---|---|---|---|---|
| **IFU** | Rhea (background spaxels, held out) | — | — | Rhea (difference images, qualitative) |
| **Multi-fibre** | SMI (a sky fibre → a pseudo-label), W&H (a sky fibre ought to be zero) | — | Zhang16 (the LAMOST simulator) | W&H, Zhang16, SMI |
| **Long slit** | (the sky rows on the slit) | Kurtz&Mink (inject a sky → recover the redshift) | — | Kolganov (by eye) |

The pattern that comes out of it:
- **A** is the most widespread source of training data; anyone who can cut out a sky-only region uses it.
- **B** turns up in work that needs a test with a known answer, as in Kurtz&Mink's redshift recovery.
- **C** turns up where a simulator happened to be at hand already, as with Zhang16's LAMOST simulator.
- **Metrics** are what almost everyone reports a result with.

### 4.2 Whether the kind of data restricts which source or metric can be used
In principle A / B / C and each of the metrics can be implemented on all three kinds of data, and nothing is shut off outright. What the kind of data changes is **how much data there is** and **how hard it is, and how much it is worth**:
- **A**: all three can do it. The IFU has the most samples and the densest, the fibres the sparsest; the difference is one of quantity.
- **B**: all three can do it. The fibres already have a precedent for it, and it is just as feasible on an IFU.
- **C**: all three can do it, provided the instrument's forward model can be built. It is harder to build for an IFU, but it gives a point-by-point truth underneath a source in 2D.
- **Metrics**: all three can do it, and they are always available.

The one thing the kind of data genuinely decides is not the source itself. It is that **underneath a source can be evaluated only by B or C**; and that **only an IFU, having a complete and continuous 2D space, makes it meaningful to speak of interpolating the sky from its surroundings to underneath the source** -- discrete fibres cannot do it.

### 4.3 The metric each paper actually presents (from a close reading of the originals)
| Paper | Data | The metric presented | How quantitative | Needs a GT? |
|---|---|---|---|---|
| **Soto 2016 (ZAP)** | MUSE IFU | A before/after spectrum overlay, a variance vs component-count curve (to choose neval), the sky-line residual plus source fidelity | Mostly qualitative | No |
| **Husemann 2022 (CubePCA)** | MUSE IFU | A single before/after overlay of the co-added spectrum (Fig 3); the residual bands are masked out downstream | **Purely qualitative, with no numbers** | No |
| **Rhea 2024** | SITELLE IFU | Difference images, flux recovery compared between methods, a reconstructed background noise said to be "lower", a scree plot to choose the components | **Purely qualitative; the 99/1 held-out split is never used to report a reconstruction error** | No |
| **Wild & Hewett 2005** | SDSS fibres | The blank residual std(λ), robust at 67%, the **noise-weighted residual → the Poisson floor**, a component count stopped where the sky/non-sky ratio = 1, source EW fidelity (a table), the detection χ² (−27%), the improvement in S/N (~20%) | **Amply quantitative** | In part, since the EW is a comparison |
| **Sharp & Parkinson 2010** | Fibres | The **local error (as a % of the sky) vs the Poisson floor**, a histogram of the residual, **the rate at which the residual falls with N (the √N test)**, the redshift recovery rate (85% vs 25%) | **Amply quantitative** | The redshift one does |
| **Kurtz & Mink 2000** | Fibres | **An injection test → the redshift recovery rate**, vs the Poisson limit (0.7 mag short, a factor of ≈ 2), the cross-correlation r value | Quantitative, mostly on redshift | **Yes, by injection** |
| **Zhang 2025 (SMI)** | LAMOST fibres | **MAE / RMSE**, a residual concentrated on 0, few outliers, an improvement at the blue end | **Quantitative** | For the sky samples |
| **Kolganov 2023 (NMF)** | Long slit | ~200 sky spectra → 20 components; mostly a demonstration of feasibility | Mostly qualitative | No |

The pattern that comes out of it: **the three IFU papers (ZAP / CubePCA / Rhea) evaluate qualitatively**, mostly with before/after overlays and difference images, while **the fibre surveys (W&H / Sharp&P / Kurtz&Mink / SMI) are the most quantitative**, and they return again and again to two shared benchmarks -- **the Poisson noise floor** and **the redshift recovery rate of an injection test**.

---

## 5. Strategy × metric, and which applies to which (the metrics that can only be computed on some strategies)

| Kind of metric | A (blank) | B (injection) | C (mock) | Note |
|---|---|---|---|---|
| Truth-free residual metrics (std / line residual / histogram / 2D image / the ratio to the noise floor) | ✅ | ✅ | ✅ | They can be computed on any data, but **they say only how clean the residual is, and are entirely blind to underneath the source** |
| The reconstruction RMSE / MAE (vs the true sky) | ✅ (on a blank) | ✅ | ✅ | It needs a known true sky; on A it can only be computed over the blank |
| Source flux / EW fidelity | ✅ (a reference is needed) | ✅ | ✅ | A comparison; a source and a reference are all it takes |
| **The reconstruction error and bias underneath a source** | ❌ **physically impossible** | ✅ | ✅ | **The one kind the strategy decides outright**: A's blank has no source in it by definition |
| Downstream scientific recovery (flux / redshift / kinematics) | ❌ (no source) | ✅ | ✅ | It needs a known scientific answer |
| The rate at which the residual falls with N (the √N test) | ✅ (on a blank) | △ | ✅ | Independent of the strategy, but it needs the data volume: many exposures, or binning |

Two conclusions follow:
1. **Truth-free residual metrics can be computed on A, B and C alike**, but they cannot see underneath the source -- subtracting cleanly is not the same as subtracting correctly, since over-subtraction flattens a blank as well.
2. **Only B and C can compute the reconstruction error underneath a source and the downstream scientific recovery**, and that is exactly the kind the literature -- the three IFU papers above all -- quantifies least and leaves emptiest.
