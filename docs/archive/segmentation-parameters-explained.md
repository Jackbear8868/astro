> **Archived on 2026-08-27. This is not how the pipeline works now.**
> It reasons from first principles about how to choose detection parameters for
> ourselves. That question was closed on 2026-08-18: the pipeline does not detect
> sources at all, it is given a segmentation map (`src/skymodel/README.md`), and the
> thresholds below are no longer measured or applied.
>
> **Two things are out of date if you read it as current.** The seeing it is built on,
> 1.2" = 6 px, was superseded by a measurement of **0.81" = 4.06 px** -- `CLAUDE.md`
> calls the earlier figure an unverified assumption -- and every value derived from it
> moves with it: kernel width 6 px to about 4, minarea 30 px to about 13, dilation 6 px
> to about 4. The 2 sigma threshold and the `bw >= 256` argument still hold. The code it
> points at, `cmd_mask` in `src/run_zap_compare.py`, no longer exists; the nearest
> surviving code is `src/zap/mask.py`.
>
> What it is still worth reading for is the reasoning, which does not depend on the
> numbers: why the kernel width should equal the seeing, why minarea is about one PSF
> area, why the background box has to be larger than the object, and why sigma must be
> estimated away from the sources.

# Segmentation parameters explained (why they are set this way)

> This document explains, from the ground up, the **physical reasoning** behind every
> parameter of the source mask (segmentation).
> It assumes the reader is not familiar with astronomical imaging, so every term is
> explained before it is used.
> The corresponding code is `cmd_mask` in `src/run_zap_compare.py`; for a summary see
> `CLAUDE.md`.

---

## 0. First, what we are doing and why a mask is needed

### 0.1 The wider picture -- sky reconstruction
Our goal is to **subtract the sky**. Every pixel the telescope records is **light from the
object + light from the sky + noise**. The sky (airglow, scattered moonlight and so on) is
what we **do not want**, and it has to be subtracted off so that only the object is left.

The method (sky reconstruction): **first learn what the sky looks like from the regions
that hold only sky and no object, then subtract that from the whole image.**

### 0.2 Why a source mask is needed
To "learn the sky from the regions that hold only sky", you first have to know **where the
objects are and where it is pure sky**.
- A **source mask** is a label image: pixels holding an object are marked 1, pure sky is
  marked 0.
- Only the places marked 0 are used when learning the sky; the object regions marked 1 are
  "masked out" and take no part.

**If the mask is not done well** (if a place that holds an object is taken for pure sky),
the "sky" that comes out has object light mixed into it → subtracting the sky subtracts
part of the object as well → and the science signal we are after is destroyed. **The
quality of the mask therefore decides success or failure directly.**

### 0.3 What makes this field (Haro11) difficult
Haro11 is a galaxy with **extended ionised gas over a large area** -- besides the bright
central core there is a large, faint, diffuse **Hα halo** around it.
- Every pixel of that halo is **very faint** (low surface brightness), faint enough to be
  close to the noise.
- But it is **real object light**, and it has to be masked, or it will contaminate the
  sky.
- **How to mask a halo that is both large and faint, reliably**, is exactly the core
  problem that all of these parameters exist to solve.

---

## 1. A few terms you will need (read this section first)

| term | in plain words | value for this data |
|---|---|---|
| **pixel** | the smallest square of the image | — |
| **pixel scale** | how large an angle on the sky one pixel covers | **0.2 arcsec/pixel** |
| **arcsec (″)** | the unit of angle on the sky. 1 degree = 3600 arcsec | — |
| **PSF (point spread function)** | what a point source (a star, say) looks like once the atmosphere and the telescope have blurred it out. Ideally a point, in practice a small blob | — |
| **seeing** | how much the atmospheric turbulence blurs the image, measured by the width of the PSF. **Smaller is sharper** | **≈1.2 arcsec** |
| **FWHM (full width at half maximum)** | the standard way of measuring "how wide a blob is": the width at the point where the brightness has dropped to half | PSF FWHM ≈1.2″ ≈ **6 pixels** |
| **noise / σ** | the random fluctuation of each pixel. σ is its standard deviation (the typical size of the fluctuation) | — |
| **S/N (signal-to-noise ratio)** | signal ÷ noise. The higher it is, the easier something is to make out | — |

**The key conversion**: seeing 1.2″ ÷ 0.2″/pixel = **6 pixels**. That "6 pixels" is used
over and over again below.

---

## 2. Every parameter, one at a time

### 2.1 Detection threshold, threshold = 2σ

**What it is**: we sweep across every pixel and ask "is the brightness here clearly above
the noise?". A threshold of 2σ means "**it has to be 2 times the noise fluctuation (σ)
above the background before it counts as a detection**".

**Why there has to be a threshold**: a pure sky region also goes up and down because of
noise. If the threshold is too low, random high points of the noise get taken for
"sources" too.

**Why 2σ**: this is the statistical "false positive rate" (the probability of taking pure
noise for a source):

| threshold | probability that pure noise exceeds it | verdict |
|---|---|---|
| 0.75σ | **23%** | too low → this is essentially collecting noise, **physically indefensible** |
| 1.5σ | 6.7% | usable, but on the low side |
| **2σ** | **2.3%** | **standard and robust** (2–5σ is the usual range in astronomy) |

→ **2σ says "there is only a 2.3% chance that this brightness came from pure noise"** --
credible enough, and the standard of the field, which is why it is the one we use.

> ⚠️ A common mistake: pushing the threshold all the way down to 0.75σ in order to catch
> the faint halo. But 0.75σ is below the noise itself (23% false positives), which amounts
> to "collecting noise" and cannot be defended. **The right answer is not to lower the
> threshold, but to first use the matched filter described below to bring the halo out
> clearly, and then use a normal 2σ.**

---

### 2.2 Matched filter, kernel width = seeing (FWHM ≈ 6 pixels)

This is the **core trick** of the whole method, and it comes in three steps.

**(a) What "smoothing" is**
Blurring the image: each pixel is replaced by **a weighted average of the neighbours
around it**. It is like looking at something short-sightedly, or through half-closed eyes
-- the detail is smoothed away.

**(b) Why smoothing makes a faint halo "surface"**
What matters is that "noise" and "halo" respond to averaging differently:
- **Random noise**: neighbouring pixels are positive and negative and independent of one
  another → after averaging they **cancel each other out and shrink** (average N pixels
  and the noise drops by roughly √N).
- **An extended halo**: neighbouring pixels are all part of one smooth signal, all in the
  same direction → after averaging it is **almost unchanged**.

The result: **the noise is pushed down and the halo is kept → the halo's S/N goes up.**
What was "halo < noise" in each pixel (invisible) becomes "halo > noise" after smoothing
(visible) → and then a normal 2σ threshold does catch the halo.

**(c) What "matched" means, and why the kernel width = seeing**
- The **kernel** is the small patch of weights the smoothing uses (a Gaussian bell: heavy
  in the middle, light around the edge).
- How wide should the kernel be? The theory (the matched filter theorem) says: **the gain
  in S/N is largest when the scale of the smoothing kernel "matches" the scale of the
  signal you are looking for.**
- For point sources, and for detection in general, the standard practice is to **set the
  kernel width = seeing (PSF FWHM)**. The reason:
  - **The seeing (6 pixels) is "the smallest scale of real structure in the image"** --
    variation finer than 6 pixels cannot possibly be a real object (everything has been
    blurred to 6 pixels by the atmosphere), so it has to be noise.
  - Smoothing at that scale of "6 pixels" is therefore the best point available: it
    **suppresses the noise as far as it can go without blurring out any real structure**.
    Too small a kernel → the noise is not suppressed enough (and the halo is still not
    caught); too large a kernel → even the shape and position of real structure is blurred
    away.

→ So **the matched filter's kernel width = seeing FWHM ≈ 6 pixels** is the choice with the
strongest physical grounding.

---

### 2.3 The background box has to be bigger than the halo → bw = 256 pixels

**What the background is**: before detecting sources, the "baseline sky brightness" has to
be estimated and subtracted. The way Source Extractor does this is to cut the image into a
grid of boxes, compute one "local background value" for each box, and subtract it. The
size of the box is called **`bw` (BACK_SIZE)**.

**Why the size of the box matters so much**:
- Think of the box as estimating "how bright it normally is around here".
- **If the box is smaller than the halo**: the box is soaking entirely in the halo, so it
  concludes "it is just this bright here" → it estimates **the halo itself as the
  background** → subtracts it → **the halo disappears** and is not detected. (No threshold,
  however low, helps, because the halo has already been subtracted as background.)
- **If the box is bigger than the halo**: every box holds, besides the halo, **enough of
  the true sky around the halo** to estimate a true background baseline → and then the
  halo stands clearly above the background and is detected.

**The rule**: the background box has to be **bigger than the object you want to keep**.
Here the halo's diameter is ≈ 226 pixels, so the box is set to **bw = 256 pixels** (bigger
than the halo).

> This also explains something that was seen earlier: Source Extractor's **default is
> bw = 64** (far smaller than the 226-pixel halo) → so it eats the halo as background →
> and no amount of adjusting the threshold catches the halo. **The real problem was the
> background box, not the threshold.**

---

### 2.4 minarea (minimum area) = 1 PSF area ≈ 30 pixels

**What it is**: how many connected pixels a detection has to be made of **at the very
least** before it counts as "a real source". Anything smaller than that is thrown away.

**Why it is needed**: even with the matched filter, a pure sky region will still throw up
the occasional few scattered pixels that happen to exceed the threshold (those 2.3% false
positives). These are **isolated little specks of noise**, not real sources.

**Why it is set to ≈ 1 PSF area (≈30 pixels)**:
- Any **real** object, having been blurred out by the PSF, will fill at least "1 resolution
  element" = the area of 1 PSF.
- PSF FWHM ≈ 6 pixels → PSF area ≈ π×(6/2)² ≈ **30 pixels**.
- So "a detection smaller than 30 pixels" cannot be a real source blurred out by the PSF →
  it is judged to be noise and thrown away.
- → **minarea = 30 pixels**, and the physical grounding is simply "a real source has to be
  at least one PSF across".

---

### 2.5 dilation = grow outwards by ≈ 6 pixels (≈ seeing)

**What it is**: once the detection is done, the boundary of the mask is **expanded
outwards by a few rings of pixels**, as a safety margin.

**Why it is needed**:
1. **The wings below the threshold**: the outermost ring of the halo is just below the
   threshold and was not detected, but it is **still real source light**. Left alone, that
   ring is treated as pure sky → contamination.
2. **PSF spill-over**: the light of a bright source is blurred by the PSF a few pixels
   beyond the boundary.

**Why it is set to ≈ 6 pixels (= 1 seeing)**: the light of the PSF reaches out to roughly
1 FWHM (6 pixels). Growing outwards by 6 pixels therefore takes in exactly that ring of
source light that slipped through the net, without expanding so far that sky samples are
wasted.

---

### 2.6 (An important trap) The noise σ must be estimated from source-free regions

The threshold above is given as "2σ", but **how is that σ (the size of the noise) to be
computed**?

- **The wrong way**: computing σ over the whole image, halo included. Because the halo is a
  large, somewhat bright area, it stretches the "fluctuation" out → σ is overestimated →
  the 2σ threshold becomes too high → and the halo is missed after all.
- **The right way**: **estimate σ only from "the regions with no source in them"**
  (statistically this is called sigma-clipping: first exclude the obviously bright pixels,
  then compute the fluctuation of what is left). Source Extractor does this internally.

→ This is the classic trap of matched filter detection: **σ has to be estimated from clean
sky, or the halo will raise the threshold itself and hide itself away.**

---

## 3. One summary table: parameter → the physical scale it matches

| parameter | value | physical quantity it matches | reason in one sentence |
|---|---|---|---|
| detection threshold | **2σ** | the size of the noise σ | only 2.3% false positives, standard and defensible |
| matched filter kernel width | **6 px** | seeing / PSF FWHM (1.2″) | smoothing at "the smallest scale of real structure" suppresses the most noise without blurring real structure |
| background box bw | **256 px** | > the halo's diameter (226 px) | only a box bigger than the halo avoids eating the halo as background |
| minarea | **30 px** | the area of 1 PSF | a real source is at least one PSF across |
| dilation | **6 px** | 1 × seeing | makes up the halo's wings plus the PSF spill-over, as a safety margin |
| how σ is estimated | after excluding the sources | the noise of clean sky | otherwise the halo raises the threshold and hides itself away |

**The principle running through all of it**: **every parameter is matched to a physical
scale of the data itself (seeing, PSF, noise), rather than being picked at random or
picked to make a number look good.** That way every single one of these numbers can be
explained clearly to the professor or to a referee.

---

## 4. Why not do it some other way? (counter-examples)

| what someone else might do | the problem | what we do instead |
|---|---|---|
| push the threshold down to 0.75σ to catch the halo | below the noise (23% false positives), physically indefensible | use a matched filter to bring the halo out clearly, then a normal 2σ |
| use Source Extractor's default (bw=64) | the box is smaller than the halo → the halo is eaten as background → the halo is not caught | bw=256 (bigger than the halo) |
| dilate alone, without filtering out the small blobs | scattered noise points get inflated into patches as well | filter the small specks out with minarea first, then dilate |
| use a morphological opening to remove the noise | opening **erodes** the outer edge of the faint halo | delete by area instead (minarea), which does not shrink the halo |
| compute σ over the whole image | the halo stretches σ out → the threshold is too high → the halo is missed | estimate σ after excluding the sources (sigma-clipping) |

---

## 5. In one sentence

**To mask a large, faint Hα halo reliably, you cannot rely on "lowering the threshold"
(which collects noise and is physically indefensible). Instead:**
1. **set the background box large** (so the halo is not taken for background),
2. **use a matched filter (kernel = seeing) to pull the halo's S/N up**,
3. **then apply the standard 2σ threshold**,
4. **use minarea (= 1 PSF) to remove the noise, and dilation (= seeing) to make up the
   edge**,
5. and **estimate σ from clean sky** throughout.

Every step is matched to a physical scale of the data, which is why every parameter stands
up and can be explained.
