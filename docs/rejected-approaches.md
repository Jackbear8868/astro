# Approaches that were tested and rejected

What is recorded here is what was **tried, measured, and decided against**. It is kept so that
none of it gets proposed a second time.

---

## Excluding blank spaxels close to a small source (formerly `step3 --r-far-src`)

**What was done**: in the blank sample the sky is learned from, exclude every spaxel within
15 px of a source other than the main one. It was run end to end on p01, p04 and p08
(step3 + step5), with both sides measured.

**What came out**:

| | far-field scatter | far-field zero point | source preservation (`main 3-10`) |
|---|---|---|---|
| p01 | −5~7% | −0.033 → −0.057 | 0.761 → 0.765 |
| p04 | +6~20% | +0.022 → +0.018 | 0.622 → 0.628 |
| p08 | −8~9% | +0.027 → +0.028 | 0.645 → 0.689 |

It cuts 30–48% out of the region the sky is learned from, and every difference it buys sits in
the third decimal place. In the mean spectrum of each ring, the two curves — with the option on
and with it off — overlap almost exactly across the whole of 4600–9350 Å (measured at the time
with `evaluation/zone_spectra.py`; that script and the whole ring criterion were deleted on
2026-08-18), while on the same figure the gap between ESO's result and ours is more than a
factor of 10.

**Why it was rejected**: the effect is smaller than the measurement precision, and the three
pointings do not even agree on its direction, so there is no way to tell a real mechanism from
noise. p04's far field holds only 1,204 spaxels — 1/5 of what the other two have — which makes
its scatter estimate unreliable in the first place.

**The asymmetry it leaves behind**: `step5 --sf-r-far 15` is still switched on — the spatial s
field trains only on spaxels more than 15 px away from a source, while the sky basis has no
such protection. The two address the same worry, and only half of it is now done. Reopening
this subject means asking whether that asymmetry ought to exist, not whether this flag ought to
come back.

---

## Three long-running questions (closed 2026-08-18)

These three used to sit in this file as open questions, and the decision has been taken not to
pursue them any further. The headings and the conclusions stay to explain why they are not on
the to-do list, not to reopen them.

**What DETECT_THRESH should be for SExtractor** — no more running our own threshold
experiments; the working pipeline uses the segmentation the project was given instead.

**The far-field residual is 1.7–2.2× the photon noise** — judged not to be a problem.

**The additive bias in the source template amplitude (about −0.044)** — the current approach is
good enough, and the mechanism is not being chased any further.
