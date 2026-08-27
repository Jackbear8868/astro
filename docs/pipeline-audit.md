# Open items in the pipeline (audited 2026-08-18)

A full audit of `run_pipeline.py`, `step1`–`step6`, `utils.py`, `templates.py` and
`src/skymodel/evaluation/` turned up 25 problems. **24 of them have been fixed and
verified**, and the reason for each is written into the comments of the code.

The fixed entries are not kept here — the full text of each one (evidence, line numbers,
consequences, the fix) is in

```
git show 665d7a9:docs/pipeline-audit.md
```

This document keeps only what has not been closed out: one item where the software side is
finished and the science is not, plus a set of parameter questions that were never part of a
software audit in the first place.

---

## One — software finished, science undecided

### step3 and step5 disagree about where the sky may be learned · verified

`sky_region` in `configs/pNN.yaml` defaults to `apply_to: [basis]`, which hands it to step3
**only**. step5 never receives it; its s-field training sample is decided by
`utils.build_s_field`, which looks at nothing except distance from the segmentation
(`--sf-r-far` 15 px, `--sf-r-far-haro` 50 px).

s-field training points that fall inside step3's excluded region:

```
  p09  --xlim 0 100      train 14,150   excluded 6,123  (43.3%)
  p11  --xlim 0 ...      train 25,436   excluded 9,682  (38.1%)
  p12                    train 24,291   excluded 7,935  (32.7%)
  p01  --xlim 0 165      train 39,286   excluded 12,412 (31.6%)
  p03  --ylim 170 9999   train 27,322   excluded 6,681  (24.5%)   median s there is 0.8% higher
  p14  --exclude-box     train 38,290   excluded 1,630  ( 4.3%)
```

The s field, multiplied by `C_sky`, is subtracted from **every** spaxel, source spaxels
included. That path is precisely the over-subtraction that restricting the sky basis's
learning region is meant to block — shut out at step3, it walks back in through step5's side
door.

**What the software already does**: step5 has `--sf-xlim/--sf-ylim`, whose inclusion rule
(LO included, HI excluded) is word for word the same as step3's `--xlim/--ylim`, and they are
combined with `--sf-exclude-box` into the single exclude mask handed to `build_s_field`; both
ranges go into `s_field_params` in `meta.json`. Verified: giving p01 `--sf-xlim 0 165` takes
the training points from 39,286 → 26,874, which is exactly the 39,286 − 12,412 of the table
above.

**What it does not do**: `apply_to` deliberately does not include `s_field`, so the range is
not forwarded to step5. That step would change the scientific result. The two sides
disagreeing is still the default; what has changed is that the disagreement can now be written
down and recorded in `meta.json`.

Trying it means adding the same set of numbers as `REGION` to the step5 command, and giving it
a directory of its own with `--run`:

```
  ... step5_fit_s_field.py ... --sf-xlim 0 165 --out results/skymodel/p01/step05_sfregion
```

---

## Two — science questions left for the project to decide

These are **not** software defects, and this document makes no recommendation about them.

1. **Whether the s field's training region should equal step3's sky-learning region** (the
   section above). The evidence has been quantified.
2. The parameters themselves: `-K = 30`, `--s-fix`, `--sf-r-far 15` / `--sf-r-far-haro 50` /
   `--sf-clip 8`, `MIN_COVERAGE = 0.9`, `CLIP_SIGMA = 30`,
   `--star-window/--gal-window 4600 8000`, `--line-mask-iter 1`, and the 14 `sky_region`
   blocks in `configs/pNN.yaml`.
3. `DV_MAX = 1468.0` (`utils.py`) **is listed separately, because it is a different kind of
   thing from the ones above**. It has been checked (`git log -S "DV_MAX" --all`): it first
   appears in `b147631` (2026-08-16, where main-source grouping moved to a redshift
   criterion), and before that neither `1468` nor `DV_MAX` existed anywhere in the repo, so it
   was **not carried in from somewhere else, and it is not a value handed to the project from
   outside**. That same commit's message says of itself that "the result is the same for
   thresholds from 300 to 100,000 km/s" — whoever wrote it already knew at the time that the
   precise value does not affect the result. No code, comment or document records where those
   four significant figures came from.

   So the question to ask is not "what does 1468 mean physically", but "should a threshold we
   produced ourselves be replaced by an integer that says honestly that it is no more than a
   safe upper bound".
4. Whether step2 should move to a cube we have subtracted the sky from ourselves. It currently
   uses ESO's nosky; switching to our own `step05/sky_subtracted.fits` would form a
   5 → 2 → 4 → 5 loop.

---

## Three — what this audit did not cover

`src/skymodel/experiments/`, `src/zap/`, `libs/`; whether the fits are physically correct; the
plotting details in `evaluation/`; performance and memory. No step was re-run during the
audit: the numbers in the first section come from rebuilding `build_s_field`'s training mask
out of the saved `sky_continuum_amplitude_per_spaxel.npy` (the `n_train` that comes out matches
each pointing's `meta.json` bit for bit).
