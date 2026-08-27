# Settled ZAP conclusions (Haro11 MUSE cube)

> The **settled conclusions and the numbers behind them** for ZAP (Zurich Atmosphere Purge,
> Soto+2016) on the Haro11 data, which is what the sky reconstruction is compared against.
> Data: `Haro11_wsky.fits` (sky included) / `Haro11_nosky.fits` (sky already subtracted by
> MUSE), WFM, 499×559×3679, λ 4750–9348 Å, z≈0.0206, Hα ~6699 Å, with ionised gas (CGM)
> extending over a large area.
> The current ZAP pipeline is `src/zap/`; the parameter reference is
> `docs/zap-parameters-reference.md`; the products go to `results/zap/` (gitignored, local).

---

## 1. The core conclusions

1. **The correct input to ZAP is `wsky`, the cube that still contains the sky.** Running ZAP on
   `nosky`, which has already had its sky subtracted, is a null test: there is no sky signal
   left to learn, so all it can do is fit noise and inject it back in (the source's Hα peak
   drops by ~70%, and the blank residual gets worse).
   And **no number of components rescues it** — sweeping nevals from 3 to 55 still leaves the
   line-free noise at ≥2.9 and only 89% of Hα preserved
   → this is **a problem with the input, not a tuning problem**; do not read it as "ZAP is
   broken" or "nevals needs adjusting".

2. **The source mask must cover the whole of the extended ionised gas, or ZAP eats the source.**
   A 2σ threshold on the white light image masks only 8%, but the extended ionised gas covers
   ~31–44% of the field (21% of it above 3σ);
   the spaxels left unmasked go into the sky basis (the 90th percentile of Hα in the "blank"
   sample is 19.4, against a noise σ≈2.2), and ZAP learns Hα as if it were sky
   → **70% of the source flux is lost**.
   Once detection plus a dilated safety margin grows the mask to 44%, the sky basis is clean
   (Hα 90th percentile 19.4 → 1.9) and 124% of the source is preserved (whole field).

3. **Preserving more than 100% of the source (110–124%) is not a bug.** The source spaxels are
   held out of the sky basis, so ZAP subtracts gently on a bright source and does not
   over-subtract it; MUSE, on a bright source, may in fact slightly over-subtract. When the
   point is to preserve faint extended signal, preserving a little too much is the better error
   to make.

---

## 2. The numbers (whole 499×559 field, `wsky` + ZAP)

| blank-sky metric | `wsky` as it comes (sky included) | **`wsky` + ZAP** | `nosky` (the MUSE truth) |
|---|---|---|---|
| sky line 5577Å (median) | 252.7 | **0.62** | 2.30 |
| sky line 6300Å (median) | 175.1 | **−0.26** | 0.45 |
| sky line 8400Å (median) | 463.9 | **1.26** | 1.08 |
| line-free pure noise | 5.71 | **1.70** | 1.30 |
| source Hα integrated flux | — | **124% preserved** | (100%) |

ZAP pushes the sky lines from ~250–460 down to ~0–1.3, essentially reproducing the MUSE sky
subtraction while keeping the source.
The main validation figure is `results/zap/fig5_zap_validation.png`.
(The 300² cutout gives the same conclusions: sky line 267→1, noise 5.71→1.70, source
preserved 110.6%.)

---

## 3. Faint CGM: MUSE `nosky` is cleaner than ZAP

(the CGM Hα analysis, `fig6`/`fig7`)

- Both sky subtractions recover the galaxy plus an Hα halo out to ~20–30″.
- Per-spaxel noise in the outskirts: `nosky` **1σ≈13.8** against `wsky`+ZAP **≈35.5**
  (ZAP is about 2.5× noisier).
- At large radius (>35″) the `nosky` median turns slightly **negative** (MUSE over-subtracts a
  little), while `wsky`+ZAP stays **positive** — but that positive offset falls inside the
  **rectangular footprint** of the sky exposure, which makes it more likely to be a
  sky-subtraction residual than real CGM.
- The conclusion: **it cannot be said that "ZAP reveals more CGM"**. ZAP's value is that it can
  **reproduce the sky subtraction independently, without relying on the pipeline**, not that it
  goes deeper. For faint CGM, `nosky` is the better choice.

---

## 4. What a run costs (whole field)

- ZAP on `wsky`: **~65 minutes** (3899 s, ncpu=16), peak memory **43.7 GB**, choosing **53**
  eigenspectra by itself.
- The mask covers 44%, leaving ~140k clean blank spaxels to build the sky basis from.

---

## 5. Known limitations

- There is only 1 cube with the sky still in it (`wsky`) → only single-exposure reconstruction
  is possible, and generalisation across exposures cannot be tested (the per-exposure data is
  ready: 14 pointings each under `data/wsky/` and `data/nosky/`).
- ZAP's residual std at the sky-line wavelengths is slightly higher than MUSE's (a normal
  property of empirical PCA).
- **STAT (the per-voxel variance) is copied through unchanged and is not propagated by ZAP**;
  anything that needs correct errors has to handle that itself.
- The positive offset at large CGM radius carries a systematic shaped like the footprint, and
  whether it can be corrected has not been established.
- The `nosky` null test has only been quantified on the 300² cutout; the whole-field version
  was never done (little value in it).

---

## 6. The main products (`results/zap/`, local)

- `fig5_zap_validation.png` — **the main validation** (wsky+ZAP against the nosky truth).
- `fig6_cgm_halpha_maps.png` — CGM Hα surface brightness maps (nosky against wsky+ZAP).
- `fig7_cgm_radial_profile.png` — the azimuthally averaged Hα radial profile, with the
  detection limit.
