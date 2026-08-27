# Stellar templates (data/stellar_templates)

> The stellar branch of the **source model** used for the sky reconstruction: The template fit sets these
> against the galaxy eigenspectra over the same set of channels, and whichever gives the lower
> reduced chi2 wins. Given to the project (2026-08-19, `stars.tgz`).
> Where the files live: `data/stellar_templates/`. The seven the pipeline uses are
> **tracked in git**, since they are small and nothing runs without them; `l5v.dat` is
> not, for the reason in section 2.

---

## 1. What is in it

The tarball held 8 two-column ASCII files (wavelength in Å, flux), all sharing one
wavelength grid; 7 of them ship with the code:

```
1200 - 24200 Å,  one point every 2.5 Å,  9201 points
```

| file | spectral type | | file | spectral type |
|---|---|---|---|---|
| `o5v.dat` | O5V | | `g5v.dat` | G5V |
| `b3v.dat` | B3V | | `k0v.dat` | K0V |
| `a0v.dat` | A0V | | `m4.5v.dat` | M4.5V |
| `f0v.dat` | F0V | | `l5v.dat` | L5V (not shipped) |

Every luminosity class is V (main sequence), and O through L makes one temperature sequence.
The flux is F_λ, with O5V through K0V normalised to 1.0 at 5556 Å; M4.5V and L5V are at 0.196
there, so they were normalised some other way. Channels with no data hold 0 rather than NaN.

**The origin is unknown.** The tarball carries no header, no README and no mark of where the
data came from. Normalising at 5556 Å, and the naming of `o5v`/`b3v`/`a0v`, both agree with
Pickles (1998) (the ESO release names them `uko5v.dat.gz` and so on), but Pickles covers
1150–25000 Å sampled every 5 Å and has neither L types nor fractional subtypes, so this set has
been resampled and had `m4.5v` and `l5v` added from somewhere else. Citing it would first
require confirming the origin with whoever supplied the tarball.

## 2. Two things done at load time

**The wavelength axis is in air wavelengths, and is converted to vacuum on load.** Measuring
the centroids of the hydrogen lines in `a0v` (Hβ, Hα, Pa16–Pa11): the median shift against air
wavelengths is −18 km/s, and against vacuum −100 km/s. The eigenspectra are on vacuum
wavelengths and everything downstream reads values on vacuum wavelengths too, so leaving the
conversion out puts a systematic −83 km/s into every stellar redshift.
See `utils.load_ascii_template`.

The Morton formula inside `air_to_vacuum` has a pole at 1602.8 Å and is neither correct nor
monotonic anywhere near it, so the wavelength axis is cut at 2000 Å before the conversion
(air is opaque below that wavelength, so an "air wavelength" there has no meaning to begin
with).

**`l5v` does not cover MUSE, and `classify_sources` excludes it.** It only holds data above
5390 Å, while the rest-frame range MUSE needs is 4578–9399 Å. Every template's coverage is
checked, and one that falls short is reported and skipped. Since the check would reject it on
every run, `l5v.dat` is not shipped with the code and is not in the repository.

## 3. Why this set

**The reason is wavelength coverage.** MUSE runs 4600–9350 Å, and the source model needs a
value across the whole of it. The other candidate was SDSS's spDR2 stellar templates, whose 23
spectra stop at about 9200 Å: `fit_source` throws away, for every spaxel, any channel whose
design matrix holds a NaN, which would leave the reddest ~120 channels of the source region
with no source model, and out of the solve for the sky coefficients as well. This set covers
1200–24200 Å, with a value everywhere in it.

The price is paid in sampling, in resolution and in how many types there are:

| | this set | SDSS spDR2-000..022 |
|---|---|---|
| wavelength coverage | 1200–24200 Å | 3806–9219 Å |
| sampling | 2.5 Å | 1.36 Å (log-λ, at the red end) |
| resolution | Hα FWHM ≈ 10 Å (the Na D doublet is not resolved) | Hα FWHM ≈ 3 Å |
| number of types | 7 usable (main sequence O–M) | 23, including white dwarfs, carbon stars, K subdwarfs, M1–M8 |

The per-source result figures are in `results/skymodel/evaluation/pNN/star_library/`.
