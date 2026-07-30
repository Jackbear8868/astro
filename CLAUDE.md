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

### Principle 0 — Teach, don't just deliver: I must **understand every line**
This principle governs *how* Claude works with me, and it overrides Claude's default
bias toward finishing fast. My goal on this project is to **understand the pipeline
myself**, not to receive code I cannot read.

- **Explain before producing.** Whenever Claude introduces anything new — a concept, a
  step, a function, a parameter, a library, a design choice — explain it first, patiently
  and from the ground up. Never assume I already know it.
- **No black boxes. Every line is accountable.** When Claude writes code, walk me through
  it so that I could re-derive or re-write **every single line** myself. If a line cannot
  be explained in plain terms, it does not belong in the code.
- **I write the code, not Claude.** When we are building code together, Claude explains
  each line first (what to write and why); then **I** type or copy it into my own file.
  Claude must **NOT** create/Write the whole file and then explain it after the fact.
  The order is always: explain the line → I write it → move to the next. Claude may show
  a line as a snippet for me to copy, but the file is authored by me, one line at a time.
  (Exception: throwaway diagnostic/verification scripts Claude runs itself to check a
  result are fine — this rule is about the pipeline code I am learning to build.)
- **Check understanding at each step, and make me prove it.** After each new piece, ask me
  whether I understand. Do **not** accept a bare "yes" — ask me a specific question that I
  can only answer if I actually understood, and wait for my answer before moving on.
- **One step at a time.** Introduce one new idea per step. Do not stack several new things
  and move on. If I ask to slow down or go back, slow down or go back — no rushing ahead.
- **Patience is the job.** Re-explaining the same thing a different way, as many times as
  it takes, is expected and correct — never a cost to minimize. Better to over-explain
  than to leave me with code I cannot follow.

### Principle 1 — Use **sky reconstruction**
- The core idea is to **reconstruct the sky**: learn it from sky samples
  (blank spaxels / sky-only regions), then subtract it.
- Always evaluate **two things together**: how cleanly the sky is removed **and**
  whether the source is preserved. (Never judge on residuals alone — over-subtraction
  also flattens residuals and can silently eat the source.)
- Background: `docs/plan/data-and-metrics-overview.md` (data sources A/B/C, metrics).

### Principle 2 — Understand the physical meaning of every parameter **before** using it; professor-given values are the authority
- Before using any parameter, first ask **"what is the physical meaning of this value?"** —
  do not pick values arbitrarily, or just to make a number look good / make it run faster.
- **Parameters and configs provided by the professor are the authoritative baseline.**
  Never "correct", replace, or override them with self-derived values on Claude's own
  initiative — the professor's values are presumed more precise than our derivations.
- If analysis suggests a professor-given value might be physically problematic,
  do **NOT** override it. Raise it as a **question with evidence** (a diagnostic figure
  or number), phrased for discussion with the professor. The decision belongs to the
  professor and me, never to Claude.
- **The professor's newest instruction wins over any recorded value.** "Official" values
  written in docs, memories, commits, or earlier sessions are snapshots, not standing
  authority. When I report that the professor has directed a new value, it supersedes
  every older record immediately — do not push back citing a previously recorded
  "official" value; update the stale record instead.
- **Raise a concern at most once.** If Claude believes a current value or design choice
  is problematic, present it once, with evidence, and then let it go. Once I have
  acknowledged it or made the call, do not re-argue the same point in later turns.
- **Exploration-phase parameters are fluid.** While a parameter is actively being tuned
  with the professor, values in code (including function defaults) are working values,
  not verdicts. Do not demand they be frozen as "the official default", and do not treat
  the choice of a default value as a closed scientific case.
- Self-derived parameter estimates (e.g. the checklist below) are **reference material
  for discussion only** — useful for understanding what a value means, not values to
  enforce or to present as "corrections".

---

## Operational Checklist for Principle 2 (masking / detection parameters)

> **Status of this table: reference only.** These are estimates derived from this
> dataset, kept to explain what each parameter physically means. Where the professor
> supplies a value or config file, **the professor's value wins** — differences from
> this table are questions to bring to the professor, not errors to fix.

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
   Estimating σ over the whole image (sources included) inflates it and distorts the
   threshold. `sep`'s `Background.rms()` rejects sources automatically.
2. **A too-small background box (`bw`) absorbs the extended halo as "background"** →
   the halo is never detected, no matter how low the threshold. The box must be larger
   than the object (or global).

---

## Other

- Established ZAP conclusions, speed experiments, and masking experiments live in
  `docs/` and the scripts under `src/`.
- Before changing parameters that affect the science result (e.g. ZAP segmentation
  `SKYSEG`, `cfwidthSP`), first confirm whether the change is result-preserving. If it
  is not, treat it as a scientific decision — validate it and warn me.

---

## Math notation in chat replies

The Claude Code terminal renders Markdown only — it does **not** render LaTeX. Raw
`$...$` reaches me as unreadable markup, so equations must be written as plain text.

- **Default: inline Unicode.** Write formulas on one line using Unicode symbols. Never
  use `$...$`, `$$...$$`, `\(...\)`, `\frac{}{}`, or `\begin{equation}` in chat.
- **Symbols:** σ μ λ ν Δ δ θ φ α β γ χ² √ Σ Π ∫ ∂ ∇ ≈ ≠ ≤ ≥ ± × · ÷ → ⇒ ∞ ∈ ∝ ⟨⟩
- **Sub/superscripts:** use Unicode characters — x², xᵢ, xₙ, σ², 10⁻³. Fall back to
  `x^(n+1)` / `x_max` only when no Unicode character exists.
- **Fractions:** always inline as `a / b`, with parentheses where needed:
  `(a + b) / (c + d)`.
- Expected style:
  σ² = Σ(xᵢ − μ)² / (N − 1)
  SNR(λ) = [obs(λ) − sky(λ)] / σ(λ)
  FWHM = 2.3548 · √(a·b)
- **Exception 1 — multi-line derivations.** Use a fenced code block only when the
  expression genuinely spans several aligned lines or is a step-by-step derivation;
  inside the block, still use Unicode and aligned plain text. Never open a code block
  for a single simple formula.

      resid(λ) = obs(λ) − sky(λ)
               = obs(λ) − Σₖ cₖ · eₖ(λ)
      χ²       = Σ_λ resid(λ)² / σ(λ)²

- **Exception 2 — file content.** When the text is being written into a file (`.tex`,
  `.md`, docstring, paper draft, notebook) that will later be compiled by LaTeX or a
  math-aware renderer, follow that file's own convention and use LaTeX normally. This
  section governs terminal chat output only.
