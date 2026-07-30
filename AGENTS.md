# AGENTS.md

These instructions apply to the entire repository.

## Project purpose

This project studies sky subtraction and sky reconstruction for Haro 11 MUSE
integral-field data. The main objective is to learn a sky model from blank
spaxels, reconstruct the sky across the field, and subtract it without removing
astronomical source signal.

The current repository layout is:

- `src/skymodel/`: the hand-built research and teaching pipeline; this is the
  active main line.
- `src/zap/`: the ZAP comparison pipeline.
- `src/cgm_halpha.py`: downstream extended H-alpha analysis.
- `docs/`: current method, metric, parameter, and literature documentation.
- `libs/zap/`: vendored ZAP code.
- `data/` and `results/`: large inputs and regenerable outputs; both are ignored
  by Git.

`docs/plan/joint-sky-factorization-spec.md` describes the intended full method.
The hand-built pipeline must still proceed from the simplest understandable
version first; do not silently enable every mechanism in the full specification.

## Teaching and collaboration

The user's goal is to understand and be able to reproduce every line of the
pipeline, not merely receive finished code.

- Explain every new concept, parameter, function, library, and design choice
  before introducing its code.
- Introduce one new idea at a time. Do not stack several unexplained changes.
- For pipeline code under `src/skymodel/`, explain the next line or small
  coherent snippet first, then let the user type or copy it. Do not author an
  entire pipeline file and explain it afterward.
- After a new concept, check understanding with a specific question that
  requires the user to explain the idea, not with a bare yes/no question.
- Re-explain patiently in a different way when needed.
- Throwaway diagnostic and verification scripts are exempt from the
  user-authored-code rule.
- Repository configuration, documentation maintenance, and changes that the
  user explicitly asks Codex to perform directly are also exempt.

## Scientific rules

### Reconstruct the sky

- Learn the sky from blank or sky-only spaxels, then reconstruct and subtract
  it.
- Use a `wsky` cube as the input to a sky-subtraction method. A `nosky` cube has
  already been sky-subtracted and is a reference or null-test input, not the
  normal training input.
- Every evaluation must report both sky-removal quality and source
  preservation. A flat residual alone is not evidence of success because
  over-subtraction can also flatten the result.
- Treat changes that alter the science result as scientific decisions. Produce
  diagnostic evidence and obtain the user's decision before adopting them.

### Parameters and professor instructions

- Explain the physical meaning of every parameter before using or changing it.
- Professor-provided values and configuration files are the authoritative
  baseline. Do not replace them with self-derived values.
- The professor's newest instruction, as relayed by the user, supersedes older
  documentation, memories, commits, and defaults immediately.
- Self-derived values are discussion references only. If evidence suggests a
  professor-provided value may be problematic, raise one evidence-backed
  question for discussion; do not label a self-derived alternative a
  correction.
- Raise the same scientific concern at most once after the user has
  acknowledged it or made a decision.
- Parameters being tuned with the professor are working values, not permanent
  defaults. Do not demand that an exploration value be frozen.

### Current pipeline decisions

- Start with the simplest pipeline the user can fully control. Add mechanisms
  such as throughput maps, exposure-region maps, source-line windows, or other
  corrections only when a diagnostic demonstrates the need, a controlled
  comparison quantifies the effect, and the user explicitly approves adoption.
- Use the MUSE `STAT` extension at face value. Do not apply a 1.8 noise
  correction. The ideal reduced-chi reference is 1; any observed discrepancy is
  an open diagnostic question unless the professor decides otherwise.
- Build the formal source mask with SExtractor on the full-spectrum white-light
  image, not an H-alpha narrow-band image. Use the professor-provided
  `src/skymodel/SExtractor/default.sex` as the baseline.
- The current line-detection recipe estimates sigma from the mean blank-sky
  spectrum with `src/skymodel/utils.py::detect_lines`. Its positive/negative
  thresholds are currently `(1, 2)` and remain exploration values subject to
  the professor's latest instruction.
- Do not revive deleted legacy pipelines or present their historical outputs as
  current results. Consult Git history or archived memories only when the user
  explicitly asks to revisit an earlier experiment.

## Execution environment

- Run project Python through the `astro` conda environment:

  ```bash
  conda run -n astro python path/to/script.py
  ```

- For code that imports the vendored ZAP package, expose `libs/zap`:

  ```bash
  PYTHONPATH=libs/zap conda run -n astro python path/to/script.py
  ```

- Do not substitute plain `python3` or `uv` without first demonstrating that the
  required scientific dependencies and local ZAP package are available.
- Follow the setup instructions in `README.md` when creating the environment.
- Large FITS files and generated results are not source files. Avoid copying,
  committing, or deleting them unless the user places those exact artifacts in
  scope.

## Documentation

- Write specifications and design documents in one consistent, authoritative
  voice.
- Include only decided and precise information in canonical documents.
- Represent unresolved items as `待討論定案`; keep option streams, deliberation
  history, verdict labels, and before/after narratives out of canonical specs.
- Keep README paths, commands, and project-structure descriptions synchronized
  with the current repository.

## Working-tree safety

- The repository may contain ongoing user edits and untracked scientific work.
  Inspect `git status` and the relevant diff before changing files.
- Preserve unrelated work. Never restore deleted files or overwrite untracked
  files merely to make the tree clean.
- Before any cleanup, distinguish tracked, untracked, ignored, and generated
  artifacts, and explain whether deletion would be recoverable.
