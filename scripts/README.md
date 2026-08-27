# Checking a change did what it says

Three scripts, each answering a different question about an edit. They are for
working on the pipeline, not for using it; nothing under `src/` imports them.

## `verify.py` — did the products change?

```bash
python scripts/verify.py p05
python scripts/verify.py p05 p14
python scripts/verify.py --compare-only p05     # compare a run already made
```

Runs the pipeline for each pointing into `results/_verify/`, then compares every
product against the run stored where that pointing's config sends it, walking
subdirectories as well: arrays value by value with NaNs in the same places, FITS
plane by plane, npz field by field, JSON key by key, everything else byte for
byte. `meta.json`'s `created` and `git_commit`, and paths that follow the output
directory, are normalised away; logs are compared for presence but not for
content.

Exit 0 and `PASS` when everything matches. On a difference it prints what
differs and leaves the new products in place to be looked at.

A change that is meant to keep the answer identical -- a refactor, a renamed
function, a faster loop -- has to pass this. p05 and p14 between them drive both
branches of the sky region: p05 restricts by a range, p14 excludes a box.

What it cannot judge is a change to what the products are called or where they
sit. Files are paired by name, so a renamed product comes out as missing and the
file that replaced it as unexpected, whatever is inside either. After such a
change the stored run has to be made again before this says anything.

## `moves.py` — was the code only moved?

```bash
python scripts/moves.py HEAD
```

Compares every top-level definition in `src/skymodel/*.py` between that revision
and the working tree, by source text, and reports each as `moved`, `deduped`,
`added`, `removed` or `CHANGED`. Module-level assignments count, so a constant
that several files defined and one file now defines shows up.

`0 changed` means no definition's text was edited on the way across. For a file
reorganisation that is a stronger statement than a run of the pipeline can make,
and it takes a second.

## `codeonly.py` — was only the prose touched?

```bash
python scripts/codeonly.py HEAD
```

Parses each module, drops the docstrings and compares the syntax tree; comments
never reach a tree at all. `prose only` for every file means the edit cannot have
changed what the program does, whatever it did to the comments.
