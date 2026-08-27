"""Run the pipeline for one or more pointings and compare every product against
the stored ones. Expect zero differences.

    conda run -n astro python verify.py p05
    conda run -n astro python verify.py p05 p14
    conda run -n astro python verify.py --compare-only p05     # skip the run

Exit code 0 when everything matches, 1 otherwise. The scratch output is deleted
on success and kept on failure so the difference can be looked at.

meta.json carries three things that follow the run rather than the result --
`created`, `git_commit`, and paths that contain the output directory name -- so
those are normalised before comparing. Everything else, including every array
and every FITS plane, has to match value for value, with NaNs in the same places.

Products are paired by name, so this says nothing useful across a change to what
the products are called or where they sit: every stored name is then reported as
missing and every new one as unexpected, whatever the values are. Such a change
is checked by making the stored run again, not here.
"""
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "results" / "_verify"
VOLATILE = {"created", "git_commit"}

def same_values(a, b):
    """Value equality that treats NaN as equal to NaN in the same place."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype.kind != b.dtype.kind:
        return False
    if a.dtype.names:
        # A structured array's kind is "V", so the float branch below never sees its
        # float fields and one NaN anywhere would fail the whole array. Compared field
        # by field each one reaches the branch that knows about NaN.
        return (a.dtype.names == b.dtype.names
                and all(same_values(a[n], b[n]) for n in a.dtype.names))
    if a.dtype.kind in "fc":
        na, nb = np.isnan(a), np.isnan(b)
        return np.array_equal(na, nb) and np.array_equal(a[~na], b[~nb])
    return np.array_equal(a, b)


def normalise(obj, out_name):
    """Drop the keys that record the run, and the output path it ran into.

    A recorded path holds the output directory's name, so two runs of the same
    pointing into different directories would differ on that alone.
    """
    if isinstance(obj, dict):
        return {k: normalise(v, out_name) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [normalise(v, out_name) for v in obj]
    if isinstance(obj, str):
        return re.sub(r"results/[^/\s]*/" + re.escape(out_name), "<out>", obj)
    return obj


def compare_file(ref, new, out_name, fails, rel):
    """Append one message per difference found in this pair of files.

    `rel` is the path the messages name the pair by, relative to the output
    directory: a bare file name no longer identifies one product now that the
    same name occurs in a step directory and in a subdirectory of it.
    """
    if ref.suffix == ".npy":
        if not same_values(np.load(ref), np.load(new)):
            fails.append(f"{rel}: values differ")
    elif ref.suffix == ".npz":
        A, B = np.load(ref, allow_pickle=True), np.load(new, allow_pickle=True)
        only = set(A.files) ^ set(B.files)
        if only:
            fails.append(f"{rel}: keys differ {sorted(only)}")
        for k in set(A.files) & set(B.files):
            if not same_values(A[k], B[k]):
                fails.append(f"{rel}[{k}]: values differ")
    elif ref.suffix == ".fits":
        with fits.open(ref) as A, fits.open(new) as B:
            if len(A) != len(B):
                fails.append(f"{rel}: {len(A)} HDUs vs {len(B)}")
                return
            for i, (ha, hb) in enumerate(zip(A, B)):
                if ha.data is None and hb.data is None:
                    continue
                if ha.data is None or hb.data is None:
                    fails.append(f"{rel}[{i}]: one HDU has no data")
                elif not same_values(ha.data, hb.data):
                    n = int(np.sum(np.asarray(ha.data) != np.asarray(hb.data)))
                    fails.append(f"{rel}[{ha.name or i}]: {n:,} elements differ")
    elif ref.suffix == ".json":
        a = normalise(json.loads(ref.read_text()), out_name)
        b = normalise(json.loads(new.read_text()), out_name)
        if a != b:
            keys = sorted(set(a) ^ set(b)) or [k for k in a if a[k] != b.get(k)]
            fails.append(f"{rel}: differs at {keys}")
    else:                                    # .png, .log and anything else
        if ref.suffix != ".log" and ref.read_bytes() != new.read_bytes():
            fails.append(f"{rel}: bytes differ")


def compare_dir(ref_dir, new_dir, out_name, fails, rel=""):
    """Compare one directory of the stored run against the new one, entry by entry.

    Subdirectories are walked as well. A step puts products in one when it has
    several kinds to keep apart -- step04 its per-source scans, step05 the runs
    kept beside the pipeline's own -- and a directory the walk stops at is a
    directory whose contents are never compared and never reported either.
    """
    ref_entries = {p.name: p for p in ref_dir.iterdir()}
    new_names = {p.name for p in new_dir.iterdir()}
    n = 0
    for name in sorted(ref_entries):
        ref, r = ref_entries[name], rel + name
        if name not in new_names:
            fails.append(f"{r}: missing from the new run")
            continue
        if ref.is_dir():
            n += compare_dir(ref, new_dir / name, out_name, fails, r + "/")
        else:
            compare_file(ref, new_dir / name, out_name, fails, r)
            n += 1
    for name in sorted(new_names - set(ref_entries)):
        fails.append(f"{rel}{name}: the new run has it, the reference has not")
    return n


def reference(pointing):
    """Where this pointing's config says its products go -- the stored run."""
    text = (ROOT / "configs" / f"{pointing}.yaml").read_text(encoding="utf-8")
    m = re.search(r"^output: *(\S+)", text, flags=re.M)
    if not m:
        raise SystemExit(f"★ configs/{pointing}.yaml has no output: line")
    return ROOT / m.group(1)


def run(pointing):
    """Run the pipeline into the scratch area; return its output directory."""
    cfg_src = ROOT / "configs" / f"{pointing}.yaml"
    out_rel = f"results/_verify/{pointing}"
    cfg = SCRATCH / f"{pointing}.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(re.sub(r"^output: .*$", f"output: {out_rel}",
                          cfg_src.read_text(encoding="utf-8"), flags=re.M),
                   encoding="utf-8")
    t0 = time.time()
    # sys.executable, so the pipeline runs under the interpreter this script was
    # started with rather than one named here.
    r = subprocess.run([sys.executable,
                        str(ROOT / "src/skymodel/pipeline.py"), str(cfg)],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"★ {pointing}: the pipeline failed")
    print(f"  {pointing}: ran in {time.time() - t0:.0f} s")
    return ROOT / out_rel


def main(argv):
    compare_only = "--compare-only" in argv
    pointings = [a for a in argv if not a.startswith("-")] or ["p05"]
    ok = True
    for p in pointings:
        new_dir = ROOT / "results/_verify" / p if compare_only else run(p)
        ref_root = reference(p)
        fails = []
        # From the top of the output directory, not from each step directory in
        # turn: a file appearing there -- a run record, a summary -- would
        # otherwise never be compared. Logs are excluded: they carry the time and
        # the path the run used.
        n = compare_dir(ref_root, new_dir, p, fails)
        print(f"  {p}: {n} files compared -> "
              + ("all identical" if not fails else f"{len(fails)} DIFFERENCES"))
        for f in fails[:40]:
            print(f"      {f}")
        if len(fails) > 40:
            print(f"      ... and {len(fails) - 40} more")
        ok &= not fails
    if ok:
        shutil.rmtree(SCRATCH, ignore_errors=True)
        print("PASS")
    else:
        print(f"FAIL -- the new products are kept in {SCRATCH}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
