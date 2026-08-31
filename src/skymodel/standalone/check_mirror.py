"""Run one pointing both ways and compare every product.

The steps in this folder are a second implementation of the ones in `pipeline.py`. Two
implementations of the same method drift: a change made to one and not the other leaves
both running, both writing plausible products, and only the science different. Neither
program can notice that on its own -- each is internally consistent, and a config key
one of them does not read is silently ignored rather than refused.

This is what notices. It runs a config through `pipeline.py` and through
`run_pipeline.py`, into two directories, and compares what they wrote:

    python check_mirror.py ../../../configs/p01.yaml

Products are compared by content and not by mtime or by size. A FITS cube is compared
on its data, not its bytes, because the two runs stamp different dates into the header
and that is not a difference in the answer. meta.json is compared on the fields that
say what was computed, with `created`, `git_commit` and `step` left out for the same
reason.

Exit status is 0 when every product matches and 1 when any does not, so a hook or a
test can call it. It is deliberately not a unit test: what has to agree is the products
of a whole run, and nothing smaller would have caught the drift it exists for.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# Written by every run, and holding what the run was rather than what it computed.
META_SKIP = {"created", "git_commit", "step", "work", "cube", "seg", "sky_dir",
             "classification", "s_field", "spectra"}


def load_yaml_out(cfg_path):
    """The `output` a config names, without importing either program's config module."""
    for line in Path(cfg_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("output:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit(f"★ {cfg_path} names no output directory")


def compare(a, b):
    """(name, verdict) for every product under either directory."""
    names = sorted({str(p.relative_to(a)) for p in a.rglob("*") if p.is_file()}
                   | {str(p.relative_to(b)) for p in b.rglob("*") if p.is_file()})
    for n in names:
        pa, pb = a / n, b / n
        # Skipped before the existence check, not after: these are the two programs'
        # own narration and bookkeeping, and only one of them writes them at all. Asked
        # about existence first, every one of them is reported as a difference.
        if pa.suffix in (".log", ".png") or pa.name == "config.json":
            continue
        if not pa.exists():
            yield n, "only the live pipeline wrote it"
            continue
        if not pb.exists():
            yield n, "only the standalone run wrote it"
            continue
        try:
            yield n, _same(pa, pb)
        except Exception as e:                       # noqa: BLE001
            yield n, f"could not be compared: {type(e).__name__}: {e}"


def _arrays_equal(x, y):
    """Equal, counting NaN in the same place as equal.

    A structured array is compared field by field. np.array_equal has no equal_nan for
    one -- the flag is rejected on a void dtype -- so without this a scan holding NaN
    compares unequal to itself. The star scans do hold NaN: `A` is as wide as the
    galaxy branch needs and a stellar template fills one column of it.
    """
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.dtype != y.dtype:
        return False
    if x.dtype.names:
        return all(_arrays_equal(x[f], y[f]) for f in x.dtype.names)
    if x.dtype.kind == "f":
        return np.array_equal(x, y, equal_nan=True)
    return np.array_equal(x, y)


def _same(pa, pb):
    if pa.suffix == ".fits":
        with fits.open(pa, memmap=True) as ha, fits.open(pb, memmap=True) as hb:
            if len(ha) != len(hb):
                return f"{len(ha)} HDUs against {len(hb)}"
            for i, (ea, eb) in enumerate(zip(ha, hb)):
                if ea.data is None and eb.data is None:
                    continue
                if not _arrays_equal(ea.data, eb.data):
                    return f"HDU {i} data differs"
        return None
    if pa.suffix == ".npy":
        return None if _arrays_equal(np.load(pa), np.load(pb)) else "arrays differ"
    if pa.suffix == ".npz":
        za, zb = np.load(pa, allow_pickle=True), np.load(pb, allow_pickle=True)
        if set(za.files) != set(zb.files):
            return (f"fields differ: only here {sorted(set(za.files) - set(zb.files))}, "
                    f"only there {sorted(set(zb.files) - set(za.files))}")
        bad = [k for k in za.files if not _arrays_equal(za[k], zb[k])]
        return f"fields differ: {bad}" if bad else None
    if pa.name == "meta.json":
        ma = {k: v for k, v in json.loads(pa.read_text()).items() if k not in META_SKIP}
        mb = {k: v for k, v in json.loads(pb.read_text()).items() if k not in META_SKIP}
        bad = sorted(k for k in set(ma) | set(mb) if ma.get(k) != mb.get(k))
        return f"fields differ: {bad}" if bad else None
    return None if pa.read_bytes() == pb.read_bytes() else "bytes differ"


def main():
    ap = argparse.ArgumentParser(
        description="run a pointing through both programs and compare every product")
    ap.add_argument("config", help="a pointing config, e.g. configs/p01.yaml")
    ap.add_argument("--work", type=Path, default=None,
                    help="where the two runs go; defaults to a mirror_check directory "
                         "beside the config's own output")
    ap.add_argument("--reuse", action="store_true",
                    help="compare what is already in --work instead of running again")
    args = ap.parse_args()

    out = load_yaml_out(args.config)
    base = args.work or (ROOT / out).parent / f"{Path(out).name}_mirror_check"
    live, alone = base / "live", base / "standalone"

    if not args.reuse:
        base.mkdir(parents=True, exist_ok=True)
        text = Path(args.config).read_text(encoding="utf-8")
        for dst, cmd in ((live, [sys.executable, str(HERE.parent / "pipeline.py")]),
                         (alone, [sys.executable, str(HERE / "run_pipeline.py")])):
            cfg = base / f"{dst.name}.yaml"
            cfg.write_text(
                text.replace(f"output: {out}",
                             f"output: {dst.relative_to(ROOT)}"), encoding="utf-8")
            print(f"=== {dst.name} ===", flush=True)
            r = subprocess.run(cmd + [str(cfg)], cwd=ROOT)
            if r.returncode:
                raise SystemExit(f"★ the {dst.name} run failed; nothing to compare")

    print("\n=== comparing products ===")
    bad = 0
    for name, verdict in compare(alone, live):
        if verdict is None:
            print(f"  {name:<52} same")
        else:
            bad += 1
            print(f"  {name:<52} DIFFERS -- {verdict}")
    if bad:
        print(f"\n{bad} product(s) differ. The two programs are not the same method; "
              f"whichever change reached one of them has to reach the other.")
        raise SystemExit(1)
    print("\nevery product matches")


if __name__ == "__main__":
    main()
