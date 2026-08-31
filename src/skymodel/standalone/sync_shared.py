"""Copy the shared modules from the live pipeline into this folder.

`utils.py` and `config.py` are libraries, not steps: the live pipeline and the steps
here call the same functions and read the same config, and nothing is gained by writing
either twice. They are copied rather than imported for the reason the tuples in
`step_io.py` are -- importing would put the live pipeline on this folder's import path,
and a step here is supposed to run without it.

A copy needs one change. Both files resolve the repository root from their own
location, and this folder is one level deeper:

    src/skymodel/utils.py             parents[2] -> the repo root
    src/skymodel/standalone/utils.py  parents[3] -> the repo root

so the depth is rewritten as the file is copied. Nothing else is touched.

    python sync_shared.py            copy, reporting what changed
    python sync_shared.py --check    report only, exit 1 if a copy is stale

--check is what a test or a hook calls: it is the difference between "the shared code
is behind" being noticed and being found out by a run that quietly used the old
version. The steps themselves are not synced -- those are the duplicated part, and
what `check_mirror.py` compares.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIVE = HERE.parent
SHARED = ("utils.py", "config.py")


def rendered(name):
    """The live module as it should read in this folder."""
    text = (LIVE / name).read_text(encoding="utf-8")
    if "parents[2]" not in text:
        raise SystemExit(
            f"★ {LIVE / name} no longer resolves the repository root with parents[2]. "
            "Whatever replaced it has to be re-checked for this folder's depth before "
            "the copy can be trusted.")
    return text.replace("parents[2]", "parents[3]")


def main():
    ap = argparse.ArgumentParser(
        description="copy utils.py and config.py from the live pipeline into this folder")
    ap.add_argument("--check", action="store_true",
                    help="report only; exit 1 if a copy is behind the live module")
    args = ap.parse_args()

    stale = []
    for name in SHARED:
        want = rendered(name)
        dst = HERE / name
        have = dst.read_text(encoding="utf-8") if dst.exists() else None
        if have == want:
            print(f"  {name:<12} up to date")
            continue
        stale.append(name)
        if args.check:
            print(f"  {name:<12} STALE -- run sync_shared.py")
        else:
            dst.write_text(want, encoding="utf-8")
            print(f"  {name:<12} copied ({'created' if have is None else 'updated'})")

    if args.check and stale:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
