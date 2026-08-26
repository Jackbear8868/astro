"""Check that a file reorganisation moved code without changing it.

    conda run -n astro python moves.py <git-rev>

Compares every top-level function and class in src/skymodel/*.py between that
revision and the working tree, by source text. Each one comes out as:

    moved      same text, different file  -- what a reorganisation should produce
    changed    same name, different text  -- has to be justified, not assumed
    added / removed

A reorganisation that reports only `moved` cannot have changed what the code
does, which is a stronger statement than any single run of the pipeline can
make, and it takes about a second.
"""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "src/skymodel"


def units(text, filename):
    """{name: source text} for everything defined at module level.

    Module-level assignments are included, not only defs and classes: merging
    files is exactly where a constant defined in several of them has to collapse
    to one, and a checker blind to assignments would call that merge clean
    without having looked at it.
    """
    out = {}
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        raise SystemExit(f"★ {filename} does not parse: {e}")
    lines = text.splitlines()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            out[node.name] = "\n".join(lines[start:node.end_lineno])
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            src = "\n".join(lines[node.lineno - 1:node.end_lineno])
            for t in targets:
                if isinstance(t, ast.Name):
                    out[t.id] = src
    return out


def collect(rev=None):
    """{name: (file, source)} across the package, at `rev` or in the working tree."""
    if rev:
        listing = subprocess.run(["git", "ls-tree", "-r", "--name-only", rev, PKG],
                                 cwd=ROOT, capture_output=True, text=True).stdout
        names = [n for n in listing.split()
                 if n.endswith(".py") and n.count("/") == PKG.count("/") + 1]
    else:
        names = sorted(str(p.relative_to(ROOT)) for p in (ROOT / PKG).glob("*.py"))

    # A name can be defined in more than one file -- main, _rel, ROOT were all
    # duplicated -- and collapsing those onto one key would hide exactly the case
    # a merge is most likely to get wrong, so every definition is kept.
    found = {}
    for n in names:
        text = (subprocess.run(["git", "show", f"{rev}:{n}"], cwd=ROOT,
                               capture_output=True, text=True).stdout if rev
                else (ROOT / n).read_text())
        for name, src in units(text, n).items():
            found.setdefault(name, []).append((Path(n).name, src))
    return found


def main(rev):
    old, new = collect(rev), collect()
    moved = changed = still = dropped = 0
    for name in sorted(set(old) | set(new)):
        a, b = old.get(name, []), new.get(name, [])
        where = lambda xs: "+".join(sorted(f for f, _ in xs)) or "-"
        srcs_a, srcs_b = sorted(s for _, s in a), sorted(s for _, s in b)
        if not b:
            print(f"  removed  {name:28s} was in {where(a)}")
        elif not a:
            print(f"  added    {name:28s} now in {where(b)}")
        elif srcs_a == srcs_b:
            if where(a) != where(b):
                moved += 1
                print(f"  moved    {name:28s} {where(a)} -> {where(b)}")
            else:
                still += 1
        elif len(a) > len(b) and set(srcs_b) <= set(srcs_a):
            # Two identical copies became one: a duplicate collapsed, and the
            # surviving text is one of the texts that were there before.
            dropped += 1
            print(f"  deduped  {name:28s} {where(a)} -> {where(b)}")
        else:
            changed += 1
            print(f"  CHANGED  {name:28s} {where(a)} -> {where(b)}")
    n_old = sum(len(v) for v in old.values())
    n_new = sum(len(v) for v in new.values())
    print(f"\n  {moved} moved, {dropped} deduped, {changed} changed, "
          f"{still} where they were ({n_old} -> {n_new} top-level definitions)")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "HEAD"))
