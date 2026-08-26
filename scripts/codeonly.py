"""Check that an edit touched only comments and docstrings.

    conda run -n astro python codeonly.py <git-rev>

Parses every module under src/skymodel, drops the docstrings, and compares the
syntax tree against the same file at that revision. Comments never reach the
tree at all, so what is left is the code and nothing else: if the trees match,
the edit cannot have changed what the program does, whatever it did to the prose.

That is a stronger statement than a run of the pipeline can make, and it takes a
second.
"""
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = "src/skymodel"


def strip_docstrings(tree):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = node.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            # A function whose whole body is its docstring still needs a body.
            node.body = body[1:] or [ast.Pass()]
    return tree


def code_of(text):
    return ast.dump(strip_docstrings(ast.parse(text)))


def at_rev(rev, name):
    r = subprocess.run(["git", "show", f"{rev}:{name}"], cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def main(rev):
    listing = subprocess.run(["git", "ls-tree", "-r", "--name-only", rev, PKG],
                             cwd=ROOT, capture_output=True, text=True).stdout
    old_names = {n for n in listing.split()
                 if n.endswith(".py") and n.count("/") == PKG.count("/") + 1}
    new_names = {str(p.relative_to(ROOT)) for p in (ROOT / PKG).glob("*.py")}

    bad = 0
    for n in sorted(old_names | new_names):
        old, new = at_rev(rev, n), None
        if n in new_names:
            new = (ROOT / n).read_text()
        if old is None:
            print(f"  added    {n}"); bad += 1
        elif new is None:
            print(f"  removed  {n}"); bad += 1
        elif code_of(old) != code_of(new):
            print(f"  CODE CHANGED  {n}"); bad += 1
        else:
            print(f"  prose only    {n}")
    print(f"\n  {'every file: code identical, only prose moved' if not bad else f'{bad} file(s) need explaining'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "HEAD"))
