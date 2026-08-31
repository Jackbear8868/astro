"""Every evaluation figure, from one command.

There are twenty-two programs here, each with its own flags. Knowing which of them to
run, in what order, and with what arguments is a separate skill from reading the
figures, and it is not one worth having. This is the front door:

    python src/skymodel/evaluation/run.py --work results/skymodel/p01
    python src/skymodel/evaluation/run.py --work results/skymodel/p01 --only sky
    python src/skymodel/evaluation/run.py --all
    python src/skymodel/evaluation/run.py --work results/skymodel/p01 --list

The list below is the whole of it. Each row says what a figure set is called, which
question it belongs to, the program that draws it, and where it lands -- so adding a
figure is a row, not a change to this file's logic, and "what does this project
measure" is answered by reading the table rather than the twenty-two programs.

`--list` is why the table records where each set lands. A figure older than the
products it was drawn from is not wrong in any way you can see: it is a picture of a
run that no longer exists. Comparing the two dates is the only way to know, and doing
it by hand across a hundred and twenty files per pointing is why nobody does it.

Each program is run as its own process, as you would from a shell. They are meant to
be run that way, matplotlib keeps global state, and one that fails should not take the
rest down with it.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import EVAL  # noqa: E402
from products import Run  # noqa: E402

ALL_POINTINGS = [f"p{i:02d}" for i in range(1, 15)]


class Fig(NamedTuple):
    """One figure set: what draws it, and where it lands.

    scope "pointing" means the program draws one pointing and takes {work} or {n};
    "all" means it draws several at once and takes {pointings} or {ns}.

    root "run" puts `glob` under that pointing's own figure directory, "eval" under
    results/skymodel/evaluation. Both are needed because a figure comparing pointings
    belongs to no one run.

    glob is what the set writes, and is what --list dates. An empty glob is a program
    that only prints.

    everyday=False keeps a set out of a plain run. Those are the ones whose content
    does not depend on the pointing being drawn -- the segmentations the professor
    supplied, the template library step 4 fits against, the slide version of a figure.
    Redrawing them after a run produces the same file, so they are asked for by name.
    """
    name: str
    group: str
    script: str
    scope: str
    args: tuple
    root: str = "run"
    glob: str = ""
    everyday: bool = True


FIGURES = [
    # --- where the sources are, and which of them is the galaxy ----------------
    Fig("prof_seg",     "masking", "prof_seg_maps.py",  "all",      ("-n", "{ns}"),
        "eval", "masking/prof_seg/*.png", everyday=False),
    Fig("seg_id",       "masking", "seg_id_map.py",     "pointing", ("--work", "{work}"),
        "eval", "masking/id_map_{name}_*.png"),
    Fig("seg_slide",    "masking", "seg_slide_map.py",  "pointing", ("--work", "{work}"),
        "eval", "masking/slide_{name}_*.png", everyday=False),
    Fig("main_group",   "masking", "main_group_map.py", "pointing", ("-n", "{n}"),
        "run", "main_group*.png"),
    Fig("main_group_spec", "masking", "main_group_spec.py", "all",  ("-n", "{ns}"), everyday=False),
    Fig("halo_sources", "masking", "halo_sources.py",   "pointing", ("--work", "{work}"),
        "run", "masking/halo_sources.png"),

    # --- what steps 3 and 5 learned the sky to be ------------------------------
    Fig("basis",        "sky", "plot_basis.py",         "pointing", ("--work", "{work}"),
        "run", "basis/*.png"),
    Fig("line_residual","sky", "sky_line_residual.py",  "pointing", ("--work", "{work}"),
        "eval", "sky_basis/line_residual_{name}.png"),
    Fig("continuum",    "sky", "continuum_compare.py",  "all", ("--pointings", "{pointings}"),
        "eval", "sky_basis/continuum_compare.png"),
    Fig("s_shape",      "sky", "s_shape_map.py",        "pointing", ("--work", "{work}"),
        "run", "s_*.png"),
    Fig("s_compare",    "sky", "s_compare.py",          "all", ("--pointings", "{pointings}"),
        "eval", "sfield/*.png"),
    Fig("sky_region",   "sky", "sky_region_map.py",     "pointing", ("--work", "{work}"),
        "run", "sky_region*.png"),

    # --- was the sky removed cleanly, and did the source survive ---------------
    Fig("whitelight_in","subtraction", "whitelight_wsky.py",    "pointing", ("--work", "{work}"),
        "run", "whitelight/wsky*.png"),
    Fig("whitelight",   "subtraction", "whitelight_compare.py", "pointing", ("--work", "{work}"),
        "run", "whitelight/compare*.png"),
    Fig("box",          "subtraction", "box_spectra.py",        "pointing", ("--work", "{work}"),
        "run", "box/*.png"),
    Fig("blank",        "subtraction", "blank_compare.py",      "pointing", ("--work", "{work}"),
        "run", "sky/blank_residual*.png"),
    Fig("blank_noise",  "subtraction", "blank_noise_floor.py",  "pointing", ("--work", "{work}"),
        "run", "sky/blank_noise_floor*.png"),
    Fig("halo",         "subtraction", "halo_spectra.py",       "pointing", ("--work", "{work}"),
        "run", "halo/halo_spectra*.png"),
    Fig("outside",      "subtraction", "outside_compare.py",    "pointing", ("--work", "{work}"),
        "run", "halo/outside_vs_*.png"),
    Fig("halo_all",     "subtraction", "halo_compare.py",       "all", ("--pointings", "{pointings}"),
        "eval", "halo/*.png"),

    # --- what step 4 fitted the sources with, and how well ---------------------
    Fig("templates_galaxy", "fit", "plot_eigen.py", "all", ("--kind", "galaxy"),
        "eval", "templates/eigen_galaxy_*.png", everyday=False),
    Fig("templates_qso",    "fit", "plot_eigen.py", "all", ("--kind", "qso"),
        "eval", "templates/eigen_qso_*.png", everyday=False),
    Fig("templates_star",   "fit", "plot_eigen.py", "all", ("--kind", "star"),
        "eval", "templates/eigen_star_*.png", everyday=False),
    Fig("chi2",         "fit", "chi2_scan.py", "pointing", ("--work", "{work}", "--id", "all"),
        "run", "template_fit/*.png"),
]

GROUPS = ["masking", "sky", "subtraction", "fit"]


def command(fig, work, pointings):
    """The argument list for one figure set, with the placeholders filled in."""
    name = Path(work).name if work else ""
    n = name[1:].lstrip("0") or "0"
    subs = {"work": str(work), "name": name, "n": n,
            "pointings": None, "ns": None}
    out = [sys.executable, str(HERE / fig.script)]
    for a in fig.args:
        if a == "{pointings}":
            out += list(pointings)
        elif a == "{ns}":
            out += [str(int(p[1:])) for p in pointings]
        else:
            out.append(a.format(**{k: v for k, v in subs.items() if v is not None}))
    return out


def figure_root(fig, work):
    """Where this set's `glob` is anchored."""
    return Run(work).figdir() if fig.root == "run" else EVAL


def newest(paths):
    """The most recent modification time among some paths, or None if there are none."""
    ts = [p.stat().st_mtime for p in paths if p.exists()]
    return max(ts) if ts else None


def drawn_from(fig, work):
    """When what this set describes last changed -- the later of the products and the
    program.

    The products, because a figure of a run that has since been redone is a picture of
    something that no longer exists. The program, because the same is true of a figure
    drawn by a version of the code that has since been changed, and that one leaves no
    trace at all on disk.
    """
    steps = sorted(Path(work).glob("step0*/meta.json")) if work else []
    return max(filter(None, [newest(steps), newest([HERE / fig.script])]), default=None)


def status(fig, work):
    """(how many figures, how old they are) as a line for --list."""
    if not fig.glob:
        return "prints only", None
    name = Path(work).name if work else ""
    root = figure_root(fig, work)
    files = sorted(root.glob(fig.glob.format(name=name)))
    if not files:
        return "not drawn", None
    made, src = newest(files), drawn_from(fig, work)
    if src and made < src:
        return f"{len(files):>3} figures  STALE", made
    return f"{len(files):>3} figures", made


def main():
    ap = argparse.ArgumentParser(
        description="Draw every evaluation figure, or say which of them are out of date")
    ap.add_argument("--work", default=None,
                    help="a pointing's output directory, e.g. results/skymodel/p01. "
                         "Without it only the sets that span pointings are run")
    ap.add_argument("--pointings", nargs="+", default=ALL_POINTINGS,
                    help="which pointings the cross-pointing sets cover")
    ap.add_argument("--only", nargs="+", default=None, metavar="NAME",
                    help=f"run only these sets or groups. Groups: {', '.join(GROUPS)}")
    ap.add_argument("--everything", action="store_true",
                    help="also the sets kept out of a plain run: the professor's "
                         "segmentations, the template library, the slide figures. "
                         "They do not change when a pointing is re-run")
    ap.add_argument("--all", action="store_true",
                    help="also run the sets that span pointings. On by default with no "
                         "--work, since those are then the only ones there are")
    ap.add_argument("--list", action="store_true",
                    help="draw nothing; report what has been drawn and what is older "
                         "than the products it describes")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands instead of running them")
    args = ap.parse_args()

    want = set(args.only) if args.only else None
    cross = args.all or args.work is None
    todo = [f for f in FIGURES
            if (want is None or f.name in want or f.group in want)
            and (f.scope == "pointing" and args.work or f.scope == "all" and cross)
            # named explicitly, it is wanted whatever its everyday flag says
            and (f.everyday or args.everything or want is not None)]
    if not todo:
        raise SystemExit("nothing selected -- see --only and --all")

    if args.list:
        if not args.work:
            raise SystemExit("--list reports on one pointing; pass --work")
        run = Run(args.work)
        print(f"{run.name}   figures under {run.figdir()}\n")
        for group in GROUPS:
            rows = [f for f in todo if f.group == group]
            if not rows:
                continue
            print(f"  {group}")
            for f in rows:
                state, made = status(f, args.work)
                when = time.strftime("%Y-%m-%d %H:%M", time.localtime(made)) if made else ""
                print(f"    {f.name:<18}{state:<22}{when}")
            print()
        return

    failed = []
    for i, f in enumerate(todo, 1):
        cmd = command(f, args.work, args.pointings)
        head = f"[{i}/{len(todo)}] {f.group}/{f.name}"
        if args.dry_run:
            print(f"{head}\n    {' '.join(cmd)}")
            continue
        print(f"\n=== {head} " + "=" * max(0, 60 - len(head)))
        if subprocess.run(cmd).returncode:
            failed.append(f.name)

    if not args.dry_run:
        print(f"\n{len(todo) - len(failed)}/{len(todo)} figure sets drawn")
        if failed:
            print("  failed: " + ", ".join(failed))
            raise SystemExit(1)


if __name__ == "__main__":
    main()
