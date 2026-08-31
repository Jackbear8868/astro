"""The spatial map of the sky continuum amplitude s -- one pointing, or many together.

For every spaxel step5 solves D(λ) = s·C_sky(λ) + Σₖ cₖ·Lₖ(λ). s is the amplitude of
the sky continuum, one number per pixel, and airglow does not change abruptly on scales
of tens of arcsec, so the spatial map of s is a direct check on the sky model.

The fields are files step5 already wrote; nothing is recomputed here.

    s_free   the free per-pixel solution in blank. Only blank has values and the
             sources are holes. It carries the solving noise, and it is where
             over-subtraction starts: source light propping s up grows the sky model,
             which then subtracts that light as if it were sky.
    s_hat    the fitted field mu + a(y) + b(x), trained only on pixels far from the
             sources, so the pixel next to a source has no say. It reaches into the
             holes, since a(y) is shared by a whole row, not a neighbourhood average.

step6 applies this same s_hat minus the spaxels it did not solve (the low-coverage
border), so that field is not drawn separately; which spaxels those were is
np.isfinite of any channel of step06/sky_model.fits.

--scale is the choice this program exists to offer. How many pointings are drawn and
how they are coloured are separate questions: one pointing can be put on the absolute
scale, and several can each be given their own.

    robust   each pointing on its own p2/p98 about its own median. That is the most
             contrast a map can have, and it is how one pointing is read on its own,
             but no two maps are then on the same ruler: the same amount of striping
             is painted a different strength in each, so structure cannot be compared
             between exposures.
    shared   every panel on one absolute scale, in s itself, with nothing recentred,
             so a difference in overall level and the structure inside one exposure
             show up together and a spaxel's colour means the same thing in every
             panel. The limits are the pooled p2/p98 of everything drawn, made
             symmetric about 1.0 -- s = 1 is the natural centre, "this spaxel has
             exactly the sky continuum the mean sky has", so red and blue read as more
             and less sky than average.

--which both puts s_free and s_hat on that one scale as well, the only way the pair can
be read against each other: s_hat is the fit and s_free is what it was fitted to, and
two rulers would make that comparison meaningless. It costs contrast in s_hat, because
s_free also carries the per-spaxel solving noise, so the single-kind runs each on their
own scale stay available.

Where a figure lands says which question it answers. --scale robust writes one file per
panel into that pointing's own evaluation directory, s_free.png and s_hat.png; --scale
shared writes the tiled figure into evaluation/sfield/, or with --separate one file per
pointing under each pointing's own sfield/, both carrying the scale in the name. No
figure gets a colour bar unless asked for -- the scale is printed to the terminal
instead.

    conda run -n astro python src/skymodel/evaluation/s_map.py --work results/skymodel/p01
    conda run -n astro python src/skymodel/evaluation/s_map.py --pointings p01 p02 \\
        --scale shared --which both --separate
"""
import argparse
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL, ROOT, diverging_range, s_panel  # noqa: E402
from products import Run, latest_run  # noqa: E402
from utils import main_source_group  # noqa: E402

FIGURES = EVAL / "sfield"

# --which's two kinds -> the file step5 writes for each.
S_FILE = {"hat":  "sky_continuum_amplitude_field.npy",
          "free": "sky_continuum_amplitude_per_spaxel.npy"}

# What each scale's figure looks like before --seg-color, --seg-width and --dpi
# override it. A robust map is read on its own: a large canvas, a thin faint outline
# that stays out of the way of the field, and the main source group drawn on top so the
# galaxy can be found in it. A shared-scale panel is one of many being compared, at a
# fraction of that size, so its outline has to be firm enough to survive being small
# and every other mark is spent on the field itself.
STYLE = {"robust": dict(dpi=150, color="k",     width=0.4, alpha=0.45, main=True),
         "shared": dict(dpi=170, color="black", width=1.2, alpha=1.0,  main=False)}

# One panel: one pointing's s of one kind, carrying the run it came out of, so nothing
# drawn later has to guess which pointing it belongs to.
Panel = namedtuple("Panel", "kind a run meta main")


def read_s(run, kind):
    """One kind of s field, with everything outside the field of view taken out.

    Outside it s is only what the fit extrapolated onto the padding, and a percentile
    taken over that is a percentile over nothing.
    """
    a = run.s_per_spaxel if kind == "free" else run.s_field
    a[~run.valid] = np.nan
    return a


def collect(args, kinds, style):
    """Every panel to be drawn, one per (pointing, kind), pointing by pointing.

    The two ways of naming a run are the two questions being asked. --work names one
    pointing outright and takes the run under step05 by its exact name, as every other
    single-pointing figure does, and a product missing there is an error. --pointings
    walks several, where no one literal name fits all of them -- so --run is a glob and
    the newest match of each pointing wins -- and one pointing without an s field must
    not stop the rest.
    """
    if args.work:
        runs = [Run(args.work, args.run, args.step04)]
    else:
        runs = []
        for name in args.pointings:
            W = ROOT / args.root / name
            d = latest_run(W, S_FILE["hat"], "step05", args.run)
            if d is None:
                print(f"  skip {name}: no run with an s field under step05")
                continue
            # latest_run answers with the directory; Run wants what that directory is
            # called under step05, and step05 itself is the one it calls "default".
            runs.append(Run(W, None if d.name == "step05" else d.name, args.step04))

    out = []
    for run in runs:
        meta = run.meta(5)
        main = None
        if style["main"]:
            main, _, _ = main_source_group(run.seg, np.where(run.valid, run.white, np.nan),
                                           run.step04)
        for k in kinds:
            # One of many pointings missing its s field must not stop the others; a
            # pointing named outright must have it, and Run's own error names the step
            # that should have written it.
            if not args.work and not (run.fit_dir / S_FILE[k]).exists():
                print(f"  skip {run.name} s_{k}: {S_FILE[k]} not in {run.fit_dir.name}")
                continue
            out.append(Panel(k, read_s(run, k), run, meta, main))
    if not out:
        raise SystemExit(f"nothing to plot for --which {args.which}")
    return out


def bar(fig, im, ax, kind, args):
    """The colour bar, if it was asked for."""
    if not args.colorbar:
        return
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.012)
    cb.set_label(f"s$_\\mathrm{{{kind}}}$", fontsize=10)


def draw_robust(panels, args, style, tag):
    """Each pointing on its own scale, one file per panel."""
    for run in dict.fromkeys(p.run for p in panels):
        group = [p for p in panels if p.run is run]
        # One colour scale for the pointing's panels, or "which is higher" could not be
        # seen. Centre and range come from s_free: it is the raw measurement, and
        # letting its fit set the ruler would hide the fit's own bias. Taken from it
        # even when only s_hat is drawn, because this filename does not carry the
        # scale, so s_hat.png has to mean the same thing however it was asked for.
        lead = next((p.a for p in group if p.kind == "free"), None)
        if lead is None:
            lead = (read_s(run, "free") if (run.fit_dir / S_FILE["free"]).exists()
                    else group[0].a)
        c, lo, hi = diverging_range(lead, pct=args.pct)
        if args.half_width is not None:
            lo, hi = c - args.half_width, c + args.half_width

        # One file each: side by side they would share a canvas width, and the
        # structure in s is a small fraction of its level, so half the width does not
        # resolve it. The shared colour scale is what makes them comparable.
        written = []
        for p in group:
            fig, ax = plt.subplots(figsize=(8.5, 7.6))
            bar(fig, s_panel(ax, p.a, run.seg, lo, hi, style["color"], style["width"],
                             style["alpha"], halo=style["halo"], main=p.main),
                ax, p.kind, args)
            o = run.figdir() / f"s_{p.kind}{tag}.png"
            fig.savefig(o, dpi=style["dpi"], bbox_inches="tight")
            plt.close(fig)
            written.append(o)

        # No colour bar on the figures, so without these lines they cannot be read
        # quantitatively.
        print(f"{run.name}  colour scale {lo:.4f} to {hi:.4f} (centre {c:.4f})")
        a = {p.kind: p.a for p in group}
        stat = []
        if "hat" in a:
            stat.append(f"s_hat median {np.nanmedian(a['hat'][run.valid]):.4f}")
        if "hat" in a and "free" in a:
            d = (a["free"] - a["hat"])[np.isfinite(a["free"]) & np.isfinite(a["hat"])]
            stat.append(f"s_free-s_hat median {np.median(d):+.4f}  spread {np.std(d):.4f}")
        if stat:
            print("  " + "   ".join(stat))
        for o in written:
            print(f"  -> {o}")


def draw_shared(panels, args, style, seg_tag):
    """Every panel on one pooled scale, tiled or one file each."""
    kinds = list(dict.fromkeys(p.kind for p in panels))

    def of(k):
        return [p for p in panels if p.kind == k]

    # Pooled over every array drawn -- across pointings, and across s_free and s_hat
    # when both are asked for, so neither one's spread becomes the other's ruler.
    pool = np.concatenate([p.a[np.isfinite(p.a)] for p in panels])
    lo, hi = np.percentile(pool, [args.pct, 100 - args.pct])
    # Symmetric about 1.0. s = 1 means "exactly the mean sky's continuum", so putting
    # it at the middle of a diverging map makes red and blue mean more and less sky.
    r = float(max(abs(lo - 1.0), abs(hi - 1.0)))
    vmin = args.vmin if args.vmin is not None else 1.0 - r
    vmax = args.vmax if args.vmax is not None else 1.0 + r

    print(f"    {'':>5}{'kind':>6}{'median':>9}{'p2':>9}{'p98':>9}{'commit':>10}"
          f"{'s_fix':>7}{'created':>12}   run")
    for k in kinds:
        for p in of(k):
            v = p.a[np.isfinite(p.a)]
            q = np.percentile(v, [args.pct, 100 - args.pct])
            print(f"    {p.run.name:>5}{k:>6}{np.median(v):>9.4f}{q[0]:>9.4f}{q[1]:>9.4f}"
                  f"{str(p.meta.get('git_commit'))[:7]:>10}{str(p.meta.get('s_fix')):>7}"
                  f"{str(p.meta.get('created'))[:10]:>12}   {p.run.fit_dir.name}")
    src = "given" if args.vmin is not None or args.vmax is not None \
        else f"pooled p{args.pct:g}/p{100 - args.pct:g}, symmetric about 1.0"
    print(f"\n  colour scale {vmin:.4f} to {vmax:.4f}  ({src})")

    def past(p):
        v = p.a[np.isfinite(p.a)]
        return np.mean((v < vmin) | (v > vmax))

    out_of = [f"{p.run.name}/{k} {100 * past(p):.1f}%"
              for k in kinds for p in of(k) if past(p) > 0.05]
    if out_of:
        # Saturation is invisible in a diverging map -- it just looks like a strong
        # colour -- so how much of each field is past the end has to be said.
        print(f"  past the end of the scale: {'  '.join(out_of)}")

    # Panels from different runs are not a comparison of the exposures, they are a
    # comparison of the runs. Only this can tell them apart.
    commits = {str(p.meta.get("git_commit"))[:7] for p in panels}
    fixes   = {str(p.meta.get("s_fix")) for p in panels}
    label_run = len(commits) > 1 or len(fixes) > 1
    if label_run:
        print(f"\n  ! these come from {len(commits)} code version(s) and {len(fixes)} "
              f"s_fix setting(s) -- differences between them are not only differences "
              f"between exposures")

    if args.separate:
        for k in kinds:
            for p in of(k):
                fig, ax = plt.subplots(figsize=(args.panel * 2.2, args.panel * 2.2))
                bar(fig, s_panel(ax, p.a, p.run.seg, vmin, vmax, style["color"],
                                 style["width"], style["alpha"], halo=style["halo"],
                                 main=p.main), ax, k, args)
                if not args.colorbar:
                    # Nothing outside the axes is left to make room for.
                    fig.subplots_adjust(0, 0, 1, 1)
                # The scale goes into the filename: the same pointing on two scales is
                # two different figures, and one name would let one replace the other.
                # It also shows which files share a ruler without opening them. The
                # directory is the drawn panel's own pointing, which it carries with
                # it: a work directory left over from the collecting loop would be the
                # last pointing's, and every figure here would land in that one.
                o = (p.run.figdir("sfield")
                     / f"s_{k}_{vmin:.3f}-{vmax:.3f}{seg_tag}.png")
                fig.savefig(o, dpi=style["dpi"], bbox_inches="tight")
                plt.close(fig)
                print(f"  -> {o}")
        return

    for k in kinds:
        group = of(k)
        rows = int(np.ceil(len(group) / args.cols))
        fig, axes = plt.subplots(rows, args.cols,
                                 figsize=(args.panel * args.cols, args.panel * rows * 1.02))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[len(group):]:
            ax.set_axis_off()
        for ax, p in zip(axes, group):
            im = s_panel(ax, p.a, p.run.seg, vmin, vmax, style["color"], style["width"],
                         style["alpha"], halo=style["halo"], main=p.main)
            t = p.run.name if not label_run else (f"{p.run.name}   "
                                                  f"{str(p.meta.get('git_commit'))[:7]}"
                                                  f"  s_fix={p.meta.get('s_fix')}")
            ax.set_title(t, fontsize=10 if label_run else 11, pad=3)
        bar(fig, im, axes.tolist(), k, args)

        out = (Path(args.out) if args.out and len(kinds) == 1
               else FIGURES / f"s_{k}_{vmin:.3f}-{vmax:.3f}{seg_tag}_compare.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=style["dpi"], bbox_inches="tight")
        plt.close(fig)
        print(f"\nsaved -> {out}")


def main():
    ap = argparse.ArgumentParser(description="The spatial map of the sky continuum amplitude s")
    ap.add_argument("--work", default=None,
                    help="one pointing's work directory, e.g. results/skymodel/p01. "
                         "Without it the pointings named by --pointings are drawn")
    ap.add_argument("--pointings", nargs="+", default=[f"p{i:02d}" for i in range(1, 15)],
                    help="which pointings to draw when --work is not given")
    ap.add_argument("--root", default="results/skymodel")
    ap.add_argument("--which", choices=["hat", "free", "both"], default="both",
                    help="hat: the fitted field mu + a(y) + b(x). free: the per-spaxel "
                         "solution, which only exists in blank. both: draw the two on "
                         "one shared scale, so they can be read against each other")
    ap.add_argument("--scale", choices=["robust", "shared"], default="robust",
                    help="robust: each pointing on its own p2/p98 about its own median, "
                         "the most contrast one map can have. shared: every panel on "
                         "the pooled p2/p98 of everything drawn, symmetric about 1.0, "
                         "which is the only way two exposures can be compared")
    ap.add_argument("--run", default=None,
                    help="the run directory under step05 to read. With --work it is "
                         "that directory's name; with --pointings it is a glob, e.g. "
                         "'blank_svdK30_*_sfield', since no one literal name fits every "
                         "pointing, and without it the newest run of each is used")
    ap.add_argument("--step04", default=None,
                    help="the step4 directory to take the redshifts from, e.g. "
                         "results/skymodel/p01/step04; by default it is read from "
                         "step5's meta.json, which records the run step5 itself used")
    ap.add_argument("--half-width", type=float, default=None,
                    help="--scale robust only: half width of the colour scale, shared "
                         "across pointings. Each map stays centred on its own median -- "
                         "the pointings sit at different airglow levels, and forcing a "
                         "common centre would colour whole fields uniformly -- so this "
                         "makes the amplitude of the structure comparable, not the "
                         "absolute value of s. Default is per-pointing, which gives "
                         "each map the most contrast but no shared ruler")
    ap.add_argument("--vmin", type=float, default=None,
                    help="--scale shared only: bottom of the colour scale, the same for "
                         "every pointing. Default is from the pooled p2/p98, symmetric "
                         "about 1.0")
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--pct", type=float, default=2.0,
                    help="percentile used for the default limits")
    ap.add_argument("--separate", action="store_true",
                    help="one file per pointing instead of one tiled figure. The scale "
                         "is the same either way, so the files stay comparable. --scale "
                         "robust is one file per panel whatever this says, since two "
                         "maps of s do not fit side by side")
    ap.add_argument("--colorbar", action="store_true",
                    help="draw the colour bar. Off by default: the scale is printed, "
                         "and under --scale shared it is in the filename too, so the "
                         "bar is the only part of the canvas that is not the field")
    ap.add_argument("--seg-color", default=None,
                    help="colour of the source outlines; the default follows --scale. "
                         "common.SEG_COLOR ('#39ff14') is the loud alternative, and it "
                         "has the property of lying outside RdBu_r entirely, so it can "
                         "never be mistaken for the data")
    ap.add_argument("--seg-width", type=float, default=None,
                    help="line width of the source outlines, the default following "
                         "--scale; 0 leaves them off")
    ap.add_argument("--seg-halo", default="none",
                    help="colour of the wider line drawn under the outlines so they "
                         "stay visible on both ends of the scale; 'none' turns it off")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--panel", type=float, default=3.6, help="panel width in inches")
    ap.add_argument("--dpi", type=int, default=None, help="the default follows --scale")
    ap.add_argument("--out", default=None,
                    help="where the tiled figure goes, when only one kind is drawn")
    args = ap.parse_args()

    # Each scale takes its limits its own way, and the other's would be silently
    # ignored -- and a colour scale that is not the one that was asked for is a figure
    # that lies.
    if args.scale == "robust" and (args.vmin is not None or args.vmax is not None):
        raise SystemExit("★ --vmin/--vmax are one scale for every panel -- that is "
                         "--scale shared")
    if args.scale == "shared" and args.half_width is not None:
        raise SystemExit("★ --half-width is a width about each map's own median -- "
                         "that is --scale robust")

    # s_free before s_hat where a pointing is read on its own: it is the measurement,
    # s_hat is the fit to it, and the robust scale is taken from it. The pooled table
    # leads with s_hat instead, the field step6 actually applies.
    order = ("free", "hat") if args.scale == "robust" else ("hat", "free")
    kinds = [k for k in order if args.which in (k, "both")]

    style = dict(STYLE[args.scale])
    for key, given in (("color", args.seg_color), ("width", args.seg_width),
                       ("dpi", args.dpi)):
        if given is not None:
            style[key] = given
    style["halo"] = None if args.seg_halo in ("none", "") else args.seg_halo

    # Only a non-default outline goes into the name, so a setting left alone does not
    # rename the figure. Non-default is measured against this scale's own look, or
    # every robust figure would be renamed for not looking like a shared-scale one.
    seg_tag = "" if (style["color"], style["width"], style["halo"]) \
        == (STYLE[args.scale]["color"], STYLE[args.scale]["width"], None) \
        else f"_seg{style['color'].lstrip('#')}w{style['width']:g}"
    if args.colorbar:
        seg_tag += "_cb"

    panels = collect(args, kinds, style)
    if args.scale == "shared":
        draw_shared(panels, args, style, seg_tag)
        return

    # The run goes into the filename because every step05 run writes the same two
    # amplitude files.
    tag = "" if args.run in (None, "default") else f"_{args.run}"
    # --half-width changes the colour scale, so it belongs in the name too.
    if args.half_width is not None:
        tag += f"_hw{args.half_width:g}"
    draw_robust(panels, args, style, tag + seg_tag)


if __name__ == "__main__":
    main()
