"""The s field of every exposure, on one fixed colour scale.

s_shape_map draws one pointing at a time, each on its own robust colour scale. That
gives every map the most contrast it can have, but no two are then on the same ruler:
the same amount of striping is painted a different strength in each, so structure
cannot be compared between exposures.

Here every panel is drawn on one absolute scale, in s itself, and nothing is
recentred, so a difference in overall level and the structure inside one exposure show
up together and a spaxel's colour means the same thing in every panel. The default
limits are the pooled p2/p98 of everything drawn, made symmetric about 1.0, and are
printed. s = 1 is the natural centre -- "this spaxel has exactly the sky continuum the
mean sky has" -- so red and blue read as more and less sky than average.

--which both puts s_free and s_hat on that one scale as well, the only way the pair
can be read against each other: s_hat is the fit and s_free is what it was fitted to,
and two rulers would make that comparison meaningless. It costs contrast in s_hat,
because s_free also carries the per-spaxel solving noise, so the single-kind runs each
on their own pooled scale stay available.

    conda run -n astro python src/skymodel/evaluation/s_compare.py --which both --separate
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL, ROOT, S_CMAP  # noqa: E402
from products import Run, latest_run  # noqa: E402

FIGURES = EVAL / "sfield"

# --which's two kinds -> the file step5 writes for each.
S_FILE = {"hat":  "sky_continuum_amplitude_field.npy",
          "free": "sky_continuum_amplitude_per_spaxel.npy"}


def draw(ax, a, seg, vmin, vmax, color, width, halo):
    """One panel. Returns the image, so a colour bar can be attached to it."""
    im = ax.imshow(a, origin="lower", cmap=S_CMAP, vmin=vmin, vmax=vmax)
    if width > 0:
        # RdBu_r runs dark blue -> white -> dark red, so no single colour is legible
        # against all of it: black weakens in the saturated corners, white weakens near
        # s = 1. A wider line of the opposite tone underneath removes that dependence
        # at the cost of a heavier line, so it is off unless a field saturates.
        if halo:
            ax.contour(seg > 0, levels=[0.5], colors=halo, linewidths=width * 3.0,
                       alpha=0.9)
        ax.contour(seg > 0, levels=[0.5], colors=color, linewidths=width)
    ax.set_axis_off()
    return im


def main():
    ap = argparse.ArgumentParser(description="The s field of every exposure, on one fixed scale")
    ap.add_argument("--pointings", nargs="+", default=[f"p{i:02d}" for i in range(1, 15)])
    ap.add_argument("--root", default="results/skymodel")
    ap.add_argument("--run", default=None,
                    help="glob naming the run directory under step05, e.g. "
                         "'blank_svdK30_*_sfield'. Without it the newest run of each "
                         "pointing is used")
    ap.add_argument("--which", choices=["hat", "free", "both"], default="hat",
                    help="hat: the fitted field mu + a(y) + b(x). free: the per-spaxel "
                         "solution, which only exists in blank. both: draw the two on "
                         "one shared scale, so they can be read against each other")
    ap.add_argument("--vmin", type=float, default=None,
                    help="bottom of the colour scale, the same for every pointing. "
                         "Default is from the pooled p2/p98, symmetric about 1.0")
    ap.add_argument("--vmax", type=float, default=None)
    ap.add_argument("--pct", type=float, default=2.0,
                    help="percentile used for the default limits")
    ap.add_argument("--separate", action="store_true",
                    help="one file per pointing instead of one tiled figure. The scale "
                         "is the same either way, so the files stay comparable")
    ap.add_argument("--colorbar", action="store_true",
                    help="draw the colour bar. Off by default: the scale is in the "
                         "filename and in the printout, so the bar is the only part of "
                         "the canvas that is not the field")
    ap.add_argument("--seg-color", default="black",
                    help="colour of the source outlines. common.SEG_COLOR ('#39ff14') "
                         "is the loud alternative, and it has the property of lying "
                         "outside RdBu_r entirely, so it can never be mistaken for "
                         "the data")
    ap.add_argument("--seg-width", type=float, default=1.2,
                    help="line width of the source outlines; 0 leaves them off")
    ap.add_argument("--seg-halo", default="none",
                    help="colour of the wider line drawn under the outlines so they "
                         "stay visible on both ends of the scale; 'none' turns it off")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--panel", type=float, default=3.6, help="panel width in inches")
    ap.add_argument("--dpi", type=int, default=170)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    kinds = ["hat", "free"] if args.which == "both" else [args.which]
    sets = {k: [] for k in kinds}
    for name in args.pointings:
        W = ROOT / args.root / name
        d = latest_run(W, S_FILE["hat"], "step05", args.run)
        if d is None:
            print(f"  skip {name}: no run with an s field under step05")
            continue
        pointing = Run(W)
        seg, valid = pointing.seg, pointing.valid
        meta = json.loads((d / "meta.json").read_text())
        for k in kinds:
            f = d / S_FILE[k]
            if not f.exists():
                print(f"  skip {name} s_{k}: {f.name} not in {d.name}")
                continue
            a = np.load(f).astype(float)
            a[~valid] = np.nan
            sets[k].append((name, a, seg, d.name, meta))
    got = [r for k in kinds for r in sets[k]]
    if not got:
        raise SystemExit(f"nothing to plot for --which {args.which}")

    # Pooled over every array drawn -- across pointings, and across s_free and s_hat
    # when both are asked for, so neither one's spread becomes the other's ruler.
    pool = np.concatenate([a[np.isfinite(a)] for _, a, _, _, _ in got])
    lo, hi = np.percentile(pool, [args.pct, 100 - args.pct])
    # Symmetric about 1.0. s = 1 means "exactly the mean sky's continuum", so putting
    # it at the middle of a diverging map makes red and blue mean more and less sky.
    r = float(max(abs(lo - 1.0), abs(hi - 1.0)))
    vmin = args.vmin if args.vmin is not None else 1.0 - r
    vmax = args.vmax if args.vmax is not None else 1.0 + r

    print(f"    {'':>5}{'kind':>6}{'median':>9}{'p2':>9}{'p98':>9}{'commit':>10}"
          f"{'s_fix':>7}{'created':>12}   run")
    for k in kinds:
        for n, a, _, run, m in sets[k]:
            v = a[np.isfinite(a)]
            q = np.percentile(v, [args.pct, 100 - args.pct])
            print(f"    {n:>5}{k:>6}{np.median(v):>9.4f}{q[0]:>9.4f}{q[1]:>9.4f}"
                  f"{str(m.get('git_commit'))[:7]:>10}{str(m.get('s_fix')):>7}"
                  f"{str(m.get('created'))[:10]:>12}   {run}")
    src = "given" if args.vmin is not None or args.vmax is not None \
        else f"pooled p{args.pct:g}/p{100 - args.pct:g}, symmetric about 1.0"
    print(f"\n  colour scale {vmin:.4f} to {vmax:.4f}  ({src})")
    out_of = [f"{n}/{k} {100 * np.mean((a[np.isfinite(a)] < vmin) | (a[np.isfinite(a)] > vmax)):.1f}%"
              for k in kinds for n, a, _, _, _ in sets[k]
              if np.mean((a[np.isfinite(a)] < vmin) | (a[np.isfinite(a)] > vmax)) > 0.05]
    if out_of:
        # Saturation is invisible in a diverging map -- it just looks like a strong
        # colour -- so how much of each field is past the end has to be said.
        print(f"  past the end of the scale: {'  '.join(out_of)}")

    # Panels from different runs are not a comparison of the exposures, they are a
    # comparison of the runs. Only this can tell them apart.
    commits = {str(m.get("git_commit"))[:7] for _, _, _, _, m in got}
    fixes   = {str(m.get("s_fix")) for _, _, _, _, m in got}
    label_run = len(commits) > 1 or len(fixes) > 1
    if label_run:
        print(f"\n  ! these come from {len(commits)} code version(s) and {len(fixes)} "
              f"s_fix setting(s) -- differences between them are not only differences "
              f"between exposures")

    halo = None if args.seg_halo in ("none", "") else args.seg_halo

    def bar(fig, im, ax, kind):
        if not args.colorbar:
            return
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.012)
        cb.set_label(f"s$_\\mathrm{{{kind}}}$", fontsize=10)

    # Only a non-default outline goes into the name, so a setting left alone does not
    # rename the figure.
    seg_tag = "" if (args.seg_color, args.seg_width, args.seg_halo) == ("black", 1.2, "none") \
        else f"_seg{args.seg_color.lstrip('#')}w{args.seg_width:g}"
    if args.colorbar:
        seg_tag += "_cb"

    if args.separate:
        for k in kinds:
            for name, a, seg, run, m in sets[k]:
                fig, ax = plt.subplots(figsize=(args.panel * 2.2, args.panel * 2.2))
                bar(fig, draw(ax, a, seg, vmin, vmax, args.seg_color, args.seg_width,
                              halo), ax, k)
                if not args.colorbar:
                    # Nothing outside the axes is left to make room for.
                    fig.subplots_adjust(0, 0, 1, 1)
                # The scale goes into the filename: the same pointing on two scales
                # is two different figures, and one name would let one replace the
                # other. It also shows which files share a ruler without opening them.
                # Built from this panel's own name. The collecting loop above has
                # ended, so a work directory left in one of its variables would be the
                # last pointing's -- and every figure here would land in that one.
                o = (Run(ROOT / args.root / name).figdir("sfield")
                     / f"s_{k}_{vmin:.3f}-{vmax:.3f}{seg_tag}.png")
                fig.savefig(o, dpi=args.dpi, bbox_inches="tight")
                plt.close(fig)
                print(f"  -> {o}")
        return

    for k in kinds:
        rows = int(np.ceil(len(sets[k]) / args.cols))
        fig, axes = plt.subplots(rows, args.cols,
                                 figsize=(args.panel * args.cols, args.panel * rows * 1.02))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[len(sets[k]):]:
            ax.set_axis_off()
        for ax, (name, a, seg, _, m) in zip(axes, sets[k]):
            im = draw(ax, a, seg, vmin, vmax, args.seg_color, args.seg_width, halo)
            t = name if not label_run else (f"{name}   {str(m.get('git_commit'))[:7]}"
                                            f"  s_fix={m.get('s_fix')}")
            ax.set_title(t, fontsize=10 if label_run else 11, pad=3)
        bar(fig, im, axes.tolist(), k)

        out = (Path(args.out) if args.out and len(kinds) == 1
               else FIGURES / f"s_{k}_{vmin:.3f}-{vmax:.3f}_compare.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
