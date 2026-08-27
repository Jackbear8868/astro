"""The spatial shape of the sky continuum coefficient s -- one figure per pointing.

What s is
---------
For every spaxel step5 solves D(λ) = s·C_sky(λ) + Σₖ cₖ·Lₖ(λ). s is the amplitude of
the sky continuum, one number per pixel. Ideally it should be smooth (airglow does not
change abruptly on scales of tens of arcsec), so the spatial map of s is a direct
check on whether the sky model is reasonable.

What the two figures are (both are files step5 has already written; this script does
not recompute them)
------------------------------------------------------------------------------------
Written into results/skymodel/evaluation/{pNN}/ as s_free.png and s_hat.png, one
each, on a colour scale they share. The figures carry no colour bar, so that scale
is printed to the terminal instead -- read it from there when a number is needed.

    s_free   the free per-pixel solution in blank. Only blank has values, and the
             positions of the sources are holes.
             It carries the solving noise, and **next to a source it gets propped up
             by the source's light** -- which is exactly the entry point of
             over-subtraction: s grows -> the sky model grows -> the source's light
             gets subtracted as if it were sky.
    s_hat    the fitted field mu + a(y) + b(x). It is trained only on the pixels far
             from the sources, so the pixel next to a source has no say, and it
             reaches into the holes (a(y) is a parameter shared by a whole row, not a
             neighbourhood average).

The field step6 actually applies is this same s_hat with the spaxels step6 did not
solve left out (the low-coverage border), so it is not drawn separately -- it would be
the same field with a piece missing. Which spaxels those were is np.isfinite of any
channel of step06/sky_model.fits.

    conda run -n astro python src/skymodel/evaluation/s_shape_map.py --work results/skymodel/p01
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (ROOT, S_CMAP, diverging_range, load_field,  # noqa: E402
                    pointing_dir, step04_tag)
from products import fit_dirs  # noqa: E402
from utils import main_source_group  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Spatial shape of s, one figure per pointing")
    ap.add_argument("--work", required=True, help="pointing work directory, e.g. results/skymodel/p01")
    ap.add_argument("--run", default=None,
                    help="alternative run directory under step05; default is the pipeline's own step05/step06")
    ap.add_argument("--tag", default=None,
                    help="name one step4 run, the part of the classification filename "
                         "after 'classification_'; by default it is read from step5's "
                         "meta.json, which records the run step5 itself used")
    ap.add_argument("--half-width", type=float, default=None,
                    help="half width of the colour scale, shared across pointings. "
                         "Each map stays centred on its own median -- the pointings sit "
                         "at different airglow levels, and forcing a common centre would "
                         "colour whole fields uniformly -- so this makes the amplitude of "
                         "the structure comparable, not the absolute value of s. Default "
                         "is per-pointing, which gives each map the most contrast but no "
                         "shared ruler")
    args = ap.parse_args()

    W = ROOT / args.work
    run, _ = fit_dirs(W, args.run)

    seg, white, valid = load_field(W)
    main, ids, _ = main_source_group(seg, np.where(valid, white, np.nan), W / "step04",
                                     tag=args.tag or step04_tag(W, args.run))

    per_spaxel = np.load(run / "sky_continuum_amplitude_per_spaxel.npy").astype(float)
    field      = np.load(run / "sky_continuum_amplitude_field.npy").astype(float)
    for a in (per_spaxel, field):
        a[~valid] = np.nan

    # The colour scale is shared by both panels, otherwise "which one is higher" could
    # not be seen. Both the centre and the range are decided by s_free -- it is the
    # raw measurement and s_hat is its fit, and letting the fit set the ruler would
    # hide the fit's own bias.
    c, lo, hi = diverging_range(per_spaxel)
    if args.half_width is not None:
        lo, hi = c - args.half_width, c + args.half_width

    # One file each. Side by side the two would have to share a canvas width, and
    # the striping in s is only a couple of percent -- at half the width it is not
    # resolved. The shared colour scale is what makes them comparable, not being
    # printed next to each other.
    # The run goes into the filename. A pointing can hold several step05 runs
    # (p03 has three), they all write the same two amplitude files, and without this
    # the second run's figures silently replace the first's.
    out = pointing_dir(W.name)
    tag = "" if args.run in (None, "default") else f"_{args.run}"
    # --half-width changes the colour scale, which changes what the figure says, so
    # it belongs in the filename for the same reason the run does.
    if args.half_width is not None:
        tag += f"_hw{args.half_width:g}"
    written = []
    for arr, name in ((per_spaxel, f"s_free{tag}.png"), (field, f"s_hat{tag}.png")):
        fig, a = plt.subplots(figsize=(8.5, 7.6))
        a.imshow(arr, origin="lower", cmap=S_CMAP, vmin=lo, vmax=hi)
        a.contour(seg > 0, levels=[0.5], colors="k", linewidths=0.4, alpha=.45)
        a.contour(main,    levels=[0.5], colors="k", linewidths=1.6)
        a.set_axis_off()
        o = out / name
        fig.savefig(o, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(o)

    d = (per_spaxel - field)[np.isfinite(per_spaxel) & np.isfinite(field)]
    # The figures carry no colour bar, so the scale they were drawn on exists
    # nowhere else -- without this line the images cannot be read quantitatively.
    print(f"{W.name}  colour scale {lo:.4f} to {hi:.4f} (centre {c:.4f})")
    print(f"  s_hat median {np.nanmedian(field[valid]):.4f}   "
          f"s_free-s_hat median {np.median(d):+.4f}  spread {np.std(d):.4f}")
    for o in written:
        print(f"  -> {o}")


if __name__ == "__main__":
    main()
