"""The spatial shape of the sky continuum coefficient s -- one figure per pointing.

For every spaxel step5 solves D(λ) = s·C_sky(λ) + Σₖ cₖ·Lₖ(λ). s is the amplitude of
the sky continuum, one number per pixel, and airglow does not change abruptly on scales
of tens of arcsec, so the spatial map of s is a direct check on the sky model.

Both figures are files step5 already wrote; nothing is recomputed here. They go into
results/skymodel/evaluation/{pNN}/ as s_free.png and s_hat.png on a colour scale they
share, and carry no colour bar -- that scale is printed to the terminal instead.

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
                    pointing_dir, step04_dir)
from products import fit_dirs  # noqa: E402
from utils import main_source_group  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Spatial shape of s, one figure per pointing")
    ap.add_argument("--work", required=True, help="pointing work directory, e.g. results/skymodel/p01")
    ap.add_argument("--run", default=None,
                    help="alternative run directory under step05; default is the pipeline's own step05/step06")
    ap.add_argument("--step04", default=None,
                    help="the step4 directory to take the redshifts from, e.g. "
                         "results/skymodel/p01/step04; by default it is read from "
                         "step5's meta.json, which records the run step5 itself used")
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
    step04 = (Path(args.step04) if args.step04
              else step04_dir(W, args.run) or W / "step04")
    main, ids, _ = main_source_group(seg, np.where(valid, white, np.nan), step04)

    per_spaxel = np.load(run / "sky_continuum_amplitude_per_spaxel.npy").astype(float)
    field      = np.load(run / "sky_continuum_amplitude_field.npy").astype(float)
    for a in (per_spaxel, field):
        a[~valid] = np.nan

    # One colour scale for both, or "which is higher" could not be seen. Centre and
    # range come from s_free: it is the raw measurement, and letting its fit set the
    # ruler would hide the fit's own bias.
    c, lo, hi = diverging_range(per_spaxel)
    if args.half_width is not None:
        lo, hi = c - args.half_width, c + args.half_width

    # One file each: side by side they would share a canvas width, and the structure
    # in s is a small fraction of its level, so half the width does not resolve it.
    # The shared colour scale is what makes them comparable. The run goes into the
    # filename because every step05 run writes the same two amplitude files.
    out = pointing_dir(W)
    tag = "" if args.run in (None, "default") else f"_{args.run}"
    # --half-width changes the colour scale, so it belongs in the name too.
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
    # No colour bar on the figures, so without this line they cannot be read
    # quantitatively.
    print(f"{W.name}  colour scale {lo:.4f} to {hi:.4f} (centre {c:.4f})")
    print(f"  s_hat median {np.nanmedian(field[valid]):.4f}   "
          f"s_free-s_hat median {np.median(d):+.4f}  spread {np.std(d):.4f}")
    for o in written:
        print(f"  -> {o}")


if __name__ == "__main__":
    main()
