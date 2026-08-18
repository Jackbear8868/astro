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

step5 also writes an s_map (the value actually used at each pixel), which is not drawn
here -- it is identical to s_hat pixel by pixel.

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
from common import ROOT, S_CMAP, diverging_range, load_field, pointing_dir  # noqa: E402
from utils import fit_dirs, main_source_group  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Spatial shape of s, one figure per pointing")
    ap.add_argument("--work", required=True, help="pointing work directory, e.g. results/skymodel/p01")
    ap.add_argument("--run", default=None,
                    help="alternative run directory under step05; default is the pipeline's own step05/step06")
    args = ap.parse_args()

    W = ROOT / args.work
    run, _ = fit_dirs(W, args.run)

    seg, white, valid = load_field(W)
    main, ids, _ = main_source_group(seg, np.where(valid, white, np.nan), W / "step04")

    s_free = np.load(run / "s_free.npy").astype(float)
    s_hat  = np.load(run / "s_hat.npy").astype(float)
    for a in (s_free, s_hat):
        a[~valid] = np.nan

    # The colour scale is shared by both panels, otherwise "which one is higher" could
    # not be seen. Both the centre and the range are decided by s_free -- it is the
    # raw measurement and s_hat is its fit, and letting the fit set the ruler would
    # hide the fit's own bias.
    c, lo, hi = diverging_range(s_free)

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.2))
    for a, arr, ttl in zip(ax, (s_free, s_hat),
                           ("s solved per spaxel",
                            "s fitted = mu + a(y) + b(x)")):
        im = a.imshow(arr, origin="lower", cmap=S_CMAP, vmin=lo, vmax=hi)
        a.contour(seg > 0, levels=[0.5], colors="k", linewidths=0.4, alpha=.45)
        a.contour(main,    levels=[0.5], colors="k", linewidths=1.6)
        a.set_title(ttl, fontsize=12); a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046)

    d = (s_free - s_hat)[np.isfinite(s_free) & np.isfinite(s_hat)]
    fig.suptitle(W.name, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    o = pointing_dir(W.name) / "s_shape.png"
    fig.savefig(o, dpi=125, bbox_inches="tight")
    print(f"{W.name}  s_hat median {np.nanmedian(s_hat[valid]):.4f}   "
          f"s_free-s_hat median {np.median(d):+.4f}  spread {np.std(d):.4f}   -> {o}")


if __name__ == "__main__":
    main()
