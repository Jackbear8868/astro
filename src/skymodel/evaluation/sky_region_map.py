"""The regions the sky is learned from -- which spaxels each of the two stages used.

There are **two** different "regions the sky is learned from" in the pipeline, with
different masks, and they must not be conflated:

    step3  sky basis      blank = inside the field of view & seg == 0, then the
                          --xlim / --ylim / --exclude-box selected visually for each
                          pointing. **No dilation.**
    step5  s spatial      on top of the above, additionally requires > r_far from
           field          **any** source and > r_far_haro from the main source group,
                          and rejects the pixels with |s − median| > clip x spread.
                          This layer is the one that "avoids the unstable regions".

Why the two layers use different criteria
-----------------------------------------
The basis learns the **shape** of the sky emission lines, a sky emission line looks
the same in every blank spaxel, and the source's PSF wings mixing in has only a
limited effect, so it only needs to avoid the halo of the main source (that is what
--xlim/--ylim are doing).

s learns the **amplitude** of the sky continuum, and the continuum of the galaxy is
too similar in shape to the sky continuum -- as soon as a training point picks up a
little source light, s gets propped up, the sky model grows with it, and subtracting
it eats the source. So it has to back off further, and it has to reject the pixels
where the fit failed.

The region parameters come from step3's own meta.json, so the figure can never
disagree with the settings that produced the products it is drawn from.

    conda run -n astro python src/skymodel/evaluation/sky_region_map.py --work results/skymodel/p01
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, arcsinh_stretch, load_field, pointing_dir  # noqa: E402
from utils import build_s_field, fit_dirs, main_source_group  # noqa: E402


def region_args(step03):
    """The spatial restrictions step3 actually applied, read from its meta.json.

    Nothing is copied here: a second copy of the numbers would let one side be
    edited without the other, and the figure would then disagree with the settings
    that produced the products, while looking perfectly normal.
    """
    m = json.loads((step03 / "meta.json").read_text())
    return {k: m[k] for k in ("xlim", "ylim", "exclude_box") if m.get(k)}


def main():
    ap = argparse.ArgumentParser(description="Sky learning regions: spaxels used by basis and s field")
    ap.add_argument("--work", required=True)
    ap.add_argument("--run", default=None,
                    help="alternative run directory under step05; default is the pipeline's own step05/step06")
    args = ap.parse_args()

    W = ROOT / args.work
    run, _ = fit_dirs(W, args.run)

    seg, white, valid = load_field(W)
    reg = region_args(W / "step03")

    # --- step3: blank, then the spatial restrictions on top ---
    blank = valid & (seg == 0)
    keep = np.ones_like(blank)
    yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
    if "xlim" in reg:
        keep &= (xx >= reg["xlim"][0]) & (xx < reg["xlim"][1])
    if "ylim" in reg:
        keep &= (yy >= reg["ylim"][0]) & (yy < reg["ylim"][1])
    if "exclude_box" in reg:
        y0, y1, x0, x1 = reg["exclude_box"]
        keep &= ~((yy >= y0) & (yy <= y1) & (xx >= x0) & (xx <= x1))
    basis_train = blank & keep

    # --- step5: back further off the sources, and reject the pixels where the fit
    #     failed ---
    p = json.loads((run / "meta.json").read_text())["s_field_params"]
    s = np.load(run / "s_free.npy").astype(float)
    main, ids, _ = main_source_group(seg, np.where(valid, white, np.nan), W / "step04")
    ok = valid & (seg == 0) & np.isfinite(s)
    _, s_train = build_s_field(s, seg, ok, p["r_far"], p["r_far_haro"], p["clip"],
                               main=main)

    img, vmax = arcsinh_stretch(white, valid)
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 7.4))
    for a, m, ttl in (
        (ax[0], basis_train,
         f"step3  sky basis     {int(basis_train.sum()):,} spaxels"),
        (ax[1], s_train,
         f"step5  s field       {int(s_train.sum()):,} spaxels")):
        a.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=vmax)
        rgba = np.zeros(m.shape + (4,))
        rgba[m] = [0.13, 0.83, 0.08, 0.45]
        a.imshow(rgba, origin="lower")
        a.contour(seg > 0, levels=[0.5], colors="#39ff14", linewidths=0.4, alpha=.6)
        a.set_title(ttl, fontsize=12)
        a.set_xticks([]); a.set_yticks([])

    # the spatial restrictions are drawn as boxes, so that "this line was set by hand"
    # is visible rather than looking like a boundary of the data
    ny, nx = seg.shape
    if "xlim" in reg or "ylim" in reg:
        x0, x1 = reg.get("xlim", [0, nx])
        y0, y1 = reg.get("ylim", [0, ny])
        ax[0].add_patch(mpatches.Rectangle((x0 - .5, y0 - .5), min(x1, nx) - x0,
                                           min(y1, ny) - y0, fill=False,
                                           ec="#ff7f0e", lw=2.0, ls="--"))
    if "exclude_box" in reg:
        y0, y1, x0, x1 = reg["exclude_box"]
        ax[0].add_patch(mpatches.Rectangle((x0 - .5, y0 - .5), x1 - x0 + 1,
                                           y1 - y0 + 1, fill=False,
                                           ec="#e8272c", lw=2.0, ls="--"))

    lim = "  ".join(f"{k} {v}" for k, v in reg.items()) or "(無空間限制)"
    fig.suptitle(f"{W.name}    step3: {lim}    "
                 f"step5: > {p['r_far']:.0f} px from any source, "
                 f"> {p['r_far_haro']:.0f} px from Haro 11, "
                 f"clip {p['clip']:.0f} sigma", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    o = pointing_dir(W.name) / "sky_region.png"
    fig.savefig(o, dpi=140, bbox_inches="tight")
    print(f"{W.name}  blank {int(blank.sum()):,} -> basis {int(basis_train.sum()):,}"
          f" ({100*basis_train.sum()/blank.sum():.1f}%)"
          f" -> s field {int(s_train.sum()):,}"
          f" ({100*s_train.sum()/blank.sum():.1f}%)   -> {o}")


if __name__ == "__main__":
    main()
