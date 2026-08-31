"""Mean spectrum inside a box -- was the sky removed cleanly, and did the source
survive.

One figure per box, statistics on the right. What is drawn is the sky_subtracted cube
step6 wrote, read and averaged over the region with nothing re-fitted (= data − sky,
source still in it). Residuals alone cannot tell good from bad, because subtracting the
source away shrinks them the same way (Principle 1); leaving it in separates the two:

    blank pixels    the line sits on 0  -> the sky was removed cleanly
    source pixels   a galaxy spectrum   -> the source survived; a line pushed down is
                                           over-subtraction

The three lines
---------------
    ESO nosky            external reference (sky subtracted by the ESO pipeline)
    s per-spaxel (old)   every blank spaxel solves its own s
    ours                 s replaced by a smooth spatial field, shared by source and
                         blank alike

How the boxes are chosen
------------------------
By content, not by typed-in coordinates: core/halo from the white-light brightness
inside the main source group, src edge from the distance to that group, blank sampled
uniformly along the same distance. A box must lie entirely inside its class's region
(checked with minimum_filter), or the label and the content would not match. map.png
marks where they landed.

Box mean vs single point
------------------------
--half 0 is a single spaxel, chosen by the same rule but not averaged. A box beats the
noise down as 1/sqrt(N) and is good for the zero point; a single point is noisy, but no
averaging can hide a systematic offset in it. Both write the same filenames, so the
output goes to evaluation/pNN/box/ or evaluation/pNN/point/ accordingly.

    conda run -n astro python src/skymodel/evaluation/box_spectra.py \\
        --work results/skymodel/p01
    conda run -n astro python src/skymodel/evaluation/box_spectra.py \\
        --work results/skymodel/p01 --half 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, load_field, pointing_dir, slug  # noqa: E402
from products import fit_dirs, spectrum_stats  # noqa: E402
from utils import main_source_group, robust_spread  # noqa: E402

Z_HARO = 0.0204
LINES  = [("[O III]", 5006.8), ("Ha", 6562.8), ("[S II]", 6716.4)]



# The colours encode roles: external reference orange, old approach grey, ours blue.
# The labels stay English -- matplotlib's default font has no Chinese glyphs.
COL = {"ESO nosky": "#ff7f0e", "s per-spaxel (old)": "0.35", "ours": "#1f77b4"}
SHORT = {"ESO nosky": "ESO", "s per-spaxel (old)": "old", "ours": "ours"}
# Fallbacks for labels given on the command line: the reference in the same grey as the
# old run, the main one in the same blue, so the roles stay readable either way.
REF_FALLBACK, RUN_FALLBACK = "0.35", "#1f77b4"

# These positions get a large canvas (see the loop). It only gives each channel more
# screen pixels; the layout is unchanged.
BIG_BOXES = {"core"}
BIG_FIG, BIG_DPI = (30, 8.4), 200


def cube_data(path):
    """Read the cube's DATA. step5 writes it in the primary HDU, ESO's is in an HDU
    named DATA."""
    h = fits.open(path, memmap=True)
    hdu = h["DATA"] if "DATA" in h else h[0]
    return h, hdu


def box_mean(hdu, y0, y1, x0, x1):
    """Mean spectrum of one box. NaN spaxels are skipped automatically (nanmean)."""
    sub = np.asarray(hdu.data[:, y0:y1 + 1, x0:x1 + 1], np.float32)
    with np.errstate(invalid="ignore"):
        return np.nanmean(sub.reshape(sub.shape[0], -1), axis=1)


def pick_boxes(seg, white, half, n_blank, edge_targets, margin, step04):
    """Pick the boxes by content.

    Returns ({name: (y0, y1, x0, x1)}, {name: note}), box endpoints included.

    Blank boxes are numbered rather than named after their distance: that distance is
    a different number in every pointing, so it makes a different filename in each.
    #1 is always the nearest, and the distance survives in the note the figure prints.
    """
    main, ids, peak = main_source_group(seg, white, step04)
    valid = white != 0
    size  = 2 * half + 1

    # "the whole box is inside a class's region" = the class's boolean map survives a
    # minimum_filter, which on a boolean map asks "is there any False in this window".
    def whole(m):
        return ndimage.minimum_filter(m.astype(np.uint8), size=size).astype(bool)

    d_main   = ndimage.distance_transform_edt(~main)
    # The edge of the field of view has to be avoided: exposures are few there and
    # step5 writes NaN below 90% coverage, so a box there measures the mosaic boundary.
    # The main source group needs the same cut -- halo takes the faintest place inside
    # it, and the group can reach the edge, faint for want of exposure.
    d_edge   = ndimage.distance_transform_edt(valid)
    far_edge = d_edge > half + margin
    in_main  = whole(main & valid) & far_edge
    in_blank = whole((seg == 0) & valid) & far_edge

    boxes = {}
    notes = {}

    def add(name, cy, cx, note=""):
        boxes[name] = (cy - half, cy + half, cx - half, cx + half)
        notes[name] = note

    # --- main source group: brightest core, faintest fully-inside place (halo) ---
    wm = ndimage.uniform_filter(np.where(valid, white, 0.0), size=size)
    if in_main.any():
        cand = np.flatnonzero(in_main.ravel())
        add("core", *divmod(int(cand[np.argmax(wm.ravel()[cand])]), seg.shape[1]))
        add("halo", *divmod(int(cand[np.argmin(wm.ravel()[cand])]), seg.shape[1]))
    else:               # main source group too small for the box -> put it on the peak
        add("core", *peak)

    # --- the ring just outside the source: where over-subtraction shows ---
    # The box lies entirely in blank, so its centre gets no closer than half+1.
    cand = np.flatnonzero(in_blank.ravel())
    dm   = d_main.ravel()[cand]
    for t in edge_targets:
        j = int(cand[np.argmin(np.abs(dm - t))])
        add(f"src edge d={d_main.ravel()[j]:.0f}px", *divmod(j, seg.shape[1]))

    # --- other sources: the two brightest, to see if only Haro 11 improves ---
    others = [i for i in np.unique(seg[seg > 0]) if i not in ids]
    flux   = {i: float(np.nansum(np.where(seg == i, white, 0))) for i in others}
    n_added = 0
    for i in sorted(flux, key=flux.get, reverse=True):
        if n_added == 2:
            break
        m = (seg == i) & valid
        k = np.unravel_index(np.nanargmax(np.where(m, white, -np.inf)), seg.shape)
        if d_edge[k] <= half + margin:      # avoid the edge of the field of view too
            continue
        add(f"source #{int(i)}", int(k[0]), int(k[1]))
        n_added += 1

    # --- blank: sampled uniformly in distance from the main source group, for the
    #     trend "the farther away, the cleaner" ---
    lo, hi = float(dm.min()), float(dm.max())
    for k, q in enumerate(np.linspace(0.25, 1.0, n_blank), start=1):
        j = int(cand[np.argmin(np.abs(dm - (lo + q * (hi - lo))))])
        add(f"blank #{k}", *divmod(j, seg.shape[1]),
            note=f"d={d_main.ravel()[j]:.0f}px from the main source")
    return boxes, notes, main, peak


def draw_map(white, seg, amplitude_field, boxes, out_path, title):
    """The boxes drawn on the white light image and on the s field -- reporting only
    the coordinates would not show which patch of the sky that is."""
    n = 1 if amplitude_field is None else 2
    fig, axes = plt.subplots(1, n, figsize=(8.5 * n, 8), squeeze=False)
    q0 = max(float(np.nanpercentile(white[white != 0], 20)), 1e-3)
    d  = np.arcsinh(white / q0)
    axes[0][0].imshow(d, origin="lower", cmap="gray_r",
                      vmin=float(np.nanpercentile(d, 30)),
                      vmax=float(np.nanpercentile(d, 99.7)))
    axes[0][0].set_title("whitelight", fontsize=10)
    if amplitude_field is not None:
        # Centred on 1.0, half width 3 x the robust spread. Percentiles would be
        # asymmetric about the centre and narrower, so the same s field would look
        # more strongly striped -- a difference of the colour scale, not of the data.
        v = 3 * robust_spread(amplitude_field[np.isfinite(amplitude_field)])
        im = axes[0][1].imshow(amplitude_field, origin="lower", cmap="RdBu_r",
                               vmin=1 - v, vmax=1 + v)
        plt.colorbar(im, ax=axes[0][1], fraction=0.045, pad=0.01)
        axes[0][1].set_title("s field (new continuum map)", fontsize=10)
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax in axes[0]:
        ax.contour(seg > 0, levels=[0.5], colors="#2ca02c", linewidths=0.45)
        for i, (nm, (y0, y1, x0, x1)) in enumerate(boxes.items()):
            c = cmap[i % 10]
            if y0 == y1 and x0 == x1:
                # a single spaxel drawn as a box is one pixel, invisible at this scale
                ax.plot(x0, y0, "+", color=c, ms=13, mew=2.2)
            else:
                ax.add_patch(mpatches.Rectangle((x0 - .5, y0 - .5), x1 - x0 + 1,
                                                y1 - y0 + 1, fill=False, ec=c,
                                                lw=1.8))
            ax.annotate(nm, (x0 + (x1 - x0) / 2, y1), xytext=(0, 5),
                        textcoords="offset points", color=c, fontsize=8,
                        fontweight="bold", ha="center",
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        ax.set_xlabel("x [pix]")
    axes[0][0].set_ylabel("y [pix]")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def draw_pdf(boxes, notes, wl, tri, out_path, title, smooth_w):
    """One box per page, two panels stacked. A small residual is not good news by
    itself, since subtracting the source away shrinks it too, so read both:

        upper panel   raw and model overplotted -- does the model sit on the data
        lower panel   resid = raw − model, enlarged. Blank should keep only noise; a
                      source should keep its shape, and a flattened one is
                      over-subtraction.

    The panels do not share a y scale: the sky continuum is orders of magnitude larger
    than the residual, so on a common scale the residual would be a line stuck on 0.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out_path) as pdf:
        for nm, b in boxes.items():
            raw, mod, res = tri[nm]
            fig, ax = plt.subplots(2, 1, figsize=(13, 7.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1.25, 1],
                                                "hspace": 0.08})
            # raw thick, model thin on top. At equal widths a hidden curve and a curve
            # that was never drawn look the same, which is the question this panel
            # answers; different widths leave a grey rim where the two coincide.
            ax[0].plot(wl, smooth(raw, smooth_w), lw=1.4, color="0.45",
                       label="raw (wsky)")
            ax[0].plot(wl, smooth(mod, smooth_w), lw=0.5, color="#d62728",
                       label="sky model")
            ax[0].set_ylabel("flux", fontsize=9)
            ax[0].legend(fontsize=9, loc="upper right", framealpha=0.85)
            ax[0].grid(alpha=0.25)

            ax[1].axhline(0, color="0.5", lw=0.7)
            ax[1].plot(wl, smooth(res, smooth_w), lw=0.5, color="#1f77b4",
                       label="residual = raw − model")
            ax[1].set_ylabel("residual", fontsize=9)
            ax[1].set_xlabel("wavelength [$\\AA$]")
            ax[1].legend(fontsize=9, loc="upper right", framealpha=0.85)
            ax[1].grid(alpha=0.25)
            v = res[np.isfinite(res)]
            lo, hi = (float(x) for x in np.percentile(v, [0.5, 99.5]))
            pad = 0.30 * max(hi - lo, 1e-9)
            ax[1].set_ylim(min(lo - pad, 0.0), max(hi + pad, 0.0))
            ax[1].set_xlim(wl[0], wl[-1])
            for _, lam in LINES:
                for a_ in ax:
                    a_.axvline(lam * (1 + Z_HARO), color="0.75", lw=0.5, ls=":")

            st = spectrum_stats(res)
            fig.suptitle(
                f"{title}\n{nm}{'  ' + notes[nm] if notes.get(nm) else ''}"
                f"   y {b[0]}-{b[1]}  x {b[2]}-{b[3]}   "
                f"{(b[1]-b[0]+1)*(b[3]-b[2]+1)} spaxels   |   residual: "
                f"mean {st['mean']:+.3f}  sigma {st['sigma']:.3f}  "
                f"rms_from_zero {st['rms_from_zero']:.3f}", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)
    print(f"saved -> {out_path}")


def smooth(a, w):
    """Moving average used for plotting. Purely to make things legible; none of the
    numbers in the table on the right have been smoothed."""
    if w <= 1:
        return a
    k = np.ones(w) / w
    return np.convolve(np.nan_to_num(a), k, mode="same") / np.convolve(
        np.isfinite(a).astype(float), k, mode="same")


def main():
    ap = argparse.ArgumentParser(description="Mean spectrum inside a box (new basis + s field)")
    ap.add_argument("--work", required=True,
                    help="pointing work directory, e.g. results/skymodel/p01")
    ap.add_argument("--run", default=None,
                    help="alternative run directory under step05; default is the "
                         "pipeline's own step05/step06")
    ap.add_argument("--ref-run", default="none",
                    help="another run directory under step05 to compare against; "
                         "none = do not draw")
    ap.add_argument("--ref-label", default="s per-spaxel (old)",
                    help="legend label for the --ref-run line. This script compares "
                         "any two runs, not just old vs new, so a hard-coded label "
                         "would be misleading")
    ap.add_argument("--run-label", default="ours",
                    help="legend label for the --run line")
    ap.add_argument("--eso", default=None,
                    help="external reference; defaults to the ESO nosky derived "
                         "from the pNN number. none = do not draw")
    ap.add_argument("--half", type=int, default=6, help="box half-width; box width = 2*half+1")
    ap.add_argument("--n-blank", type=int, default=4)
    ap.add_argument("--edge", type=float, nargs="+", default=[7, 20],
                    help="target distance (px) from the main source for outer boxes")
    ap.add_argument("--margin", type=float, default=10,
                    help="minimum distance (px) from the field-of-view edge. Edges "
                         "have low exposure, and step5 writes NaN for spaxels with "
                         "coverage < 90%%")
    ap.add_argument("--ypct", type=float, default=0.5,
                    help="percentile that sets the y range; the range runs from this "
                         "to 100 minus this. Smaller keeps more of the spikes")
    ap.add_argument("--ypad", type=float, default=0.55,
                    help="extra room above and below, as a fraction of that range")
    ap.add_argument("--smooth", type=int, default=1, help="moving-average width for plotting (channels)")
    ap.add_argument("--pdf", action="store_true",
                    help="output PDF: one page per box, upper panel raw vs sky "
                         "model, lower panel residual. Uses only this run's own files "
                         "(raw / model / resid), no ESO or old run needed, so it works "
                         "for every pointing. The raw path is read from the run's "
                         "meta.json")
    ap.add_argument("--out", default=None,
                    help="output directory; default results/skymodel/evaluation/"
                         "pNN/box/. One figure per box, filenames derived from "
                         "box names")
    args = ap.parse_args()

    W    = ROOT / args.work
    # Derived from the working directory: a hard-coded default would silently take
    # another pointing's cube as the reference, on a figure that looks normal.
    if args.eso is None:
        args.eso = f"data/nosky/DATACUBE_FINAL_ESOSKY_{int(W.name[1:])}.fits"
    seg, white, _ = load_field(W)
    s_dir, run = fit_dirs(W, args.run)
    wl    = np.load(W / "step03/wavelength.npy")
    s_file = s_dir / "sky_continuum_amplitude_field.npy"
    amplitude_field = np.load(s_file) if s_file.exists() else None

    # The grouping redshifts must come from the fit being drawn -- a workspace can hold
    # several step4 runs. meta records the classification file's path from the repo
    # root, so its parent is that run; older products name the key "best".
    meta = json.loads((run / "meta.json").read_text())
    step04 = ROOT / Path(meta.get("classification") or meta["best"]).parent
    boxes, notes, main, peak = pick_boxes(seg, white, args.half, args.n_blank,
                                          args.edge, args.margin, step04)
    print(f"main source {int(main.sum()):,} px, brightest pixel (y, x) = {peak}")
    for nm, (y0, y1, x0, x1) in boxes.items():
        print(f"  {nm:<18} y {y0}-{y1}  x {x0}-{x1}   {notes.get(nm, '')}")

    if args.pdf:
        cube_path = ROOT / meta["cube"]
        tri = {}
        for kind, p in (("raw", cube_path), ("mod", run / "sky_model.fits"),
                        ("res", run / "sky_subtracted.fits")):
            h, hdu = cube_data(p)
            for nm, b in boxes.items():
                tri.setdefault(nm, {})[kind] = box_mean(hdu, *b)
            h.close()
            print(f"loaded {kind}  {p.name}")
        tri = {nm: (d["raw"], d["mod"], d["res"]) for nm, d in tri.items()}
        out = (Path(args.out) / "box_raw_model_resid.pdf" if args.out
               else pointing_dir(W, "box") / "box_raw_model_resid.pdf")
        out.parent.mkdir(parents=True, exist_ok=True)
        draw_pdf(boxes, notes, wl, tri, out, f"{args.work}  [{run.name}]", args.smooth)
        draw_map(white, seg, amplitude_field, boxes, out.with_suffix(".map.png"),
                 f"{'box' if args.half else 'point'} locations   {W.name}")
        return

    srcs = {}
    if args.eso.lower() != "none":
        srcs["ESO nosky"] = ROOT / args.eso
    if args.ref_run.lower() != "none":
        srcs[args.ref_label] = fit_dirs(W, args.ref_run)[1] / "sky_subtracted.fits"
    srcs[args.run_label] = run / "sky_subtracted.fits"
    srcs = {k: v for k, v in srcs.items() if v.exists()}

    curves = {nm: {} for nm in boxes}
    for m, p in srcs.items():
        h, hdu = cube_data(p)
        for nm, b in boxes.items():
            curves[nm][m] = box_mean(hdu, *b)
        h.close()
        print(f"loaded {m}")

    # half=0 is a single spaxel, a different kind of sampling, and both write the same
    # filenames -- mixed in one directory the later run would overwrite the earlier.
    kind = "box" if args.half else "point"
    outdir = Path(args.out) if args.out else pointing_dir(W, kind)
    outdir.mkdir(parents=True, exist_ok=True)
    # The label is the key itself: "rms" alone is unambiguous only while sigma is next
    # to it, and a number quoted out of the figure loses that.
    keys = ("mean", "sigma", "rms_from_zero")
    # The statistics column is laid out from a character count, not from fixed
    # coordinates: the number of columns depends on which of --eso / --ref-run are on.
    NW, VW = len(max(keys, key=len)) + 2, 10   # label column width, value column width
    ncol   = len(srcs)
    span   = NW + VW * ncol
    for nm, b in boxes.items():
        # core is drawn large: at the default size several channels share one screen
        # pixel and a MUSE line is narrower than that, so the detail is thrown away
        # before it is drawn. Only core needs it -- the other positions are noise.
        big = nm in BIG_BOXES
        w, h, dpi = (BIG_FIG + (BIG_DPI,)) if big else (15, 4.2, 135)
        sc = w / 15.0        # font size follows the canvas, or it is unreadable when big
        fig = plt.figure(figsize=(w, h))
        # 0.04 is the width fraction of one monospace character at fontsize 8.5, so
        # span sizes the axis: too narrow truncates, too wide strands the numbers.
        gs  = fig.add_gridspec(1, 2, width_ratios=[6, 0.04 * span], wspace=0.02,
                               left=0.055, right=0.985,
                               top=1 - 0.59 / h, bottom=0.55 / h)
        ax = fig.add_subplot(gs[0, 0])
        ax.axhline(0, color="0.5", lw=0.6)
        for lab, y in curves[nm].items():
            ax.plot(wl, smooth(y, args.smooth), lw=0.5, alpha=0.8, label=lab,
                    color=COL.get(lab, REF_FALLBACK if lab == args.ref_label
                                  else RUN_FALLBACK))
        # The range is asymmetric: on a source pixel the whole spectrum is above 0, and
        # forcing symmetry would squash the curve into half the canvas. Percentiles
        # rather than min/max, or a handful of bad channels set the range; 0 stays
        # inside it either way, being the line "clean sky" is read against. The padding
        # is generous because the percentiles choose what sets the scale, not what is
        # visible, and a spike cut off would be a residual sky line.
        v = np.concatenate([y[np.isfinite(y)] for y in curves[nm].values()])
        lo, hi = (float(x) for x in np.percentile(v, [args.ypct, 100 - args.ypct]))
        pad = args.ypad * max(hi - lo, 1e-9)
        ax.set_ylim(min(lo - pad, 0.0), max(hi + pad, 0.0))
        ax.set_xlim(wl[0], wl[-1])
        for lname, lam in LINES:        # redshifted; the sky model cannot remove these
            ax.axvline(lam * (1 + Z_HARO), color="0.75", lw=0.5, ls=":")
        ax.set_ylabel("flux", fontsize=9 * sc)
        ax.set_xlabel("wavelength [$\\AA$]", fontsize=9 * sc)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8 * sc)
        ax.legend(fontsize=9 * sc, loc="upper left", ncol=3, framealpha=0.85)

        sax = fig.add_subplot(gs[0, 1]); sax.axis("off")
        sax.text(0.0, 0.98, "\n".join([""] + list(keys)),
                 va="top", ha="left", family="monospace", fontsize=8.5 * sc,
                 transform=sax.transAxes)
        for j, (lab, y) in enumerate(curves[nm].items()):
            st = spectrum_stats(y)
            sax.text((NW + VW * (j + 1)) / span, 0.98,
                     "\n".join([SHORT.get(lab, lab[:9])]
                               + [f"{st[k]:.3f}" for k in keys]),
                     va="top", ha="right", family="monospace", fontsize=8.5 * sc,
                     color=COL.get(lab, REF_FALLBACK if lab == args.ref_label
                                   else RUN_FALLBACK),
                     transform=sax.transAxes)
        npx = (b[1] - b[0] + 1) * (b[3] - b[2] + 1)
        where = (f"y {b[0]}  x {b[2]}" if npx == 1 else
                 f"y {b[0]}-{b[1]}  x {b[2]}-{b[3]}")
        note = notes.get(nm, "")
        fig.suptitle(f"{W.name}  {nm}   {where}   "
                     f"{npx} spaxel{'' if npx == 1 else 's'}"
                     + (f"   {note}" if note else ""), fontsize=12 * sc)
        o = outdir / f"{slug(nm)}.png"
        fig.savefig(o, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {o}")

    draw_map(white, seg, amplitude_field, boxes, outdir / "map.png",
             f"{kind} locations   {W.name}")


if __name__ == "__main__":
    main()
