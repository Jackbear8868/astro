"""Step 1, on its own: collapse the cube to a white light image and place the
segmentation on it.

Everything downstream that has to say "where is the source" works on this image rather
than on the cube: the segmentation is checked against it, and the main source is the
blob holding its brightest pixel.

    python step1_whitelight.py --cube data/nosky/DATACUBE_FINAL_ESOSKY_1.fits \
                               --seg  data/wsky_seg/DATACUBE_FINAL_1_seg.fits \
                               --out  results/skymodel/p01/step01

This is the standalone copy of `Pipeline.whitelight` and `Pipeline.place_segmentation`.
It writes what those write, so a later step can be run against its output.
"""

import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import matplotlib
matplotlib.use("Agg")              # must be set before importing pyplot
import matplotlib.pyplot as plt

from step_io import Seg, WhiteLight

ROOT = Path(__file__).resolve().parents[3]

# The offset two headers may describe the same grid within, in pixels. Above it a
# pointing is refused unless its config raises the limit, which is a decision to run on
# headers that disagree.
MAX_GRID_OFFSET = 0.1


def repo_path(p):
    """A path written against the repository root.

    What is recorded then does not depend on where the run was started from.
    """
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT)
    except ValueError:
        return p


def write_meta(out, step, **fields):
    """Write out/meta.json: what the step was given, plus who wrote it and when.

    `step` is passed in rather than read off the calling frame. The live pipeline reads
    it from the frame so it cannot fall behind a method rename; here the steps are
    files with fixed names, and naming the file is what a reader of the products needs.
    """
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, cwd=ROOT)
    meta = dict(step=step,
                created=datetime.datetime.now().isoformat(timespec="seconds"),
                git_commit=head.stdout.strip(), **fields)
    (out / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"meta -> {out / 'meta.json'}")


def whitelight(cube, out, rows=32, keep_intermediate=True):
    """Collapse `cube` along wavelength; return the image and its WCS.

    The filename names the cube because the pointing has two -- a sky-included one and
    this sky-subtracted one -- and an evaluation script collapses the other. rows is
    the number of image rows collapsed at a time: memory and speed only.
    """
    cube, out = Path(cube), Path(out)
    if keep_intermediate:
        out.mkdir(parents=True, exist_ok=True)
    white_fits = out / "whitelight_nosky.fits"
    white_png = out / "whitelight_preview.png"

    with fits.open(cube, memmap=True) as hdul:
        data = hdul["DATA"].data
        # A band of image rows at a time, so nanmean's copy of its input is one band's
        # worth, not the cube's. The split must stay spatial: splitting along
        # wavelength would change each pixel's summation order.
        white = np.concatenate([np.nanmean(data[:, y:y + rows, :], axis=0)
                                for y in range(0, data.shape[1], rows)])
        white = np.nan_to_num(white, nan=0.0)
        # The cube's celestial WCS, the two sky axes without the wavelength one;
        # without it the segmentation check could compare only shapes.
        hdr = WCS(hdul["DATA"].header).celestial.to_header()

    if keep_intermediate:
        fits.writeto(white_fits, white, hdr, overwrite=True)

        fig = plt.figure(figsize=(6, 6))
        plt.imshow(white, origin="lower", cmap="gray",
                   vmin=np.nanpercentile(white, 5),
                   vmax=np.nanpercentile(white, 99))
        plt.colorbar()
        fig.savefig(white_png, dpi=130)
        # Closed explicitly: this runs in-process, so open figures accumulate.
        plt.close(fig)
        print(f"saved -> {white_fits}")

    print(f"white light {white.shape} {white.dtype}")
    return WhiteLight(white, hdr)


def place_segmentation(white, seg_src, out, cube, max_offset=MAX_GRID_OFFSET,
                       keep_intermediate=True):
    """Confirm the segmentation shares a pixel grid with the white light; return it.

    The pipeline does not detect sources. Which spaxels hold one is an input, and the
    only thing checked here is that it describes the same sky as the cube.

    Equal shapes do not prove the same grid, so the check is "where on the sky does
    this pixel point", not a keyword-by-keyword comparison: the seg carries a CD matrix
    while the cube uses PC + CDELT, and their CRPIX differ, both of which a literal
    comparison would report as a mismatch.

    The map is copied next to the white light, so the evaluation scripts have it at a
    fixed path and need to know nothing about where the inputs live. The copy is
    byte-identical, so what it cannot show is that it was checked -- meta.json beside
    it carries the measured offset and the source's checksum, which is what makes the
    copy answer "which segmentation, verified how".

    max_offset above the default is a decision to run anyway on a pointing whose
    headers disagree; it is printed when it is above the default, so the bypass is
    recorded in the log.
    """
    out = Path(out)
    dst = out / "segmentation_input.fits"
    s, hs = fits.getdata(seg_src, header=True)
    if keep_intermediate:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(seg_src, dst)
    w, hw = white.data, white.header
    if s.shape != w.shape:
        raise SystemExit(f"★ seg {s.shape} and white light {w.shape} differ in shape")
    if "CTYPE1" not in hw:
        raise SystemExit("★ the white light carries no WCS -- the cube's DATA "
                         "header has none to copy")

    ny, nx = s.shape
    yy = np.array([0, 0, ny - 1, ny - 1, ny // 2])
    xx = np.array([0, nx - 1, 0, nx - 1, nx // 2])
    ws, ww = WCS(hs).celestial, WCS(hw).celestial
    sep = ws.pixel_to_world(xx, yy).separation(ww.pixel_to_world(xx, yy)).arcsec
    off = sep.max() / (proj_plane_pixel_scales(ww)[0] * 3600)
    if off > max_offset:
        raise SystemExit(f"★ seg and white light grids are {off:.2f} px apart "
                         "(largest of the four corners and the centre); "
                         f"the limit is {max_offset:g} px. Raise "
                         "--max-grid-offset to run anyway")
    print(f"    {len(np.unique(s)) - 1} sources, mask {100 * (s > 0).mean():.1f}%, "
          f"grid offset {off:.3f} px")
    if off > MAX_GRID_OFFSET:
        print(f"    ! grid offset {off:.3f} px exceeds the usual limit "
              f"{MAX_GRID_OFFSET:g} px and was allowed by max_grid_offset "
              f"{max_offset:g}. Anything this pointing produces from sky coordinates "
              f"carries that offset.")
    if keep_intermediate:
        write_meta(
            out, "step1_whitelight.py",
            cube=str(repo_path(cube)),
            seg_source=str(repo_path(seg_src)),
            # The copy beside this file is byte-identical to its source, and the
            # source can be replaced. The digest is what still identifies it then.
            seg_md5=hashlib.md5(Path(seg_src).read_bytes()).hexdigest(),
            grid_offset_px=round(float(off), 4),
            max_grid_offset=max_offset,
            n_sources=int(len(np.unique(s)) - 1),
            mask_fraction=round(float((s > 0).mean()), 4))
    return Seg(s, dst)


def main():
    ap = argparse.ArgumentParser(
        description="cube -> white light image, and the segmentation placed on it")
    ap.add_argument("--cube", type=Path, required=True,
                    help="the sky-subtracted cube the white light is collapsed from")
    ap.add_argument("--seg", type=Path, default=None,
                    help="segmentation map; given, it is checked against the white "
                         "light and copied next to it. Without it only the white "
                         "light is written, which is enough for a look at the field "
                         "but not for any later step")
    # --out has no default. The alternative would be to derive a directory name from
    # the cube filename, but that means feeding the wrong cube silently creates an
    # unrecognisable directory under results/ with no warning.
    ap.add_argument("--out", type=Path, required=True,
                    help="output directory, normally <work>/step01")
    ap.add_argument("--rows", type=int, default=32,
                    help="image rows collapsed at a time; memory and speed only")
    ap.add_argument("--max-grid-offset", type=float, default=MAX_GRID_OFFSET,
                    help="how far apart the seg and white-light grids may be, in "
                         "pixels, before the pointing is refused")
    args = ap.parse_args()

    w = whitelight(args.cube, args.out, rows=args.rows)
    if args.seg:
        place_segmentation(w, args.seg, args.out, args.cube,
                           max_offset=args.max_grid_offset)


if __name__ == "__main__":
    main()
