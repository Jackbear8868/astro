"""把幾個 step5 的 run 並排成白光圖 —— 灰階/熱色階 + zscale,一般檢視器的觀感。

四個 run **共用同一組拉伸上下限**(從第一個 run 算出來)。這一點是必要的:各自
autoscale 的話,扣得越乾淨的那版拉伸越緊、看起來反而雜訊越大,比較就失去意義。

壞 voxel 用 sigma-clip 剔除,做法與門檻沿用 step3_sky_basis.py 的 mean_sky
(同一通道、跨 spaxel,穩健散布,30 sigma),但**只剪 blank**:源只在少數 spaxel
亮,跨 spaxel 看它本來就是離群值,用同一把尺會把源的核心剪掉。

    conda run -n astro python src/skymodel/evaluation/compare_runs.py \\
        --band 4700 9300 --cmap inferno
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, LinearStretch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import EVAL, ROOT, collapse

STEP01  = ROOT / "results/skymodel/ne_pointing/step01"
STEP03  = ROOT / "results/skymodel/ne_pointing/step03"
STEP05  = ROOT / "results/skymodel/ne_pointing/step05"
FIGURES = EVAL / "masking/edge_oversub"

DEFAULT = [("baseline",            "blank_svdK30_tpl68_s1"),
           ("dilation r=4",        "blank_dil4_svdK30_tpl68_s1"),
           ("sky from x 0-170",    "blank_x0-170_svdK30_tpl68_s1"),
           ("both",                "blank_dil4_x0-170_svdK30_tpl68_s1")]


def main():
    ap = argparse.ArgumentParser(description="並排比較幾個 run 的白光圖")
    ap.add_argument("--run", nargs="+", default=None,
                    help="要比較的 run 資料夾名;省略時用內建的四個")
    ap.add_argument("--label", nargs="+", default=None)
    ap.add_argument("--band", type=float, nargs=2, default=(4700, 9300))
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--seg", default=str(STEP01 / "seg.fits"),
                    help="畫輪廓用的 segmentation。"
                         "這樣四張圖的輪廓一致、比得出遮罩以外的差別")
    ap.add_argument("--plain", action="store_true",
                    help="不畫整體標題,每格只留 --label 的字(拿去簡報時用)")
    ap.add_argument("--outdir", default=str(FIGURES))
    args = ap.parse_args()

    pairs = (list(zip(args.label or args.run, args.run)) if args.run else DEFAULT)
    seg   = fits.getdata(args.seg).astype(int)
    wl    = np.load(STEP03 / "wavelength.npy")
    name  = f"{args.band[0]:.0f}-{args.band[1]:.0f}"

    imgs = {}
    for lab, run in pairs:
        f = STEP05 / run / "sky_subtracted.fits"
        if not f.exists():
            raise SystemExit(f"找不到 {f}")
        imgs[lab], nbad, _ = collapse(f, args.band, wl, seg)
        print(f"{lab:20s} {run:38s} sigma-clip 剔除 {nbad:,}")

    # 共用拉伸:用第一個 run 算,四張都套同一組
    ref = imgs[pairs[0][0]]
    lo, hi = ZScaleInterval().get_limits(ref[np.isfinite(ref)])
    nrm = ImageNormalize(vmin=lo, vmax=hi, stretch=LinearStretch())
    print(f"共用 zscale 上下限:{lo:.3f} 到 {hi:.3f}(取自 {pairs[0][0]})")

    n  = len(pairs)
    nc = 2 if n <= 4 else 3
    nr = int(np.ceil(n / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(7.4 * nc, 6.9 * nr), squeeze=False)
    for a, (lab, run) in zip(ax.ravel(), pairs):
        im = a.imshow(imgs[lab], origin="lower", cmap=args.cmap, norm=nrm)
        a.contour(seg > 0, levels=[0.5], colors="#66ccff", linewidths=0.4)
        a.set_title(lab if args.plain else f"{lab}\n{run}",
                    fontsize=13 if args.plain else 10)
        a.set_xticks([]); a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    for a in ax.ravel()[n:]:
        a.axis("off")
    if not args.plain:
        fig.suptitle(f"sky-subtracted white-light image   {name} A   "
                     f"{args.cmap}, zscale   (all panels share one stretch "
                     f"{lo:.2f} to {hi:.2f})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 1 if args.plain else 0.95))
    out = Path(args.outdir) / f"27_compare{len(pairs)}_{args.cmap}_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
