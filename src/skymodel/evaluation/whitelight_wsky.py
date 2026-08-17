"""輸入 cube(含天空)的白光影像 —— 天空長什麼樣、遮罩有沒有蓋住星系。

這張畫的是**還沒扣天空的原始資料**,不是我們的產品。它回答兩個進入 pipeline
之前就該確認的問題:

    天空的空間結構    平不平?有沒有條紋?振幅多大?
    遮罩夠不夠大      seg 的輪廓有沒有蓋住星系實際延伸到的範圍 ——
                      沒蓋到的部分會被當成 blank 拿去學天空,等於把星系
                      自己的光學成天空再扣掉

對比度怎麼來的
--------------
天空連續譜是一個約 52 的底座,而我們要看的結構只有它的 1%。直接對原值拉伸的話
底座會把整條色階用掉,圖上就是一片均勻的顏色 —— 不是資料平,是色階被浪費了。

所以先扣掉 blank 的中位、除以 blank 的穩健散布,再做 asinh:

    z = (flux − sky) / sigma_blank

0 就是天空,色階的單位變成「高過天空幾個 sigma」,天空的條紋(約 1 sigma)、
暗源(2-4)、星系核心(約 9)三個量級同時看得見。

壞 voxel 用 common.collapse 剔除(跨 spaxel 的 sigma-clip,只剪 blank),
所以宇宙射線不會在圖上留下假的亮點。

    conda run -n astro python src/skymodel/evaluation/whitelight_wsky.py --work results/skymodel/p01
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, asinh_bar, collapse, load_field, pointing_dir  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="含天空的輸入 cube 的白光影像")
    ap.add_argument("--work", required=True)
    ap.add_argument("--cube", default=None,
                    help="輸入 cube;預設由 pNN 的編號推出 data/wshy/DATACUBE_FINAL_N.fits")
    ap.add_argument("--band", type=float, nargs=2, default=(4600, 9350))
    ap.add_argument("--vmin-sigma", type=float, default=-3.0,
                    help="色階下界,單位是 blank 的 sigma")
    args = ap.parse_args()

    W = ROOT / args.work
    n = int(W.name[1:])
    cube = ROOT / (args.cube or f"data/wshy/DATACUBE_FINAL_{n}.fits")

    seg, _, _ = load_field(W)
    wl = np.load(W / "step03/wavelength.npy")
    img, nbad, ntot = collapse(cube, args.band, wl, seg)
    valid = np.isfinite(img) & (img != 0)
    img = np.where(valid, img, np.nan)

    blank = valid & (seg == 0)
    sky = float(np.nanmedian(img[blank]))
    # 散布用分位數而不是標準差:星系溢出遮罩的部分也落在 blank 裡,
    # 標準差會被它拉大,色階跟著變鬆,條紋就看不見了。
    sig = float(np.nanpercentile(img[blank], 84)
                - np.nanpercentile(img[blank], 16)) / 2
    z = (img - sky) / sig
    print(f"  {cube.name}  剔除壞 voxel {nbad:,}/{ntot:,} ({100*nbad/ntot:.3f}%)")
    print(f"  天空 {sky:.2f}   blank 散布 {sig:.3f} ({100*sig/sky:.2f}% of sky)"
          f"   峰值 {np.nanmax(img):.0f} = 天空的 {np.nanmax(img)/sky:.1f} 倍")

    fig, ax = plt.subplots(figsize=(9.5, 9))
    im = ax.imshow(np.arcsinh(z), origin="lower", cmap="magma",
                   vmin=np.arcsinh(args.vmin_sigma),
                   vmax=np.arcsinh(np.nanmax(z)))
    ax.contour(seg > 0, levels=[0.5], colors="#39ff14", linewidths=0.6, alpha=.8)
    ax.set_xticks([]); ax.set_yticks([])
    asinh_bar(fig, im, ax, "signal above sky   [$\\sigma$ of blank]",
              args.vmin_sigma, float(np.nanmax(z)))
    ax.set_title(f"{W.name}   {cube.name} (with sky)   "
                 f"{args.band[0]:.0f}-{args.band[1]:.0f} A", fontsize=13)
    fig.tight_layout()
    o = pointing_dir(W.name) / "whitelight_wsky.png"
    fig.savefig(o, dpi=150, bbox_inches="tight")
    print(f"saved -> {o}")


if __name__ == "__main__":
    main()
