"""扣完天空的白光影像 —— ESO、我們、以及兩者的差。

為什麼看白光影像
----------------
逐環、逐方框的數字告訴你「差多少」,但告訴不了你「差在哪裡」。天空扣得不乾淨
如果是全場一致的偏移,那和「源旁邊被多扣一圈」是完全不同的病,而兩者在區域平均
上可能給出一樣的數字。影像把空間結構攤開。

三張圖各是什麼
--------------
    ESO / ours   同一個色階,所以「誰比較亮」看得出來。用的是 asinh 拉伸:
                 扣完天空之後主星系仍然比背景亮好幾個量級,線性顯示只看得到核心。
    ours - ESO   發散色階、以 0 為中心。**這張才是重點** —— 紅色代表我們留下的
                 光比 ESO 多,藍色代表我們扣得比較多。

怎麼判讀
--------
源身上和源周圍偏紅、blank 區接近白 = 我們少扣了源的光而天空扣得一樣乾淨,
那是我們要的。blank 區整片偏紅 = 我們天空扣得比較少,不是好事。

    conda run -n astro python src/skymodel/evaluation/whitelight_compare.py --work results/skymodel/p01
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (ROOT, S_CMAP, arcsinh_stretch, collapse, diverging_range,
                    load_field, pointing_dir)  # noqa: E402
from utils import main_source_group, scale  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="扣完天空的白光影像:ESO vs 我們")
    ap.add_argument("--work", required=True)
    ap.add_argument("--run", default=None, help="預設取唯一的 *_sfield")
    ap.add_argument("--eso", default=None,
                    help="ESO 的 nosky cube;預設由 pNN 的編號推出來")
    ap.add_argument("--band", type=float, nargs=2, default=(4600, 9350))
    args = ap.parse_args()

    W = ROOT / args.work
    runs = sorted((W / "step05").glob("*_sfield"))
    run = W / "step05" / args.run if args.run else runs[0]
    n = int(W.name[1:])
    eso = ROOT / (args.eso or f"data/nosky/DATACUBE_FINAL_ESOSKY_{n}.fits")

    seg, white, valid = load_field(W)
    main, ids, _ = main_source_group(seg, np.where(valid, white, np.nan),
                                     W / "step04")
    wl = np.load(W / "step03/wavelength.npy")

    imgs = {}
    for lab, p in (("ESO nosky", eso), ("ours", run / "sky_subtracted.fits")):
        img, nbad, ntot = collapse(p, args.band, wl, seg)
        imgs[lab] = np.where(valid, img, np.nan)
        print(f"  {lab:>10}  剔除壞 voxel {nbad:,}/{ntot:,} ({100*nbad/ntot:.3f}%)")

    # 兩者的差裡有一個全場一致的零點:ESO 在整個視場留著一層背景。那一層在
    # blank 和源上都一樣,所以它會把色階整個吃掉,真正要看的空間結構反而看不見。
    # 減掉 blank 的中位之後,0 的意思變成「和背景一樣」,源上偏紅才是「我們多留
    # 了源的光」。減的是中位不是平均 —— 源的殘留會把平均拉走。
    d0  = imgs["ours"] - imgs["ESO nosky"]
    off = float(np.nanmedian(d0[valid & (seg == 0)]))
    d   = d0 - off
    print(f"  差:blank 中位 {off:+.4f}(全場零點,已從第三張圖扣掉)")
    print(f"      扣掉零點後  主源上中位 {np.nanmedian(d[main]):+.4f}   "
          f"blank {np.nanmedian(d[valid & (seg == 0)]):+.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(19, 6.4))
    # 前兩張共用同一個拉伸,否則「誰比較亮」是色階造成的假象。基準用 ESO ——
    # 它是外部參考,拿我們自己的當基準會把我們的偏差藏進尺規裡。
    _, vmax = arcsinh_stretch(imgs["ESO nosky"], valid)
    q0 = max(float(np.nanpercentile(imgs["ESO nosky"][valid], 99.5)), 1e-3)
    for a, lab in zip(ax[:2], ("ESO nosky", "ours")):
        im = a.imshow(np.arcsinh(imgs[lab] / (0.02 * q0)), origin="lower",
                      cmap="magma", vmin=0, vmax=vmax)
        a.set_title(lab, fontsize=12)
        fig.colorbar(im, ax=a, fraction=0.046)

    c, lo, hi = diverging_range(d[valid], centre=0.0)
    im = ax[2].imshow(d, origin="lower", cmap=S_CMAP, vmin=lo, vmax=hi)
    ax[2].set_title(f"(ours - ESO) - {off:+.3f}      red = we keep more light",
                    fontsize=12)
    fig.colorbar(im, ax=ax[2], fraction=0.046)

    for a in ax:
        a.contour(seg > 0, levels=[0.5], colors="#2ca02c", linewidths=0.4,
                  alpha=.6)
        a.contour(main, levels=[0.5], colors="k", linewidths=1.3)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{W.name}    {args.band[0]:.0f}-{args.band[1]:.0f} A", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    o = pointing_dir(W.name) / "whitelight.png"
    fig.savefig(o, dpi=140, bbox_inches="tight")
    print(f"saved -> {o}")


if __name__ == "__main__":
    main()
