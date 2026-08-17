"""扣完天空的白光影像 —— ESO 與我們並排。

為什麼看白光影像
----------------
逐環、逐方框的數字告訴你「差多少」,但告訴不了你「差在哪裡」。天空扣得不乾淨
如果是全場一致的偏移,那和「源旁邊被多扣一圈」是完全不同的病,而兩者在區域平均
上可能給出一樣的數字。影像把空間結構攤開。

兩張圖
------
同一個色階,所以「誰比較亮」看得出來 —— 不同色階的話,兩邊的差別會分不清是資料
還是尺規。尺規是 arcsinh(flux / blank 散布),0 就是「天空扣乾淨」的真值,單位是
sigma。不扣任何底座:ESO 留下的那層背景在圖上表現成一片均勻的亮,那是真的資訊。

怎麼判讀
--------
    blank 區       誰比較接近黑,誰的天空就扣得比較乾淨
    源身上與周圍   兩邊亮度相同 = 源都保住了;我們這邊比較暗 = 我們把源扣掉了

兩者的差值印在終端機(全場零點、以及扣掉零點之後源上還剩多少),不畫成圖 ——
差值圖被那個零點主導,色階全給了它,空間結構反而看不見。

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
from common import (ROOT, SEG_COLOR, asinh_bar, collapse, load_field,
                    pointing_dir)  # noqa: E402
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
    # 多個 run 並存時不能默默挑一個 —— 挑錯了圖看起來完全正常
    if not args.run and len(runs) != 1:
        raise SystemExit(f"★ {W/'step05'} 底下有 {len(runs)} 個 *_sfield,請用 --run 指定")
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

    # 兩者的差有一個全場一致的零點:ESO 在整個視場留著一層背景,blank 和源上
    # 都一樣。扣掉它之後,源上剩下的才是「我們比 ESO 多留了多少源的光」。
    # 減的是中位不是平均 —— 源的殘留會把平均拉走。
    d0  = imgs["ours"] - imgs["ESO nosky"]
    off = float(np.nanmedian(d0[valid & (seg == 0)]))
    d   = d0 - off
    print(f"  差:blank 中位 {off:+.4f}(全場零點)")
    print(f"      扣掉零點後  主源上中位 {np.nanmedian(d[main]):+.4f}   "
          f"blank {np.nanmedian(d[valid & (seg == 0)]):+.4f}")

    # 尺規:除以**我們的** blank 散布,不扣任何底座。扣完天空之後 blank 的真值
    # 就是 0,所以 0 可以直接當基準 —— 而 ESO 留下的那層背景會在圖上表現成一片
    # 均勻的亮,那是真的資訊,扣掉底座反而會把它藏起來。
    # 兩張共用同一個尺規,否則「誰比較亮」是色階造成的假象。
    sig = scale(imgs["ours"][valid & (seg == 0)])
    zmax = np.arcsinh(np.nanmax(imgs["ours"][valid]) / sig)
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.8))
    for a, lab in zip(ax, ("ESO nosky", "ours")):
        im = a.imshow(np.arcsinh(imgs[lab] / sig), origin="lower", cmap="magma",
                      vmin=np.arcsinh(-3.0), vmax=zmax)
        a.set_title(lab, fontsize=12)
        asinh_bar(fig, im, a, "signal   [$\\sigma$ of blank]",
                  -3.0, float(np.nanmax(imgs["ours"][valid]) / sig))

    for a in ax:
        # 只畫 segmentation。主源不另外描一條 —— 它和其他源一樣是 seg 裡的源,
        # 額外強調會讓人以為那條線代表別的東西。
        a.contour(seg > 0, levels=[0.5], colors=SEG_COLOR, linewidths=0.5,
                  alpha=.75)
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{W.name}    {args.band[0]:.0f}-{args.band[1]:.0f} A", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    band = ("" if tuple(args.band) == (4600.0, 9350.0)
            else f"_{args.band[0]:.0f}-{args.band[1]:.0f}")
    o = pointing_dir(W.name, "whitelight") / f"compare{band}.png"
    fig.savefig(o, dpi=140, bbox_inches="tight")
    print(f"saved -> {o}")


if __name__ == "__main__":
    main()
