"""把 over-subtraction 畫成看得見的白光圖。

前面所有的量都是數字。這支只做一件事:把扣完天空的 cube 沿波長壓成一張影像,
用**對稱、以 0 為中心、而且非常緊**的色階去看 —— 天空扣乾淨的地方應該是白的,
被扣過頭的地方會是藍的,源的光留下來的地方是紅的。

色階是重點。用一般的天文拉伸(0 到 max)只會看到源本體,而 over-subtraction 是
「比零還低一點點」,在那種拉伸下和背景一樣都是黑的。所以這裡的上下限是從
**blank 區的散布**定出來的,不是從影像最大值。

兩個波段各畫一次:
    全段 4700-9300   一般意義的白光圖
    5000-6000        天光線最少、Haro 11 的 [O III]5007(z=0.0204 -> 5109 A)落在
                     這裡,是訊噪比最好的一段

差異圖用 x0-170 那個 run 當基準:「基準 − 某個 run」就是那個 run 比基準多吃掉
的光。

    conda run -n astro python src/skymodel/experiments/oversub_whitelight.py
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.visualization import ZScaleInterval, ImageNormalize, LinearStretch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/ne_pointing/step01"
STEP03  = ROOT / "results/skymodel/ne_pointing/step03"
STEP05  = ROOT / "results/skymodel/ne_pointing/step05"
CUBE    = ROOT / "data/Haro11_NEpointing_wsky.fits"
OUTDIR  = ROOT / "results/skymodel/figures/edge_oversub"

CLIP_SIGMA = 30   # 沿用 step3_sky_basis.py 的門檻與做法

RUNS = [("s free",          "blank_svdK54_src9_s1_seg2s68"),
        ("s=1, full-field", "blank_svdK54_tpl68_s1_bs1"),
        ("s=1, x0-170",     "blankx0-170_svdK54_src68_s1_bs1_seg2s68")]


def collapse(path, band, wl, seg, verbose=False):
    """沿波長壓成一張影像,壞 voxel 先剔除再平均。

    剔除的做法與門檻**沿用 step3_sky_basis.py 的 mean_sky**:同一通道、跨 spaxel,
    中心取中位數、散布取 (p84 - p16) / 2(對離群值穩健),超過 30 sigma 的剔掉。
    這樣就不需要另外挑一個絕對門檻。

    但**只作用在 blank spaxel 上**。step3 那段註解給的理由是「一條天光線在每個
    spaxel 都亮,它的亮度就在該通道的中位數裡,不會被剪掉」—— 這個理由對源
    **不成立**:源只在少數 spaxel 亮,跨 spaxel 看它本來就是離群值。用同一把尺
    去剪會把源的 spaxel 整條剪光,nanmean 變成 NaN。所以源區不剪。
    """
    m = (wl >= band[0]) & (wl < band[1])
    with fits.open(path, memmap=True) as h:
        d = np.asarray(h[0].data[m] if h[0].data is not None else h["DATA"].data[m],
                       np.float32)
    blank = (seg == 0)
    bl    = d[:, blank]                                  # (nchan, n_blank)
    p16, med, p84 = np.nanpercentile(bl, [16, 50, 84], axis=1)
    sg   = np.maximum((p84 - p16) / 2, 1e-6)
    bad  = np.abs(bl - med[:, None]) > CLIP_SIGMA * sg[:, None]
    bl   = np.where(bad, np.nan, bl)
    d    = d.astype(np.float32).copy()
    d[:, blank] = bl
    if verbose:
        print(f"  sigma-clip {CLIP_SIGMA} sigma(只剪 blank):剔除 "
              f"{int(np.nansum(bad)):,} / {bad.size:,} 個元素 "
              f"({100 * np.nanmean(bad):.6f}%)")
    return np.nanmean(d, axis=0)


def show(a, img, seg, v, title, cmap="RdBu_r"):
    im = a.imshow(img, origin="lower", cmap=cmap, vmin=-v, vmax=v)
    a.contour(seg > 0, levels=[0.5], colors="k", linewidths=0.4)
    a.set_title(title, fontsize=9)
    a.set_xticks([]); a.set_yticks([])
    return im


def viewer_figure(raw, imgs, seg, band, out, cmap="inferno", contour=True):
    """一般天文檢視器的樣子:熱色階 + zscale 自動拉伸(DS9 / QFitsView 的觀感)。

    zscale 取影像的中位數附近做線性擬合再外推,對「大片天空 + 少數亮源」這種
    分布特別穩 —— 用 min/max 的話 Haro 11 會把整個範圍撐開,天空全變成一片黑。

    三個扣完天空的 run **共用同一組上下限**(用參考 run 算),否則各自 autoscale
    會讓「比較」失去意義:扣得越乾淨的那版拉伸越緊,看起來反而雜訊越大。
    """
    ref  = imgs["s=1, x0-170"]
    z    = ZScaleInterval()
    lo, hi = z.get_limits(ref[np.isfinite(ref)])
    n_sub  = ImageNormalize(vmin=lo, vmax=hi, stretch=LinearStretch())
    n_raw  = ImageNormalize(raw[np.isfinite(raw)], interval=z, stretch=LinearStretch())

    panels = [("raw data (sky included)", raw, n_raw)] + \
             [(f"sky subtracted — {k}", v, n_sub) for k, v in imgs.items()]
    fig, ax = plt.subplots(2, 2, figsize=(13, 12.5))
    for a, (title, img, nrm) in zip(ax.ravel(), panels):
        im = a.imshow(img, origin="lower", cmap=cmap, norm=nrm)
        if contour:
            a.contour(seg > 0, levels=[0.5], colors="#66ccff", linewidths=0.4)
        a.set_title(title, fontsize=10)
        a.set_xticks([]); a.set_yticks([])
        plt.colorbar(im, ax=a, fraction=0.046, pad=0.02)
    fig.suptitle(f"white-light image   {band[0]:.0f}-{band[1]:.0f} A   "
                 f"{cmap}, zscale stretch"
                 f"   (the 3 sky-subtracted panels share one stretch: "
                 f"{lo:.2f} to {hi:.2f})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    f = out / f"07_whitelight_{cmap}_{band[0]:.0f}-{band[1]:.0f}.png"
    fig.savefig(f, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {f}")


def zoom_figure(imgs, seg, v, band, out, n_show=6, margin=16):
    """放大看甜甜圈:挑出足跡外一圈最負的幾個源,三個 run 並排。

    哪幾個源由程式挑,不用眼睛挑 —— 用眼睛挑等於先看到答案再選證據。
    判準是 s 自由那個 run 在 1-3 px 環上的平均值,最負的前 n_show 個。
    """
    from scipy import ndimage
    dist, (iy, ix) = ndimage.distance_transform_edt(seg == 0, return_indices=True)
    owner = seg[iy, ix]
    ring  = (dist >= 1) & (dist < 3)

    ref = imgs["s free"]
    ny, nx = seg.shape
    ids = [i for i in np.unique(seg[seg > 0]) if i != 1]     # Haro 11 太大,另外看
    score = {}
    for i in ids:
        m = ring & (owner == i)
        y, x = np.nonzero(seg == i)
        # 排除視場邊緣的源。DAR 讓有效視野逐波長平移,邊界一圈本來就有過渡帶,
        # 那裡的藍不是 over-subtraction 而是覆蓋率不足,混進來會蓋過要看的東西。
        edge = (y.min() < margin or x.min() < margin
                or y.max() >= ny - margin or x.max() >= nx - margin)
        if m.sum() < 20 or edge or not np.all(np.isfinite(ref[m])):
            continue
        score[i] = float(np.nanmean(ref[m]))
    pick = sorted(score, key=score.get)[:n_show]
    print(f"可用的源 {len(score)}/{len(ids)}(扣掉邊緣與資料不全的)")
    print("環最負的源:" + "  ".join(f"#{i}({score[i]:+.2f})" for i in pick))

    # 小源的環比 blank 的逐像素散布還小,單看像素本來就看不出來。平滑
    # sigma=1.5 px 讓 ~7 個像素一起發言,雜訊掉約 sqrt(7),環才浮得出來。
    # 這是**顯示用**的處理,前面所有數字都沒有平滑過。
    def smooth(img):
        w = np.isfinite(img).astype(float)
        z = ndimage.gaussian_filter(np.nan_to_num(img) * w, 1.5)
        n = ndimage.gaussian_filter(w, 1.5)
        return np.where(n > 0.3, z / np.maximum(n, 1e-9), np.nan)

    sm = {k: smooth(im) for k, im in imgs.items()}
    cols = [(f"{k}\nraw", imgs[k]) for k in imgs] + \
           [(f"{k}\nsmoothed 1.5px", sm[k]) for k in imgs]
    fig, ax = plt.subplots(len(pick), len(cols), squeeze=False,
                           figsize=(2.8 * len(cols), 2.9 * len(pick)))
    for row, i in enumerate(pick):
        y, x = np.nonzero(seg == i)
        y0, y1 = max(y.min() - margin, 0), min(y.max() + margin + 1, seg.shape[0])
        x0, x1 = max(x.min() - margin, 0), min(x.max() + margin + 1, seg.shape[1])
        for col, (name, img) in enumerate(cols):
            a  = ax[row][col]
            vv = v if col < len(imgs) else v / 3      # 平滑過雜訊小,色階跟著收緊
            a.imshow(img[y0:y1, x0:x1], origin="lower", cmap="RdBu_r",
                     vmin=-vv, vmax=vv)
            a.contour(seg[y0:y1, x0:x1] == i, levels=[0.5], colors="k", linewidths=0.8)
            a.set_xticks([]); a.set_yticks([])
            if row == 0:
                a.set_title(name, fontsize=9)
            if col == 0:
                a.set_ylabel(f"#{i}   {int((seg == i).sum())} px", fontsize=9)
    fig.suptitle(f"the donut, zoomed   {band[0]:.0f}-{band[1]:.0f} A   "
                 f"blue ring = source light removed from around the source\n"
                 f"left 3 = raw (+-{v:.2f}), right 3 = smoothed for display only "
                 f"(+-{v/3:.2f})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    f = out / f"06_donut_zoom_{band[0]:.0f}-{band[1]:.0f}.png"
    fig.savefig(f, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {f}")


def main():
    ap = argparse.ArgumentParser(description="over-subtraction 的白光圖")
    ap.add_argument("--band", type=float, nargs=2, action="append", default=None,
                    help="可以給多次;預設全段與 5000-6000 各畫一張")
    ap.add_argument("--nsig", type=float, default=6.0,
                    help="色階上下限 = blank 區標準差的幾倍。預設 6 —— 夠緊才看得到"
                         "「比零低一點」的區域,太緊則源本體整片飽和、看不出結構")
    ap.add_argument("--cmap", default="inferno",
                    help="檢視器那張圖的色階。inferno / magma(黑-紫-橘-黃)、"
                         "afmhot / hot(DS9 的 heat:黑-紅-黃-白)、gray")
    ap.add_argument("--outdir", default=str(OUTDIR))
    args = ap.parse_args()

    bands = args.band or [(4700, 9300), (5000, 6000)]
    out   = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    seg   = fits.getdata(STEP01 / "seg.fits").astype(int)
    wl    = np.load(STEP03 / "wavelength.npy")

    for band in bands:
        raw  = collapse(CUBE, band, wl, seg)
        imgs = {name: collapse(STEP05 / r / "sky_subtracted.fits",
                               band, wl, seg, verbose=True)
                for name, r in RUNS}
        base = imgs["s=1, x0-170"]

        # 色階從 blank 區的散布定出來 —— 不是影像最大值。用中位絕對偏差換算 sigma,
        # 對殘留的源光免疫。
        blank = (seg == 0) & np.isfinite(base)
        sig   = 1.4826 * np.median(np.abs(base[blank] - np.median(base[blank])))
        v     = args.nsig * sig
        print(f"{band[0]:.0f}-{band[1]:.0f} A   blank sigma {sig:.4f}   色階 +-{v:.4f}")

        fig, ax = plt.subplots(2, 3, figsize=(16, 11))
        # 上排:各 run 扣完天空的樣子。原始資料含天空,不能用同一個色階,單獨拉伸。
        im0 = ax[0][0].imshow(raw, origin="lower", cmap="magma",
                              vmin=np.nanpercentile(raw, 1),
                              vmax=np.nanpercentile(raw, 99.5))
        ax[0][0].contour(seg > 0, levels=[0.5], colors="w", linewidths=0.4)
        ax[0][0].set_title(f"raw data (sky included)   {band[0]:.0f}-{band[1]:.0f} A",
                           fontsize=9)
        ax[0][0].set_xticks([]); ax[0][0].set_yticks([])
        plt.colorbar(im0, ax=ax[0][0], fraction=0.046, pad=0.02)
        for a, (name, _) in zip(ax[0][1:], RUNS[:2]):
            plt.colorbar(show(a, imgs[name], seg, v, f"sky subtracted — {name}"),
                         ax=a, fraction=0.046, pad=0.02)
        # 下排:基準本身,以及「基準 − 某個 run」= 那個 run 多吃掉的光
        plt.colorbar(show(ax[1][0], base, seg, v,
                          f"sky subtracted — {RUNS[2][0]}  (reference)"),
                     ax=ax[1][0], fraction=0.046, pad=0.02)
        for a, (name, _) in zip(ax[1][1:], RUNS[:2]):
            plt.colorbar(show(a, base - imgs[name], seg, v,
                              f"light eaten by  {name}\n(reference - run)"),
                         ax=a, fraction=0.046, pad=0.02)
        fig.suptitle(f"over-subtraction seen directly   {band[0]:.0f}-{band[1]:.0f} A"
                     f"   colour scale +-{args.nsig:g} sigma of blank sky"
                     f"   (blue = removed too much)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        f = out / f"05_whitelight_{band[0]:.0f}-{band[1]:.0f}.png"
        fig.savefig(f, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {f}")

        viewer_figure(raw, imgs, seg, band, out, cmap=args.cmap)
        zoom_figure(imgs, seg, v, band, out)


if __name__ == "__main__":
    main()
