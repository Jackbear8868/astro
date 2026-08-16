"""教授的 segmentation 是用多少 DETECT_THRESH 切的?—— 從遮罩本身反推。

教授只給了 seg.fits 和 pseudo-r,沒給設定檔(Drive 上的 default.sex 寫 1.0,
本機是 1.5,我們實跑是 2.0,三個數字互不相同)。門檻可以直接從遮罩量回來:

SExtractor 的偵測是「**濾波後**的影像 > DETECT_THRESH x sigma_背景」,而 sigma
是從**未濾波**的背景量的(濾波提高 SNR,門檻不跟著縮放)。所以遮罩的**邊界**
剛好落在那條等值線上:

    DETECT_THRESH  ~=  中位{ 濾波後影像(邊界像素) }  /  sigma_未濾波

一步步照著 SExtractor 做,不能省:
    ① 背景用 64 x 64 網格的中位(BACK_SIZE 64),再對網格做 3 x 3 中位濾波
       (BACK_FILTERSIZE 3)—— 用單一常數會被條紋與大尺度梯度帶偏
    ② sigma 從**扣完背景、沒有源**的格量穩健散布
    ③ 濾波核就是 default.conv(3x3 高斯,FWHM 2 px),分子分母各自摺積再相除,
       這樣視野邊緣不會被無效格拉低
    ④ 邊界像素 = seg > 0 而四鄰居中有 seg == 0 的那些

**先驗證再相信**:同一套量法套在我們自己的 seg 上,已知答案是 2.0。量出來對不上
就代表量法有問題,教授那個數字也不能信。

    conda run -n astro python src/skymodel/experiments/measure_seg_thresh.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import scale  # noqa: E402

ROOT   = Path(__file__).resolve().parents[3]
SEGDIR = ROOT / "data/wsky_seg"
KERNEL = np.array([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.0


def background(img, valid, src, size=64, fsize=3):
    """SExtractor 式的網格背景:每格取中位,網格再做中位濾波,最後放大回原尺寸。"""
    ny, nx = img.shape
    gy, gx = int(np.ceil(ny / size)), int(np.ceil(nx / size))
    mesh = np.full((gy, gx), np.nan)
    ok = valid & ~src
    for j in range(gy):
        for i in range(gx):
            m = ok[j * size:(j + 1) * size, i * size:(i + 1) * size]
            v = img[j * size:(j + 1) * size, i * size:(i + 1) * size][m]
            if v.size >= 50:
                mesh[j, i] = np.median(v)
    # 空格用全場中位補,否則中位濾波會把 NaN 擴散出去
    mesh[~np.isfinite(mesh)] = np.nanmedian(mesh) if np.isfinite(mesh).any() \
        else float(np.median(img[ok]))
    mesh = ndimage.median_filter(mesh, size=fsize, mode="nearest")
    return ndimage.zoom(mesh, (ny / gy, nx / gx), order=1)[:ny, :nx]


def measure(img, seg, tag):
    valid = np.isfinite(img) & (img != 0)
    src   = seg > 0
    bg    = background(img, valid, src)
    res   = np.where(valid, img - bg, 0.0)

    sd = float(scale(res[valid & ~src]))              # 未濾波的背景 sigma

    num  = ndimage.convolve(res, KERNEL, mode="nearest")
    den  = ndimage.convolve(valid.astype(float), KERNEL, mode="nearest")
    filt = np.where(den > 0.5, num / np.maximum(den, 1e-9), np.nan)

    # 門檻的等值線落在遮罩「最外一圈源像素」與「緊貼在外的第一圈非源像素」之間
    # —— 前者按定義在門檻之上,後者在門檻之下。只量內圈會系統性高估,所以兩圈
    # 都量,夾出一個區間。
    st = ndimage.generate_binary_structure(2, 1)
    ring_in  = src & ~ndimage.binary_erosion(src, structure=st)
    ring_out = ndimage.binary_dilation(src, structure=st) & ~src
    ring_in  &= valid & np.isfinite(filt)
    ring_out &= valid & np.isfinite(filt)
    if ring_in.sum() < 100 or ring_out.sum() < 100:
        return None
    hi = float(np.median(filt[ring_in] / sd))
    lo = float(np.median(filt[ring_out] / sd))
    return dict(tag=tag, n=int(ring_in.sum()), sd=sd, hi=hi, lo=lo,
                mid=0.5 * (hi + lo))


def prof_image(n):
    p = SEGDIR / f"DATACUBE_FINAL_{n}_pseudo_r_nosky.fits"
    if not p.exists():
        p = SEGDIR / f"DATACUBE_FINAL_{n}_pseudo_r.fits"
    return np.asarray(fits.getdata(p), float), p.name


def main():
    ap = argparse.ArgumentParser(description="從遮罩反推 DETECT_THRESH")
    ap.add_argument("-n", type=int, nargs="+", default=list(range(1, 15)))
    args = ap.parse_args()

    print("=== 驗證:我們自己的 seg(已知 DETECT_THRESH = 2.0)===")
    ours = [("NE", ROOT / "results/skymodel/step01")]
    ours += [(f"p{k:02d}", ROOT / f"results/skymodel/p{k:02d}/step01") for k in (4, 8, 12)]
    for tag, d in ours:
        if not (d / "seg.fits").exists():
            continue
        r = measure(np.asarray(fits.getdata(d / "whitelight.fits"), float),
                    fits.getdata(d / "seg.fits").astype(int), tag)
        if r:
            print(f"  {r['tag']:<5} 邊界 {r['n']:>6,} px   sigma {r['sd']:.4f}   "
                  f"門檻介於 {r['lo']:.2f} - {r['hi']:.2f} sigma   中點 {r['mid']:.2f}")

    print("\n=== 教授的 seg(門檻未知)===")
    vals = []
    for n in args.n:
        img, src = prof_image(n)
        seg = fits.getdata(SEGDIR / f"DATACUBE_FINAL_{n}_seg.fits").astype(int)
        r = measure(img, seg, f"#{n}")
        if r:
            vals.append(r["mid"])
            print(f"  {r['tag']:<5} 邊界 {r['n']:>6,} px   sigma {r['sd']:.4f}   "
                  f"門檻介於 {r['lo']:.2f} - {r['hi']:.2f} sigma   中點 {r['mid']:.2f}"
                  f"   {src}")
    if vals:
        print(f"\n  14 顆中點的中位 {np.median(vals):.2f} sigma   "
              f"範圍 {min(vals):.2f} - {max(vals):.2f}")


if __name__ == "__main__":
    main()
