"""② 學 basis 時加權(白化 SVD)會不會比較好?

現在 step3 的 SVD 吃的是未加權的殘差,所以它挑的方向是「解釋最多流量變異」
的方向 —— 最亮的幾條天光線把 K 條 basis 吃掉大半。白化之後挑的是「解釋最多
顯著性」的方向。ZAP 就是這樣做的。

逐通道診斷指出的問題正好在這裡:誤差集中在最亮的線(前 100 個通道佔 31%),
而且我們在那幾條線上輸給 ESO 15-32%。

三個變體,差別只在 SVD 之前怎麼縮放通道:

    A  不白化              現行
    B  白化 sigma_stat     用 STAT 的雜訊。但 STAT 對線外低估 37%、線內只
                           低估 8%,相對權重本身就錯了約 27%
    C  白化 sigma_emp      用殘差自己跨 spaxel 的標準差。完全不碰 STAT,
                           等價於「用相關矩陣做 PCA」而不是共變異數矩陣

白化只用在「學 basis」這一步;擬合一律維持未加權。這是刻意的 —— 上一個實驗
(step5 --blank-chi2)已經證明加權擬合會讓零點漂掉、rms 惡化 6%。兩件事要分開,
才知道是哪一步造成差別。

basis 還原回流量空間:白化空間的方向 v 對應到流量空間的 sigma·v。

    conda run -n astro python src/skymodel/experiments/whitened_basis.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import estimate_continuum

ROOT   = Path(__file__).resolve().parents[3]
STEP01 = ROOT / "results/skymodel/step01"
STEP03 = ROOT / "results/skymodel/step03"
WSKY   = ROOT / "data/Haro11_NEpointing_wsky.fits"

SEED, WINDOW, THRESHOLDS, MAX_ITER, CLIP_SIGMA, TILE = 0, 300, (1, 2), 5, 30, 32


def main():
    ap = argparse.ArgumentParser(description="白化 SVD 的效果")
    ap.add_argument("--ks", type=int, nargs="+", default=[20, 30, 40])
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    IN    = np.load(STEP03 / "iter_line_mask.npy")[0]
    bm    = (white != 0) & (seg == 0)
    ny, nx = bm.shape

    with fits.open(WSKY, memmap=True) as h:
        nz = h["DATA"].header["NAXIS3"]
        blank = np.empty((nz, int(bm.sum())), np.float32)
        V     = np.empty((nz, int(bm.sum())), np.float32)
        for j in range(0, nz, 200):
            blank[j:j+200] = np.asarray(h["DATA"].data[j:j+200], np.float32)[:, bm]
            V[j:j+200]     = np.asarray(h["STAT"].data[j:j+200], np.float32)[:, bm]
    ok = np.isfinite(blank).all(axis=0) & np.isfinite(V).all(axis=0)
    blank, V = blank[:, ok].astype(np.float64), V[:, ok]

    yy, xx = np.mgrid[0:ny, 0:nx]
    tr = (((yy // TILE) + (xx // TILE)) % 2).astype(bool)[bm][ok]
    Dtr, Dte = blank[:, tr], blank[:, ~tr]
    print(f"blank {blank.shape[1]:,}   train {Dtr.shape[1]:,}  test {Dte.shape[1]:,}"
          f"   線內 {int(IN.sum())}/{nz} 通道\n")

    # ---- C_sky 與殘差,與改過的 step3 完全一致 ----
    p16, med, p84 = np.percentile(Dtr, [16, 50, 84], axis=1)
    sg   = np.maximum((p84 - p16) / 2, 1e-6)
    keep = np.abs(Dtr - med[:, None]) <= CLIP_SIGMA * sg[:, None]
    mean_sky = (Dtr * keep).sum(axis=1) / keep.sum(axis=1)
    C = estimate_continuum(mean_sky, thresholds=THRESHOLDS,
                           window=WINDOW, max_iter=MAX_ITER)[0]
    R = np.where(keep, Dtr - C[:, None], (med - C)[:, None])

    s_stat = np.sqrt(V[:, tr].astype(np.float64).mean(axis=1))
    s_emp  = R.std(axis=1)
    del V
    print(f"兩種尺度的比值 sigma_emp / sigma_stat:"
          f"線內中位 {np.median((s_emp/s_stat)[IN]):.3f}   "
          f"線外中位 {np.median((s_emp/s_stat)[~IN]):.3f}")
    print("(相差越大,B 和 C 的行為就越不同 —— 那正是 STAT 相對權重錯掉的量)\n")

    VAR = [("A  不白化", None), ("B  白化 sigma_stat", s_stat),
           ("C  白化 sigma_emp", s_emp)]

    print(f"{'變體':>20}{'K':>5}{'test rms':>10}{'test 線內':>11}{'test 線外':>11}"
          f"{'vs A':>9}{'basis 前100通道佔':>19}")
    print("-" * 86)
    # 「前 100 通道」= 逐通道超額變異數最大的 100 個線內通道
    ex = np.where(IN, s_emp ** 2 - s_stat ** 2, 0.0)
    top = np.argsort(ex)[::-1][:100]

    for K in args.ks:
        base = None
        for name, w in VAR:
            X = (R / w[:, None]) if w is not None else R
            B = TruncatedSVD(n_components=K,
                             random_state=SEED).fit(X.T.astype(np.float32)).components_
            if w is not None:
                B = B * w[None, :]                       # 還原回流量空間
                B /= np.linalg.norm(B, axis=1, keepdims=True)
            sky = np.vstack([C, B])
            Rte = Dte - sky.T @ (np.linalg.pinv(sky.T) @ Dte)   # 擬合一律未加權
            rm = float(np.median(np.sqrt(np.mean(Rte ** 2, axis=0))))
            si = float(np.median(Rte[IN].std(axis=0)))
            so = float(np.median(Rte[~IN].std(axis=0)))
            frac = (B[:, top] ** 2).sum() / (B ** 2).sum()
            rel = "" if base is None else f"{100*(rm/base-1):>8.2f}%"
            print(f"{name:>20}{K:>5}{rm:>10.4f}{si:>11.4f}{so:>11.4f}{rel:>9}"
                  f"{100*frac:>18.1f}%", flush=True)
            if base is None:
                base = rm
        print()

    print("「basis 前100通道佔」= K 條 basis 的能量有多少落在最difficult的 100 個")
    print("線內通道上。白化若真的重新分配了自由度,這個數字應該明顯改變。")


if __name__ == "__main__":
    main()
