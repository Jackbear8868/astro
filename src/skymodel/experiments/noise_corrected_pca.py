"""把雜訊底線從共變異數裡減掉再取特徵向量

SVD 的定義是「找變異數最大的方向」,它分不出哪些變異數是訊號、哪些是雜訊。
而這份資料的雜訊極度異方差 —— 亮天光線通道的光子雜訊比連續譜區大兩個數量級,
正好就是 basis 集中的地方。

因子分析的標準做法是把雜訊底線先減掉:

    二階矩   M = R Rᵀ / n = m mᵀ + Cov_訊號 + diag(v)
    修正後   M − alpha · diag(v)      再取前 K 個特徵向量

這和白化不是同一件事:
    白化      特徵分解 D^(-1/2) M D^(-1/2)   重新縮放度量,訊號也被扭曲
    雜訊修正  特徵分解 M − alpha·diag(v)      只減掉雜訊底線,訊號方向不動

alpha 是「STAT 佔真實雜訊的比例」。STAT 不見得等於真實的雜訊,所以不硬性設
alpha=1,而是掃描讓留出資料決定 —— alpha=0 就是現行做法。

自由度不變(仍是 K 條 basis),所以不會有吃掉源的風險。

    conda run -n astro python src/skymodel/experiments/noise_corrected_pca.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import estimate_continuum

ROOT   = Path(__file__).resolve().parents[3]
STEP01 = ROOT / "results/skymodel/ne_pointing/step01"
STEP03 = ROOT / "results/skymodel/ne_pointing/step03"
WSKY   = ROOT / "data/Haro11_NEpointing_wsky.fits"

WINDOW, THRESHOLDS, MAX_ITER, CLIP_SIGMA, TILE = 300, (1, 2), 5, 30, 32


def main():
    ap = argparse.ArgumentParser(description="雜訊修正的 PCA")
    ap.add_argument("--ks", type=int, nargs="+", default=[20, 30, 40])
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.5, 0.8, 1.0, 1.3, 1.75])
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
    blank = blank[:, ok].astype(np.float64)

    yy, xx = np.mgrid[0:ny, 0:nx]
    tr = (((yy // TILE) + (xx // TILE)) % 2).astype(bool)[bm][ok]
    Dtr, Dte = blank[:, tr], blank[:, ~tr]
    Vok = V[:, ok]
    v    = Vok[:, tr].astype(np.float64).mean(axis=1)        # 逐通道的雜訊變異數
    sig_te = np.sqrt(Vok[:, ~tr].astype(np.float64))         # chi 指標用,逐元素
    del V, Vok

    # ---- C_sky 與殘差,與 step3 一致 ----
    p16, med, p84 = np.percentile(Dtr, [16, 50, 84], axis=1)
    sg   = np.maximum((p84 - p16) / 2, 1e-6)
    keep = np.abs(Dtr - med[:, None]) <= CLIP_SIGMA * sg[:, None]
    C = estimate_continuum((Dtr * keep).sum(axis=1) / keep.sum(axis=1),
                           thresholds=THRESHOLDS, window=WINDOW, max_iter=MAX_ITER)[0]
    R = np.where(keep, Dtr - C[:, None], (med - C)[:, None])
    n = R.shape[1]
    print(f"blank {blank.shape[1]:,}   train {n:,}  test {Dte.shape[1]:,}")

    emp = R.var(axis=1)
    print(f"跨 spaxel 的變異數 / STAT:線內中位 {np.median((emp/v)[IN]):.3f}"
          f"   線外中位 {np.median((emp/v)[~IN]):.3f}")
    print(f"→ 線內約 {100/np.median((emp/v)[IN]):.0f}% 的變異數是光子雜訊\n")

    t0 = time.time()
    M = (R @ R.T) / n                                         # (nz, nz) 二階矩
    print(f"二階矩 {M.shape} 算完 ({time.time()-t0:.0f}s)")

    ex = np.where(IN, np.maximum(emp - v, 0), 0.0)
    top = np.argsort(ex)[::-1][:100]

    # flux 指標(未加權)與 chi 指標(除以雜訊)一起報 —— 兩個指標可能往相反的
    # 方向走,只量一個等於預設了答案。
    print(f"\n{'alpha':>7}{'K':>5}{'rms':>9}{'線內':>9}{'線外':>9}{'vs a=0':>9}"
          f"{'  |':>3}{'chi 全':>8}{'chi 內':>8}{'chi 外':>8}{'vs a=0':>9}"
          f"{'前100佔':>10}{'負特徵值':>9}")
    print("-" * 104)
    base, basec = {}, {}
    for a in args.alphas:
        w, U = np.linalg.eigh(M - a * np.diag(v))             # 對稱 → eigh
        o = np.argsort(w)[::-1]
        w, U = w[o], U[:, o]
        nneg = int((w < 0).sum())
        for K in args.ks:
            B = U[:, :K].T
            sky = np.vstack([C, B])
            Rte = Dte - sky.T @ (np.linalg.pinv(sky.T) @ Dte)
            rm = float(np.median(np.sqrt(np.mean(Rte ** 2, axis=0))))
            si = float(np.median(Rte[IN].std(axis=0)))
            so = float(np.median(Rte[~IN].std(axis=0)))
            ch = Rte / sig_te
            ca = float(np.median(np.sqrt(np.mean(ch ** 2, axis=0))))
            ci = float(np.median(np.sqrt(np.mean(ch[IN] ** 2, axis=0))))
            co = float(np.median(np.sqrt(np.mean(ch[~IN] ** 2, axis=0))))
            frac = (B[:, top] ** 2).sum() / (B ** 2).sum()
            r1 = "" if a == 0.0 else f"{100*(rm/base[K]-1):>8.2f}%"
            r2 = "" if a == 0.0 else f"{100*(ca/basec[K]-1):>8.2f}%"
            print(f"{a:>7.2f}{K:>5}{rm:>9.4f}{si:>9.4f}{so:>9.4f}{r1:>9}{'  |':>3}"
                  f"{ca:>8.4f}{ci:>8.4f}{co:>8.4f}{r2:>9}"
                  f"{100*frac:>9.1f}%{nneg:>9,}", flush=True)
            if a == 0.0:
                base[K], basec[K] = rm, ca
        print()

    print("alpha = STAT 佔真實雜訊的比例;alpha=0 就是現行做法。負號 = 更好。")
    print("兩組指標若往相反方向走,這就不是「改善或退步」,而是指標的選擇。")


if __name__ == "__main__":
    main()
