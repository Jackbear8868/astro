"""超參數搜尋:有沒有一組設定能同時壓低 rms 與線內/線外 sigma?

搜尋的每一個超參數都必須有物理意義,不是為了把數字壓低而存在:

    K            天空變化的自由度。天光線的相對強度隨時間/位置變化,
                 K 就是「需要幾個獨立方向才描述得完」
    WINDOW       連續譜 running median 的視窗(px)。1.25 A/px,所以 300 px
                 = 375 A。物理限制:必須遠寬於一條天光線(FWHM ~2.5 px),
                 又必須窄於天空連續譜本身的結構尺度(~1000 A)
    THRESHOLDS   線偵測門檻(正, 負)。目前用 (1, 2) —— 這裡只是量出
                 鄰近值的行為供討論,不是要取代它
    CLIP_SIGMA   離群值門檻,掃它是為了知道結果對這個選擇有多敏感
    method       分解方式 svd/pca/nmf/rpca

三個防過擬合的設計:

    ① 空間分塊切訓練/測試(32 px 棋盤格),不用隨機切。隨機切時測試 spaxel
       的鄰居就在訓練集裡,basis 等於看過答案;棋盤格測的是「能不能推廣到
       探測器上另一塊區域」
    ② 評分用的 line mask 全程固定(現行 pipeline 的第一輪遮罩)。否則改
       THRESHOLDS 會同時改變模型和「線內/線外」的定義,數字變好可能只是
       重新切分的假象
    ③ 同時報 train 與 test。兩者的差距就是過擬合的量;只看 test 會漏掉
       「這組設定本來就學不動」的情況

評分方式對齊 step5 的 blank 區實際做法:設計矩陣 [C_sky, e1..eK],s 自由,
未加權最小平方,共用 pinv。

    conda run -n astro python src/skymodel/experiments/hyper_search.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from sklearn.decomposition import PCA, NMF, TruncatedSVD

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import estimate_continuum
from step3_sky_basis import robust_pca

ROOT   = Path(__file__).resolve().parents[3]
STEP01 = ROOT / "results/skymodel/ne_pointing/step01"
STEP03 = ROOT / "results/skymodel/ne_pointing/step03"
WSKY   = ROOT / "data/Haro11_NEpointing_wsky.fits"

SEED, MAX_ITER, TILE = 0, 5, 32
BASE = dict(window=300, thr=(1, 2), clip=30.0)     # 現行設定


def build(blank, window, thr, clip):
    """從 train spaxel 學出 (C_sky, residual)。與 step3 的流程逐步對應。"""
    p16, med, p84 = np.percentile(blank, [16, 50, 84], axis=1)
    sg = np.maximum((p84 - p16) / 2, 1e-6)
    if clip is None:
        mean = blank.mean(axis=1)
        keep = None
    else:
        keep = np.abs(blank - med[:, None]) <= clip * sg[:, None]
        mean = (blank * keep).sum(axis=1) / keep.sum(axis=1)
    C = estimate_continuum(mean, thresholds=thr, window=window,
                           max_iter=MAX_ITER)[0]
    R = blank - C[:, None]
    if keep is not None:
        R = np.where(keep, R, (med - C)[:, None])
    return C, R


def decompose(R, K, method):
    """(K, nz) 的 basis。svd/pca 是巢狀的 —— 前 K 列就是 K 條 basis。"""
    X = R.T.astype(np.float32)
    if method == "svd":
        return TruncatedSVD(n_components=K, random_state=SEED).fit(X).components_
    if method == "pca":
        p = PCA(n_components=K - 1, random_state=SEED).fit(X)
        return np.vstack([p.mean_[None, :], p.components_])
    if method == "nmf":
        return NMF(n_components=K, init="nndsvda", max_iter=300,
                   random_state=SEED).fit(np.clip(X, 0, None)).components_
    if method == "rpca":
        return robust_pca(X, K=K, seed=SEED)[0]
    raise ValueError(method)


def score(sky, D, IN):
    """回傳 (rms, 線內 sigma, 線外 sigma) 的逐 spaxel 中位數。

    s 自由、未加權、共用 pinv —— 和 step5 的 fit_blank 預設路徑相同。
    """
    R = D - sky.T @ (np.linalg.pinv(sky.T) @ D)
    return (float(np.median(np.sqrt(np.mean(R ** 2, axis=0)))),
            float(np.median(R[IN].std(axis=0))),
            float(np.median(R[~IN].std(axis=0))))


def main():
    ap = argparse.ArgumentParser(description="天空模型的超參數搜尋")
    ap.add_argument("--windows", type=int, nargs="+",
                    default=[100, 150, 200, 300, 450, 600])
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 20, 30, 40, 60, 80])
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    IN    = np.load(STEP03 / "iter_line_mask.npy")[0]      # 固定的評分遮罩
    bm    = (white != 0) & (seg == 0)
    ny, nx = bm.shape

    with fits.open(WSKY, memmap=True) as h:
        nz = h["DATA"].header["NAXIS3"]
        blank = np.empty((nz, int(bm.sum())), np.float32)
        for j in range(0, nz, 200):
            blank[j:j+200] = np.asarray(h["DATA"].data[j:j+200], np.float32)[:, bm]
    good  = np.isfinite(blank).all(axis=0)
    blank = blank[:, good].astype(np.float64)

    # 32 px 棋盤格:兩半都覆蓋整個視野,照明梯度不會變成訓練/測試的系統差異
    yy, xx = np.mgrid[0:ny, 0:nx]
    tile = (((yy // TILE) + (xx // TILE)) % 2).astype(bool)[bm][good]
    Dtr, Dte = blank[:, tile], blank[:, ~tile]
    print(f"blank {blank.shape[1]:,}   train {Dtr.shape[1]:,}  test {Dte.shape[1]:,}"
          f"   ({TILE} px 棋盤格)   評分遮罩:線內 {int(IN.sum())}/{nz} 通道\n")

    # ---------- 第 1 階段:WINDOW x K ----------
    print("第 1 階段  WINDOW x K   (method=svd, THRESHOLDS=(1,2), CLIP=30)")
    print(f"{'WINDOW':>8}{'= A':>8}{'K':>5}"
          f"{'test rms':>10}{'test 線內':>11}{'test 線外':>11}"
          f"{'train rms':>11}{'差距':>8}")
    print("-" * 74)
    rows = []
    for w in args.windows:
        t0 = time.time()
        C, R = build(Dtr, w, BASE["thr"], BASE["clip"])
        B80 = decompose(R, max(args.ks), "svd")          # 巢狀:一次算到最大 K
        for K in args.ks:
            sky = np.vstack([C, B80[:K]])
            te, tr_ = score(sky, Dte, IN), score(sky, Dtr, IN)
            rows.append((w, K, *te, tr_[0]))
            print(f"{w:>8}{w*1.25:>7.0f}A{K:>5}{te[0]:>10.4f}{te[1]:>11.4f}"
                  f"{te[2]:>11.4f}{tr_[0]:>11.4f}{100*(te[0]/tr_[0]-1):>7.2f}%")
        print(f"{'':>8}{'':>8}{'':>5}   ({time.time()-t0:.0f}s)")
    best = min(rows, key=lambda r: r[2])
    print(f"\ntest rms 最低:WINDOW={best[0]} K={best[1]}  rms={best[2]:.4f}")

    # ---------- 第 2 階段:在最佳 WINDOW 上掃其餘超參數 ----------
    w = best[0]
    print(f"\n第 2 階段  其餘超參數 (WINDOW={w})")
    print(f"{'超參數':>26}{'K':>5}{'test rms':>10}{'test 線內':>11}"
          f"{'test 線外':>11}{'vs 現行':>9}")
    print("-" * 72)

    C0, R0 = build(Dtr, w, BASE["thr"], BASE["clip"])
    B0 = decompose(R0, 30, "svd")
    ref = score(np.vstack([C0, B0]), Dte, IN)
    print(f"{'現行 svd/thr(1,2)/clip30':>26}{30:>5}{ref[0]:>10.4f}"
          f"{ref[1]:>11.4f}{ref[2]:>11.4f}{'—':>9}")

    def show(name, C, B, K):
        s = score(np.vstack([C, B]), Dte, IN)
        print(f"{name:>26}{K:>5}{s[0]:>10.4f}{s[1]:>11.4f}{s[2]:>11.4f}"
              f"{100*(s[0]/ref[0]-1):>8.2f}%", flush=True)

    for m in ["pca", "nmf", "rpca"]:
        show(f"method = {m}", C0, decompose(R0, 30, m), 30)

    for c in [10.0, 100.0, None]:
        C, R = build(Dtr, w, BASE["thr"], c)
        show(f"CLIP_SIGMA = {c if c else '不剔除'}", C, decompose(R, 30, "svd"), 30)

    # 目前用 (1,2);列出鄰近值只是為了知道這個選擇有多敏感,不是要取代它
    for thr in [(1, 1), (2, 2), (1, 3), (0.5, 2)]:
        C, R = build(Dtr, w, thr, BASE["clip"])
        show(f"THRESHOLDS = {thr}", C, decompose(R, 30, "svd"), 30)


if __name__ == "__main__":
    main()
