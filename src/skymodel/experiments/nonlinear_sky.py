"""完整的非線性天空模型:逐 spaxel 的波長偏移與 LSF 寬度,basis 放進迭代重學

物理模型
    S(p, λ) = Σᵢ Aᵢ(p) · Lᵢ(λ − δ(p) ; w(p))
              強度是線性的;δ(波長偏移)與 w(LSF 寬度)是非線性的。

先前 shift_fit.py 的測試不完整,而且不完整的方式會系統性地低估效果:它拿一組
「從未對齊的資料學來的 basis」再在上面加 δ。如果偏移真的存在,那組 basis 早就
用額外的成分把它吸收了(描述「有些 spaxel 偏左、有些偏右」),所以事後再加 δ
必然找不到東西 —— 它要找的東西已經在 basis 裡了。

這支把 basis 放進迭代:

    初始化  δ(p)=0, η(p)=0
    重複:
        ① 固定 basis,逐 spaxel 在 (δ, η) 格點上求最佳解
        ② 把資料校正到共同框架
        ③ 在校正後的資料上重學 basis

η 是寬度變化的線性化參數:LSF 的 sigma 增加 Δσ 時 m → m + (σ₀·Δσ)·m″,
所以 η = σ₀·Δσ,單位是 px²。σ₀ 取 MUSE 的 LSF(FWHM ~2.5 px → σ₀ ~1.06 px)。

校正的方向刻意選成「永遠展寬、不銳化」:參考框架取格點裡最寬的那一個,於是
每個 spaxel 都是正向摺積,數值穩定(反摺積會放大雜訊)。代價是校正會讓雜訊
稍微相關,這對 SVD 是個已知的小偏差,在結論裡要記得。

公平性:非線性模型每 spaxel 有 K+3 個參數(C_sky, K 條 basis, δ, η),所以
除了 K=30 之外也跑 K=32 當作「相同自由度」的對照。

    conda run -n astro python src/skymodel/experiments/nonlinear_sky.py
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import shift as ndshift
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import estimate_continuum

ROOT   = Path(__file__).resolve().parents[3]
STEP01 = ROOT / "results/skymodel/step01"
STEP03 = ROOT / "results/skymodel/step03"
WSKY   = ROOT / "data/Haro11_NEpointing_wsky.fits"

SEED, WINDOW, THRESHOLDS, MAX_ITER, CLIP_SIGMA, TILE = 0, 300, (1, 2), 5, 30, 32
SIGMA0 = 1.06          # MUSE LSF sigma (px),FWHM ~2.5 px


def learn(X, K):
    """從一批 spaxel 學出 (C_sky, basis),流程與 step3 一致。"""
    p16, med, p84 = np.percentile(X, [16, 50, 84], axis=1)
    sg = np.maximum((p84 - p16) / 2, 1e-6)
    kp = np.abs(X - med[:, None]) <= CLIP_SIGMA * sg[:, None]
    C = estimate_continuum((X * kp).sum(axis=1) / kp.sum(axis=1),
                           thresholds=THRESHOLDS, window=WINDOW, max_iter=MAX_ITER)[0]
    R = np.where(kp, X - C[:, None], (med - C)[:, None])
    B = TruncatedSVD(n_components=K, random_state=SEED).fit(
        R.T.astype(np.float32)).components_
    return C, B


def transformed(sky, d, e):
    """把天空模型平移 d、再加上寬度的一階項 e·m″。

    兩者都是對「整個模型」的線性算子,所以變換後的設計矩陣對所有 spaxel
    仍然相同 —— pinv 捷徑成立,不必逐 spaxel 解非線性問題。
    """
    S = sky if d == 0 else ndshift(sky, (0, d), order=3, mode="nearest")
    if e == 0:
        return S
    return S + e * np.gradient(np.gradient(S, axis=1), axis=1)


def fit_grid(sky, D, gd, ge, rows):
    """逐 spaxel 挑 (δ, η) 格點裡平方誤差最小者。回傳 (δ, η, 殘差)。"""
    n = D.shape[1]
    best = np.full(n, np.inf)
    dd = np.zeros(n); ee = np.zeros(n)
    R = np.empty_like(D)
    for d in gd:
        for e in ge:
            S = transformed(sky, d, e)
            c = np.linalg.pinv(S[:, rows].T) @ D[rows]
            r = D - S.T @ c
            q = (r[rows] ** 2).sum(axis=0)
            m = q < best
            best[m], dd[m], ee[m] = q[m], d, e
            R[:, m] = r[:, m]
    return dd, ee, R


def correct(D, dd, ee, gd, ge):
    """把資料反向平移 −δ,校正到共同的波長框架。

    只校正平移,不校正寬度。原因是可逆性:平移的反運算還是平移(精確、穩定),
    而展寬的反運算是反摺積(放大雜訊、不穩定)。

    先前的版本把資料「展寬到格點裡最寬的參考」,想藉此避開反摺積 —— 但那讓
    basis 活在展寬後的框架裡,拿去擬合沒展寬的資料就對不上(連 η=0 的 spaxel
    都被展寬 0.825 px,而天光線的 sigma 才 1.06 px),結果比基準差 48%。

    寬度的一階效應本來就是線性的(m → m + η·m″),留在擬合裡當線性項即可;
    真要把它也做成非線性,正解是用伴隨算子 T_pᵀ 解正規方程,不是校正資料。
    """
    out = np.empty_like(D)
    for d in gd:
        m = dd == d
        if not m.any():
            continue
        out[:, m] = D[:, m] if d == 0 else ndshift(D[:, m], (-d, 0), order=3,
                                                   mode="nearest")
    return out


def score(sky, D, IN, gd=None, ge=None):
    rows = np.ones(D.shape[0], bool)
    if gd is None:
        R = D - sky.T @ (np.linalg.pinv(sky.T) @ D)
    else:
        R = fit_grid(sky, D, gd, ge, rows)[2]
    return (float(np.median(np.sqrt(np.mean(R ** 2, axis=0)))),
            float(np.median(R[IN].std(axis=0))),
            float(np.median(R[~IN].std(axis=0))))


def main():
    ap = argparse.ArgumentParser(description="完整的非線性天空模型")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--dmax", type=float, default=0.20)
    ap.add_argument("--dstep", type=float, default=0.05)
    ap.add_argument("--emax", type=float, default=0.30)
    ap.add_argument("--estep", type=float, default=0.15)
    args = ap.parse_args()

    gd = np.round(np.arange(-args.dmax, args.dmax + args.dstep / 2, args.dstep), 4)
    ge = np.round(np.arange(-args.emax, args.emax + args.estep / 2, args.estep), 4)

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    IN    = np.load(STEP03 / "iter_line_mask.npy")[0]
    bm    = (white != 0) & (seg == 0)
    ny, nx = bm.shape

    with fits.open(WSKY, memmap=True) as h:
        nz = h["DATA"].header["NAXIS3"]
        blank = np.empty((nz, int(bm.sum())), np.float32)
        for j in range(0, nz, 200):
            blank[j:j+200] = np.asarray(h["DATA"].data[j:j+200], np.float32)[:, bm]
    ok = np.isfinite(blank).all(axis=0)
    blank = blank[:, ok].astype(np.float64)

    yy, xx = np.mgrid[0:ny, 0:nx]
    tr = (((yy // TILE) + (xx // TILE)) % 2).astype(bool)[bm][ok]
    Dtr, Dte = blank[:, tr], blank[:, ~tr]
    print(f"train {Dtr.shape[1]:,}  test {Dte.shape[1]:,}")
    print(f"δ 格點 {gd.size} 個 ({gd[0]:+.2f}…{gd[-1]:+.2f} px)   "
          f"η 格點 {ge.size} 個 ({ge[0]:+.2f}…{ge[-1]:+.2f} px²,"
          f"對應 Δσ {ge[-1]/SIGMA0:+.2f} px = 寬度 {100*ge[-1]/SIGMA0**2:+.0f}%)")
    print(f"每輪 {gd.size*ge.size} 次共用設計矩陣的解\n")

    allrows = np.ones(nz, bool)
    print(f"{'':>32}{'rms':>10}{'線內':>10}{'線外':>10}{'vs 基準':>10}")
    print("-" * 74)

    C, B = learn(Dtr, args.K)
    base = score(np.vstack([C, B]), Dte, IN)
    print(f"{f'A  線性 K={args.K} (現行)':>32}{base[0]:>10.4f}{base[1]:>10.4f}"
          f"{base[2]:>10.4f}")

    C2, B2 = learn(Dtr, args.K + 2)
    s2 = score(np.vstack([C2, B2]), Dte, IN)
    print(f"{f'B  線性 K={args.K+2} (相同自由度)':>32}{s2[0]:>10.4f}{s2[1]:>10.4f}"
          f"{s2[2]:>10.4f}{100*(s2[0]/base[0]-1):>9.2f}%")

    # ---- 非線性:basis 在迴圈裡重學 ----
    Ci, Bi = C, B
    for it in range(1, args.iters + 1):
        t0 = time.time()
        sky = np.vstack([Ci, Bi])
        dd, ee, _ = fit_grid(sky, Dtr, gd, ge, allrows)
        Dc = correct(Dtr, dd, ee, gd, ge)
        Ci, Bi = learn(Dc, args.K)
        s = score(np.vstack([Ci, Bi]), Dte, IN, gd, ge)
        print(f"{f'C  非線性 迭代 {it}':>32}{s[0]:>10.4f}{s[1]:>10.4f}{s[2]:>10.4f}"
              f"{100*(s[0]/base[0]-1):>9.2f}%"
              f"   δ {np.std(dd):.3f}px  η {np.std(ee):.3f}  ({time.time()-t0:.0f}s)",
              flush=True)

    # ---- 對照:參數轉不轉移得到未見的通道 ----
    lines = np.flatnonzero(IN)
    fitrow = np.zeros(nz, bool); fitrow[lines[0::2]] = True
    evrow  = np.zeros(nz, bool); evrow[lines[1::2]]  = True
    sky = np.vstack([Ci, Bi])
    print()
    for nm, g1, g2 in [("δ=η=0", np.array([0.0]), np.array([0.0])),
                       ("(δ,η) 自由", gd, ge)]:
        R = fit_grid(sky, Dte, g1, g2, fitrow)[2]
        v = float(np.median(R[evrow].std(axis=0)))
        print(f"{'對照 ' + nm + ' (奇數線通道擬合→偶數評分)':>40}   sigma {v:.4f}")
        if "0" in nm:
            v0 = v
        else:
            print(f"{'':>40}   轉移到未見通道 {100*(v/v0-1):+.2f}%")

    dd, ee, _ = fit_grid(sky, Dte, gd, ge, allrows)
    img = np.full(bm.size, np.nan)
    img[np.flatnonzero(bm.ravel())[ok][~tr]] = dd
    img = img.reshape(bm.shape); fin = np.isfinite(img)
    p = fin[:, :-1] & fin[:, 1:]
    print(f"\nδ 中位 {np.nanmedian(img):+.3f} px  散布 {np.nanstd(img):.3f} px"
          f"  觸及格點邊界 {100*np.mean(np.abs(dd) >= args.dmax):.1f}%")
    print(f"η 中位 {np.median(ee):+.3f}  散布 {np.std(ee):.3f}"
          f"  觸及邊界 {100*np.mean(np.abs(ee) >= args.emax):.1f}%")
    print(f"δ 的相鄰 spaxel 相關 {np.corrcoef(img[:, :-1][p], img[:, 1:][p])[0,1]:+.3f}")


if __name__ == "__main__":
    main()
