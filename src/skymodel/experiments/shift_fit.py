"""逐 spaxel 的波長偏移 delta(p):線性一階版本

模型從
    D(p,λ) = s·C_sky(λ) + Σₖ cₖ(p)·eₖ(λ)
改成
    D(p,λ) = [ s·C_sky + Σₖ cₖ(p)·eₖ ](λ − δ(p))

也就是整個天空模型平移 δ(p)。物理上 δ 是波長解的誤差 —— 那是儀器性質,對
所有天光線是同一個值,所以只有「一個」δ,不是每條線各自漂移。每個 spaxel
只多 1 個自由度。

結構上的保證:一條被平移的天光線永遠還在天光線的波長上,不可能變成一條紅移
的源發射線。這比「K 維子空間」的保證更強 —— 但保證要用注入-回收證明,不是
用論證,所以這支只做 blank 這一側,源那一側另外跑。

求解:δ 取格點,每個格點把整組 basis 平移後仍是「所有 spaxel 共用的設計矩陣」,
於是 pinv 捷徑仍然成立 —— 每個格點只要算一次 pinv,不必逐 spaxel 非線性求解。

三個測試,因為 δ 是挑出來讓自己殘差最小的,它必定會讓殘差下降,即使在擬合雜訊:
    主測試   δ 用全部通道擬合,在留出的 test spaxel 上評分
    對照 A   δ 只用奇數線通道擬合,在偶數線通道上評分 —— 擬合雜訊的 δ 轉移不了
    對照 B   δ 的空間結構。真的波長解誤差有 slice 結構,雜訊沒有

    conda run -n astro python src/skymodel/experiments/shift_fit.py
"""
import argparse
import sys
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


def best_shift(sky, D, grid, fit_rows):
    """對每個 spaxel 在 delta 格點上挑最小平方誤差的那一個。

    每個格點的設計矩陣對所有 spaxel 相同,所以 pinv 只算一次 —— 這正是
    「只有一個共用的 delta」帶來的計算優勢。

    Returns
    -------
    delta : (n,)      每個 spaxel 選中的位移
    coef  : (p, n)    在該位移下的係數
    skies : list      各格點平移後的天空矩陣(評分時要用)
    """
    n = D.shape[1]
    best = np.full(n, np.inf)
    delta = np.zeros(n)
    coef = np.zeros((sky.shape[0], n))
    skies = []
    for g in grid:
        Sg = sky if g == 0 else ndshift(sky, (0, g), order=3, mode="nearest")
        skies.append(Sg)
        c = np.linalg.pinv(Sg[:, fit_rows].T) @ D[fit_rows]
        r = ((D[fit_rows] - Sg[:, fit_rows].T @ c) ** 2).sum(axis=0)
        m = r < best
        best[m], delta[m], coef[:, m] = r[m], g, c[:, m]
    return delta, coef, skies


def residual(D, skies, grid, delta, coef):
    """用各 spaxel 自己選中的位移重建全通道殘差。"""
    R = np.empty_like(D)
    for i, g in enumerate(grid):
        m = delta == g
        if m.any():
            R[:, m] = D[:, m] - skies[i].T @ coef[:, m]
    return R


def report(name, R, IN, ref=None):
    rm = float(np.median(np.sqrt(np.mean(R ** 2, axis=0))))
    si = float(np.median(R[IN].std(axis=0)))
    so = float(np.median(R[~IN].std(axis=0)))
    rel = "" if ref is None else f"{100*(rm/ref-1):>8.2f}%"
    print(f"{name:>26}{rm:>10.4f}{si:>11.4f}{so:>11.4f}{rel:>9}")
    return rm


def main():
    ap = argparse.ArgumentParser(description="逐 spaxel 波長偏移")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--max-shift", type=float, default=0.5, help="格點半寬 (px)")
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()

    grid = np.round(np.arange(-args.max_shift, args.max_shift + args.step / 2,
                              args.step), 4)

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
    print(f"blank {blank.shape[1]:,}   train {Dtr.shape[1]:,}  test {Dte.shape[1]:,}")
    print(f"delta 格點 {grid[0]:+.2f} … {grid[-1]:+.2f} px,共 {grid.size} 點"
          f"   (1 px = {1.25:.2f} A = {1.25/5577*3e5:.0f} km/s @ 5577 A)\n")

    # ---- basis 與 step3 一致 ----
    p16, med, p84 = np.percentile(Dtr, [16, 50, 84], axis=1)
    sg   = np.maximum((p84 - p16) / 2, 1e-6)
    keep = np.abs(Dtr - med[:, None]) <= CLIP_SIGMA * sg[:, None]
    C = estimate_continuum((Dtr * keep).sum(axis=1) / keep.sum(axis=1),
                           thresholds=THRESHOLDS, window=WINDOW, max_iter=MAX_ITER)[0]
    R = np.where(keep, Dtr - C[:, None], (med - C)[:, None])
    B = TruncatedSVD(n_components=args.K,
                     random_state=SEED).fit(R.T.astype(np.float32)).components_
    sky = np.vstack([C, B])

    allrows = np.ones(nz, bool)
    print(f"{'':>26}{'rms':>10}{'線內 sigma':>11}{'線外 sigma':>11}{'vs δ=0':>9}")
    print("-" * 68)

    # ---- 主測試 ----
    d0, c0, sk = best_shift(sky, Dte, np.array([0.0]), allrows)
    base = report("δ = 0 (現行)", residual(Dte, sk, [0.0], d0, c0), IN)
    dd, cc, sk = best_shift(sky, Dte, grid, allrows)
    report("δ 自由 (全通道擬合)", residual(Dte, sk, grid, dd, cc), IN, base)

    # ---- 對照 A:δ 在奇數線通道擬合,偶數線通道評分 ----
    lines = np.flatnonzero(IN)
    fitrow = np.zeros(nz, bool); fitrow[lines[0::2]] = True
    evrow  = np.zeros(nz, bool); evrow[lines[1::2]]  = True
    print()
    for nm, g in [("δ = 0", np.array([0.0])), ("δ 自由", grid)]:
        d, c, s = best_shift(sky, Dte, g, fitrow)
        Rr = residual(Dte, s, g, d, c)
        v = float(np.median(Rr[evrow].std(axis=0)))
        print(f"{'對照A ' + nm + ' (奇數擬合→偶數評分)':>32}"
              f"   偶數線通道 sigma {v:.4f}")
        if nm == "δ = 0":
            v0 = v
        else:
            print(f"{'':>32}   轉移到未見通道的改善 {100*(v/v0-1):+.2f}%")

    # ---- 對照 B:δ 的空間結構 ----
    img = np.full(bm.size, np.nan)
    pos = np.flatnonzero(bm.ravel())[ok][~tr]
    img[pos] = dd
    img = img.reshape(bm.shape)
    fin = np.isfinite(img)
    pair = fin[:, :-1] & fin[:, 1:]
    a, b = img[:, :-1][pair], img[:, 1:][pair]
    print(f"\n對照B  δ 的分布:中位 {np.nanmedian(img):+.3f} px   "
          f"散布 {np.nanstd(img):.3f} px   "
          f"落在格點邊界的比例 {100*np.mean(np.abs(dd) >= args.max_shift):.1f}%")
    print(f"       相鄰 spaxel 的相關係數 {np.corrcoef(a, b)[0,1]:+.3f}"
          f"   (雜訊 → 0;真的波長解誤差 → 明顯為正)")


if __name__ == "__main__":
    main()
