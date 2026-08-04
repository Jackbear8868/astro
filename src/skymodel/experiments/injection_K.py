"""注入-回收:basis 條數 K 對「源保不保得住」的影響。

blank 區的量測只能說天空扣得乾不乾淨,說不了源有沒有被吃掉 —— 源區域沒有
真值。唯一能造出真值的方法就是在 blank 區注入已知強度的假源再量回來。

K 變大在 blank 區是好事(天光線描述得更好),但在源區域是風險:每多一條天空
成分,天空就多一種吸收源流量的方式。這支腳本量的就是那個代價。

判讀:
    A_meas / A_true 系統性 < 1   → 源被天空吃掉(過度扣除)
    A_meas / A_true 系統性 > 1   → 天空殘留被算成源
    散布大                       → 簡併造成的不確定度
    A_true = 0 那一列            → 什麼都沒注入卻量到的 A = 假訊號底線

擬合設定與現行 step5 的 source 區一致:chi2 加權、A >= 0、s 固定。

    conda run -n astro python src/skymodel/experiments/injection_K.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import lsq_linear
from scipy.ndimage import binary_dilation
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from templates import load_sdss_template, redshift_to_grid, air_to_vacuum

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP03  = ROOT / "results/skymodel/step03"
TPL_DIR = ROOT / "data/sdss_templates"
CUBE    = ROOT / "data/Haro11_NEpointing_wsky.fits"

SEED       = 0
CLIP_SIGMA = 30         # 與 step3_sky_basis.CLIP_SIGMA 相同
TEMPLATE = "027"        # step4 對 Haro 11 選中的那一條
Z        = 0.0205
FWHM     = 4.06         # 實測 PSF,px
RADIUS   = 4            # 孔徑半徑 → 9x9 = 81 px
S_FIX    = 1.0
# 模板 T 的值域只有 0.003–0.100,所以 A 要到千位數才對應到雜訊以上的流量
# (Haro 11 實測 A = 2741)。振幅設低了會全部落在雜訊以下,量到的是「偵測極限」
# 而不是「回收率」。
AMPS     = np.array([0.0, 300.0, 1000.0, 3000.0, 10000.0])


def gaussian_stamp(radius, fwhm):
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return np.exp(-(x ** 2 + y ** 2) / (2.0 * (fwhm / 2.3548) ** 2))


def pick_sites(blank2d, n, radius, seed):
    """孔徑完全落在 blank 內、彼此不重疊。"""
    ok = blank2d & ~binary_dilation(~blank2d, iterations=radius + 1)
    ys, xs = np.nonzero(ok)
    rng = np.random.default_rng(seed)
    used, sites = np.zeros_like(blank2d), []
    for i in rng.permutation(ys.size):
        y, x = int(ys[i]), int(xs[i])
        sl = (slice(y - radius, y + radius + 1), slice(x - radius, x + radius + 1))
        if used[sl].any():
            continue
        used[sl] = True
        sites.append((y, x))
        if len(sites) == n:
            break
    return sites


def fit_patch(d, v, design, lb, cover):
    """逐 spaxel 加權最小平方,回傳每個 spaxel 的 A(第 0 個參數)。"""
    p = design.shape[0]
    ub = np.full(p, np.inf)
    base = np.all(np.isfinite(design), axis=0) & cover
    out = np.zeros(d.shape[1])
    for j in range(d.shape[1]):
        g = base & np.isfinite(d[:, j]) & np.isfinite(v[:, j]) & (v[:, j] > 0)
        if g.sum() <= p:
            out[j] = np.nan
            continue
        sig = np.sqrt(v[g, j])
        out[j] = lsq_linear(design[:, g].T / sig[:, None], d[g, j] / sig,
                            bounds=(lb, ub), method="bvls").x[0]
    return out


def main():
    ap = argparse.ArgumentParser(description="注入-回收:K 對源的影響")
    ap.add_argument("-K", type=int, nargs="+", default=[10, 25],
                    help="要比較的 basis 條數")
    ap.add_argument("--n-site", type=int, default=25)
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    wl    = air_to_vacuum(np.load(STEP03 / "wavelength.npy"))
    C_sky = np.load(STEP03 / "sky_continuum.npy")

    with fits.open(CUBE, memmap=True) as hdul:
        D0 = np.asarray(hdul["DATA"].data, np.float32)
        V0 = np.asarray(hdul["STAT"].data, np.float32)
    nz, ny, nx = D0.shape

    blank2d = (white != 0) & (seg == 0)
    flat = D0.reshape(nz, -1)
    comp = np.zeros(ny * nx, bool)
    b = blank2d.ravel()
    comp[np.flatnonzero(b)] = np.isfinite(flat[:, b]).all(axis=0)

    # 殘差要和改過的 step3 一致 —— 逐通道剔除離群值,被剔除的位置填該通道的
    # 典型殘差。不做的話這裡學到的 basis 會有一條被 12 個壞 spaxel 佔走,
    # 等於拿舊的 basis 去測新的設定。逐段處理是為了控制峰值記憶體
    # (整個 cube 的 DATA 與 STAT 已經在記憶體裡)。
    blank = flat[:, comp]
    med = np.empty(nz); sg = np.empty(nz)
    for j in range(0, nz, 200):
        p16, m, p84 = np.percentile(blank[j:j+200], [16, 50, 84], axis=1)
        med[j:j+200], sg[j:j+200] = m, np.maximum((p84 - p16) / 2, 1e-6)
    resid = np.empty((int(comp.sum()), nz), np.float32)
    nbad = 0
    for j in range(0, nz, 200):
        x = blank[j:j+200]
        k = np.abs(x - med[j:j+200, None]) <= CLIP_SIGMA * sg[j:j+200, None]
        nbad += int((~k).sum())
        resid[:, j:j+200] = np.where(
            k, x - C_sky[j:j+200, None],
            (med[j:j+200] - C_sky[j:j+200])[:, None]).T
    print(f"學 basis 的殘差:clip {CLIP_SIGMA} sigma 剔除 {nbad:,} 個元素")

    T  = redshift_to_grid(load_sdss_template(TPL_DIR / f"spDR2-{TEMPLATE}.fit"), Z, wl)
    Tz = np.nan_to_num(T, nan=0.0)
    cover = np.isfinite(T)
    stamp = gaussian_stamp(RADIUS, FWHM)
    prof  = stamp.ravel()
    sites = pick_sites(blank2d & comp.reshape(ny, nx), args.n_site, RADIUS, SEED)

    patches = []
    for (y, x) in sites:
        sl = (slice(y - RADIUS, y + RADIUS + 1), slice(x - RADIUS, x + RADIUS + 1))
        patches.append((D0[:, sl[0], sl[1]].reshape(nz, -1).astype(np.float64),
                        V0[:, sl[0], sl[1]].reshape(nz, -1).astype(np.float64)))
    del D0, V0, flat

    print(f"模板 {TEMPLATE} @ z={Z}   孔徑 {stamp.size} px   {len(sites)} 個位置")
    print(f"s 固定 {S_FIX}   模板覆蓋 {int(cover.sum()):,}/{nz} 通道\n")

    summary = []
    for K in args.K:
        B = TruncatedSVD(n_components=K, random_state=SEED).fit(resid).components_
        design = np.vstack([Tz, B]).astype(np.float64)          # s 固定 → 不放 C_sky
        p  = design.shape[0]
        lb = np.r_[0.0, np.full(p - 1, -np.inf)]                # A >= 0
        print(f"===== K = {K}   (參數 {p} 個) =====")
        # 把所有位置、所有振幅的 spaxel 匯總,依「該 spaxel 的真實 A」分層
        rows = []
        for amp in AMPS:
            for (d0, v) in patches:
                d = d0 + amp * prof[None, :] * Tz[:, None] - S_FIX * C_sky[:, None]
                a = fit_patch(d, v, design, lb, cover)
                rows.append(np.c_[amp * prof, a])
        AT, AM = np.vstack(rows).T
        m = np.isfinite(AM)
        AT, AM = AT[m], AM[m]

        print(f"{'A_true 區間':>16}{'n':>8}{'A_meas 中位':>13}{'回收率':>10}"
              f"{'散布':>10}{'量到 0 的比例':>14}")
        print("-" * 72)
        bins = [(0, 0), (0, 30), (30, 100), (100, 300), (300, 1000),
                (1000, 3000), (3000, 1e9)]
        for lo, hi in bins:
            sel = (AT == 0) if hi == 0 else ((AT > lo) & (AT <= hi))
            if sel.sum() < 10:
                continue
            t, a = AT[sel], AM[sel]
            lab = "0 (未注入)" if hi == 0 else f"{lo:.0f}–{hi:.0f}"
            if hi == 0:
                print(f"{lab:>16}{sel.sum():>8,}{np.median(a):>13.2f}{'—':>10}"
                      f"{a.std():>10.2f}{100*np.mean(a == 0):>13.1f}%")
            else:
                r = a / t
                print(f"{lab:>16}{sel.sum():>8,}{np.median(a):>13.1f}"
                      f"{np.median(r):>10.3f}{r.std():>10.3f}"
                      f"{100*np.mean(a == 0):>13.1f}%")
        print()

        # 偵測對比 = 微弱源量到的訊號 / 什麼都沒注入時量到的假訊號散布。
        # 這是唯一同時看兩側的判準:分子大代表源留得住,分母小代表不會無中生有。
        z = AM[AT == 0]
        w = (AT > 100) & (AT <= 300)
        summary.append((K, z.std(), np.median(z), np.median(AM[w] / AT[w]),
                        np.median(AM[w]) / z.std() if z.std() > 0 else np.inf))

    print("=" * 72)
    print(f"{'K':>5}{'假訊號散布':>13}{'假訊號中位':>13}{'100–300 回收率':>16}"
          f"{'偵測對比':>11}")
    print("-" * 72)
    for K, s, m, r, c in summary:
        print(f"{K:>5}{s:>13.2f}{m:>13.2f}{r:>16.3f}{c:>11.1f}")
    print("\n偵測對比 = 微弱源(A_true 100–300)量到的 A 中位 / 假訊號底線的散布。")
    print("分子大 = 源留得住,分母小 = 不會把雜訊當成源 —— 兩側必須一起看。")

if __name__ == "__main__":
    main()
