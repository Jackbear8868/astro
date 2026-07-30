"""注入-回收:量 step5 對源振幅的回收率。

在 blank 區域注入已知強度的假源,用同一條模板擬合,比較 A_meas 與 A_true。
源區域沒有真值,這是唯一能把真值造出來的方法。

同時比較兩種 s(p) 的處理:

    free   s 逐 spaxel 自由      = 現行 step5
    fixed  s 由周圍 blank 決定   = 移除 A·T 與 s·C_sky 的簡併

判讀:
    A_meas / A_true 系統性 < 1  → 過度扣除,源被天空模型吃掉
    A_meas / A_true 系統性 > 1  → 欠扣,天空殘留被算成源
    散布大                      → 簡併造成的不確定度

A_true = 0 的那一列給出「什麼都沒注入」時的 dchi2 分佈,
也就是純雜訊能碰出多大的 dchi2 —— 篩選門檻的統計依據。

    conda run -n astro python src/skymodel/experiments/injection_recovery.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import lsq_linear
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from templates import load_sdss_template, redshift_to_grid, air_to_vacuum

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP03  = ROOT / "results/skymodel/step03"
STEP05  = ROOT / "results/skymodel/step05"
TPL_DIR = ROOT / "data/sdss_templates"
CUBE    = ROOT / "data/Haro11_NEpointing_wsky.fits"

BASIS    = "svd"
TEMPLATE = "027"          # Haro 11 選中的那一條
Z        = 0.0202
FWHM     = 4.06           # 實測 PSF,單位 px
RADIUS   = 4              # 假源孔徑半徑,面積 49 px,接近真實小源的規模
N_SITE   = 40             # 注入位置數
SEED     = 0

# A_true 的對數網格。0 代表什麼都不注入,用來量純雜訊的 dchi2。
AMPS = np.array([0.0, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0])


def gaussian_stamp(radius, fwhm):
    """半徑 radius 的方形孔徑內的高斯剖面,峰值為 1。"""
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    sigma = fwhm / 2.3548
    return np.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))


def pick_sites(blank, n, radius, seed):
    """挑出孔徑完全落在 blank 內、彼此不重疊的位置。"""
    ok = blank.copy()
    # 孔徑要整個在 blank 裡:把非 blank 的區域向外膨脹一個孔徑
    ok &= ~binary_dilation(~blank, iterations=radius + 1)
    ys, xs = np.nonzero(ok)
    rng = np.random.default_rng(seed)
    order = rng.permutation(ys.size)
    used = np.zeros_like(blank)
    sites = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        sl = (slice(y - radius, y + radius + 1), slice(x - radius, x + radius + 1))
        if used[sl].any():
            continue
        used[sl] = True
        sites.append((y, x))
        if len(sites) == n:
            break
    return sites


def fit_patch(D, var, sky, T=None, s_fixed=None, cover=None):
    """擬合一小塊 spaxel,回傳 (A(p), s(p), chi2(p))。

    T = None        只有天空,沒有模板 —— dchi2 的基準
    s_fixed = None  s 逐 spaxel 自由(現行 step5)
    s_fixed 給定    把 s·C_sky 先從資料扣掉,只解 A 與天光線係數

    cover 指定要使用的通道;三種擬合必須用同一批通道,dchi2 才有意義。
    A 一律受非負限制。
    """
    n_spx = D.shape[1]
    A  = np.full(n_spx, np.nan)
    S  = np.full(n_spx, np.nan)
    c2 = np.full(n_spx, np.nan)

    if T is None:
        design, lb = sky, np.r_[0.0, np.full(sky.shape[0] - 1, -np.inf)]
        i_A, i_s = None, 0
    elif s_fixed is None:
        design, lb = np.vstack([T, sky]), np.r_[0.0, 0.0, np.full(sky.shape[0] - 1, -np.inf)]
        i_A, i_s = 0, 1
    else:
        design, lb = np.vstack([T, sky[1:]]), np.r_[0.0, np.full(sky.shape[0] - 1, -np.inf)]
        i_A, i_s = 0, None
    p  = design.shape[0]
    ub = np.full(p, np.inf)
    base = np.all(np.isfinite(design), axis=0) & cover

    for j in range(n_spx):
        g = base & np.isfinite(D[:, j]) & np.isfinite(var[:, j]) & (var[:, j] > 0)
        if g.sum() <= p:
            continue
        sig = np.sqrt(var[g, j])
        y   = D[g, j] if s_fixed is None else D[g, j] - s_fixed[j] * sky[0, g]
        fit = lsq_linear(design[:, g].T / sig[:, None], y / sig,
                         bounds=(lb, ub), method="bvls")
        c2[j] = 2.0 * fit.cost
        if i_A is not None:
            A[j] = fit.x[i_A]
        if i_s is not None:
            S[j] = fit.x[i_s]
    return A, S, c2


def main():
    seg    = fits.getdata(STEP01 / "seg.fits")
    white  = fits.getdata(STEP01 / "whitelight.fits")
    wl_vac = air_to_vacuum(np.load(STEP03 / "wavelength.npy"))
    sky    = np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                        np.load(STEP03 / f"sky_basis_{BASIS}.npy")])
    s_map  = np.load(STEP05 / f"s_map_{BASIS}.npy")

    with fits.open(CUBE, memmap=True) as hdul:
        D0 = np.asarray(hdul["DATA"].data, np.float32)
        V0 = np.asarray(hdul["STAT"].data, np.float32)

    valid = white != 0
    blank = valid & (seg == 0)
    T = redshift_to_grid(load_sdss_template(TPL_DIR / f"spDR2-{TEMPLATE}.fit"), Z, wl_vac)
    Tz = np.nan_to_num(T, nan=0.0)

    stamp = gaussian_stamp(RADIUS, FWHM)
    sites = pick_sites(blank, N_SITE, RADIUS, SEED)
    cover = np.isfinite(T)          # 三種擬合共用同一批通道
    print(f"模板 {TEMPLATE}  z={Z}   孔徑 {2*RADIUS+1}x{2*RADIUS+1} = {stamp.size} px"
          f"   注入位置 {len(sites)} 個   模板覆蓋 {int(cover.sum()):,} 通道\n")

    # 每個位置的固定 s:用「未注入」資料做一次加權擬合。
    # 不能沿用 step5 的 s_map —— 那是未加權的解,和這裡的加權擬合不一致。
    prof = stamp.ravel()
    patches, s_ref = [], []
    for (y, x) in sites:
        sl = (slice(y - RADIUS, y + RADIUS + 1), slice(x - RADIUS, x + RADIUS + 1))
        d = D0[:, sl[0], sl[1]].reshape(D0.shape[0], -1).astype(np.float64)
        v = V0[:, sl[0], sl[1]].reshape(V0.shape[0], -1).astype(np.float64)
        patches.append((d, v))
        s_ref.append(fit_patch(d, v, sky, cover=cover)[1])
    s_ref = np.array(s_ref)
    print(f"固定用的 s(加權,未注入)  中位數 {np.nanmedian(s_ref):.4f}"
          f"   1σ {(np.nanpercentile(s_ref,84)-np.nanpercentile(s_ref,16))/2:.4f}\n")

    print(f"{'A_true':>8}{'——— s 自由 ———':^32}{'——— s 固定 ———':^32}{'':>2}")
    print(f"{'':>8}{'A_meas':>11}{'/A_true':>10}{'散布':>10}"
          f"{'A_meas':>13}{'/A_true':>10}{'散布':>10}{'dchi2 中位數':>16}")
    print("-" * 96)

    for amp in AMPS:
        af, ax, dch = [], [], []
        for (d0, v), sf in zip(patches, s_ref):
            d = d0 + amp * prof[None, :] * Tz[:, None]
            A_free, _, c2_free = fit_patch(d, v, sky, T, cover=cover)
            A_fix,  _, _       = fit_patch(d, v, sky, T, s_fixed=sf, cover=cover)
            _,      _, c2_null = fit_patch(d, v, sky, cover=cover)
            af.append(np.nansum(A_free))
            ax.append(np.nansum(A_fix))
            dch.append(np.nansum(c2_null) - np.nansum(c2_free))
        af, ax, dch = np.array(af), np.array(ax), np.array(dch)
        true_tot = amp * prof.sum()
        if amp > 0:
            print(f"{amp:>8.1f}{np.median(af):>11.1f}{np.median(af)/true_tot:>10.3f}"
                  f"{af.std()/true_tot:>10.3f}"
                  f"{np.median(ax):>13.1f}{np.median(ax)/true_tot:>10.3f}"
                  f"{ax.std()/true_tot:>10.3f}{np.median(dch):>16,.0f}")
        else:
            print(f"{'0 (空)':>8}{np.median(af):>11.1f}{'—':>10}{af.std():>10.1f}"
                  f"{np.median(ax):>13.1f}{'—':>10}{ax.std():>10.1f}"
                  f"{np.median(dch):>16,.0f}")
            print(f"{'':>8}{'↑ 假訊號底線(注入 0 卻量到的 A)':<44}"
                  f"純雜訊 dchi2 90% 分位 {np.percentile(dch, 90):>10,.0f}")
            print("-" * 96)


if __name__ == "__main__":
    main()
