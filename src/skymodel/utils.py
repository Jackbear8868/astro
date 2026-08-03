"""Shared sky-spectrum utilities: running median + iterative line detection."""

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import matplotlib
matplotlib.use("Agg")              # 必須在 import pyplot 之前
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


def running_median(spectrum, window=300):
    half = window // 2
    n = len(spectrum)
    result = np.empty(n)
    for j in range(n):
        result[j] = np.nanmedian(spectrum[max(0, j - half):j + half])
    return result


def detect_lines(mean_sky, exclude=None, thresholds = (1, 2), window=300):
    m = mean_sky.copy()
    if exclude is not None:
        m[exclude] = np.nan                        # 上一輪抓到的線 → 不參與 continuum
    continuum = running_median(m, window)
    
    x = np.arange(len(m))
    good = np.isfinite(m) & np.isfinite(continuum)
    xg, yg = x[good], continuum[good]
    spl = UnivariateSpline(xg, yg, k=3, s=len(xg) * 0.05**2, ext=3)
    continuum = spl(x)

    abs_diff = np.abs(m - continuum)
    sigma = running_median(abs_diff, window)       # 對波長方向取 running median

    line_mask = (mean_sky > continuum + thresholds[0] * sigma) | (mean_sky < continuum - thresholds[1] * sigma)
    return continuum, sigma, line_mask


def estimate_continuum(mean_sky, thresholds=(1, 2), window=300, max_iter=5, min_unmasked_frac=0.16):
    line_mask = None
    history = []

    for i in range(max_iter):
        continuum, sigma, new_mask = detect_lines(mean_sky, exclude=line_mask, thresholds=thresholds, window=window)
        
        unmasked_frac = 1.0 - new_mask.sum() / new_mask.size
        if unmasked_frac < min_unmasked_frac:
            print(f"Iteration {i+1}: unmasked fraction {unmasked_frac:.1%} < floor {min_unmasked_frac:.0%}. Stop iteration.")
            break

        if line_mask is not None and np.array_equal(new_mask, line_mask):
            break

        line_mask = new_mask
        history.append((continuum, sigma, line_mask))
    
    if not history:
        raise ValueError(f"First iteration already masked more than {1 - min_unmasked_frac:.0%} of the spectrum.\n"
        "Check the input spectrum and parameters.")

    return history[-1][0], history[-1][1], history[-1][2], history


def fit_chi2_coefficients(residual, variance, basis):
    """固定 sky basis，以 inverse-variance weighted least squares 求一條光譜的係數。

    Parameters
    ----------
    residual : ndarray, shape (nz,)
        一個 spaxel 中準備拿來擬合 sky lines 的殘差光譜。
    variance : ndarray, shape (nz,)
        同一個 spaxel 的 MUSE STAT；其數值是每個波長的 variance。
    basis : ndarray, shape (K, nz)
        只從 blank spaxels 學到的 K 條固定 sky-line basis。

    Returns
    -------
    coefficients : ndarray, shape (K,)
        讓 chi-square 最小的 K 個 basis 係數。若有效資料不足或 basis
        不具完整 rank，回傳 K 個 NaN，表示無法可靠求解。
    """
    # 每一個條件的 shape 都是 (nz,)。只有 data、STAT 與全部 basis
    # 都是有限值，且 STAT > 0 的波長，才能參與 1/STAT 加權擬合。
    good = (
        np.isfinite(residual)
        & np.isfinite(variance)
        & (variance > 0)
        & np.all(np.isfinite(basis), axis=0)
    )

    n_coeff = basis.shape[0]  # K：需要求解的 sky-basis 係數數量

    # 有效觀測數必須多於未知係數數量，否則沒有多餘資料約束這個 fit。
    if good.sum() <= n_coeff:
        return np.full(n_coeff, np.nan)

    y = residual[good].astype(np.float64)          # (n_good,)
    A = basis[:, good].T.astype(np.float64)        # (n_good, K)

    # STAT = sigma^2；除以 sigma 等價於讓平方殘差帶有 1/STAT 權重。
    sigma = np.sqrt(variance[good].astype(np.float64))
    y_white = y / sigma                            # (n_good,)
    A_white = A / sigma[:, None]                   # (n_good, K)

    # 同時求解 K 個可正可負的係數，使 ||y_white - A_white @ w||^2 最小。
    coefficients, _, rank, _ = np.linalg.lstsq(
        A_white,
        y_white,
        rcond=None,
    )

    # rank < K 表示 basis 中沒有 K 個獨立方向，因此係數不是唯一解。
    if rank < n_coeff:
        return np.full(n_coeff, np.nan)

    return coefficients


def spectrum_stats(spec):
    """把一條光譜濃縮成摘要統計。"""
    spec = spec[np.isfinite(spec)]                     # 先丟掉 NaN
    return {
        "mean":          np.mean(spec),
        "sigma":         np.std(spec),
        "skewness":      skew(spec),
        "kurtosis":      kurtosis(spec),
        "rms_from_zero": np.sqrt(np.mean(spec**2)),    # sqrt(平方的平均) = 離 0 的均方根
    }


def plot_compare(wl, spec, spec_compare, out_path, label="ours", label_compare="nosky", ylim=(-20, 20), title=None):
    """對照圖：左光譜（藍=spec、橘虛線=spec_compare），右兩組 stats。"""
    fig, (ax, stat_ax) = plt.subplots(1, 2, figsize=(15.5, 4.5), gridspec_kw={"width_ratios": [5, 1]})
    ax.axhline(0, color="0.5", lw=0.5)
    ax.plot(wl, spec, lw=0.9, color="#1f77b4", label=label)
    ax.plot(wl, spec_compare, lw=0.9, ls="--", alpha=0.7, color="#e8710a", label=label_compare)
    ax.set_ylim(*ylim)
    ax.set_xlabel("wavelength [A]"); ax.set_ylabel("flux")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)

    stat_ax.axis("off")
    def fmt(name, s):
        st = spectrum_stats(s)
        return f"[{name}]\n" + "\n".join(f"{k:<13} = {v:.4g}" for k, v in st.items())
    stat_ax.text(0, 0.95, fmt(label, spec), color="#1f77b4", va="top", family="monospace", fontsize=8, transform=stat_ax.transAxes)
    stat_ax.text(0, 0.45, fmt(label_compare, spec_compare), color="#e8710a", va="top", family="monospace", fontsize=8, transform=stat_ax.transAxes)

    fig.tight_layout()
    fig.savefig(out_path, dpi=145)
    plt.close()


def per_spaxel_continuum(
    spectra,
    line_mask,
    window=300,
    chunk=8000,
):
    """估計每個 spaxel 自己的平滑 continuum。

    Parameters
    ----------
    spectra : ndarray, shape (nz, n_spaxels)
        多個 spaxels 的光譜；每一欄是一個 spaxel。
    line_mask : ndarray of bool, shape (nz,)
        從 mean blank-sky spectrum 偵測出的全域 sky-line mask。
        True 表示該 wavelength 不參與 running median。
    window : int
        running median 的 wavelength window，單位是 spectral pixels。
        目前沿用既有值 300，不在這次 comparison 中改變。
    chunk : int
        每次同時處理的 spaxel 數量，只控制記憶體與速度，
        不改變 continuum 的科學定義。

    Returns
    -------
    continuum_own : ndarray, shape (nz, n_spaxels)
        每個 spaxel 各自估計的平滑 continuum。
    """
    nz, n_spaxels = spectra.shape

    continuum_own = np.empty(
        (nz, n_spaxels),
        dtype=np.float32,
    )

    for low in range(0, n_spaxels, chunk):
        high = min(low + chunk, n_spaxels)

        chunk_spectra = spectra[:, low:high].astype(
            np.float64,
            copy=True,
        )

        chunk_spectra[line_mask, :] = np.nan

        chunk_continuum = (
            pd.DataFrame(chunk_spectra)
            .rolling(
                window=window,
                center=True,
                min_periods=1,
            )
            .median()
            .to_numpy()
            .astype(np.float32)
        )

        continuum_own[:, low:high] = chunk_continuum

    return continuum_own