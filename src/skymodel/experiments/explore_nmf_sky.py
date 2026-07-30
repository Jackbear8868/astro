import numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, PCA
import time

import sys
from pathlib import Path

# 搬到子目錄之後,同層的 templates / utils 不再自動可見。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import *

ROOT = Path(__file__).resolve().parents[3]
STEP01 = ROOT / "results/skymodel/step01"
WSKY = ROOT / "data/Haro11_NEpointing_wsky.fits"
NOSKY = ROOT / "data/Haro11_NEpointing_esonosky.fits"
WHITELIGHT = STEP01 / "whitelight.fits"
SEG = STEP01 / "seg.fits"

white_mask = fits.getdata(WHITELIGHT)
seg_mask   = fits.getdata(SEG)
valid_mask = white_mask != 0
blank_mask = valid_mask & ~((seg_mask > 0) & valid_mask)

hdr = fits.getheader(WSKY, "DATA")
wsky = fits.getdata(WSKY, "DATA")
wsky_variance = fits.getdata(WSKY, "STAT", memmap=True)
nosky = fits.getdata(NOSKY, "DATA")

mean_nosky = np.nanmean(nosky[:, blank_mask], axis=1)   # nosky 的 blank 平均光譜

nz = hdr["NAXIS3"]
wl = hdr["CRVAL3"] + (np.arange(hdr["NAXIS3"]) + 1 - hdr["CRPIX3"]) * hdr["CD3_3"]
blank_spectra = wsky[:, blank_mask]  # 2D (nz, n_blank)
blank_variance = wsky_variance[:, blank_mask]
mean_sky = np.nanmean(blank_spectra, axis=1)

start_time = time.time()

thresholds = (1, 2)
N_iterations = 5
LINE_MASK_DIR = STEP01 / "line_masks"
LINE_MASK_DIR.mkdir(parents=True, exist_ok=True)

continuum, sigma, line_mask, history = estimate_continuum(mean_sky, thresholds=thresholds, window=300, max_iter=N_iterations)

for i, (plt_continuum, plt_sigma, plt_line_mask) in enumerate(history):
    # --- Draw a chart for each iteration ---
    plt.figure(figsize=(12, 4))
    plt.plot(wl, mean_sky, lw=0.5, color="gray", label="mean sky")
    plt.plot(wl, plt_continuum, lw=0.5, color="blue", label="continuum")
    plt.plot(wl, plt_continuum + thresholds[0]*plt_sigma, lw=0.5, color="red", label=f"positive threshold ({thresholds[0]}σ)")
    plt.plot(wl, plt_continuum - thresholds[-1]*plt_sigma, lw=0.5, color="purple", label=f"negative threshold ({thresholds[-1]}σ)")
    plt.fill_between(wl, 0, mean_sky.max(), where=plt_line_mask, color="orange", alpha=0.2, label="detected lines")
    plt.ylim(0, 100)
    plt.xlabel("wavelength [A]"); plt.ylabel("flux")
    plt.title(f"iteration {i+1}: {int(plt_line_mask.sum())} lines")
    plt.legend()
    plt.savefig(LINE_MASK_DIR / f"line_mask_iter{i+1}.png", dpi=240)
    plt.close()

    plot_sky = mean_sky.copy()
    plot_sky[plt_line_mask] = np.nan       # 被遮的 bins → NaN → 線在那裡自動斷開留白
    # --- Draw a chart for non maked region ---
    plt.figure(figsize=(12, 4))
    plt.plot(wl, mean_sky, lw=0.5, color="gray", alpha=0.3 ,label="mean sky")
    plt.plot(wl, plot_sky, lw=0.5, color="red", label="mean sky (unmasked)")
    plt.ylim(0, 100)
    plt.xlabel("wavelength [A]"); plt.ylabel("flux")
    plt.title(f"iteration {i+1}: {int(nz - plt_line_mask.sum())} non masked lines")
    plt.legend()
    plt.savefig(LINE_MASK_DIR / f"non_line_mask_iter{i+1}.png", dpi=240)
    plt.close()

end_time = time.time()
print(f"Generate sky continuum took {end_time - start_time:.2f} seconds")
start_time = time.time()

# Use NMF to model the sky lines
L = blank_spectra - continuum[:, None]

X = np.nan_to_num(np.clip(L.T, 0, None))
K = 10
model = NMF(n_components=K, init="nndsvda", max_iter=300)
W = model.fit_transform(X)                       # (n_blank, K)：這些格子的振幅
basis = model.components_                            # (K, nz)：K 條基底模板
print("basis shape:", basis.shape)

end_time = time.time()
print(f"Generate sky line spectrum with NMF took {end_time - start_time:.2f} seconds")

sky_line = (W @ basis).T
sky = continuum[:, None] + sky_line
subtracted = blank_spectra - sky
mean_after  = np.nanmean(subtracted, axis=1)           
stats = spectrum_stats(mean_after)
print("Spectrum status with NMF: ",stats)
plot_compare(wl, mean_after, mean_nosky, STEP01 / "sky_subtracted_NMF.png")


# Use semi-NMF(official algorithm) to model the sky lines
L = blank_spectra - continuum[:, None]
start_time = time.time()
W_mu, B_mu = semi_NMF(np.nan_to_num(L.T).astype(np.float32), K=10, n_iter=10)
end_time = time.time()
print(f"semi-NMF (paper: kmeans+MU) took {end_time - start_time:.2f} s")

sky_line = (W_mu @ B_mu).T
sky = continuum[:, None] + sky_line
subtracted = blank_spectra - sky
mean_after_mu = np.nanmean(subtracted, axis=1)
print("Spectrum status with semi-NMF: ",spectrum_stats(mean_after_mu))
print("line:     ", spectrum_stats(mean_after_mu[line_mask]))
print("line-free:", spectrum_stats(mean_after_mu[~line_mask]))
plot_compare(wl, mean_after_mu, mean_nosky, STEP01 / "sky_subtracted_semi_NMF.png")


start_time = time.time()

W_chi2 = np.full(
    (blank_spectra.shape[1], B_mu.shape[0]),
    np.nan,
    dtype=np.float64,
)

for j in range(blank_spectra.shape[1]):
    W_chi2[j] = fit_chi2_coefficients(
        residual=L[:, j],
        variance=blank_variance[:, j],
        basis=B_mu,
    )

end_time = time.time()

print(
    f"Weighted coefficient fitting took "
    f"{end_time - start_time:.2f} s"
)

# 檢查有多少 spaxels 因資料不足或 basis rank 不足而無法求解。
failed_chi2 = np.any(~np.isfinite(W_chi2), axis=1)
print(
    f"Failed chi-square fits: "
    f"{failed_chi2.sum()}/{W_chi2.shape[0]}"
)

# 使用 weighted coefficients 重建 blank region 的 sky。
sky_line_chi2 = (W_chi2 @ B_mu).T
sky_chi2 = continuum[:, None] + sky_line_chi2
subtracted_chi2 = blank_spectra - sky_chi2
mean_after_chi2 = np.nanmean(subtracted_chi2, axis=1)

print(
    "Spectrum status with chi-square fitting: ",
    spectrum_stats(mean_after_chi2),
)
print(
    "line:     ",
    spectrum_stats(mean_after_chi2[line_mask]),
)
print(
    "line-free:",
    spectrum_stats(mean_after_chi2[~line_mask]),
)

plot_compare(
    wl,
    mean_after_chi2,
    mean_nosky,
    STEP01 / "sky_subtracted_semi_NMF_chi2.png",
)


# ----------------------------------------------------------------
# Per-spaxel continuum：只用來建立 coefficient fitting residual
# ----------------------------------------------------------------

start_time = time.time()

continuum_own = per_spaxel_continuum(
    spectra=blank_spectra,
    line_mask=line_mask,
    window=300,
    chunk=8000,
)

R_fit_own = blank_spectra - continuum_own

end_time = time.time()

print(
    f"Per-spaxel continuum filtering took "
    f"{end_time - start_time:.2f} s"
)

# R_fit_own 已經保留所需資訊，釋放約 1.1 GB 的 continuum_own 陣列。
del continuum_own


# ----------------------------------------------------------------
# 固定相同的 B_mu，改用 per-spaxel continuum residual 求 coefficients
# ----------------------------------------------------------------

start_time = time.time()

W_chi2_own = np.full(
    (blank_spectra.shape[1], B_mu.shape[0]),
    np.nan,
    dtype=np.float64,
)

for j in range(blank_spectra.shape[1]):
    W_chi2_own[j] = fit_chi2_coefficients(
        residual=R_fit_own[:, j],
        variance=blank_variance[:, j],
        basis=B_mu,
    )

end_time = time.time()

print(
    f"Per-spaxel-continuum weighted fitting took "
    f"{end_time - start_time:.2f} s"
)

failed_chi2_own = np.any(
    ~np.isfinite(W_chi2_own),
    axis=1,
)

print(
    f"Failed per-spaxel-continuum chi-square fits: "
    f"{failed_chi2_own.sum()}/{W_chi2_own.shape[0]}"
)


# 使用 per-spaxel residual 求得的 coefficients 重建 sky lines。
sky_line_chi2_own = (W_chi2_own @ B_mu).T

# 注意：最後扣除的 continuum 仍是從 blank region 得到的 shared sky continuum。
sky_chi2_own = continuum[:, None] + sky_line_chi2_own

subtracted_chi2_own = blank_spectra - sky_chi2_own
mean_after_chi2_own = np.nanmean(
    subtracted_chi2_own,
    axis=1,
)

print(
    "Spectrum status with per-spaxel-continuum chi-square fitting: ",
    spectrum_stats(mean_after_chi2_own),
)
print(
    "line:     ",
    spectrum_stats(mean_after_chi2_own[line_mask]),
)
print(
    "line-free:",
    spectrum_stats(mean_after_chi2_own[~line_mask]),
)

plot_compare(
    wl,
    mean_after_chi2_own,
    mean_nosky,
    STEP01 / "sky_subtracted_semi_NMF_chi2_own_continuum.png",
)

# A. 對每個 source spaxel,估它「自己」的連續譜 cont_own  → 算殘差 r_fit = s − cont_own
# B. 把 r_fit 投影到我們的天光線 basis,量出每個 spaxel 的天光線振幅 W
# C. 重建天光線:sky_lines = basis.T @ W
# D. 相減:clean = s − continuum(統一天光連續) − sky_lines
# E. 塞回 cube、重存 FITS,然後評估(source 保住沒)
