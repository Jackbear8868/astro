"""用 NMF 從 blank spaxels 學天光線 basis,並檢查扣完之後的殘差。"""
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
print("line:     ", spectrum_stats(mean_after[line_mask]))
print("line-free:", spectrum_stats(mean_after[~line_mask]))
