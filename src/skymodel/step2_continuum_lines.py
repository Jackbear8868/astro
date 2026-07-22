from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter, binary_dilation

import matplotlib
matplotlib.use("Agg")              # 關鍵：先設定「畫到檔案」模式
import matplotlib.pyplot as plt    # 畫圖的主介面，慣例縮寫成 plt

ROOT = Path(__file__).resolve().parents[2]

WSKY_CUBE = ROOT / "data/Haro11_NEpointing_wsky.fits"
NOSKY_CUBE = ROOT / "data/Haro11_NEpointing_esonosky.fits"

ROOT = Path(__file__).resolve().parents[2]
STEP01_OUTPUT = ROOT / "results/skymodel/step01"
STEP02_OUTPUT = ROOT / "results/skymodel/step02"

white = fits.getdata(STEP01_OUTPUT / "whitelight.fits")
seg   = fits.getdata(STEP01_OUTPUT / "seg.fits")
valid_mask  = white != 0
source_mask = (seg > 0) & valid_mask
blank_mask  = valid_mask & ~source_mask

mean_wsky = np.load(STEP01_OUTPUT / "mean_wsky.npy")

wsky_mean_C = median_filter(mean_wsky, size=300)         # 跟 step2 一樣：continuum
wsky_mean_L = mean_wsky - wsky_mean_C                          # 純線譜
median = np.median(wsky_mean_L)

# Median Absolute Deviation

mad = np.median(np.abs(wsky_mean_L - median)) # median of the absolute deviations from the median
print(mad)


with fits.open(WSKY_CUBE) as hdul:
    hdr = hdul["DATA"].header
    nz = hdr["NAXIS3"]
    data = np.asarray(hdul["DATA"].data, np.float32)   # 讀取整個 cube
    wavelength_range = hdr["CRVAL3"] + (np.arange(nz) + 1 - hdr["CRPIX3"]) * hdr["CD3_3"]

    blank_spectra = data[:, blank_mask]                    # 先挑 blank 格子 → (3801, 72819)
    blank_L = np.abs(blank_spectra - wsky_mean_C[:, None])         # 扣 continuum（二維，只一個 None）

    step = 25
    half = 150                                          # 窗寬 300 的一半
    idx = np.arange(0, nz, step)                        # 取樣點：0, 25, 50, ... 共約 153 個
    sampled = np.stack([np.nanmedian(blank_L[max(0, j-half):j+half], axis=0) for j in idx])
    # sampled 形狀 (153, 72819)：每個取樣點、每個 blank 格子的中位數
    sampled_mean = np.nanmean(sampled, axis=1)                 # 對 blank 平均 → (153,)
    median_spectrum = np.interp(np.arange(nz), idx, sampled_mean)   # 內插回 3801 個波長 → (3801,)

    plt.figure(figsize=(13, 4))
    plt.plot(wavelength_range, median_spectrum, lw=0.5, label="wsky_mean_L - median")
    plt.xlabel("wavelength [A]")
    plt.ylabel("flux")
    plt.ylim(0, np.nanpercentile(mean_wsky, 99))
    plt.title("Spectra from blank spaxels (WSKY cube)")
    plt.tight_layout()
    plt.legend()
    plt.savefig(STEP02_OUTPUT / "spectrum.png", dpi=130)
    plt.close()
    print("saved", STEP02_OUTPUT / "spectrum.png")
    # np.save(STEP02_OUTPUT / "median_wsky.npy", median_wsky)
    # print("saved", STEP02_OUTPUT / "median_wsky.npy")