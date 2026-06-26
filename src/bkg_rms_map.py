"""畫出 SEP 估計的背景 RMS (2D 雜訊地圖)。"""
import numpy as np
import sep
from scipy.ndimage import binary_dilation
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FITS_FILE = "Haro11_nosky.fits"
WAVE_MIN, WAVE_MAX = 4750.0, 7000.0

# 1. 取波段影像（和 segment_background.py 相同）
hdul = fits.open(FITS_FILE)
cube = hdul["DATA"].data
hdr = hdul["DATA"].header
crval, crpix, cd = hdr["CRVAL3"], hdr["CRPIX3"], hdr["CD3_3"]
w = crval + (np.arange(cube.shape[0]) + 1 - crpix) * cd
band = (w >= WAVE_MIN) & (w <= WAVE_MAX)
image = np.nansum(cube[band], axis=0).astype(np.float32)
hdul.close()

# 2. 估背景，取出 2D RMS 地圖
#    負值的成因：bkg.rms() 把低解析網格用雙三次樣條放大回原尺寸，
#    中央亮星系讓 RMS 網格暴衝(~500 -> ~24000)，樣條在陡邊過衝而掉到負值。
#    解法：估背景前先把「補零區 + 亮源」都遮掉，讓 RMS 網格保持平坦。
zero = (image == 0)
b0 = sep.Background(image, mask=zero)          # 第一次粗估，用來找亮源
src = (image - b0) > 2 * b0.globalrms          # 比背景亮 2sigma 視為源
src = binary_dilation(src, iterations=5)       # 膨脹一圈，連星系外暈一起蓋住
full_mask = zero | src                         # 補零 + 亮源 一起遮

bkg = sep.Background(image, mask=full_mask)    # 只用純天空像素估背景
rms_map = bkg.rms()   # 與影像同尺寸的 2D 陣列
valid = ~zero
print(f"globalrms = {bkg.globalrms:.2f}")
print(f"資料區 rms_map 範圍: {rms_map[valid].min():.1f} - {rms_map[valid].max():.1f}")
print(f"資料區負值像素數: {int(((rms_map < 0) & valid).sum())}")

# 3. 畫圖：左邊波段影像，右邊 RMS 地圖
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

vmin, vmax = np.percentile(image[image != 0], [5, 99])
axes[0].imshow(image, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
axes[0].set_title(f"Band image {WAVE_MIN:.0f}-{WAVE_MAX:.0f} A")

im = axes[1].imshow(rms_map, origin="lower", cmap="viridis")
axes[1].set_title("Background RMS map  bkg.rms()")
fig.colorbar(im, ax=axes[1], label="RMS (noise)")

for ax in axes:
    ax.set_xlabel("x [pixel]")
    ax.set_ylabel("y [pixel]")
fig.tight_layout()
fig.savefig("results/bkg_rms_map.png", dpi=150)
print("已存檔: results/bkg_rms_map.png")
