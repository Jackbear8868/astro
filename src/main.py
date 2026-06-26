"""
手把手：用 SEP 提取 source / background 區域，並畫出 segmentation 圖。

從專案根目錄執行：  uv run python src/main.py
"""
import numpy as np
import sep
from astropy.io import fits
import matplotlib.pyplot as plt

# ============================================================
# 步驟 1：把資料立方體收成一張 2D 偵測影像
# ------------------------------------------------------------
# cube 形狀是 (波長 3679, Y 499, X 559)。
# 這裡先取一段波長 (6500-6700 A，含 Halpha 發射線) 加總，
# 訊號集中、雜訊少，segmentation 會比較乾淨。
# ============================================================
WAVE_MIN, WAVE_MAX = 6500.0, 6700.0

hdul = fits.open("data/Haro11_nosky.fits")
cube = hdul["DATA"].data
hdr = hdul["DATA"].header

# 用標頭的 WCS 把「channel 編號」換算成「波長(埃)」
crval, crpix, cd = hdr["CRVAL3"], hdr["CRPIX3"], hdr["CD3_3"]
wavelengths = crval + (np.arange(cube.shape[0]) + 1 - crpix) * cd
band = (wavelengths >= WAVE_MIN) & (wavelengths <= WAVE_MAX)

# 只加總這段波長。nansum 忽略 NaN；轉成 native float32 (SEP 的要求)
image = np.nansum(cube[band], axis=0).astype(np.float32)
hdul.close()
print(f"步驟1: 影像 shape={image.shape}, 加總了 {band.sum()} 個 channel")

# 視場外圍是補零區，標記出「真實資料」像素，之後背景區會用到
valid = image != 0

# ============================================================
# 步驟 2：估計天空背景，並從影像扣除
# ------------------------------------------------------------
# sep.Background 把畫面分格、做 sigma-clipping，估出 2D 背景與雜訊(RMS)。
# image_sub = 扣掉背景後的乾淨影像，源才會凸顯出來。
# ============================================================
bkg = sep.Background(image)
image_sub = image - bkg
print(f"步驟2: 背景 level={bkg.globalback:.1f}, RMS(雜訊)={bkg.globalrms:.1f}")

# ============================================================
# 步驟 3：偵測源，並取得 segmentation map
# ------------------------------------------------------------
# segmentation_map=True 會多回傳一張「標籤影像」segmap：
#   - 值 = 0   -> 背景
#   - 值 = 1,2,3...-> 第 1,2,3... 個源所佔的像素
# THRESH: 要比雜訊亮幾倍(sigma)才算源。  MINAREA: 至少幾個相連像素。
# ============================================================
THRESH, MINAREA, DEBLEND_CONT = 5.0, 5, 0.005
sep.set_sub_object_limit(4096)
objects, segmap = sep.extract(
    image_sub,
    THRESH,
    err=bkg.globalrms,
    minarea=MINAREA,
    deblend_cont=DEBLEND_CONT,
    segmentation_map=True,
)
print(f"步驟3: 偵測到 {len(objects)} 個源")

# ============================================================
# 步驟 4：用 segmap 切出 source 區與 background 區
# ------------------------------------------------------------
# source 區     = segmap 有標籤的地方 (>0)
# background 區  = 真實資料 (valid) 中、不屬於任何源的地方
# ============================================================
source_mask = segmap > 0
background_mask = valid & ~source_mask
print(f"步驟4: source 像素={int(source_mask.sum())}, "
      f"background 像素={int(background_mask.sum())}")

# ============================================================
# 步驟 5：畫四格圖並存檔
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
vmin, vmax = np.percentile(image_sub[valid], [5, 99])

axes[0, 0].imshow(image_sub, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
axes[0, 0].set_title(f"Band image {WAVE_MIN:.0f}-{WAVE_MAX:.0f} A (bkg-subtracted)")

axes[0, 1].imshow(segmap, origin="lower", cmap="nipy_spectral")
axes[0, 1].set_title(f"Segmentation map ({len(objects)} objects)")

axes[1, 0].imshow(source_mask, origin="lower", cmap="gray")
axes[1, 0].set_title("Source region (segmap > 0)")

axes[1, 1].imshow(background_mask, origin="lower", cmap="gray")
axes[1, 1].set_title("Background region (segmap == 0)")

for ax in axes.ravel():
    ax.set_xlabel("x [pixel]")
    ax.set_ylabel("y [pixel]")
fig.tight_layout()
fig.savefig("results/segmentation.png", dpi=130)
print("步驟5: 已存檔 results/segmentation.png")
