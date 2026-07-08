"""fig4 — 源遮罩（segmentation）：Hα 窄帶影像 + 遮罩輪廓，檢查「哪些像素被當成源」。

資料流：讀 results/zap/fits/source_mask.fits（由 run_zap_compare.py mask 產生）+ crop_nosky.fits。
執行：  python3 src/fig4_source_mask.py
輸出：  results/zap/fig4_source_mask.png

遮罩參數（此處僅記錄；真正的偵測在 run_zap_compare.py 的 cmd_mask，依 CLAUDE.md 原則2）：
  Source Extractor (sep)：背景框 bw=256、matched filter 高斯 FWHM=6px(=seeing)、
  門檻 2σ、minarea=30(≈1 PSF 面積)、事後 dilation=6px(≈1×seeing)。
左圖：純 Hα 窄帶影像；右圖：同影像疊上紅色源遮罩輪廓（遮罩內＝源，之後餵給 ZAP 保護）。
"""
import numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "results/zap"
HA_LINE = (6692.0, 6708.0)                       # Hα(+核心) 發射線窗（z≈0.0206）
HA_CONT = ((6605.0, 6645.0), (6760.0, 6795.0))   # 兩側乾淨連續譜窗（避開 [NII]）

# ---- 重建 Hα 窄帶影像（線內平均 − 兩側連續譜平均），只讀需要的波段 ----
hd = fits.open(f"{OUT}/fits/crop_nosky.fits", memmap=True)   # memmap：不整份載入，省記憶體
h = hd["DATA"].header
wl = h["CRVAL3"] + (np.arange(h["NAXIS3"]) + 1 - h["CRPIX3"]) * h["CD3_3"]   # 由 header 算波長軸
data = hd["DATA"].data
li = (wl > HA_LINE[0]) & (wl < HA_LINE[1])       # Hα 線窗平面
ci = np.zeros_like(wl, bool)
for a, b in HA_CONT:                             # 兩側連續譜窗平面（聯集）
    ci |= (wl > a) & (wl < b)
ha = (np.nanmean(data[li], 0) - np.nanmean(data[ci], 0)).astype(np.float32)  # 純 Hα 影像
mid = data[h["NAXIS3"] // 2]                     # 取中間一個波長平面來判斷有效視場
valid = mid != 0
hd.close()

# ---- 讀源遮罩，算覆蓋率與顯示範圍 ----
mask = fits.open(f"{OUT}/fits/source_mask.fits")[0].data.astype(bool)
cov = 100 * mask.sum() / valid.sum()             # 源遮罩覆蓋了多少比例的有效視場
sig = 1.4826 * np.median(np.abs(ha[valid] - np.median(ha[valid])))  # 穩健噪音 σ_MAD（當顯示下限）
vmax = np.percentile(ha[valid], 99)              # 顯示上限取 99 百分位，避免被極亮點洗掉對比

# ---- 左：Hα 影像；右：Hα 影像 + 遮罩輪廓 ----
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax[0].imshow(ha, origin="lower", vmin=-sig, vmax=vmax, cmap="gray")
ax[0].set_title(r"H$\alpha$ narrowband", fontsize=14)
ax[1].imshow(ha, origin="lower", vmin=-sig, vmax=vmax, cmap="gray")
ax[1].imshow(np.where(mask, 1, np.nan), origin="lower", cmap="autumn", alpha=0.30)  # 遮罩半透明填色
ax[1].contour(mask.astype(float), levels=[0.5], colors="red", linewidths=0.8)       # 遮罩邊界紅線
ax[1].set_title(r"H$\alpha$ source mask", fontsize=14)
for a in ax:
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_source_mask.png", dpi=130); plt.close(fig)
print(f"saved {OUT}/fig4_source_mask.png  (coverage {cov:.0f}%)")
