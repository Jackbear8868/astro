"""fig1 — ZAP 對 wsky 的天空線扣除效果（天空線是否被壓到接近 MUSE 真值）。

資料流：src/run_zap_compare.py 的 collect 已把「空白天空區的中位光譜」算好存進 npz，
        本腳本只讀 npz、畫圖，完全不碰幾 GB 的大 cube（記憶體極省）。
執行：  PYTHONPATH=libs/zap python3 src/fig1_wsky_effect.py
輸出：  results/zap/fig1_wsky_effect.png

看什麼：三條「空白天空 spaxel 的中位光譜」對照，證明 wsky+ZAP 有效——
  - wsky raw   (灰) 還含天空的原始 cube → 高聳的天空線森林
  - wsky + ZAP (紅) 用 ZAP 扣天空後     → 應被壓到接近真值
  - nosky      (藍) MUSE pipeline 已扣天空 = 參考真值
"""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # Agg：無視窗環境也能存圖

OUT = "results/zap"
# _cache.npz 只提供波長軸 wl（畫圖 x 軸）；summ_*.npz 提供各 cube 的中位光譜 med
c = np.load(f"{OUT}/npz/_cache.npz"); wl = c["wl"]
W = np.load(f"{OUT}/npz/summ_wsky.npz")     # W = wsky（含天空）
N = np.load(f"{OUT}/npz/summ_nosky.npz")    # N = nosky（MUSE 真值）

# 每條線：(圖例文字, y 資料, 顏色, 線型, 線寬)
curves = [
    ("wsky raw (sky present)", W["raw_med"], "0.55",     "-", 0.6),  # 原始：天空線還在
    ("wsky + ZAP",             W["zap_med"], "tab:red",  "-", 0.8),  # ZAP 扣天空後
    ("nosky (MUSE truth)",     N["raw_med"], "tab:blue", "-", 0.8),  # MUSE 參考真值
]

fig, ax = plt.subplots(figsize=(13, 5.5))
for lab, y, col, ls, lw in curves:
    ax.plot(wl, y, col, ls=ls, lw=lw, label=lab)
# symlog：linthresh=1 以內線性、以外對數 → 同一張圖同時容納「高聳天空線」與「近零殘餘」
ax.set_yscale("symlog", linthresh=1.0)
ax.set_ylim(-1, 1000)
ax.set_xlim(wl.min(), wl.max())
ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
ax.set_ylabel(r"Blank-sky median flux [$10^{-20}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]")
ax.set_title("ZAP on wsky")
ax.legend(loc="upper left")
for lam in (5577, 6300, 8400):                     # 標出三條主要天空線位置
    ax.axvline(lam, color="k", ls=":", lw=0.4, alpha=0.35)

fig.tight_layout()
fig.savefig(f"{OUT}/fig1_wsky_effect.png", dpi=135); plt.close(fig)
print("saved", f"{OUT}/fig1_wsky_effect.png")
