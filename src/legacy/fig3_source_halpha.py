"""fig3 — 源 Hα 保真：三種處理對星系發射線的保留程度。

資料流：src/run_zap_compare.py 的 collect 已把「最亮 spaxel 的整條光譜」算好存進 npz。
執行：  PYTHONPATH=libs/zap python3 src/fig3_source_halpha.py
輸出：  results/zap/fig3_source_halpha.png

看什麼：最亮 spaxel（星系核心）在 Hα 附近的光譜，檢查源有沒有被保住 / 被吃掉——
  - nosky raw  (藍)   MUSE 真值：源本來該長的樣子
  - wsky + ZAP (紅虛) 正確用法：應貼合真值 → 源保住
  - nosky + ZAP(橘)   錯誤用法：對已無天空的 cube 跑 ZAP → 把源吃壞（暴衝失真）
"""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "results/zap"
c = np.load(f"{OUT}/npz/_cache.npz"); wl = c["wl"]
W = np.load(f"{OUT}/npz/summ_wsky.npz"); N = np.load(f"{OUT}/npz/summ_nosky.npz")

s = (wl > 6600) & (wl < 6800)               # 只畫 Hα 附近窗口，聚焦發射線細節
# 每條線：(圖例文字, y 資料, 顏色, 線型, 線寬)
curves = [
    ("nosky raw (MUSE truth)", N["raw_srcspec"], "tab:blue",   "-",  1.4),
    ("wsky + ZAP",             W["zap_srcspec"], "tab:red",    "--", 1.2),
    ("nosky + ZAP",            N["zap_srcspec"], "tab:orange", "-",  1.0),
]

fig, ax = plt.subplots(figsize=(11, 6))
for lab, y, col, ls, lw in curves:
    ax.plot(wl[s], y[s], col, ls=ls, lw=lw, label=lab)
ax.set_xlim(6600, 6800)
ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
ax.set_ylabel(r"Brightest-spaxel flux [$10^{-20}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]")
ax.set_title(r"Source H$\alpha$ fidelity")
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_source_halpha.png", dpi=135); plt.close(fig)
print("saved", f"{OUT}/fig3_source_halpha.png")
