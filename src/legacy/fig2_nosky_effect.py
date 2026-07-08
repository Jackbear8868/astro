"""fig2 — nosky null test：對「已無天空」的 cube 跑 ZAP，只會灌入雜訊。

資料流：src/run_zap_compare.py 的 collect 已把「空白天空區的逐波長標準差」算好存進 npz。
執行：  PYTHONPATH=libs/zap python3 src/fig2_nosky_effect.py
輸出：  results/zap/fig2_nosky_effect.png

看什麼：nosky 已被 MUSE 扣過天空、空白譜已平，ZAP 沒有天空可學；硬跑只會把每個 spaxel 的
        雜訊放大（per-spaxel std 暴增）。這是「對照組 / null test」，不是 ZAP 壞掉，而是餵錯 cube。
  - nosky raw   (藍) MUSE 已扣天空後的雜訊基準
  - nosky + ZAP (橘) 再跑 ZAP 後，雜訊被灌高
"""
import numpy as np
from scipy import ndimage as ndi
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "results/zap"
c = np.load(f"{OUT}/npz/_cache.npz"); wl = c["wl"]
N = np.load(f"{OUT}/npz/summ_nosky.npz")

# 輕度中值平滑：壓掉逐波長的尖峰森林、保留基準高度，兩條線的高低差才看得清楚
sm = lambda a: ndi.median_filter(a, 11)
curves = [
    ("nosky raw",   sm(N["raw_std"]), "tab:blue",   "-", 0.9),
    ("nosky + ZAP", sm(N["zap_std"]), "tab:orange", "-", 0.9),
]

fig, ax = plt.subplots(figsize=(13, 5.5))
for lab, y, col, ls, lw in curves:
    ax.plot(wl, y, col, ls=ls, lw=lw, label=lab)
ax.set_yscale("log")                                # 雜訊跨越量級 → 對數 y 軸
ax.set_xlim(wl.min(), wl.max())
ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
ax.set_ylabel(r"Blank-sky noise, per-spaxel std [$10^{-20}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]")
ax.set_title("ZAP on nosky")
ax.legend(loc="upper left")
for lam in (5577, 6300, 8400):
    ax.axvline(lam, color="k", ls=":", lw=0.4, alpha=0.35)

fig.tight_layout()
fig.savefig(f"{OUT}/fig2_nosky_effect.png", dpi=135); plt.close(fig)
print("saved", f"{OUT}/fig2_nosky_effect.png")
# 順便印出中位雜訊放大倍數，量化 null test 的傷害
print("median std: raw=%.2f  zap=%.2f  (x%.0f)" % (
    np.median(N["raw_std"]), np.median(N["zap_std"]), np.median(N["zap_std"])/np.median(N["raw_std"])))
