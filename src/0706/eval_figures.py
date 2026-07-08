"""
eval_figures — 從摘要 npz 畫出 fig1-4（evaluate 的「present」階段，完全不碰大 cube）。
  輸入： results/zap/npz/summ_*.npz, _cache.npz  +  settings.SOURCE_MASK_PATH（fig4 用）
  輸出： results/zap/fig1_wsky_effect.png / fig2_nosky_effect.png
         results/zap/fig3_source_halpha.png / fig4_source_mask.png
  跑法： conda run -n astro python src/0706/eval_figures.py          # 全畫
         conda run -n astro python src/0706/eval_figures.py fig2     # 只畫一張

  每張圖看什麼：
    fig1 天空線扣除 (wsky raw / wsky+ZAP / nosky 真值)
    fig2 nosky null test（對已無天空的 cube 跑 ZAP → 雜訊放大）
    fig3 源 Hα 保真（wsky+ZAP 應貼合真值；nosky+ZAP 吃壞源）
    fig4 源遮罩（Hα 影像 + 遮罩輪廓）
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

_YL_FLUX = r"[$10^{-20}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]"

def _load():
    wl = np.load(str(settings.CACHE_NPZ_PATH))["wl"]
    nosky = np.load(str(settings.summary_npz_path("nosky")))
    wsky  = np.load(str(settings.summary_npz_path("wsky")))
    return wl, nosky, wsky

def fig1():
    """天空線扣除：三條空白區中位光譜。"""
    wl, nosky, wsky = _load()
    curves = [("wsky raw (sky present)", wsky["raw_med"], "0.55", "-", 0.6),
              ("wsky + ZAP", wsky["zap_med"], "tab:red", "-", 0.8),
              ("nosky (MUSE truth)", nosky["raw_med"], "tab:blue", "-", 0.8)]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for lab, y, col, ls, lw in curves:
        ax.plot(wl, y, col, ls=ls, lw=lw, label=lab)
    ax.set_yscale("symlog", linthresh=1.0); ax.set_ylim(-1, 1000); ax.set_xlim(wl.min(), wl.max())
    ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]"); ax.set_ylabel("Blank-sky median flux " + _YL_FLUX)
    ax.set_title("ZAP on wsky"); ax.legend(loc="upper left")
    for lam in settings.SKY_LINE_WAVELENGTHS:
        ax.axvline(lam, color="k", ls=":", lw=0.4, alpha=0.35)
    _save(fig, "fig1_wsky_effect.png")

def fig2():
    """nosky null test：對已無天空的 cube 跑 ZAP，雜訊放大。"""
    from scipy import ndimage as ndi
    wl, nosky, wsky = _load()
    smooth = lambda a: ndi.median_filter(a, 11)           # 輕度平滑，線條看得清
    curves = [("nosky raw", smooth(nosky["raw_std"]), "tab:blue", "-", 0.9),
              ("nosky + ZAP", smooth(nosky["zap_std"]), "tab:orange", "-", 0.9)]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for lab, y, col, ls, lw in curves:
        ax.plot(wl, y, col, ls=ls, lw=lw, label=lab)
    ax.set_yscale("log"); ax.set_xlim(wl.min(), wl.max())
    ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]"); ax.set_ylabel("Blank-sky noise, per-spaxel std " + _YL_FLUX)
    ax.set_title("ZAP on nosky"); ax.legend(loc="upper left")
    for lam in settings.SKY_LINE_WAVELENGTHS:
        ax.axvline(lam, color="k", ls=":", lw=0.4, alpha=0.35)
    _save(fig, "fig2_nosky_effect.png")
    print("  median std: raw=%.2f  zap=%.2f  (x%.0f)" % (
        np.median(nosky["raw_std"]), np.median(nosky["zap_std"]), np.median(nosky["zap_std"]) / np.median(nosky["raw_std"])))

def fig3():
    """源 Hα 保真：最亮 spaxel 在 Hα 附近的光譜。"""
    wl, nosky, wsky = _load()
    s = (wl > 6600) & (wl < 6800)
    curves = [("nosky raw (MUSE truth)", nosky["raw_srcspec"], "tab:blue", "-", 1.4),
              ("wsky + ZAP", wsky["zap_srcspec"], "tab:red", "--", 1.2),
              ("nosky + ZAP", nosky["zap_srcspec"], "tab:orange", "-", 1.0)]
    fig, ax = plt.subplots(figsize=(11, 6))
    for lab, y, col, ls, lw in curves:
        ax.plot(wl[s], y[s], col, ls=ls, lw=lw, label=lab)
    ax.set_xlim(6600, 6800)
    ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]"); ax.set_ylabel("Brightest-spaxel flux " + _YL_FLUX)
    ax.set_title(r"Source H$\alpha$ fidelity"); ax.legend(loc="upper right")
    _save(fig, "fig3_source_halpha.png")

def fig4():
    """源遮罩：Hα 窄帶影像 + 紅色遮罩輪廓。"""
    from astropy.io import fits
    hd = fits.open(str(settings.input_cube_path("nosky")), memmap=True)
    h = hd["DATA"].header; wl = settings.wavelength_axis(h)
    ha = settings.halpha_narrowband_image(hd["DATA"].data, wl)
    valid = hd["DATA"].data[h["NAXIS3"] // 2] != 0
    hd.close()
    mask = fits.open(str(settings.SOURCE_MASK_PATH))[0].data.astype(bool)
    cov = 100 * mask.sum() / valid.sum()
    sig = 1.4826 * np.median(np.abs(ha[valid] - np.median(ha[valid])))
    vmax = np.percentile(ha[valid], 99)
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    ax[0].imshow(ha, origin="lower", vmin=-sig, vmax=vmax, cmap="gray")
    ax[0].set_title(r"H$\alpha$ narrowband", fontsize=14)
    ax[1].imshow(ha, origin="lower", vmin=-sig, vmax=vmax, cmap="gray")
    ax[1].imshow(np.where(mask, 1, np.nan), origin="lower", cmap="autumn", alpha=0.30)
    ax[1].contour(mask.astype(float), levels=[0.5], colors="red", linewidths=0.8)
    ax[1].set_title(r"H$\alpha$ source mask", fontsize=14)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    _save(fig, "fig4_source_mask.png", extra=f"(coverage {cov:.0f}%)")

def _save(fig, name, extra=""):
    fig.tight_layout()
    out = settings.FIGURE_DIR / name
    fig.savefig(str(out), dpi=135 if name.startswith("fig") and "mask" not in name else 130)
    plt.close(fig)
    print(f"saved {out}  {extra}".rstrip())

ALL = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4}

def all_figures():
    for f in (fig1, fig2, fig3, fig4):
        f()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        ALL[sys.argv[1]]()
    else:
        all_figures()
