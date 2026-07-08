"""
CGM Hα 延展結構分析：比較 MUSE(nosky) 與 ZAP(wsky+ZAP) 清過天空的整張 cube,
看哪個能把 Haro11 外圍的 circumgalactic Hα 暈揭露得更深 / 更乾淨。

看什麼：外圍的電離氣(CGM)很暗、容易被天空殘餘與雜訊淹沒。天空扣得越乾淨,
        暗的 Hα 暈就能被偵測到越遠 → 用「方位平均徑向剖面跌到 3σ 的半徑」量化延展程度。

只讀少數波長平面 (記憶體輕), 從專案根目錄執行:
  PYTHONPATH=libs/zap python3 src/cgm_halpha.py
  （實際請用 conda env astro：conda run -n astro python src/cgm_halpha.py）
產出:
  results/zap/fig6_cgm_halpha_maps.png   (Hα 表面亮度圖: nosky vs wsky+ZAP, 全視場 log)
  results/zap/fig7_cgm_radial_profile.png(方位平均 Hα SB vs 半徑 + 偵測極限)
"""
import sys, os, warnings, numpy as np
sys.path.insert(0, "libs/zap"); warnings.filterwarnings("ignore")
from astropy.io import fits
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt  # Agg：無視窗環境也能存圖

OUT = "results/zap"
# 三個要比較的 cube：MUSE 扣天空(真值)、wsky+ZAP(正確用法)、nosky+ZAP(null test 對照)
CUBES = {"nosky (MUSE)": f"{OUT}/fits/crop_nosky.fits", "wsky + ZAP": f"{OUT}/fits/wsky_zap.fits",
         "nosky + ZAP": f"{OUT}/fits/nosky_zap.fits"}
PIXSCALE = 0.2                       # MUSE WFM: 0.2 arcsec / pixel（把像素半徑換算成角秒用）
# Hα(z≈0.0206)≈6699; 線窗 + 兩側乾淨連續譜(避開 [NII]6548/6583 @ ~6682/6718)
HA_LINE = (6692.0, 6708.0)
HA_CONT = ((6605.0, 6645.0), (6760.0, 6795.0))

def wl_axis(hdr):
    # 由 header (CRVAL3/CRPIX3/CD3_3) 算出每個波長平面的波長 [Å]
    return hdr["CRVAL3"] + (np.arange(hdr["NAXIS3"]) + 1 - hdr["CRPIX3"]) * hdr["CD3_3"]

def halpha_map(path):
    """回傳 (Hα 積分通量面, 連續譜面)。單位 1e-20 erg/s/cm2 / spaxel。"""
    hd = fits.open(path); d = hd["DATA"].data; hdr = hd["DATA"].header
    wl = wl_axis(hdr); dlam = abs(hdr["CD3_3"])            # dlam = 每個波長平面的寬度 (積分用)
    li = (wl > HA_LINE[0]) & (wl < HA_LINE[1])             # Hα 線窗平面
    ci = np.zeros_like(wl, bool)
    for a, b in HA_CONT: ci |= (wl > a) & (wl < b)         # 兩側連續譜平面 (聯集)
    cont = np.nanmean(d[ci], axis=0)                       # 每 spaxel 的連續譜基準
    line = np.nansum(d[li] - cont, axis=0) * dlam          # 扣連續譜後沿波長積分 = 線通量
    hd.close()
    return line.astype(np.float32), cont.astype(np.float32)

def main():
    from astropy.stats import sigma_clipped_stats
    raw = {k: halpha_map(v) for k, v in CUBES.items()}     # 每個 cube → (線通量面, 連續譜面)
    maps = {k: m for k, (m, _) in raw.items()}             # 只取線通量面來畫圖 / 做剖面
    # 共同有效視場 = nosky 連續譜非零 (wsky_zap 被 ZAP 補滿到邊界, 那圈是假的)
    valid = np.isfinite(raw["nosky (MUSE)"][1]) & (raw["nosky (MUSE)"][1] != 0)
    for k in maps:                                         # 視場外設 NaN, 不參與統計/繪圖
        maps[k] = np.where(valid, maps[k], np.nan)
    ny, nx = valid.shape
    # 星系中心 = nosky 線通量最亮的 spaxel
    cy, cx = np.unravel_index(np.nanargmax(np.where(valid, raw["nosky (MUSE)"][1], -np.inf)), valid.shape)
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - cy, xx - cx) * PIXSCALE             # 每 spaxel 到中心的半徑 (arcsec)

    # 穩健噪聲: 外圈(r>25") 做 sigma-clip (剔除前景星/熱點), 取 clip 後 std
    outer = valid & (r > 25.0)
    sigma = {k: float(sigma_clipped_stats(m[outer], sigma=3.0, maxiters=5)[2]) for k, m in maps.items()}

    # ---- fig6: Hα SB 圖 (log, 全視場, 兩格用同一色階) ----
    fig6_keys = ["nosky (MUSE)", "wsky + ZAP"]            # 地圖只放這兩個; nosky+ZAP 是 null test, 太吵不放
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.2))
    vmax = np.nanpercentile(maps["nosky (MUSE)"], 99.9)   # 色階上限取 99.9 百分位 (避免被極亮核心洗掉對比)
    floor = max(min(sigma[k] for k in fig6_keys), 1e-3)   # 色階下限 = 較乾淨者的 1σ (log 不能吃 <=0)
    for a, k in zip(ax, fig6_keys):
        m = maps[k]
        im = a.imshow(np.log10(np.clip(m, floor, None)), origin="lower",
                      cmap="inferno", vmin=np.log10(floor), vmax=np.log10(vmax))
        a.plot(cx, cy, "c+", ms=10)                       # 標出星系中心
        a.set_title(f"{k}   (1σ/spaxel = {sigma[k]:.1f})")
        a.set_xlabel("x [pix]"); a.set_ylabel("y [pix]")
        plt.colorbar(im, ax=a, fraction=0.046, label="log10 Hα flux [1e-20 erg/s/cm2]")
    fig.suptitle("Haro11 CGM Hα surface brightness (full field, log)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig6_cgm_halpha_maps.png", dpi=130); plt.close(fig)

    # ---- fig7: 方位平均徑向剖面 + 偵測極限 ----
    edges = np.arange(0, r.max(), 1.0)                    # 每 1 arcsec 切一個環
    rc = 0.5 * (edges[:-1] + edges[1:])                  # 每個環的中心半徑
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"nosky (MUSE)": "tab:blue", "wsky + ZAP": "tab:red", "nosky + ZAP": "tab:orange"}
    print("\n===== CGM Hα 徑向剖面 (方位中位 SB) =====")
    print(f"  星系中心@({cy},{cx}), pixscale={PIXSCALE}\"/pix")
    for k, m in maps.items():
        # 每個環內取中位 SB = 方位平均剖面 (中位數對殘餘亮點穩健)
        prof = np.array([np.nanmedian(m[(r >= lo) & (r < hi)]) for lo, hi in zip(edges[:-1], edges[1:])])
        # 每個環內有多少有效 spaxel (環平均後噪音會 /sqrt(N) 下降)
        npix = np.array([int((np.isfinite(m) & (r >= lo) & (r < hi)).sum()) for lo, hi in zip(edges[:-1], edges[1:])])
        lim = sigma[k] / np.sqrt(np.maximum(npix, 1))     # 環平均後的 1σ 偵測極限
        ax.plot(rc, prof, color=colors[k], label=k)       # 剖面實線
        ax.plot(rc, 3 * lim, color=colors[k], ls=":", lw=1)  # 3σ 偵測極限 (虛線)
        # CGM 延展量化: 剖面仍高於 3σ 極限的最遠半徑
        det = prof > 3 * lim
        rext = rc[det].max() if det.any() else 0.0
        print(f"  {k:14s}: 3σ 可偵測 Hα 延展到 ~{rext:.1f}\"  (1σ/spaxel={sigma[k]:.1f})")
    ax.set_yscale("symlog", linthresh=1.0)               # symlog: 近零線性、之外對數
    ax.set_xlabel("radius [arcsec]"); ax.set_ylabel("Hα flux [1e-20 erg/s/cm2] / spaxel")
    ax.set_title("Azimuthally-averaged Hα radial profile")
    ax.legend(fontsize=8, ncol=1, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig7_cgm_radial_profile.png", dpi=130); plt.close(fig)
    print(f"  → 圖: {OUT}/fig6_cgm_halpha_maps.png, {OUT}/fig7_cgm_radial_profile.png")

if __name__ == "__main__":
    main()
