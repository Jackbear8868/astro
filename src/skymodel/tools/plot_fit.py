"""畫出某個源的最佳擬合:資料、模型、源/天空分解、殘差。

回答的問題:
  - 模板在哪裡不合(連續譜斜率?發射線強度?)
  - 殘差是否集中在天光線位置(天空 basis 夠不夠)
  - 需不需要教授說的 PCA 補正,以及補正該長什麼樣

用法:  python src/skymodel/step4_plot_fit.py --id 1 --basis svd
"""
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# 搬到子目錄之後,同層的 templates / utils 不再自動可見。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from templates import load_sdss_template, redshift_to_grid, air_to_vacuum
from utils import fit_chi2_coefficients

ROOT    = Path(__file__).resolve().parents[3]
STEP02  = ROOT / "results/skymodel/step02"
STEP03  = ROOT / "results/skymodel/step03"
STEP04  = ROOT / "results/skymodel/step04"
TPL_DIR = ROOT / "data/sdss_templates"

# 常見的光學譜線(空氣波長,Å)。Haro 11 是星爆星系,這些應該很強。
LINES = {
    "Hb":       4861.3,
    "[OIII]a":  4958.9,
    "[OIII]b":  5006.8,
    "[OI]":     6300.3,
    "Ha":       6562.8,
    "[NII]":    6583.5,
    "[SII]a":   6716.4,
    "[SII]b":   6730.8,
}


def main():
    ap = argparse.ArgumentParser(description="畫某個源的最佳擬合與殘差")
    ap.add_argument("--id",    type=int, default=1)
    ap.add_argument("--basis", default="svd")
    args = ap.parse_args()

    # --- 最佳解(step4 掃描的第一名) ---
    d = np.load(STEP04 / f"scan_id{args.id}_{args.basis}.npz")
    tpl_name, z_best = str(d["template"][0]), float(d["z"][0])
    print(f"best: template {tpl_name}   z = {z_best:.5f}")

    # --- 資料 ---
    ids   = np.load(STEP02 / "object_ids.npy")
    k     = int(np.flatnonzero(ids == args.id)[0])
    flux  = np.load(STEP02 / "object_flux.npy")[k]
    var   = np.load(STEP02 / "object_var.npy")[k]
    nspax = np.load(STEP02 / "object_nspax.npy")[k]
    with np.errstate(invalid="ignore", divide="ignore"):
        mflux = flux / nspax
        mvar  = var / nspax ** 2

    # --- 天空成分 ---
    wl_air    = np.load(STEP03 / "wavelength.npy")
    wl_vac    = air_to_vacuum(wl_air)
    C_sky     = np.load(STEP03 / "sky_continuum.npy")
    B         = np.load(STEP03 / f"sky_basis_{args.basis}.npy")
    line_mask = np.load(STEP03 / "line_mask.npy")
    sky       = np.vstack([C_sky, B])

    # --- 重解一次,拿完整的係數 ---
    spline = load_sdss_template(TPL_DIR / f"spDR2-{tpl_name}.fit")
    tpl    = redshift_to_grid(spline, z_best, wl_vac)
    design = np.vstack([tpl, sky])
    coeff  = fit_chi2_coefficients(mflux, mvar, design)

    A, s, c = coeff[0], coeff[1], coeff[2:]
    print(f"A = {A:.4g}   s = {s:.4f}   line coeff = {np.round(c, 3)}")

    src   = A * tpl
    skym  = s * C_sky + c @ B
    model = src + skym
    resid = mflux - model

    good = (np.isfinite(mflux) & np.isfinite(mvar) & (mvar > 0)
            & np.all(np.isfinite(design), axis=0))
    lm, fm = good & line_mask, good & ~line_mask
    print(f"residual rms:  all {np.sqrt(np.mean(resid[good]**2)):.3f}   "
          f"line {np.sqrt(np.mean(resid[lm]**2)):.3f}   "
          f"line-free {np.sqrt(np.mean(resid[fm]**2)):.3f}")
    print(f"源佔總流量:    {np.nansum(src[good]) / np.nansum(mflux[good]) * 100:.1f}%")

    # 殘差最大的 8 個位置
    r = np.where(good, np.abs(resid), 0)
    print("\n殘差最大的波長:")
    for i in np.argsort(r)[::-1][:8]:
        near = min(LINES, key=lambda n: abs(LINES[n] * (1 + z_best) - wl_air[i]))
        off  = wl_air[i] - LINES[near] * (1 + z_best)
        tag  = f"  <- {near} 偏 {off:+.1f}A" if abs(off) < 15 else ""
        print(f"  {wl_air[i]:8.1f} A   resid {resid[i]:+9.2f}   "
              f"{'線' if line_mask[i] else '無線'}{tag}")

    # ---------------- 圖 ----------------
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)

    a = axes[0]
    a.plot(wl_air, mflux, lw=0.5, color="0.3", label="data (summed / nspax)")
    a.plot(wl_air, model, lw=0.5, color="#d62728", label="model")
    a.set_ylim(0, np.nanpercentile(mflux[good], 99.5))
    a.set_ylabel("flux"); a.legend(fontsize=8)
    a.set_title(f"ID {args.id}   template {tpl_name}   z = {z_best:.5f}   basis {args.basis}")

    a = axes[1]
    a.plot(wl_air, skym, lw=0.5, color="#1f77b4", label=f"sky  (s={s:.3f})")
    a.plot(wl_air, src,  lw=0.7, color="#2ca02c", label=f"source  (A={A:.3g})")
    a.set_ylim(0, np.nanpercentile(mflux[good], 99.5))
    a.set_ylabel("flux"); a.legend(fontsize=8)

    a = axes[2]
    a.fill_between(wl_air, -1e4, 1e4, where=line_mask, color="orange", alpha=0.12,
                   label="sky-line channels")
    a.plot(wl_air, resid, lw=0.5, color="k")
    a.axhline(0, color="0.5", lw=0.5)
    lim = np.nanpercentile(np.abs(resid[good]), 99) * 1.6
    a.set_ylim(-lim, lim)
    for name, lam0 in LINES.items():
        x = lam0 * (1 + z_best)
        if wl_air[0] < x < wl_air[-1]:
            a.axvline(x, color="#2ca02c", ls=":", lw=0.8)
            a.text(x, lim * 0.85, name, fontsize=7, rotation=90,
                   color="#2ca02c", ha="right", va="top")
    a.set_xlabel("observed wavelength (air) [A]"); a.set_ylabel("residual")
    a.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out = STEP04 / f"fit_id{args.id}_{args.basis}.png"
    fig.savefig(out, dpi=140)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
