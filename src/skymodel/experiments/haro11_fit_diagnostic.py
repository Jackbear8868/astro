"""Haro 11:兩種比較方式選出的模型,以及它們的逐通道 chi2。

改前  固定的共同通道集合,比 total chi2          → 全部候選用同一批通道
改後  各候選用自己覆蓋到的通道,比 reduced chi2   → 教授指定

搜尋(改後)完全依教授指定:z ∈ [0, 1.5] 步長 1e-4、全部 33 條模板、
唯一下限是數學上必需的 n > p。改前的解以 REF 給定,是舊方法在 z ≤ 0.06
的共同通道上得到的最小 total chi2。

    conda run -n astro python src/skymodel/experiments/haro11_fit_diagnostic.py

掃描結果會快取;只調圖時不必重跑。要重算就刪掉 CACHE 檔。
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from templates import load_sdss_template, redshift_to_grid, air_to_vacuum

ROOT    = Path(__file__).resolve().parents[3]
STEP02  = ROOT / "results/skymodel/step02"
STEP03  = ROOT / "results/skymodel/step03"
TPL_DIR = ROOT / "data/sdss_templates"
OBJ_ID = 1
BASIS  = "seminmf"

# 檔名與快取都帶 basis 與判準,不同組合並存不互相覆蓋。
FIGURES = ROOT / "results/skymodel/figures"
FIGURES.mkdir(parents=True, exist_ok=True)
OUT   = {k: (FIGURES / f"haro11_spectrum_{k}_{BASIS}.png",
             FIGURES / f"haro11_chi2_{k}_{BASIS}.png",
             FIGURES / f"haro11_residual_{k}_{BASIS}.png") for k in ("A", "B")}
CACHE = Path(__file__).parent / f"haro11_fit_diagnostic.cache_{BASIS}.npz"

N_TPL  = 33
Z_GRID = np.arange(0.0, 1.5 + 5e-5, 1e-4)      # 教授指定
REF    = ("027", 0.0205)                        # 改前的解

C_DATA, C_BEFORE, C_AFTER = "#6b7280", "#2563eb", "#ea580c"


# ---------------------------------------------------------------- 載入
wl_air = np.load(STEP03 / "wavelength.npy")
wl_vac = air_to_vacuum(wl_air)
sky    = np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                    np.load(STEP03 / f"sky_basis_{BASIS}.npy")])
ids    = np.load(STEP02 / "object_ids.npy")
flux   = np.load(STEP02 / "object_flux.npy")
var    = np.load(STEP02 / "object_var.npy")
nspax  = np.load(STEP02 / "object_nspax.npy")

k = int(np.flatnonzero(ids == OBJ_ID)[0])
with np.errstate(invalid="ignore", divide="ignore"):
    f = flux[k] / nspax[k]
    v = var[k] / nspax[k] ** 2
p       = sky.shape[0] + 1
base    = np.isfinite(f) & np.isfinite(v) & (v > 0) & np.all(np.isfinite(sky), axis=0)
sig_all = np.sqrt(np.where(v > 0, v, 1.0))

tpls = {f"{i:03d}": load_sdss_template(TPL_DIR / f"spDR2-{i:03d}.fit") for i in range(N_TPL)}


def fit(name, z):
    """在該候選自己覆蓋到的通道上求解,回傳模型與逐通道 chi2。"""
    T = redshift_to_grid(tpls[name], z, wl_vac)
    g = base & np.isfinite(T)
    n = int(g.sum())
    if n <= p:
        return None
    sig = sig_all[g]
    M   = np.vstack([T[g], sky[:, g]]).T / sig[:, None]
    y   = f[g] / sig
    th, res, rank, _ = np.linalg.lstsq(M, y, rcond=None)
    model = (M @ th) * sig
    per   = ((f[g] - model) / sig) ** 2

    # 擬合只用 g 的通道(教授指定);這裡另外把解出的係數套到全部波長,
    # 看這個解在被 discount 掉的區域預測成什麼樣。模板蓋不到處源項為零。
    m_all   = th[0] * np.nan_to_num(T, nan=0.0) + th[1:] @ sky
    per_all = ((f - m_all) / sig_all) ** 2

    return dict(name=name, z=z, n=n, g=g, model=model, sig=sig, theta=th,
                per=per, chi2=per.sum(), red=per.sum() / (n - p),
                model_all=m_all, per_all=per_all,
                chi2_all=float(per_all[base].sum()))


# ---------------------------------------------------------------- 掃描(改後)
if CACHE.exists():
    c = np.load(CACHE)
    win = {k: (str(c[f"{k}_name"]), float(c[f"{k}_z"])) for k in ("A", "B")}
    print(f"讀取快取 {CACHE.name}")
else:
    print(f"掃描 {N_TPL} 模板 x {Z_GRID.size} z = {N_TPL*Z_GRID.size:,} 個候選 ...")
    win, bestv = {}, {"A": np.inf, "B": np.inf}
    for z in Z_GRID:
        for nm in tpls:
            r = fit(nm, z)
            if r is None:
                continue
            if r["red"] < bestv["A"]:                 # A:只算自己覆蓋的通道
                bestv["A"], win["A"] = r["red"], (nm, z)
            if r["chi2_all"] < bestv["B"]:            # B:套到全部通道再算
                bestv["B"], win["B"] = r["chi2_all"], (nm, z)
    np.savez(CACHE, **{f"{k}_name": win[k][0] for k in win},
                    **{f"{k}_z": win[k][1] for k in win})

sol = {k: fit(*win[k]) for k in ("A", "B")}
ref = fit(*REF)

print(f"\n{'判準':<32}{'tpl':>5}{'z':>10}{'n':>7}{'reduced chi2':>16}{'chi2 overall':>18}")
print("-" * 88)
for tag, r in (("A  min reduced chi2 (自己的通道)", sol["A"]),
               ("B  min total chi2 (全部通道)", sol["B"]),
               ("   文獻對照 tpl 027 z=0.0205", ref)):
    print(f"{tag:<32}{r['name']:>5}{r['z']:>10.4f}{r['n']:>7}"
          f"{r['red']:>16.4g}{r['chi2_all']:>18,.0f}")


# ---------------------------------------------------------------- 圖
lam        = wl_air[base]
per_raw    = (f[base] / sig_all[base]) ** 2        # 完全不擬合:模型 = 0
chi2_raw   = float(per_raw.sum())
print(f"\n完全不擬合 (模型 = 0):  chi2 = {chi2_raw:,.0f}  over {int(base.sum())} channels")

def make_figures(r, out_spec, out_chi2, out_resid):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(lam, f[base], lw=0.6, color=C_DATA, label="data")
    ax.plot(wl_air[r["g"]], r["model"], lw=1.4, color=C_AFTER,
            label=f"fitted template: tpl {r['name']}, z = {r['z']:.4f},  "
                  f"{r['n']} channels")
    ax.set_ylabel("flux")
    ax.set_xlabel("observed wavelength [Å]  (air)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Haro 11 spectrum", fontsize=12)
    ax.grid(alpha=.25, lw=.5)
    fig.savefig(out_spec, dpi=140, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(13, 5))
    lam_fit = wl_air[r["g"]]
    h_span  = ax.axvspan(lam_fit.min(), lam_fit.max(), color=C_AFTER, alpha=.30, lw=0, zorder=1)
    h_mod,  = ax.plot(lam, r["per_all"][base], lw=0.7, color=C_AFTER, zorder=2)
    h_raw,  = ax.plot(lam, per_raw, lw=0.7, color=C_DATA, zorder=3)
    ax.legend([h_raw, h_mod, h_span],
              [fr"$\chi^2$ = {chi2_raw:,.0f}",
               fr"model fitted: tpl {r['name']}, z = {r['z']:.4f}  —  "
               fr"reduced $\chi^2$ = {r['red']:.4g},  "
               fr"$\chi^2$ overall = {r['chi2_all']:,.0f}",
               f"wavelengths the template covers  "
               f"({lam_fit.min():.0f}–{lam_fit.max():.0f} Å, "
               f"{r['n']} of {int(base.sum())} channels)"],
              fontsize=9, loc="upper left", framealpha=0.92, edgecolor="none")
    ax.set_ylabel(r"per-channel $\chi^2$")
    ax.set_xlabel("observed wavelength [Å]  (air)")
    ax.set_ylim(0, per_raw.max() * 1.05)
    ax.set_title(r"Haro 11 per-channel $\chi^2$", fontsize=12)
    ax.grid(alpha=.25, lw=.5)
    fig.savefig(out_chi2, dpi=140, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(13, 5))
    lam_fit = wl_air[r["g"]]
    ax.axvspan(lam_fit.min(), lam_fit.max(), color=C_AFTER, alpha=.18, lw=0, zorder=1)
    ax.axhline(0, lw=0.9, color=C_DATA, zorder=2)
    ax.plot(lam, (f - r["model_all"])[base], lw=0.7, color=C_AFTER, zorder=3,
            label=f"data − model:  tpl {r['name']}, z = {r['z']:.4f},  "
                  f"fitted on {r['n']} of {int(base.sum())} channels")
    ax.set_ylabel("residual  (data − model)")
    ax.set_xlabel("observed wavelength [Å]  (air)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Haro 11 residual", fontsize=12)
    ax.grid(alpha=.25, lw=.5)
    fig.savefig(out_resid, dpi=140, bbox_inches="tight")
    print(f"saved -> {out_spec.name}, {out_chi2.name}, {out_resid.name}")


for key in ("A", "B"):
    make_figures(sol[key], *OUT[key])
