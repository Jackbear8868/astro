"""扣掉源模型之後,源區的 spaxel 還算不算「天空」?—— 用 blank 自己定門檻。

判準必須是**形狀不合**,不是**強度不合**。天空在源區可以整體亮一點或暗一點,
那正是 s(p) 這個自由參數在描述的東西;不能不同的是形狀。所以指標直接取
「允許 s 和 cₖ 自由調整之後,還剩多少殘差」—— 也就是逐 spaxel 的 reduced chi2。

    blank spaxel     D = s·C_sky + Σ cₖ·Lₖ                  dof = n − K
    source spaxel    D = Σ aⱼ·Tⱼ + s·C_sky + Σ cₖ·Lₖ        dof = n − K − n_comp

兩邊都做了自由度修正,可以直接比。門檻不自己拍:blank 的分布就是「一個真的
天空 spaxel 長什麼樣」,落在它之外的源區 spaxel 才剔除。

blank 有 84,052 個,逐一解太慢也沒必要 —— 抽樣幾千個就足以定出分布。

    conda run -n astro python src/skymodel/experiments/step4d_spaxel_quality.py \\
        --basis svd -K 30 \\
        --best results/skymodel/step04b/classification_nobasis_s0.0_4700-8000_4700-8000_L1cum__eso.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step5_fit_spaxels import (build_templates, fit_source, N_SRC,
                               MIN_COVERAGE, CUBE, STEP01, STEP03)
from templates import air_to_vacuum

ROOT    = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "results/skymodel/figures"


def red_chi2(D, V, sky, T, coef):
    """逐 spaxel 的 reduced chi2。模型由 fit_source 的係數重建。

    係數的排列是固定的 (a₁…a₄, s, c₁…c_{K−1}),和 T 的欄數無關 —— 恆星只有
    a₁ 有值,沒有模板的區域四個 a 全是 NaN,所以重建時要用 nan_to_num。
    """
    K      = sky.shape[0]
    n_comp = 0 if T is None else T.shape[1]
    a      = np.nan_to_num(coef[:N_SRC])
    model  = (sky[0][:, None] * coef[N_SRC][None, :]
              + sky[1:].T @ coef[N_SRC + 1:])
    if T is not None:
        model = model + T @ a[:n_comp]
    good = np.isfinite(D) & np.isfinite(V) & (V > 0) & np.isfinite(model)
    with np.errstate(invalid="ignore", divide="ignore"):
        c2 = np.where(good, (D - model) ** 2 / V, 0.0).sum(axis=0)
    dof = good.sum(axis=0) - (K + n_comp)
    return np.where(dof > 0, c2 / np.maximum(dof, 1), np.nan)


def main():
    ap = argparse.ArgumentParser(description="源區 spaxel 扣源之後的品質")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--s-fix", type=float, default=None,
                    help="預設 None = s 自由。品管要量的是形狀,強度必須放行")
    ap.add_argument("--best", required=True)
    ap.add_argument("--n-blank", type=int, default=5000,
                    help="抽樣幾個 blank spaxel 來定分布")
    ap.add_argument("--pct", type=float, default=99.5,
                    help="門檻取 blank 分布的第幾百分位")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seg    = fits.getdata(STEP01 / "seg.fits")
    white  = fits.getdata(STEP01 / "whitelight.fits")
    wl_vac = air_to_vacuum(np.load(STEP03 / "wavelength.npy"))
    sky    = np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                        np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")])
    best   = np.load(args.best)

    with fits.open(CUBE, memmap=True) as h:
        D = np.asarray(h["DATA"].data, np.float32)
        V = np.asarray(h["STAT"].data, np.float32)
    nz, ny, nx = D.shape
    D, V  = D.reshape(nz, -1), V.reshape(nz, -1)
    seg_f = seg.reshape(-1)
    valid = (white != 0).reshape(-1) & (np.isfinite(D).sum(axis=0) / nz >= MIN_COVERAGE)

    rng   = np.random.default_rng(0)
    bidx  = np.flatnonzero(valid & (seg_f == 0))
    samp  = rng.choice(bidx, size=min(args.n_blank, bidx.size), replace=False)
    print(f"blank {bidx.size:,} 個,抽樣 {samp.size:,} 個定分布", flush=True)
    cb    = fit_source(D[:, samp], V[:, samp], sky, None, s_fix=args.s_fix)
    r_bl  = red_chi2(D[:, samp], V[:, samp], sky, None, cb)
    r_bl  = r_bl[np.isfinite(r_bl)]
    thr   = float(np.percentile(r_bl, args.pct))
    print(f"blank 的 reduced chi2:中位 {np.median(r_bl):.3f}   "
          f"{args.pct} 百分位 {thr:.3f}   最大 {r_bl.max():.3f}")

    templates = build_templates(best, wl_vac)
    res = {}
    print(f"\n{'ID':>4}{'nspax':>7}{'中位':>9}{'90%':>9}{'最大':>10}"
          f"{'超過門檻':>12}")
    print("-" * 51)
    for t in [int(i) for i in best["id"]]:
        m = valid & (seg_f == t)
        if not m.any():
            continue
        c = fit_source(D[:, m], V[:, m], sky, templates.get(t), s_fix=args.s_fix)
        r = red_chi2(D[:, m], V[:, m], sky, templates.get(t), c)
        r = r[np.isfinite(r)]
        res[t] = r
        bad = int((r > thr).sum())
        print(f"{t:>4}{r.size:>7}{np.median(r):>9.3f}{np.percentile(r, 90):>9.3f}"
              f"{r.max():>10.3f}{bad:>7} ({100*bad/r.size:>4.1f}%)")

    tot = sum(v.size for v in res.values())
    bad = sum(int((v > thr).sum()) for v in res.values())
    print(f"\n合計 {tot:,} 個源區 spaxel,超過門檻的 {bad:,} 個 "
          f"({100*bad/tot:.1f}%)")

    # ---------------- 圖 ----------------
    fig, ax = plt.subplots(1, 2, figsize=(15, 5),
                           gridspec_kw=dict(width_ratios=[1.5, 1]))
    bins = np.linspace(0, max(thr * 2.5, np.percentile(r_bl, 99.9) * 1.5), 120)
    ax[0].hist(r_bl, bins=bins, density=True, color="0.6",
               label=f"blank ({r_bl.size:,} sampled)")
    for t, r in sorted(res.items()):
        ax[0].hist(r, bins=bins, density=True, histtype="step", lw=1.2,
                   label=f"ID {t} ({r.size})")
    ax[0].axvline(thr, color="k", ls="--", lw=1.2,
                  label=f"blank {args.pct}% = {thr:.2f}")
    ax[0].set_xlabel("reduced chi2 per spaxel")
    ax[0].set_ylabel("density")
    ax[0].legend(fontsize=8, ncol=2)
    ax[0].grid(alpha=0.25)

    # 右:每個源的分布用箱線圖並排,和 blank 的門檻比高低
    ids = sorted(res)
    ax[1].boxplot([res[t] for t in ids], tick_labels=[str(t) for t in ids],
                  showfliers=False, widths=0.6)
    ax[1].axhline(thr, color="k", ls="--", lw=1.2)
    ax[1].axhline(float(np.median(r_bl)), color="0.5", ls=":", lw=1.2)
    ax[1].set_xlabel("source ID")
    ax[1].set_ylabel("reduced chi2 per spaxel")
    ax[1].grid(alpha=0.25)

    fig.suptitle("per-spaxel sky-model quality after removing the source "
                 f"template   [{args.basis} K={args.K}, s free]\n"
                 "dashed = blank "
                 f"{args.pct}th percentile,  dotted = blank median", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = Path(args.out) if args.out else FIGURES / "step4d_spaxel_quality.png"
    fig.savefig(out, dpi=135, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
