"""天空扣得乾不乾淨 —— 0709 那套「sky status」比較,套在 step4d 前後的天空模型上。

量的是**扣完之後剩下什麼**,不是模型本身變了多少:

    blank 區   residual = D − 天空模型
    源區       residual = D − 天空模型 − 源模型

兩邊的理想值都是 0,所以同一組統計可以一致地套用,源區也成為公平的檢驗。
源區必須連源模型一起扣 —— 只扣天空的話源還在,「殘差接近 0」就沒有意義。

沿用 0709 的摘要統計(utils.spectrum_stats):

    mean            理想 0     扣過頭是負的,扣不夠是正的
    sigma           越小越好   殘餘天光線會把它撐大
    skewness        理想 0     天光線殘留是單邊的,會讓它偏正
    kurtosis                   少數大殘差會讓它爆高
    rms_from_zero              把 mean 和 sigma 併成一個數

不需要跑 step5:扣天空只要逐 spaxel 解一次線性最小平方,不必產出整個 cube。
blank 區更有捷徑 —— 不加權時設計矩陣對所有 spaxel 相同,一次 pinv 解完。

    conda run -n astro python src/skymodel/experiments/step4d_sky_status.py \\
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
from step5_fit_spaxels import (build_templates, fit_blank, fit_source,
                               CUBE, STEP01, STEP03)
from templates import air_to_vacuum
from utils import spectrum_stats

ROOT    = Path(__file__).resolve().parents[3]
STEP04D = ROOT / "results/skymodel/step04d"
FIGURES = ROOT / "results/skymodel/figures"
ESONOSKY = ROOT / "data/Haro11_NEpointing_esonosky.fits"

BANDS = {"5000-9000": (5000, 9000), "5000-7000": (5000, 7000),
         "7000-9000": (7000, 9000)}
COL = {"step3": "#7f7f7f", "step4d": "#d62728", "ESO": "#1f77b4"}


def band_stats(wl, spec):
    """三個波段各自的摘要統計。分段是因為紅端天光線密,兩邊的行為不同。"""
    out = {}
    for nm, (lo, hi) in BANDS.items():
        m = (wl >= lo) & (wl < hi) & np.isfinite(spec)
        out[nm] = spectrum_stats(spec[m])
    return out


def main():
    ap = argparse.ArgumentParser(description="step4d 前後的 sky status 比較")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--best", required=True)
    ap.add_argument("--s-fix", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    wl    = np.load(STEP03 / "wavelength.npy")
    seg   = fits.getdata(STEP01 / "seg.fits").reshape(-1)
    white = fits.getdata(STEP01 / "whitelight.fits").reshape(-1)
    skies = {
        "step3":  np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                             np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")]),
        "step4d": np.vstack([np.load(STEP04D / "sky_continuum.npy"),
                             np.load(STEP04D / f"sky_basis_{args.basis}_K{args.K}.npy")]),
    }
    best = np.load(args.best)

    with fits.open(CUBE, memmap=True) as h:
        D = np.asarray(h["DATA"].data, np.float32)
        V = np.asarray(h["STAT"].data, np.float32)
    nz = D.shape[0]
    D, V = D.reshape(nz, -1), V.reshape(nz, -1)
    valid = (white != 0) & np.isfinite(D).all(axis=0)

    fit_ids = [int(i) for i in best["id"]]
    regions = {"blank": valid & (seg == 0),
               "Haro 11 (#1)": valid & (seg == 1),
               "other sources": valid & np.isin(seg, [i for i in fit_ids if i != 1])}

    T_all = build_templates(best, air_to_vacuum(wl))
    resid = {}                                   # (region, model) -> 殘差光譜

    for rname, mask in regions.items():
        idx = np.flatnonzero(mask)
        print(f"{rname}: {idx.size:,} spaxel", flush=True)
        for mname, sky in skies.items():
            if rname == "blank":
                c = fit_blank(D[:, idx], sky)
                r = D[:, idx] - sky.T @ c
            else:
                # 源區:天空與源一起解,兩者都扣掉,理想值才會是 0
                parts = []
                for t in (fit_ids if rname != "Haro 11 (#1)" else [1]):
                    if rname == "other sources" and t == 1:
                        continue
                    mm = np.flatnonzero(valid & (seg == t))
                    if mm.size == 0:
                        continue
                    T = T_all[t]
                    co = fit_source(D[:, mm], V[:, mm], sky, T, s_fix=args.s_fix)
                    src = T @ np.nan_to_num(co[:T.shape[1]])
                    src = np.where(np.all(np.isfinite(T), axis=1)[:, None],
                                   src, np.nan)
                    model = (sky[0][:, None] * co[4][None, :]
                             + sky[1:].T @ co[5:] + src)
                    parts.append(D[:, mm] - model)
                r = np.hstack(parts)
            with np.errstate(invalid="ignore"):
                resid[(rname, mname)] = np.nanmean(r, axis=1)

    # ESO 的 nosky 只有 blank 可比 —— 它沒有我們的源模型
    with fits.open(ESONOSKY, memmap=True) as h:
        E = np.asarray(h["DATA"].data, np.float32).reshape(nz, -1)
    with np.errstate(invalid="ignore"):
        resid[("blank", "ESO")] = np.nanmean(E[:, np.flatnonzero(regions["blank"])],
                                             axis=1)

    # ---------------- 表 ----------------
    print()
    for rname in regions:
        print(f"=== {rname} ===")
        print(f"{'model':>8}{'band':>12}" +
              "".join(f"{k:>16}" for k in
                      ("mean", "sigma", "skewness", "kurtosis", "rms_from_zero")))
        for mname in ("step3", "step4d", "ESO"):
            if (rname, mname) not in resid:
                continue
            st = band_stats(wl, resid[(rname, mname)])
            for b, s in st.items():
                print(f"{mname:>8}{b:>12}" +
                      "".join(f"{s[k]:>16.4f}" for k in
                              ("mean", "sigma", "skewness", "kurtosis",
                               "rms_from_zero")))
        print()

    # ---------------- 圖 ----------------
    nrow = len(regions)
    fig, axes = plt.subplots(nrow, 2, figsize=(17, 3.6 * nrow), squeeze=False,
                             gridspec_kw={"width_ratios": [5, 1.6]})
    for k, rname in enumerate(regions):
        ax, sax = axes[k]
        ax.axhline(0, color="0.5", lw=0.6)
        curves = [(m, resid[(rname, m)]) for m in ("step3", "step4d", "ESO")
                  if (rname, m) in resid]
        for m, r in curves:
            ax.plot(wl, r, lw=0.7, color=COL[m], label=m, alpha=0.85)
        v = np.concatenate([np.abs(r[np.isfinite(r)]) for _, r in curves])
        hi = float(np.percentile(v, 99.5)) * 1.4
        ax.set_ylim(-hi, hi)
        ax.set_ylabel("residual [$10^{-20}$]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title(f"{rname}   ({int(regions[rname].sum()):,} spaxel)",
                     fontsize=10)

        sax.axis("off")
        y = 0.98
        for m, r in curves:
            s = band_stats(wl, r)["5000-9000"]
            sax.text(0, y, f"[{m}]  5000-9000\n" + "\n".join(
                f"{kk:<14}= {vv:+.4f}" for kk, vv in s.items()),
                color=COL[m], va="top", family="monospace", fontsize=7.5,
                transform=sax.transAxes)
            y -= 0.34
    axes[-1][0].set_xlabel("wavelength [$\\AA$]")
    fig.suptitle("sky status: what is left after subtracting the sky "
                 "(and, in source regions, the source)   "
                 f"[{args.basis} K={args.K}]", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = Path(args.out) if args.out else FIGURES / "step4d_sky_status.png"
    fig.savefig(out, dpi=135, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
