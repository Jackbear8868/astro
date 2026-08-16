"""亮線上剩下的殘差,是強度對不上還是形狀對不上?

要檢驗的假說是「線性的強度 basis 表達不了線的形狀變化」。

在每條亮線的窗口內,把該線的平均輪廓 P 及其導數當成基底:

    P     強度誤差        線被整體多扣或少扣
    P'    波長偏移        S(λ+δ) ≈ S(λ) + δ·S'(λ) —— 一階展開,長得像一正一負的雙峰
    P''   LSF 寬度變化    線變寬或變窄的一階效應
    常數  局部連續譜殘留

把每個 spaxel 在窗口內的殘差投影到這四個方向(先做 QR 正交化,各方向解釋的
變異數才可加),看各佔多少。

判讀:
    P   佔優   → 強度沒對上。但 basis 本來就能表達強度,這代表 K 不夠
    P'  佔優   → 波長偏移。注意這在數學上是線性的,basis 若學到 P' 就能處理,
                 所以 P' 殘留代表 basis 沒有把這個自由度學進去
    P'' 佔優   → LSF 寬度隨視場變化
    都不佔優   → 就是雜訊,沒東西可修

    conda run -n astro python src/skymodel/experiments/line_shape.py
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/ne_pointing/step01"
STEP03  = ROOT / "results/skymodel/ne_pointing/step03"
STEP05  = ROOT / "results/skymodel/ne_pointing/step05"
FIGURES = ROOT / "results/skymodel/figures"
WSKY    = ROOT / "data/Haro11_NEpointing_wsky.fits"
ESO     = ROOT / "data/Haro11_NEpointing_esonosky.fits"

HALF = 7          # 窗口半寬 (px);線 FWHM ~2.5 px,±7 蓋得到整條線與兩側底盤


def main():
    ap = argparse.ArgumentParser(description="亮線殘差的形狀分解")
    ap.add_argument("--nline", type=int, default=10)
    ap.add_argument("--tag", default="svd_K30_s_1.0")
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    wl    = np.load(STEP03 / "wavelength.npy")
    mean  = np.load(STEP03 / "mean_sky.npy")
    C     = np.load(STEP03 / "sky_continuum.npy")
    nz    = wl.size
    prof  = mean - C                        # 純天光線的平均輪廓

    S = fits.getdata(STEP05 / f"sky_subtracted_{args.tag}.fits").reshape(nz, -1)
    E = fits.getdata(ESO).reshape(nz, -1)
    okm = ((white != 0) & (seg == 0))
    ok = okm.ravel() & np.isfinite(S).all(0) & np.isfinite(E).all(0)
    idx = np.flatnonzero(ok)
    print(f"blank {idx.size:,} spaxel")

    # 最亮的 nline 條線:局部極大且比鄰居高
    cand = np.flatnonzero((prof[1:-1] > prof[:-2]) & (prof[1:-1] > prof[2:])) + 1
    cand = cand[(cand > HALF + 2) & (cand < nz - HALF - 2)]
    peaks, used = [], np.zeros(nz, bool)
    for c in cand[np.argsort(prof[cand])[::-1]]:
        if used[max(0, c - 3 * HALF):c + 3 * HALF].any():
            continue
        used[max(0, c - HALF):c + HALF + 1] = True
        peaks.append(int(c))
        if len(peaks) == args.nline:
            break

    print(f"\n窗口 ±{HALF} px,四個方向:常數(連續譜) / P(強度) / "
          f"P'(波長偏移) / P''(LSF 寬度)")
    print(f"{'wl (A)':>9}{'峰高':>9}{'':>3}"
          f"{'我們 殘差rms':>13}{'常數':>7}{'P':>7}{'P′':>7}{'P″':>7}{'剩餘':>7}"
          f"{'':>3}{'ESO 殘差rms':>12}{'P′':>7}{'剩餘':>7}")
    print("-" * 100)

    keep_map = None
    for c in peaks:
        sl = slice(c - HALF, c + HALF + 1)
        P = prof[sl]
        M = np.c_[np.ones(P.size), P, np.gradient(P), np.gradient(np.gradient(P))]
        Q = np.linalg.qr(M)[0]                       # 正交化 → 各方向的變異數可加

        out = []
        for X in (S, E):
            R = X[sl][:, idx].astype(np.float64)     # (win, n_spaxel)
            tot = (R ** 2).sum(axis=0)
            pr = Q.T @ R                             # (4, n_spaxel) 各方向的投影
            frac = (pr ** 2).sum(axis=1) / tot.sum()
            out.append((np.sqrt(tot.mean() / P.size), frac,
                        1 - frac.sum(), pr[2]))
        (r1, f1, rest1, c1), (r2, f2, rest2, _) = out
        print(f"{wl[c]:>9.1f}{P.max():>9.0f}{'':>3}{r1:>13.3f}"
              f"{100*f1[0]:>6.1f}%{100*f1[1]:>6.1f}%{100*f1[2]:>6.1f}%"
              f"{100*f1[3]:>6.1f}%{100*rest1:>6.1f}%{'':>3}"
              f"{r2:>12.3f}{100*f2[2]:>6.1f}%{100*rest2:>6.1f}%")
        if keep_map is None:
            keep_map = (wl[c], c1)

    # ---- 最亮那條線的 P' 係數空間分布:平滑 → 波長解的梯度 ----
    lam, coef = keep_map
    img = np.full(okm.size, np.nan)
    img[idx] = coef
    img = img.reshape(okm.shape)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    v = np.nanpercentile(np.abs(img), 98)
    im = ax.imshow(img, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_title(f"coefficient on the derivative P' at {lam:.1f} A\n"
                 "(smooth spatial pattern = wavelength-solution gradient)",
                 fontsize=11)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    out = FIGURES / f"line_shape_deriv_{args.tag}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")

    fin = np.isfinite(img)
    print(f"\n最亮線 {lam:.1f} A 的 P' 係數:中位 {np.nanmedian(img):+.3f}   "
          f"散布 {np.nanstd(img):.3f}")
    # 相鄰 spaxel 的相關:平滑的空間結構會讓它明顯大於 0
    a = img[:, :-1][fin[:, :-1] & fin[:, 1:]]
    b = img[:, 1:][fin[:, :-1] & fin[:, 1:]]
    print(f"   相鄰 spaxel 的相關係數 {np.corrcoef(a, b)[0,1]:+.3f}"
          f"   (接近 0 = 雜訊;明顯為正 = 有空間結構)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
