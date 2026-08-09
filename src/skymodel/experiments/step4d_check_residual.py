"""源區扣掉模板之後,還像不像天空?

step4d 的前提是「D − 源模型 = 天空」。這個前提只有在源模型夠準時才成立,
而 Haro 11 的擬合 reduced chi2 是 4429 —— 它一個源就佔了要新增的天空樣本的
94%,如果殘差夠大,送進去的不是天空而是「天空 + 一大坨源殘差」,會把 C_sky
和 basis 往「看起來像 Haro 11」的方向汙染。這支先量,再決定納不納入。

做法:對每個源的 spaxel 逐一擬合

    D(p,λ) = Σⱼ aⱼ(p)·Tⱼ(λ) + s(p)·C_sky(λ) + Σₖ cₖ(p)·Lₖ(λ)

扣掉源那一項,對該源的 spaxel 取平均,和 blank 的 mean sky 並排比較。平均時
用和 step3 完全相同的逐通道 sigma-clip(CLIP_SIGMA = 30,σ 由跨 spaxel 的
16-84 百分位定),這樣兩邊的統計處理一致,差異才是真的差異。

    conda run -n astro python src/skymodel/experiments/step4d_check_residual.py \\
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
from step3_sky_basis import CLIP_SIGMA
from templates import air_to_vacuum

ROOT    = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "results/skymodel/figures"


def clipped_mean(X):
    """逐通道 sigma-clip 之後取平均 —— 和 step3 算 mean_sky 的方式逐字相同。

    σ 用 (p84 − p16)/2 而不是標準差:標準差本身會被離群值撐大,用它定門檻等於
    讓離群值決定自己會不會被剪掉。
    """
    p16, med, p84 = np.nanpercentile(X, [16, 50, 84], axis=1)
    sg   = (p84 - p16) / 2
    keep = np.abs(X - med[:, None]) <= CLIP_SIGMA * sg[:, None]
    keep &= np.isfinite(X)
    n    = keep.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (np.nan_to_num(X) * keep).sum(axis=1, dtype=np.float64)
                        / np.maximum(n, 1), np.nan), n


def main():
    ap = argparse.ArgumentParser(description="源區扣掉模板後和 mean sky 的差異")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--s-fix", type=float, default=1.0)
    ap.add_argument("--best", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seg    = fits.getdata(STEP01 / "seg.fits")
    white  = fits.getdata(STEP01 / "whitelight.fits")
    wl_air = np.load(STEP03 / "wavelength.npy")
    wl_vac = air_to_vacuum(wl_air)
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

    blank = valid & (seg_f == 0)
    mean_sky, _ = clipped_mean(D[:, blank])
    print(f"mean sky 來自 {int(blank.sum()):,} 個 blank spaxel")

    templates = build_templates(best, wl_vac)
    ids = [int(i) for i in best["id"]]

    # 差異要相對於什麼才有意義?天空連續譜的量級。用線外通道的中位數當尺度。
    from utils import load_line_masks
    line = load_line_masks(STEP03 / "iter_line_mask.npy", cumulative=True)[0]
    scale = float(np.nanmedian(mean_sky[~line]))

    res = {}
    print(f"\n{'ID':>4}{'nspax':>7}{'線外中位差':>12}{'相對天空':>10}"
          f"{'線內中位差':>12}{'相對天空':>10}")
    print("-" * 56)
    for t in ids:
        m = valid & (seg_f == t)
        if not m.any():
            continue
        c   = fit_source(D[:, m], V[:, m], sky, templates.get(t), s_fix=args.s_fix)
        T   = templates[t]
        src = T @ np.nan_to_num(c[:T.shape[1]])       # (nz, n) 每個 spaxel 的源模型
        red, _ = clipped_mean(D[:, m] - src)          # 扣掉源之後的平均
        d = red - mean_sky
        res[t] = (red, d, int(m.sum()))
        o, i = float(np.nanmedian(d[~line])), float(np.nanmedian(d[line]))
        print(f"{t:>4}{int(m.sum()):>7}{o:>12.3f}{100*o/scale:>9.1f}%"
              f"{i:>12.3f}{100*i/scale:>9.1f}%")

    print(f"\n天空連續譜的量級(線外中位) = {scale:.3f}")

    # ---------------- 圖 ----------------
    n = len(res)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 3.0 * nrow),
                             squeeze=False, sharex=True)
    for k, t in enumerate(sorted(res)):
        ax = axes[k // ncol][k % ncol]
        red, d, ns = res[t]
        # 灰色是 mean sky 本身,當作「多大才算大」的參照;紅色是差異。
        ax.plot(wl_air, mean_sky, lw=0.4, color="0.75", label="mean sky (blank)")
        ax.plot(wl_air, d, lw=0.5, color="#d62728",
                label="source region − template  −  mean sky")
        ax.axhline(0, color="0.3", lw=0.6)
        hi = float(np.nanpercentile(np.abs(d), 99.5)) * 1.3
        ax.set_ylim(-hi, max(hi, np.nanpercentile(mean_sky, 99) * 1.05))
        ax.grid(alpha=0.25)
        ax.set_title(f"ID {t}   {ns} spaxel", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("flux [$10^{-20}$]", fontsize=7)
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    for a in axes[-1]:
        a.set_xlabel("wavelength [$\\AA$]", fontsize=8)
    fig.suptitle("source region after removing the fitted template, "
                 "compared with the mean sky   "
                 f"[{args.basis} K={args.K}, s = {args.s_fix}]", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out) if args.out else FIGURES / "step4d_source_minus_template.png"
    fig.savefig(out, dpi=135, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
