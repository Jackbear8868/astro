"""天空的變化,時間軸和空間軸哪個大?

我們的 basis 學的是「同一顆曝光內、跨 spaxel 的空間變化」。但天空真正在變的
是時間 —— OH 帶在十幾分鐘內會變好幾個百分點。data/NEpointing_exps/ 有四顆
間隔 13.2 分鐘的獨立曝光,從來沒有被任何程式讀過,正好可以量這件事。

三個量放在一起比:

    時間變化   四顆曝光各自的平均天空光譜之間的散布。每個平均是數萬個 spaxel
               的平均,雜訊被壓到 sigma/sqrt(N),所以這是幾乎純淨的天空變化訊號
    空間變化   單顆曝光內,跨 blank spaxel 的散布(扣掉光子雜訊之後)
    光子雜訊   STAT

判讀:
    時間 >> 空間  → 合成 cube 裡的空間變化,主要來自「不同 spaxel 平均到
                    不同的曝光子集」(dither + 90 度旋轉造成),那是一個
                    低維(~曝光數)的結構,不需要那麼大的 K
    時間 ~ 空間  → 天空真的有空間結構,現行做法合理

四顆曝光的空間網格不同(還有 90 度旋轉),但 basis 是純光譜的量 —— 不需要
空間對齊,只要各自取得 blank spaxel 即可。

    conda run -n astro python src/skymodel/experiments/exposure_variation.py
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[3]
STEP03  = ROOT / "results/skymodel/ne_pointing/step03"
FIGURES = ROOT / "results/skymodel/evaluation/subtraction_check"
EXPDIR  = ROOT / "data/NEpointing_exps"

CHUNK = 200


def exposure_stats(path, pct):
    """回傳 (blank 平均光譜, 跨 spaxel 標準差, 平均 STAT, blank spaxel 數)。

    blank 的界定用白光影像的百分位切 —— 各顆曝光的網格不同,沒有共用的 seg 圖,
    而這裡只需要「以天空為主」的樣本,不需要精確的源遮罩。
    """
    with fits.open(path, memmap=True) as h:
        nz = h["DATA"].header["NAXIS3"]
        d, v = h["DATA"].data, h["STAT"].data
        ny, nx = d.shape[1], d.shape[2]
        # 白光影像:分段累加,避免一次載入整個 cube
        acc = np.zeros((ny, nx)); cnt = np.zeros((ny, nx))
        for j in range(0, nz, CHUNK):
            x = np.asarray(d[j:j+CHUNK], np.float32)
            g = np.isfinite(x)
            acc += np.where(g, x, 0).sum(axis=0); cnt += g.sum(axis=0)
        white = np.where(cnt > 0.9 * nz, acc / np.maximum(cnt, 1), np.nan)
        thr = np.nanpercentile(white, pct)
        m = np.isfinite(white) & (white < thr)

        mean = np.empty(nz); sd = np.empty(nz); stat = np.empty(nz)
        for j in range(0, nz, CHUNK):
            x = np.asarray(d[j:j+CHUNK], np.float64)[:, m]
            w = np.asarray(v[j:j+CHUNK], np.float64)[:, m]
            ok = np.isfinite(x).all(axis=0) & np.isfinite(w).all(axis=0)
            x, w = x[:, ok], w[:, ok]
            mean[j:j+CHUNK] = x.mean(axis=1)
            sd[j:j+CHUNK]   = x.std(axis=1)
            stat[j:j+CHUNK] = w.mean(axis=1)
    return mean, sd, stat, int(m.sum())


def main():
    ap = argparse.ArgumentParser(description="曝光之間 vs 曝光之內的天空變化")
    ap.add_argument("--pct", type=float, default=50.0,
                    help="白光影像的百分位門檻,低於此者當作 blank")
    args = ap.parse_args()

    wl = np.load(STEP03 / "wavelength.npy")
    IN = np.load(STEP03 / "iter_line_mask.npy")[0]

    means, sds, stats = [], [], []
    for i in (1, 2, 3, 4):
        p = EXPDIR / f"Haro11_NEpointing_EXP{i}_wsky.fits"
        m, s, v, n = exposure_stats(p, args.pct)
        means.append(m); sds.append(s); stats.append(v)
        print(f"EXP{i}  blank {n:,} spaxel   平均天空(線內中位) {np.median(m[IN]):8.2f}"
              f"   線外 {np.median(m[~IN]):7.3f}", flush=True)

    M = np.array(means)                       # (4, nz)
    S = np.array(sds).mean(axis=0)            # 空間散布(四顆平均)
    V = np.array(stats).mean(axis=0)          # 光子雜訊變異數

    t_sd  = M.std(axis=0, ddof=1)             # 時間散布
    s_sig = np.sqrt(np.maximum(S ** 2 - V, 0))   # 空間散布扣掉光子雜訊
    n_mean = np.sqrt(V / 40000.)              # 平均光譜自身的雜訊(~4 萬 spaxel)

    print(f"\n{'':>10}{'時間散布':>11}{'空間變化(扣雜訊)':>18}{'平均譜的雜訊':>14}"
          f"{'時間/空間':>11}")
    print("-" * 66)
    for nm, m in [("線內", IN), ("線外", ~IN)]:
        print(f"{nm:>10}{np.median(t_sd[m]):>11.3f}{np.median(s_sig[m]):>18.3f}"
              f"{np.median(n_mean[m]):>14.4f}{np.median(t_sd[m]/np.maximum(s_sig[m],1e-9)):>11.2f}")

    print(f"\n相對變化(相對於平均天空):")
    lvl = M.mean(axis=0)
    for nm, m in [("線內", IN), ("線外", ~IN)]:
        print(f"{nm:>10}  時間 {100*np.median((t_sd/np.abs(lvl))[m]):.3f}%"
              f"   空間 {100*np.median((s_sig/np.abs(lvl))[m]):.3f}%")

    # 四顆曝光的天空張開幾維?對 (4, nz) 做 SVD 看奇異值衰減
    Mc = M - M.mean(axis=0)
    sv = np.linalg.svd(Mc, compute_uv=False)
    print(f"\n四顆曝光的平均天空(扣掉共同成分後)的奇異值:"
          f"{np.array2string(sv, precision=1)}")
    print(f"   前 1 個佔 {100*sv[0]**2/ (sv**2).sum():.1f}% 的變異數,"
          f"前 2 個佔 {100*(sv[:2]**2).sum()/(sv**2).sum():.1f}%")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for i in range(4):
        ax[0].plot(wl, M[i] - M.mean(axis=0), lw=0.6, label=f"EXP{i+1}")
    ax[0].set_ylabel("mean sky − 4-exposure mean")
    ax[0].set_title("sky variation between four 700 s exposures, 13.2 min apart",
                    fontsize=11)
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)
    ax[1].plot(wl, t_sd, lw=0.7, color="#d62728", label="temporal scatter (4 exposures)")
    ax[1].plot(wl, s_sig, lw=0.7, color="#1f77b4",
               label="spatial scatter within an exposure (noise removed)")
    ax[1].set_yscale("log"); ax[1].set_xlabel("wavelength (A)")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = FIGURES / "exposure_variation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
