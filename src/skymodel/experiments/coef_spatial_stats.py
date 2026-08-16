"""blank 區係數圖的空間統計:雜訊佔多少、真實結構有多長。

決定「空間正則化」值不值得做,以及 lambda 大概該在什麼量級。

量兩件事:

  雜訊下限   相鄰兩格看到的是同一片天空,它們的差幾乎純粹是求解雜訊。
             sigma_noise = scale(c(p+1) - c(p)) / sqrt(2)
             和該係數圖的總散布相比,就知道有多少比例可以被平滑壓掉。

  自相關     rho(d) = <dc(p)·dc(p+d)> / <dc(p)²>,dc 是扣掉整體中位數的殘差。
             rho 若在 d=1 就掉到 0 → 相鄰格無共同結構,散布純雜訊,平滑有效。
             rho 若拖長尾 → 有真實空間結構,平滑核不能超過那個尺度。

兩個方向分開量:x 沿 IFU slice 方向,y 跨 slice,兩者的結構可能不同。

    conda run -n astro python src/skymodel/experiments/coef_spatial_stats.py --basis svd
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
CUBE    = ROOT / "data/Haro11_NEpointing_wsky.fits"
FIGURES = ROOT / "results/skymodel/figures"

MIN_COVERAGE = 0.9
MAX_LAG      = 40          # 自相關算到幾格


def scale(a):
    """穩健散布:(p84 - p16)/2。係數有少數極端值,np.std 會被拉高。"""
    a = a[np.isfinite(a)]
    return float((np.percentile(a, 84) - np.percentile(a, 16)) / 2)


def noise_floor(img, mask, axis):
    """相鄰格差分估雜訊。兩格都必須在 mask 內才算。"""
    if axis == 0:
        d, m = img[1:] - img[:-1], mask[1:] & mask[:-1]
    else:
        d, m = img[:, 1:] - img[:, :-1], mask[:, 1:] & mask[:, :-1]
    return scale(d[m]) / np.sqrt(2.0)


def autocorr(img, mask, axis, max_lag):
    """rho(d),只用兩端都在 mask 內的配對。img 應已扣掉整體中位數。"""
    out = []
    var = np.nanmean(img[mask] ** 2)
    for d in range(1, max_lag + 1):
        if axis == 0:
            a, b, m = img[d:], img[:-d], mask[d:] & mask[:-d]
        else:
            a, b, m = img[:, d:], img[:, :-d], mask[:, d:] & mask[:, :-d]
        out.append(np.nanmean(a[m] * b[m]) / var if m.sum() > 100 else np.nan)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description="blank 係數圖的空間統計")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("--tag", default=None, help="step05 的 tag;預設 {basis}_s_free")
    args = ap.parse_args()
    tag = args.tag or f"{args.basis}_s_free"

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    sky   = np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                       np.load(STEP03 / f"sky_basis_{args.basis}.npy")])
    K = sky.shape[0]

    # 係數圖:s_map 只存了 s,其餘 K-1 條要重解。直接重跑 fit_blank 最省事。
    with fits.open(CUBE, memmap=True) as hdul:
        D = np.asarray(hdul["DATA"].data, np.float32)
    nz, ny, nx = D.shape
    D = D.reshape(nz, -1)
    cov   = np.isfinite(D).sum(axis=0) / nz
    valid = (white != 0).reshape(-1) & (cov >= MIN_COVERAGE)
    blank = valid & (seg.reshape(-1) == 0)
    Db    = D[:, blank].astype(np.float64)
    del D
    clean = np.isfinite(Db).all(axis=0)
    coef  = np.full((K, int(blank.sum())), np.nan)
    coef[:, clean] = np.linalg.pinv(sky.T) @ Db[:, clean]
    del Db

    maps = np.full((K, ny * nx), np.nan)
    maps[:, blank] = coef
    maps = maps.reshape(K, ny, nx)
    bmask = blank.reshape(ny, nx) & np.isfinite(maps[0])

    print(f"basis={args.basis}   blank {int(bmask.sum()):,} 個可用格子   K={K}\n")
    print(f"{'係數':>6}{'總散布':>11}{'雜訊(y向)':>12}{'雜訊(x向)':>12}"
          f"{'雜訊佔比':>11}{'可壓縮空間':>12}")
    print("-" * 66)
    frac = []
    for k in range(K):
        img = maps[k]
        tot = scale(img[bmask])
        ny_, nx_ = noise_floor(img, bmask, 0), noise_floor(img, bmask, 1)
        n = min(ny_, nx_)                       # 取較小者當雜訊估計,較保守
        f = (n / tot) ** 2 if tot > 0 else np.nan
        frac.append(f)
        name = "s" if k == 0 else f"c{k}"
        print(f"{name:>6}{tot:>11.5f}{ny_:>12.5f}{nx_:>12.5f}"
              f"{100*f:>10.1f}%{'  可壓 ' + f'{100*(1-np.sqrt(max(1-f,0))):.0f}%' if f==f else '':>12}")
    print(f"\n雜訊佔比 = (雜訊/總散布)²;越接近 100% 代表散布幾乎全是求解雜訊,平滑越有效。")
    print(f"11 條的雜訊佔比中位數 {100*np.nanmedian(frac):.1f}%")

    # ---------------- 自相關 ----------------
    lags = np.arange(1, MAX_LAG + 1)
    print(f"\n自相關 rho(d)(扣掉整體中位數之後):")
    print(f"{'係數':>6}" + "".join(f"{f'd={d}':>9}" for d in [1, 2, 3, 5, 10, 20, 40]))
    print("-" * 69)
    rhos = {}
    for k in [0, 1, 2, 5, 10]:
        img = maps[k] - np.nanmedian(maps[k][bmask])
        ry = autocorr(img, bmask, 0, MAX_LAG)
        rx = autocorr(img, bmask, 1, MAX_LAG)
        rhos[k] = (ry, rx)
        name = "s" if k == 0 else f"c{k}"
        print(f"{name:>6}" + "".join(f"{np.nanmean([ry[d-1], rx[d-1]]):>9.3f}"
                                     for d in [1, 2, 3, 5, 10, 20, 40]))
    print("\nrho(1) 接近 0 → 相鄰格無共同結構,散布是雜訊 → 平滑有效")
    print("rho(d) 拖長尾  → 有真實空間結構,平滑核不可超過該尺度")

    # ---------------- 圖 ----------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for k, (ry, rx) in rhos.items():
        name = "s" if k == 0 else f"c{k}"
        ax[0].plot(lags, ry, label=name, lw=1.2)
        ax[1].plot(lags, rx, label=name, lw=1.2)
    for a, t in zip(ax, ["y direction (across slices)", "x direction (along slices)"]):
        a.axhline(0, color="0.6", lw=0.6)
        a.set_xlabel("lag [px]"); a.set_ylabel(r"$\rho(d)$")
        a.set_title(t); a.legend(fontsize=8); a.grid(alpha=0.25)
    fig.suptitle(f"blank coefficient maps, spatial autocorrelation  (basis={args.basis})",
                 fontsize=10)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / f"coef_autocorr_{tag}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
