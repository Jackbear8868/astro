"""sigma-clip 的門檻該設多少?

先釐清剪的方向:這裡是「同一個通道、跨所有 blank spaxel」比較,不是沿著波長。
一條天光線在每一個 spaxel 都亮,它的亮度就在該通道的中位數裡面 —— 不可能被
剪掉。會被剪掉的只有「在同一個波長上和其他 spaxel 不一樣」的東西。

所以門檻不是靠「設大一點比較安全」來選,而是看資料裡有沒有一道間隙:
真實的 spaxel 間變化到多少 sigma 為止,壞值從幾 sigma 開始。間隙夠寬的話,
門檻放在裡面任何位置結果都一樣 —— 那就不是一個要調的參數。

兩件事要量:
    ① 偏離量的分布   |x − med| / sg 超過各門檻的元素有幾個,間隙在哪
    ② 門檻掃描       各門檻下,天光線通道與連續譜通道的估計值各偏移多少
                     —— 若門檻太緊,右偏分布的正尾巴被切掉,亮線通道會系統性偏低

    conda run -n astro python src/skymodel/experiments/clip_threshold.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import estimate_continuum

ROOT   = Path(__file__).resolve().parents[3]
STEP01 = ROOT / "results/skymodel/ne_pointing/step01"
STEP03 = ROOT / "results/skymodel/ne_pointing/step03"
WSKY   = ROOT / "data/Haro11_NEpointing_wsky.fits"

WINDOW, THRESHOLDS, MAX_ITER = 300, (1, 2), 5
CHUNK = 200


def main():
    ap = argparse.ArgumentParser(description="sigma-clip 門檻的選擇")
    ap.add_argument("--grid", type=float, nargs="+",
                    default=[3, 5, 10, 20, 30, 50, 100, 200])
    args = ap.parse_args()

    white = fits.getdata(STEP01 / "whitelight.fits")
    seg   = fits.getdata(STEP01 / "seg.fits")
    bm    = (white != 0) & ~((seg > 0) & (white != 0))
    lm    = np.load(STEP03 / "iter_line_mask.npy")[0]        # 第 1 輪的線遮罩

    with fits.open(WSKY, memmap=True) as h:
        hdr = h["DATA"].header
        nz  = hdr["NAXIS3"]
        wl  = hdr["CRVAL3"] + (np.arange(nz) + 1 - hdr["CRPIX3"]) * hdr["CD3_3"]
        blank = np.empty((nz, int(bm.sum())), np.float32)
        for j in range(0, nz, CHUNK):
            blank[j:j+CHUNK] = np.asarray(h["DATA"].data[j:j+CHUNK],
                                          np.float32)[:, bm]
    blank = blank[:, np.isfinite(blank).all(axis=0)]
    N = blank.shape[1]
    print(f"blank spaxels {N:,}   {nz} channels   "
          f"共 {nz*N:,} 個元素   線通道佔 {100*lm.mean():.1f}%\n")

    p16, med, p84 = np.percentile(blank, [16, 50, 84], axis=1)
    sg = np.maximum((p84 - p16).astype(np.float64) / 2, 1e-6)
    ref = blank.mean(axis=1, dtype=np.float64)               # 現行的 nanmean

    # ---------- ① 偏離量的分布 ----------
    THR = np.array([3, 5, 10, 20, 30, 50, 100, 200, 500, 1e3, 1e4, 1e5])
    cnt = np.zeros((THR.size, 2), np.int64)                  # [門檻, (線, 線外)]
    for j in range(0, nz, CHUNK):
        sl = slice(j, j + CHUNK)
        d  = np.abs(blank[sl].astype(np.float64) - med[sl, None]) / sg[sl, None]
        for i, t in enumerate(THR):
            b = d > t
            cnt[i, 0] += int(b[lm[sl]].sum())
            cnt[i, 1] += int(b[~lm[sl]].sum())

    print("① 偏離量的分布 —— |x − med| / sg 超過門檻的元素數")
    print(f"{'門檻':>10}{'線通道':>14}{'線外通道':>14}{'合計':>12}{'佔全部':>14}")
    print("-" * 66)
    for t, (a, b) in zip(THR, cnt):
        print(f"{t:>10,.0f}{a:>14,}{b:>14,}{a+b:>12,}{100*(a+b)/(nz*N):>13.6f}%")

    # ---------- ② 門檻掃描 ----------
    print(f"\n② 門檻掃描 —— 相對現行 nanmean 的偏移(中位數,以 % 計)")
    print(f"{'門檻':>8}{'剔除比例':>12}{'線通道偏移':>14}{'線外偏移':>13}"
          f"{'mean[151]':>12}{'151 被遮?':>11}")
    print("-" * 72)
    for t in args.grid:
        num = np.zeros(nz); den = np.zeros(nz, np.int64)
        for j in range(0, nz, CHUNK):
            sl = slice(j, j + CHUNK)
            x  = blank[sl].astype(np.float64)
            k  = np.abs(x - med[sl, None]) <= t * sg[sl, None]
            num[sl] = (x * k).sum(axis=1)
            den[sl] = k.sum(axis=1)
        m   = num / den
        rel = 100 * (m / ref - 1)
        mask = estimate_continuum(m, thresholds=THRESHOLDS, window=WINDOW,
                                  max_iter=MAX_ITER)[2]
        print(f"{t:>8,.0f}{100*(1-den.mean()/N):>11.4f}%{np.median(rel[lm]):>13.4f}%"
              f"{np.median(rel[~lm]):>12.4f}%{m[151]:>12.3f}"
              f"{('是' if mask[151] else '否'):>11}")

    print(f"\n參考:通道 151 的 nanmean {ref[151]:.3f}、median {med[151]:.3f}、"
          f"sg {sg[151]:.3f}")
    print("線通道偏移若隨門檻變緊而變得更負,代表右偏分布的正尾巴被切掉 —— "
          "那才是「訊號被剪到」的真正形式。")


if __name__ == "__main__":
    main()
