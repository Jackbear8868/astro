"""mean_sky 用哪一種估計量?nanmean / median / sigma-clipped mean

mean_sky 不是拿來看的,它是 estimate_continuum 的唯一輸入,往下決定三樣東西:

    C_sky      天空連續譜        step4/step5 設計矩陣的第一欄
    sigma      線偵測的雜訊尺度
    line_mask  哪些通道算天光線

所以換估計量要看的不是「mean_sky 好不好看」,而是這三樣有沒有變。特別要查
line_mask:通道 151 的 nanmean 是 1.8 而連續譜是 29.2,殘差 −27.4 是個很深的
負值,而門檻含「負 2 sigma」—— 那個通道很可能被誤判成一條負的線而遮掉。

三種估計量:
    nanmean          現行,崩潰點 0%(一個極端值就能拉走)
    median           崩潰點 50%,但精度要多花 57% 的樣本
    sigma-clipped    先用 median/百分位定中心與散布,剔除 >N sigma,剩下取平均
                     —— 剔除那步穩健,平均那步精確

    conda run -n astro python src/skymodel/experiments/mean_estimator.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import estimate_continuum

ROOT   = Path(__file__).resolve().parents[3]
STEP01 = ROOT / "results/skymodel/step01"
STEP03 = ROOT / "results/skymodel/step03"
WSKY   = ROOT / "data/Haro11_NEpointing_wsky.fits"

WINDOW, THRESHOLDS, MAX_ITER = 300, (1, 2), 5


def main():
    ap = argparse.ArgumentParser(description="mean_sky 估計量的比較")
    ap.add_argument("--nsigma", type=float, default=5.0,
                    help="sigma-clip 的門檻(對 mean 而言 5 已經很寬鬆)")
    args = ap.parse_args()

    white = fits.getdata(STEP01 / "whitelight.fits")
    seg   = fits.getdata(STEP01 / "seg.fits")
    bm    = (white != 0) & ~((seg > 0) & (white != 0))

    with fits.open(WSKY, memmap=True) as h:
        hdr = h["DATA"].header
        nz  = hdr["NAXIS3"]
        wl  = hdr["CRVAL3"] + (np.arange(nz) + 1 - hdr["CRPIX3"]) * hdr["CD3_3"]
        blank = np.empty((nz, int(bm.sum())), np.float32)
        for j in range(0, nz, 200):
            blank[j:j+200] = np.asarray(h["DATA"].data[j:j+200], np.float32)[:, bm]
    blank = blank[:, np.isfinite(blank).all(axis=0)].astype(np.float64)
    N = blank.shape[1]
    print(f"blank spaxels(完整光譜) {N:,}   {nz} channels\n")

    # ---------- 三種估計量 ----------
    med = np.median(blank, axis=1)
    sg  = np.maximum((np.percentile(blank, 84, axis=1)
                      - np.percentile(blank, 16, axis=1)) / 2, 1e-6)
    keep = np.abs(blank - med[:, None]) <= args.nsigma * sg[:, None]
    nkeep = keep.sum(axis=1)
    clipped = (blank * keep).sum(axis=1) / nkeep

    EST = [("nanmean", blank.mean(axis=1)),
           ("median", med),
           (f"clipped {args.nsigma:g}s", clipped)]

    # 有多少通道被離群值污染?用「nanmean 和 clipped 差多少個 clipped 自己的誤差」衡量
    err = sg / np.sqrt(nkeep)                    # clipped mean 自身的統計精度
    pull = np.abs(EST[0][1] - clipped) / err
    print(f"sigma-clip({args.nsigma:g}s) 平均剔除 {N - nkeep.mean():.1f} / {N:,} "
          f"個 spaxel ({100*(1-nkeep.mean()/N):.4f}%)")
    print(f"估計量自身的統計精度 sg/sqrt(N):中位 {np.median(err):.4f} "
          f"(相對連續譜 ~{100*np.median(err/med):.3f}%)")
    print(f"nanmean 偏離 clipped 超過 3 個自身誤差的通道:"
          f"{int((pull > 3).sum())} / {nz}")
    worst = np.argsort(pull)[::-1][:8]
    print(f"\n最嚴重的 8 個通道:")
    print(f"{'ch':>6}{'wl (A)':>10}{'nanmean':>13}{'median':>10}"
          f"{'clipped':>10}{'偏離(自身誤差)':>16}")
    print("-" * 66)
    for c in worst:
        print(f"{c:>6}{wl[c]:>10.1f}{EST[0][1][c]:>13.3f}{med[c]:>10.3f}"
              f"{clipped[c]:>10.3f}{pull[c]:>16,.0f}")

    # ---------- 下游:C_sky / sigma / line_mask ----------
    print(f"\n{'估計量':>16}{'mean[151]':>12}{'C_sky[151]':>12}{'sigma[151]':>12}"
          f"{'151 被遮?':>11}{'遮罩比例':>10}{'迭代':>6}")
    print("-" * 80)
    OUT = {}
    for name, m in EST:
        C, s, lmask, hist = estimate_continuum(m, thresholds=THRESHOLDS,
                                               window=WINDOW, max_iter=MAX_ITER)
        OUT[name] = (C, s, lmask)
        print(f"{name:>16}{m[151]:>12.3f}{C[151]:>12.3f}{s[151]:>12.3f}"
              f"{('是' if lmask[151] else '否'):>11}"
              f"{100*lmask.mean():>9.2f}%{len(hist):>6}")

    ref = OUT["nanmean"]
    print(f"\n{'vs nanmean':>16}{'C_sky 最大差':>14}{'C_sky 中位差':>14}"
          f"{'遮罩不同的通道':>16}")
    print("-" * 62)
    for name, _ in EST[1:]:
        C, s, lmask = OUT[name]
        d = np.abs(C - ref[0])
        diff = np.flatnonzero(lmask != ref[2])
        print(f"{name:>16}{d.max():>14.4f}{np.median(d):>14.4f}"
              f"{diff.size:>16}")
        if diff.size:
            print(f"{'':>16}不同的通道:"
                  + ", ".join(f"{c}({wl[c]:.1f}A,"
                              f"{'新增' if lmask[c] else '移除'})" for c in diff[:12]))

    print("\n『151 被遮?』= 那個壞掉的通道有沒有被 estimate_continuum 誤判成天光線。")
    print("被誤遮的通道會從線外統計中消失,也不會進入線偵測 —— 是靜默的資料損失。")


if __name__ == "__main__":
    main()
