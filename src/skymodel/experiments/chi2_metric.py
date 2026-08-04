"""用 chi2 式的指標重新評分:① 的否決是不是指標選擇造成的?

先前所有的比較都用未加權的量(rms、逐通道散布)。問題是未加權擬合的目標函數
就是「最小化未加權平方誤差」—— 拿同一個東西當指標,它當然贏。這不叫作弊,
但也不能拿來證明加權擬合比較差。

chi2 式的指標問的是不同的問題:

    未加權 sigma   殘差有多大(以流量計)
    chi = R/sigma  殘差相對於該通道的雜訊有多大 —— 這才是決定「一個微弱源
                   能不能在這個通道上被偵測到」的量

如果加權擬合在 chi 指標上贏,結論就要改寫成「取決於科學目標」;
如果它連 chi 指標都輸,那它就是單純比較差。

理想值:sigma 若正確,chi 應該是 1。實測 STAT 低估約 1.32,所以底線在 1.32 附近。

    conda run -n astro python src/skymodel/experiments/chi2_metric.py
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT   = Path(__file__).resolve().parents[3]
STEP01 = ROOT / "results/skymodel/step01"
STEP03 = ROOT / "results/skymodel/step03"
STEP05 = ROOT / "results/skymodel/step05"
WSKY   = ROOT / "data/Haro11_NEpointing_wsky.fits"
ESO    = ROOT / "data/Haro11_NEpointing_esonosky.fits"

MIN_COVERAGE = 0.9


def main():
    ap = argparse.ArgumentParser(description="chi2 式指標的比較")
    ap.add_argument("--tags", nargs="+",
                    default=["svd_K30_s_1.0", "svd_K30_s_1.0_bchi2"])
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    IN    = np.load(STEP03 / "iter_line_mask.npy")[0]

    with fits.open(WSKY, memmap=True) as h:
        V = np.asarray(h["STAT"].data, np.float32)
    nz = V.shape[0]
    V = V.reshape(nz, -1)
    cov = np.isfinite(V).sum(axis=0) / nz

    E  = fits.getdata(ESO).reshape(nz, -1)
    ok = ((white != 0) & (seg == 0)).ravel() & (cov >= MIN_COVERAGE) \
         & np.isfinite(E).all(0) & np.isfinite(V).all(0)
    cubes = {}
    for t in args.tags:
        S = fits.getdata(STEP05 / f"sky_subtracted_{t}.fits").reshape(nz, -1)
        ok &= np.isfinite(S).all(0)
        cubes[t] = S
    idx = np.flatnonzero(ok)
    sig = np.sqrt(V[:, idx].astype(np.float64))
    del V
    print(f"blank {idx.size:,} spaxel   線內 {int(IN.sum())}/{nz} 通道")
    print("chi = 殘差 / sqrt(STAT);逐 spaxel 取均方根,再取中位數\n")

    print(f"{'':>22}{'chi 全部':>11}{'chi 線內':>11}{'chi 線外':>11}"
          f"{'|chi 平均|':>12}")
    print("-" * 68)
    rows = {}
    for nm, X in [("ESO nosky", E)] + [(t, cubes[t]) for t in args.tags]:
        c = X[:, idx].astype(np.float64) / sig
        rows[nm] = (float(np.median(np.sqrt(np.mean(c ** 2, axis=0)))),
                    float(np.median(np.sqrt(np.mean(c[IN] ** 2, axis=0)))),
                    float(np.median(np.sqrt(np.mean(c[~IN] ** 2, axis=0)))),
                    float(np.median(np.abs(c.mean(axis=0)))))
        print(f"{nm:>22}{rows[nm][0]:>11.4f}{rows[nm][1]:>11.4f}"
              f"{rows[nm][2]:>11.4f}{rows[nm][3]:>12.4f}")

    a, b = args.tags[0], args.tags[1] if len(args.tags) > 1 else None
    if b:
        print(f"\n{b}\n  相對於 {a}:", end="")
        print("  ".join(f"{k} {100*(rows[b][i]/rows[a][i]-1):+.2f}%"
                        for i, k in enumerate(["全部", "線內", "線外", "|平均|"])))
    print("\n正號 = 更差。若加權擬合在 chi 指標上仍然輸,它就是單純比較差,")
    print("不是「換個指標就會贏」。")


if __name__ == "__main__":
    main()
