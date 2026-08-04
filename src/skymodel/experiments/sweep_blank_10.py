"""在固定的 10 個 blank spaxel 上掃描 basis 與 K。

blank 區的擬合完全不用模板,所以不必跑 step4/step5 —— 直接載入 basis、
對那 10 個 spaxel 解係數就好(和 step5 的 fit_blank 同一套:未加權最小平方,
事後補 s >= 0)。一個設定幾秒鐘,而不是 22 分鐘。

報告拆成三塊,因為它們的物理意義不同:
    mean    零點。我們的優勢來源(s 固定成 1.0 的效果)
    sigma   抖動。再拆成天光線區與連續譜區 —— K 只對前者有效
    模型誤差 sqrt(sigma² − 雜訊²)。扣掉光子雜訊之後才是模型真正的誤差

    conda run -n astro python src/skymodel/experiments/sweep_blank_10.py
    conda run -n astro python src/skymodel/experiments/sweep_blank_10.py -K 10 25 30
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_blank_spectra import FIXED_SPAXELS

ROOT   = Path(__file__).resolve().parents[3]
STEP01 = ROOT / "results/skymodel/step01"
STEP03 = ROOT / "results/skymodel/step03"
WSKY   = ROOT / "data/Haro11_NEpointing_wsky.fits"
ESO    = ROOT / "data/Haro11_NEpointing_esonosky.fits"


def fit_one(d, sky):
    """單一 spaxel 的 blank 係數,和 step5 的 fit_blank 同一套。"""
    g = np.isfinite(d)
    c = np.linalg.lstsq(sky[:, g].T, d[g], rcond=None)[0]
    if c[0] < 0:                       # s 不能是負的
        lb = np.r_[0.0, np.full(sky.shape[0] - 1, -np.inf)]
        c = lsq_linear(sky[:, g].T, d[g], bounds=(lb, np.full(sky.shape[0], np.inf)),
                       method="bvls").x
    return c


def stats(r, lm):
    """回傳 (mean, sigma_全, rms, sigma_線內, sigma_線外)。"""
    return (r.mean(), r.std(), np.sqrt(np.mean(r ** 2)), r[lm].std(), r[~lm].std())


def main():
    ap = argparse.ArgumentParser(description="固定 10 個 blank spaxel 上掃 basis 與 K")
    ap.add_argument("--basis", nargs="+", default=["svd", "pca", "nmf", "rpca"])
    ap.add_argument("-K", type=int, nargs="+", default=[10, 25, 30])
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    lm    = np.load(STEP03 / "iter_line_mask.npy")[0]
    C_sky = np.load(STEP03 / "sky_continuum.npy")
    nx    = seg.shape[1]
    idx   = [y * nx + x for y, x in FIXED_SPAXELS]

    with fits.open(WSKY, memmap=True) as h:
        nz = h["DATA"].data.shape[0]
        D  = h["DATA"].data.reshape(nz, -1)[:, idx].astype(np.float64)
        V  = h["STAT"].data.reshape(nz, -1)[:, idx].astype(np.float64)
    E = fits.getdata(ESO).reshape(nz, -1)[:, idx].astype(np.float64)

    # 光子雜訊下限:STAT 的變異數開根號,分線內線外
    nf_in, nf_out = np.sqrt(V[lm].mean(0)), np.sqrt(V[~lm].mean(0))

    def report(lab, R):
        s = np.array([stats(R[:, j], lm) for j in range(R.shape[1])])
        me_in  = np.sqrt(np.maximum(np.median(s[:, 3]) ** 2 - np.median(nf_in) ** 2, 0))
        me_out = np.sqrt(np.maximum(np.median(s[:, 4]) ** 2 - np.median(nf_out) ** 2, 0))
        print(f"{lab:>14}{np.median(np.abs(s[:,0])):>10.4f}{np.median(s[:,2]):>9.4f}"
              f"{np.median(s[:,3]):>11.4f}{me_in:>10.4f}"
              f"{np.median(s[:,4]):>11.4f}{me_out:>10.4f}", flush=True)
        return s

    print(f"固定 10 個 blank spaxel   線內 {int(lm.sum())}/{nz} 通道")
    print(f"光子雜訊下限   線內 {np.median(nf_in):.4f}   線外 {np.median(nf_out):.4f}\n")
    print(f"{'設定':>14}{'|mean|':>10}{'rms':>9}"
          f"{'線內 sigma':>11}{'模型誤差':>10}{'線外 sigma':>11}{'模型誤差':>10}")
    print("-" * 75)
    ref = report("ESO nosky", E)
    print("-" * 75)

    for b in args.basis:
        for K in args.K:
            f = STEP03 / f"sky_basis_{b}_K{K}.npy"
            if not f.exists():
                print(f"{b+' K='+str(K):>14}   (缺 {f.name})")
                continue
            sky = np.vstack([C_sky, np.load(f)])
            R = np.column_stack([D[:, j] - sky.T @ fit_one(D[:, j], sky)
                                 for j in range(D.shape[1])])
            s = report(f"{b} K={K}", R)
            win = (np.array([np.sqrt(np.mean(R[:, j] ** 2)) for j in range(R.shape[1])])
                   < np.array([np.sqrt(np.mean(E[:, j] ** 2)) for j in range(E.shape[1])]))
            print(f"{'':>14}{'':>10}{'贏 '+str(int(win.sum()))+'/10':>9}")
        print()


if __name__ == "__main__":
    main()
