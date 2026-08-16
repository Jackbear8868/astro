"""學 basis 之前,連續譜該怎麼扣?

現在 step3 的做法是每個 spaxel 都扣同一條 C_sky:
    residual = blank - C_sky[:, None]
但每個 spaxel 的連續譜強度不同,扣完之後會有殘留的連續譜留在訓練資料裡。
SVD 會把「連續譜整體亮一點/暗一點」當成一個變化方向去描述 —— 那是一條
被浪費掉的成分,因為 step5 的設計矩陣本來就有 C_sky 這一欄在管它。

三個版本:
    A  扣全域的 C_sky                          現行做法
    B  逐 spaxel 擬合 s(p) 再扣 s(p)·C_sky      全部通道上擬合
    C  同 B,但 s(p) 只用線外通道擬合            天光線會把 s 拉高,線外比較乾淨

評分一律用相同的方式:設計矩陣 = [C_sky, basis],未加權最小平方,
basis 只用 train spaxel 學,殘差只在 test spaxel 上算。
額外報告每條 basis 和 C_sky 的重疊,直接看有沒有成分被浪費在連續譜上。

    conda run -n astro python src/skymodel/experiments/continuum_prefit.py
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from sklearn.decomposition import TruncatedSVD

ROOT   = Path(__file__).resolve().parents[3]
STEP01 = ROOT / "results/skymodel/ne_pointing/step01"
STEP03 = ROOT / "results/skymodel/ne_pointing/step03"
WSKY   = ROOT / "data/Haro11_NEpointing_wsky.fits"

SEED = 0


def main():
    ap = argparse.ArgumentParser(description="學 basis 前連續譜的扣法")
    ap.add_argument("-K", type=int, nargs="+", default=[10, 25, 30, 40])
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    lm    = np.load(STEP03 / "iter_line_mask.npy")[0]
    C     = np.load(STEP03 / "sky_continuum.npy")

    with fits.open(WSKY, memmap=True) as h:
        D = np.asarray(h["DATA"].data, np.float32)
    nz, ny, nx = D.shape
    D = D.reshape(nz, -1)
    ok = (white != 0).ravel() & (seg.ravel() == 0) & np.isfinite(D).all(axis=0)
    idx = np.flatnonzero(ok)
    De = D[:, idx].astype(np.float64)
    del D

    rng = np.random.default_rng(SEED)
    tr = rng.random(idx.size) < 0.5
    Dtr, Dte = De[:, tr], De[:, ~tr]

    # 逐 spaxel 的 s:對 C_sky 做一維最小平方,s = (C·d)/(C·C)
    def s_of(X, rows):
        c = C[rows]
        return (c @ X[rows]) / (c @ c)

    s_all  = s_of(Dtr, np.ones(nz, bool))
    s_out  = s_of(Dtr, ~lm)
    print(f"blank {idx.size:,}   train {int(tr.sum()):,}   test {int((~tr).sum()):,}")
    print(f"逐 spaxel 的 s:全部通道 {np.median(s_all):.4f} ± "
          f"{(np.percentile(s_all,84)-np.percentile(s_all,16))/2:.4f}   "
          f"線外通道 {np.median(s_out):.4f} ± "
          f"{(np.percentile(s_out,84)-np.percentile(s_out,16))/2:.4f}")
    print(f"(天光線會把 s 拉高:兩者中位數差 "
          f"{100*(np.median(s_all)/np.median(s_out)-1):+.2f}%)\n")

    RES = {"A  global C_sky":      Dtr - C[:, None],
           "B  per-spaxel s, all": Dtr - C[:, None] * s_all[None, :],
           "C  per-spaxel s, off-line": Dtr - C[:, None] * s_out[None, :]}

    print(f"{'K':>4}{'做法':>26}{'線內 sigma':>13}{'線外 sigma':>13}{'rms':>10}"
          f"{'vs A':>9}{'第1條與C的重疊':>16}")
    print("-" * 92)
    for K in args.K:
        base = None
        for name, R in RES.items():
            B = TruncatedSVD(n_components=K, random_state=SEED).fit(R.T).components_
            sky = np.vstack([C, B])
            c = np.linalg.pinv(sky.T) @ Dte
            Rte = Dte - sky.T @ c
            si = np.median(Rte[lm].std(0)); so = np.median(Rte[~lm].std(0))
            rm = np.median(np.sqrt((Rte ** 2).mean(0)))
            # 每條 basis 和 C_sky 的夾角餘弦;接近 1 = 這條在描述連續譜
            ov = np.abs(B @ C) / (np.linalg.norm(B, axis=1) * np.linalg.norm(C))
            rel = "" if base is None else f"{100*(rm/base-1):>8.2f}%"
            print(f"{K:>4}{name:>26}{si:>13.4f}{so:>13.4f}{rm:>10.4f}{rel:>9}"
                  f"{ov.max():>16.4f}")
            if base is None:
                base = rm
        print()

    print("負的 vs A = 比現行做法好。「第1條與C的重疊」是所有 basis 中和 C_sky")
    print("夾角最小者的餘弦 —— 接近 1 代表有一條成分被浪費在描述連續譜強度上。")


if __name__ == "__main__":
    main()
