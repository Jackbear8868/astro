"""學 basis 之前先剔除離群值,會改善嗎?

發現的問題:通道 151(4788.4 A)有少數 blank spaxel 的值遠低於正常範圍。它們在
空間上成團,是宇宙射線或壞畫素在重採樣時被抹開的形態。

兩層後果:
    mean_sky 用 np.nanmean,那個通道的平均被那些值拉走
    更嚴重:SVD 的訓練矩陣裡就有這些值,於是產生一條能量幾乎全在通道 151
    的 basis —— 它描述的不是天空,是那些壞 spaxel。
    然後任何模板只要藍端覆蓋越過 4788 A,那條 basis 的係數就失去約束,
    chi2_all 暴增,該模板在那個 z 之上被假性禁止。

三個做法:
    A  不處理                         現行
    B  逐通道穩健截斷,離群值設 0       殘差空間的 0 = 「等於連續譜」
    C  整條 spaxel 剔除                只要有任一通道離群就不用這個 spaxel

評分:train/test 分半,basis 只用 train 學,殘差在 test 上算;
test 的統計排除離群 spaxel,否則比較會被離群值本身主導。
另外報告 basis 的能量集中度 —— 那條被浪費的成分有沒有消失。

    conda run -n astro python src/skymodel/experiments/outlier_reject.py
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


def n90(v):
    """能量的 90% 落在幾個通道 —— 個位數代表這條 basis 被單一壞點佔走。"""
    e = np.sort(v ** 2)[::-1]
    return int(np.searchsorted(np.cumsum(e), 0.9 * e.sum()) + 1)


def main():
    ap = argparse.ArgumentParser(description="學 basis 前的離群值處理")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--nsigma", type=float, nargs="+", default=[100.0, 30.0, 15.0])
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
    Rtr = De[:, tr] - C[:, None]
    Rte_raw = De[:, ~tr] - C[:, None]

    # 離群的判準要看「偏離該通道的中位數多少」,不是 R 本身 ——
    # R = D − C_sky 裡還含著天光線,天光線通道上每個 spaxel 的 R 都很大,
    # 直接除以散布會讓每個 spaxel 都被判成離群。
    med = np.median(Rtr, axis=1)
    sg = (np.percentile(Rtr, 84, axis=1) - np.percentile(Rtr, 16, axis=1)) / 2
    sg = np.maximum(sg, 1e-6)
    dev_tr = np.abs(Rtr - med[:, None]) / sg[:, None]
    dev_te = np.abs(Rte_raw - med[:, None]) / sg[:, None]

    def stats(B, clean_mask):
        sky = np.vstack([C, B])
        c = np.linalg.pinv(sky.T) @ De[:, ~tr]
        R = De[:, ~tr] - sky.T @ c
        R = R[:, clean_mask]
        return (np.median(R[lm].std(0)), np.median(R[~lm].std(0)),
                np.median(np.sqrt((R ** 2).mean(0))))

    # test 端的乾淨 spaxel:用最嚴的門檻界定,所有做法共用同一批,比較才公平
    te_bad = (dev_te > min(args.nsigma)).any(axis=0)
    clean = ~te_bad
    print(f"blank {idx.size:,}   train {int(tr.sum()):,}   test {int((~tr).sum()):,}")
    print(f"test 端排除 {int(te_bad.sum()):,} 個含離群值的 spaxel,"
          f"統計用剩下的 {int(clean.sum()):,} 個\n")

    print(f"{'做法':>22}{'線內 sigma':>12}{'線外 sigma':>12}{'rms':>9}{'vs A':>9}"
          f"{'最集中 basis 的 n90':>20}{'訓練 spaxel':>12}")
    print("-" * 98)

    B = TruncatedSVD(n_components=args.K, random_state=SEED).fit(Rtr.T).components_
    a = stats(B, clean)
    base = a[2]
    print(f"{'A  不處理':>22}{a[0]:>12.4f}{a[1]:>12.4f}{a[2]:>9.4f}{'':>9}"
          f"{min(n90(b) for b in B):>20}{int(tr.sum()):>12,}")

    for ns in args.nsigma:
        bad = dev_tr > ns
        Rb = np.where(bad, 0.0, Rtr)                    # B:離群值設 0
        B = TruncatedSVD(n_components=args.K, random_state=SEED).fit(Rb.T).components_
        r = stats(B, clean)
        print(f"{f'B  截斷 {ns:g} sigma → 0':>22}{r[0]:>12.4f}{r[1]:>12.4f}"
              f"{r[2]:>9.4f}{100*(r[2]/base-1):>8.2f}%"
              f"{min(n90(b) for b in B):>20}{int(tr.sum()):>12,}"
              f"   (剔除 {100*bad.mean():.4f}% 的元素)")

    for ns in args.nsigma:
        keep = ~(dev_tr > ns).any(axis=0)
        if keep.sum() < 4 * args.K:
            continue
        B = TruncatedSVD(n_components=args.K,
                         random_state=SEED).fit(Rtr[:, keep].T).components_
        r = stats(B, clean)
        print(f"{f'C  剔除 spaxel {ns:g} sigma':>22}{r[0]:>12.4f}{r[1]:>12.4f}"
              f"{r[2]:>9.4f}{100*(r[2]/base-1):>8.2f}%"
              f"{min(n90(b) for b in B):>20}{int(keep.sum()):>12,}")

    print("\n最集中 basis 的 n90 = 所有 basis 裡「90% 能量落在最少通道」的那條。")
    print("個位數代表有一條成分被單一壞點佔走;變大代表那條成分回去描述真正的天空。")


if __name__ == "__main__":
    main()
