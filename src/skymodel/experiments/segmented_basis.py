"""分波長區段學 basis(ZAP 的做法)vs 全波段一組。

ZAP 把光譜切成 11 段各自做 SVD(Soto+2016 Table 1),邊界照 OH 振動帶的族群切:
同一段裡主導的是同一組躍遷,強度會一起變化,所以段內的自由度少,約 10 條就夠。
11 段 x 10 條 = 約 110 個係數,但每個只作用在自己那一段 —— 設計矩陣是塊對角的。

我們現在是全波段一組 K 條,每一條都得同時描述所有族群的變化。

公平的比較必須固定「總參數數目」:
    全波段    K_total 條,每條作用在 3801 個通道
    分區段    每段 K_total/11 條(依段長加權),只作用在該段
兩者的自由度相同,差別只在「基底能不能跨段耦合」。

驗證一樣是樣本外:basis 只用 train spaxel 學,殘差只在 test spaxel 上算。

    conda run -n astro python src/skymodel/experiments/segmented_basis.py
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from sklearn.decomposition import TruncatedSVD

ROOT   = Path(__file__).resolve().parents[3]
STEP01 = ROOT / "results/skymodel/step01"
STEP03 = ROOT / "results/skymodel/step03"
WSKY   = ROOT / "data/Haro11_NEpointing_wsky.fits"

SEED = 0

# Soto+2016 Table 1。第一段的下界改成 4600,因為 NE pointing 的藍端多 150 A。
ZAP_SEGMENTS = [(4600, 5400), (5400, 5850), (5850, 6440), (6440, 6750),
                (6750, 7200), (7200, 7700), (7700, 8265), (8265, 8602),
                (8602, 8731), (8731, 9275), (9275, 9400)]


def fit_eval(sky, Dte, lm):
    """未加權最小平方(和 fit_blank 的快路徑相同),回傳 test 的三個指標。"""
    c = np.linalg.pinv(sky.T) @ Dte
    R = Dte - sky.T @ c
    return (np.median(R[lm].std(0)), np.median(R[~lm].std(0)),
            np.median(np.sqrt((R ** 2).mean(0))))


def main():
    ap = argparse.ArgumentParser(description="分波長區段 vs 全波段")
    ap.add_argument("--totals", type=int, nargs="+", default=[11, 22, 33, 55, 110],
                    help="總參數數目(不含連續譜);分段時平均分給 11 段")
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    wl    = np.load(STEP03 / "wavelength.npy")
    lm    = np.load(STEP03 / "iter_line_mask.npy")[0]
    C_sky = np.load(STEP03 / "sky_continuum.npy")

    with fits.open(WSKY, memmap=True) as h:
        D = np.asarray(h["DATA"].data, np.float32)
    nz, ny, nx = D.shape
    D = D.reshape(nz, -1)
    ok = (white != 0).ravel() & (seg.ravel() == 0) & np.isfinite(D).all(axis=0)
    idx = np.flatnonzero(ok)
    De = D[:, idx].astype(np.float32)
    del D

    rng = np.random.default_rng(SEED)
    tr = rng.random(idx.size) < 0.5
    Dtr, Dte = De[:, tr], De[:, ~tr]
    Rtr = Dtr - C_sky[:, None]

    masks = [(wl >= a) & (wl < b) for a, b in ZAP_SEGMENTS]
    print(f"blank {idx.size:,}   train {int(tr.sum()):,}   test {int((~tr).sum()):,}")
    print(f"{len(masks)} 個波長區段,通道數:"
          f"{[int(m.sum()) for m in masks]}\n")

    print(f"{'總參數':>8}{'做法':>12}{'線內 sigma':>13}{'線外 sigma':>13}"
          f"{'rms':>10}{'分段 vs 全波段':>16}")
    print("-" * 74)
    for tot in args.totals:
        # --- 全波段 ---
        B = TruncatedSVD(n_components=tot, random_state=SEED).fit(Rtr.T).components_
        g = fit_eval(np.vstack([C_sky, B]), Dte, lm)
        print(f"{tot:>8}{'全波段':>12}{g[0]:>13.4f}{g[1]:>13.4f}{g[2]:>10.4f}")

        # --- 分區段:每段的條數依段長比例分配,至少 1 條 ---
        lens = np.array([m.sum() for m in masks], float)
        per = np.maximum(1, np.round(tot * lens / lens.sum()).astype(int))
        rows = []
        for m, k in zip(masks, per):
            sub = Rtr[m]                       # 只用該段的通道學
            kk = min(k, int(m.sum()) - 1)
            if kk < 1:
                continue
            c = TruncatedSVD(n_components=kk, random_state=SEED).fit(sub.T).components_
            blk = np.zeros((kk, nz))           # 塊對角:段外一律為 0
            blk[:, m] = c
            rows.append(blk)
        Bseg = np.vstack(rows)
        s = fit_eval(np.vstack([C_sky, Bseg]), Dte, lm)
        rel = 100 * (s[2] / g[2] - 1)
        print(f"{Bseg.shape[0]:>8}{'分 11 段':>12}{s[0]:>13.4f}{s[1]:>13.4f}"
              f"{s[2]:>10.4f}{rel:>15.2f}%")
        print(f"{'':>8}{'':>12}{'每段條數 ' + str(per.tolist()):>60}")
        print()

    print("負的 = 分段比較好。注意分段的實際條數會因四捨五入略多於指定的總數。")


if __name__ == "__main__":
    main()
