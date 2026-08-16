"""blank 係數圖的空間平滑,以 ESO nosky 為外部基準評估。

分兩部分:

  第一部分 留一格驗證 —— 鄰居到底能不能預測這一格?
      排除中心格,只用周圍的鄰居預測這一格的係數,和它自己解出來的值比。
      參考尺規是雜訊 sigma_n(相鄰格差分估):真值若在局部近似常數,誤差應為
          err ≈ sigma_n · sqrt(1 + 1/k)  ≈ 1.06 · sigma_n   (k ≈ 7 個有效鄰居)
      err/sigma_n 接近 1.06 → 鄰居把雜訊以外的都預測到了,平滑安全
      err/sigma_n 遠大於    → 有鄰居預測不到的真實結構,平滑會抹掉它

  第二部分 實際套用,用 ESO nosky 當外部基準評估
      不能用「blank 殘差」當判準 —— 未平滑就是讓殘差最小的解,任何平滑都必然
      讓它變大,那個判準的結論必然是「平滑有害」且毫無資訊。改用對 ESO 的
      指標:
          rms(全通道) 最直接的總結:這個 cube 到底好不好
          |偏移|      殘差沿波長的平均,即整體的高低偏移
          天光線抖動  線內通道的殘差散布
          連續譜抖動  線外通道的殘差散布
      後三者是 rms 的分解(rms² = 偏移² + sigma²,而 sigma 依通道分成線內/線外
      兩個不相交的集合),用來知道「好在哪裡」;rms 本身則是總結。

為什麼不用切通道的交叉驗證:basis 太稀疏(能量集中在極少數通道),任何通道
分半都會讓某幾條在某一半近乎消失,量到的是分半造成的假象。

    conda run -n astro python src/skymodel/experiments/blank_smooth_coef.py --basis svd -K 30
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT   = Path(__file__).resolve().parents[3]
STEP01 = ROOT / "results/skymodel/ne_pointing/step01"
STEP03 = ROOT / "results/skymodel/ne_pointing/step03"
CUBE   = ROOT / "data/Haro11_NEpointing_wsky.fits"
ESO    = ROOT / "data/Haro11_NEpointing_esonosky.fits"

MIN_COVERAGE = 0.9
# 八鄰域,權重 ∝ 1/距離²:上下左右 1,對角 1/2。這樣平滑核才是等向的。
OFFSETS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
           (-1, -1, 0.5), (-1, 1, 0.5), (1, -1, 0.5), (1, 1, 0.5)]


def shift(a, dy, dx, fill=0.0):
    out = np.full_like(a, fill)
    ys_s = slice(max(0, -dy), a.shape[-2] - max(0, dy))
    ys_d = slice(max(0, dy),  a.shape[-2] - max(0, -dy))
    xs_s = slice(max(0, -dx), a.shape[-1] - max(0, dx))
    xs_d = slice(max(0, dx),  a.shape[-1] - max(0, -dx))
    out[..., ys_d, xs_d] = a[..., ys_s, xs_s]
    return out


def scale(a):
    a = a[np.isfinite(a)]
    return float((np.percentile(a, 84) - np.percentile(a, 16)) / 2)


def noise_sigma(img, mask):
    """相鄰格差分估雜訊;取 y、x 較小者(較保守,不含真實結構)。"""
    out = []
    for dy, dx in [(1, 0), (0, 1)]:
        d = img - shift(img, dy, dx)
        m = mask & shift(mask.astype(float), dy, dx).astype(bool)
        out.append(scale(d[m]) / np.sqrt(2.0))
    return min(out)


def predict_neighbours(C, mask, how, radius=1, keep_centre=False):
    """用鄰居預測中心格。keep_centre=False 時排除中心(留一格驗證用)。"""
    Cn = np.where(mask, C, np.nan)
    stack, wts = [], []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                if not keep_centre:
                    continue
                w = 1.0
            else:
                w = 1.0 / (dy * dy + dx * dx)
            stack.append(shift(Cn, dy, dx, fill=np.nan))
            wts.append(w)
    S = np.stack(stack, axis=0)
    W = np.array(wts).reshape(-1, *([1] * (S.ndim - 1)))
    with np.errstate(invalid="ignore"):
        if how == "median":
            return np.nanmedian(S, axis=0)
        ok = np.isfinite(S)
        return np.nansum(np.where(ok, S, 0) * W, axis=0) / np.sum(ok * W, axis=0)


def main():
    ap = argparse.ArgumentParser(description="blank 係數平滑,以 ESO 為基準")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    lm1   = np.load(STEP03 / "iter_line_mask.npy")[0]
    sky   = np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                       np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")]).astype(np.float64)
    K = sky.shape[0]

    with fits.open(CUBE, memmap=True) as h:
        D = np.asarray(h["DATA"].data, np.float32)
        V = np.asarray(h["STAT"].data, np.float32)
    nz, ny, nx = D.shape
    D, V = D.reshape(nz, -1), V.reshape(nz, -1)
    E = fits.getdata(ESO).reshape(nz, -1)

    cov   = np.isfinite(D).sum(axis=0) / nz
    blank = (white != 0).ravel() & (seg.ravel() == 0) & (cov >= MIN_COVERAGE)
    ok = blank & np.isfinite(D).all(0) & np.isfinite(V).all(0) & np.isfinite(E).all(0)
    idx = np.flatnonzero(ok)
    De = D[:, idx].astype(np.float64)
    Ve = V[:, idx].astype(np.float64)
    Ee = E[:, idx].astype(np.float64)
    del D, V, E

    C0f = np.linalg.pinv(sky.T) @ De
    maps = np.full((K, ny * nx), np.nan)
    maps[:, idx] = C0f
    maps = maps.reshape(K, ny, nx)
    mask = np.zeros(ny * nx, bool); mask[idx] = True
    mask = mask.reshape(ny, nx)
    print(f"basis={args.basis}  K={args.K}  ({K} 個成分,含連續譜)  blank {idx.size:,} 個\n")

    # ---------------- 第一部分:留一格驗證 ----------------
    print("=" * 76)
    print("留一格驗證:err / sigma_n   (理論下限 1.06;越接近 = 平滑越安全)")
    print("=" * 76)
    print(f"{'係數':>6}{'總散布':>12}{'雜訊 sigma_n':>14}"
          f"{'加權均值':>11}{'中位數':>10}{'5x5 均值':>11}{'損失結構/sn':>14}")
    print("-" * 76)
    show = [0] + list(range(1, K, max(1, (K - 1) // 6)))
    ratios = []
    for k in range(K):
        img = maps[k]
        sig, tot = noise_sigma(img, mask), scale(img[mask])
        row = []
        for how, r in [("mean", 1), ("median", 1), ("mean", 2)]:
            pred = predict_neighbours(img, mask, how, radius=r)
            row.append(scale((pred - img)[mask & np.isfinite(pred)]) / sig if sig > 0 else np.nan)
        ratios.append(row)
        if k in show:
            lost = np.sqrt(max(row[0] ** 2 - 1.06 ** 2, 0))
            name = "s" if k == 0 else f"c{k}"
            print(f"{name:>6}{tot:>12.5g}{sig:>14.5g}"
                  f"{row[0]:>11.3f}{row[1]:>10.3f}{row[2]:>11.3f}{lost:>14.2f}")
    R = np.array(ratios)
    print(f"{'中位數':>6}{'':>26}{np.median(R[:,0]):>11.3f}"
          f"{np.median(R[:,1]):>10.3f}{np.median(R[:,2]):>11.3f}")
    print(f"{'  其中 s':>6}{'':>26}{R[0,0]:>11.3f}{R[0,1]:>10.3f}{R[0,2]:>11.3f}")

    # ---------------- 第二部分:套用平滑,以 ESO 為基準 ----------------
    IN, OUT = lm1, ~lm1
    e_off = np.median(np.abs(Ee.mean(0)))
    e_in, e_out = Ee[IN].std(0), Ee[OUT].std(0)
    e_rms = np.sqrt(np.mean(Ee ** 2, axis=0))
    print("\n" + "=" * 76)
    print("套用平滑,以 ESO nosky 為外部基準")
    print("=" * 76)
    print(f"ESO:  rms {np.median(e_rms):.4f}   |偏移| {e_off:.4f}   "
          f"線內 sigma {np.median(e_in):.4f}   線外 sigma {np.median(e_out):.4f}\n")
    print(f"{'設定':>18}{'rms':>10}{'rms贏':>8}{'|偏移|':>10}{'偏移贏':>8}"
          f"{'線內 sigma':>12}{'線內贏':>8}{'線外 sigma':>12}{'線外贏':>8}{'s 散布':>10}")
    print("-" * 106)
    for which, lab in [("none", "未平滑"), ("s", "只平滑 s"),
                       ("lines", "只平滑線"), ("all", "全部平滑"),
                       ("s_median", "只平滑 s (中位數)"), ("s_5x5", "只平滑 s (5x5)")]:
        M = maps.copy()
        rows_k = {"none": [], "s": [0], "lines": list(range(1, K)),
                  "all": list(range(K)), "s_median": [0], "s_5x5": [0]}[which]
        how = "median" if which == "s_median" else "mean"
        rad = 2 if which == "s_5x5" else 1
        for k in rows_k:
            M[k] = np.where(mask, predict_neighbours(M[k], mask, how, rad, keep_centre=True),
                            np.nan)
        c = M.reshape(K, -1)[:, idx]
        Rr = De - sky.T @ c
        off = np.abs(Rr.mean(0))
        si, so = Rr[IN].std(0), Rr[OUT].std(0)
        rms = np.sqrt(np.mean(Rr ** 2, axis=0))
        print(f"{lab:>18}{np.median(rms):>10.4f}{100*np.mean(rms<e_rms):>7.1f}%"
              f"{np.median(off):>10.4f}{100*np.mean(off<np.abs(Ee.mean(0))):>7.1f}%"
              f"{np.median(si):>12.4f}{100*np.mean(si<e_in):>7.1f}%"
              f"{np.median(so):>12.4f}{100*np.mean(so<e_out):>7.1f}%"
              f"{scale(c[0]):>10.5f}")


if __name__ == "__main__":
    main()
