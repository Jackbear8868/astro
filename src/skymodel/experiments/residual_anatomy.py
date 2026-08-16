"""blank 殘差的解剖:剩下的誤差是雜訊,還是我們沒建好的結構?

把殘差的 sigma 和 STAT 給的光子雜訊下限相減,差額通常被當成「模型誤差」。
但那個差額也可能只是 STAT 低估了真實雜訊,兩種解釋給出的數字一樣。

要分辨這兩者,不能再用 STAT(它正是被懷疑的對象)。改用相關結構:

    真的白雜訊     相鄰通道獨立   →  var[R(λ) − R(λ+1)] = 2 · var[R]
    平滑的模型誤差  相鄰通道相關   →  差分把它消掉,差分變異數 < 2 · var[R]

所以 sqrt(var[差分]/2) 是一個「不依賴 STAT 的雜訊估計」。把它和 R 本身的
sigma 比,差額就是有相關結構的部分 —— 那才是建模有可能拿掉的東西。

三個方向各查一次:
    波長方向  lag 1-5 的自相關      平滑的連續譜殘留會在這裡現形
    空間方向  相鄰 spaxel 的相關    共同的天空/儀器結構會在這裡現形
    spaxel 級 逐 spaxel 的超額      未偵測到的暗源會讓少數 spaxel 系統性偏高

    conda run -n astro python src/skymodel/experiments/residual_anatomy.py
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


def lag_sigma(R, mask, lag):
    """用 lag 階差分估雜訊:sqrt(var[R(i)−R(i+lag)] / 2),逐 spaxel 後取中位。

    只取「兩端都在 mask 內」的通道對,才不會跨到天光線去。
    """
    i = np.flatnonzero(mask[:-lag] & mask[lag:])
    d = R[i + lag] - R[i]
    return np.median(d.std(axis=0)) / np.sqrt(2.0)


def main():
    ap = argparse.ArgumentParser(description="blank 殘差的組成")
    ap.add_argument("--tag", default="svd_K30_s_1.0")
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    IN    = np.load(STEP03 / "iter_line_mask.npy")[0]
    OUT   = ~IN

    with fits.open(WSKY, memmap=True) as h:
        V = np.asarray(h["STAT"].data, np.float32)
    nz, ny, nx = V.shape
    V = V.reshape(nz, -1)

    S = fits.getdata(STEP05 / f"sky_subtracted_{args.tag}.fits").reshape(nz, -1)
    E = fits.getdata(ESO).reshape(nz, -1)
    ok = ((white != 0) & (seg == 0)).ravel() & np.isfinite(S).all(0) \
         & np.isfinite(E).all(0) & np.isfinite(V).all(0)
    idx = np.flatnonzero(ok)
    R  = S[:, idx].astype(np.float64)
    Re = E[:, idx].astype(np.float64)
    Vs = V[:, idx].astype(np.float64)
    print(f"blank {idx.size:,} spaxel   線外 {int(OUT.sum())}/{nz} 通道\n")

    # ---------- ① 波長方向:差分估雜訊 vs STAT ----------
    print("① 不依賴 STAT 的雜訊估計(相鄰通道差分)")
    print(f"{'':>6}{'':>12}{'sigma':>10}{'lag3 估雜訊':>13}{'超出雜訊':>11}"
          f"{'STAT':>9}{'真雜訊/STAT':>13}")
    print("-" * 76)
    for reg, M in [("線內", IN), ("線外", OUT)]:
        for nm, X in [("我們", R), ("ESO nosky", Re)]:
            s  = float(np.median(X[M].std(axis=0)))
            n3 = lag_sigma(X, M, 3)                  # lag3:已越過重採樣的相關長度
            st = float(np.median(np.sqrt(np.mean(Vs[M], axis=0))))
            print(f"{reg:>6}{nm:>12}{s:>10.4f}{n3:>13.4f}"
                  f"{np.sqrt(max(s**2-n3**2,0)):>11.4f}{st:>9.4f}{n3/st:>13.4f}")
    print("\n  超出雜訊 ~ 0     → 已在雜訊底線,沒有結構可以再建模")
    print("  真雜訊/STAT > 1  → STAT 低估,和它相比得到的「模型誤差」是假的\n")

    # ---------- ② 自相關隨 lag 的變化 ----------
    print("② 差分估計隨 lag 的變化")
    print(f"{'lag':>6}{'線內 我們':>13}{'線內 ESO':>12}{'線外 我們':>13}{'線外 ESO':>12}")
    print("-" * 58)
    for lag in [1, 2, 3, 5, 10, 30]:
        print(f"{lag:>6}{lag_sigma(R, IN, lag):>13.4f}{lag_sigma(Re, IN, lag):>12.4f}"
              f"{lag_sigma(R, OUT, lag):>13.4f}{lag_sigma(Re, OUT, lag):>12.4f}")
    print("  隨 lag 上升後持平 → 相關長度就是持平之處;持平值 = 真正的雜訊水準\n")

    # ---------- ③ 空間方向:相鄰 spaxel 的相關 ----------
    m2 = ok.reshape(ny, nx)
    pair = m2[:, :-1] & m2[:, 1:]                       # 左右相鄰且都可用
    pos = np.full(ny * nx, -1); pos[idx] = np.arange(idx.size)
    a = pos.reshape(ny, nx)[:, :-1][pair]
    b = pos.reshape(ny, nx)[:, 1:][pair]
    sub = np.random.default_rng(0).choice(a.size, min(4000, a.size), replace=False)
    Xa, Xb = R[:, a[sub]], R[:, b[sub]]
    ca = np.mean((Xa[OUT] * Xb[OUT]).mean(0)
                 / (Xa[OUT].std(0) * Xb[OUT].std(0) + 1e-12))
    # 隨機配對當對照:沒有空間結構的話兩者應該一樣
    rnd = np.random.default_rng(1).permutation(a[sub])
    Xr = R[:, rnd]
    cr = np.mean((Xa[OUT] * Xr[OUT]).mean(0)
                 / (Xa[OUT].std(0) * Xr[OUT].std(0) + 1e-12))
    print(f"③ 空間相關(線外通道,{sub.size:,} 對)")
    print(f"   相鄰 spaxel  {ca:+.4f}      隨機配對  {cr:+.4f}")
    print("   相鄰明顯大於隨機 → 殘差有空間結構(共同的天空/儀器成分沒扣乾淨)\n")

    # ---------- ④ spaxel 級的超額 ----------
    ex = R[OUT].std(axis=0) / np.sqrt(np.mean(Vs[OUT], axis=0))
    q = np.percentile(ex, [50, 84, 95, 99, 99.9])
    print("④ 逐 spaxel 的超額 sigma/STAT(線外)")
    print(f"   中位 {q[0]:.3f}   84% {q[1]:.3f}   95% {q[2]:.3f}   "
          f"99% {q[3]:.3f}   99.9% {q[4]:.3f}   最大 {ex.max():.3f}")
    for t in [1.5, 2.0, 3.0]:
        print(f"   超額 > {t}: {int((ex > t).sum()):,} 個 spaxel "
              f"({100*np.mean(ex > t):.2f}%)")
    print("   少數 spaxel 大幅超額 → 未偵測到的暗源或壞區,可在學 basis 前剔除")


if __name__ == "__main__":
    main()
