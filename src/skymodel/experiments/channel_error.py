"""線內那點模型誤差集中在哪些通道?

線外的殘差落在雜訊底線上,線內才是我們和 ESO 有差別的地方。所以要問的是:線內
剩下的誤差是平均分布在所有線內通道上(那就是雜訊,沒得改),還是集中在少數
幾條線(那就可行動)。

逐通道對所有 blank spaxel 取兩個量,拆法不依賴 STAT:

    mean(c)   同調誤差。雜訊會被平均掉(尺度 sigma/sqrt(N)),所以顯著
              不為零的平均值只能是系統性的模型誤差 —— 某條線被統一多扣或少扣
    std(c)    雜訊 + 誤差的逐 spaxel 變化

同時對 ESO nosky 做一次:同一批通道對兩個方法都壞 → 資料本身的問題;
只對我們壞 → 我們的模型的問題。

    conda run -n astro python src/skymodel/experiments/channel_error.py
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
FIGURES = ROOT / "results/skymodel/figures"
WSKY    = ROOT / "data/Haro11_NEpointing_wsky.fits"
ESO     = ROOT / "data/Haro11_NEpointing_esonosky.fits"

CHUNK = 200

# 光學波段最亮的幾條天光線(空氣波長, A),用來認出出問題的通道
SKYLINES = [(4861.3, "Hb sky"), (5577.3, "[O I]"), (5889.9, "Na D1"),
            (5895.9, "Na D2"), (6300.3, "[O I]"), (6363.8, "[O I]"),
            (6863.9, "O2 B-band"), (7276.4, "OH"), (7340.9, "OH"),
            (7750.6, "OH"), (7913.7, "OH"), (7993.3, "OH"), (8344.6, "OH"),
            (8399.2, "OH"), (8452.3, "OH"), (8827.1, "OH"), (8885.9, "OH"),
            (8919.6, "OH"), (9001.3, "OH"), (9337.9, "OH"), (9375.9, "OH")]


def ident(w, tol=6.0):
    d = [(abs(w - l), n) for l, n in SKYLINES]
    d.sort()
    return d[0][1] if d[0][0] < tol else ""


def stats(path, idx, nz):
    """逐通道的 (平均, 標準差),分段讀以控制記憶體。"""
    mu = np.empty(nz); sd = np.empty(nz)
    with fits.open(path, memmap=True) as h:
        d = h[0].data if h[0].data is not None else h[1].data
        for j in range(0, nz, CHUNK):
            x = np.asarray(d[j:j+CHUNK], np.float64).reshape(min(CHUNK, nz-j), -1)[:, idx]
            mu[j:j+CHUNK], sd[j:j+CHUNK] = x.mean(axis=1), x.std(axis=1)
    return mu, sd


def main():
    ap = argparse.ArgumentParser(description="逐通道的模型誤差")
    ap.add_argument("--tag", default="svd_K30_s_1.0")
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    wl    = np.load(STEP03 / "wavelength.npy")
    IN    = np.load(STEP03 / "iter_line_mask.npy")[0]
    nz    = wl.size

    with fits.open(WSKY, memmap=True) as h:
        V = np.asarray(h["STAT"].data, np.float32).reshape(nz, -1)
    S = fits.getdata(STEP05 / f"sky_subtracted_{args.tag}.fits").reshape(nz, -1)
    E = fits.getdata(ESO).reshape(nz, -1)
    ok = ((white != 0) & (seg == 0)).ravel() & np.isfinite(S).all(0) \
         & np.isfinite(E).all(0) & np.isfinite(V).all(0)
    idx = np.flatnonzero(ok)
    n = idx.size
    del S, E

    stat = np.empty(nz)
    for j in range(0, nz, CHUNK):
        stat[j:j+CHUNK] = np.sqrt(V[j:j+CHUNK][:, idx].astype(np.float64).mean(axis=1))
    del V

    mo, so = stats(STEP05 / f"sky_subtracted_{args.tag}.fits", idx, nz)
    me, se = stats(ESO, idx, nz)

    err_mu = np.median(so) / np.sqrt(n)          # 平均值本身的雜訊尺度
    print(f"blank {n:,} spaxel   線內 {int(IN.sum())}/{nz} 通道")
    print(f"逐通道平均值的雜訊尺度 sigma/sqrt(N) ~ {err_mu:.4f}"
          f"   → |mean| 大於這個的都是系統性的\n")

    # ---------- 同調誤差 ----------
    print("① 同調誤差(逐通道平均值)")
    print(f"{'':>12}{'線內 |mean| 中位':>18}{'線外 |mean| 中位':>18}"
          f"{'線內最大':>11}{'>10x 雜訊的通道':>17}")
    print("-" * 78)
    for nm, m in [("我們", mo), ("ESO", me)]:
        print(f"{nm:>12}{np.median(np.abs(m[IN])):>18.4f}"
              f"{np.median(np.abs(m[~IN])):>18.4f}{np.abs(m[IN]).max():>11.4f}"
              f"{int((np.abs(m) > 10*err_mu).sum()):>17}")

    # ---------- 超額變異數的集中度 ----------
    ex = np.maximum(so ** 2 - stat ** 2, 0.0)     # 逐通道超出 STAT 的變異數
    tot = ex[IN].sum()
    order = np.argsort(ex * IN)[::-1]
    print(f"\n② 線內超額變異數的集中度(總量 {tot:,.0f})")
    for k in [1, 5, 10, 30, 100, 300]:
        print(f"   前 {k:>3} 個通道佔 {100*ex[order[:k]].sum()/tot:>5.1f}%"
              f"   (線內共 {int(IN.sum())} 個通道,均勻分布的話會是 "
              f"{100*k/IN.sum():.1f}%)")

    print(f"\n③ 線內超額最大的 {args.top} 個通道")
    print(f"{'ch':>6}{'wl (A)':>10}{'認定':>12}{'我們 sd':>10}{'ESO sd':>10}"
          f"{'STAT':>9}{'我們 mean':>11}{'ESO mean':>11}")
    print("-" * 80)
    for c in order[:args.top]:
        print(f"{c:>6}{wl[c]:>10.1f}{ident(wl[c]):>12}{so[c]:>10.3f}{se[c]:>10.3f}"
              f"{stat[c]:>9.3f}{mo[c]:>11.3f}{me[c]:>11.3f}")

    # ---------- 圖 ----------
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax[0].plot(wl, mo, lw=0.6, color="#1f77b4", label="ours")
    ax[0].plot(wl, me, lw=0.6, color="#d62728", alpha=0.7, label="ESO nosky")
    ax[0].axhline(0, color="0.4", lw=0.8)
    ax[0].set_ylabel("mean residual over blank spaxels")
    ax[0].set_title("per-channel residual: coherent error (top) and excess scatter "
                    "(bottom)", fontsize=11)
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    ax[1].plot(wl, np.sqrt(np.maximum(so**2 - stat**2, 0)), lw=0.6,
               color="#1f77b4", label="ours")
    ax[1].plot(wl, np.sqrt(np.maximum(se**2 - stat**2, 0)), lw=0.6,
               color="#d62728", alpha=0.7, label="ESO nosky")
    ax[1].set_xlabel("wavelength (A)")
    ax[1].set_ylabel("sqrt(sd² - STAT²)")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out = FIGURES / f"channel_error_{args.tag}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
