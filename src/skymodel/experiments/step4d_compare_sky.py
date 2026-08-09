"""step4d 前後的天空模型比較 —— 把源區納入樣本,天空模型變了多少?

六格,回答六個不同的問題:

    C_sky           連續譜本身變了嗎
    C_sky 的差      變在哪些波長
    basis 前三條    最主要的天光線成分變了嗎
    主角度          兩組 basis 張開的子空間差多少(0 度 = 同一個方向)
    line mask       線與連續譜的劃分變了嗎
    殘差            擬合實際上變好了多少,分 blank 與各源

殘差用「允許係數自由調整之後還剩多少」,以 1/sigma 加權 —— 量的是形狀合不合,
不是強度合不合。blank 對兩個模型都是訓練樣本,源區則只有新模型看過,所以
源區的改善是「新模型在它新學到的地方好多少」,blank 的改善是「有沒有付出代價」。

    conda run -n astro python src/skymodel/experiments/step4d_compare_sky.py \\
        --basis svd -K 30 \\
        --best results/skymodel/step04b/classification_nobasis_s0.0_4700-8000_4700-8000_L1cum__eso.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step5_fit_spaxels import build_templates, fit_source, CUBE, STEP01, STEP03
from templates import air_to_vacuum

ROOT    = Path(__file__).resolve().parents[3]
STEP04D = ROOT / "results/skymodel/step04d"
FIGURES = ROOT / "results/skymodel/figures"

OLD_C, NEW_C = "#7f7f7f", "#d62728"


def wrms(sky, Y, Var):
    """逐 spaxel:係數自由的加權最小平方之後,殘差的 rms(以 sigma 為單位)。"""
    A, out = sky.T, []
    for j in range(Y.shape[1]):
        g = np.isfinite(Y[:, j]) & np.isfinite(Var[:, j]) & (Var[:, j] > 0)
        if g.sum() <= A.shape[1]:
            continue
        w = 1 / np.sqrt(Var[g, j].astype(np.float64))
        c, *_ = np.linalg.lstsq(A[g] * w[:, None],
                                Y[g, j].astype(np.float64) * w, rcond=None)
        out.append(np.sqrt(((((Y[g, j] - A[g] @ c)) * w) ** 2).mean()))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description="step4d 前後的天空模型比較")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--best", required=True)
    ap.add_argument("--n-blank", type=int, default=1500)
    ap.add_argument("--n-per-source", type=int, default=250)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    wl = np.load(STEP03 / "wavelength.npy")
    old = {k: np.load(STEP03 / f"{k}.npy") for k in
           ("sky_continuum", "mean_sky", "iter_line_mask")}
    new = {k: np.load(STEP04D / f"{k}.npy") for k in
           ("sky_continuum", "mean_sky", "iter_line_mask")}
    b_old = np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")
    b_new = np.load(STEP04D / f"sky_basis_{args.basis}_K{args.K}.npy")
    sky_old = np.vstack([old["sky_continuum"], b_old])
    sky_new = np.vstack([new["sky_continuum"], b_new])

    # ---- 殘差 ----
    best  = np.load(args.best)
    seg   = fits.getdata(STEP01 / "seg.fits").reshape(-1)
    white = fits.getdata(STEP01 / "whitelight.fits").reshape(-1)
    with fits.open(CUBE, memmap=True) as h:
        D = np.asarray(h["DATA"].data, np.float32)
        V = np.asarray(h["STAT"].data, np.float32)
    nz = D.shape[0]
    D, V = D.reshape(nz, -1), V.reshape(nz, -1)
    valid = (white != 0) & np.isfinite(D).all(axis=0)

    rng = np.random.default_rng(0)
    imp, labels = [], []

    bl = rng.choice(np.flatnonzero(valid & (seg == 0)), args.n_blank, replace=False)
    r_o, r_n = wrms(sky_old, D[:, bl], V[:, bl]), wrms(sky_new, D[:, bl], V[:, bl])
    imp.append(100 * (np.median(r_n) / np.median(r_o) - 1))
    labels.append("blank")
    print(f"blank    {np.median(r_o):.4f} -> {np.median(r_n):.4f}   {imp[-1]:+.2f}%",
          flush=True)

    T_all = build_templates(best, air_to_vacuum(wl))
    for t in [int(i) for i in best["id"]]:
        m = np.flatnonzero(valid & (seg == t))
        if m.size == 0:
            continue
        if m.size > args.n_per_source:
            m = rng.choice(m, args.n_per_source, replace=False)
        T = T_all[t]
        c = fit_source(D[:, m], V[:, m], sky_old, T, s_fix=1.0)
        src = T @ np.nan_to_num(c[:T.shape[1]])
        src = np.where(np.all(np.isfinite(T), axis=1)[:, None], src, np.nan)
        Y = D[:, m] - src
        a, b = wrms(sky_old, Y, V[:, m]), wrms(sky_new, Y, V[:, m])
        imp.append(100 * (np.median(b) / np.median(a) - 1))
        labels.append(f"#{t}")
        print(f"ID {t:>3}   {np.median(a):.4f} -> {np.median(b):.4f}   "
              f"{imp[-1]:+.2f}%", flush=True)

    # ---------------- 圖 ----------------
    fig, ax = plt.subplots(2, 2, figsize=(16, 8.5))

    a = ax[0][0]
    a.plot(wl, old["sky_continuum"], lw=0.8, color=OLD_C, label="step3 (blank only)")
    a.plot(wl, new["sky_continuum"], lw=0.8, color=NEW_C, label="step4d (+ source)")
    a.set_ylabel("C$_{sky}$ [$10^{-20}$]"); a.legend(fontsize=8); a.grid(alpha=0.25)
    a.set_title("sky continuum", fontsize=10)

    a = ax[0][1]
    d = new["sky_continuum"] - old["sky_continuum"]
    a.plot(wl, d, lw=0.7, color="k")
    a.axhline(0, color="0.5", lw=0.6)
    a.set_ylabel("new − old"); a.grid(alpha=0.25)
    a.set_title(f"difference   median {np.median(d):+.4f} "
                f"({100*np.median(d)/np.median(old['sky_continuum']):+.2f}%)",
                fontsize=10)

    a = ax[1][0]
    for i in range(3):
        # basis 的正負號由分解決定,不具意義 —— 對齊之後才比得出形狀。
        s = np.sign(np.dot(b_old[i], b_new[i])) or 1.0
        # 舊的畫粗、放底下,新的畫細、疊上去 —— 兩者幾乎重合,同樣粗細會讓
        # 上面那條把下面那條完全蓋掉,看起來像只畫了一條。
        a.plot(wl, b_old[i] - i * 0.6, lw=2.0, color="0.75",
               label="step3" if i == 0 else None)
        a.plot(wl, s * b_new[i] - i * 0.6, lw=0.5, color=NEW_C,
               label="step4d" if i == 0 else None)
    a.legend(fontsize=8)
    a.set_ylabel("basis 1-3 (offset)"); a.grid(alpha=0.25)
    a.set_title("first three sky-line basis vectors", fontsize=10)
    a.set_xlabel("wavelength [$\\AA$]")

    a = ax[1][1]
    col = ["0.4"] + ["#2ca02c" if v < 0 else "#d62728" for v in imp[1:]]
    a.bar(labels, imp, color=col)
    a.axhline(0, color="k", lw=0.8)
    a.set_ylabel("residual rms change [%]   (negative = better)")
    a.grid(alpha=0.25, axis="y")
    a.set_title("weighted residual rms, step4d vs step3", fontsize=10)

    fig.suptitle("does including the source regions improve the sky model?   "
                 f"[{args.basis} K={args.K}]", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out) if args.out else FIGURES / "step4d_compare_sky.png"
    fig.savefig(out, dpi=135, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
