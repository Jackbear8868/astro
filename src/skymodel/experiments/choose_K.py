"""K 該取多少?三個判準的對照。

判準 1  ZAP 的做法(libs/zap/zap/zap.py:926 `_compute_deriv`)
    看 explained_variance_ 曲線的一階差分。曲線還在陡降時,成分描述的是
    天光線殘差;一旦降幅變成線性(二階導數歸零),再往下就是在移除天體訊號。
    實作:取前 25% 的成分算差分,用其中第 15% 之後那段當「平坦區」的基準,
    找第一個差分回到 (平均 − 5 sigma) 以上的位置。
    注意 ZAP 是逐波長區段做的,我們是全波段一組,所以數字不能直接互相引用。

判準 2  交叉驗證(直接量測)
    spaxel 隨機分半,basis 只用 train 學,殘差只在 test 上算。
    這直接回答「多學一條對沒看過的 spaxel 有沒有幫助」。

判準 3  每多一條成分,降低的變異數有沒有超過雜訊
    先用 STAT 把每個通道白化,再看特徵值。純雜訊的特徵值有一個平台
    (Marchenko-Pastur);特徵值掉到平台上就代表那條成分和雜訊分不出來。
    lambda_k / lambda_noise 就是「這條成分帶的訊號是雜訊的幾倍」。

    conda run -n astro python src/skymodel/experiments/choose_K.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from sklearn.decomposition import TruncatedSVD
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 判準的實作在 pipeline 裡(step3_sky_basis.zap_k),這裡只是呼叫它。
# 各留一份的話,改了 pipeline 而診斷程式還在用舊版,兩邊會靜靜地不一致。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step3_sky_basis import zap_k

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP03  = ROOT / "results/skymodel/step03"
FIGURES = ROOT / "results/skymodel/figures"
WSKY    = ROOT / "data/Haro11_NEpointing_wsky.fits"

SEED = 0


def main():
    ap = argparse.ArgumentParser(description="選 K 的三個判準")
    ap.add_argument("--kmax", type=int, default=120, help="交叉驗證掃到多少")
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    lm    = np.load(STEP03 / "iter_line_mask.npy")[0]
    C_sky = np.load(STEP03 / "sky_continuum.npy")

    with fits.open(WSKY, memmap=True) as h:
        D = np.asarray(h["DATA"].data, np.float32)
        V = np.asarray(h["STAT"].data, np.float32)
    nz, ny, nx = D.shape
    D, V = D.reshape(nz, -1), V.reshape(nz, -1)
    ok = ((white != 0).ravel() & (seg.ravel() == 0)
          & np.isfinite(D).all(axis=0) & np.isfinite(V).all(axis=0))
    idx = np.flatnonzero(ok)
    X = (D[:, idx] - C_sky[:, None]).astype(np.float64)     # (nz, n) 殘差
    nsig = np.sqrt(V[:, idx].astype(np.float64).mean(axis=1))   # 每通道的雜訊
    del D, V
    n = idx.size
    print(f"blank {n:,} 個 spaxel x {nz} 通道\n")

    # ---------- 全部特徵值(一次算完,三個判準共用) ----------
    # 直接對 (nz x nz) 的共變異數做特徵分解,比對 (n x nz) 做完整 SVD 便宜得多,
    # 而 explained_variance_ 本來就是共變異數的特徵值。
    Xc = X - X.mean(axis=1, keepdims=True)
    cov = (Xc @ Xc.T) / (n - 1)
    ev = np.linalg.eigvalsh(cov)[::-1]                       # 由大到小
    ev = np.maximum(ev, 0)

    k_zap, deriv, mn1, std1 = zap_k(ev)
    print(f"判準 1  ZAP 的差分法(全部 {ev.size} 條成分)")
    print(f"        平坦區基準 mean {mn1:.4g}   5 sigma {std1:.4g}")
    print(f"        → K = {k_zap}\n")

    # ---------- 判準 3:白化之後的特徵值 vs 雜訊平台 ----------
    Xw = Xc / nsig[:, None]
    covw = (Xw @ Xw.T) / (n - 1)
    evw = np.maximum(np.linalg.eigvalsh(covw)[::-1], 0)
    plateau = np.median(evw[int(0.5 * nz):])                 # 後半段 = 純雜訊平台
    ratio = evw / plateau
    k_noise = int(np.searchsorted(-ratio, -2.0))             # 訊號 >= 2x 雜訊
    print(f"判準 3  白化後的特徵值 vs 雜訊平台")
    print(f"        雜訊平台 {plateau:.4g}   (理想值 1.0 → 實測倍率 "
          f"{np.sqrt(plateau):.3f}x STAT)")
    print(f"        lambda_k / 平台 >= 2 的成分數 → K = {k_noise}")
    print(f"        {'k':>5}{'lambda/平台':>14}")
    for k in (1, 5, 10, 20, 25, 30, 35, 40, 50, 60, 80, 100):
        if k <= nz:
            print(f"        {k:>5}{ratio[k-1]:>14.2f}")
    print()

    # ---------- 判準 2:交叉驗證 ----------
    rng = np.random.default_rng(SEED)
    tr = rng.random(n) < 0.5
    Dtr = (X[:, tr] + C_sky[:, None])
    Dte = (X[:, ~tr] + C_sky[:, None])
    Ks = sorted(set([1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 80, args.kmax]))
    Ks = [k for k in Ks if k <= args.kmax]
    big = TruncatedSVD(n_components=max(Ks), random_state=SEED).fit(
        (Dtr - C_sky[:, None]).T)
    comp = big.components_                                   # 巢狀:前 K 條就是 K 的解
    print("判準 2  交叉驗證(basis 只用 train 學,殘差在 test 上算)")
    print(f"{'K':>5}{'線內 sigma':>13}{'線外 sigma':>13}{'rms':>10}"
          f"{'每加一條的增益':>16}")
    prev = None
    curve = []
    for k in Ks:
        sky = np.vstack([C_sky, comp[:k]])
        c = np.linalg.pinv(sky.T) @ Dte
        Rte = Dte - sky.T @ c
        si, so = np.median(Rte[lm].std(0)), np.median(Rte[~lm].std(0))
        rm = np.median(np.sqrt((Rte ** 2).mean(0)))
        gain = "" if prev is None else f"{(prev - rm) / (k - prevk) * 1000:>13.3f}/千"
        print(f"{k:>5}{si:>13.4f}{so:>13.4f}{rm:>10.4f}{gain:>16}")
        curve.append((k, si, so, rm))
        prev, prevk = rm, k

    # ---------- 圖 ----------
    FIGURES.mkdir(parents=True, exist_ok=True)
    cur = np.array(curve)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].semilogy(np.arange(1, 201), ev[:200], lw=1.2)
    ax[0].axvline(k_zap, color="#d62728", ls="--",
                  label=f"ZAP criterion: K = {k_zap}")
    ax[0].set_xlabel("component"); ax[0].set_ylabel("explained variance")
    ax[0].set_title("scree curve (ZAP criterion)", fontsize=10)
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3)

    ax[1].plot(np.arange(1, 201), ratio[:200], lw=1.2)
    ax[1].axhline(1.0, color="0.4", ls=":", label="noise plateau")
    ax[1].axhline(2.0, color="#d62728", ls="--", label=f"2x noise: K = {k_noise}")
    ax[1].set_yscale("log"); ax[1].set_xlabel("component")
    ax[1].set_ylabel("eigenvalue / noise plateau")
    ax[1].set_title("whitened eigenvalues vs noise", fontsize=10)
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)

    ax[2].plot(cur[:, 0], cur[:, 1], "o-", lw=1.2, label="line region sigma")
    ax[2].plot(cur[:, 0], cur[:, 3], "s-", lw=1.2, label="rms")
    ax[2].set_xlabel("K"); ax[2].set_ylabel("test-set residual")
    ax[2].set_title("cross-validation (held-out spaxels)", fontsize=10)
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3)
    fig.suptitle("how many sky basis components?  three criteria", fontsize=12)
    fig.tight_layout()
    out = FIGURES / "choose_K.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
