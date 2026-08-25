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

    conda run -n astro python src/skymodel/experiments/choose_K.py --work results/skymodel/p01
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from sklearn.decomposition import TruncatedSVD
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 判準本體放在這裡。它曾經住在 step3 的檔案裡,但 step3 從來不呼叫它 ——
# 留在 pipeline 主檔裡只會讓人以為 step3 是用 ZAP 的方式選 K 的。唯一的使用者
# 就是這支診斷程式,所以搬過來,和它畫的圖放在一起。
def zap_k(var, nsigma=5):
    """ZAP 的成分數判準。照抄 libs/zap/zap/zap.py:926 的 `_compute_deriv`。

    回傳 (K, deriv, mn1, std1),後三個是畫圖用的中間量。

    想法:把特徵值曲線(降冪)的一階差分看成「每多一條成分,還能再解釋掉多少
    變異數」。曲線一開始陡降 —— 那些成分描述的是天光線殘差,是我們要的;
    降幅逐漸趨於**線性**(二階導數歸零)之後,再往下就只是在移除雜訊與天體
    訊號。所以要找的是「降幅第一次回到平坦區水準」的那個位置。

        ① 只看前 25% 的成分 —— 後面早就進入平坦區,納進來只會稀釋統計
        ② deriv = diff(var[:npix])
        ③ 平坦區的基準取 deriv 的後 85%(跳過最前面 15% 的陡降段)
               mn1 = mean(deriv[ind:])   std1 = nsigma * std(deriv[ind:])
        ④ K = 第一個滿足 deriv >= mn1 - std1 的位置
           也就是降幅第一次不再顯著陡於平坦區

    nsigma=5 是 ZAP 的預設值,不是我們調的。門檻越鬆(nsigma 越大)K 越小。

    注意 ZAP 是**逐波長區段**各自做這件事(它把光譜切成數段,每段自己選 K),
    我們是全波段一組,所以兩邊的數字不能互相引用。
    """
    npix  = int(0.25 * var.shape[0])
    deriv = np.diff(var[:npix])
    ind   = int(0.15 * deriv.size)
    mn1   = deriv[ind:].mean()
    std1  = deriv[ind:].std() * nsigma
    # 第一個元素補 False:deriv[i] 描述的是「從第 i 條到第 i+1 條」的降幅,
    # 所以位置要往後挪一格才對得上成分編號。
    hit   = np.flatnonzero(np.append([False], deriv >= (mn1 - std1)))
    return (int(hit[0]) if hit.size else -1), deriv, mn1, std1

ROOT    = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "results/skymodel/evaluation/sky_basis"

SEED = 0


def main():
    ap = argparse.ArgumentParser(description="選 K 的三個判準")
    ap.add_argument("--work", required=True,
                    help="pointing 的工作區,例如 results/skymodel/p01")
    ap.add_argument("--cube", default=None,
                    help="含天空的 cube;預設由 pNN 的編號推出 "
                         "data/wsky/DATACUBE_FINAL_N.fits")
    ap.add_argument("--kmax", type=int, default=120, help="交叉驗證掃到多少")
    args = ap.parse_args()

    W = ROOT / args.work
    STEP01, STEP03 = W / "step01", W / "step03"
    WSKY = ROOT / (args.cube or f"data/wsky/DATACUBE_FINAL_{int(W.name[1:])}.fits")

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
    fig.suptitle(f"{W.name}: how many sky basis components?  three criteria",
                 fontsize=12)
    fig.tight_layout()
    # 檔名帶 pointing:每顆的 blank 樣本不同,K 的判準本來就可能給出不同答案,
    # 寫成同一個檔會讓後跑的那顆蓋掉前一顆。
    out = FIGURES / f"choose_K_{W.name}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
