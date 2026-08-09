"""K 是怎麼決定的 —— 把 ZAP 的判準畫出來。

天光線 basis 要留幾條(K)沒有客觀的最佳解:它是「把天光線扣乾淨」和「不要
吃掉天體訊號」之間的取捨點。我們採用的是 ZAP 自己的判準(Soto+ 2016),
理由是它是這個方法的原作者定義的,不是我們調出來的。

判準的想法(實作見 step3_sky_basis.zap_k,照抄 zap.py:926 的 _compute_deriv):

    把特徵值曲線的**一階差分**看成「每多一條成分,還能再解釋掉多少變異數」。
    曲線一開始陡降 —— 那些成分描述的是天光線殘差,是我們要的。降幅逐漸趨於
    線性(二階導數歸零)之後,再往下就只是在移除雜訊與天體訊號。
    所以要找的是「降幅第一次回到平坦區水準」的那個位置。

這張圖把那三步攤開:
    左   特徵值曲線本身(scree),K 標在上面
    中   一階差分、平坦區基準 mn1、5 sigma 帶、以及第一個穿越點
    右   穿越點附近放大,看清楚它是怎麼被判定的

    conda run -n astro python src/skymodel/experiments/step3_zap_k.py
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
from step3_sky_basis import zap_k

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP03  = ROOT / "results/skymodel/step03"
FIGURES = ROOT / "results/skymodel/figures"
WSKY    = ROOT / "data/Haro11_NEpointing_wsky.fits"


def main():
    ap = argparse.ArgumentParser(description="畫出 ZAP 判準怎麼決定 K")
    ap.add_argument("--nsigma", type=float, default=5.0,
                    help="ZAP 的門檻,預設 5 是 ZAP 自己的值")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # 和 step3 完全相同的樣本與前處理 —— 判準必須算在真正拿去分解的那組殘差上,
    # 換一組資料就是換一個問題,圖會失去說明力。
    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    C_sky = np.load(STEP03 / "sky_continuum.npy")
    with fits.open(WSKY, memmap=True) as h:
        nz = h["DATA"].header["NAXIS3"]
        D  = np.asarray(h["DATA"].data, np.float32).reshape(nz, -1)
    blank = ((white != 0) & (seg == 0)).reshape(-1) & np.isfinite(D).all(axis=0)
    X = (D[:, blank] - C_sky[:, None]).T.astype(np.float32)
    print(f"blank spaxel {int(blank.sum()):,}   通道 {nz}")

    X = X - X.mean(axis=0)
    sv = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    var = sv ** 2 / (X.shape[0] - 1)          # explained_variance_,降冪

    K, deriv, mn1, std1 = zap_k(var, args.nsigma)
    npix = int(0.25 * var.shape[0])
    ind  = int(0.15 * deriv.size)
    print(f"前 25% = {npix} 條成分   平坦區基準取 deriv[{ind}:]")
    print(f"mn1 = {mn1:.4f}   {args.nsigma:g} sigma = {std1:.4f}   "
          f"門檻 deriv >= {mn1 - std1:.4f}")
    print(f"→ K = {K}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    a = ax[0]
    a.semilogy(np.arange(1, npix + 1), var[:npix], color="#1f77b4", lw=1.2)
    a.axvline(K, color="#d62728", ls="--", lw=1.2, label=f"ZAP criterion: K = {K}")
    a.set_xlabel("component"); a.set_ylabel("explained variance")
    a.set_title("① scree curve", fontsize=10)
    a.legend(fontsize=9); a.grid(alpha=0.25)

    a = ax[1]
    x = np.arange(1, deriv.size + 1)
    a.plot(x, deriv, color="#1f77b4", lw=1.0, label="deriv = diff(variance)")
    a.axhline(mn1, color="#7f7f7f", lw=1.0, label=f"flat-region mean = {mn1:.3f}")
    a.axhspan(mn1 - std1, mn1 + std1, color="#7f7f7f", alpha=0.18,
              label=f"$\\pm{args.nsigma:g}\\sigma$")
    a.axhline(mn1 - std1, color="#7f7f7f", ls=":", lw=1.0)
    a.axvline(K, color="#d62728", ls="--", lw=1.2)
    a.axvspan(0, ind, color="#ffcc80", alpha=0.35,
              label=f"skipped for the baseline (first 15%)")
    # symlog:第一個差分是 -1.07e7,線性軸會把其餘 949 個點全部壓成一條零線,
    # 而判準要看的正是那 949 個點怎麼趨於平坦。linthresh 取 1 —— 平坦區的
    # 尺度是個位數,小於 1 的部分用線性顯示才不會把零附近撐開成噪音。
    a.set_yscale("symlog", linthresh=1)
    a.set_xlabel("component"); a.set_ylabel("$\\Delta$ explained variance (symlog)")
    a.set_title("② first difference: where does the drop stop steepening?",
                fontsize=10)
    a.legend(fontsize=8, loc="lower right"); a.grid(alpha=0.25)

    a = ax[2]
    lo, hi = max(0, K - 25), min(deriv.size, K + 25)
    a.plot(x[lo:hi], deriv[lo:hi], "o-", color="#1f77b4", ms=3, lw=1.0)
    a.axhline(mn1, color="#7f7f7f", lw=1.0)
    a.axhline(mn1 - std1, color="#7f7f7f", ls=":", lw=1.2,
              label=f"threshold = mn1 $-$ {args.nsigma:g}$\\sigma$")
    a.axvline(K, color="#d62728", ls="--", lw=1.2, label=f"first crossing → K = {K}")
    a.plot([K], [deriv[K - 1]], "o", color="#d62728", ms=8, zorder=5)
    a.set_xlabel("component"); a.set_ylabel("$\\Delta$ explained variance")
    a.set_title("③ zoom on the crossing", fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.25)

    fig.suptitle("how K is chosen: the ZAP criterion "
                 "(Soto+ 2016, zap.py `_compute_deriv`)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.out) if args.out else FIGURES / "step3_zap_k.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
