"""畫出 step3 學到的 K 條天空 basis,一條一張圖,全部存進一個資料夾。

回答的問題:
  - 每條 basis 的結構落在哪些波長(和 mean_sky.png 對照)
  - 條數夠不夠(後面幾條若只剩雜訊,代表超過真實自由度)
  - 有沒有 basis 被浪費掉(能量全集中在單一通道 = 壞畫素,不是天光線)
  - 不同分解法的差異:svd 的右奇異向量單位正交(每條 norm = 1),
    pca 的第 0 條是平均光譜(未正規化),其餘才是主成分

輸出 results/skymodel/evaluation/sky_basis/basis_{方法}/ 底下:
    mean_sky.png   平均天空與連續譜,對照用
    e00.png …      每條 basis 一張

用法:
    python src/skymodel/tools/plot_basis.py --basis svd
    python src/skymodel/tools/plot_basis.py --basis svd --ylim -0.1 0.3
    python src/skymodel/tools/plot_basis.py --basis pca --ylim-sky 0 60
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[3]
STEP03  = ROOT / "results/skymodel/ne_pointing/step03"
FIGURES = ROOT / "results/skymodel/evaluation/sky_basis"


def main():
    ap = argparse.ArgumentParser(description="畫 step3 學到的天空 basis")
    ap.add_argument("--basis", default="svd", help="pca / svd / nmf / rpca")
    ap.add_argument("-K", type=int, default=25, help="basis 條數;要和 step3 相同")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="basis 圖的 y 軸範圍;不給則每張自動縮放")
    ap.add_argument("--ylim-sky", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="mean_sky 圖的 y 軸範圍;不給則 0 到 99.5 百分位")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    wl = np.load(STEP03 / "wavelength.npy")
    B  = np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")
    C  = np.load(STEP03 / "sky_continuum.npy")
    lm = np.load(STEP03 / "line_mask.npy")
    K  = B.shape[0]          # 不寫死 10 —— step3 的 -K 可以改條數

    # --- 診斷:能量集中度揭露被浪費在壞畫素上的 basis ---
    print(f"basis={args.basis}   K={K}   {wl.size} channels "
          f"({wl.min():.1f}-{wl.max():.1f} A air)")
    print(f"line_mask {lm.sum()}/{lm.size} ({100*lm.mean():.1f}%)\n")
    print(f"{'條':>3}{'L2 norm':>11}{'峰值':>10}{'負值':>8}"
          f"{'90% 能量所在通道數':>20}{'峰值波長':>12}")
    print("-" * 66)
    for k in range(K):
        e   = np.sort(B[k] ** 2)[::-1]
        n90 = int(np.searchsorted(np.cumsum(e), 0.9 * e.sum()) + 1)
        i   = int(np.argmax(np.abs(B[k])))
        print(f"{k:>3}{np.linalg.norm(B[k]):>11.4g}{np.abs(B[k]).max():>10.4g}"
              f"{100.0 * (B[k] < 0).mean():>7.0f}%{n90:>16}{wl[i]:>12.1f}")
    print("\n90% 能量只落在個位數通道 = 那條 basis 被單一壞畫素佔走,不是天光線。")

    # ---------------- 圖:一條 basis 一張,全部進同一個資料夾 ----------------
    ms  = np.load(STEP03 / "mean_sky.npy")
    out = FIGURES / f"basis_{args.basis}_K{args.K}"
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(out / name, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)                       # 不關會累積,K 大時吃光記憶體

    fig, a = plt.subplots(figsize=(15, 4))
    a.plot(wl, ms, lw=0.4, color="0.3", label="mean sky (blank)")
    a.plot(wl, C,  lw=0.8, color="#d62728", label="continuum $C_{sky}$")
    # 天光線峰值比連續譜高幾十倍,自動範圍會把 18-30 的連續譜壓成一條線
    a.set_ylim(*(args.ylim_sky if args.ylim_sky else (0, np.percentile(ms, 99.5))))
    a.set_xlabel("observed wavelength (air) [$\\AA$]")
    a.set_ylabel("flux")
    a.legend(fontsize=8, loc="upper right")
    a.set_title(f"mean sky   basis={args.basis}   K={K}   "
                f"line_mask {lm.sum()}/{lm.size} ({100*lm.mean():.1f}%)")
    save(fig, "mean_sky.png")

    for k in range(K):
        b    = B[k].astype(np.float64)
        flat = b.mean() ** 2 * b.size / (b ** 2).sum()   # 一個常數能解釋多少能量

        fig, a = plt.subplots(figsize=(15, 4))
        a.plot(wl, b, lw=0.5, color="#1f77b4")
        a.axhline(0, color="0.6", lw=0.5)    # 零線:一眼看出這組 basis 跨不跨零
        # if args.ylim:
        #     a.set_ylim(*args.ylim)
        a.set_ylim(np.min(b), np.max(b)*1.1)
        a.set_xlabel("observed wavelength (air) [$\\AA$]")
        a.set_ylabel(f"$e_{{{k}}}$")
        # 標題只用 ASCII —— DejaVu Sans 沒有中日韓字形,中文會變成方框
        save(fig, f"e{k:02d}.png")

    print(f"\nsaved {K + 1} figures -> {out}")


if __name__ == "__main__":
    main()