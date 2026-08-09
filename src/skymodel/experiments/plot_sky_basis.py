"""天光線 basis 長什麼樣 —— K 條成分一次看完。

54 條 3801 通道的曲線疊在一起是看不懂的,所以拆成兩種看法,各自回答不同的問題:

    上   全部 K 條畫成一張影像(成分 x 波長),用發散色階、以 0 為中心。
         回答「哪些成分在哪些波長有力量」—— 天光線的位置會變成一條條垂直亮紋,
         而如果某條成分整條都很平,那它多半在描述雜訊而不是天光線。
    下   前幾條各自畫出來、上下錯開。
         回答「單一條成分的形狀」—— 影像看得出位置,看不出正負與線型。

色階用百分位而不是最大值:第一條成分的振幅比後面大一個量級,用最大值當上限
會讓其餘 50 幾條全部變成同一個顏色。

    conda run -n astro python src/skymodel/experiments/plot_sky_basis.py \\
        --basis results/skymodel/_backup_prof2sigma/step03/sky_basis_svd_K54.npy
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[3]
STEP03  = ROOT / "results/skymodel/step03"
FIGURES = ROOT / "results/skymodel/figures"


def main():
    ap = argparse.ArgumentParser(description="畫出天光線 basis")
    ap.add_argument("--basis", default=str(STEP03 / "sky_basis_svd_K54.npy"))
    ap.add_argument("--wavelength", default=None,
                    help="預設用 basis 同一個資料夾裡的 wavelength.npy")
    ap.add_argument("--n-trace", type=int, default=6, help="下方單獨畫幾條")
    ap.add_argument("--per-component", action="store_true",
                    help="每條成分各存一張 eNN.png,放進 figures/step3_basis/"
                         "basis_{method}_K{K}/。沿用舊 pipeline 的命名 —— 那批圖的"
                         "產生程式在 2026-07-23 重構時被刪了,只留下產物")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bp = Path(args.basis)
    B  = np.load(bp)
    wl = np.load(Path(args.wavelength) if args.wavelength else bp.parent / "wavelength.npy")
    K  = B.shape[0]
    print(f"{bp}  {B.shape}   波長 {wl[0]:.1f}-{wl[-1]:.1f} A")

    # SVD 的正負號不具意義(分解本身的自由度)。統一成「最大絕對值為正」,
    # 不然同一條成分在不同 run 之間可能整條翻轉,看起來像變了。
    B = B * np.sign(B[np.arange(K), np.abs(B).argmax(axis=1)])[:, None]

    if args.per_component:
        # 名稱從 basis 檔名還原:sky_basis_svd_K54.npy -> basis_svd_K54。
        # 但檔名不含「這是哪一版 seg 跑的」,不同 seg 的同 K 會撞名而互相覆蓋 ——
        # 需要區分時用 --out 指定資料夾。
        d = (Path(args.out) if args.out
             else FIGURES / "step3_basis" / bp.stem.replace("sky_basis_", "basis_"))
        d.mkdir(parents=True, exist_ok=True)
        # y 範圍全部成分共用 —— 各自 autoscale 的話,一條只有雜訊的成分會被
        # 放大到和真正的天光線成分看起來一樣強,逐張翻閱時完全誤導。
        v = float(np.percentile(np.abs(B), 99.9)) * 1.15
        for i in range(K):
            f, a = plt.subplots(figsize=(13, 3.2))
            a.plot(wl, B[i], lw=0.6, color="#1f77b4")
            a.axhline(0, color="0.6", lw=0.6)
            a.set_ylim(-v, v)
            a.set_xlim(wl[0], wl[-1])
            a.set_xlabel("wavelength [$\\AA$]")
            a.set_ylabel("amplitude")
            a.set_title(f"{d.name}   component {i + 1} / {K}", fontsize=10)
            a.grid(alpha=0.25)
            f.tight_layout()
            f.savefig(d / f"e{i:02d}.png", dpi=110, bbox_inches="tight")
            plt.close(f)
        print(f"saved {K} 張 -> {d}")
        return

    fig, ax = plt.subplots(2, 1, figsize=(15, 10),
                           gridspec_kw={"height_ratios": [1.35, 1]})

    v = float(np.percentile(np.abs(B), 99))
    im = ax[0].imshow(B, aspect="auto", origin="lower", cmap="RdBu_r",
                      vmin=-v, vmax=v,
                      extent=[wl[0], wl[-1], 0.5, K + 0.5], interpolation="nearest")
    ax[0].set_ylabel("component")
    ax[0].set_title(f"all {K} sky-line basis vectors", fontsize=11)
    plt.colorbar(im, ax=ax[0], fraction=0.025, pad=0.01, label="amplitude")

    off = 2.2 * np.percentile(np.abs(B[:args.n_trace]), 99)
    for i in range(min(args.n_trace, K)):
        ax[1].plot(wl, B[i] - i * off, lw=0.6,
                   color=plt.cm.viridis(i / max(args.n_trace - 1, 1)))
        ax[1].text(wl[0] + 30, -i * off + 0.35 * off, f"#{i + 1}", fontsize=8,
                   color=plt.cm.viridis(i / max(args.n_trace - 1, 1)))
    ax[1].set_yticks([])
    ax[1].set_xlabel("wavelength [$\\AA$]")
    ax[1].set_ylabel("first components (offset)")
    ax[1].grid(alpha=0.25, axis="x")

    fig.suptitle(f"sky-line basis: {bp.parent.parent.name}/{bp.parent.name}/{bp.name}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.out) if args.out else FIGURES / f"sky_basis_{bp.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
