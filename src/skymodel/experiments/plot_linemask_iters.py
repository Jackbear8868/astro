"""畫出 estimate_continuum 每一輪的連續譜、門檻與線遮罩。

回答的問題:
  - 遮罩為什麼一路成長?(遮掉發射線 → 連續譜下降、sigma 縮小 → 門檻下移)
  - 遮罩是逐輪累加的嗎?(不是 —— 每輪用原始 mean_sky 重算,少數通道會被放回)
  - 迭代停在哪裡、為什麼停(收斂 or 撞到 min_unmasked_frac 地板)

需要 step3 存下的 iter_*.npy。若還沒有,重跑一次 step3 即可:
    python src/skymodel/step3_sky_basis.py --methods svd

輸出 results/skymodel/figures/linemask_iters/ 底下:
    iter1.png …    每一輪:mean_sky + 連續譜 + 上下門檻 + 遮罩
    summary.png    遮罩比例、連續譜中位數、sigma 中位數隨迭代的變化

用法:
    python src/skymodel/experiments/plot_linemask_iters.py
    python src/skymodel/experiments/plot_linemask_iters.py --ylim 0 120
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

THRESHOLDS = (1, 2)      # (正, 負);與 step3_sky_basis.py 一致


def main():
    ap = argparse.ArgumentParser(description="畫 line_mask 每一輪的演變")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=(0, 120),
                    help="光譜圖的 y 軸範圍")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    need = ["iter_continuum.npy", "iter_sigma.npy", "iter_line_mask.npy"]
    missing = [f for f in need if not (STEP03 / f).exists()]
    if missing:
        raise SystemExit(
            f"缺少 {', '.join(missing)}。step3 必須用有存 history 的版本重跑一次:\n"
            f"  conda run -n astro python src/skymodel/step3_sky_basis.py --methods svd")

    wl = np.load(STEP03 / "wavelength.npy")
    ms = np.load(STEP03 / "mean_sky.npy")
    C  = np.load(STEP03 / "iter_continuum.npy")     # (n_iter, nz)
    S  = np.load(STEP03 / "iter_sigma.npy")
    M  = np.load(STEP03 / "iter_line_mask.npy")
    n_iter = M.shape[0]

    # ---------------- 診斷表 ----------------
    print(f"{n_iter} 輪迭代   {wl.size} 通道 ({wl.min():.1f}-{wl.max():.1f} A air)\n")
    print(f"{'輪':>3}{'遮罩通道':>10}{'比例':>9}{'新增':>8}{'移除':>8}"
          f"{'continuum 中位數':>18}{'sigma 中位數':>14}")
    print("-" * 72)
    for i in range(n_iter):
        if i == 0:
            add = rem = "—"
        else:
            add = int((M[i] & ~M[i-1]).sum())
            rem = int((~M[i] & M[i-1]).sum())
        print(f"{i+1:>3}{M[i].sum():>10}{100*M[i].mean():>8.1f}%{add:>8}{rem:>8}"
              f"{np.median(C[i]):>18.3f}{np.median(S[i]):>14.4f}")

    union = np.logical_or.reduce(M)
    print(f"\n各輪聯集 = 最後一輪嗎: {np.array_equal(union, M[-1])}"
          f"   (聯集 {union.sum()} vs 最後一輪 {M[-1].sum()})")
    print("→ 不相等代表遮罩不是逐輪累加,每輪都用原始 mean_sky 重新判定。")

    # ---------------- 每一輪一張圖 ----------------
    out = FIGURES / "linemask_iters"
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(out / name, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    for i in range(n_iter):
        fig, a = plt.subplots(figsize=(15, 4.5))
        # 陰影畫在資料座標之外的 y 軸座標上,不必配合 ylim
        a.fill_between(wl, 0, 1, where=M[i], transform=a.get_xaxis_transform(),
                       color="orange", alpha=0.15, lw=0, label="masked as line")
        a.plot(wl, ms,   lw=0.4, color="0.35", label="mean sky")
        a.plot(wl, C[i], lw=0.9, color="#1f77b4", label="continuum")
        a.plot(wl, C[i] + THRESHOLDS[0] * S[i], lw=0.7, color="#d62728",
               label=f"+{THRESHOLDS[0]}$\\sigma$")
        a.plot(wl, C[i] - THRESHOLDS[1] * S[i], lw=0.7, color="#9467bd",
               label=f"-{THRESHOLDS[1]}$\\sigma$")
        a.set_ylim(*args.ylim)
        a.set_xlabel("observed wavelength (air) [$\\AA$]")
        a.set_ylabel("flux")
        a.legend(fontsize=8, loc="upper right", ncol=2)
        a.set_title(f"iteration {i+1}/{n_iter}   masked {M[i].sum()}/{M[i].size} "
                    f"({100*M[i].mean():.1f}%)   "
                    f"continuum med {np.median(C[i]):.2f}   "
                    f"sigma med {np.median(S[i]):.3f}")
        save(fig, f"iter{i+1}.png")

    # ---------------- 收斂總覽 ----------------
    it = np.arange(1, n_iter + 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(it, 100 * M.mean(axis=1), "o-", color="#ff7f0e")
    ax[0].axhline(100 * (1 - 0.16), ls="--", lw=0.8, color="0.5",
                  label="min_unmasked_frac floor (84%)")
    ax[0].set_ylabel("masked channels [%]"); ax[0].legend(fontsize=7)

    ax[1].plot(it, np.median(C, axis=1), "o-", color="#1f77b4")
    ax[1].set_ylabel("continuum median")

    ax[2].plot(it, np.median(S, axis=1), "o-", color="#d62728")
    ax[2].set_ylabel("sigma median"); ax[2].set_yscale("log")

    for a in ax:
        a.set_xlabel("iteration"); a.set_xticks(it); a.grid(alpha=0.25)
    # 標題只用 ASCII —— DejaVu Sans 沒有中日韓字形,中文會變成方框
    fig.suptitle("masking lines -> continuum drops & sigma shrinks "
                 "-> thresholds move down -> mask grows", fontsize=10)
    fig.tight_layout()
    save(fig, "summary.png")

    print(f"\nsaved {n_iter + 1} figures -> {out}")


if __name__ == "__main__":
    main()
