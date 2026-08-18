"""畫出 estimate_continuum 每一輪的連續譜、門檻與線遮罩。

回答的問題:
  - 遮罩為什麼一路成長?(遮掉發射線 → 連續譜下降、sigma 縮小 → 門檻下移)
  - 遮罩是逐輪累加的嗎?(不是 —— 每輪都用原始 mean_sky 重新判定)
  - 迭代停在哪裡、為什麼停(收斂 or 撞到 min_unmasked_frac 地板)

需要 step3 存下的 iter_*.npy。若還沒有,重跑一次 step3 即可:
    python src/skymodel/step3_sky_basis.py --methods svd -K 30 \
        --work <工作區> --cube <wsky cube> <該顆的 REGION>

輸出 results/skymodel/evaluation/sky_basis/linemask_iters_{pNN}/ 底下,每一輪兩張:
    iter{N}.png          被判成線的通道(橘)
    iter{N}_nonline.png  沒被判成線的通道(綠)—— 連續譜就是在這些通道上擬的

用法:
    python src/skymodel/experiments/plot_linemask_iters.py --work results/skymodel/p01
    python src/skymodel/experiments/plot_linemask_iters.py --work results/skymodel/p01 --ylim 0 120
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "results/skymodel/evaluation/sky_basis"

THRESHOLDS = (1, 2)      # (正, 負);與 step3_sky_basis.py 一致


def main():
    ap = argparse.ArgumentParser(description="畫 line_mask 每一輪的演變")
    ap.add_argument("--work", required=True,
                    help="pointing 的工作區,例如 results/skymodel/p01")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=(0, 120),
                    help="光譜圖的 y 軸範圍")
    ap.add_argument("--dpi", type=int, default=140)
    args = ap.parse_args()

    W = ROOT / args.work
    STEP03 = W / "step03"

    need = ["iter_continuum.npy", "iter_sigma.npy", "iter_line_mask.npy"]
    missing = [f for f in need if not (STEP03 / f).exists()]
    if missing:
        raise SystemExit(
            f"{STEP03} 缺少 {', '.join(missing)}。step3 必須用有存 history 的版本重跑一次:\n"
            f"  conda run -n astro python src/skymodel/step3_sky_basis.py "
            f"--methods svd -K 30 --work {args.work} --cube <wsky cube>\n"
            "  (學天空的空間範圍要一併帶上,見 configs/pNN.yaml 的 sky_region)")

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
    # 目錄名帶上工作區名稱:每顆 pointing 的 mean_sky 不同,遮罩也不同,
    # 寫進同一個目錄的話後跑的那顆會無聲蓋掉前一顆。
    out = FIGURES / f"linemask_iters_{W.name}"
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        fig.savefig(out / name, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    def panel(i, mask, colour, label, title, name):
        """一輪、一種遮罩(線 or 非線)畫一張。

        兩張圖的差別只有陰影蓋在哪一半的通道上 —— 底下的 mean_sky、連續譜、
        門檻完全相同。分開畫是因為兩者疊在一起會互相遮蔽,看不出哪一半才是
        重點。
        """
        fig, a = plt.subplots(figsize=(15, 4.5))
        # 陰影畫在資料座標之外的 y 軸座標上,不必配合 ylim
        a.fill_between(wl, 0, 1, where=mask, transform=a.get_xaxis_transform(),
                       color=colour, alpha=0.18, lw=0, label=label)
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
        a.set_title(title, fontsize=13)
        save(fig, name)

    def panel_nonline(i, name, title):
        """只有 mean_sky 和「沒被遮掉的通道」—— 不畫連續譜與門檻。

        線遮罩那張要看的是「門檻怎麼把線切出來」,所以連續譜與 ±sigma 是主角。
        這張要看的是「剩下哪些通道」,多畫三條線只會蓋住那些點。直接把未遮罩的
        通道畫成紅點疊在 mean_sky 上,點的疏密就是可用資料的分布。
        """
        keep = ~M[i]
        fig, a = plt.subplots(figsize=(15, 4.5))
        a.plot(wl, ms, lw=0.4, color="0.55", label="mean sky")
        a.plot(wl[keep], ms[keep], ".", ms=1.2, color="#ff7b7b", lw=0,
               label="not masked")
        a.set_ylim(*args.ylim)
        a.set_xlabel("observed wavelength (air) [$\\AA$]")
        a.set_ylabel("flux")
        a.legend(fontsize=9, loc="upper right", markerscale=7)
        a.set_title(title, fontsize=13)
        save(fig, name)

    for i in range(n_iter):
        n_line = int(M[i].sum())
        panel(i, M[i], "orange", "masked as line",
              f"iteration {i+1}: {n_line} lines", f"iter{i+1}.png")
        panel_nonline(i, f"iter{i+1}_nonline.png",
                      f"iteration {i+1}: {M[i].size - n_line} non-lines")

    print(f"\nsaved {2 * n_iter} figures -> {out}")


if __name__ == "__main__":
    main()
