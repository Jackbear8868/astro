"""紅移對不上的源:我們的解 vs 教授給的 z,同一條光譜上並排。

只回報「差了多少」沒有用,要看的是**為什麼擬合偏好我們那個解**。所以同一個源
畫兩次:上半用我們的 z,下半用教授的 z(振幅在該 z 下重新解到最佳),並且把
兩個 z 各自預期的譜線位置用垂直虛線標出來。

判讀的方式:一個 z 若是對的,它預言的**每一條**強線都要在資料上找得到;
只要有一條該出現卻沒出現(殘差在那裡出現一根朝下的尖峰),那個 z 就被否定
了。反過來,如果教授的 z 預言的線都在,只是深度不夠,那就是模板的表達力
不足,不是紅移錯。

    conda run -n astro python src/skymodel/experiments/step4b_mismatch.py \\
        --spec-dir results/skymodel/ne_pointing/step02_eso --s-fix 0 \\
        --star-window 4700 8000 --gal-window 4700 8000 --iter 1 --ids 12 27 30
"""
import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from templates import load_eigen_galaxy, load_sdss_template, redshift_to_grid
from step4_fit_source import STAR_WINDOW, GAL_WINDOW, make_tag, EIGEN_GAL, TPL_DIR
from step4b_stage1 import load_common, source_spectrum

ROOT    = Path(__file__).resolve().parents[3]
STEP04 = ROOT / "results/skymodel/ne_pointing/step04"
FIGURES = ROOT / "results/skymodel/evaluation/template_fit"

PROF_Z = {1: 0.021, 10: 0.99, 12: 1.033, 13: 0.422, 14: 0.336, 24: 1.055,
          27: 1.349, 28: 1.033, 30: 0.295, 34: 0.605}

OURS_C, GT_C = "#d62728", "#2ca02c"


def model_at(sc, z_target, lam_vac):
    """取掃描結果裡最接近 z_target 的那一格,重建模型光譜。

    掃描已經在每個 z 上把振幅解到最佳,所以「在指定 z 的最佳擬合」不需要重算,
    直接查表即可 —— 也保證和 chi2 曲線上的值完全一致。
    """
    j = int(np.argmin(np.abs(sc["z"] - z_target)))
    z = float(sc["z"][j])
    nm = str(sc["template"][j])
    if nm == "eigen":
        T = redshift_to_grid(load_eigen_galaxy(EIGEN_GAL), z, lam_vac)
        m = T @ np.asarray(sc["A"][j][:T.shape[1]], float)
    else:
        spl = load_sdss_template(TPL_DIR / f"spDR2-{int(nm):03d}.fit")
        m = float(sc["A"][j][0]) * redshift_to_grid(spl, z, lam_vac)
    return z, m, float(sc["red_chi2"][j])


def main():
    ap = argparse.ArgumentParser(description="紅移不合的源:我們的 z vs 教授的 z")
    ap.add_argument("--star-window", type=float, nargs=2, default=STAR_WINDOW)
    ap.add_argument("--gal-window", type=float, nargs=2, default=GAL_WINDOW)
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--sky-basis", action="store_true")
    ap.add_argument("--aperture", action="store_true")
    ap.add_argument("--raw-mask", action="store_true")
    ap.add_argument("--iter", type=int, default=1)
    ap.add_argument("--s-fix", type=float, default=0.0)
    ap.add_argument("--spec-dir", default=None)
    ap.add_argument("--gal-model", choices=["eigen", "sdss"], default="eigen")
    ap.add_argument("--ids", type=int, nargs="+", default=[12, 27, 30])
    args = ap.parse_args()

    suffix = (f"_{Path(args.spec_dir).name.replace('step02', '')}"
              if args.spec_dir else "") \
             + ("_galtpl" if args.gal_model == "sdss" else "")
    tag = make_tag(args.basis, args.K, args.s_fix, args.star_window,
                   args.gal_window, args.sky_basis, args.iter,
                   not args.raw_mask, args.aperture, suffix)
    D = load_common(Namespace(**vars(args)))
    wl = D["wl_air"]

    for t in args.ids:
        f2 = STEP04 / f"scan2_id{t}_{tag}.npz"
        if not f2.exists():
            print(f"ID {t}: 找不到 {f2.name},跳過")
            continue
        sc = np.load(f2)
        y, sig, ok = source_spectrum(D, t, args.s_fix)

        z_ours, m_ours, r_ours = model_at(sc, float(sc["z"][np.argmin(sc["chi2"])]),
                                          D["wl_vac"])
        z_gt,   m_gt,   r_gt   = model_at(sc, PROF_Z[t], D["wl_vac"])

        # 五列:兩個模型各自配資料、sigma、兩個殘差疊在一起、累積 Δchi2。
        # sigma 緊接在流量之後 —— 它是把流量殘差換算成 sigma 單位的分母,
        # 擺在兩者中間,順序就和讀圖的順序一致。
        fig, ax = plt.subplots(5, 1, figsize=(16, 11.5), sharex=True,
                               gridspec_kw=dict(height_ratios=[3, 3, 1, 2, 1.7],
                                                hspace=0.07))
        # 兩個流量列共用 y 範圍,兩個殘差列也共用 —— 不共用的話眼睛比的是軸的
        # 縮放而不是資料。
        q = np.nanpercentile(np.concatenate([y[ok], m_ours[ok], m_gt[ok]]),
                             [0.5, 99.5])
        pad = 0.25 * (q[1] - q[0]) + 1e-6
        with np.errstate(invalid="ignore", divide="ignore"):
            res_o, res_g = (y - m_ours) / sig, (y - m_gt) / sig
        rr = float(np.nanpercentile(
            np.concatenate([np.abs(res_o[ok]), np.abs(res_g[ok])]), 99.5)) * 1.25

        info = [(z_ours, m_ours, r_ours, OURS_C, "our fit"),
                (z_gt,   m_gt,   r_gt,   GT_C,   "gt")]

        # ---- 第 1、2 列:各自配資料 ----
        for k, (z, m, r, col, who) in enumerate(info):
            a = ax[k]
            a.plot(wl, y, lw=0.4, color="0.72", label="observed")
            a.plot(wl, m, lw=0.5, color=col, alpha=0.3)
            a.plot(wl, np.where(ok, m, np.nan), lw=1.0, color=col,
                   label=f"{who}:  z = {z:.4f}   reduced chi2 = {r:.2f}")
            a.set_ylim(q[0] - pad, q[1] + pad)
            a.grid(alpha=0.25)
            a.set_ylabel("flux [$10^{-20}$]", fontsize=8)
            a.legend(fontsize=8, loc="upper left")

        # ---- 第 4 列:兩個模型的殘差疊在一起 ----
        # chi2 就是這兩條線的平方和,所以誰比較貼近 0 就是誰贏 —— 疊在同一格
        # 才比得出來,分開畫時眼睛沒辦法在格與格之間做細部比對。
        ax[3].axhline(0, color="0.3", lw=0.6)
        for h in (-1, 1):
            ax[3].axhline(h, color="0.6", lw=0.5, ls=":")
        for (z, m, r, col, who), res in zip(info, (res_o, res_g)):
            ax[3].plot(wl, np.where(ok, res, np.nan), lw=0.45, color=col,
                       label=f"{who}  z = {z:.4f}")
        ax[3].set_ylim(-rr, rr)
        ax[3].grid(alpha=0.25)
        ax[3].set_ylabel(r"(obs$-$mod)/$\sigma$", fontsize=8)
        ax[3].legend(fontsize=8, loc="upper left")

        # ---- 第 5 列:累積 Δchi2 ----
        # C(λ) = Σ_{λ' ≤ λ} [res_ours² − res_gt²],只累加進 chi2 的通道。
        # 往下走 = 我們的解在那一段贏,往上 = gt 贏。曲線的終點就是兩者的
        # chi2 差。這條線直接回答「偏好是從哪些波長來的」,不必用眼睛猜。
        d2 = np.where(ok, res_o ** 2 - res_g ** 2, 0.0)
        cum = np.cumsum(np.nan_to_num(d2))
        ax[4].axhline(0, color="0.3", lw=0.6)
        ax[4].plot(wl, cum, lw=1.0, color="k")
        ax[4].fill_between(wl, cum, 0, where=cum < 0, color=OURS_C, alpha=0.25)
        ax[4].fill_between(wl, cum, 0, where=cum > 0, color=GT_C, alpha=0.25)
        ax[4].grid(alpha=0.25)
        ax[4].set_ylabel(r"cumulative $\Delta\chi^2$", fontsize=8)
        ax[4].annotate(f"total = {cum[-1]:+,.0f}   "
                       f"({'our fit' if cum[-1] < 0 else 'gt'} wins)",
                       (0.995, 0.06), xycoords="axes fraction", ha="right",
                       fontsize=8)

        ax[2].plot(wl, sig, lw=0.4, color="0.72")
        ax[2].plot(wl, np.where(ok, sig, np.nan), lw=0.6, color="0.25",
                   label=fr"$\sigma$, {int(ok.sum())} channels in chi2")
        ax[2].set_yscale("log")
        vis = (wl >= args.star_window[0]) & (wl <= args.star_window[1]) \
            & np.isfinite(sig) & (sig > 0)
        ax[2].set_ylim(float(np.nanpercentile(sig[vis], 0.5)) * 0.7,
                       float(np.nanpercentile(sig[vis], 99.9)) * 1.4)
        ax[2].yaxis.set_major_locator(matplotlib.ticker.LogLocator(
            base=10, subs=(1.0, 2.0, 5.0), numticks=6))
        ax[2].yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax[2].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:g}"))
        ax[2].grid(alpha=0.25)
        ax[2].set_ylabel(r"$\sigma$ [$10^{-20}$]", fontsize=8)
        ax[2].legend(fontsize=8, loc="upper left")
        ax[4].set_xlabel("wavelength [$\\AA$]", fontsize=9)
        ax[0].set_xlim(*args.star_window)

        fig.suptitle(f"ID {t}   our z = {z_ours:.4f} "
                     f"(reduced chi2 {r_ours:.2f})   "
                     f"vs   gt z = {z_gt:.4f} (reduced chi2 {r_gt:.2f})   "
                     f"[{args.star_window[0]:.0f}-{args.star_window[1]:.0f} "
                     f"$\\AA$]", fontsize=12, y=0.992)
        # tight_layout 會自己在標題和圖之間留一段固定的內距,rect 壓不掉;
        # 之後再用 subplots_adjust 直接指定 axes 區域的上緣才有效。
        fig.tight_layout(rect=(0, 0, 1, 0.99))
        fig.subplots_adjust(top=0.972)
        out = FIGURES / f"step4b_mismatch_id{t:02d}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
