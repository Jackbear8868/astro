"""把指定的幾條恆星模板疊在同一個源的光譜上直接比較。

`per_template.png` 給的是 23 條模板各自的最佳 chi2(一根長條一條模板),回答
「誰贏」;這支回答的是**為什麼贏** —— 兩條模板在哪一段波長開始分家。

每條模板用的是它自己在掃描裡的最佳 (z, A),不是共用贏家的參數 —— 否則比的
就變成「用別人的參數配得多差」,那不公平。

一條模板一張圖,存進 id{NN}/compare/。疊在一起時三條模板會互相遮蔽,而且
遮蔽最嚴重的正是吸收線的谷底 —— 那是判別型別的地方。分開畫,但 y 範圍共用,
不然眼睛比的是軸的縮放而不是資料。

    conda run -n astro python src/skymodel/experiments/step4b_compare_templates.py \\
        --id 35 --templates 10 7
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from templates import load_sdss_template, redshift_to_grid, air_to_vacuum
from utils import load_line_masks
from step4_fit_source import STAR_WINDOW, GAL_WINDOW, make_tag

ROOT    = Path(__file__).resolve().parents[3]
STEP02  = ROOT / "results/skymodel/step02"
STEP02B = ROOT / "results/skymodel/step02b"
STEP03  = ROOT / "results/skymodel/step03"
STEP04 = ROOT / "results/skymodel/step04"
FIGURES = ROOT / "results/skymodel/figures"
TPL_DIR = ROOT / "data/sdss_templates"

STAR_TYPE = {
    0: "O", 1: "O/B", 2: "B", 3: "A", 4: "A", 5: "F/A", 6: "F", 7: "F",
    8: "G", 9: "G", 10: "K", 11: "M1", 12: "M3", 13: "M5", 14: "M8", 15: "L1",
    16: "mag WD", 17: "C", 18: "C", 19: "C", 20: "WD", 21: "B WD", 22: "K sd",
}
MODEL_COLOR = "#1f77b4"     # 光譜圖三張共用 —— 換顏色會讓眼睛以為在比顏色
DOT_COLOR   = "#ff3b3b"
# chi2(z) 那張反過來:同一張圖上多條曲線,必須用顏色分開
CURVE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def main():
    ap = argparse.ArgumentParser(description="多條恆星模板疊在同一個源上比較")
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--templates", type=int, nargs="+", required=True,
                    help="要比較的模板編號(0-22)。最佳解那條會自動加入")
    ap.add_argument("--ylim", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"), help="固定 y 範圍;不給則由進 chi2 的通道決定")
    ap.add_argument("--width", type=float, default=40.0, help="圖寬(吋)")
    ap.add_argument("--height", type=float, default=7.0, help="圖高(吋)")
    ap.add_argument("--star-window", type=float, nargs=2, default=STAR_WINDOW)
    ap.add_argument("--gal-window", type=float, nargs=2, default=GAL_WINDOW)
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--sky-basis", action="store_true")
    ap.add_argument("--aperture", action="store_true")
    ap.add_argument("--raw-mask", action="store_true")
    ap.add_argument("--iter", type=int, default=1)
    ap.add_argument("--s-fix", type=float, default=1.0)
    args = ap.parse_args()

    tag = make_tag(args.basis, args.K, args.s_fix, args.star_window,
                   args.gal_window, args.sky_basis, args.iter,
                   not args.raw_mask, args.aperture)
    f = STEP04 / f"scan1_id{args.id}_{tag}.npz"
    if not f.exists():
        raise SystemExit(f"找不到 {f.name}。先跑 step4_fit_source.py")
    sc = np.load(f)

    wl   = np.load(STEP03 / "wavelength.npy")
    vac  = air_to_vacuum(wl)
    Csky = np.load(STEP03 / "sky_continuum.npy")
    line = load_line_masks(STEP03 / "iter_line_mask.npy",
                           cumulative=not args.raw_mask)[args.iter - 1]
    win  = (wl >= args.star_window[0]) & (wl < args.star_window[1])

    src   = STEP02B if args.aperture else STEP02
    ids   = np.load(src / "object_ids.npy")
    k     = int(np.flatnonzero(ids == args.id)[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        nsp = np.load(src / "object_nspax.npy")[k]
        y   = np.load(src / "object_flux.npy")[k] / nsp - args.s_fix * Csky
        sig = np.sqrt(np.load(src / "object_var.npy")[k] / nsp ** 2)
    ok = win & ~line & np.isfinite(y) & np.isfinite(sig) & (sig > 0)

    # 最佳解那條一定要在圖上 —— 只比使用者指定的兩條,會漏掉「其實還有更好的」。
    best_t = int(sc["template"][int(np.argmin(sc["chi2"]))])
    want   = list(dict.fromkeys([best_t] + args.templates))

    models, info = {}, []
    dof = int(sc["n_good"][0]) - 1
    for t in want:
        m = sc["template"] == f"{t:03d}"
        if not m.any():
            print(f"模板 {t:03d} 不在掃描結果裡,跳過")
            continue
        j = int(np.flatnonzero(m)[np.argmin(sc["chi2"][m])])
        z, A, c = float(sc["z"][j]), float(sc["A"][j][0]), float(sc["chi2"][j])
        spl = load_sdss_template(TPL_DIR / f"spDR2-{t:03d}.fit")
        models[t] = A * redshift_to_grid(spl, z, vac)
        info.append((t, z, A, c, c / dof))

    base = min(x[3] for x in info)
    print(f"ID {args.id}   iter{args.iter}   {int(ok.sum())} channels in chi2   "
          f"dof = {dof}")
    print(f"{'tpl':>6}{'type':>9}{'z':>10}{'A':>12}{'chi2':>11}{'chi2/dof':>10}"
          f"{'vs best':>10}")
    print("-" * 68)
    for t, z, A, c, r in info:
        print(f"{t:>6}{STAR_TYPE[t]:>9}{z:>10.4f}{A:>12.5g}{c:>11,.0f}{r:>10.2f}"
              f"{100*(c/base-1):>9.1f}%")

    # ---------------- 圖:一條模板一張 ----------------
    # 三張共用同一個顏色與同一個 y 範圍 —— 要比的是模板的形狀,不是顏色或縮放。
    # 紅點畫在「模型」上不是資料上:資料是鋸齒狀的,紅點跟著它跳,同時編碼了
    # 波長與流量;模型平滑,紅點沿著它排成一條線,只傳達「哪一段進了 chi2」。
    if args.ylim is not None:
        lo, hi = args.ylim
    else:
        # 預設由「進 chi2 的通道」決定 —— 紅端的天光線殘差會衝到幾百,
        # 拿它定範圍會把源壓成一條線。
        q = np.nanpercentile(
            np.concatenate([y[ok]] + [models[t][ok] for t in models]), [0.5, 99.5])
        pad = 0.25 * (q[1] - q[0])
        lo, hi = q[0] - pad, q[1] + pad

    out = FIGURES / f"step4b_{tag}" / f"id{args.id:02d}" / "compare"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for t, z, A, c, r in info:
        fig, a = plt.subplots(figsize=(args.width, args.height))
        a.plot(wl, y, lw=0.5, color="0.72", zorder=1,
               label=f"observed − {args.s_fix}·C_sky")
        a.plot(wl, models[t], lw=1.3, color=MODEL_COLOR, zorder=2,
               label=f"A · tpl {t:03d} ({STAR_TYPE[t]})")
        a.plot(wl[ok], models[t][ok], ".", ms=2.6, color=DOT_COLOR, lw=0, zorder=3,
               label=f"included in chi2 ({int(ok.sum())} channels)")
        a.set_xlim(wl.min(), wl.max())
        a.set_ylim(lo, hi)
        a.grid(alpha=0.3)
        a.set_xlabel("wavelength [$\\AA$] (air)")
        a.set_ylabel("flux  [$10^{-20}$ cgs]")
        a.legend(fontsize=10, markerscale=5, loc="upper right")
        a.set_title(f"ID {args.id}   template {t:03d} ({STAR_TYPE[t]})      "
                    f"z = {z:+.4f}      reduced chi2 = {r:.2f}      A = {A:.4g}",
                    fontsize=13)
        p = out / f"tpl_{t:03d}_{STAR_TYPE[t].replace(' ', '')}.png"
        fig.savefig(p, dpi=125, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # ---------------- chi2(z):只畫選中的幾條 ----------------
    # landscape.png 有全部 23 條,擠在一起看不出誰是誰。這裡只留要比較的幾條,
    # 並且標出每條自己的極小值 —— 那就是上面每張光譜圖用的 (z, A)。
    fig, a = plt.subplots(figsize=(13, 6.5))
    for i, (t, z, A, c, r) in enumerate(info):
        m = sc["template"] == f"{t:03d}"
        o = np.argsort(sc["z"][m])
        col = CURVE_COLORS[i % len(CURVE_COLORS)]
        a.plot(sc["z"][m][o], sc["chi2"][m][o], lw=1.4, color=col,
               label=f"{t:03d} {STAR_TYPE[t]}   min chi2 = {c:,.0f}"
                     f"  (chi2/dof = {r:.2f})")
        a.plot(z, c, "*", ms=15, color=col, mec="k", mew=0.6, zorder=5)
    a.set_xlabel("redshift  z")
    a.set_ylabel("chi2")
    a.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
    # 第二個刻度:reduced chi2。通道數與自由參數在這張圖裡都固定,所以它和 chi2
    # 是同一條曲線除以 dof —— 畫成第二個 y 軸,不另外畫一張一模一樣的圖。
    a2 = a.twinx()
    a2.set_ylim(*[v / dof for v in a.get_ylim()])
    a2.set_ylabel(f"reduced chi2   (dof = {dof})")
    a.grid(alpha=0.3)
    a.legend(fontsize=10, loc="upper right")
    a.set_title(f"ID {args.id}   chi2 vs redshift for the selected templates   "
                f"[{args.star_window[0]:.0f}-{args.star_window[1]:.0f} $\\AA$, "
                f"{int(ok.sum())} channels]", fontsize=13)
    pz = out / ("chi2_vs_z_" + "_".join(f"{t:03d}" for t, *_ in info) + ".png")
    fig.savefig(pz, dpi=130, bbox_inches="tight")
    plt.close(fig)
    paths.append(pz)

    print()
    for p in paths:
        print(f"saved -> {p}")


if __name__ == "__main__":
    main()
