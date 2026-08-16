"""step4b 第 1 段(是不是恆星)的視覺化 —— 訂門檻用。

第 1 段的模型只有一個自由參數:

    flux(λ) − s·C_sky(λ)  =  A · T(λ / (1+z)),   A ≥ 0

通道集合固定(擬合視窗扣掉天光線),自由參數固定 1 個,所以
**chi2 與 reduced chi2 是同一條曲線**,只差 dof 這個常數。兩個刻度
畫在同一張圖的左右兩側,不畫兩張一樣的圖。

判讀一個 chi2 之前要先知道它在量什麼。這裡有兩種完全不同的高 chi2:

    源很亮、模板不對    → chi2 高,而且 A 的偵測顯著度也高
    源很暗、看不到源    → chi2 高,但那是天空扣除的殘差,和模板無關;
                          A 的顯著度接近 0,換哪條模板 chi2 都一樣

只看 chi2 分不出這兩件事,所以每一張圖都同時給出

    A 的顯著度  √(chi2(A=0) − chi2(最佳))   —— 這條光譜裡到底有沒有源
    模板的張力  23 條模板之間 chi2 的散布   —— 資料分不分得出模板的差別

產出 figures/step4b_{tag}/
    stage1_all_sources.png   所有源排在一起,用來挑門檻
    chi2_vs_z.png            所有源的 chi2(z) 小圖總覽
    id{NN}/landscape.png     單一源:chi2(z) 地形
    id{NN}/per_template.png  單一源:23 條模板各自的最佳 chi2
    id{NN}/spectrum.png      單一源:全波段光譜 + 最佳模板 + 進 chi2 的通道,
                             下面附一列誤差譜 σ(λ) —— 就是 chi2 的分母

    conda run -n astro python src/skymodel/experiments/step4b_stage1.py
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
# 視窗常數與檔名規則都從 step4b 進口 —— 兩邊各寫一份的話,改了一邊忘了另一邊,
# 圖上標的視窗就會和實際算 chi2 的視窗不一樣,而那種錯誤看不出來。
from step4b_window_fit import STAR_WINDOW, GAL_WINDOW, FULL_RANGE, make_tag
from utils import load_line_masks

ROOT    = Path(__file__).resolve().parents[3]
STEP02  = ROOT / "results/skymodel/step02"
STEP02B = ROOT / "results/skymodel/step02b"
STEP03  = ROOT / "results/skymodel/step03"
STEP04B = ROOT / "results/skymodel/step04b"
FIGURES = ROOT / "results/skymodel/figures"
TPL_DIR = ROOT / "data/sdss_templates"

# docs/sdss-templates.md 第 2 節。分組是為了上色 —— 同一族的模板長得像,
# 分開上色才看得出「贏的是哪一族」,而不是「贏的是第幾號」。
STAR_TYPE = {
    0: "O", 1: "O/B", 2: "B", 3: "A", 4: "A", 5: "F/A", 6: "F", 7: "F",
    8: "G", 9: "G", 10: "K", 11: "M1", 12: "M3", 13: "M5", 14: "M8", 15: "L1",
    16: "mag WD", 17: "C", 18: "C", 19: "C", 20: "WD", 21: "B WD", 22: "K sd",
}
FAMILY = [("OBA", range(0, 5), "#1f77b4"), ("FG", range(5, 10), "#2ca02c"),
          ("KM L", range(10, 16), "#d62728"), ("WD", [16, 20, 21], "#9467bd"),
          ("carbon", [17, 18, 19], "#ff7f0e"), ("K subdwarf", [22], "#8c564b")]
COLOR = {i: c for _, idxs, c in FAMILY for i in idxs}


def load_common(args):
    """所有圖共用的資料。窗與遮罩的定義必須和 step4b 逐字相同,否則圖騙人。"""
    wl_air = np.load(STEP03 / "wavelength.npy")
    line   = load_line_masks(STEP03 / "iter_line_mask.npy",
                             cumulative=not args.raw_mask)[args.iter - 1]
    win    = (wl_air >= args.star_window[0]) & (wl_air < args.star_window[1])
    # --spec-dir 讓診斷程式可以讀「扣過天空的 cube 萃取出來的」源光譜
    # (results/skymodel/step02_ours、step02_eso),而不只是原始的 step02/。
    src    = (Path(args.spec_dir) if getattr(args, "spec_dir", None)
              else (STEP02B if args.aperture else STEP02))
    return dict(wl_air=wl_air, wl_vac=air_to_vacuum(wl_air), line=line, win=win,
                fit=win & ~line,
                C_sky=np.load(STEP03 / "sky_continuum.npy"),
                ids=np.load(src / "object_ids.npy"),
                flux=np.load(src / "object_flux.npy"),
                var=np.load(src / "object_var.npy"),
                nspax=np.load(src / "object_nspax.npy"))


def source_spectrum(D, t, s_fix):
    """回傳 (扣掉天空連續譜的光譜, sigma, 可用通道)。"""
    k = int(np.flatnonzero(D["ids"] == t)[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        f = D["flux"][k] / D["nspax"][k]
        v = D["var"][k]  / D["nspax"][k] ** 2
    y   = f - s_fix * D["C_sky"]
    sig = np.sqrt(np.where(v > 0, v, np.nan))
    ok  = D["fit"] & np.isfinite(y) & np.isfinite(sig) & (sig > 0)
    return y, sig, ok


def null_chi2(D, t, s_fix):
    """A = 0 的 chi2 —— 完全不放源時的殘差。偵測顯著度的基準線。"""
    y, sig, ok = source_spectrum(D, t, s_fix)
    return float(((y[ok] / sig[ok]) ** 2).sum()), int(ok.sum())


def summarise(D, t, sc, best, args, out, draw=True):
    """單一源的統計值;draw=True 時順便畫三張圖,存進 id{NN}/ 底下。

    分成三張而不是一張三格,是因為它們回答三個獨立的問題,而且各自需要的寬度
    差很多 —— 光譜要橫跨 4750 A、切成好幾段才讀得出通道;地形圖和長條圖擠在
    旁邊只會互相壓縮。
    """
    dof   = int(best["n_good"]) - 1
    chi0, _ = null_chi2(D, t, args.s_fix)
    names = sorted(set(sc["template"]))
    # 統計值先算完 —— 總覽圖需要它們,即使不畫逐源的三張。
    bt = sorted(((int(nm), sc["chi2"][sc["template"] == nm].min())
                 for nm in names), key=lambda x: x[1])
    spread = 100 * (bt[-1][1] - bt[0][1]) / bt[0][1]
    snr = np.sqrt(max(chi0 - float(best["chi2"]), 0.0))
    stats = dict(id=t, red_chi2=float(best["red_chi2"]),
                 chi2=float(best["chi2"]), dof=dof, snr=snr, spread=spread,
                 tpl=int(best["template"]), A=float(best["A"][0]),
                 lo=bt[0][1], hi=bt[-1][1],
                 nspax=int(best["nspax"]) if "nspax" in best else 0)
    if not draw:
        return stats

    d = out / f"id{t:02d}"
    d.mkdir(parents=True, exist_ok=True)

    # ---- ① chi2(z) 地形 ----
    fig, ax = plt.subplots(figsize=(13, 6))
    for nm in names:
        m = sc["template"] == nm
        ax.plot(sc["z"][m], sc["chi2"][m], lw=0.9, color=COLOR[int(nm)], alpha=0.75)
    ax.axhline(chi0, color="0.3", ls="--", lw=1.2,
               label=f"A = 0, no source   chi2 = {chi0:,.0f}")
    ax.plot(best["z"], best["chi2"], "k*", ms=15,
            label=f"best: tpl {best['template']} ({STAR_TYPE[int(best['template'])]})")
    # z 的範圍只有 ±0.005,預設會被 matplotlib 改成科學記號加偏移量,讀起來很痛苦。
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.4f"))
    ax.set_xlabel("redshift  z")
    ax.set_ylabel("chi2")
    a2 = ax.twinx()
    a2.set_ylim(*[v / dof for v in ax.get_ylim()])
    a2.set_ylabel(f"reduced chi2   (dof = {dof})")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    ax.set_title(f"ID {t}   stage 1 landscape — 23 stellar templates   "
                 f"[{args.star_window[0]:.0f}-{args.star_window[1]:.0f} $\\AA$]",
                 fontsize=12)
    fig.savefig(d / "landscape.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- ② 逐模板的最佳 chi2 ----
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([f"{i:03d} {STAR_TYPE[i]}" for i, _ in bt], [c for _, c in bt],
            color=[COLOR[i] for i, _ in bt])
    ax.axvline(chi0, color="0.3", ls="--", lw=1.2)
    ax.invert_yaxis()
    ax.set_xlabel("best chi2 for that template")
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.3, axis="x")
    ax.set_title(f"ID {t}   template tension {spread:.0f}%\n"
                 "(small = the data cannot tell the templates apart)", fontsize=12)
    fig.savefig(d / "per_template.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- ③ 光譜:全波段,切成數段 ----
    y, sig, ok = source_spectrum(D, t, args.s_fix)
    spl = load_sdss_template(TPL_DIR / f"spDR2-{int(best['template']):03d}.fit")
    mod = float(best["A"][0]) * redshift_to_grid(spl, float(best["z"]), D["wl_vac"])
    wl  = D["wl_air"]
    # y 範圍只由「進 chi2 的通道」決定。全波段畫出來是為了看得見視窗以外長什麼
    # 樣子,但紅端的天光線殘差會衝到幾百,拿它定範圍會把源壓成一條線。
    q   = np.nanpercentile(np.concatenate([y[ok], mod[ok]]), [0.5, 99.5])
    pad = 0.25 * (q[1] - q[0]) + 1e-6
    lo, hi = q[0] - pad, q[1] + pad

    # 一整段畫完就要夠寬:3801 個通道畫在 20 吋、125 dpi 上是每像素 1.5 個通道,
    # 通道解析不出來。40 吋 = 5000 px,約 1.3 像素一個通道才看得見個別通道。
    edges = np.linspace(wl.min(), wl.max(), args.panels + 1)
    # 每一段波長佔兩列:上列資料 + 模型,下列 σ。高度比 3:1 —— σ 是輔助資訊,
    # 給它和光譜一樣的高度會讓人以為兩者同等重要,而且會把光譜壓扁。
    # 圖高乘 4/3:每段從 3 個單位變成 3+1 個,這樣每一列的實際高度和加 σ 之前相同。
    fig, axes = plt.subplots(2 * args.panels, 1,
                             figsize=(args.width, args.height * args.panels * 4 / 3),
                             squeeze=False,
                             gridspec_kw=dict(height_ratios=[3, 1] * args.panels))
    for k in range(args.panels):
        ax, axs = axes[2 * k][0], axes[2 * k + 1][0]
        m = (wl >= edges[k]) & (wl <= edges[k + 1])
        ax.plot(wl[m], y[m], lw=0.5, color="0.72",
                label=f"observed − {args.s_fix}·C_sky" if k == 0 else None)
        ax.plot(wl[m], mod[m], lw=1.3, color="#1f77b4",
                label=(f"A · tpl {best['template']}  (A = {best['A'][0]:.4g})"
                       if k == 0 else None))
        # 紅點畫在「模型」上,不是畫在觀測資料上。觀測是鋸齒狀的,紅點跟著它上下
        # 亂跳,同時編碼了波長與流量 —— 但我們只想說「哪一段波長進了 chi2」。
        # 模型是平滑的,紅點沿著它排成一條連續的線,缺口就是被排除的天光線通道。
        mo = m & ok
        ax.plot(wl[mo], mod[mo], ".", ms=2.6, color="#ff3b3b", lw=0,
                label=f"included in chi2  ({int(ok.sum())} channels)" if k == 0 else None)
        ax.set_xlim(edges[k], edges[k + 1])
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.3)
        ax.set_ylabel("flux  [$10^{-20}$ cgs]")
        ax.tick_params(labelbottom=False)   # x 標籤只寫在下面那列,兩列共用同一個波長軸
        if k == 0:
            ax.legend(fontsize=9, markerscale=5, ncol=3, loc="upper right")

        # σ 是 chi2 的分母,和 step4b 擬合時用的完全是同一個量:
        #     sig = sqrt(var / nspax**2) = sqrt(Σ var) / N
        # 配的是平均光譜(flux / nspax),所以除的是 N 而不是 sqrt(N)。
        # 天光線處的 σ 遠高於連續譜區,線性軸會把連續譜區壓進面板底部,
        # 看不出任何結構,所以用 log 軸。
        # 紅點的意義和上面那一列一模一樣 —— 同一個符號在同一張圖裡只能有一個意思。
        axs.plot(wl[m], sig[m], lw=0.5, color="0.45",
                 label=r"$\sigma = \sqrt{\Sigma\,\mathrm{var}}\,/\,N$"
                       if k == 0 else None)
        axs.plot(wl[mo], sig[mo], ".", ms=2.6, color="#ff3b3b", lw=0)
        axs.set_yscale("log")
        axs.set_xlim(edges[k], edges[k + 1])
        axs.grid(alpha=0.3)
        axs.set_ylabel("$\\sigma$  [$10^{-20}$ cgs]")
        if k == 0:
            axs.legend(fontsize=9, loc="upper right")
    axes[-1][0].set_xlabel("wavelength [$\\AA$] (air)")
    fig.suptitle(f"ID {t}      z = {float(best['z']):+.4f}      "
                 f"reduced chi2 = {float(best['red_chi2']):.2f}      "
                 f"A = {float(best['A'][0]):.4g}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(d / "spectrum.png", dpi=125, bbox_inches="tight")
    plt.close(fig)

    return stats


def all_sources(rows, args, out):
    """所有源排在一起 —— 這是挑門檻真正要看的圖。"""
    rows = sorted(rows, key=lambda r: r["red_chi2"])
    x    = np.arange(len(rows))
    dof  = rows[0]["dof"]
    fig, _ax = plt.subplots(figsize=(17, 7.5))
    ax = [_ax]

    # 上:每個源畫一條「23 條模板的 chi2 範圍」,實心點是最佳。
    # 慣例和 step4b_iters/ 那四張一致:線性軸 + 整數刻度、頂端截斷、
    # A = 0 用紅色 X 標出來。三張圖用同一套語言,不必每張重新學一次怎麼讀。
    best_v = np.array([r["red_chi2"] for r in rows])
    hi = np.sort(best_v)[-2] * 1.12
    if args.thresholds:                      # 有畫門檻線時要留得下它們
        hi = max(hi, max(args.thresholds) * 1.25)
    lo = best_v.min() * 0.92
    for i, r in enumerate(rows):
        ax[0].plot([i, i], [r["lo"] / dof, min(r["hi"] / dof, hi)], color="0.75",
                   lw=3, solid_capstyle="butt", zorder=1)
    zero = np.array([r["A"] <= 0 for r in rows])
    ax[0].scatter(x[~zero], np.minimum(best_v, hi)[~zero], s=45, zorder=3,
                  c=[COLOR[r["tpl"]] for r, z in zip(rows, zero) if not z],
                  edgecolor="k", linewidth=0.5)
    if zero.any():
        ax[0].scatter(x[zero], np.minimum(best_v, hi)[zero], s=120, marker="X",
                      color="#d62728", zorder=4, edgecolor="k", linewidth=0.6,
                      label="A = 0  (no source)")
    for i, r in enumerate(rows):
        if r["red_chi2"] > hi:
            ax[0].annotate(f"{r['red_chi2']:,.0f}", (i, hi), xytext=(0, -22),
                           textcoords="offset points", ha="center", fontsize=9,
                           color="#d62728", fontweight="bold",
                           arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=1.4))
    for T in args.thresholds:
        n = sum(1 for r in rows if r["red_chi2"] < T)
        ax[0].axhline(T, ls="--", lw=1.3, color="#2ca02c")
        ax[0].text(len(rows) - 0.3, T,
                   f"  T = {T:g}  ->  {n} star / {len(rows) - n} non-star",
                   va="center", fontsize=10, color="#2ca02c")
    ax[0].set_ylim(lo, hi)
    ax[0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax[0].set_ylabel("reduced chi2   (best stellar template)")
    a2 = ax[0].twinx()
    a2.set_ylim(lo * dof, hi * dof)
    a2.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    a2.set_ylabel(f"chi2   (dof = {dof})")
    ax[0].grid(alpha=0.3)
    ax[0].set_title(f"stage 1, window {args.star_window[0]:.0f}-"
                    f"{args.star_window[1]:.0f} $\\AA$  "
                    f"({rows[0]['dof'] + 1} channels)", fontsize=13)
    ax[0].legend(handles=[plt.Line2D([], [], color="0.75", lw=4,
                                     label="range over the 23 templates")]
                 + [plt.Line2D([], [], marker="o", ls="none", color=c,
                               mec="k", label=nm) for nm, _, c in FAMILY]
                 + ([plt.Line2D([], [], marker="X", ls="none", color="#d62728",
                                mec="k", ms=10, label="A = 0 (no source)")]
                    if zero.any() else []),
                 fontsize=9, ncol=7, loc="upper left")

    # 下:A 的偵測顯著度 —— 沒有這一列,上面那張圖會被誤讀
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([r["id"] for r in rows], fontsize=8)
    ax[0].set_xlabel("source ID   (sorted by reduced chi2)")

    p = out / "stage1_all_sources.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return p


def chi2_vs_z_grid(scans, args, tag, out):
    """所有源的 chi2(z) 擺在一起 —— 每格一個源。

    每格畫兩層:淡灰是 23 條模板各自的曲線,粗線是「每個 z 上最好的那條模板」
    的包絡線。掃描實際比較的就是包絡線 —— 對每個 z,擬合器可以自由挑模板。

    y 軸各格獨立(源的亮度差三個數量級,共用軸的話大部分格子會是平的),
    所以格子之間比的是**形狀**:曲線有沒有明顯的谷?谷有多深?

    第 1 段的 z 只掃 ±0.005(前景星的本動速度範圍),所以這裡的曲線本來就
    應該很平 —— 恆星沒有紅移可找。真正有結構的 chi2(z) 在第 2 段。
    """
    ids = sorted(scans)
    ncol = 5
    nrow = int(np.ceil(len(ids) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.5 * nrow),
                             squeeze=False)
    for k, sid in enumerate(ids):
        ax = axes[k // ncol][k % ncol]
        sc = scans[sid]
        zs = np.unique(sc["z"])
        dof = int(sc["n_good"][0]) - 1
        best_t = None
        env = np.full(zs.size, np.inf)
        for nm in sorted(set(sc["template"])):
            m = sc["template"] == nm
            o = np.argsort(sc["z"][m])
            y = sc["chi2"][m][o] / dof
            ax.plot(zs, y, lw=0.6, color="0.8", zorder=1)
            better = y < env
            env = np.where(better, y, env)
            if better.any() and (best_t is None or y.min() < env.min() + 1e-12):
                pass
        ax.plot(zs, env, lw=1.6, color="#1f77b4", zorder=3)
        j = int(np.argmin(sc["chi2"]))
        ax.plot(sc["z"][j], sc["chi2"][j] / dof, "r*", ms=12, zorder=4)
        ax.set_title(f"ID {sid}   tpl {sc['template'][j]} "
                     f"({STAR_TYPE[int(sc['template'][j])]})   "
                     f"best z = {sc['z'][j]:+.4f}", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.3f"))
        ax.grid(alpha=0.3)
    for k in range(len(ids), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"stage 1 — reduced chi2 vs redshift, one panel per source   "
                 f"[{tag}]\n"
                 "grey = each of the 23 stellar templates,  blue = best template "
                 "at each z,  red star = the reported solution", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    p = out / "chi2_vs_z.png"
    fig.savefig(p, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return p


def iter_compare(per_iter, args, out):
    """同一批源在四輪遮罩下的變化 —— 遮得越多,結論會不會跟著變?

    chi2 本來就會隨通道數變小,所以第一列看的不是絕對值,是**排名有沒有翻**。
    第二列直接畫贏家模板的編號:同一個源在四輪裡選到不同模板,代表那個
    「最像哪種星」的答案不穩,不能拿來當分類依據。
    """
    its  = sorted(per_iter)
    vals = {it: {r["id"]: r["red_chi2"] for r in per_iter[it]} for it in its}
    # 每一輪依「自己的」chi2 排序 —— 四輪疊在同一條 x 軸上時,只有基準那一輪
    # 是單調的,其餘三輪上下亂跳,看起來像雜訊。各自排序之後每張都是單調曲線,
    # 要比較的東西改成:x 軸標籤(源的順序)有沒有變、曲線的高度與陡峭度。
    rank = {it: {sid: k for k, sid in
                 enumerate(sorted(vals[it], key=lambda s: vals[it][s]), 1)}
            for it in its}
    amps = {it: {r["id"]: r["A"] for r in per_iter[it]} for it in its}
    ref  = its[0]

    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for it in its:
        order = sorted(vals[it], key=lambda s: vals[it][s])
        x     = np.arange(len(order))
        y     = np.array([vals[it][s] for s in order])
        dmove = np.array([rank[ref][s] - rank[it][s] for s in order])
        av    = amps[it]

        fig, ax = plt.subplots(figsize=(16, 6.5))
        ax.plot(x, y, "-", lw=1.0, color="0.65", zorder=1)
        # 上色的判準是「A 有沒有被非負限制壓到 0」。A = 0 代表無限制的最佳解
        # 是負的 —— 資料說這裡的流量比天空連續譜還低,也就是根本沒有源。
        # 那件事比 chi2 的高低更根本:chi2 在那時量的純粹是天空扣除的殘差。
        amp  = np.array([av[s_] for s_ in order])
        zero = amp <= 0
        ax.scatter(x[~zero], y[~zero], s=70, color="#4c72b0", zorder=3,
                   edgecolor="k", linewidth=0.5, label="A > 0  (source detected)")
        if zero.any():
            ax.scatter(x[zero], y[zero], s=110, color="#d62728", marker="X",
                       zorder=4, edgecolor="k", linewidth=0.6,
                       label="A = 0  (no source: unconstrained fit went negative)")
        ax.legend(fontsize=10, loc="upper left")

        # 主源的 chi2 比其餘的源高好幾個數量級,不截頂的話其餘的會壓成
        # 一條線。紅色不是資料,是「這裡有一個點被切掉了」的記號。
        hi = np.sort(y)[-2] * 1.12
        lo = y.min() * 0.92
        for i, v in enumerate(y):
            if v > hi:
                ax.annotate(f"{v:,.0f}", (i, hi), xytext=(0, -20),
                            textcoords="offset points", ha="center", fontsize=9,
                            color="#d62728", fontweight="bold",
                            arrowprops=dict(arrowstyle="-|>", color="#d62728",
                                            lw=1.4))
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_ylabel("reduced chi2   (best stellar template)")
        ax.set_xticks(x)
        ax.set_xticklabels(order, fontsize=8)
        ax.set_xlabel("source ID   (sorted by this iteration's reduced chi2)")
        ax.grid(alpha=0.3)
        nch = per_iter[it][0]["dof"] + 1
        # A = 0 的源已經用 ✕ 標在圖上、也寫在圖例裡了,標題不重複。
        ax.set_title(f"sky-line mask iter{it} (cumulative) — {nch} clean channels "
                     f"in {args.star_window[0]:.0f}-{args.star_window[1]:.0f} "
                     f"$\\AA$", fontsize=13)
        p = out / f"iter{it}.png"
        fig.savefig(p, dpi=140, bbox_inches="tight")
        plt.close(fig)
        paths.append(p)

    # 數字版:排名一致性與模板穩定度
    ids = sorted(vals[ref])
    print(f"\n{'':>14}" + "".join(f"{f'iter{i}':>10}" for i in its))
    print(f"{'乾淨通道':>12}" + "".join(
        f"{per_iter[i][0]['dof'] + 1:>10}" for i in its))
    print(f"{'chi2/dof 中位':>12}" + "".join(
        f"{np.median(list(vals[i].values())):>10.2f}" for i in its))
    print(f"{f'vs iter{ref} 排名相關':>12}" + "".join(
        f"{'—' if i == ref else f'{np.corrcoef([rank[ref][s] for s in ids], [rank[i][s] for s in ids])[0, 1]:+.3f}':>10}"
        for i in its))
    tpl = {s: {it: next(r["tpl"] for r in per_iter[it] if r["id"] == s)
               for it in its} for s in ids}
    same = sum(1 for s in ids if len(set(tpl[s].values())) == 1)
    print(f"\n四輪都選到同一條模板的源: {same} / {len(ids)}")
    return paths


def main():
    ap = argparse.ArgumentParser(description="step4b 第 1 段的視覺化")
    ap.add_argument("--star-window", type=float, nargs=2, default=STAR_WINDOW)
    ap.add_argument("--gal-window", type=float, nargs=2, default=GAL_WINDOW)
    ap.add_argument("--full-range", action="store_true",
                    help="對照組:兩段都用 MUSE 全波段")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--sky-basis", action="store_true")
    ap.add_argument("--line-mask-iter", type=int, nargs="+", default=[1, 2, 3, 4],
                    help="要畫哪幾輪遮罩的結果,預設四輪全畫並附上跨輪比較")
    ap.add_argument("--aperture", action="store_true",
                    help="讀 step02b/ 的圓形孔徑光譜(要和 step4b 的設定一致)")
    ap.add_argument("--raw-mask", action="store_true",
                    help="四輪各自獨立,不累加(要和 step4b 跑的設定一致)")
    ap.add_argument("--s-fix", type=float, default=1.0)
    ap.add_argument("--thresholds", type=float, nargs="*", default=[],
                    help="在總覽圖上畫出候選門檻的水平線;預設不畫")
    ap.add_argument("--panels", type=int, default=1,
                    help="光譜切成幾段(預設 1 = 不切,靠加寬解決)")
    ap.add_argument("--width", type=float, default=40.0,
                    help="光譜圖寬(吋)。不切段時要夠寬才解析得出通道")
    ap.add_argument("--height", type=float, default=7.0,
                    help="光譜圖每一段的高(吋)")
    ap.add_argument("--only-summary", action="store_true",
                    help="只重畫總覽圖(stage1_all_sources / chi2_vs_z / 跨輪比較),"
                         "跳過逐源的三張 —— 只改總覽的版面時快很多")
    ap.add_argument("--spec-dir", default=None,
                    help="源光譜的目錄。分類用的是扣過天空的版本(step02_eso),"
                         "和 step4b 保持一致才比得上")
    ap.add_argument("--id", default="all")
    args = ap.parse_args()
    if args.full_range:
        args.star_window = args.gal_window = FULL_RANGE

    per_iter, root = {}, None
    for it in args.line_mask_iter:
        args.iter = it                      # load_common 與圖的標題都要用到
        # 字尾要和 step4b 用同一套規則 —— 它把光譜來源編進檔名(step02_eso -> _eso),
        # 這裡漏掉的話會去找不存在的檔案而整批跳過。
        suffix = (f"_{Path(args.spec_dir).name.replace('step02', '')}"
                  if args.spec_dir else "")
        tag = make_tag(args.basis, args.K, args.s_fix, args.star_window,
                       args.gal_window, args.sky_basis, it, not args.raw_mask,
                       args.aperture, suffix)
        bf  = STEP04B / f"best_{tag}.npz"
        if not bf.exists():
            print(f"iter{it}: 找不到 {bf.name},跳過。先跑 step4b_window_fit.py")
            continue
        D    = load_common(args)
        best = np.load(bf)
        out  = FIGURES / f"step4b_{tag}"
        out.mkdir(parents=True, exist_ok=True)
        root = root or out.parent

        targets = (best["id"].tolist() if args.id == "all" else [int(args.id)])
        rows, scans = [], {}
        for t in targets:
            f = STEP04B / f"scan1_id{t}_{tag}.npz"
            if not f.exists():
                print(f"ID {t}: 找不到 {f.name},跳過")
                continue
            sc = np.load(f)
            j  = int(np.flatnonzero(best["id"] == t)[0])
            # 第 1 段的最佳解要從 scan1 取,不是 best —— best 是兩段之後的結果。
            i  = int(np.argmin(sc["chi2"]))
            b  = {k: sc[k][i] for k in
                  ("template", "z", "A", "chi2", "red_chi2", "n_good")}
            b["nspax"] = int(best["nspax"][j])
            rows.append(summarise(D, t, sc, b, args, out,
                                  draw=not args.only_summary))
            scans[t] = sc
        if not rows:
            continue
        per_iter[it] = rows
        print(f"iter{it}  {rows[0]['dof'] + 1:>4} 乾淨通道   "
              f"chi2/dof 中位 {np.median([r['red_chi2'] for r in rows]):>7.2f}   "
              f"{len(rows)} 張 -> {out.name}")
        if len(rows) > 1:
            all_sources(rows, args, out)
            chi2_vs_z_grid(scans, args, tag, out)

    if len(per_iter) > 1:
        # 資料夾名要帶上「所有會改變結果的設定」—— 不帶的話換一個視窗或換成
        # 孔徑版就會把前一次蓋掉,而且蓋掉的方式看不出來(檔名一樣、圖也很像)。
        sub = (f"step4b_iters_{args.star_window[0]:.0f}-{args.star_window[1]:.0f}"
               + ("_ap" if args.aperture else ""))
        for p in iter_compare(per_iter, args, root / sub):
            print(f"saved -> {p}")


if __name__ == "__main__":
    main()
