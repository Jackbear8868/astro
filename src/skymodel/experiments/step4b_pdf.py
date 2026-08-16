"""把源的「光譜 + 最佳模板 + 殘差 + 誤差譜」合成一份 PDF —— 寄給教授用。

預設畫 best 檔裡的全部源。
`--all-ids` 才畫全部。


step4b_stage1.py 產的 id{NN}/spectrum.png 又寬又大,整批沒辦法寄也沒辦法一次
讀。這支把同樣的內容重畫進一份 A4 橫式 PDF,一頁 N 個源,外加第一頁的總表。

版面語言和 spectrum.png 逐項一致 —— 灰線是觀測、藍線是模型、紅點是真正進 chi2
的通道、下面那一小列是誤差譜 σ(log 軸)。兩份東西看起來必須是同一種圖,不然
對照著看的人要重學一次怎麼讀。

資料線用 rasterized=True 轉成點陣,文字與座標軸維持向量:每個源好幾條線、每條
線數千個通道,全部走向量的話 PDF 會很大、開啟很慢;文字仍是向量,放大不會糊。

    conda run -n astro python src/skymodel/experiments/step4b_pdf.py \\
        --star-window 4600 5800 --iter 1
"""
import argparse
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from templates import load_sdss_template, load_eigen_galaxy, redshift_to_grid
from step4b_window_fit import (STAR_WINDOW, GAL_WINDOW, make_tag,
                               EIGEN_GAL)
# 圖的內容必須和 step4b_stage1.py 完全同源 —— 光譜怎麼算、遮罩怎麼定義都直接
# 進口它的函式,不另外寫一份。兩份各自實作的話,哪天改了一邊,兩種圖會悄悄不一樣。
from step4b_stage1 import load_common, source_spectrum, STAR_TYPE

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"        # 白光圖 + segmentation map
STEP04B = ROOT / "results/skymodel/step04b"
FIGURES = ROOT / "results/skymodel/figures"
TPL_DIR = ROOT / "data/sdss_templates"

A4_LAND = (11.69, 8.27)     # A4 橫式(吋)
# 恆星與星系兩個模型的顏色。光譜列和殘差列共用同一組 —— 同一個模型在兩個地方
# 必須是同一個顏色,否則要在腦裡再做一次對應。
STAR_COLOR = "#1f77b4"
GAL_COLOR  = "#d62728"
# spDR2-023..028 的星系模板(docs/sdss-templates.md 第 2 節)
GAL_TYPE = {23: "early-type", 24: "galaxy", 25: "galaxy", 26: "galaxy",
            27: "late-type", 28: "LRG"}


def stage1_best(tag, t, best):
    """第 1 段的最佳解。和 step4b_stage1.py 一樣從 scan1 取,不是從 best 取 ——
    best 是兩段跑完的結果,若該源進了第 2 段,裡面存的會是星系解。"""
    f = STEP04B / f"scan1_id{t}_{tag}.npz"
    if not f.exists():
        return None
    sc = np.load(f)
    i  = int(np.argmin(sc["chi2"]))
    b  = {k: sc[k][i] for k in ("template", "z", "A", "chi2", "red_chi2", "n_good")}
    j  = int(np.flatnonzero(best["id"] == t)[0])
    b["nspax"] = int(best["nspax"][j])
    return b


def stage2_best(tag, t):
    """第 2 段(星系)的最佳解。沒跑過第 2 段就回傳 None。"""
    f = STEP04B / f"scan2_id{t}_{tag}.npz"
    if not f.exists():
        return None
    sc = np.load(f)
    i  = int(np.argmin(sc["chi2"]))
    return {k: sc[k][i]
            for k in ("template", "z", "A", "chi2", "red_chi2", "n_good")}


def model_of(b, lam_vac):
    """把存下來的解重建成模型光譜,由 template 欄位決定用哪一種。

    'eigen'  Bolton 本徵譜:4 條的線性組合,係數在 A 的前 4 欄。高階成分會跨過
             零點,所以係數可以是負的 —— 那不是錯誤。
    '023'    一條 SDSS 模板(恆星 000-022、星系 023-028 都走這條路):單一光譜
             乘上一個非負的振幅。
    """
    name = str(b["template"])
    if name == "eigen":
        T = redshift_to_grid(load_eigen_galaxy(EIGEN_GAL), float(b["z"]), lam_vac)
        return T @ np.asarray(b["A"][:T.shape[1]], float)
    spl = load_sdss_template(TPL_DIR / f"spDR2-{int(name):03d}.fit")
    return float(b["A"][0]) * redshift_to_grid(spl, float(b["z"]), lam_vac)


def gal_name(g):
    """星系模型在圖上的名字。"""
    n = str(g["template"])
    return ("galaxy eigenspectra" if n == "eigen"
            else f"galaxy template {n} ({GAL_TYPE.get(int(n), '?')})")


def draw_source(fig, cell, D, t, b, args, xlim, legend=False, g=None):
    """在版面的一格裡畫一個源,五列。

    上兩列是恆星模型(流量 + 殘差),中間兩列是星系模型(流量 + 殘差),最下面
    一列是兩者共用的 σ。分成上下兩塊而不是把兩個模型疊在同一格,是因為分類
    比的是「同一段資料,兩個模型各自配得多好」—— 上下對照才讀得出哪一個在
    哪一段波長開始不行;疊在一起時,線多的地方互相遮蔽,而那正是有資訊的地方。

    兩個流量列共用 y 範圍、兩個殘差列也共用 —— 不共用的話,眼睛比的是軸的
    縮放而不是資料。

    xlim 決定橫軸範圍。同一個源會畫兩次:第一節只畫擬合視窗(進 chi2 的通道
    在頁面上攤得開,看得出模板的吸收線對不對得上),第二節畫全波段(看視窗
    以外的地方模型有沒有跑掉)。
    """
    y, sig, ok = source_spectrum(D, t, args.s_fix)
    mod = model_of(b, D["wl_vac"])
    gal = model_of(g, D["wl_vac"]) if g is not None else None
    wl  = D["wl_air"]

    with np.errstate(invalid="ignore", divide="ignore"):
        res  = (y - mod) / sig
        resg = (y - gal) / sig if gal is not None else None

    rs  = float(b["red_chi2"])
    rg  = float(g["red_chi2"]) if g is not None else np.nan
    win = "STAR" if (g is None or rs < rg) else "GALAXY"

    # 沒有第 2 段的結果時退回三列(流量 + 殘差 + σ),不留空格子。
    n_model = 2 if gal is not None else 1
    ratios  = [3, 1.2] * n_model + [1]
    inner = cell.subgridspec(len(ratios), 1, height_ratios=ratios, hspace=0.06)
    axes  = [fig.add_subplot(inner[0])]
    axes += [fig.add_subplot(inner[i], sharex=axes[0])
             for i in range(1, len(ratios))]
    axs   = axes[-1]

    # 兩個流量列共用的 y 範圍。只由進 chi2 的通道決定 —— 紅端的天光線殘差
    # 會衝到幾百,拿它定範圍會把源壓成一條線。
    stack = [y[ok], mod[ok]] + ([gal[ok]] if gal is not None else [])
    q   = np.nanpercentile(np.concatenate(stack), [0.5, 99.5])
    pad = 0.25 * (q[1] - q[0]) + 1e-6
    # 兩個殘差列共用的 y 範圍,而且左右對稱 —— 殘差的重點是「偏哪一邊、偏多少」,
    # 不對稱的軸會讓一面看起來比另一面嚴重。
    allres = np.abs(res[ok]) if resg is None else \
        np.concatenate([np.abs(res[ok]), np.abs(resg[ok])])
    rr = float(np.nanpercentile(allres, 99.5)) * 1.25 if ok.any() else 5.0

    obs = ("observed" if args.s_fix == 0 else
           f"observed − {args.s_fix}·C$_{{sky}}$")
    shade = xlim[1] > args.star_window[1]      # 只有全波段那一節要標出視窗

    blocks = [("star", mod, res, STAR_COLOR, rs,
               f"stellar template {int(b['template']):03d} "
               f"({STAR_TYPE[int(b['template'])]})")]
    if gal is not None:
        blocks.append(("galaxy", gal, resg, GAL_COLOR, rg,
                       f"{gal_name(g)}, z = {float(g['z']):.4f}"))

    for k, (kind, m, r, col, rchi, name) in enumerate(blocks):
        ax, axr = axes[2 * k], axes[2 * k + 1]

        # ---- 流量 ----
        if shade:
            ax.axvspan(*args.star_window, color="#ffd966", alpha=0.18, lw=0)
        ax.plot(wl, y, lw=0.4, color="0.72", rasterized=True, label=obs)
        # 模型畫兩次:淡的一條畫全段(看得出模型在視窗外長什麼樣),濃的一條只畫
        # 進 chi2 的通道。缺口就是被排除的天光線通道 —— 用同一條線的濃淡表示,
        # 不必另外引入一個顏色。
        ax.plot(wl, m, lw=0.5, color=col, alpha=0.3, rasterized=True)
        ax.plot(wl, np.where(ok, m, np.nan), lw=0.9, color=col, rasterized=True,
                label=f"{name}   reduced chi2 = {rchi:.2f}")
        ax.set_xlim(*xlim)
        ax.set_ylim(q[0] - pad, q[1] + pad)
        ax.tick_params(labelbottom=False, labelsize=6)
        ax.grid(alpha=0.25)
        ax.set_ylabel("flux [$10^{-20}$]", fontsize=6.5)
        # 圖例每一塊都畫 —— 兩塊的線是不同的模型,只在第一塊標會讓下半失去說明。
        ax.legend(fontsize=5.8, ncol=2, loc="upper right", framealpha=0.85)
        if k == 0:
            ax.set_title(f"ID {t}   nspax {b['nspax']}   ->  {win}"
                         + (f"   (star {rs:.2f} vs galaxy {rg:.2f})"
                            if gal is not None else ""),
                         fontsize=8.5, pad=3)

        # ---- 殘差 (obs − model)/σ:chi2 的逐項開根號 ----
        # 教授說的兩種大 chi2,數字上分不出來,殘差的形狀一眼就分得出來:
        #     模型錯     -> 殘差寬、連貫、整段偏一邊
        #     模型不夠精 -> 殘差窄、集中在吸收線的線心、正負交錯
        if shade:
            axr.axvspan(*args.star_window, color="#ffd966", alpha=0.18, lw=0)
        axr.axhline(0, color="0.3", lw=0.5)
        # ±1σ 的參考線:眼睛需要一把尺才知道殘差算大還是算小。模型完美且 σ
        # 正確時,應該有約 68% 的點落在這兩條線之間。
        for h in (-1, 1):
            axr.axhline(h, color="0.6", lw=0.4, ls=":")
        axr.plot(wl, np.where(ok, r, np.nan), lw=0.35, color=col, rasterized=True,
                 label=f"{kind} residual")
        axr.set_ylim(-rr, rr)
        axr.tick_params(labelbottom=False, labelsize=6)
        axr.grid(alpha=0.25)
        axr.set_ylabel(r"(obs$-$mod)/$\sigma$", fontsize=6)
        axr.legend(fontsize=5.8, loc="upper right", framealpha=0.85)

    # σ:chi2 的分母。天光線處遠高於連續譜,線性軸會把連續譜區壓扁 -> log。
    axs.plot(wl, sig, lw=0.4, color="0.72", rasterized=True)
    axs.plot(wl, np.where(ok, sig, np.nan), lw=0.5, color="0.25", rasterized=True,
             label=fr"$\sigma$, {int(ok.sum())} channels in chi2")
    axs.set_yscale("log")
    # y 範圍只由「畫得出來的那一段」決定。sharex 只同步橫軸,縱軸的自動縮放
    # 會把整條 σ(含紅端的天光線尖峰)算進去 —— 那會讓視窗那一節的 σ 被壓成
    # 一條線,而那一節的重點正好是視窗裡的 σ。
    vis = ((wl >= xlim[0]) & (wl <= xlim[1]) & np.isfinite(sig) & (sig > 0))
    if vis.any():
        axs.set_ylim(float(np.nanpercentile(sig[vis], 0.5)) * 0.7,
                     float(np.nanpercentile(sig[vis], 99.9)) * 1.4)
    # log 軸的預設刻度是給整張圖用的:這一列只有 1 吋高,預設會把 4×10⁰、6×10⁰
    # 這種次要刻度全部標上去,標籤互相疊成一團。
    # 只留 1/2/5 三個位置,而且用普通數字寫(0.5 1 2 5 10),不用科學記號 ——
    # 這一列要傳達的是量級,不是精確值。
    axs.yaxis.set_major_locator(
        matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=6))
    axs.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axs.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{v:g}"))
    axs.tick_params(labelsize=6)
    axs.grid(alpha=0.25)
    axs.set_ylabel(r"$\sigma$ [$10^{-20}$]", fontsize=6.5)
    axs.set_xlabel("wavelength [$\\AA$]", fontsize=7, labelpad=1)
    axs.legend(fontsize=5.8, loc="upper right", framealpha=0.85)


def cover(pdf, rows, args, tag):
    """第一頁:左邊 source ID map,右邊逐源的數值表。

    後面每一頁的標題只有 ID,光看數字沒辦法知道那是天上的哪一塊。這一頁的
    作用就是把 ID 和位置綁在一起,讓人翻到第 8 頁看到 ID 24 時可以回來查。
    """
    fig = plt.figure(figsize=A4_LAND)

    # ---- 左:白光圖 + segmentation 輪廓 + ID ----
    ax    = fig.add_axes((0.05, 0.045, 0.55, 0.92))
    white = fits.getdata(STEP01 / "whitelight.fits").astype(float)
    seg   = fits.getdata(STEP01 / "seg.fits")

    # asinh 拉伸。白光圖的動態範圍約 10³(Haro 11 本體 vs 暗源),線性只看得到
    # 本體;log 又處理不了扣過天空後的負值。asinh 在 0 附近是線性、在大值趨近
    # log,兩邊都保得住。除以 q20 是把「一個典型的背景值」定為拉伸的尺度。
    q0 = max(float(np.nanpercentile(white[white != 0], 20)), 1e-3)
    d  = np.arcsinh(white / q0)
    # 顯示範圍取拉伸後的 30–99.7 百分位。下限用 5% 的話,整張圖的背景會擠在
    # 色階最亮的 1/4,暗源和背景分不開;Haro 11 本體在上限外會飽和成純黑,
    # 但它的形狀已經由藍色輪廓交代了,不需要靠灰階去讀。
    ax.imshow(d, origin="lower", cmap="gray_r",
              vmin=float(np.nanpercentile(d, 30)),
              vmax=float(np.nanpercentile(d, 99.7)))
    # 輪廓而不是填色:填色會蓋掉源本身的形狀,而形狀正是要對照的東西。
    ax.contour(seg > 0, levels=[0.5], colors="#1f77b4", linewidths=0.45)

    ny = seg.shape[0]
    for r in rows:
        yy, xx = np.nonzero(seg == r["id"])
        if yy.size == 0:
            continue
        # 標籤不寫在源身上,往外挪 8 點。最小的源在圖上只有幾個像素寬,
        # 字直接壓上去會把源整個蓋掉,看起來像「這裡標了一個編號卻沒有東西」。
        # 靠近上緣的源改成往下標,否則字會跑到圖框外面。
        up = yy.mean() < 0.92 * ny
        ax.annotate(str(r["id"]), (xx.mean(), yy.mean()),
                    xytext=(0, 8 if up else -8), textcoords="offset points",
                    fontsize=6.5, color="#d62728", fontweight="bold",
                    ha="center", va="bottom" if up else "top",
                    # 白色描邊 —— 標籤會同時落在亮的源上和暗的背景上,單一顏色
                    # 一定有一邊看不見。
                    path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])
    ax.set_xlabel("x [pix]", fontsize=7)
    ax.set_ylabel("y [pix]", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("source ID map", fontsize=9)

    # ---- 右:數值表。兩個模型並排,最後一欄是判定 ----
    L = [f"{'ID':>4}{'nspax':>7}{'tpl':>6}{'type':>7}"
         f"{'star X2/dof':>12}{'gal z':>9}{'gal X2/dof':>11}{'class':>8}", "-" * 64]
    ns = 0
    for r in sorted(rows, key=lambda r: r["id"]):
        i, gg = int(r["template"]), r.get("gal")
        rs = float(r["red_chi2"])
        rg = float(gg["red_chi2"]) if gg is not None else np.nan
        cls = "-" if gg is None else ("STAR" if rs < rg else "galaxy")
        ns += cls == "STAR"
        L.append(f"{r['id']:>4}{r['nspax']:>7}{i:>6d}{STAR_TYPE[i]:>7}{rs:>12.2f}"
                 + (f"{float(gg['z']):>9.4f}{rg:>11.2f}" if gg is not None
                    else f"{'-':>9}{'-':>11}") + f"{cls:>8}")
    L += ["", f"  {ns} star / {len(rows) - ns} galaxy"
              "   (lower reduced chi2 on the same channels wins)"]
    fig.text(0.628, 0.975, "\n".join(L), family="monospace", fontsize=6.8,
             va="top", ha="left", linespacing=1.34)

    pdf.savefig(fig)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="把每個源的擬合圖合成一份 PDF")
    ap.add_argument("--star-window", type=float, nargs=2, default=STAR_WINDOW)
    ap.add_argument("--gal-window", type=float, nargs=2, default=GAL_WINDOW)
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--sky-basis", action="store_true")
    ap.add_argument("--aperture", action="store_true")
    ap.add_argument("--raw-mask", action="store_true")
    ap.add_argument("--iter", type=int, default=1, help="用第幾輪天光線遮罩的結果")
    ap.add_argument("--s-fix", type=float, default=1.0)
    ap.add_argument("--per-page", type=int, default=1,
                    help="一頁畫幾個源。一個源現在佔五列(恆星流量/殘差、"
                         "星系流量/殘差、σ),一頁放兩個的話每一列只剩 0.8 吋")
    ap.add_argument("--all-ids", action="store_true",
                    help="保留給相容性,現在預設就是全部 —— "
                         "其餘的是雜訊尖峰/宇宙線/探測器條紋")
    ap.add_argument("--dpi", type=int, default=200, help="點陣化資料線的解析度")
    ap.add_argument("--spec-dir", default=None,
                    help="源光譜的目錄,要和 step4b_window_fit.py 跑時用的一致,"
                         "例如 results/skymodel/step02_eso")
    ap.add_argument("--gal-model", choices=["eigen", "sdss"], default="eigen",
                    help="要和 step4b_window_fit.py 跑時用的一致")
    ap.add_argument("--out", default=None, help="輸出檔名,預設由 tag 決定")
    args = ap.parse_args()

    # tag 的組法必須和 step4b_window_fit.py 逐字相同,否則會讀到別的檔案。
    suffix = (f"_{Path(args.spec_dir).name.replace('step02', '')}"
              if args.spec_dir else "") \
             + ("_galtpl" if args.gal_model == "sdss" else "")
    tag  = make_tag(args.basis, args.K, args.s_fix, args.star_window,
                    args.gal_window, args.sky_basis, args.iter,
                    not args.raw_mask, args.aperture, suffix)
    bf   = STEP04B / f"best_{tag}.npz"
    if not bf.exists():
        raise SystemExit(f"找不到 {bf.name}。先跑 step4b_window_fit.py")
    best = np.load(bf)
    D    = load_common(Namespace(**vars(args)))

    targets = best["id"].tolist()

    rows = []
    for t in targets:
        b = stage1_best(tag, t, best)
        if b is None:
            print(f"ID {t}: 找不到 scan1,跳過")
            continue
        b["id"]  = t
        b["gal"] = stage2_best(tag, t)      # 沒跑第 2 段就是 None,圖只畫恆星
        rows.append(b)

    # 檔名要能分辨「只畫真源」和「全偵測」兩版 —— 兩份的頁數和結論都不同,
    # 同名互相覆蓋的話,看不出手上這份是哪一種。
    out = (Path(args.out) if args.out else
           FIGURES / f"step4b_{tag}{'' if args.all_ids else '_keep11'}.pdf")
    out.parent.mkdir(parents=True, exist_ok=True)

    # 兩節,每節都把所有源走完再換下一節:先只畫擬合視窗,再畫全波段。
    # 一個源的兩張圖分開在兩節,而不是相鄰兩頁 —— 這樣「所有源在同一個尺度下
    # 排在一起」,翻過去比較的是源與源,不是同一個源的兩種畫法。
    wl       = D["wl_air"]
    sections = [tuple(args.star_window), (float(wl.min()), float(wl.max()))]

    with PdfPages(out) as pdf:
        cover(pdf, rows, args, tag)
        npage = 1
        for xlim in sections:
            for p0 in range(0, len(rows), args.per_page):
                page = rows[p0:p0 + args.per_page]
                fig  = plt.figure(figsize=A4_LAND)
                # hspace 給的是「格與格之間」的空隙 —— 每一格裡面還要塞標題,
                # 所以留得比預設大。左右邊界留窄一點,波長軸能拉多長就多長。
                gs   = fig.add_gridspec(args.per_page, 1, hspace=0.55,
                                        left=0.055, right=0.985, top=0.955,
                                        bottom=0.065)
                for k, r in enumerate(page):
                    draw_source(fig, gs[k], D, r["id"], r, args, xlim,
                                legend=(k == 0), g=r.get("gal"))
                pdf.savefig(fig, dpi=args.dpi)
                plt.close(fig)
                npage += 1
                print(f"page {npage}  [{xlim[0]:.0f}-{xlim[1]:.0f}]  "
                      + " ".join(f"ID{r['id']}" for r in page), flush=True)

    print(f"\n{len(rows)} sources × {len(sections)} sections   {npage} pages   "
          f"{out.stat().st_size / 1e6:.1f} MB")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
