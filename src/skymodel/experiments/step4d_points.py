"""特定幾個點 / 幾個方框上,step4d 的天空模型有沒有把天空扣得更乾淨。

區域平均(step4d_sky_status)會把兩件事混成同一個數字:「改善集中在少數 spaxel」
和「所有 spaxel 均勻改善一點」。Haro 11 的 −32% 到底是哪一種,只有看個別
spaxel 才分得出來。

兩種粒度,各一張圖:

    (a) 逐 spaxel   挑幾個具體的 spaxel,每個一列。看得到單一光譜的細節,
                    但單一 spaxel 的雜訊大,趨勢要靠 (b)。
    (b) 空間方框    幾個具名的方框取平均。統計穩,但仍會平均掉個別差異。

殘差的定義和 sky_status 一致:blank 只扣天空,源區連源模型一起扣,兩邊的
理想值都是 0。源的振幅對每個天空模型各自重解 —— 換了天空模型,源的解也會變,
沿用同一組振幅等於偷偷把兩個模型混在一起。

灰色的 sqrt(STAT) 是該 spaxel 的光子雜訊下限:殘差貼著它就代表模型已經到頂,
剩下的是雜訊,再改模型也沒有用。

**這是 in-sample 的比較** —— 兩個天空模型都用了這些 spaxel(step4d 還多用了
源區)去訓練,所以殘差小不代表預測能力好。要嚴謹必須留出檢查用的 spaxel
不參與訓練。

    conda run -n astro python src/skymodel/experiments/step4d_points.py \\
        --basis svd -K 30 \\
        --best results/skymodel/step04b/classification_nobasis_s0.0_4700-8000_4700-8000_L1cum__eso.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step5_fit_spaxels import (build_templates, fit_blank, fit_source,
                               CUBE, STEP01, STEP03)
from templates import air_to_vacuum
from utils import spectrum_stats

ROOT    = Path(__file__).resolve().parents[3]
STEP04D = ROOT / "results/skymodel/step04d"
FIGURES = ROOT / "results/skymodel/figures"

# 顏色編的是**角色**不是方法:每一格都是「參照(深) vs 我們要看的那條(淡藍)」,
# 三格的對比強度因此一致,線寬也全部相同。哪條是哪個方法由圖例交代。
# 兩個比較基準(ESO、精煉前)都用黑色,要看的那條一律淡藍 —— 三格的
# 視覺語言完全一致,不必每格重新對照哪個顏色代表什麼。
REF_COL = {"ESO nosky": "#000000", "before refine": "#000000"}
SUB_COL = "#6baed6"
COL = {"ESO nosky": "#000000", "before refine": "#000000",
       "after refine": "#6baed6"}
ESONOSKY = ROOT / "data/Haro11_NEpointing_esonosky.fits"


def residual(D, V, idx, seg_f, sky, T_all, s_fix, two_stage=True):
    """這些 spaxel 扣完模型之後剩下什麼。每個方法都扣掉「它自己聲稱有模型的東西」。

    sky 給定    我們的做法:天空由 basis 描述,源區再加上源模型
    sky = None  資料已經扣過天空(ESO nosky),只需要再扣源模型 —— 這樣三個
                方法的理想值都是 0,才比得起來

    每個 spaxel 依所屬的 segmentation ID 分組 —— 同一組共用一條模板。
    """
    out = np.full((D.shape[0], idx.size), np.nan)
    for rid in np.unique(seg_f[idx]):
        sel  = np.flatnonzero(seg_f[idx] == rid)
        cols = idx[sel]
        T = T_all.get(int(rid))
        if sky is None:
            if rid == 0 or T is None:
                out[:, sel] = D[:, cols]          # blank:ESO 已經扣完,直接就是殘差
                continue
            # 只解源的振幅:設計矩陣就是模板本身,以 1/sigma 加權
            for c_, j in zip(sel, cols):
                g = (np.isfinite(D[:, j]) & np.isfinite(V[:, j]) & (V[:, j] > 0)
                     & np.all(np.isfinite(T), axis=1))
                if g.sum() <= T.shape[1]:
                    continue
                w = 1 / np.sqrt(V[g, j].astype(np.float64))
                a, *_ = np.linalg.lstsq(T[g] * w[:, None],
                                        D[g, j].astype(np.float64) * w, rcond=None)
                r = np.full(D.shape[0], np.nan)
                r[g] = D[g, j] - T[g] @ a
                out[:, c_] = r
            continue
        if rid == 0 or T is None:
            c = fit_blank(D[:, cols], sky)
            out[:, sel] = D[:, cols] - sky.T @ c
        else:
            # 階段 1:源和天空一起解,s 固定 —— A·T 和 s·C_sky 的形狀都平滑,
            # 同時放自由的話天空會吸走源的光,源就扣得不夠。
            co  = fit_source(D[:, cols], V[:, cols], sky, T, s_fix=s_fix)
            src = T @ np.nan_to_num(co[:T.shape[1]])
            src = np.where(np.all(np.isfinite(T), axis=1)[:, None], src, np.nan)
            if not two_stage:
                model = (sky[0][:, None] * co[4][None, :] + sky[1:].T @ co[5:] + src)
                out[:, sel] = D[:, cols] - model
                continue
            # 階段 2:源已經扣掉,設計矩陣裡沒有 A·T 了 —— 簡併不存在,
            # s 沒有理由再被綁住。這一步和 blank 用完全相同的處理。
            sky_only = D[:, cols] - src
            c2 = fit_blank(sky_only, sky)
            out[:, sel] = sky_only - sky.T @ c2
    return out


def draw_box_map(white2, seg2, boxes, out_path, points=None, title="box locations"):
    """方框(或單點)畫在白光圖上 —— 只報座標的話,看的人沒辦法判斷那是天上的哪一塊。

    拉伸與輪廓的做法和 source ID map 一致,兩張圖並排時不必重新學怎麼讀。
    """
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(8.5, 8))
    q0 = max(float(np.nanpercentile(white2[white2 != 0], 20)), 1e-3)
    d  = np.arcsinh(white2 / q0)
    ax.imshow(d, origin="lower", cmap="gray_r",
              vmin=float(np.nanpercentile(d, 30)),
              vmax=float(np.nanpercentile(d, 99.7)))
    ax.contour(seg2 > 0, levels=[0.5], colors="#1f77b4", linewidths=0.45)
    cmapb = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (nm, (y0, y1, x0, x1)) in enumerate(boxes.items()):
        c = cmapb[i % 10]
        ax.add_patch(mpatches.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0 + 1,
                                        y1 - y0 + 1, fill=False, ec=c, lw=1.8))
        ax.annotate(nm, (x0 + (x1 - x0) / 2, y1), xytext=(0, 5),
                    textcoords="offset points", color=c, fontsize=9,
                    fontweight="bold", ha="center",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    # points:{名稱: (x, y)}。單點只有一個像素,畫成十字才看得見;標籤往外
    # 挪並加白色描邊 —— Haro 11 的三個點彼此很近,不挪會疊在一起。
    # 標籤用短代號:19 個點的全名一定互相疊。全名留在 PDF 每一頁的標題裡,
    # 這張圖只負責「哪個代號在哪裡」。顏色用 modulo 循環,超過 10 個不會被截斷。
    def short(nm):
        if nm.startswith("Haro"):
            return "H" + nm.split()[-1].rstrip("%")
        if nm.startswith("source"):
            return nm.split()[1]
        if nm.startswith("blank"):
            return "b" + nm.split("#")[1].split()[0]
        return nm[:6]

    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, (nm, (x, y)) in enumerate((points or {}).items()):
        c = cmap[i % 20]
        ax.plot(x, y, "+", ms=11, mew=1.8, color=c)
        ax.annotate(short(nm), (x, y), xytext=(7, 6 - 12 * (i % 2)),
                    textcoords="offset points", color=c, fontsize=8,
                    fontweight="bold", ha="left",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.set_xlabel("x [pix]"); ax.set_ylabel("y [pix]")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def draw(rows, wl, out_path, title):
    """一頁一個對象,兩格,每格都是「我們的其中一版 vs ESO nosky」。

    ESO 當共同參照出現在兩格裡,所以兩格之間比的是「精煉前後誰更接近 ESO
    的行為」,而不是三條互相纏在一起。兩條線都給透明度 —— 它們大部分時間
    幾乎重合,不透明的話後畫的會把先畫的整條蓋掉,而重合處正是要看的地方。

    兩格共用 y 範圍,輸出成 PDF、一頁一個對象。
    """
    from matplotlib.backends.backend_pdf import PdfPages
    keys = ("mean", "sigma", "skewness", "kurtosis", "rms_from_zero")
    with PdfPages(out_path) as pdf:
        for name, curves, _ in rows:
            fig = plt.figure(figsize=(14, 8.6))
            d = dict(curves)
            ref = "ESO nosky"
            # 三格:前兩格各自和 ESO 比(共同參照),第三格把我們的兩版直接對打
            # —— 前兩格看「相對於外部基準各在哪」,第三格看「精煉到底改了什麼」。
            pairs = [(ref, "before refine"), (ref, "after refine"),
                     ("before refine", "after refine")]
            pairs = [(a, b) for a, b in pairs if a in d and b in d]
            gs  = fig.add_gridspec(len(pairs), 2, width_ratios=[6, 1.45],
                                   hspace=0.08, wspace=0.02, left=0.06,
                                   right=0.985, top=0.90, bottom=0.07)
            # y 範圍由所有曲線共同決定 —— 各自縮放的話,眼睛比的是軸不是資料
            v  = np.concatenate([np.abs(r[np.isfinite(r)]) for _, r in curves])
            hi = float(np.percentile(v, 99.5)) * 1.4
            axes = []
            for k, (bg, fg) in enumerate(pairs):
                ax = fig.add_subplot(gs[k, 0], sharex=axes[0] if axes else None)
                axes.append(ax)
                ax.axhline(0, color="0.5", lw=0.6)
                # 底下那條畫粗、上面那條畫細 —— 同粗細疊在一起時,不管透明度
                # 怎麼調都會糊成一片。灰色當底時不必再淡化,有顏色時要。
                # 兩條都半透明:不透明時後畫的會把先畫的整條蓋掉,而兩者
                # 大部分時間幾乎重合 —— 重合處正是要看的地方。
                ax.plot(wl, d[bg], lw=0.6, color=REF_COL[bg], alpha=0.65,
                        label=bg)
                ax.plot(wl, d[fg], lw=0.6, color=SUB_COL, alpha=0.65, label=fg)
                ax.set_ylim(-hi, hi)
                ax.set_ylabel("residual", fontsize=8)
                ax.grid(alpha=0.25)
                ax.tick_params(labelsize=7, labelbottom=(k == len(pairs) - 1))
                ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.85)
            axes[-1].set_xlabel("wavelength [$\\AA$]")

            sax = fig.add_subplot(gs[:, 1]); sax.axis("off")
            # 標籤靠左對齊,整塊表才貼著左緣 —— 右對齊會把它整體推到右邊,
            # 和左邊的圖之間留下一片空白。
            sax.text(0.02, 0.98, "\n".join([""] + [f"{k:<13}" for k in keys]),
                     va="top", family="monospace", fontsize=8,
                     transform=sax.transAxes)
            # 一個方法一欄,和曲線同色。三欄橫向並排,同一個統計量在同一列。
            for j, (m, r) in enumerate(curves):
                st = spectrum_stats(r)
                # 欄位起點按字元寬算:標籤 13 字 + 三欄各 9 字 = 40 字,
                # 一欄佔 9/40 = 0.225,起點 0.33 剛好接在標籤後面。
                # 表格一律黑字:顏色的意義由每一格的圖例交代,表格再上一套
                # 顏色只會多一層要記的對應。
                sax.text(0.35 + 0.225 * j, 0.98,
                         "\n".join([f"{m.split()[0]:>9}"]
                                   + [f"{st[k]:>9.3f}" for k in keys]),
                         va="top", ha="left", family="monospace",
                         fontsize=8, transform=sax.transAxes)
            fig.suptitle(f"{title}\n{name}", fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description="特定 spaxel / 方框的 sky status")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--best", required=True)
    ap.add_argument("--s-fix", type=float, default=1.0)
    ap.add_argument("--one-stage", action="store_true",
                    help="源區用單階段(s 固定 1.0 解到底)。預設兩階段:先固定 s "
                         "解源的振幅,扣掉源之後再讓 s 自由重解天空")
    ap.add_argument("--n-haro", type=int, default=6,
                    help="Haro 11 內沿亮度分位取幾個 spaxel")
    ap.add_argument("--n-blank-points", type=int, default=5,
                    help="沿距離取幾個 blank spaxel")
    ap.add_argument("--n-blank-boxes", type=int, default=5,
                    help="沿距離均勻取幾個 blank 方框")
    ap.add_argument("--half", type=int, default=6,
                    help="方框的半寬(px),方框大小 = 2*half+1")
    args = ap.parse_args()

    wl    = np.load(STEP03 / "wavelength.npy")
    seg2  = fits.getdata(STEP01 / "seg.fits")
    white2 = fits.getdata(STEP01 / "whitelight.fits")
    ny, nx = seg2.shape
    seg_f, white = seg2.reshape(-1), white2.reshape(-1)
    skies = {
        "before refine":  np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                             np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")]),
        "after refine": np.vstack([np.load(STEP04D / "sky_continuum.npy"),
                             np.load(STEP04D / f"sky_basis_{args.basis}_K{args.K}.npy")]),
    }
    best = np.load(args.best)

    with fits.open(CUBE, memmap=True) as h:
        D = np.asarray(h["DATA"].data, np.float32)
        V = np.asarray(h["STAT"].data, np.float32)
    nz = D.shape[0]
    D, V = D.reshape(nz, -1), V.reshape(nz, -1)
    with fits.open(ESONOSKY, memmap=True) as h:
        E  = np.asarray(h["DATA"].data, np.float32).reshape(nz, -1)
        VE = np.asarray(h["STAT"].data, np.float32).reshape(nz, -1)
    valid = (white != 0) & np.isfinite(D).all(axis=0)
    T_all = build_templates(best, air_to_vacuum(wl))
    # 三個方法:資料、變異數、天空模型(ESO 的天空已經扣掉,所以是 None)
    runs = {"before refine": (D, V, skies["before refine"]),
            "after refine":  (D, V, skies["after refine"]),
            "ESO nosky":     (E, VE, None)}

    # ---------------- (a) 逐 spaxel ----------------
    # 挑選依白光亮度的分位,不是隨手指座標 —— 這樣「核心/halo/邊緣」的意義
    # 由亮度定義,換一份資料也選得出對應的位置。四類各自取樣:
    #   Haro 11   亮度分位掃過整個範圍(它一個源就佔 12,126 個 spaxel)
    #   其他源    各取最亮的一個,看改善是不是只發生在 Haro 11 身上
    #   blank     沿「離 Haro 11 中心的距離」均勻取,對照距離趨勢
    picks = []
    h1 = np.flatnonzero(valid & (seg_f == 1))
    order = h1[np.argsort(white[h1])]
    for q in np.linspace(1.0, 0.0, args.n_haro):
        j = int(round(q * (order.size - 1)))
        picks.append((f"Haro 11  brightness {100 * q:.0f}%", int(order[j])))

    for t in [int(i) for i in best["id"]]:
        if t == 1:
            continue
        ii = np.flatnonzero(valid & (seg_f == t))
        if ii.size:
            picks.append((f"source #{t} (brightest)", int(ii[np.argmax(white[ii])])))

    bl = np.flatnonzero(valid & (seg_f == 0))
    cy0, cx0 = divmod(int(order[-1]), nx)
    by, bx = np.divmod(bl, nx)
    dist = np.hypot(by - cy0, bx - cx0)
    for i, q in enumerate(np.linspace(0, 1, args.n_blank_points)):
        j = int(np.argmin(np.abs(dist - (dist.min() + q * np.ptp(dist)))))
        y, x = divmod(int(bl[j]), nx)
        picks.append((f"blank #{i + 1} ({x}, {y})", int(bl[j])))
        print(f"blank point #{i + 1}  (x, y) = ({x}, {y})   "
              f"離 Haro 11 中心 {dist[j]:.0f} px")

    idx = np.array([p for _, p in picks])
    draw_box_map(white2, seg2, {},
                 FIGURES / "step4d_point_map.png",
                 points={nm: tuple(reversed(divmod(int(p), nx)))
                         for nm, p in picks},
                 title="individual spaxels")
    res = {m: residual(dat, var, idx, seg_f, sky, T_all, args.s_fix,
                            two_stage=not args.one_stage)
           for m, (dat, var, sky) in runs.items()}
    rows = []
    for k, (nm, p) in enumerate(picks):
        y, x = divmod(int(p), nx)
        rows.append((f"{nm}   (x, y) = ({x}, {y})",
                     [(m, res[m][:, k]) for m in runs], None))
        for m in runs:
            s = spectrum_stats(res[m][:, k])
            print(f"{nm:<28}{m:>7}  rms {s['rms_from_zero']:.4f}  "
                  f"mean {s['mean']:+.4f}  sigma {s['sigma']:.4f}")
    # 檔名帶上 s 的處理方式 —— 兩者是不同的科學產物(單階段保住源但殘差有
    # 系統性偏移,兩階段偏移消失但吃掉 11% 的源流量),必須並存才比得起來。
    sfx = "sfix" if args.one_stage else "sfree"
    draw(rows, wl, FIGURES / f"step4d_points_{sfx}.pdf",
         f"sky status at individual spaxels   [{args.basis} K={args.K}]")

    # ---------------- (b) 空間方框 ----------------
    # 方框用「內容」定義,不用盲目的偏移量:先前用固定偏移取的 blank near
    # 其實整個落在 Haro 11 的 segmentation 裡面,標籤和內容不符。
    # core/halo 由 seg == 1 內的亮度分位決定,blank 則要求整框 seg == 0 且有效,
    # 再依離 Haro 11 中心的距離取最近與最遠。
    cy, cx = divmod(int(order[-1]), nx)
    hw = args.half
    v2 = valid.reshape(ny, nx)
    s2 = seg2

    def box_at(y, x):
        return (y - hw, y + hw, x - hw, x + hw)

    # halo:seg == 1 裡亮度落在中位附近的位置
    hy, hx = divmod(int(order[order.size // 2]), nx)

    # blank:掃描所有可能的框心,只留整框都是 blank 且有效的
    ys, xs = np.mgrid[hw:ny - hw, hw:nx - hw]
    cand = []
    for y, x in zip(ys.ravel()[::3], xs.ravel()[::3]):
        sl = (slice(y - hw, y + hw + 1), slice(x - hw, x + hw + 1))
        if (s2[sl] == 0).all() and v2[sl].all():
            cand.append((np.hypot(y - cy, x - cx), y, x))
    cand.sort()
    boxes = {"core": box_at(cy, cx), "halo": box_at(hy, hx)}

    # 其他兩個源也各取一框:恆星 #35 和最大的星系 #24。它們的源模型和
    # Haro 11 完全不同,可以看改善是不是只發生在 Haro 11 身上。
    for t, nm in ((35, "star #35"), (24, "source #24")):
        ii = np.flatnonzero(valid & (seg_f == t))
        if ii.size:
            y, x = divmod(int(ii[np.argmax(white[ii])]), nx)
            boxes[nm] = box_at(min(max(y, hw), ny - hw - 1),
                               min(max(x, hw), nx - hw - 1))

    # blank 沿「離 Haro 11 中心的距離」均勻取幾個 —— 我的說法是「越近改善越大」,
    # 那就該讓距離本身變成圖上的一個變數,而不是只取最近與最遠兩點。
    if cand:
        # 名字只放編號與框心座標。先前把「離 Haro 11 的距離」放進名字,結果
        # 和標題裡的 spaxel 數並排時被讀成同一類量(86 px 是距離,169 是
        # 13×13 個 spaxel)—— 距離改印在終端機的表格裡。
        d = np.array([c[0] for c in cand])
        for i, q in enumerate(np.linspace(0, 1, args.n_blank_boxes)):
            j = int(np.argmin(np.abs(d - (d.min() + q * (d.max() - d.min())))))
            y, x = cand[j][1], cand[j][2]
            boxes[f"blank #{i + 1} ({x}, {y})"] = box_at(y, x)
            print(f"blank #{i + 1}  框心 (x, y) = ({x}, {y})   "
                  f"離 Haro 11 中心 {d[j]:.0f} px")

    boxes = {nm: (max(y0, 0), min(y1, ny - 1), max(x0, 0), min(x1, nx - 1))
             for nm, (y0, y1, x0, x1) in boxes.items()}
    draw_box_map(white2, seg2, boxes, FIGURES / "step4d_box_map.png")

    print()
    rows = []
    for nm, (y0, y1, x0, x1) in boxes.items():
        m2 = np.zeros((ny, nx), bool)
        m2[y0:y1 + 1, x0:x1 + 1] = True
        ii = np.flatnonzero(m2.reshape(-1) & valid)
        if ii.size == 0:
            print(f"{nm}: 方框內沒有有效 spaxel,跳過")
            continue
        r = {m: residual(dat, var, ii, seg_f, sky, T_all, args.s_fix,
                              two_stage=not args.one_stage)
             for m, (dat, var, sky) in runs.items()}
        with np.errstate(invalid="ignore"):
            curves = [(m, np.nanmean(r[m], axis=1)) for m in runs]
        rows.append((f"{nm}   y {y0}-{y1}  x {x0}-{x1}   {ii.size} spaxel",
                     curves, None))
        for m, c in curves:
            s = spectrum_stats(c)
            print(f"{nm:<12}{m:>7}  rms {s['rms_from_zero']:.4f}  "
                  f"mean {s['mean']:+.4f}  sigma {s['sigma']:.4f}")
    draw(rows, wl, FIGURES / f"step4d_boxes_{sfx}.pdf",
         f"sky status in spatial boxes   [{args.basis} K={args.K}]")


if __name__ == "__main__":
    main()
