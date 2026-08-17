"""方框裡的平均光譜 —— 天空扣乾淨了嗎、源有沒有被扣掉。

一個方框一張圖、右邊放統計。畫的是 step5 實際輸出的 sky_subtracted
(= 資料 − 天空),**源留在裡面**,不是 資料 − 天空 − 源模型。

理由是原則 1:只看殘差判斷不了好壞 —— 把源一起扣光,殘差同樣會變小。
留著源,兩件事才在同一張圖上分得開:

    blank 的格   線壓在 0 上   -> 天空扣乾淨了
    源上的格     線是星系光譜  -> 源留住了;線被壓低就是 over-subtraction

而且這裡不重新擬合任何東西,直接讀 step5 寫出來的 cube 取區域平均。圖上看到的
就是使用者拿到的那份資料,沒有「圖是另外算的」這層落差。

三條線
------
    ESO nosky      外部基準(ESO pipeline 的天空扣完)
    s 逐格自由     舊做法:每個 blank spaxel 各自解自己的 s
    s 空間場       新做法:s 換成一張平滑的空間場,源區與 blank 共用同一張

方框怎麼選
----------
用**內容**選,不用手打座標:core/halo 由主源內的白光亮度決定,src edge 由
「離主源的距離」決定,blank 沿距離均勻取。方框必須整框落在該類區域裡(用
minimum_filter 檢查),否則標籤和內容會不符。

方框平均 vs 單點
----------------
--half 是方框的半寬,框寬 = 2*half+1。**--half 0 就是單一 spaxel** —— 選位置的
規則完全一樣,只是不做平均。方框把雜訊壓成 1/sqrt(N),適合看零點;單點很吵,
但它是「這一個位置到底如何」的誠實答案,不靠平均把系統性偏差藏起來。

兩者的檔名相同,所以輸出目錄跟著 --half 走,不會互相覆蓋:

    evaluation/pNN/box/     --half > 0
    evaluation/pNN/point/   --half 0

兩種輸出
--------
    預設(PNG)  一個方框一張圖,三條線疊在一起比。`map.png` 標出位置在視場
               的哪裡。缺 ESO 或舊 run 時用 --eso none / --ref-run none。
    --pdf      一頁一個方框,上格 raw vs sky model、下格殘差。**只用這個 run
               自己的三份資料**(原始 cube / sky_model.fits / sky_subtracted.fits),
               不需要任何外部對照。

    conda run -n astro python src/skymodel/evaluation/box_spectra.py \\
        --work results/skymodel/p01 --run blank_svdK30_tpl60_sfield
    conda run -n astro python src/skymodel/evaluation/box_spectra.py \\
        --work results/skymodel/p01 --run blank_svdK30_tpl60_sfield --half 0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, load_field, pointing_dir, slug  # noqa: E402
from utils import main_source_group, scale, spectrum_stats  # noqa: E402

Z_HARO = 0.0204
LINES  = [("[O III]", 5006.8), ("Ha", 6562.8), ("[S II]", 6716.4)]



# 顏色編的是角色:外部基準橘、舊做法灰、要看的那條藍。
# 標籤一律英文 —— matplotlib 預設字型沒有中文字符,中文會變成一格格的豆腐。
COL = {"ESO nosky": "#ff7f0e", "s per-spaxel (old)": "0.35",
       "s field (new)": "#1f77b4"}
SHORT = {"ESO nosky": "ESO", "s per-spaxel (old)": "old", "s field (new)": "new"}


def cube_data(path):
    """讀 cube 的 DATA。step5 寫在 primary HDU,ESO 的在名為 DATA 的 HDU。"""
    h = fits.open(path, memmap=True)
    hdu = h["DATA"] if "DATA" in h else h[0]
    return h, hdu


def box_mean(hdu, y0, y1, x0, x1):
    """一個方框的平均光譜。NaN 的 spaxel 自動略過(nanmean)。"""
    sub = np.asarray(hdu.data[:, y0:y1 + 1, x0:x1 + 1], np.float32)
    with np.errstate(invalid="ignore"):
        return np.nanmean(sub.reshape(sub.shape[0], -1), axis=1)


def pick_boxes(seg, white, half, n_blank, edge_targets, margin, step04):
    """依內容挑方框,回傳 {名稱: (y0, y1, x0, x1)}(含端點)。"""
    main, ids, peak = main_source_group(seg, white, step04)
    valid = white != 0
    size  = 2 * half + 1

    # 「整框都在某類區域裡」= 該類的布林圖經過 minimum_filter 之後仍為 True。
    # minimum_filter 對布林圖就是「這個視窗裡有沒有 False」的檢查。
    def whole(m):
        return ndimage.minimum_filter(m.astype(np.uint8), size=size).astype(bool)

    in_main  = whole(main & valid)
    d_main   = ndimage.distance_transform_edt(~main)
    # 視野邊緣要避開:那裡曝光次數少,step5 還會把覆蓋率 < 90% 的 spaxel 寫成
    # NaN。方框落在那裡,量到的是拼接的邊界效應,不是天空扣得好不好。
    d_edge   = ndimage.distance_transform_edt(valid)
    in_blank = whole((seg == 0) & valid) & (d_edge > half + margin)

    boxes = {}

    def add(name, cy, cx):
        boxes[name] = (cy - half, cy + half, cx - half, cx + half)

    # --- 主源:最亮的核心、以及仍整框在源內的最暗處(halo) ---
    wm = ndimage.uniform_filter(np.where(valid, white, 0.0), size=size)
    if in_main.any():
        cand = np.flatnonzero(in_main.ravel())
        add("core", *divmod(int(cand[np.argmax(wm.ravel()[cand])]), seg.shape[1]))
        add("halo", *divmod(int(cand[np.argmin(wm.ravel()[cand])]), seg.shape[1]))
    else:                                    # 主源太小,框放不進去 -> 就放在峰值上
        add("core", *peak)

    # --- 源外那一圈:over-subtraction 的案發現場 ---
    # 方框整框在 blank,框心離主源最近只能到 half+1 —— 這已經是「緊貼源邊」。
    cand = np.flatnonzero(in_blank.ravel())
    dm   = d_main.ravel()[cand]
    for t in edge_targets:
        j = int(cand[np.argmin(np.abs(dm - t))])
        add(f"src edge d={d_main.ravel()[j]:.0f}px", *divmod(j, seg.shape[1]))

    # --- 其他源:兩個最亮的非主源,看改善是不是只發生在 Haro 11 身上 ---
    others = [i for i in np.unique(seg[seg > 0]) if i not in ids]
    flux   = {i: float(np.nansum(np.where(seg == i, white, 0))) for i in others}
    n_added = 0
    for i in sorted(flux, key=flux.get, reverse=True):
        if n_added == 2:
            break
        m = (seg == i) & valid
        k = np.unravel_index(np.nanargmax(np.where(m, white, -np.inf)), seg.shape)
        if d_edge[k] <= half + margin:            # 同樣避開視野邊緣
            continue
        add(f"source #{int(i)}", int(k[0]), int(k[1]))
        n_added += 1

    # --- blank:沿離主源的距離均勻取,對照「距離越遠越乾淨」的趨勢 ---
    lo, hi = float(dm.min()), float(dm.max())
    for q in np.linspace(0.25, 1.0, n_blank):
        j = int(cand[np.argmin(np.abs(dm - (lo + q * (hi - lo))))])
        add(f"blank d={d_main.ravel()[j]:.0f}px", *divmod(j, seg.shape[1]))
    return boxes, main, peak


def draw_map(white, seg, s_hat, boxes, out_path, title):
    """方框畫在白光圖與 s 場上 —— 只報座標的話看不出那是天上的哪一塊。"""
    n = 1 if s_hat is None else 2
    fig, axes = plt.subplots(1, n, figsize=(8.5 * n, 8), squeeze=False)
    q0 = max(float(np.nanpercentile(white[white != 0], 20)), 1e-3)
    d  = np.arcsinh(white / q0)
    axes[0][0].imshow(d, origin="lower", cmap="gray_r",
                      vmin=float(np.nanpercentile(d, 30)),
                      vmax=float(np.nanpercentile(d, 99.7)))
    axes[0][0].set_title("whitelight", fontsize=10)
    if s_hat is not None:
        # 色階以 1.0 為中心、半寬 3 x 穩健散布。改用分位數的話上下不對稱、
        # 範圍又更窄,同一張 s 場會看起來條紋強得多 —— 那是色階的差別,
        # 不是資料變了。
        v = 3 * scale(s_hat[np.isfinite(s_hat)])
        im = axes[0][1].imshow(s_hat, origin="lower", cmap="RdBu_r",
                               vmin=1 - v, vmax=1 + v)
        plt.colorbar(im, ax=axes[0][1], fraction=0.045, pad=0.01)
        axes[0][1].set_title("s field (new continuum map)", fontsize=10)
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    for ax in axes[0]:
        ax.contour(seg > 0, levels=[0.5], colors="#2ca02c", linewidths=0.45)
        for i, (nm, (y0, y1, x0, x1)) in enumerate(boxes.items()):
            c = cmap[i % 10]
            if y0 == y1 and x0 == x1:
                # 單一 spaxel 畫成方框的話只有 1 px,在 320 px 的視場上看不見
                ax.plot(x0, y0, "+", color=c, ms=13, mew=2.2)
            else:
                ax.add_patch(mpatches.Rectangle((x0 - .5, y0 - .5), x1 - x0 + 1,
                                                y1 - y0 + 1, fill=False, ec=c,
                                                lw=1.8))
            ax.annotate(nm, (x0 + (x1 - x0) / 2, y1), xytext=(0, 5),
                        textcoords="offset points", color=c, fontsize=8,
                        fontweight="bold", ha="center",
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])
        ax.set_xlabel("x [pix]")
    axes[0][0].set_ylabel("y [pix]")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def draw_pdf(boxes, wl, tri, out_path, title, smooth_w):
    """一頁一個方框,上下兩格 —— 判斷「扣得好不好」不能只看殘差。

    把源扣光也會讓殘差變小,所以殘差小本身不是好消息。要三條一起看:

        上格   raw 與 model 同尺度疊在一起。天空模型有沒有貼上去,看這格。
        下格   resid = raw − model,尺度放大。blank 的框應該只剩雜訊;
               源上的框應該留著星系/恆星的光譜形狀,被壓平就是 over-subtraction。

    上下不共用 y 尺度是刻意的:天空連續譜比殘差大好幾個數量級,同尺度的話殘差
    會是一條貼在 0 上的直線,什麼都看不出來。
    """
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out_path) as pdf:
        for nm, b in boxes.items():
            raw, mod, res = tri[nm]
            fig, ax = plt.subplots(2, 1, figsize=(13, 7.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1.25, 1],
                                                "hspace": 0.08})
            # raw 畫粗、model 畫細疊在上面。同粗細的話兩條在 blank 完全重合,
            # 看圖的人分不出「黑線藏在紅線底下」還是「黑線根本沒畫出來」——
            # 而那正是這一格要回答的問題。粗細差開之後,重合處會露出灰色邊。
            ax[0].plot(wl, smooth(raw, smooth_w), lw=1.4, color="0.45",
                       label="raw (wsky)")
            ax[0].plot(wl, smooth(mod, smooth_w), lw=0.5, color="#d62728",
                       label="sky model")
            ax[0].set_ylabel("flux", fontsize=9)
            ax[0].legend(fontsize=9, loc="upper right", framealpha=0.85)
            ax[0].grid(alpha=0.25)

            ax[1].axhline(0, color="0.5", lw=0.7)
            ax[1].plot(wl, smooth(res, smooth_w), lw=0.5, color="#1f77b4",
                       label="residual = raw − model")
            ax[1].set_ylabel("residual", fontsize=9)
            ax[1].set_xlabel("wavelength [$\\AA$]")
            ax[1].legend(fontsize=9, loc="upper right", framealpha=0.85)
            ax[1].grid(alpha=0.25)
            v = res[np.isfinite(res)]
            lo, hi = (float(x) for x in np.percentile(v, [0.5, 99.5]))
            pad = 0.30 * max(hi - lo, 1e-9)
            ax[1].set_ylim(min(lo - pad, 0.0), max(hi + pad, 0.0))
            ax[1].set_xlim(wl[0], wl[-1])
            for _, lam in LINES:
                for a_ in ax:
                    a_.axvline(lam * (1 + Z_HARO), color="0.75", lw=0.5, ls=":")

            st = spectrum_stats(res)
            fig.suptitle(
                f"{title}\n{nm}   y {b[0]}-{b[1]}  x {b[2]}-{b[3]}   "
                f"{(b[1]-b[0]+1)*(b[3]-b[2]+1)} spaxels   |   residual: "
                f"mean {st['mean']:+.3f}  sigma {st['sigma']:.3f}  "
                f"rms {st['rms_from_zero']:.3f}", fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            pdf.savefig(fig)
            plt.close(fig)
    print(f"saved -> {out_path}")


def smooth(a, w):
    """畫圖用的移動平均。只是為了看得清楚,右邊表格的數字都沒有平滑過。"""
    if w <= 1:
        return a
    k = np.ones(w) / w
    return np.convolve(np.nan_to_num(a), k, mode="same") / np.convolve(
        np.isfinite(a).astype(float), k, mode="same")


def main():
    ap = argparse.ArgumentParser(description="方框裡的平均光譜(新 basis + s 場)")
    ap.add_argument("--work", default="results/skymodel/ne_pointing")
    ap.add_argument("--run", default="step03_half_svdK30_tpl68_sfield",
                    help="要看的 step05 輸出目錄名(新 basis + s 空間場)")
    ap.add_argument("--ref-run", default="blank_svdK30_tpl68_s1",
                    help="對照的舊 run;none = 不畫")
    ap.add_argument("--eso", default="data/Haro11_NEpointing_esonosky.fits",
                    help="外部基準;none = 不畫")
    ap.add_argument("--half", type=int, default=6, help="方框半寬,框寬 = 2*half+1")
    ap.add_argument("--n-blank", type=int, default=4)
    ap.add_argument("--edge", type=float, nargs="+", default=[7, 20],
                    help="源外方框的目標距離(px,離主源)")
    ap.add_argument("--margin", type=float, default=10,
                    help="方框要離視野邊緣這麼遠(px)。邊緣曝光少,step5 還會把覆蓋率 "
                         "< 90%% 的 spaxel 寫成 NaN")
    ap.add_argument("--smooth", type=int, default=1, help="畫圖用的移動平均寬度(通道)")
    ap.add_argument("--pdf", action="store_true",
                    help="改出 PDF:一頁一個方框,上格 raw vs sky model、下格殘差。"
                         "這個模式只看**這一個 run 自己**(raw / model / resid),"
                         "不需要 ESO 或舊 run 當對照,所以每顆 pointing 都畫得出來。"
                         "raw 的路徑從 run 的 meta.json 讀 —— 手打的話遲早會拿錯一顆")
    ap.add_argument("--out", default=None,
                    help="輸出**目錄**;預設 results/skymodel/evaluation/pNN/box/。"
                         "一個方框一張圖,檔名由方框名稱轉成")
    args = ap.parse_args()

    W    = ROOT / args.work
    seg, white, _ = load_field(W)
    run   = W / "step05" / args.run
    wl    = np.load(W / "step03/wavelength.npy")
    s_hat = np.load(run / "s_hat.npy") if (run / "s_hat.npy").exists() else None

    boxes, main, peak = pick_boxes(seg, white, args.half, args.n_blank,
                                   args.edge, args.margin, W / "step04")
    print(f"主源 {int(main.sum()):,} px,最亮像素 (y, x) = {peak}")
    for nm, (y0, y1, x0, x1) in boxes.items():
        print(f"  {nm:<18} y {y0}-{y1}  x {x0}-{x1}")

    if args.pdf:
        cube_path = ROOT / json.loads((run / "meta.json").read_text())["cube"]
        tri = {}
        for tag, p in (("raw", cube_path), ("mod", run / "sky_model.fits"),
                       ("res", run / "sky_subtracted.fits")):
            h, hdu = cube_data(p)
            for nm, b in boxes.items():
                tri.setdefault(nm, {})[tag] = box_mean(hdu, *b)
            h.close()
            print(f"讀完 {tag}  {p.name}")
        tri = {nm: (d["raw"], d["mod"], d["res"]) for nm, d in tri.items()}
        out = (Path(args.out) / "box_raw_model_resid.pdf" if args.out
               else pointing_dir(W.name, "box") / "box_raw_model_resid.pdf")
        out.parent.mkdir(parents=True, exist_ok=True)
        draw_pdf(boxes, wl, tri, out, f"{args.work}  [{args.run}]", args.smooth)
        draw_map(white, seg, s_hat, boxes, out.with_suffix(".map.png"),
                 f"{'box' if args.half else 'point'} locations   {W.name}")
        return

    srcs = {}
    if args.eso.lower() != "none":
        srcs["ESO nosky"] = ROOT / args.eso
    if args.ref_run.lower() != "none":
        srcs["s per-spaxel (old)"] = W / "step05" / args.ref_run / "sky_subtracted.fits"
    srcs["s field (new)"] = run / "sky_subtracted.fits"
    srcs = {k: v for k, v in srcs.items() if v.exists()}

    curves = {nm: {} for nm in boxes}
    for m, p in srcs.items():
        h, hdu = cube_data(p)
        for nm, b in boxes.items():
            curves[nm][m] = box_mean(hdu, *b)
        h.close()
        print(f"讀完 {m}")

    # half=0 的「方框」就是單一 spaxel,那是另一種取樣(誠實但很吵),兩者的圖
    # 不能混在同一個目錄裡 —— 檔名相同,後跑的會蓋掉先跑的。
    kind = "box" if args.half else "point"
    outdir = Path(args.out) if args.out else pointing_dir(W.name, kind)
    outdir.mkdir(parents=True, exist_ok=True)
    keys = ("mean", "sigma", "rms_from_zero")
    # 統計欄的版面用**字元數**算,不用寫死的座標:欄數會隨 --eso / --ref-run
    # 開不開而變,寫死的位置只在某一種組合下不重疊。
    NW, VW = len(max(keys, key=len)) + 2, 10        # 標籤欄寬、每個數值欄寬
    ncol   = len(srcs)
    span   = NW + VW * ncol
    for nm, b in boxes.items():
        fig = plt.figure(figsize=(15, 4.2))
        # 0.04 是「一個字元佔多少寬度比例」,由 fontsize 8.5 的等寬字元量出來的:
        # 欄位靠 span 換算成軸內比例,軸本身太窄會截斷、太寬會把數字推到天邊。
        gs  = fig.add_gridspec(1, 2, width_ratios=[6, 0.04 * span], wspace=0.02,
                               left=0.055, right=0.985, top=0.86, bottom=0.13)
        ax = fig.add_subplot(gs[0, 0])
        ax.axhline(0, color="0.5", lw=0.6)
        for lab, y in curves[nm].items():
            ax.plot(wl, smooth(y, args.smooth), lw=0.5, color=COL[lab],
                    alpha=0.8, label=lab)
        # y 範圍不對稱 —— 源上的格整條都在 0 以上,硬要對稱的話一半的畫布是空的,
        # 曲線被壓成一條。用 0.5/99.5 分位切掉少數尖峰,再確保 0 一定在範圍內
        # (0 是「天空扣乾淨」的基準線,看不到它就沒得判斷)。
        v = np.concatenate([y[np.isfinite(y)] for y in curves[nm].values()])
        lo, hi = (float(x) for x in np.percentile(v, [0.5, 99.5]))
        pad = 0.30 * max(hi - lo, 1e-9)
        ax.set_ylim(min(lo - pad, 0.0), max(hi + pad, 0.0))
        ax.set_xlim(wl[0], wl[-1])
        for lname, lam in LINES:                      # 紅移後的位置,天空模型扣不掉
            ax.axvline(lam * (1 + Z_HARO), color="0.75", lw=0.5, ls=":")
        ax.set_ylabel("flux", fontsize=9)
        ax.set_xlabel("wavelength [$\\AA$]", fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=9, loc="upper left", ncol=3, framealpha=0.85)

        sax = fig.add_subplot(gs[0, 1]); sax.axis("off")
        sax.text(0.0, 0.98, "\n".join([""] + list(keys)),
                 va="top", ha="left", family="monospace", fontsize=8.5,
                 transform=sax.transAxes)
        for j, (lab, y) in enumerate(curves[nm].items()):
            st = spectrum_stats(y)
            sax.text((NW + VW * (j + 1)) / span, 0.98,
                     "\n".join([SHORT[lab]] + [f"{st[k]:.3f}" for k in keys]),
                     va="top", ha="right", family="monospace", fontsize=8.5,
                     color=COL[lab], transform=sax.transAxes)
        npx = (b[1] - b[0] + 1) * (b[3] - b[2] + 1)
        where = (f"y {b[0]}  x {b[2]}" if npx == 1 else
                 f"y {b[0]}-{b[1]}  x {b[2]}-{b[3]}")
        fig.suptitle(f"{W.name}  {nm}   {where}   "
                     f"{npx} spaxel{'' if npx == 1 else 's'}", fontsize=12)
        o = outdir / f"{slug(nm)}.png"
        fig.savefig(o, dpi=135, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {o}")

    draw_map(white, seg, s_hat, boxes, outdir / "map.png",
             f"{kind} locations   {W.name}")


if __name__ == "__main__":
    main()
