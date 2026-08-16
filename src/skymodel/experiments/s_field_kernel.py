"""把天空連續譜係數 s 建成一個空間場 —— 三種模型,以及它們到底能外插多遠。

動機
----
現在 step5 給每個 blank spaxel 一組完全自由的 (s, c1..cK),只用它自己的光譜解。
源旁邊那格看到多出來的源光,唯一能解釋它的辦法就是把自己的 s 抬高 —— **那個逐
spaxel 的自由度就是源光被吃掉的通道**。

路線 2 的想法:讓 s 不再屬於某一格,而是一個空間場在該位置的取值,只用**遠離所有
源**的 spaxel 去定它,再外插到源附近。一格的資料撼動不了那個場,源的光在天空模型
裡就無處可去,只能留在殘差裡 —— 也就是被保留下來。

這支不改 pipeline。它只回答「這條路成不成立、該用哪個模型」。

三個模型
--------
① kernel  歸一化卷積(核回歸)

                    Σ_q  w(q) · K(p − q) · s(q)
        s_hat(p) = ─────────────────────────────
                    Σ_q  w(q) · K(p − q)

   w 是訓練點指示函數(遠場 blank = 1,其餘 0),K 是高斯核。分子分母各自是一個
   卷積,兩次 gaussian_filter 就算完。**分母是關鍵**:它是「這個位置實際收集到
   多少真實權重」,值域 0-1。源區內部 w 全是 0,分子分母都只拿得到外圈的貢獻,
   於是洞裡的值完全由周圍遠場決定 —— **外插不是另外寫的程式,是這條式子的自然
   結果**。省略分母等於把源區當成「s = 0 的真實觀測」,估計值會被往 0 拉。

   den 太小的地方必須宣告無法預測:那是拿幾乎沒有的資料除以幾乎沒有的權重,
   雜訊被放大幾十倍。而「有多少格無法預測」正是外插能力的答案,所以 min_weight
   不是防呆,是實驗的一部分。

② row+col  s ≈ mu + a(y) + b(x),交替以中位數求解

   為什麼需要它:s 的空間結構**不是等向的平滑起伏,而是對齊影像軸的週期條紋**。
   週期性、軸對齊的圖案不可能是天空(天空平滑)也不可能是源(源是團塊),那是
   儀器結構 —— IFU / slice / 曝光拼接的邊界。等向高斯核是**錯的基底**,它會抹過
   條紋的銳利邊界,所以再怎麼調 sigma 都抓不完整。

   而這個模型有一個核回歸永遠做不到的性質:**它伸得進 Haro 11**。橫條紋橫跨
   整個視場,Haro 11 佔住的每一條 row 在遠處仍有大量訓練點,資訊來自**同一條
   row 的遠方**,不是來自附近的格子。

③ hybrid  row+col 之後,再對殘差做一次核回歸

   條紋由 ① 拿走,剩下的局部起伏由 ② 拿走;核伸不到的地方殘差項自然歸零,
   模型平滑退化成 row+col。

三個非做不可的清理
------------------
① **訓練集要剔掉擬合失敗的格。** s 的物理值應該在 1 附近,但有少數 spaxel 會解
   出離群的值。它們不是天空的空間結構,是壞掉的解。
② **指標要穩健。** 用 rms 的話那些少數壞格會單獨主宰整個數字,結論可能整個
   反過來。
③ **各個模型必須用同一組挖洞位置。** 否則比的是「不同的洞」而不是「不同的模型」。

挖洞測試(唯一誠實的驗證)
------------------------
模型一定給得出一個值,問題是那個值準不準。把一塊**本來知道答案**的遠場區域挖掉,
只用剩下的預測它,再和真值比。兩把尺規讓誤差有意義:

    雜訊底線 sigma_n   相鄰格差分估的求解雜訊。誤差不可能低於它。
    常數模型 sigma_tot 在**同一批格**上用訓練集中位數預測的誤差 = 不建模的成績。

**覆蓋率必須和誤差一起看。** 核回歸在大洞裡只伸得到邊緣一圈,它的誤差只算在
那些格上,看起來漂亮但那是「只考容易的題目」。

    conda run -n astro python src/skymodel/experiments/s_field_kernel.py
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 這些函式是 pipeline 也要用的,住在 src/skymodel/utils.py。
# pipeline 不可以 import experiments(方向錯了),所以共用的部分放在 utils。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import (scale, nanmed, noise_floor, kernel_field, rowcol_field,
                   ridge_from_data, TRUNCATE)  # noqa: E402

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP05  = ROOT / "results/skymodel/step05"
FIGURES = ROOT / "results/skymodel/figures/s_field"


def predict(s, w, model, sigma, min_weight):
    """統一介面。回傳 (預測, 有效遮罩)。

    min_weight 現在只用來定義「覆蓋率」這個報表用的統計量(den > min_weight),
    值本身一律由 ridge 形式算出 —— 它處處連續,沒有門檻。
    """
    if model == "kernel":
        ridge = ridge_from_data(s, w, sigma)
        v, den = kernel_field(s, w, sigma, ridge)
        ok = den > min_weight
        return np.where(ok, v, np.nan), ok
    M, _, _ = rowcol_field(s, w)
    if model == "rowcol":
        return M, np.ones(s.shape, bool)
    R = s - M
    r, _ = kernel_field(R, w, sigma, ridge_from_data(R, w, sigma))
    return M + r, np.ones(s.shape, bool)


def main():
    ap = argparse.ArgumentParser(description="把 s 建成空間場,並量它能外插多遠")
    ap.add_argument("--run", default="blank_svdK30_tpl68_s1",
                    help="從哪個 step5 run 讀 s_map.npy")
    ap.add_argument("--r-far", type=float, default=12.0,
                    help="訓練點必須離最近的源這麼遠(px)。代價只是樣本變少,"
                         "而 blank 的樣本本來就多,所以這個值可以取得凶")
    ap.add_argument("--r-far-haro", type=float, default=None,
                    help="Haro 11 (seg == 1) 專用的遠場半徑。它的暈延伸得比小源"
                         "遠得多(見 experiments/s_ring_check.py),用同一個 r_far "
                         "的話訓練點本身就被暈污染,模型會把暈學成天空")
    ap.add_argument("--clip", type=float, default=8.0,
                    help="訓練集剔除門檻:|s − 中位| > clip × 穩健散布 的格不採用")
    ap.add_argument("--sigma", type=float, default=2.0,
                    help="核回歸的核寬(px)")
    ap.add_argument("--sigma-scan", type=float, nargs="+",
                    default=[2, 3, 5, 8, 12, 20, 30], help="核寬掃描範圍")
    ap.add_argument("--radius", type=float, nargs="+",
                    default=[3, 5, 8, 12, 20, 40], help="挖洞測試的洞半徑(px)")
    ap.add_argument("--min-weight", type=float, default=0.2,
                    help="收集到的核權重低於此值就宣告無法預測")
    ap.add_argument("--n-place", type=int, default=25, help="每個洞半徑挖幾次")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figdir", default=str(FIGURES))
    args = ap.parse_args()

    fig_dir = Path(args.figdir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    seg   = fits.getdata(STEP01 / "seg.fits").astype(int)
    white = fits.getdata(STEP01 / "whitelight.fits")
    s     = np.load(STEP05 / args.run / "s_map.npy").astype(np.float64)
    ny, nx = s.shape
    Y, X = np.mgrid[0:ny, 0:nx]

    valid = white != 0
    blank = valid & (seg == 0) & np.isfinite(s)
    # distance_transform_edt 對每個 True 算「離最近的 False 多遠」,餵 seg == 0
    # 得到的就是每個 blank 格離最近源像素的距離。
    dist  = ndimage.distance_transform_edt(seg == 0)

    print(f"run {args.run}   s_map {s.shape}")
    print(f"視場內 {int(valid.sum()):,}   blank(有解) {int(blank.sum()):,}   "
          f"源 {int((valid & (seg > 0)).sum()):,}")
    print(f"\n訓練點數 vs r_far(離源多遠才算遠場):")
    for r in (0, 4, 8, 12, 16, 20, 30):
        n = int((blank & (dist > r)).sum())
        print(f"  r_far = {r:>2} px   {n:>7,} 格 "
              f"({100 * n / max(int(blank.sum()), 1):5.1f}% 的 blank)")

    far = blank & (dist > args.r_far)
    if args.r_far_haro:
        d_har = ndimage.distance_transform_edt(seg != 1)
        n0 = int(far.sum())
        far &= d_har > args.r_far_haro
        print(f"\nHaro 11 專用遠場半徑 {args.r_far_haro:g} px:"
              f"訓練點 {n0:,} -> {int(far.sum()):,}")
    med0, sc0 = float(np.median(s[far])), scale(s[far])
    print(f"\n訓練集清理(中位 {med0:.5f},穩健散布 {sc0:.5f}):")
    for c in (4, 6, 8, 12, 20, 50):
        n = int((np.abs(s - med0) > c * sc0)[far].sum())
        print(f"  clip = {c:>2g} sigma  剔除 {n:>4} 格 "
              f"({100 * n / max(int(far.sum()), 1):.3f}%)"
              + ("   <- 採用" if c == args.clip else ""))
    train = far & (np.abs(s - med0) <= args.clip * sc0)
    print(f"採用 r_far = {args.r_far:g} px, clip = {args.clip:g} sigma "
          f"-> 訓練點 {int(train.sum()):,} 格")

    # ---- 兩把尺規 --------------------------------------------------------
    tot   = scale(s[train])
    n_y, n_x = noise_floor(s, train, 0), noise_floor(s, train, 1)
    sig_n = min(n_y, n_x)                     # 取較小者,對雜訊較保守
    print(f"\n尺規(都只在訓練區內量,穩健散布):")
    print(f"  總散布   sigma_tot = {tot:.5f}")
    print(f"  求解雜訊 sigma_n   = {sig_n:.5f}  (y {n_y:.5f} / x {n_x:.5f})"
          f"  <- 誤差的下限")
    print(f"  雜訊佔變異數 {100 * (sig_n / tot) ** 2:.1f}%  ->  "
          f"真實空間結構佔 {100 * (1 - (sig_n / tot) ** 2):.1f}%")

    def captured(e):
        """誤差 -> 「抓到多少可解釋的結構」。誤差² = 結構誤差² + 雜訊²,
        所以先扣掉雜訊再和常數模型的結構誤差比。0% = 不建模,100% = 打到雜訊底線。"""
        e2 = max(e ** 2 - sig_n ** 2, 0.0)
        t2 = max(tot ** 2 - sig_n ** 2, 1e-12)
        return 100 * (1 - np.sqrt(e2 / t2))

    # ---- 結構長什麼樣:條紋 -----------------------------------------------
    S = np.where(train, s, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prof_y = np.nanmedian(S, axis=1)      # 每一條 row 的偏移(沿 y 變化)
        prof_x = np.nanmedian(S, axis=0)      # 每一條 column 的偏移(沿 x 變化)
    prof_y -= np.nanmedian(prof_y)
    prof_x -= np.nanmedian(prof_x)
    print(f"\n空間結構的形狀(這決定該用哪個模型):")
    for p, nm in ((prof_y, "沿 y (row 剖面)"), (prof_x, "沿 x (col 剖面)")):
        f = np.isfinite(p)
        q = np.where(f, p, 0.0); q = q - q.mean()
        ac = np.correlate(q, q, "full")[len(q) - 1:]
        ac = ac / ac[0]
        pk = [(d, ac[d]) for d in range(5, min(140, len(ac) - 1))
              if ac[d] > ac[d - 1] and ac[d] > ac[d + 1] and ac[d] > 0.15]
        print(f"  {nm}:振幅 {scale(p[f]):.5f}  自相關峰 "
              + ("  ".join(f"(lag {d}, {v:.2f})" for d, v in pk[:5]) or "無明顯週期"))
    print("  -> 週期性、軸對齊的條紋不是天空也不是源,是儀器結構。"
          "等向高斯核抹得過它,所以需要 row+col 模型。")

    # ---- 核寬掃描(只針對 kernel) ----------------------------------------
    dt_train = ndimage.distance_transform_edt(train)
    places = {}
    for R in args.radius:
        cand = np.argwhere(dt_train > R + 2)
        if len(cand):
            places[R] = cand[rng.integers(len(cand), size=args.n_place)]
    radii = sorted(places)

    def run_holes(R, model, sigma):
        """挖洞、預測、回傳 (深度, 誤差, 覆蓋率)。所有模型共用同一組洞。"""
        dep, err = [], []
        n_h = n_p = 0
        for cy, cx in places[R]:
            hole = (Y - cy) ** 2 + (X - cx) ** 2 <= R * R
            w2   = train & ~hole
            p, _ = predict(s, w2, model, sigma, args.min_weight)
            held = hole & train
            ok   = held & np.isfinite(p)
            n_h += int(held.sum()); n_p += int(ok.sum())
            if ok.any():
                # 深度 = 離最近的**剩餘訓練點**多遠。用這個而不是「離圓心多遠」,
                # 因為它對任何形狀都成立,真實源的不規則足跡也適用。
                dep.append(ndimage.distance_transform_edt(~w2)[ok])
                err.append((p - s)[ok])
        if not dep:
            return np.array([]), np.array([]), 0.0
        return np.concatenate(dep), np.concatenate(err), n_p / max(n_h, 1)

    print(f"\n核寬掃描(模型 = kernel,每個洞半徑挖 {args.n_place} 次,"
          f"所有 sigma 共用同一組洞)")
    hdr = f"{'sigma\\R':>9}" + "".join(f"{f'R={r:g}':>13}" for r in radii)
    print(hdr); print("-" * len(hdr))
    scan = {}
    for sg in args.sigma_scan:
        row = []
        for R in radii:
            _, e, c = run_holes(R, "kernel", sg)
            scan[(sg, R)] = (scale(e), c)
            row.append(f"{scale(e):>7.5f}({100*c:>3.0f}%)")
        print(f"{sg:>9g}" + "".join(f"{v:>13}" for v in row))
    print("  括號是覆蓋率:被挖掉的格裡有多少真的算得出值。核寬小 -> 準但伸不遠。")

    # ---- 三個模型正面比較 -------------------------------------------------
    print(f"\n三個模型正面比較(kernel 與 hybrid 都用 sigma = {args.sigma:g})")
    print(f"{'洞半徑':>7}{'kernel':>16}{'row+col':>16}{'hybrid':>16}"
          f"{'常數模型':>12}{'kernel覆蓋':>11}")
    print("-" * 78)
    comp, curves = {}, {}
    for R in radii:
        cells, kcov = [], 0.0
        for model in ("kernel", "rowcol", "hybrid"):
            dep, e, c = run_holes(R, model, args.sigma)
            v = scale(e)
            comp[(model, R)] = (v, c)
            curves[(model, R)] = (dep, e)
            if model == "kernel":
                kcov = c
            cells.append(f"{v:>7.5f}({captured(v):>3.0f}%)")
        # 常數模型:同一批洞,用剩下訓練點的中位數預測
        ec = []
        for cy, cx in places[R]:
            hole = (Y - cy) ** 2 + (X - cx) ** 2 <= R * R
            w2 = train & ~hole
            held = hole & train
            ec.append(np.median(s[w2]) - s[held])
        vc = scale(np.concatenate(ec))
        comp[("const", R)] = (vc, 1.0)
        print(f"{R:>7g}" + "".join(f"{c:>16}" for c in cells)
              + f"{vc:>12.5f}{100 * kcov:>10.1f}%")
    print("  括號 = 抓到多少可解釋的結構(0% 不建模,100% 打到雜訊底線)。")
    print("  **kernel 的誤差只算在它伸得到的格上** —— 大洞時那只是洞的邊緣一圈,")
    print("  row+col 與 hybrid 一律是 100% 的洞,所以大洞時只有後兩者的數字可比。")

    # ---- 誤差 vs 洞內深度 -------------------------------------------------
    print(f"\n誤差 vs 洞內深度(model = hybrid,sigma = {args.sigma:g}):")
    edges = np.array([0, 1, 2, 3, 5, 8, 12, 20, 30, 50, 999])
    print(f"{'深度 [px]':>12}" + "".join(f"{f'R={r:g}':>11}" for r in radii))
    print("-" * (12 + 11 * len(radii)))
    depth_tab = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        row = []
        for R in radii:
            dep, e = curves[("hybrid", R)]
            m = (dep > lo) & (dep <= hi) if dep.size else np.zeros(0, bool)
            row.append(scale(e[m]) if m.sum() >= 50 else np.nan)
        depth_tab[f"{lo}-{hi}"] = row
        if not all(v != v for v in row):
            lbl = f"{lo}-{hi}" if hi < 999 else f">{lo}"
            print(f"{lbl:>12}" + "".join(f"{v:>11.5f}" if v == v else f"{'-':>11}"
                                         for v in row))

    # ---- Haro 11 -----------------------------------------------------------
    haro   = seg == 1
    allsrc = valid & (seg > 0)
    d2t    = ndimage.distance_transform_edt(~train)
    print(f"\nHaro 11 (seg == 1) {int(haro.sum()):,} 格:")
    print(f"  離最近訓練點 中位 {np.median(d2t[haro]):.1f} px,"
          f"最遠 {d2t[haro].max():.1f} px")
    rows_n = train.sum(axis=1)
    hy = np.unique(np.nonzero(haro)[0])
    print(f"  它佔住 {len(hy)} 條 row,每條 row 在遠處仍有訓練格 "
          f"中位 {np.median(rows_n[hy]):.0f},最少 {rows_n[hy].min()}"
          f"   <- row+col 的資訊來源")
    for model in ("kernel", "rowcol", "hybrid"):
        p, _ = predict(s, train, model, args.sigma, args.min_weight)
        n_ok = int(np.isfinite(p[haro]).sum())
        print(f"  {model:>7}: 給得出值 {n_ok:>6,}/{int(haro.sum()):,} "
              f"({100 * n_ok / int(haro.sum()):5.1f}%)"
              + (f"   值域 {np.nanmin(p[haro]):.4f} - {np.nanmax(p[haro]):.4f}"
                 if n_ok else ""))

    s_hat, _ = predict(s, train, "hybrid", args.sigma, args.min_weight)
    dlt = s_hat[allsrc] - 1.0
    print(f"\n源區 {int(allsrc.sum()):,} 格:hybrid 的 s_hat 與現行常數 1.0 的差")
    print(f"  中位 {np.median(dlt):+.4f}   16-84% [{np.percentile(dlt,16):+.4f}, "
          f"{np.percentile(dlt,84):+.4f}]   最大 |diff| {np.max(np.abs(dlt)):.4f}")

    np.save(fig_dir / "s_hat_hybrid.npy", s_hat.astype(np.float32))

    # ---- 圖 ---------------------------------------------------------------
    M, a, b = rowcol_field(s, train)
    v = 3 * tot
    fig, ax = plt.subplots(2, 2, figsize=(13, 11.5))
    panels = [
        ("$s$ from the per-spaxel fit  (blank only)", np.where(blank, s, np.nan),
         dict(cmap="RdBu_r", vmin=1 - v, vmax=1 + v)),
        ("row+col model  $\\mu + a(y) + b(x)$   (the instrument stripes)", M,
         dict(cmap="RdBu_r", vmin=1 - v, vmax=1 + v)),
        (f"hybrid  =  row+col  +  kernel on the residual "
         f"($\\sigma$ = {args.sigma:g})", s_hat,
         dict(cmap="RdBu_r", vmin=1 - v, vmax=1 + v)),
        ("residual  $s - \\hat{s}_{hybrid}$  (blank only)",
         np.where(blank, s - s_hat, np.nan), dict(cmap="RdBu_r", vmin=-v, vmax=v)),
    ]
    for a_, (t, img, kw) in zip(ax.ravel(), panels):
        im = a_.imshow(img, origin="lower", **kw)
        a_.contour(seg > 0, levels=[0.5], colors="k", linewidths=0.4)
        a_.set_title(t, fontsize=10)
        a_.set_xticks([]); a_.set_yticks([])
        plt.colorbar(im, ax=a_, fraction=0.046, pad=0.02)
    fig.suptitle(f"the sky-continuum coefficient as a spatial field   "
                 f"(run {args.run},  training = blank with dist > {args.r_far:g} px "
                 f"from any source)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p1 = fig_dir / "01_s_field.png"
    fig.savefig(p1, dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.0))
    ax[0].plot(np.arange(ny), prof_y, lw=1.0, label="along $y$ (row profile)")
    ax[0].plot(np.arange(nx), prof_x, lw=1.0, alpha=0.8,
               label="along $x$ (column profile)")
    ax[0].axhline(0, color="0.6", lw=0.7)
    ax[0].set_xlabel("pixel"); ax[0].set_ylabel("median $s$ − global median")
    ax[0].set_title("the structure is axis-aligned striping, not smooth undulation",
                    fontsize=10)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    for sg in args.sigma_scan:
        ax[1].plot(radii, [scan[(sg, R)][0] for R in radii], "o-", ms=4, lw=1.3,
                   label=f"kernel $\\sigma$ = {sg:g}")
    ax[1].axhline(sig_n, color="k", ls="--", lw=1.2, label=f"noise floor {sig_n:.4f}")
    ax[1].set_xlabel("hole radius [px]"); ax[1].set_ylabel("hold-out error in $s$")
    ax[1].set_title("kernel width scan  (error only over the pixels it reaches)",
                    fontsize=10)
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)

    style = {"kernel": ("#1f77b4", "o-"), "rowcol": ("#2ca02c", "s-"),
             "hybrid": ("#d62728", "^-"), "const": ("0.5", ":")}
    for model in ("kernel", "rowcol", "hybrid", "const"):
        c, ls = style[model]
        ax[2].plot(radii, [comp[(model, R)][0] for R in radii], ls, ms=5, lw=1.6,
                   color=c, label=model + (" (partial coverage)"
                                           if model == "kernel" else ""))
    ax[2].axhline(sig_n, color="k", ls="--", lw=1.2, label="noise floor")
    ax[2].set_xlabel("hole radius [px]"); ax[2].set_ylabel("hold-out error in $s$")
    ax[2].set_title(f"three models head to head  ($\\sigma$ = {args.sigma:g})",
                    fontsize=10)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)
    fig.suptitle("can the far field predict what we masked out?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p2 = fig_dir / "02_holdout.png"
    fig.savefig(p2, dpi=140, bbox_inches="tight")
    plt.close(fig)

    (fig_dir / "s_field_kernel.json").write_text(json.dumps(dict(
        run=args.run, r_far=args.r_far, clip=args.clip, sigma=args.sigma,
        min_weight=args.min_weight, n_train=int(train.sum()),
        sigma_tot=tot, sigma_noise=sig_n,
        prof_y=[None if x != x else float(x) for x in prof_y],
        prof_x=[None if x != x else float(x) for x in prof_x],
        kernel_scan={f"{sg:g}|{R:g}": dict(err=scan[(sg, R)][0],
                                           coverage=scan[(sg, R)][1])
                     for (sg, R) in scan},
        compare={f"{m}|{R:g}": dict(err=comp[(m, R)][0], coverage=comp[(m, R)][1])
                 for (m, R) in comp},
        depth_bins={k: [None if x != x else x for x in v]
                    for k, v in depth_tab.items()},
        radius=radii, sigma_scan=list(args.sigma_scan)),
        ensure_ascii=False, indent=2))
    print(f"\nsaved -> {p1}\n         {p2}\n         "
          f"{fig_dir / 's_hat_hybrid.npy'}")


if __name__ == "__main__":
    main()
