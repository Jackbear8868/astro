"""源周圍那一圈是真的嗎 —— s 的徑向剖面 + 平移虛無對照。

問題
----
把 s 畫成影像之後,每個源的外圍看起來都有一圈。有兩種完全不同的成因,而它們的
處置方式相反:

    真的   源的光漏進了訓練樣本 -> 那些格的 s 被抬高 -> 環是**污染的證據**,
           代表 r_far 不夠遠,必須加大。
    假的   洞被模型填平之後紋理和周圍不同(裡面平滑、外面有逐格雜訊),
           眼睛把紋理的交界讀成一圈。這種環只存在於「看」,不存在於數字。

怎麼分辨
--------
量 s 對「離最近源多遠」的徑向剖面(只用 blank 格,逐環取中位數),然後做
**平移虛無對照**:把源的足跡整塊搬到隨機的空白位置,用完全相同的程式再算一次。

    虛無也有環  ->  幾何或估計方法造成的假象
    只有真源有環 ->  真的污染

這個對照是必要的,不是保險。環的振幅只有千分之幾,而逐環中位數本身就有估計
雜訊;沒有虛無分布就無法判斷「看起來有」是不是統計上的必然。

三條剖面一起看,因為它們回答不同的問題:

    s          資料本身。有環 = 真的有污染。
    s_hat      模型。它把環傳播到哪裡去了。
    s − s_hat  殘差。模型有沒有把環吸收掉(吸收掉 = 污染被寫進了天空模型)。

Haro 11 單獨處理:它太大,搬不進任何空白區,所以**沒有虛無對照**,只報剖面本身
與它衰減到基線的半徑。

    conda run -n astro python src/skymodel/experiments/s_ring_check.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP05  = ROOT / "results/skymodel/step05"
FIGURES = ROOT / "results/skymodel/figures/s_field"


def scale(a):
    a = a[np.isfinite(a)]
    return float((np.percentile(a, 84) - np.percentile(a, 16)) / 2) if a.size else np.nan


def profile(vals, dist, usable, rmax):
    """逐環中位數。回傳 (每環中位數, 每環格數)。

    中位數而非平均:少數壞格會把平均拉走,而我們要量的振幅只有千分之幾。
    """
    med = np.full(rmax, np.nan)
    cnt = np.zeros(rmax, int)
    for r in range(rmax):
        m = usable & (dist > r) & (dist <= r + 1)
        cnt[r] = int(m.sum())
        if cnt[r] >= 20:
            med[r] = float(np.median(vals[m]))
    return med, cnt


def main():
    ap = argparse.ArgumentParser(description="源周圍那一圈是真的還是假的")
    ap.add_argument("--run", default="blank_svdK30_tpl68_s1")
    ap.add_argument("--s-hat", default=str(FIGURES / "s_hat_hybrid.npy"),
                    help="s_field_kernel.py 產生的模型場")
    ap.add_argument("--rmax", type=int, default=30, help="剖面算到幾 px")
    ap.add_argument("--min-area", type=int, default=25, help="只看夠大的源")
    ap.add_argument("--edge-margin", type=float, default=15.0,
                    help="離視場邊界這麼近的源不看。視場邊緣因 DAR 有低覆蓋過渡帶,"
                         "那裡的 s 本來就偏高,混進來會假裝成環")
    ap.add_argument("--far", type=float, default=30.0, help="基線取離源多遠之外")
    ap.add_argument("--n-null", type=int, default=40, help="虛無:整組印章平移幾次")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figdir", default=str(FIGURES))
    args = ap.parse_args()

    fig_dir = Path(args.figdir); fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    seg   = fits.getdata(STEP01 / "seg.fits").astype(int)
    white = fits.getdata(STEP01 / "whitelight.fits")
    s     = np.load(STEP05 / args.run / "s_map.npy").astype(np.float64)
    s_hat = np.load(args.s_hat).astype(np.float64)
    ny, nx = s.shape

    valid = white != 0
    blank = valid & (seg == 0) & np.isfinite(s)
    edge  = ndimage.distance_transform_edt(valid)      # 離視場邊界多遠

    # 只用「離視場邊界夠遠」的 blank 格量剖面。這一條不是可有可無:視場邊緣的
    # s 本來就偏高,不擋掉的話任何靠近邊緣的源都會憑空長出一圈。
    usable = blank & (edge > args.edge_margin)
    print(f"run {args.run}")
    print(f"blank {int(blank.sum()):,}  ->  離視場邊界 > {args.edge_margin:g} px 的 "
          f"{int(usable.sum()):,} 格可用於剖面")

    # 分兩組:Haro 11(seg == 1)與其餘的源。兩者的物理完全不同 ——
    # Haro 11 有延展到數十 px 的暈,小源沒有。
    ids = [int(i) for i in np.unique(seg[seg > 0])
           if i != 1 and (seg == i).sum() >= args.min_area]
    ys, xs = np.nonzero(np.isin(seg, ids))
    keep = [i for i in ids
            if edge[seg == i].min() > args.edge_margin]
    print(f"源(面積 >= {args.min_area},排除 Haro 11):{len(ids)} 個,"
          f"其中離視場邊界夠遠的 {len(keep)} 個 -> 採用")

    other = np.isin(seg, keep)
    d_oth = ndimage.distance_transform_edt(~other)
    d_har = ndimage.distance_transform_edt(seg != 1)
    d_all = ndimage.distance_transform_edt(seg == 0)

    base = float(np.median(s[usable & (d_all > args.far)]))
    sg_px = scale(s[usable & (d_all > args.far)])
    print(f"遠場基線(離任何源 > {args.far:g} px):s 中位 {base:.5f},"
          f"逐格散布 {sg_px:.5f}")

    # ---- 真實剖面 --------------------------------------------------------
    prof = {}
    for nm, d, extra in (("other", d_oth, ~(seg == 1) & (d_har > args.far)),
                         ("haro",  d_har, ~other & (d_oth > args.far))):
        # extra:量某一組時,把另一組及其鄰近區域排除,免得兩者的環互相污染
        u = usable & extra
        for key, v in (("s", s), ("s_hat", s_hat), ("resid", s - s_hat)):
            m, c = profile(v, d, u, args.rmax)
            prof[f"{nm}|{key}"] = m
            prof[f"{nm}|{key}|n"] = c

    # ---- 虛無:每個源各自獨立平移 ----------------------------------------
    # 每一次realization 都把每個足跡**各自**搬到隨機的空白位置,再合成一張印章,
    # 用完全相同的程式算剖面。這樣每一條虛無曲線的統計量(源數、面積分布、
    # 每環的格數)都和真實測量一致,只有「那裡沒有源」不同。
    #
    # 不能整組剛體平移 —— 源散布在整個視場,包絡框比視場還大,永遠塞不進空白區。
    stamps = []
    for i in keep:
        yy, xx = np.nonzero(seg == i)
        stamps.append((yy - yy.min(), xx - xx.min()))
    free = blank & (edge > args.edge_margin) & (d_all > args.far)
    print(f"\n虛無對照:{len(keep)} 個足跡**各自**獨立平移,合成一張印章,"
          f"重複 {args.n_null} 次")

    nulls = {k: [] for k in ("s", "s_hat", "resid")}
    u_null = usable & ~other & (d_all > args.far)
    placed_n = []
    for _ in range(args.n_null):
        stamp = np.zeros_like(other)
        occupied = ~free
        n_ok = 0
        for fy, fx in stamps:
            for _try in range(200):
                dy = int(rng.integers(0, ny - fy.max()))
                dx = int(rng.integers(0, nx - fx.max()))
                yy, xx = fy + dy, fx + dx
                if np.any(occupied[yy, xx]):
                    continue
                stamp[yy, xx] = True
                # 佔位時連同 rmax 的緩衝一起標掉,避免兩個假源的環互相重疊
                occupied[max(yy.min() - 3, 0):yy.max() + 4,
                         max(xx.min() - 3, 0):xx.max() + 4] = True
                n_ok += 1
                break
        placed_n.append(n_ok)
        if n_ok < len(stamps) // 2:
            continue
        dn = ndimage.distance_transform_edt(~stamp)
        for key, v in (("s", s), ("s_hat", s_hat), ("resid", s - s_hat)):
            nulls[key].append(profile(v, dn, u_null, args.rmax)[0])
    got = len(nulls["s"])
    print(f"  成功 {got} / {args.n_null} 次,每次平均擺進 "
          f"{np.mean(placed_n):.1f} / {len(stamps)} 個足跡")
    nul = {k: (np.array(v) if v else np.zeros((0, args.rmax)))
           for k, v in nulls.items()}

    # ---- 報表 -------------------------------------------------------------
    print(f"\n【其他源】逐環 s 相對遠場基線的偏移(×1000)。"
          f"虛無給的是同樣計算在空白區的分布\n")
    print(f"{'r [px]':>7}{'n':>7}{'s':>9}{'虛無中位':>10}{'虛無16-84%':>16}"
          f"{'百分位':>8}   {'s_hat':>8}{'殘差':>8}")
    print("-" * 76)
    verdict = []
    for r in range(min(args.rmax, 16)):
        n = prof["other|s|n"][r]
        v = prof["other|s"][r]
        if not np.isfinite(v):
            continue
        col = (nul["s"][:, r] if nul["s"].shape[0] else np.zeros(0))
        col = col[np.isfinite(col)]
        # 虛無不足時仍然印真實剖面 —— 沒有對照只是不能下判定,不是沒有資料
        if col.size >= 10:
            pct = 100.0 * np.mean(col <= v)
            verdict.append((r, (v - base) * 1e3, pct))
            nul_s = (f"{(np.median(col)-base)*1e3:>10.2f}"
                     f"{f'[{(np.percentile(col,16)-base)*1e3:+.2f}, {(np.percentile(col,84)-base)*1e3:+.2f}]':>16}"
                     f"{pct:>7.1f}%")
        else:
            nul_s = f"{'-':>10}{'-':>16}{'-':>8}"
        print(f"{r:>7}{n:>7}{(v-base)*1e3:>9.2f}{nul_s}"
              f"   {(prof['other|s_hat'][r]-base)*1e3:>8.2f}"
              f"{prof['other|resid'][r]*1e3:>8.2f}")
    print("\n百分位 = 真實值落在虛無分布的第幾百分位。")
    print("  > 95% 代表「空白區幾乎不會出現這麼高的值」-> 環是真的。")
    print("  50% 附近代表和空白區沒有差別 -> 那一圈是假象。")

    hits = [r for r, _, p in verdict if p > 95]
    print(f"\n判定(其他源):顯著偏高的環 r = {hits if hits else '無'}")
    if hits:
        print(f"  -> 源的光確實漏到 {max(hits) + 1} px 之外;"
              f"訓練用的 r_far 必須大於這個值")
    else:
        print("  -> 資料裡沒有可偵測的環,看到的圈是模型填洞造成的視覺假象")

    print(f"\n【Haro 11】無法平移(太大),只報剖面。"
          f"逐格散布 {sg_px:.5f},所以單環的估計雜訊約 sigma/sqrt(n)\n")
    print(f"{'r [px]':>7}{'n':>7}{'s (×1000)':>12}{'/(sigma/sqrt n)':>17}"
          f"   {'s_hat':>8}{'殘差':>8}")
    print("-" * 62)
    for r in range(min(args.rmax, 30)):
        n = prof["haro|s|n"][r]
        v = prof["haro|s"][r]
        if not np.isfinite(v):
            continue
        z = (v - base) / (sg_px / np.sqrt(max(n, 1)))
        print(f"{r:>7}{n:>7}{(v-base)*1e3:>12.2f}{z:>17.1f}"
              f"   {(prof['haro|s_hat'][r]-base)*1e3:>8.2f}"
              f"{prof['haro|resid'][r]*1e3:>8.2f}")

    # ---- 圖 ---------------------------------------------------------------
    rr = np.arange(args.rmax) + 0.5
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    for a, nm, ttl in ((ax[0], "other", f"other sources ({len(keep)}), "
                                        f"with translated-stamp null"),
                       (ax[1], "haro", "Haro 11 (no null — too large to translate)")):
        a.axhline(0, color="0.6", lw=0.8)
        if nm == "other" and nul["s"].size:
            lo = np.nanpercentile(nul["s"], 2.5, axis=0) - base
            hi = np.nanpercentile(nul["s"], 97.5, axis=0) - base
            a.fill_between(rr, lo * 1e3, hi * 1e3, color="0.8", zorder=0,
                           label="null 2.5–97.5%")
        for key, c, lab in (("s", "#1f77b4", "$s$ (data)"),
                            ("s_hat", "#d62728", "$\\hat{s}$ (model)")):
            a.plot(rr, (prof[f"{nm}|{key}"] - base) * 1e3, "o-", ms=3.5, lw=1.5,
                   color=c, label=lab)
        a.plot(rr, prof[f"{nm}|resid"] * 1e3, "^:", ms=3, lw=1.2, color="#2ca02c",
               label="$s-\\hat{s}$ (residual)")
        a.set_xlabel("distance from the source footprint [px]")
        a.set_ylabel("offset from the far-field baseline  [$\\times 10^{3}$]")
        a.set_title(ttl, fontsize=10)
        a.legend(fontsize=8); a.grid(alpha=0.3)

    # 第三張:直接把源附近的 s 疊起來看(不分環),確認環不是分箱造成的
    ax[2].axhline(0, color="0.6", lw=0.8)
    for nm, d, c, lab in (("other", d_oth, "#1f77b4", "other sources"),
                          ("haro", d_har, "#ff7f0e", "Haro 11")):
        m = usable & (d <= args.rmax)
        dd = d[m]
        vv = (s[m] - base) * 1e3
        o = np.argsort(dd)
        k = max(len(o) // 60, 1)
        xs = [np.median(dd[o][i:i + k]) for i in range(0, len(o) - k, k)]
        ys = [np.median(vv[o][i:i + k]) for i in range(0, len(o) - k, k)]
        ax[2].plot(xs, ys, "-", lw=1.4, color=c, label=lab)
    ax[2].set_xlabel("distance from the nearest source [px]")
    ax[2].set_ylabel("median $s$ − baseline  [$\\times 10^{3}$]")
    ax[2].set_title("same thing without annulus binning (sanity check)", fontsize=10)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    fig.suptitle("is the ring around each source real?   "
                 "grey band = what the same measurement gives on empty sky",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = fig_dir / "03_ring_check.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)

    (fig_dir / "s_ring_check.json").write_text(json.dumps(dict(
        run=args.run, rmax=args.rmax, min_area=args.min_area,
        edge_margin=args.edge_margin, far=args.far, n_null=got,
        baseline=base, sigma_pixel=sg_px, n_sources=len(keep),
        profiles={k: [None if (isinstance(x, float) and x != x) else float(x)
                      for x in v] for k, v in prof.items()},
        null_percentile={str(r): p for r, _, p in verdict},
        significant_radii=hits), ensure_ascii=False, indent=2))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
