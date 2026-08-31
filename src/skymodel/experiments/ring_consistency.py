"""膨脹到哪裡才停 —— 問「這一圈還是不是同一個天體」,而不是「這一圈夠不夠亮」。

不能比係數的大小:表面亮度本來就往外遞減,振幅變小是物理,照「差太多就否決」的
規則任何真實的源都會在第一圈被判死。要比的是形狀 —— 這一圈的光譜是不是足跡光譜
的一個縮放版。

① 參考光譜 P(λ) = <足跡> − <遠場>,兩個都是原始資料的平均,沒有經過任何擬合:
   遮罩決定 blank 樣本、blank 樣本決定 basis,拿擬合結果回頭定遮罩會形成迴圈。

② 對每一圈,用同一組固定的 step03 basis(不重學)做兩個巢狀擬合:

       模型 A   <環> − <遠場> = alpha · P(λ) + Σ cₖ · Lₖ(λ)
       模型 B   <環> − <遠場> =                Σ cₖ · Lₖ(λ)

   alpha 是這一圈裡有多少同一個天體,Δchi2 = chi2(B) − chi2(A) 是加進 P 有沒有
   幫助;多一個參數,虛無假設下 Δchi2 ~ chi2_1。天光線 basis 當干擾項是保守的,
   它有機會吸走一部分 P,判到的「有源」只會少報不會多報。

③ 門檻不指定:整塊印章平移到隨機的空白位置做完全相同的計算,空白區沒有這個天體,
   那裡的 Δchi2 就是虛無分布,門檻直接取它的百分位。

    conda run -n astro python src/skymodel/experiments/ring_consistency.py \\
        --work results/skymodel/p01
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

ROOT    = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "results/skymodel/evaluation/masking/edge_oversub"

WILD  = 1e3     # 壞 voxel 的絕對值遠超物理範圍,見 edge_oversubtraction.py
CHI2_1_MEDIAN = 0.4549364    # chi2 分布(自由度 1)的中位數
CHI2_1_95     = 3.8414588    # 同上,95 百分位


def label_aware(seg, rmax):
    dist, (iy, ix) = ndimage.distance_transform_edt(seg == 0, return_indices=True)
    owner = seg[iy, ix]
    owner[dist > rmax] = 0
    owner[seg > 0] = seg[seg > 0]
    return dist, owner


def region_mean(D, V, idx):
    """一塊區域的平均光譜,以及那個平均的變異數。

    變異數 Σvar / n^2 假設 spaxel 之間互相獨立,而重採樣讓相鄰 voxel 相關,所以它
    是低估的。不在這裡修:虛無分布會把低估的倍率直接量出來。
    """
    d = D[:, idx]
    v = V[:, idx]
    bad = ~np.isfinite(d) | ~np.isfinite(v) | (v <= 0) | (np.abs(d) > WILD)
    d = np.where(bad, np.nan, d)
    v = np.where(bad, np.nan, v)
    n = np.sum(~bad, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m  = np.nanmean(d, axis=1)
        s2 = np.nansum(v, axis=1) / np.maximum(n, 1) ** 2
    return m, s2, n


def nested_fit(y, s2, P, L, ok):
    """模型 A(含 P)對模型 B(不含 P)。回傳 (alpha, Δchi2)。

    不需要解兩次。把 y 和 P 都先投影掉 basis 之後,問題退化成一維:

        alpha = (r_P · r_y) / (r_P · r_P)        Δchi2 = (r_P · r_y)^2 / (r_P · r_P)

    r_y、r_P 是各自扣掉 basis 最佳擬合後的殘差;同一個設計矩陣分解一次,兩個右手邊
    一起解。
    """
    g = ok & np.isfinite(y) & np.isfinite(s2) & (s2 > 0) & np.isfinite(P)
    if g.sum() < L.shape[0] + 10:
        return np.nan, np.nan
    w   = 1.0 / np.sqrt(s2[g])
    Bw  = L[:, g].T * w[:, None]
    rhs = np.column_stack([y[g] * w, P[g] * w])
    c, *_ = np.linalg.lstsq(Bw, rhs, rcond=None)
    r   = rhs - Bw @ c
    ry, rp = r[:, 0], r[:, 1]
    den = float(rp @ rp)
    if den <= 0:
        return np.nan, np.nan
    num = float(rp @ ry)
    return num / den, num * num / den


def stamp_of(seg, dist, owner, i, rmax):
    """源 i 的足跡與各圈,存成相對於足跡左下角的位移,好整塊平移去量虛無分布。"""
    ys, xs = np.nonzero(seg == i)
    y0, x0 = ys.min(), xs.min()
    rings = [np.nonzero((owner == i) & (dist > r - 1) & (dist <= r))
             for r in range(1, rmax + 1)]
    return dict(fy=ys - y0, fx=xs - x0,
                rings=[(ry - y0, rx - x0) for ry, rx in rings])


def main():
    ap = argparse.ArgumentParser(description="逐圈問「還是不是同一個天體」")
    ap.add_argument("--work", required=True,
                    help="pointing 的工作區,例如 results/skymodel/p01")
    ap.add_argument("--cube", default=None,
                    help="含天空的 cube;預設由 pNN 的編號推出 "
                         "data/wsky/DATACUBE_FINAL_N.fits")
    ap.add_argument("--seg", default=None,
                    help="segmentation;預設為工作區的 step01/segmentation_input.fits")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--rmax", type=int, default=12)
    ap.add_argument("--min-area", type=int, default=30, help="只看夠大的源")
    ap.add_argument("--n-null", type=int, default=30, help="每個源平移幾次當對照")
    ap.add_argument("--far", type=float, default=40.0)
    ap.add_argument("--null-margin", type=float, default=6.0)
    ap.add_argument("--threshold", choices=["pooled", "per-source"],
                    default="per-source",
                    help="門檻用合池的還是逐源自己的。逐源原理上正確,但要夠多"
                         "平移樣本才估得準")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figdir", default=str(FIGURES))
    args = ap.parse_args()

    W = ROOT / args.work
    STEP01, STEP03 = W / "step01", W / "step03"
    CUBE = ROOT / (args.cube or f"data/wsky/DATACUBE_FINAL_{int(W.name[1:])}.fits")
    seg_path = Path(args.seg) if args.seg else STEP01 / "segmentation_input.fits"

    fig_dir = Path(args.figdir); fig_dir.mkdir(parents=True, exist_ok=True)
    seg   = fits.getdata(seg_path).astype(int)
    white = fits.getdata(STEP01 / "whitelight_nosky.fits")
    wl    = np.load(STEP03 / "wavelength.npy")
    # 固定用現有的 step03 basis,不重學 —— 重學會讓遮罩透過 basis 影響自己的判定。
    L = np.load(STEP03 / f"sky_line_basis_{args.basis}_K{args.K}.npy")
    print(f"basis: sky_line_basis_{args.basis}_K{args.K}.npy  {L.shape}(固定,不重學)")

    with fits.open(CUBE, memmap=True) as h:
        D = np.asarray(h["DATA"].data, np.float32)
        V = np.asarray(h["STAT"].data, np.float32)
    nz, ny, nx = D.shape
    D = D.reshape(nz, -1)
    V = V.reshape(nz, -1)

    dist, owner = label_aware(seg, args.rmax)
    valid = (white != 0)
    far   = ((dist > args.far) & (seg == 0) & valid).ravel()
    far_m, _, _ = region_mean(D, V, np.flatnonzero(far))
    ok = np.isfinite(far_m) & np.all(np.isfinite(L), axis=0)
    print(f"遠場 {int(far.sum()):,} spaxel   可用通道 {int(ok.sum()):,}/{nz}")

    free = ((seg == 0) & valid & (dist > args.null_margin))
    rng  = np.random.default_rng(args.seed)
    ids  = [int(i) for i in np.unique(seg[seg > 0])
            if (seg == i).sum() >= args.min_area]
    print(f"源 {len(ids)} 個(面積 >= {args.min_area})，每個平移 {args.n_null} 次\n")

    res, null_all = {}, []
    for i in ids:
        st = stamp_of(seg, dist, owner, i, args.rmax)
        fp_m, fp_s2, _ = region_mean(D, V, np.flatnonzero((seg == i).ravel()))
        P = fp_m - far_m                       # 參考光譜:沒有經過任何擬合
        a_r, d_r, n_r = [], [], []
        for r in range(1, args.rmax + 1):
            ry, rx = np.nonzero((owner == i) & (dist > r - 1) & (dist <= r))
            n_r.append(ry.size)
            if ry.size < 5:
                a_r.append(np.nan); d_r.append(np.nan); continue
            m, s2, _ = region_mean(D, V, ry * nx + rx)
            a, dc = nested_fit(m - far_m, s2, P, L, ok)
            a_r.append(a); d_r.append(dc)

        # 虛無分布:整塊印章(足跡 + 各圈)平移到隨機的空白位置,就地重算 P。P 若
        # 沿用原位置的光譜,就切斷了「足跡和貼著它的環共享同一塊局部天空結構」,
        # 而那正是最可能製造偽陽性的機制,門檻會被低估。
        fy = np.concatenate([st["fy"]] + [r[0] for r in st["rings"] if r[0].size])
        fx = np.concatenate([st["fx"]] + [r[1] for r in st["rings"] if r[1].size])
        got, tries, nulls = 0, 0, []
        while got < args.n_null and tries < 80 * args.n_null:
            tries += 1
            dy = int(rng.integers(-fy.min(), ny - fy.max()))
            dx = int(rng.integers(-fx.min(), nx - fx.max()))
            if not np.all(free[fy + dy, fx + dx]):
                continue
            got += 1
            pm, _, _ = region_mean(D, V, (st["fy"] + dy) * nx + (st["fx"] + dx))
            Pn = pm - far_m
            for (ryy, rxx) in st["rings"]:
                if ryy.size < 5:
                    continue
                m, s2, _ = region_mean(D, V, (ryy + dy) * nx + (rxx + dx))
                a, dc = nested_fit(m - far_m, s2, Pn, L, ok)
                if np.isfinite(dc):
                    nulls.append(dc)
        null_all += nulls
        res[i] = dict(area=int((seg == i).sum()), alpha=a_r, dchi2=d_r,
                      npix=n_r, n_null=got, null=nulls)
        print(f"  #{i:3d}  area {res[i]['area']:6,}  平移 {got:3d}  "
              f"alpha(r=1..4) " + " ".join(f"{x:7.3f}" for x in a_r[:4]))

    # ---- 門檻:直接取虛無分布的百分位 --------------------------------------
    # 不繞道「量出 STAT 低估的倍率 f 再乘上 chi2_1 的 95 百分位」:那要先假設尾巴
    # 服從 chi2_1,而天空的空間結構不是高斯白雜訊。f 仍然印出來當診斷用。
    nn  = np.array([x for x in null_all if np.isfinite(x)])
    f   = float(np.median(nn) / CHI2_1_MEDIAN) if nn.size else 1.0
    thr = float(np.percentile(nn, 95))
    per = {i: float(np.percentile(v, 95)) for i, v in
           ((i, [x for x in res[i]["null"] if np.isfinite(x)]) for i in ids)
           if len(v) >= 50}
    # 逐源估計有多少是雜訊:從合池抽同樣多的樣本,看 95 百分位自己會抖多寬。
    if per:
        m = int(np.median([len(res[i]["null"]) for i in per]))
        bs = np.array([np.percentile(rng.choice(nn, m, replace=False), 95)
                       for _ in range(2000)])
        w_o = float(np.diff(np.percentile(list(per.values()), [16, 84]))[0])
        w_b = float(np.diff(np.percentile(bs, [16, 84]))[0])
        real = float(np.sqrt(max(w_o ** 2 - w_b ** 2, 0)))
        print(f"  逐源估計的散布 16-84% 寬 {w_o:.1f};若純粹是估計雜訊(抽 {m} 個)"
              f"應該只有 {w_b:.1f}")
        print(f"  -> 真實源間差異 {real:.1f},佔總變異數 "
              f"{100 * real ** 2 / max(w_o ** 2, 1e-9):.0f}%"
              f"  (越高代表逐源門檻越值得用)")
    print(f"\n虛無分布:{nn.size:,} 個樣本")
    print(f"  中位 {np.median(nn):.2f}  (chi2_1 的中位數是 {CHI2_1_MEDIAN:.3f}"
          f" -> STAT 對「區域平均」的變異數低估 {f:.1f} 倍)")
    print(f"  **門檻 = 實測 95 百分位 = {thr:.2f}**"
          f"  (若硬套 chi2_1 會得到 {f * CHI2_1_95:.2f},尾巴比 chi2_1 重)")
    if per:
        v = np.array(list(per.values()))
        print(f"  逐源各自算的 95 百分位:中位 {np.median(v):.1f},"
              f"範圍 {v.min():.1f} - {v.max():.1f}(合池是否合理的檢查)")

    # 逐源在原理上正確(每個源的 P 形狀、環幾何都不同),但要夠多平移樣本才估得準,
    # 樣本太少時逐源門檻的散布大半只是估計雜訊,所以樣本不足的源退回合池。
    def thr_of(i):
        if args.threshold == "pooled":
            return thr
        return per.get(i, thr)

    print(f"\n門檻模式:{args.threshold}"
          + ("" if args.threshold == "pooled" else
             f"(樣本 < 50 的源退回合池 {thr:.1f})"))
    print(f"\n{'ID':>4}{'area':>7}{'門檻':>8}{'r_stop':>8}   Δchi2(r=1..8)")
    rows = []
    for i in ids:
        d = np.array(res[i]["dchi2"], float)
        below = ~(d > thr_of(i))
        r_stop = args.rmax
        for r in range(len(d) - 1):
            if below[r] and below[r + 1]:
                r_stop = r
                break
        res[i]["r_stop"] = int(r_stop)
        res[i]["threshold"] = float(thr_of(i))
        rows.append((i, res[i]["area"], r_stop))
        print(f"{i:>4}{res[i]['area']:>7}{thr_of(i):>8.1f}{r_stop:>8}   "
              + " ".join(f"{x:8.1f}" if np.isfinite(x) else f"{'-':>8}"
                         for x in d[:8]))
    print(f"\nr_stop 中位 {np.median([r for _, _, r in rows]):.0f} px   "
          f"分布 {np.percentile([r for _, _, r in rows], [16, 84])}")

    # ---- 圖 ---------------------------------------------------------------
    show = sorted(ids, key=lambda i: -res[i]["area"])[:12]
    rr   = np.arange(1, args.rmax + 1)
    fig, ax = plt.subplots(3, 4, figsize=(18, 11), squeeze=False)
    for a, i in zip(ax.ravel(), show):
        d = np.array(res[i]["dchi2"], float)
        al = np.array(res[i]["alpha"], float)
        a.plot(rr, d, "o-", ms=3.5, lw=1.3, color="#1f77b4", label="$\\Delta\\chi^2$")
        a.axhline(thr, color="#d62728", lw=1.2, label=f"threshold {thr:.1f}")
        a.axvline(res[i]["r_stop"] + 0.5, color="k", ls="--", lw=1.0)
        a.set_yscale("symlog", linthresh=1.0)
        a.set_xlabel("r [px]")
        a.set_title(f"#{i}  {res[i]['area']} px   $r_{{stop}}$={res[i]['r_stop']}",
                    fontsize=9)
        a.grid(alpha=0.25)
        b = a.twinx()
        b.plot(rr, al, "s--", ms=3, lw=1.0, color="#2ca02c", alpha=0.7)
        b.set_ylabel("alpha", color="#2ca02c", fontsize=8)
        b.tick_params(axis="y", labelcolor="#2ca02c", labelsize=7)
    ax.ravel()[0].legend(fontsize=7)
    fig.suptitle("is this ring still the same object?   blue = $\\Delta\\chi^2$ for "
                 "adding the footprint spectrum,  green = its amplitude alpha\n"
                 f"threshold from translating each stamp into blank sky "
                 f"({nn.size:,} null samples)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = fig_dir / f"16_ring_consistency_{W.name}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)

    (fig_dir / f"ring_consistency_{W.name}.json").write_text(json.dumps(dict(
        basis=f"{args.basis}_K{args.K}", rmax=args.rmax, min_area=args.min_area,
        n_null=args.n_null, far=args.far, null_margin=args.null_margin,
        stat_underestimate=f, threshold=thr,
        pooled_null=[float(x) for x in nn],
        sources={str(k): v for k, v in res.items()}),
        ensure_ascii=False, indent=2))
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
