"""低秩補全 vs 現行的 row+col 場 —— 多洞、配對的挖洞測試。

為什麼要找一個比 row+col 更強的模型
----------------------------------
任何「拿鄰近訓練點做加權平均」的補法都只能在有樣本的地方作用,被源遮住的
大洞裡無值可用。a(y) 之所以伸得進洞裡,是因為它是**被整列共用的一個參數**,
不是鄰域的平均。

所以方向是讓剩下的結構也變成參數共用的模型:

    s(y, x)  ≈  Sum_{k=1..R}  u_k(y) · v_k(x)

u_k 由整列決定、v_k 由整行決定,全域共用,自動伸得進洞。現行的
mu + a(y) + b(x) 是它的特例(加法模型 = 秩 3),所以這是嚴格的推廣。

怎麼判定贏
----------
**判準在跑之前訂好**,不是跑完挑對自己有利的:

    ① 配對勝率 > 55%     同一格比同一格。真值的雜訊對兩個方法是同一份,配對
                          比較把它消掉,比「比散布」敏感得多
    ② 扣掉雜訊底後誤差更小   sqrt(scatter² − sigma_n²)。完美的預測器散布也只能
                          到 sigma_n,不扣掉就是在比雜訊
    ③ >= 3 個不同位置的洞都成立   單一個洞可能只是運氣

**這支不回答 over-subtraction。** 洞挖在 blank 區,源旁邊會怎樣要靠端到端的量測,
而那個量測目前沒有(原本的 check_pointing 已於 2026-08-18 刪除)。

    conda run -n astro python src/skymodel/experiments/s_lowrank.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from products import fit_dirs, sky_amplitude_params  # noqa: E402
from utils import build_amplitude_field, main_source_group, median_polish, robust_spread  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
# 圖與量測值一律寫中央,檔名帶 pointing —— 放在各自的工作區裡的話,
# 要比較幾顆就得開幾個目錄,而且檔名相同排不到一起。
EVAL = ROOT / "results/skymodel/evaluation/s_field"


CUR = "current"      # 現行的場 = mu + a(y) + b(x),配對比較的對照組


def lowrank_complete(s, obs, rank, init, n_iter=60, tol=1e-7):
    """缺失資料的低秩補全(iterative SVD / hard-impute)。

    init 用 mu + a(y) + b(x) —— 起點不是隨便挑的:觀測很稀疏的 row/col,SVD
    會被少數幾格帶著走,用一個已經合理的曲面當起點可以把它們釘住。
    """
    X = np.where(obs, s, init)
    prev = None
    for _ in range(n_iter):
        U, sv, Vt = np.linalg.svd(X, full_matrices=False)
        Xr = (U[:, :rank] * sv[:rank]) @ Vt[:rank]
        if prev is not None and np.max(np.abs(Xr - prev)) < tol:
            break
        prev = Xr
        X = np.where(obs, s, Xr)
    return Xr


def holes(blank, seg, valid, radius, margin, n, rng):
    """挑 n 個互不重疊的圓洞,都放在離源、離邊緣夠遠的地方。

    一個洞可能只是運氣。多個洞的結論必須一致,否則不算改善。
    """
    d_src  = ndimage.distance_transform_edt(seg == 0)
    d_edge = ndimage.distance_transform_edt(valid)
    score  = np.where(blank, np.minimum(d_src, d_edge - margin), -np.inf)
    yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
    out, taken = [], np.zeros(seg.shape, bool)
    sc = score.copy()
    for _ in range(n):
        cy, cx = np.unravel_index(np.argmax(sc), sc.shape)
        if not np.isfinite(sc[cy, cx]):
            break
        h = ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2)
        out.append((h & blank, (int(cy), int(cx))))
        taken |= ((yy - cy) ** 2 + (xx - cx) ** 2 <= (2.2 * radius) ** 2)
        sc = np.where(taken, -np.inf, sc)
    return out


def main():
    ap = argparse.ArgumentParser(description="低秩補全 vs 現行做法")
    ap.add_argument("--work", default="results/skymodel/p01")
    ap.add_argument("--run", default=None)
    ap.add_argument("--radius", type=float, default=45.0)
    ap.add_argument("--margin", type=float, default=15.0)
    ap.add_argument("--n-hole", type=int, default=4)
    ap.add_argument("--ranks", type=int, nargs="+",
                    default=[2, 3, 4, 6, 8, 10, 14, 20])
    ap.add_argument("--sigma-n", type=float, default=0.0047,
                    help="真值自己的量測雜訊(分半測試量到的)。用來扣掉雜訊底")
    args = ap.parse_args()

    W = ROOT / args.work
    run, _ = fit_dirs(W, args.run)
    meta = json.loads((run / "meta.json").read_text())
    p = sky_amplitude_params(meta)
    seg   = fits.getdata(W / "step01/seg.fits").astype(int)
    white = np.asarray(fits.getdata(W / "step01/whitelight.fits"), float)
    s     = np.load(run / "s_free.npy").astype(float)
    valid = white != 0
    main, _, _ = main_source_group(seg, np.where(valid, white, np.nan),
                                        W / "step04")
    blank = valid & (seg == 0) & np.isfinite(s)

    rng = np.random.default_rng(0)
    hs = holes(blank, seg, valid, args.radius, args.margin, args.n_hole, rng)
    print(f"{len(hs)} 個洞,半徑 {args.radius:g} px:"
          + "  ".join(f"{c} ({int(h.sum()):,}格)" for h, c in hs))
    print(f"真值的量測雜訊 sigma_n = {args.sigma_n:.4f}(散布的下限)\n")

    tab = {}
    for hi, (hole, ctr) in enumerate(hs):
        _, train = build_amplitude_field(s, seg, blank, p["min_source_distance"], p["min_main_source_distance"],
                                        p["train_clip_sigma"],
                                main=main, exclude=hole)
        M, _, _ = median_polish(s, train)
        truth = s[hole]
        # build_amplitude_field 回傳的就是 median_polish 的結果 —— 現行的場**就是** M,
        # 兩者分開列會看起來像兩個方法,實際上是同一個陣列
        cur   = M[hole]
        preds = {CUR: cur}
        for R in args.ranks:
            Xr = lowrank_complete(s, train, R, M)
            preds[f"rank {R}"] = Xr[hole]

        print(f"--- 洞 {hi+1} @ {ctr}   {truth.size:,} 格")
        print(f"{'方法':>10}{'偏差':>10}{'散布':>9}{'扣雜訊底':>10}"
              f"{'配對勝率':>10}{'配對差中位':>12}")
        for nm, pr in preds.items():
            e = pr - truth
            sc_ = robust_spread(e)
            deconv = np.sqrt(max(sc_**2 - args.sigma_n**2, 0.0))
            win = np.mean(np.abs(e) < np.abs(cur - truth))
            dmed = np.median(np.abs(e) - np.abs(cur - truth))
            tab.setdefault(nm, []).append((float(np.median(e)), sc_, deconv,
                                           float(win), float(dmed)))
            mark = "  <- 基準" if nm == CUR else ""
            print(f"{nm:>10}{np.median(e):>+10.5f}{sc_:>9.5f}{deconv:>10.5f}"
                  f"{100*win:>9.0f}%{dmed:>+12.6f}{mark}")
        print()

    print("=== 四個洞的總結(勝率是對 current 的配對比較)===")
    # |偏差| 也要報。散布和配對勝率都被雜訊主導,但**對 over-subtraction 有影響的
    # 是偏差** —— 場在源旁系統性偏高,保留率就跟著掉,而散布會在環上平均掉。
    print(f"{'方法':>10}{'|偏差| 中位':>14}{'扣雜訊底 中位':>16}"
          f"{'配對勝率 中位':>16}{'每個洞的偏差':>34}")
    for nm, v in tab.items():
        v = np.array(v)
        print(f"{nm:>10}{np.median(np.abs(v[:,0])):>14.5f}{np.median(v[:,2]):>16.5f}"
              f"{100*np.median(v[:,3]):>15.0f}%"
              + "   " + " ".join(f"{x:+.4f}" for x in v[:, 0]))

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    names = [n for n in tab if n.startswith("rank")]
    rk = [int(n.split()[1]) for n in names]
    ax[0].plot(rk, [np.median(np.array(tab[n])[:, 2]) for n in names], "o-",
               label="low-rank completion")
    ax[0].axhline(np.median(np.array(tab[CUR])[:, 2]), color="#d62728", ls="--",
                  label=CUR)
    ax[0].set_xlabel("rank"); ax[0].set_ylabel("error, noise floor removed")
    ax[0].grid(alpha=.25); ax[0].legend(fontsize=8)
    ax[1].plot(rk, [100*np.median(np.array(tab[n])[:, 3]) for n in names], "o-")
    ax[1].axhline(50, color="k", lw=0.8)
    ax[1].axhline(55, color="#d62728", ls="--", lw=1.0, label="55% (判準)")
    ax[1].set_xlabel("rank"); ax[1].set_ylabel("paired win rate vs current [%]")
    ax[1].grid(alpha=.25); ax[1].legend(fontsize=8)
    fig.suptitle(f"low-rank completion vs current field   {args.work}   "
                 f"{len(hs)} holes, r={args.radius:g} px", fontsize=12)
    fig.tight_layout()
    o = EVAL / f"s_lowrank_{W.name}.png"
    o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(o, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {o}")


if __name__ == "__main__":
    main()
