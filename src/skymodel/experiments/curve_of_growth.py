"""累積 Δchi2 的成長曲線 —— 峰值就是最佳半徑,不需要門檻。

ring_consistency.py 算的是**逐圈獨立**的 Δchi2:第 r 圈自己一塊區域,問「這一圈
還是不是同一個天體」。那條曲線一定隨 r 遞減,因為表面亮度往外掉。所以它需要一個
門檻(從平移空白區量出來的)來決定什麼時候算「掉進雜訊裡」。

這支改算**累積**的:把足跡 + 第 1 到第 r 圈當成**一整塊區域**去擬合。

    Δchi2_cum(r) = 對 <足跡 ∪ 環1..環r> 的平均光譜做同樣的巢狀擬合

行為完全不同,而且更有用:

    加進一圈**有訊號**的區域   -> 訊號增加得比雜訊快 -> 累積 Δchi2 上升
    加進一圈**只有雜訊**的區域 -> 把平均值稀釋掉     -> 累積 Δchi2 下降

所以曲線會有一個**極大值**,那個峰就是「再長下去只會變差」的位置。這正是天文裡
量測孔徑光度時的 curve of growth 論證 —— 最佳孔徑在 S/N 的峰值,不是在某個
人為門檻上。**峰值不需要門檻,所以也不需要虛無分布。**

同時畫三條線,好對照:

    per-ring     只有第 r 圈(= ring_consistency.py 在算的)
    cum shell    第 1 到第 r 圈的聯集,不含足跡
    cum total    足跡 + 第 1 到第 r 圈的聯集   <- 峰值取這條

    conda run -n astro python src/skymodel/experiments/curve_of_growth.py \\
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ring_consistency import label_aware, region_mean, nested_fit

ROOT    = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "results/skymodel/evaluation/masking/edge_oversub"


def main():
    ap = argparse.ArgumentParser(description="累積 Δchi2 的成長曲線")
    ap.add_argument("--work", required=True,
                    help="pointing 的工作區,例如 results/skymodel/p01")
    ap.add_argument("--cube", default=None,
                    help="含天空的 cube;預設由 pNN 的編號推出 "
                         "data/wsky/DATACUBE_FINAL_N.fits")
    ap.add_argument("--seg", default=None,
                    help="segmentation;預設為工作區的 step01/seg.fits")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--rmax", type=int, default=12)
    ap.add_argument("--min-area", type=int, default=60)
    ap.add_argument("--far", type=float, default=40.0)
    ap.add_argument("--figdir", default=str(FIGURES))
    args = ap.parse_args()

    W = ROOT / args.work
    STEP01, STEP03 = W / "step01", W / "step03"
    CUBE = ROOT / (args.cube or f"data/wsky/DATACUBE_FINAL_{int(W.name[1:])}.fits")
    seg_path = Path(args.seg) if args.seg else STEP01 / "seg.fits"

    fig_dir = Path(args.figdir); fig_dir.mkdir(parents=True, exist_ok=True)
    seg   = fits.getdata(seg_path).astype(int)
    white = fits.getdata(STEP01 / "whitelight.fits")
    L = np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")

    with fits.open(CUBE, memmap=True) as h:
        D = np.asarray(h["DATA"].data, np.float32)
        V = np.asarray(h["STAT"].data, np.float32)
    nz, ny, nx = D.shape
    D = D.reshape(nz, -1)
    V = V.reshape(nz, -1)

    dist, owner = label_aware(seg, args.rmax)
    valid = white != 0
    far   = ((dist > args.far) & (seg == 0) & valid).ravel()
    far_m, _, _ = region_mean(D, V, np.flatnonzero(far))
    ok = np.isfinite(far_m) & np.all(np.isfinite(L), axis=0)

    ids = [int(i) for i in np.unique(seg[seg > 0])
           if (seg == i).sum() >= args.min_area]
    print(f"源 {len(ids)} 個   rmax {args.rmax}")

    res = {}
    for i in ids:
        fp   = (seg == i).ravel()
        fp_m, _, _ = region_mean(D, V, np.flatnonzero(fp))
        P    = fp_m - far_m                    # 參考光譜:足跡 − 遠場,沒有擬合
        ring = [((owner == i) & (dist > r - 1) & (dist <= r)).ravel()
                for r in range(1, args.rmax + 1)]

        def dchi2(mask):
            """一塊區域的 Δchi2。mask 是攤平的布林陣列。"""
            m, s2, _ = region_mean(D, V, np.flatnonzero(mask))
            return nested_fit(m - far_m, s2, P, L, ok)[1]

        per, shell, total, npx = [], [], [], []
        acc = np.zeros_like(fp)                # 已經累積進來的環
        for r in range(args.rmax):
            per.append(dchi2(ring[r]) if ring[r].sum() >= 5 else np.nan)
            acc = acc | ring[r]
            shell.append(dchi2(acc))           # 只有加進來的那些圈
            total.append(dchi2(acc | fp))      # 足跡也算進去
            npx.append(int(acc.sum()))
        t0 = dchi2(fp)                         # r = 0:只有足跡,累積曲線的起點

        # 峰值取「累積殼」那條。**不能用含足跡的那條** —— P 就是從足跡量出來的,
        # 「用 P 去擬合足跡」必然完美,那個起點是循環論證灌出來的,之後加任何
        # 東西都只會稀釋它,峰值一律落在 r=0。
        t = np.array([t0] + total, float)
        rbest = int(np.nanargmax(np.array(shell, float))) + 1
        res[i] = dict(area=int(fp.sum()), per=per, shell=shell,
                      total=t.tolist(), npix=npx, r_best=rbest)
        print(f"  #{i:3d}  area {res[i]['area']:6,}  峰值在 r = {rbest:2d}   "
              f"cum shell(r=1..5) " + " ".join(f"{x:8.0f}"
                                                for x in shell[:5]))

    rb = [v["r_best"] for v in res.values()]
    print(f"\n峰值 r 的中位 {np.median(rb):.0f},分布 {np.percentile(rb,[16,84])}"
          f",範圍 {min(rb)}-{max(rb)}")
    print(f"  撞到 rmax={args.rmax} 的有 {sum(1 for x in rb if x == args.rmax)} 個"
          f" —— 那些源的光在 rmax 之內沒有用完,峰值不是量出來的")

    show = sorted(ids, key=lambda i: -res[i]["area"])[:12]
    rr   = np.arange(1, args.rmax + 1)
    fig, ax = plt.subplots(3, 4, figsize=(19, 12), squeeze=False)
    for a, i in zip(ax.ravel(), show):
        v = res[i]
        a.plot(rr, v["shell"], "o-", ms=4, lw=1.7, color="#d62728",
               label="cumulative, added shell only")
        a.plot(rr, v["per"], "^:", ms=3, lw=1.1, color="#1f77b4",
               label="per-ring (what ring_consistency used)")
        a.axvline(v["r_best"], color="k", ls="--", lw=1.1)
        a.set_yscale("symlog", linthresh=10)
        a.set_xlabel("r [px]")
        a.set_title(f"#{i}  {v['area']} px   peak at r = {v['r_best']}", fontsize=9)
        a.grid(alpha=0.25)
    ax.ravel()[0].set_ylabel("$\\Delta\\chi^2$")
    ax.ravel()[0].legend(fontsize=7)
    fig.suptitle("curve of growth in $\\Delta\\chi^2$   —   cumulative over the added "
                 "shell (footprint excluded: P comes from it, so including it is "
                 "circular)\ndashed line = the maximum, i.e. where adding more area "
                 "starts to dilute more than it adds", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = fig_dir / f"21_curve_of_growth_{W.name}.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    (fig_dir / f"curve_of_growth_{W.name}.json").write_text(
        json.dumps(dict(rmax=args.rmax, basis=f"{args.basis}_K{args.K}",
                        sources={str(k): v for k, v in res.items()}),
                   ensure_ascii=False, indent=2))
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
