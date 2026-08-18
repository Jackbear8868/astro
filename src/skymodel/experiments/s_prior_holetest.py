"""挖洞測試:把 s 的先驗當成擬合的一部分,能不能贏過現行的「事後平滑」。

現行做法的限制
--------------
現在是兩步:①每格獨立解 s ②丟掉源區的、平滑剩下的。問題出在①—— 源區的
chi2 地形是一條長平谷(a·T 和 s·C_sky 都是平滑的連續譜,沿著「a 加一點、s 減
一點」走 chi2 幾乎不變),逐格擬合會在谷底上任意挑一點。等②拿到 s 的時候,
看到的是那個任意的選擇,不是資料。**事後平滑救不回沒被記錄下來的東西。**

結果是核回歸那一項只在訓練點附近有值:離訓練點夠遠的地方 den = 0,那些地方
s_hat 就是純粹的 mu + a(y) + b(x)。

要測的改法
----------
把兩步合成一個目標函數,先驗直接進擬合:

    最小化   || W·(D − A·theta) ||²  +  lambda·( s − M )²        M = mu + a(y) + b(x)

**先驗中心用 M,不是「要求平滑」** —— s 的真實結構是週期 37 px 的儀器條紋,
|grad s|² 那種平滑懲罰會把真的結構磨掉。

實作上不需要組大矩陣:給定 M 之後每一格仍然獨立,而 lambda·(s−M)² 只是在設計
矩陣底下**多加一行** [0…sqrt(lambda)…0],目標值 sqrt(lambda)·M。

怎麼測才乾淨
------------
在 **blank 區**挖一個洞,洞裡有真值(那些格本來就是 blank,自由解出來的 s 可信)。

    ① 洞裡的格從「建場的訓練樣本」中排除 —— 模擬源區的處境
    ② 對洞裡的格**注入一個真實的星系模板**,強度用「源連續譜 / 天空連續譜」的
       比值 f 控制。不注入的話沒有東西可以被 s 吸收,簡併的偏差不會出現,
       測到的只有變異數,不是我們要修的那個病。
    ③ 用三種方法解洞裡的 s,和真值比:
           free     lambda = 0,s 完全自由            = 沒有先驗
           field    s 固定成場的值(不含洞的訓練)     = **現行做法**
           prior L  s 自由但被 lambda 拉向 M         = 提案
    ④ 兩個指標一起看(原則 1):
           s 的偏差   ->  天空扣得對不對
           源流量回收 ->  源有沒有被吃掉

求解器和正式流程一致:lsq_linear / BVLS,s >= 0,單一模板時 A >= 0。

    conda run -n astro python src/skymodel/experiments/s_prior_holetest.py
    conda run -n astro python src/skymodel/experiments/s_prior_holetest.py -f 0.1 0.3 1.0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
from scipy.optimize import lsq_linear
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fitting import build_templates                     # noqa: E402
from templates import air_to_vacuum                      # noqa: E402
from utils import build_s_field, fit_dirs, main_source_group, rowcol_field, scale  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
# 圖與量測值一律寫中央,檔名帶 pointing —— 放在各自的工作區裡的話,
# 要比較幾顆就得開幾個目錄,而且檔名相同排不到一起。
EVAL = ROOT / "results/skymodel/evaluation/s_field"
BAND = (5500.0, 6500.0)      # 量「源 / 天空」比值用的乾淨窗口


def pick_hole(blank, seg, valid, radius, margin):
    """在 blank 區挑一個圓洞:圓心取離任何源、離視野邊緣都最遠的那一格。

    洞要放在「本來就乾淨」的地方,否則真值本身就不可信 —— 那會讓整個測試
    失去基準。半徑由呼叫端給,並回報它相對於 Haro 11 足跡的大小。
    """
    d_src  = ndimage.distance_transform_edt(seg == 0)
    d_edge = ndimage.distance_transform_edt(valid)
    score  = np.where(blank, np.minimum(d_src, d_edge - margin), -np.inf)
    cy, cx = np.unravel_index(np.argmax(score), score.shape)
    yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
    hole = ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2)
    return hole, (int(cy), int(cx))


def solve(y, var, design, lb, ub, lam=None, prior=None):
    """單一 spaxel 的加權最小平方,可選擇加一行先驗。

    先驗那一行是 [0 … sqrt(lam) … 0](只在 s 那一欄非零),目標值 sqrt(lam)·M。
    這一行和資料的行**用同一個最小平方一起解**,所以它不是事後修正,
    而是真的參與了「谷底上要挑哪一點」的決定。
    """
    g = np.isfinite(y) & np.isfinite(var) & (var > 0)
    if g.sum() <= design.shape[0]:
        return None
    sig = np.sqrt(var[g])
    A = design[:, g].T / sig[:, None]
    b = y[g] / sig
    if lam:
        row = np.zeros((1, design.shape[0]))
        row[0, prior[0]] = np.sqrt(lam)
        A = np.vstack([A, row])
        b = np.concatenate([b, [np.sqrt(lam) * prior[1]]])
    return lsq_linear(A, b, bounds=(lb, ub), method="bvls").x


def main():
    ap = argparse.ArgumentParser(description="s 先驗正則化的挖洞測試")
    ap.add_argument("--work", default="results/skymodel/p01")
    ap.add_argument("--run", default=None,
                    help="step05 底下的替代 run 目錄;預設用 pipeline 自己的 step05")
    ap.add_argument("--radius", type=float, default=45.0, help="洞的半徑(px)")
    ap.add_argument("--margin", type=float, default=15.0, help="洞要離視野邊緣多遠")
    ap.add_argument("-f", type=float, nargs="+", default=[0.3],
                    help="注入強度:源連續譜 / 天空連續譜 的比值")
    ap.add_argument("--lam", type=float, nargs="+",
                    default=[1e2, 1e3, 1e4, 1e5, 1e6],
                    help="先驗權重 lambda")
    ap.add_argument("--n-sample", type=int, default=300,
                    help="洞裡抽幾格來解(逐格 BVLS 很慢,所以抽樣)")
    ap.add_argument("--tpl-id", type=int, default=None,
                    help="注入用的模板來自哪個源;省略 = 主源")
    ap.add_argument("--tilt", type=float, nargs="+", default=[0.0],
                    help="注入的源和擬合用的模板之間的**失配**強度:注入的光譜再乘上 "
                         "(lambda/6000)^tilt。tilt = 0 時注入恰好落在模板的行空間裡,"
                         "加上去只改變模板的係數、**對 s 完全沒有影響**。真實情況一定"
                         "有失配,而失配才是把 s 推歪的東西")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    W = ROOT / args.work
    run, _ = fit_dirs(W, args.run)
    meta = json.loads((run / "meta.json").read_text())
    p = meta["s_field_params"]
    cube = ROOT / meta["cube"]

    seg   = fits.getdata(W / "step01/seg.fits").astype(int)
    white = np.asarray(fits.getdata(W / "step01/whitelight.fits"), float)
    wl    = np.load(W / "step03/wavelength.npy")
    C_sky = np.load(W / "step03/sky_continuum.npy")
    basis = np.load(W / f"step03/sky_basis_{meta['basis']}_K{meta['K']}.npy")
    sky   = np.vstack([C_sky, basis])
    s_free = np.load(run / "s_free.npy").astype(float)
    valid  = white != 0
    main, _, _ = main_source_group(seg, np.where(valid, white, np.nan),
                                        W / "step04")
    blank = valid & (seg == 0) & np.isfinite(s_free)

    hole, ctr = pick_hole(blank, seg, valid, args.radius, args.margin)
    hole &= blank
    print(f"洞:圓心 (y, x) = {ctr}  半徑 {args.radius:g} px  "
          f"{int(hole.sum()):,} 格 = Haro 11 足跡的 {100*hole.sum()/main.sum():.0f}%")

    # 場與 M 都在「洞被排除」的條件下建 —— 模擬源區的處境
    s_hat, train = build_s_field(
        s_free, seg, blank, p["r_far"], p["r_far_haro"], p["clip"],
                                main=main, exclude=hole)
    M, _, _ = rowcol_field(s_free, train)
    print(f"訓練點 {int(train.sum()):,} 格(洞已排除)")
    print(f"洞裡:真值 s 中位 {np.median(s_free[hole]):.5f}   "
          f"M 中位 {np.median(M[hole]):.5f}   場 中位 {np.median(s_hat[hole]):.5f}")

    # ---- 注入用的模板 ----
    best = np.load(ROOT / meta["best"])
    T_all = build_templates(best, air_to_vacuum(wl))
    tid = args.tpl_id if args.tpl_id else int(best["id"][np.nanargmax(
        np.nansum(np.abs(best["A"]), axis=1))])
    T = T_all[tid]
    a_vec = np.nan_to_num(best["A"][list(best["id"]).index(tid)][:T.shape[1]])
    src = T @ a_vec
    bnd = (wl >= BAND[0]) & (wl < BAND[1])
    unit = src / np.median(src[bnd]) * np.median(C_sky[bnd])   # f = 1 時源=天空
    print(f"注入模板:源 ID {tid}({best['group'][list(best['id']).index(tid)]}),"
          f" {T.shape[1]} 欄")

    ys, xs = np.nonzero(hole)
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(ys.size, size=min(args.n_sample, ys.size), replace=False)
    ys, xs = ys[sel], xs[sel]
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    with fits.open(cube, memmap=True) as h:
        D = np.asarray(h["DATA"].data[:, y0:y1, x0:x1], np.float32)
        V = np.asarray(h["STAT"].data[:, y0:y1, x0:x1], np.float32)
    D = D[:, ys - y0, xs - x0].astype(np.float64)
    V = V[:, ys - y0, xs - x0].astype(np.float64)
    s_true, M_h = s_free[ys, xs], M[ys, xs]
    sf_h = s_hat[ys, xs]
    n_comp = T.shape[1]

    # 設計矩陣與正式流程一致:[a₁…a_ncomp, s, c₁…c_{K−1}],s >= 0,
    # 單一模板(恆星)時 A >= 0
    design_free = np.vstack([T.T, sky])
    lb = np.full(design_free.shape[0], -np.inf)
    if n_comp == 1:
        lb[0] = 0.0
    lb[n_comp] = 0.0
    ub = np.full(design_free.shape[0], np.inf)
    design_fix = np.vstack([T.T, sky[1:]])          # s 固定時 C_sky 不在設計矩陣裡
    lb_fix = np.full(design_fix.shape[0], -np.inf)
    if n_comp == 1:
        lb_fix[0] = 0.0
    ub_fix = np.full(design_fix.shape[0], np.inf)

    rows = []
    for tilt in args.tilt:
      # 失配:注入的光譜乘上一個平滑的斜率,讓它離開模板的行空間
      warp = (wl / 6000.0) ** tilt
      for f in args.f:
        inj = f * unit * warp                            # 注入的源光譜
        Dj = D + inj[:, None]
        flux_true = float(np.sum(inj[bnd]))

        def run_one(name, lam=None, s_fix=None):
            ds, fr = [], []
            for j in range(ys.size):
                if s_fix is None:
                    th = solve(Dj[:, j], V[:, j], design_free, lb, ub,
                               lam=lam, prior=(n_comp, M_h[j]))
                    if th is None:
                        continue
                    ds.append(th[n_comp] - s_true[j])
                    a = th[:n_comp]
                else:
                    th = solve(Dj[:, j] - s_fix[j] * C_sky, V[:, j],
                               design_fix, lb_fix, ub_fix)
                    if th is None:
                        continue
                    ds.append(s_fix[j] - s_true[j])
                    a = th[:n_comp]
                fr.append(float(np.sum((T @ a)[bnd])) / flux_true)
            ds, fr = np.array(ds), np.array(fr)
            rows.append(dict(f=f, tilt=tilt, method=name, n=ds.size,
                             bias=float(np.median(ds)), sct=float(scale(ds)),
                             flux=float(np.median(fr))))
            print(f"  f={f:<4g} t={tilt:<4g} {name:<12} n={ds.size:<4} "
                  f"s 偏差 {np.median(ds):+.5f}  散布 {scale(ds):.5f}   "
                  f"源流量回收 {np.median(fr):.3f}")

        print(f"\n注入 f = {f:g}(源 = 天空的 {100*f:.0f}%),失配 tilt = {tilt:g}"
              f"(藍紅端相差 {100*(warp[-1]/warp[0]-1):+.0f}%)")
        run_one("free")                                  # lambda = 0
        run_one("field", s_fix=sf_h)                     # 現行做法
        for lam in args.lam:
            # 標籤直接印 lambda 本身 —— 印 log10 取整的話,相近的 lambda 會撞成
            # 同一個標籤,表格裡就分不出是哪一個。
            run_one(f"prior {lam:.0e}", lam=lam)

    # ---- 圖 ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for f in args.f:
        r = [x for x in rows if x["f"] == f and x["tilt"] == args.tilt[-1]
             and x["method"].startswith("prior")]
        lam = [float(x["method"].split()[1]) for x in r]   # 標籤就是 lambda
        for k, (a_, key, lab) in enumerate(
                ((ax[0], "bias", "s bias  (median $\\hat{s}-s_{true}$)"),
                 (ax[1], "flux", "source flux recovered"))):
            a_.plot(lam, [abs(x[key]) if key == "bias" else x[key] for x in r],
                    "o-", label=f"prior, f={f:g}")
            for m, c, ls in (("free", "0.4", "--"), ("field", "#d62728", ":")):
                v = [x for x in rows if x["f"] == f and x["tilt"] == args.tilt[-1]
                     and x["method"] == m][0]
                a_.axhline(abs(v[key]) if key == "bias" else v[key],
                           color=c, ls=ls, lw=1.2, label=f"{m}, f={f:g}")
            a_.set_xscale("log"); a_.set_xlabel("$\\lambda$")
            a_.set_ylabel(lab); a_.grid(alpha=.25)
    ax[0].set_yscale("log")
    ax[1].axhline(1.0, color="k", lw=0.8)
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=8)
    fig.suptitle(f"hole test — {args.work}   hole r={args.radius:g} px "
                 f"({int(hole.sum()):,} spaxels), {ys.size} sampled", fontsize=12)
    fig.tight_layout()
    out = EVAL / f"s_prior_holetest_{W.name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {out}")
    (out.with_suffix(".json")).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
