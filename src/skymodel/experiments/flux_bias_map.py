"""源流量的加法偏差:它在全場是常數嗎?源模型放軟一點會不會放大得更兇?

背景:在**沒有源**的 blank 格上放一組源模板去擬合,擬合會給出一個固定的負流量
(單位:源亮度 = 天空亮度時為 1)。它是加法的,所以源越暗被吃掉的比例越大。
機制:s 固定之後模型裡沒有自由的平滑成分,天空連續譜的形狀誤差只能倒進源模板。

這支同時回答兩個問題,因為它們共用同一套量測:

    ②  偏差在全場是常數嗎?
        是 -> 它就是一個**可校正的系統誤差**,量出來直接扣掉即可
        否 -> 得修模型
        量法:blank 格散布在整個視場上各量一次,畫成圖,並看它和位置/亮度的關係

    ③  源模型放軟(成分多)是不是放大得更兇?
        四條星系本徵譜彼此高度相關,要湊出一個小傾斜需要一組大係數。
        量法:同一批格,把模板截成 1 / 2 / 3 / 4 欄各算一次

不需要注入任何東西 —— blank 格本來就沒有源,真值就是 0。

    conda run -n astro python src/skymodel/experiments/flux_bias_map.py
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
from utils import (air_to_vacuum, build_amplitude_field, build_templates,  # noqa: E402
                   main_source_group, robust_spread)
from s_flux_bias import wls                              # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
# 圖與量測值一律寫中央,檔名帶 pointing —— 放在各自的工作區裡的話,
# 要比較幾顆就得開幾個目錄,而且檔名相同排不到一起。
EVAL = ROOT / "results/skymodel/evaluation/s_field"
BAND = (5500.0, 6500.0)


def main():
    ap = argparse.ArgumentParser(description="加法偏差的空間分布與模板欄數的影響")
    ap.add_argument("--work", default="results/skymodel/p01")
    ap.add_argument("--run", default=None)
    ap.add_argument("--n-sample", type=int, default=600)
    ap.add_argument("--ncomp", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--tpl-id", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    W = ROOT / args.work
    run, _ = fit_dirs(W, args.run)
    meta = json.loads((run / "meta.json").read_text())
    p = sky_amplitude_params(meta)
    seg   = fits.getdata(W / "step01/seg.fits").astype(int)
    white = np.asarray(fits.getdata(W / "step01/whitelight.fits"), float)
    wl    = np.load(W / "step03/wavelength.npy")
    C_sky = np.load(W / "step03/sky_continuum.npy")
    basis = np.load(W / f"step03/sky_basis_{meta['basis']}_K{meta['K']}.npy")
    s_free = np.load(run / "s_free.npy").astype(float)
    valid = white != 0
    mg, _, _ = main_source_group(seg, np.where(valid, white, np.nan),
                                        W / "step04")
    blank = valid & (seg == 0) & np.isfinite(s_free)
    s_hat, _ = build_amplitude_field(s_free, seg, blank, p["min_source_distance"], p["min_main_source_distance"],
                                p["train_clip_sigma"],
                                main=mg)

    # "classification" is the key; "best" is what step5 wrote before the
    # parameter was renamed, and products made then are still on disk.
    best = np.load(ROOT / (meta.get("classification") or meta["best"]))
    T_all = build_templates(best, air_to_vacuum(wl))
    tid = args.tpl_id if args.tpl_id else int(best["id"][np.nanargmax(
        np.nansum(np.abs(best["A"]), axis=1))])
    T = T_all[tid]
    a_vec = np.nan_to_num(best["A"][list(best["id"]).index(tid)][:T.shape[1]])
    bnd = (wl >= BAND[0]) & (wl < BAND[1])
    src = T @ a_vec
    ref = float(np.sum((src / np.median(src[bnd]) * np.median(C_sky[bnd]))[bnd]))
    print(f"模板 ID {tid},{T.shape[1]} 欄   參考流量(源 = 天空)= {ref:.0f}")

    # 全場均勻抽樣 —— 只在一個角落抽的話,看不出「是不是常數」
    ys, xs = np.nonzero(blank)
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(ys.size, min(args.n_sample, ys.size), replace=False)
    ys, xs = ys[sel], xs[sel]
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    with fits.open(ROOT / meta["cube"], memmap=True) as h:
        D = np.asarray(h["DATA"].data[:, y0:y1, x0:x1], np.float32)[:, ys - y0, xs - x0]
        V = np.asarray(h["STAT"].data[:, y0:y1, x0:x1], np.float32)[:, ys - y0, xs - x0]
    D, V = D.astype(np.float64), V.astype(np.float64)
    print(f"blank 抽樣 {ys.size:,} 格,散布在整個視場\n")

    res = {}
    for nc in args.ncomp:
        Tn = T[:, :nc]
        f = []
        for j in range(ys.size):
            y = D[:, j] - s_hat[ys[j], xs[j]] * C_sky
            c = wls(basis, y, V[:, j])
            if c is None:
                f.append(np.nan); continue
            a = wls(Tn.T, y - basis.T @ c, V[:, j])
            f.append(float(np.sum((Tn @ a)[bnd])) / ref)
        res[nc] = np.array(f)

    print(f"{'模板欄數':>10}{'偏差 中位':>12}{'逐格散布':>12}{'中位的標準誤':>14}")
    print("-" * 48)
    for nc in args.ncomp:
        v = res[nc][np.isfinite(res[nc])]
        se = 1.253 * robust_spread(v) / np.sqrt(v.size)
        print(f"{nc:>10}{np.median(v):>12.4f}{robust_spread(v):>12.4f}{se:>14.4f}")

    full = res[args.ncomp[-1]]
    ok = np.isfinite(full)
    d_main = ndimage.distance_transform_edt(~mg)[ys, xs]
    print(f"\n② 空間相依性(用 {args.ncomp[-1]} 欄):")
    print(f"  與 x 的相關 {np.corrcoef(xs[ok], full[ok])[0,1]:+.3f}   "
          f"與 y 的相關 {np.corrcoef(ys[ok], full[ok])[0,1]:+.3f}   "
          f"與離主源距離 {np.corrcoef(d_main[ok], full[ok])[0,1]:+.3f}")
    for lo, hi in ((0, 80), (80, 160), (160, 240), (240, 400)):
        m = ok & (d_main >= lo) & (d_main < hi)
        if m.sum() > 20:
            print(f"    離主源 {lo:>3}-{hi:<3} px  {int(m.sum()):>4} 格   "
                  f"偏差 {np.median(full[m]):+.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    cm = np.full(seg.shape, np.nan); cm[ys, xs] = full
    v = 3 * robust_spread(full[ok])
    im = ax[0].imshow(cm, origin="lower", cmap="RdBu_r",
                      vmin=np.median(full[ok]) - v, vmax=np.median(full[ok]) + v)
    plt.colorbar(im, ax=ax[0], fraction=.046)
    ax[0].contour(seg > 0, levels=[.5], colors="k", linewidths=.4)
    ax[0].set_title(f"additive bias per spaxel ({args.ncomp[-1]} template cols)",
                    fontsize=10)
    ax[1].plot(d_main[ok], full[ok], ".", ms=2, alpha=.4)
    ax[1].axhline(np.median(full[ok]), color="#d62728", lw=1.2)
    ax[1].set_xlabel("distance from main source [px]")
    ax[1].set_ylabel("bias"); ax[1].grid(alpha=.25)
    ax[1].set_title("② is it constant across the field?", fontsize=10)
    ax[2].errorbar(args.ncomp,
                   [np.median(res[n][np.isfinite(res[n])]) for n in args.ncomp],
                   yerr=[1.253 * robust_spread(res[n][np.isfinite(res[n])])
                         / np.sqrt(np.isfinite(res[n]).sum()) for n in args.ncomp],
                   fmt="o-")
    ax[2].axhline(0, color="k", lw=.8)
    ax[2].set_xlabel("number of template columns")
    ax[2].set_ylabel("bias"); ax[2].grid(alpha=.25)
    ax[2].set_xticks(args.ncomp)
    ax[2].set_title("③ does a stiffer source model help?", fontsize=10)
    fig.suptitle(f"additive source-flux bias measured on blank spaxels "
                 f"(truth = 0)   {args.work}", fontsize=12)
    fig.tight_layout()
    o = EVAL / f"flux_bias_map_{W.name}.png"
    o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(o, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {o}")


if __name__ == "__main__":
    main()
