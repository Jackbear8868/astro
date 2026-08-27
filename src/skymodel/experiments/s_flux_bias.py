"""源流量的加法偏差是誰造成的?

在**沒有源**的 blank 格上放一組源模板去擬合(見 s_prior_holetest),擬合會給出
一個固定的負流量(單位:源亮度 = 天空亮度時為 1)。它不隨注入強度按比例改變,
所以是**加法**的偏差,不是按比例吸收 —— 也因此源越暗,被吃掉的比例越大。

這支就查它從哪來。兩個互補的量法:

    A  joint     [T, sky] 一起解                  = 正式流程的做法
    B  project   先只用 sky 解(和 fit_blank 一樣),
                 再把殘差投影到 T 上              = 沒有競爭的情況

    A ≈ B   ->  偏差在**資料/殘差**裡:天空模型本身留下一個長得像負的源的東西
    A ≪ B   ->  偏差來自**競爭**:設計矩陣裡的其他欄把源的光搶走了

再加一個 K 掃描(天光線 basis 用幾條)。若 |偏差| 隨 K 增大,兇手就是那些自由的
basis —— 30 個自由參數去擬合一條平滑的星系連續譜,吸走一部分是很合理的。

不需要注入任何東西:blank 格本來就沒有源,真值就是 0。

    conda run -n astro python src/skymodel/experiments/s_flux_bias.py
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
from products import fit_dirs, sky_amplitude_params  # noqa: E402
from utils import (air_to_vacuum, build_amplitude_field, build_templates,  # noqa: E402
                   main_source_group, robust_spread)
from s_prior_holetest import pick_hole                   # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
# 圖與量測值一律寫中央,檔名帶 pointing —— 放在各自的工作區裡的話,
# 要比較幾顆就得開幾個目錄,而且檔名相同排不到一起。
EVAL = ROOT / "results/skymodel/evaluation/s_field"
BAND = (5500.0, 6500.0)


def wls(design, y, var, lb=None):
    """加權最小平方。lb 給定時走 BVLS,和正式流程一致。"""
    g = np.isfinite(y) & np.isfinite(var) & (var > 0)
    if g.sum() <= design.shape[0]:
        return None
    sig = np.sqrt(var[g])
    A, b = design[:, g].T / sig[:, None], y[g] / sig
    if lb is None:
        return np.linalg.lstsq(A, b, rcond=None)[0]
    return lsq_linear(A, b, bounds=(lb, np.full(design.shape[0], np.inf)),
                      method="bvls").x


def main():
    ap = argparse.ArgumentParser(description="源流量加法偏差的來源")
    ap.add_argument("--work", default="results/skymodel/p01")
    ap.add_argument("--run", default=None)
    ap.add_argument("--radius", type=float, default=45.0)
    ap.add_argument("--margin", type=float, default=15.0)
    ap.add_argument("--n-sample", type=int, default=250)
    ap.add_argument("--klist", type=int, nargs="+", default=[1, 2, 5, 10, 20, 30])
    ap.add_argument("--tpl-id", type=int, default=None)
    ap.add_argument("--fit-window", type=float, nargs=2, default=[5100, 7500],
                    metavar=("LO", "HI"),
                    help="第三欄:只用這段通道去擬合模板振幅。動機:8000 A 之後是 OH "
                         "帶,天光線殘差極大,若模板的振幅其實是被那一段決定的,"
                         "限制在乾淨窗口就會看到偏差塌掉")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    W = ROOT / args.work
    run, _ = fit_dirs(W, args.run)
    meta = json.loads((run / "meta.json").read_text())
    p = sky_amplitude_params(meta)

    seg   = fits.getdata(W / "step01/segmentation_input.fits").astype(int)
    white = np.asarray(fits.getdata(W / "step01/whitelight_nosky.fits"), float)
    wl    = np.load(W / "step03/wavelength.npy")
    C_sky = np.load(W / "step03/sky_continuum.npy")
    basis = np.load(W / f"step03/sky_line_basis_{meta['basis']}_K{meta['K']}.npy")
    s_free = np.load(run / "sky_continuum_amplitude_per_spaxel.npy").astype(float)
    valid = white != 0
    main, _, _ = main_source_group(seg, np.where(valid, white, np.nan),
                                        W / "step04")
    blank = valid & (seg == 0) & np.isfinite(s_free)

    hole, ctr = pick_hole(blank, seg, valid, args.radius, args.margin)
    hole &= blank
    s_hat, train = build_amplitude_field(s_free, seg, blank, p["min_source_distance"],
                                    p["min_main_source_distance"], p["train_clip_sigma"],
                                main=main, exclude=hole)

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
    unit = src / np.median(src[bnd]) * np.median(C_sky[bnd])   # 源 = 天空 -> 1
    ref = float(np.sum(unit[bnd]))
    n_comp = T.shape[1]
    print(f"洞 {int(hole.sum()):,} 格   模板 ID {tid} "
          f"({best['group'][list(best['id']).index(tid)]}, {n_comp} 欄)")
    print(f"參考流量(源 = 天空)= {ref:.1f}\n")

    ys, xs = np.nonzero(hole)
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(ys.size, size=min(args.n_sample, ys.size), replace=False)
    ys, xs = ys[sel], xs[sel]
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    with fits.open(ROOT / meta["cube"], memmap=True) as h:
        D = np.asarray(h["DATA"].data[:, y0:y1, x0:x1], np.float32)
        V = np.asarray(h["STAT"].data[:, y0:y1, x0:x1], np.float32)
    D = D[:, ys - y0, xs - x0].astype(np.float64)
    V = V[:, ys - y0, xs - x0].astype(np.float64)
    sf = s_hat[ys, xs]

    fitw = (wl >= args.fit_window[0]) & (wl < args.fit_window[1])
    print(f"擬合窗口(第三欄):{args.fit_window[0]:.0f}-{args.fit_window[1]:.0f} A"
          f",{int(fitw.sum())} / {len(wl)} 通道\n")
    print(f"{'K':>4}{'joint':>11}{'project':>11}{'proj(窗口)':>13}{'競爭':>10}")
    print("-" * 46)
    out = []
    spec_keep = {}
    for K in args.klist:
        sky_K = basis[:K]
        d_joint = np.vstack([T.T, sky_K])
        lb = np.full(d_joint.shape[0], -np.inf)
        if n_comp == 1:
            lb[0] = 0.0
        fj, fp, fw, sj = [], [], [], []
        for j in range(ys.size):
            y = D[:, j] - sf[j] * C_sky            # s 固定成場的值,和正式流程一致
            # A:一起解 —— T 和 basis 互相競爭
            th = wls(d_joint, y, V[:, j], lb)
            if th is None:
                continue
            aj = th[:n_comp]
            fj.append(float(np.sum((T @ aj)[bnd])) / ref)
            sj.append(T @ aj)
            # B:先只用 sky 解,殘差再投影到 T —— 沒有競爭
            c = wls(sky_K, y, V[:, j])
            r = y - sky_K.T @ c
            ap_ = wls(T.T, r, V[:, j])
            fp.append(float(np.sum((T @ ap_)[bnd])) / ref)
            # C:同樣沒有競爭,但**只用乾淨窗口的通道**決定振幅
            vv = np.where(fitw, V[:, j], np.nan)
            aw = wls(T.T, r, vv)
            if aw is not None:
                fw.append(float(np.sum((T @ aw)[bnd])) / ref)
        fj, fp, fw = np.array(fj), np.array(fp), np.array(fw)
        mj, mp = float(np.median(fj)), float(np.median(fp))
        mw = float(np.median(fw)) if fw.size else float("nan")
        print(f"{K:>4}{mj:>11.4f}{mp:>11.4f}{mw:>13.4f}{mj - mp:>10.4f}")
        out.append(dict(K=K, joint=mj, project=mp, proj_window=mw, n=int(fj.size),
                        joint_sct=float(robust_spread(fj)), proj_sct=float(robust_spread(fp))))
        spec_keep[K] = np.median(np.array(sj), axis=0)

    print(f"\n讀法:joint 是正式流程的做法。project 沒有競爭。")
    print(f"      兩者接近 -> 偏差在殘差裡;joint 明顯更負 -> 是 basis 搶走的。")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    k = [o["K"] for o in out]
    ax[0].plot(k, [o["joint"] for o in out], "o-", label="joint  [T, sky] together")
    ax[0].plot(k, [o["project"] for o in out], "s--",
               label="project  sky first, then project onto T")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].axhline(-0.049, color="#d62728", ls=":", lw=1.2,
                  label="-0.049 (measured in the hole test)")
    ax[0].set_xlabel("K  (number of sky-line components)")
    ax[0].set_ylabel("recovered source flux / (source = sky)")
    ax[0].grid(alpha=.25); ax[0].legend(fontsize=8)
    for K in args.klist:
        ax[1].plot(wl, spec_keep[K], lw=0.6, label=f"K={K}")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("wavelength [$\\AA$]")
    ax[1].set_ylabel("median recovered source spectrum  T·a")
    ax[1].grid(alpha=.25); ax[1].legend(fontsize=7, ncol=2)
    fig.suptitle(f"where does the additive source-flux bias come from?   "
                 f"{args.work}, {ys.size} blank spaxels (no source injected)",
                 fontsize=12)
    fig.tight_layout()
    o = EVAL / f"s_flux_bias_{W.name}.png"
    o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(o, dpi=140, bbox_inches="tight")
    o.with_suffix(".json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {o}")


if __name__ == "__main__":
    main()
