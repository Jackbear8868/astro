"""逐源畫出「換掉恆星模板庫」前後的擬合,用光譜圖直接核對分類。

step4 已經把每個源的完整掃描寫成 scanN_id{ID}_{tag}.npz,所以這裡不重跑擬合,
只把兩份 tag 的冠軍模型讀出來畫。畫的是 step02 的光譜,也就是 **ESO nosky**
cube 上該源所有 spaxel 的平均(step2 讀的是 nosky),模型是 A x T —— step4 把 s
固定在 0,nosky 上沒有天空連續譜要扣。

一張圖三條線:
    observed        ESO nosky 的源光譜
    star (SDSS)     spDR2 那 23 條裡的冠軍
    star (dwarf)    data/stellar_templates 底下的冠軍
外加星系分支的冠軍(兩份 tag 相同,因為只有恆星庫換掉),因為分類就是這兩支的
reduced chi2 比大小決定的,只看恆星那一支看不出為什麼會翻。

    conda run -n astro python src/skymodel/experiments/star_library_spectra.py \
        --work results/skymodel/p01
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from templates import (air_to_vacuum, load_ascii_template,  # noqa: E402
                       load_eigen_galaxy, load_sdss_template, redshift_to_grid)
from utils import load_line_masks  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
SDSS_DIR  = ROOT / "data/sdss_templates"
DWARF_DIR = ROOT / "data/stellar_templates"
EIGEN_GAL = ROOT / "data/eigen_galaxy_Bolton2012.fits"


def spline_for(name, dwarf):
    """模板名字 -> 樣條。名字是 step4 存進 scan 檔的那個。"""
    if name == "eigen":
        return load_eigen_galaxy(EIGEN_GAL)
    if dwarf:
        return load_ascii_template(DWARF_DIR / f"{name}.dat")
    return load_sdss_template(SDSS_DIR / f"spDR2-{name}.fit")


def best_of(path, dwarf):
    """一份 scan 檔的冠軍 -> (名稱, z, 係數, reduced chi2)。掃描結果已依 chi2 排序。"""
    if not path.exists():
        return None
    d = np.load(path)
    if len(d["red_chi2"]) == 0:
        return None
    i = int(np.argmin(d["red_chi2"]))
    A = d["A"][i]
    return str(d["template"][i]), float(d["z"][i]), A[np.isfinite(A)], float(d["red_chi2"][i])


def model_on_grid(entry, dwarf, lam_vac):
    name, z, A, _ = entry
    T = redshift_to_grid(spline_for(name, dwarf), z, lam_vac)
    if T.ndim == 1:
        T = T[:, None]
    return np.nan_to_num(T[:, :len(A)], nan=np.nan) @ A


def main():
    ap = argparse.ArgumentParser(
        description="逐源比較兩個恆星模板庫的擬合(讀 step4 存下的掃描結果)")
    ap.add_argument("--work", required=True)
    ap.add_argument("--tag-a", default="nobasis_s0.0_4600-8000_4600-8000_L1cum",
                    help="基準那一份的 tag(SDSS 恆星庫)")
    ap.add_argument("--tag-b", default=None,
                    help="要比較的 tag;預設是 --tag-a 加上 _dwarfstar")
    ap.add_argument("--ids", type=int, nargs="+", default=None,
                    help="只畫這些 ID;預設自動挑「有改變」的源")
    ap.add_argument("--ratio", type=float, default=1.10,
                    help="自動挑選時,恆星分支 reduced chi2 變動超過這個倍數就算有改變")
    ap.add_argument("--window", type=float, nargs=2, default=[4600.0, 8000.0],
                    metavar=("LO", "HI"),
                    help="step4 的擬合視窗;殘差圖用它標出哪些通道真的進了 chi2")
    ap.add_argument("--line-mask-iter", type=int, default=1,
                    help="step4 用的那一輪天光線遮罩")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    W = ROOT / args.work
    tag_a = args.tag_a
    tag_b = args.tag_b or f"{tag_a}_dwarfstar"
    S4 = W / "step04"
    out = Path(args.out) if args.out else \
        ROOT / f"results/skymodel/evaluation/{W.name}/star_library"
    out.mkdir(parents=True, exist_ok=True)

    a = np.load(S4 / f"best_{tag_a}.npz")
    b = np.load(S4 / f"best_{tag_b}.npz")
    ga, gb = a["group"].astype(str), b["group"].astype(str)
    ratio = b["star_red_chi2"] / a["star_red_chi2"]

    if args.ids:
        ids = list(args.ids)
    else:
        # 「有改變」= 分類翻了,或恆星分支的 chi2 變動超過門檻
        changed = (ga != gb) | (np.abs(np.log(ratio)) > np.log(args.ratio))
        ids = a["id"][changed].tolist()
    print(f"{len(ids)} sources: {ids}")

    ids_all = np.load(W / "step02/object_ids.npy")
    flux    = np.load(W / "step02/object_flux.npy")
    nspax   = np.load(W / "step02/object_nspax.npy")
    wl_air  = np.load(W / "step03/wavelength.npy")
    lam_vac = air_to_vacuum(wl_air)
    # 進 chi2 的通道 = 視窗內 且 沒被天光線遮罩蓋到。殘差圖若不分開畫,天光線
    # 位置的尖峰看起來像模型擬得差,但那些通道根本沒有參與擬合。
    line = load_line_masks(W / "step03/iter_line_mask.npy",
                           cumulative=True)[args.line_mask_iter - 1]
    used = (wl_air >= args.window[0]) & (wl_air < args.window[1]) & ~line

    for i in ids:
        k = int(np.flatnonzero(ids_all == i)[0])
        j = int(np.flatnonzero(a["id"] == i)[0])
        with np.errstate(invalid="ignore", divide="ignore"):
            obs = flux[k] / nspax[k]

        star_a = best_of(S4 / f"scan1_id{i}_{tag_a}.npz", dwarf=False)
        star_b = best_of(S4 / f"scan1_id{i}_{tag_b}.npz", dwarf=True)
        gal    = best_of(S4 / f"scan2_id{i}_{tag_a}.npz", dwarf=False)

        panels = [(star_a, False, "#ff7f0e", "star  SDSS"),
                  (star_b, True,  "#1f77b4", "star  dwarf"),
                  (gal,    False, "#2ca02c", "galaxy  eigen")]
        panels = [q for q in panels if q[0] is not None]

        # 一列一個模型,每一列都是「ESO nosky vs 這一個模型」,底下接它自己的
        # 殘差。所有列共用同一組座標範圍,否則高低是刻度造成的還是模型造成的
        # 分不出來。
        fig = plt.figure(figsize=(15.5, 3.5 * len(panels) + 0.6))
        gs = fig.add_gridspec(2 * len(panels), 1,
                              height_ratios=[2.6, 1.0] * len(panels),
                              hspace=0.0, top=0.93, bottom=0.06)
        curves = [model_on_grid(e, d, lam_vac) for e, d, _, _ in panels]
        shown = (wl_air >= 4600)
        allv = np.concatenate([obs[shown]] + [c[shown] for c in curves])
        allv = allv[np.isfinite(allv)]
        ylim = np.percentile(allv, [0.2, 99.8])
        pad = 0.12 * (ylim[1] - ylim[0])
        res = [obs - c for c in curves]
        rv = np.concatenate([r[used][np.isfinite(r[used])] for r in res])
        rlim = np.percentile(np.abs(rv), 99.0)

        axes = []
        for row, ((entry, dwarf, colour, lbl), mod, r) in enumerate(
                zip(panels, curves, res)):
            top = fig.add_subplot(gs[2 * row, 0])
            bot = fig.add_subplot(gs[2 * row + 1, 0], sharex=top)
            axes += [top, bot]
            top.plot(wl_air, obs, lw=0.6, color="0.45", label="ESO nosky")
            top.plot(wl_air, mod, lw=1.0, color=colour,
                     label=f"{lbl}   {entry[0]}   z={entry[1]:+.4f}   "
                           f"X2/dof={entry[3]:.2f}")
            top.set_ylim(ylim[0] - pad, ylim[1] + pad)
            top.legend(fontsize=10, loc="upper right", ncol=2)
            top.set_ylabel("flux")
            top.tick_params(labelbottom=False)

            bot.axhline(0, color="0.6", lw=0.7)
            bot.plot(wl_air, np.where(used, np.nan, r), lw=0.4, color="0.78")
            bot.plot(wl_air, np.where(used, r, np.nan), lw=0.5, color=colour,
                     label=f"in chi2: {int(used.sum()):,} channels")
            bot.set_ylim(-rlim, rlim)
            bot.set_ylabel("obs - model", fontsize=9)
            if row == 0:
                bot.legend(fontsize=8, loc="upper right",
                           title="grey = masked or outside the window",
                           title_fontsize=8)
            if row < len(panels) - 1:
                bot.tick_params(labelbottom=False)

        for axis in axes:
            axis.set_xlim(4600, wl_air[-1])
            # 擬合視窗只到 8000 A;右邊是模板外推出去的,不是擬合結果
            axis.axvspan(8000, wl_air[-1], color="0.85", alpha=0.4, zorder=0)
        axes[-1].set_xlabel("wavelength [A, air]")
        axes[0].text(8060, ylim[0] - pad, "outside the fitting window",
                     fontsize=9, color="0.35", va="bottom")
        fig.suptitle(f"{W.name}  ID {i}   {nspax[k].max():.0f} spaxels   "
                     f"{ga[j]} (SDSS)  ->  {gb[j]} (dwarf)"
                     f"   star X2 ratio {ratio[j]:.2f}", fontsize=13)
        o = out / f"id{i:03d}.png"
        fig.savefig(o, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {o}")


if __name__ == "__main__":
    main()
