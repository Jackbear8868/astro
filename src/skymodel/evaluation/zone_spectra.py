"""同一個環上,ESO 與我們的兩個設定各扣出什麼 —— 逐環一張光譜。

check_pointing 的四個環各自回答一個問題,這支腳本把那四個數字背後的**光譜**畫
出來,讓「差在哪個波長」看得見。

    far          離所有源都遠。沒有源,所以三條線的真值都是 0 ——
                 誰貼近 0 就是誰的天空扣得準。
    small 1-3    小源外圍 1-3 px。那裡有小源的 PSF 翼,真值是正的。
    main 1-3     Haro 11 外圍 1-3 px。
    main 3-10    Haro 11 外圍 3-10 px,延展的暈。
                 後兩者是我們要保住的東西,線壓得越低代表源被吃掉越多。

環的定義來自 common.zones,和 check_pointing 是同一份 —— 一律取 seg == 0 的格,
偵測不到的地方不代表沒有源的光,而那正是最容易被過度扣除的位置。

每個環畫兩層:細線是逐通道的平均光譜,粗線是移動平均。雜訊被壓成 1/sqrt(N) 之後,
零點的偏移才浮得出來。

    conda run -n astro python src/skymodel/evaluation/zone_spectra.py -n 1 4 8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL, ROOT, load_field, zones  # noqa: E402
from utils import galaxy_redshifts, main_source_group, scale  # noqa: E402

FIG = EVAL / "acceptance"

# 主星系的強發射線,靜止空氣波長。step03/wavelength.npy 存的是空氣波長
# (step4b 讀進去的變數就叫 wl_air),所以這裡也必須用空氣值,不能混真空值。
# 標在圖上是為了看出「源環上多出來的東西是不是真的來自這個星系」。
LINES = [(4861.33, "Hb"), (4958.91, "[O III]"), (5006.84, "[O III]"),
         (5875.6, "He I"), (6300.30, "[O I]"), (6548.05, "[N II]"),
         (6562.82, "Ha"), (6583.45, "[N II]"), (6716.44, "[S II]"),
         (6730.81, "[S II]"), (7135.8, "[Ar III]"), (9068.6, "[S III]")]


def zone_mean(path, mask, hdu=None):
    """一個 cube 在遮罩上的平均光譜。逐波長平面讀,避免整個 cube 進記憶體。"""
    with fits.open(path, memmap=True) as h:
        d = h[hdu] if hdu else (h[0] if h[0].data is not None else h["DATA"])
        nz = d.data.shape[0]
        out = np.empty(nz, np.float64)
        idx = np.nonzero(mask.ravel())[0]
        for k in range(nz):
            out[k] = np.nanmean(d.data[k].ravel()[idx])
    return out


def main():
    ap = argparse.ArgumentParser(description="逐環的光譜:ESO vs 我們的兩個設定")
    ap.add_argument("-n", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--smooth", type=int, default=51,
                    help="粗線的移動平均寬度(通道)")
    ap.add_argument("--ylim", type=float, nargs=2, default=None)
    ap.add_argument("--runs", nargs="+", default=["blank_*"],
                    help="要比較的 step05 run(可用萬用字元)。ESO nosky 一律畫,"
                         "當作共同的 baseline")
    ap.add_argument("--z", type=float, default=None,
                    help="標線用的紅移。省略 = 用 step4b 對最亮像素所在那塊擬合的值")
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    for n in args.n:
        W   = ROOT / f"results/skymodel/p{n:02d}"
        seg, white, valid = load_field(W)
        wl  = np.load(W / "step03/wavelength.npy")
        main, _, pk = main_source_group(seg, np.where(valid, white, np.nan))
        Z   = zones(seg, white, main)

        # 紅移取自這顆自己的擬合,不是寫死的常數 —— 逐顆略有差異,
        # 用別顆的值會讓標線和鼓包對不上而看不出來是對不上
        pid = int(seg[pk])
        z = args.z if args.z is not None else galaxy_redshifts(W / "step04",
                                                               [pid])[pid]

        runs = [("ESO nosky", ROOT / f"data/nosky/DATACUBE_FINAL_ESOSKY_{n}.fits",
                 "#d62728", "-")]
        # 設定接近時兩條線會重疊,必須用線型分開 —— 只換顏色的話後畫的會把先畫的
        # 整條蓋掉,圖上看起來像少了一條線
        palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]
        styles  = ["-", "--", "-.", ":"]
        for k, pat in enumerate(args.runs):
            hit = sorted((W / "step05").glob(pat + "/sky_subtracted.fits"))
            if not hit:
                print(f"  p{n:02d}: 沒有符合 {pat} 的 run,略過")
                continue
            runs.append((hit[0].parent.name, hit[0],
                         palette[k % len(palette)], styles[k % len(styles)]))

        kern = np.ones(args.smooth) / args.smooth
        # 每個 run 在每個環的平均光譜先全部算好 —— 源環的數字要減掉該 run
        # **自己的**遠場零點才可比:ESO 整體高 0.39,不扣掉的話它每個環都虛高
        spec = {nm: {lbl: zone_mean(path, m) for lbl, path, _, _ in runs}
                for nm, m in Z.items()}
        off = {lbl: float(np.nanmean(spec["far"][lbl])) for lbl, *_ in runs}

        fig, axes = plt.subplots(len(Z), 1, figsize=(19, 3.4 * len(Z)),
                                 sharex=True)
        for ax, (nm, m) in zip(np.atleast_1d(axes), Z.items()):
            ax.axhline(0, color="0.5", lw=0.8, zorder=1)
            for rest, nm_l in LINES:
                obs = rest * (1 + z)
                if wl[0] <= obs <= wl[-1]:
                    ax.axvline(obs, color="#9467bd", lw=0.8, ls=":", alpha=0.75,
                               zorder=1.5)
            lo, hi = [], []
            for lbl, path, col, ls in runs:
                y = spec[nm][lbl]
                sm = np.convolve(y, kern, "same")
                ax.plot(wl, y, lw=0.5, color=col, alpha=0.30, ls=ls, zorder=2)
                ax.plot(wl, sm, lw=1.8, color=col, ls=ls, zorder=3,
                        label=f"{lbl}    mean {np.nanmean(y):+.4f}"
                              + ("" if nm == "far" else
                                 f"  (−far = {np.nanmean(y) - off[lbl]:+.4f})")
                              + f"    scatter {scale(y):.4f}")
                lo.append(np.nanpercentile(sm, 1)); hi.append(np.nanpercentile(sm, 99))
            # y 範圍照**平滑後**的曲線定,細線的尖峰不該決定看得到什麼
            a, b = min(lo), max(hi)
            pad = 0.35 * max(b - a, 1e-3)
            ax.set_ylim(min(a - pad, -pad), b + pad)
            # 標籤只寫在最上面那一格,每格都寫會把圖蓋掉
            if nm == "far":
                y0, y1 = ax.get_ylim()
                for rest, nm_l in LINES:
                    obs = rest * (1 + z)
                    if wl[0] <= obs <= wl[-1]:
                        ax.text(obs, y1, f" {nm_l}", color="#6a3d9a", fontsize=8,
                                rotation=90, va="top", ha="left", zorder=5)
            ax.set_ylabel("flux  [$10^{-20}$ cgs]", fontsize=9)
            ax.legend(fontsize=9, loc="upper right", framealpha=0.92, ncol=1)
            ax.set_title(f"{nm}    {int(m.sum()):,} spaxels"
                         + ("    (no source here — truth is 0)"
                            if nm == "far" else
                            "    (source light lives here — lower = more eaten)"),
                         fontsize=10, loc="left")
            if args.ylim:
                ax.set_ylim(*args.ylim)
        np.atleast_1d(axes)[-1].set_xlabel("wavelength [$\\AA$]", fontsize=10)
        fig.suptitle(f"p{n:02d}   zone mean spectra   "
                     f"(thin = per channel, thick = {args.smooth}-channel mean)"
                     f"   —   dotted violet = Haro 11 emission lines at z = {z:.4f}",
                     fontsize=13)
        fig.tight_layout()
        out = FIG / f"zone_spectra_p{n:02d}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
