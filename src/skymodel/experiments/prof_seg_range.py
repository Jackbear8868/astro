"""教授的遮罩之外,還剩多少 Haro 11 的光?—— 決定「學天光的範圍」。

學天光的範圍必須排除星系:遮罩外面若還有星系的延展暈,不多排除一圈,它就會
被當成天空學進 basis 裡。教授的遮罩比我們自己的大得多,所以要重問一次:
**遮罩外面還有星系嗎?**有的話還要往外推多遠?這支就量這件事,量完才談範圍
怎麼設。

怎麼量
------
沿「離主源遮罩的距離 d」把 blank 切成同心環,每環取 pseudo-r 的中位數:

    profile(d) = median{ pseudo_r(p) : p 是 blank, 離主源 d 到 d+bin }

遠處的環是純天空(對 #1/#2/#14 是天空+ESO 殘留),取 d > far 的中位當基線。
星系的暈會讓近處的環高出基線;**基線之上的超出量掉進雜訊裡的那個 d,就是
遮罩外面還受污染的範圍**。

判定的門檻不能用每環中位數的統計誤差 sigma/sqrt(N) —— 每環的格數很多,那個誤差
極小,等於在問「profile 是不是數學上完全平坦」,而背景本來就不平坦(ESO 扣完
天空留下的底隨位置起伏)。門檻改用**遠場自己的環對環起伏**,見 noise_floor_sys。
這量的是「量得出來的污染」,不是「物理上為零」。

不過「乾淨起點」由各顆自己的背景起伏決定,而那個起伏各顆並不相同,所以它不能
直接當成一條通用規則。要選一個能套用到全部 pointing 的排除半徑,看的是最後那張
**取捨表**:同一個 R 之下,各顆各自還剩多少污染、還剩多少 blank。

小源的翼會混進來,所以另外要求離**任何**源 > r_other(預設 5 px)。

    conda run -n astro python src/skymodel/experiments/prof_seg_range.py
    conda run -n astro python src/skymodel/experiments/prof_seg_range.py -n 4 8 12
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import main_source_group, scale  # noqa: E402

ROOT    = Path(__file__).resolve().parents[3]
SEGDIR  = ROOT / "data/wsky_seg"
FIGURES = ROOT / "results/skymodel/evaluation/masking/prof_seg"


def profile(n, bin_px, r_other, margin, far):
    seg = fits.getdata(SEGDIR / f"DATACUBE_FINAL_{n}_seg.fits").astype(int)
    img = np.asarray(fits.getdata(SEGDIR / f"DATACUBE_FINAL_{n}_pseudo_r.fits"), float)
    valid = np.isfinite(img) & (img != 0)

    main, ids, _ = main_source_group(seg, np.where(valid, img, np.nan))
    d_main  = ndimage.distance_transform_edt(~main)
    others  = (seg > 0) & ~main
    d_other = (ndimage.distance_transform_edt(~others) if others.any()
               else np.full(seg.shape, np.inf))
    d_edge  = ndimage.distance_transform_edt(valid)

    blank = valid & (seg == 0) & (d_other > r_other) & (d_edge > margin)
    d, v  = d_main[blank], img[blank]

    base_m = d > far
    if base_m.sum() < 200:                      # 視野不夠大就用最遠的四分之一
        base_m = d > np.percentile(d, 75)
    base = float(np.median(v[base_m]))

    edges = np.arange(0, d.max() + bin_px, bin_px)
    rows  = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi)
        if m.sum() < 30:
            continue
        rows.append((0.5 * (lo + hi), int(m.sum()), float(np.median(v[m]) - base),
                     float(scale(v[m]) / np.sqrt(m.sum()))))
    return np.array(rows), base, blank, main, seg, valid


def noise_floor_sys(rows, far):
    """判定用的雜訊底 —— **不能**只用每環中位數的統計誤差 sigma/sqrt(N)。

    每環的格數很多,統計誤差極小。用它當門檻等於在問「profile 是不是數學上完全
    平坦」,而背景本來就不平坦:ESO 扣完天空之後留下的底本身就隨位置起伏。拿
    統計誤差去比,答案永遠是「到視野邊緣都還不乾淨」,那不是星系的訊息,
    是門檻定錯了。

    正確的底是**遠場自己的環對環起伏**:那裡沒有星系,剩下的散布就是背景系統性
    起伏的大小。近處的超出量小於它,就代表「即使有殘留,也已經埋在背景本身的
    不平坦裡,量不出來了」。
    """
    m = rows[:, 0] > far
    if m.sum() < 3:
        m = rows[:, 0] > np.percentile(rows[:, 0], 60)
    return max(float(scale(rows[m, 2])), float(np.median(rows[:, 3])))


def first_clean(rows, floor, nsigma):
    """從最遠往回走,找出「從這裡開始一路到底都乾淨」的最小距離。

    從近往遠找第一個乾淨的環是錯的 —— profile 有起伏,可能剛好穿過一次基線
    又升回去。要的是「之後全部都乾淨」,所以必須從遠端往回掃。
    """
    ok = rows[:, 2] < nsigma * floor
    r  = np.inf
    for i in range(len(rows) - 1, -1, -1):
        if not ok[i]:
            break
        r = rows[i, 0]
    return r


def main():
    ap = argparse.ArgumentParser(description="教授遮罩外的殘留星系光 vs 距離")
    ap.add_argument("-n", type=int, nargs="+", default=list(range(1, 15)))
    ap.add_argument("--bin", type=float, default=5.0, help="環寬(px)")
    ap.add_argument("--r-other", type=float, default=5.0, help="離其他源至少多遠")
    ap.add_argument("--margin", type=float, default=10.0, help="離視野邊緣至少多遠")
    ap.add_argument("--far", type=float, default=150.0, help="定基線用的遠場距離")
    ap.add_argument("--nsigma", type=float, default=3.0)
    ap.add_argument("--radii", type=float, nargs="+",
                    default=[0, 10, 20, 30, 40, 60],
                    help="取捨表要試哪幾個排除半徑(px)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    ncol = 5
    nrow = int(np.ceil(len(args.n) / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow),
                           squeeze=False)
    for a in np.ravel(ax):
        a.axis("off")

    trade = []
    print(f"{'':>4}{'基線':>9}{'背景起伏':>10}{'d=0-5':>9}{'5-10':>8}{'10-20':>8}"
          f"{'20-40':>8}{'乾淨起點':>10}{'blank 佔視野':>13}")
    for k, n in enumerate(args.n):
        rows, base, blank, main, seg, valid = profile(
            n, args.bin, args.r_other, args.margin, args.far)
        floor = noise_floor_sys(rows, args.far)
        r0 = first_clean(rows, floor, args.nsigma)

        def at(lo, hi):
            m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
            return np.average(rows[m, 2], weights=rows[m, 1]) if m.any() else np.nan

        keep = blank & (ndimage.distance_transform_edt(~main) > (0 if not
                        np.isfinite(r0) else r0))
        print(f"#{n:<3}{base:>9.3f}{floor:>10.3f}{at(0, 5):>9.3f}{at(5, 10):>8.3f}"
              f"{at(10, 20):>8.3f}{at(20, 40):>8.3f}"
              + (f"{r0:>9.0f}px" if np.isfinite(r0) else f"{'—':>10}")
              + f"{100 * keep.sum() / valid.sum():>12.1f}%")

        a = np.ravel(ax)[k]
        a.axis("on")
        a.axhline(0, color="0.6", lw=0.7)
        a.axhspan(-args.nsigma * floor, args.nsigma * floor, color="0.85", zorder=0)
        a.errorbar(rows[:, 0], rows[:, 2], yerr=rows[:, 3],
                   fmt="o-", ms=2.5, lw=0.8, elinewidth=0.6, color="#1f77b4")
        if np.isfinite(r0):
            a.axvline(r0, color="#d62728", lw=1.0, ls="--")
            a.text(r0, a.get_ylim()[1], f" {r0:.0f}px", color="#d62728",
                   fontsize=8, va="top")
        a.set_xscale("symlog", linthresh=20)
        a.set_title(f"#{n}   baseline {base:.2f}", fontsize=9)
        a.grid(alpha=0.25)
        a.tick_params(labelsize=7)
        if k % ncol == 0:
            a.set_ylabel("median − baseline", fontsize=8)
        a.set_xlabel("distance from main mask [px]", fontsize=8)
        # 給選範圍用的取捨表:排除半徑 R 換到的「殘留污染」與「還剩多少 blank」。
        # 只報一個「乾淨起點」是不夠的 —— 它由每顆自己的背景起伏決定,而背景
        # 起伏各顆並不相同。真正要看的是同一個 R 之下各顆的污染量是否都夠小,
        # 那才是能套用到全部 pointing 的規則。
        d_main_all = ndimage.distance_transform_edt(~main)
        cells = []
        for R in args.radii:
            m = rows[:, 0] > R
            ex = (np.average(rows[m, 2], weights=rows[m, 1]) if m.any() else np.nan)
            frac = 100 * (blank & (d_main_all > R)).sum() / valid.sum()
            cells.append((ex, frac))
        trade.append((n, floor, cells))

    print(f"\n取捨表:排除主源 R px 之後,剩下的 blank 還有多少殘留星系光")
    print(f"{'':>4}{'背景起伏':>9}"
          + "".join(f"{'R=' + str(int(R)):>16}" for R in args.radii))
    print(f"{'':>13}" + "".join(f"{'污染':>9}{'blank':>7}" for _ in args.radii))
    for n, floor, cells in trade:
        print(f"#{n:<3}{floor:>9.3f}"
              + "".join(f"{ex:>9.3f}{fr:>6.0f}%" for ex, fr in cells))

    fig.suptitle("residual galaxy light OUTSIDE the professor's mask "
                 f"(error bars = {args.nsigma:g} sigma)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.out) if args.out else FIGURES / "range_profile.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
