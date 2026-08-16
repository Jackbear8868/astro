"""主源怎麼從被拆散的 seg ID 拼回來 —— 處理前後對照。

SExtractor 的 deblender 會把 Haro 11 拆成好幾塊(它是並合星系,本來就有數個亮結),
而拆成幾塊、怎麼拆,逐次觀測不同。任何「選一個 seg ID」的規則都只會拿到其中一塊,
下游用它決定「排除周圍多少 px」與「遮掉哪半邊」時就會遮錯位置。

utils.main_source_group 的做法:把 seg > 0 膨脹 bridge 像素讓被拆開的結重新相連,
取最亮像素所在的整個連通塊,回傳該塊涵蓋的所有 seg ID。

    左  處理前:主源那一團裡的每個 seg ID 各給一個顏色,標上編號
    右  處理後:合併成一塊

    conda run -n astro python src/skymodel/experiments/main_group_map.py -n 12 5 1
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
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import main_source_group  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FIG  = ROOT / "results/skymodel/figures/main_group"


def main():
    ap = argparse.ArgumentParser(description="主源合併的處理前後對照")
    ap.add_argument("-n", type=int, nargs="+", default=[12, 5, 1])
    ap.add_argument("--bridge", type=int, default=3)
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)
    for n in args.n:
        W = ROOT / f"results/skymodel/p{n:02d}"
        seg = fits.getdata(W / "step01/seg.fits").astype(int)
        white = np.asarray(fits.getdata(W / "step01/whitelight.fits"), float)
        valid = white != 0
        mg, ids, pk = main_source_group(seg, np.where(valid, white, np.nan),
                                        bridge=args.bridge)

        # 只畫主源那一團的附近,不然整個視場裡星系只佔一小塊,看不出拆成幾片
        ys, xs = np.nonzero(mg)
        pad = 30
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, seg.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, seg.shape[1])
        sub = np.s_[y0:y1, x0:x1]

        v = np.nanpercentile(white[valid], 99.5)
        bg = np.arcsinh(np.where(valid, white, np.nan) / (0.02 * v))[sub]
        vmax = np.arcsinh(1 / 0.02)

        fig, ax = plt.subplots(1, 2, figsize=(15, 7.2))
        cmap = plt.cm.tab20(np.linspace(0, 1, 20))
        for a in ax:
            a.imshow(bg, origin="lower", cmap="gray", vmin=0, vmax=vmax)
            a.set_xticks([]); a.set_yticks([])

        # 左:每個 seg ID 一個顏色
        for k, i in enumerate(ids):
            m = (seg == i)[sub]
            rgba = np.zeros(m.shape + (4,))
            rgba[m] = list(cmap[k % 20][:3]) + [0.55]
            ax[0].imshow(rgba, origin="lower")
            yy, xx = np.nonzero(m)
            if yy.size > 40:
                ax[0].text(xx.mean(), yy.mean(), str(i), color="w", fontsize=11,
                           fontweight="bold", ha="center", va="center",
                           path_effects=[pe.withStroke(linewidth=2.5,
                                                       foreground="k")])
        # 圖上一律英文 —— matplotlib 預設字型沒有中文字符
        ax[0].set_title(f"before — the deblender split the main source into "
                        f"{len(ids)} seg IDs   ({seg.max()} sources in total)",
                        fontsize=11)

        # 右:合併之後
        m = mg[sub]
        rgba = np.zeros(m.shape + (4,))
        rgba[m] = [1.0, 0.5, 0.05, 0.5]
        ax[1].imshow(rgba, origin="lower")
        ax[1].contour(m, levels=[0.5], colors="#ff7f0e", linewidths=1.6)
        # 膨脹後的連通塊 —— 合併靠的就是這一步跨過縫隙
        br = ndimage.binary_dilation((seg > 0), iterations=args.bridge)[sub]
        lab, _ = ndimage.label(br)
        ax[1].contour(lab == lab[pk[0] - y0, pk[1] - x0], levels=[0.5],
                      colors="#00e5ff", linewidths=0.9, linestyles="--")
        ax[1].plot(pk[1] - x0, pk[0] - y0, "w+", ms=14, mew=2)
        ax[1].set_title(f"after — main_source_group: one blob, {int(mg.sum()):,} px\n"
                        f"cyan dashed = connected blob after {args.bridge} px "
                        f"dilation,  white cross = brightest pixel", fontsize=11)

        f = {int(i): float(np.nansum(np.where(seg == i, white, 0)))
             for i in np.unique(seg) if i > 0}
        tot = sum(f.values()); mf = sum(f[i] for i in ids)
        rest = sorted((v_ for k_, v_ in f.items() if k_ not in ids), reverse=True)
        fig.suptitle(f"p{n:02d}   main source = {len(ids)} seg IDs merged,   "
                     f"{100*mf/tot:.1f}% of all source flux   "
                     f"(runner-up: {100*rest[0]/tot:.1f}%)", fontsize=13)
        fig.tight_layout()
        out = FIG / f"main_group_p{n:02d}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"p{n:02d}: {len(ids)} 個 ID {ids} -> {int(mg.sum()):,} px   "
              f"佔源流量 {100*mf/tot:.1f}%   saved -> {out}")


if __name__ == "__main__":
    main()
