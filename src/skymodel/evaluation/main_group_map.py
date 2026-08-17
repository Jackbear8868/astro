"""主源怎麼從被拆散的 seg ID 拼回來 —— 處理前後對照。

SExtractor 的 deblender 會把 Haro 11 拆成好幾塊(它是並合星系,本來就有數個亮結),
而拆成幾塊、怎麼拆,逐次觀測不同。任何「選一個 seg ID」的規則都只會拿到其中一塊,
下游用它決定「排除周圍多少 px」與「遮掉哪半邊」時就會遮錯位置。

utils.main_source_group 的做法:取最亮像素所在的連通塊(不做膨脹),再要求成員的
星系分支紅移和主源相差在 dv_max 以內 —— 相鄰只說明它們是同一次 deblend 的兄弟,
疊在星系上的另一個天體同樣會相鄰,要靠紅移把它分出去。

    左  處理前:相鄰的整團裡每個 seg ID 各給一個顏色,標上編號
    右  處理後:通過紅移判準、留下來的那些

    conda run -n astro python src/skymodel/evaluation/main_group_map.py -n 12 5 1
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, arcsinh_stretch, load_field, pointing_dir  # noqa: E402
from utils import DV_MAX, main_source_group  # noqa: E402



def main():
    ap = argparse.ArgumentParser(description="主源合併的處理前後對照")
    ap.add_argument("-n", type=int, nargs="+", default=[12, 5, 1])
    ap.add_argument("--dv-max", type=float, default=DV_MAX,
                    help="成員的紅移離主源多少 km/s 之內才收")
    ap.add_argument("--out-suffix", default="",
                    help="輸出檔名加後綴,不同設定的圖才不會互相覆蓋")
    args = ap.parse_args()

    for n in args.n:
        W = ROOT / f"results/skymodel/p{n:02d}"
        seg, white, valid = load_field(W)
        wn = np.where(valid, white, np.nan)
        mg, ids, pk = main_source_group(seg, wn, W / "step04", args.dv_max)
        # 不給紅移就只做相鄰判準 —— 左圖要畫的正是「還沒篩」的那一團
        all_ids = main_source_group(seg, wn)[1]

        # 畫整個視場 —— 裁切會看不到被剔除的成員落在哪裡,
        # 而「哪些東西沒被收進來」正是這張圖要回答的
        y0, x0 = 0, 0
        sub = np.s_[:, :]

        stretched, vmax = arcsinh_stretch(white, valid)
        bg = stretched[sub]

        fig, ax = plt.subplots(1, 2, figsize=(15, 7.2))
        cmap = plt.cm.tab20(np.linspace(0, 1, 20))
        for a in ax:
            a.imshow(bg, origin="lower", cmap="gray", vmin=0, vmax=vmax)
            a.set_xticks([]); a.set_yticks([])

        # 左:相鄰整團裡每個 seg ID 一個顏色(篩選前)
        for k, i in enumerate(all_ids):
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
        ax[0].set_title(f"before ({len(all_ids)} sources in total)", fontsize=12)

        # 右:合併之後
        m = mg[sub]
        rgba = np.zeros(m.shape + (4,))
        rgba[m] = [1.0, 0.5, 0.05, 0.5]
        ax[1].imshow(rgba, origin="lower")
        ax[1].contour(m, levels=[0.5], colors="#ff7f0e", linewidths=1.6)
        # 相鄰判準用的連通塊 —— 紅移篩掉的成員也在這一塊裡。和合併後的輪廓
        # 同色,用虛線分:兩者是同一件事的前後,換顏色會讓它看起來像另一種東西。
        lab, _ = ndimage.label(seg > 0)
        ax[1].contour((lab == lab[pk])[sub], levels=[0.5],
                      colors="#ff7f0e", linewidths=0.9, linestyles="--")
        ax[1].plot(pk[1] - x0, pk[0] - y0, "w+", ms=14, mew=2)
        ax[1].set_title(f"after ({len(ids)} sources in total)", fontsize=12)

        f = {int(i): float(np.nansum(np.where(seg == i, white, 0)))
             for i in np.unique(seg) if i > 0}
        tot = sum(f.values()); mf = sum(f[i] for i in ids)
        rest = sorted((v_ for k_, v_ in f.items() if k_ not in ids), reverse=True)
        fig.suptitle(f"p{n:02d}", fontsize=14)
        fig.tight_layout()
        out = pointing_dir(f"p{n:02d}") / f"main_group{args.out_suffix}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"p{n:02d}: 相鄰 {len(all_ids)} 個 -> 紅移留下 {len(ids)} 個 {ids}"
              f" -> {int(mg.sum()):,} px   佔源流量 {100*mf/tot:.1f}%"
              f"   剔除 {sorted(set(all_ids) - set(ids))}   saved -> {out}")


if __name__ == "__main__":
    main()
