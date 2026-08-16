"""每一條模板的 chi2 如何隨紅移變化 —— 紅移是怎麼被決定的(或沒被決定)。

step4b_template_chi2.py 畫的是「每條模板最好能配到多好」,一條模板一個點,
掃描過程被壓成一個最小值。這支把那個過程攤開:一條模板一條曲線,橫軸是 z。

要看的是曲線的形狀,不是最低點的位置:

    深而窄的單一谷      紅移被資料決定了
    兩個差不多深的谷    紅移在兩個解之間擲骰子
    幾乎平的            資料裡沒有紅移的資訊

灰色虛線是教授給的 z,紅色星號是我們的解。兩者不重合時,看教授那條線的位置
上有沒有一個次要的谷 —— 有的話是「兩個解競爭」,沒有的話才是「真的不合」。

水平藍虛線是第 1 段最佳恆星的 reduced chi2。曲線落在它下面的區間,就是
「星系模型贏過恆星模型」的紅移範圍。

    conda run -n astro python src/skymodel/experiments/step4b_chi2_vs_z.py \\
        --s-fix 0 --star-window 4700 8000 --gal-window 4700 8000 --iter 1 \\
        --runs __eso __ours __eso_galtpl __ours_galtpl \\
        --labels A_ESOsky_eigen B_oursky_eigen C_ESOsky_sdsstpl D_oursky_sdsstpl
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step4_fit_source import STAR_WINDOW, GAL_WINDOW, make_tag

ROOT    = Path(__file__).resolve().parents[3]
STEP04 = ROOT / "results/skymodel/ne_pointing/step04"
FIGURES = ROOT / "results/skymodel/evaluation/template_fit"

GAL_TYPE = {23: "early", 24: "gal24", 25: "gal25", 26: "gal26",
            27: "late", 28: "LRG"}
# 一條模板一個顏色。本徵譜只有一條,SDSS 版有六條 —— 六條就必須靠顏色分開。
CURVE = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b",
         "#17becf"]
PROF_Z = {1: 0.021, 10: 0.99, 12: 1.033, 13: 0.422, 14: 0.336, 24: 1.055,
          27: 1.349, 28: 1.033, 30: 0.295, 34: 0.605, 35: None}


def main():
    ap = argparse.ArgumentParser(description="每條模板的 chi2 隨紅移的變化")
    ap.add_argument("--star-window", type=float, nargs=2, default=STAR_WINDOW)
    ap.add_argument("--gal-window", type=float, nargs=2, default=GAL_WINDOW)
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--sky-basis", action="store_true")
    ap.add_argument("--aperture", action="store_true")
    ap.add_argument("--raw-mask", action="store_true")
    ap.add_argument("--iter", type=int, default=1)
    ap.add_argument("--s-fix", type=float, default=0.0)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--titles", nargs="+", default=None,
                    help="每個 run 印在圖上的標題,預設同 --labels")
    ap.add_argument("--ids", type=int, nargs="+", default=None,
                    help="要畫哪些源。省略 = best 檔裡的全部")
    ap.add_argument("--yscale", choices=["clip", "zero", "log"], default="zero",
                    help="縱軸範圍。clip = 最小值到它的 2 倍(谷最清楚);"
                         "zero = 從 0 起算;log = 對數軸涵蓋完整範圍")
    ap.add_argument("--zoom", type=float, default=None,
                    help="只畫最佳解附近 +-ZOOM 的範圍,預設畫整個掃描範圍")
    args = ap.parse_args()
    labels = args.labels or [s.lstrip("_") for s in args.runs]
    titles = args.titles or labels

    tags = [make_tag(args.basis, args.K, args.s_fix, args.star_window,
                     args.gal_window, args.sky_basis, args.iter,
                     not args.raw_mask, args.aperture, s) for s in args.runs]

    # --ids 省略時取第一個 tag 的 best 檔裡的全部源 —— 每顆 pointing 的
    # SExtractor 編號都不同,寫死一串編號只在某一顆上成立
    if args.ids is None:
        bf = STEP04 / f"best_{tags[0]}.npz"
        if not bf.exists():
            raise SystemExit(f"找不到 {bf.name},先跑 step4_fit_source.py")
        args.ids = [int(i) for i in np.load(bf)["id"]]

    ncol = 3
    nrow = int(np.ceil(len(args.ids) / ncol))

    for tag, lab, ttl in zip(tags, labels, titles):
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.1 * nrow),
                                 squeeze=False)
        for k, t in enumerate(args.ids):
            ax = axes[k // ncol][k % ncol]
            f2 = STEP04 / f"scan2_id{t}_{tag}.npz"
            f1 = STEP04 / f"scan1_id{t}_{tag}.npz"
            if not f2.exists():
                ax.axis("off")
                continue
            sc = np.load(f2)

            names = sorted(set(sc["template"]))
            for i, nm in enumerate(names):
                m = sc["template"] == nm
                o = np.argsort(sc["z"][m])
                lbl = ("eigen" if nm == "eigen"
                       else GAL_TYPE.get(int(nm), str(nm)))
                ax.plot(sc["z"][m][o], sc["red_chi2"][m][o], lw=0.9,
                        color=CURVE[i % len(CURVE)], label=lbl)

            j = int(np.argmin(sc["chi2"]))
            zbest = float(sc["z"][j])
            ax.plot(zbest, float(sc["red_chi2"][j]), "*", ms=13, color="k",
                    zorder=5)

            # 第 1 段最佳恆星的水平線:曲線在它下面的地方,星系才贏過恆星。
            s1 = None
            if f1.exists():
                s1 = float(np.load(f1)["red_chi2"].min())
                ax.axhline(s1, color="#1f77b4", lw=1.0, ls="--", alpha=0.8)

            pz = PROF_Z.get(t)
            if pz is not None:
                ax.axvline(pz, color="0.4", lw=1.0, ls=":")

            ax.set_xlim(sc["z"].min(), sc["z"].max())
            if args.zoom:
                ax.set_xlim(max(zbest - args.zoom, sc["z"].min()),
                            min(zbest + args.zoom, sc["z"].max()))
            # y 的範圍只由畫得出來的那一段決定。下限一定要把恆星那條線含進去:
            # 恆星贏的源,恆星的值比整條星系曲線都低,只用星系曲線定範圍
            # 會把那條線切到框外 —— 而那正好是唯一需要看到它的一格。
            vis = ((sc["z"] >= ax.get_xlim()[0]) & (sc["z"] <= ax.get_xlim()[1]))
            v   = sc["red_chi2"][vis]
            lo  = min(v.min(), s1) if s1 is not None else v.min()

            if args.yscale == "clip":
                # 截在最小值的 2 倍。谷看得最清楚,代價是曲線主體遠在上限之上的
                # 源,整格幾乎是空的。
                ax.set_ylim(lo * 0.97, max(v.min(), lo) * 2.0)
            elif args.yscale == "zero":
                # 從 0 起算,上限是最小值的 2 倍:谷底落在面板的正中間,深度
                # 相對於「零」直接讀得出來。代價是曲線主體高過 2 倍最小值的源,
                # 主體會被切在框外,那一格只看得到谷。
                ax.set_ylim(0, max(v.min(), lo) * 2.0)
            else:                                   # log
                # 對數軸涵蓋完整範圍:深谷和曲線主體同時看得到,代價是
                # 「深多少」要用比例去讀,不是用高度差。
                ax.set_yscale("log")
                ax.set_ylim(lo * 0.95, float(v.max()) * 1.1)
                # log 軸跨度小於一個數量級時,matplotlib 會改用「次要刻度」來標,
                # 而次要刻度有自己的格式器 —— 只換主要刻度的話,標籤還是會出現
                # 3×10⁰ 那種寫法。兩個都要換。
                ax.yaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(
                    lambda x, _: f"{x:,.0f}" if x >= 1000 else f"{x:g}"))

            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda x, _: f"{x:,.0f}" if x >= 1000 else f"{x:g}"))
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3)
            ax.set_xlabel("redshift  z", fontsize=7)
            ax.set_ylabel("reduced chi2", fontsize=7)
            # 判定寫進標題 —— 這張圖上「誰贏」是靠兩條線的相對高低表達的,
            # 而那在恆星贏的情形最容易被讀反,直接寫出來最保險。
            win = "STAR" if (s1 is not None and s1 < v.min()) else "GALAXY"
            ax.set_title(f"ID {t}   " + (f"gt z = {pz}" if pz
                                         else "gt: star")
                         + f"   our z = {zbest:.4f}   ->  {win}", fontsize=9)
            if k == 0:
                ax.legend(fontsize=6.5, ncol=2, loc="upper right")

        for k in range(len(args.ids), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")

        fig.suptitle(f"{ttl}   "
                     f"[{args.star_window[0]:.0f}-{args.star_window[1]:.0f} "
                     f"$\\AA$]\n"
                     "black star = our solution,  dotted vertical = gt z,  "
                     "blue dashed = best stellar template", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.955))
        stem = lab.replace(":", "").replace(" ", "_")
        out = FIGURES / f"step4b_chi2_vs_z_{stem}.png"
        fig.savefig(out, dpi=135, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
