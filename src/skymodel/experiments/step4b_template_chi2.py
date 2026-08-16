"""每個源在各個模板上的最佳 chi2 —— 分類的判準攤開來看。

step4b 的分類只回報「誰贏」。這支把中間過程畫出來:23 條恆星模板各自的最佳
chi2 排成一列,星系本徵譜的最佳 chi2 畫成一條水平線。三件事一眼可見:

    星系那條線離恆星那一群多遠  = 分類的裕度。差得遠才叫分得出來。
    23 條恆星之間的散布          = 資料分不分得出恆星的型別。平的就是分不出來。
    兩個扣天空版本疊在一起       = 結論對天空模型敏不敏感。

y 用 log:同一個源裡 chi2 可以跨一個數量級,線性軸會把差別壓平;而各源之間
又差三個數量級,所以每格自己的 y 範圍獨立。

    conda run -n astro python src/skymodel/experiments/step4b_template_chi2.py \\
        --s-fix 0 --star-window 4700 8000 --gal-window 4700 8000 --iter 1
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
STEP04 = ROOT / "results/skymodel/step04"
FIGURES = ROOT / "results/skymodel/figures"

STAR_TYPE = {0: "O", 1: "O/B", 2: "B", 3: "A", 4: "A", 5: "F/A", 6: "F", 7: "F",
             8: "G", 9: "G", 10: "K", 11: "M1", 12: "M3", 13: "M5", 14: "M8",
             15: "L1", 16: "magWD", 17: "C", 18: "C", 19: "C", 20: "WD",
             21: "BWD", 22: "Ksd"}
# 教授給的紅移,拿來當標題上的對照
PROF_Z = {1: 0.021, 10: 0.99, 12: 1.033, 13: 0.422, 14: 0.336, 24: 1.055,
          27: 1.349, 28: 1.033, 30: 0.295, 34: 0.605, 35: None}


GAL_TYPE = {23: "early", 24: "gal24", 25: "gal25", 26: "gal26",
            27: "late", 28: "LRG"}


def per_template(tag, t):
    """回傳 (23 條恆星模板的最佳 reduced chi2, 星系候選的 [(名字, 值)])。

    每條模板取它自己在 z 掃描裡的最低點 —— 比的是「這條模板最好能配到多好」,
    不是「用贏家的 z 去配得多差」。星系那邊,本徵譜只有一個候選('eigen'),
    SDSS 星系模板有 6 個,所以回傳一個清單而不是單一數字。
    """
    f1 = STEP04 / f"scan1_id{t}_{tag}.npz"
    if not f1.exists():
        return None, None
    sc = np.load(f1)
    star = np.full(23, np.nan)
    for nm in set(sc["template"]):
        m = sc["template"] == nm
        star[int(nm)] = sc["red_chi2"][m].min()

    f2 = STEP04 / f"scan2_id{t}_{tag}.npz"
    if not f2.exists():
        return star, []
    s2 = np.load(f2)
    gal = []
    for nm in sorted(set(s2["template"])):
        v = float(s2["red_chi2"][s2["template"] == nm].min())
        lab = "eigen" if nm == "eigen" else GAL_TYPE.get(int(nm), str(nm))
        gal.append((lab, v))
    return star, gal


def main():
    ap = argparse.ArgumentParser(description="每個源在各模板上的最佳 chi2")
    ap.add_argument("--star-window", type=float, nargs=2, default=STAR_WINDOW)
    ap.add_argument("--gal-window", type=float, nargs=2, default=GAL_WINDOW)
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--sky-basis", action="store_true")
    ap.add_argument("--aperture", action="store_true")
    ap.add_argument("--raw-mask", action="store_true")
    ap.add_argument("--iter", type=int, default=1)
    ap.add_argument("--s-fix", type=float, default=0.0)
    ap.add_argument("--runs", nargs="+", default=["_eso", "_ours"],
                    help="要疊在一起比的 tag 後綴,例如 _eso _ours")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="每個 run 的檔名字尾,預設用 tag 後綴")
    ap.add_argument("--titles", nargs="+", default=None,
                    help="每個 run 印在圖上的標題,預設同 --labels")
    ap.add_argument("--ids", type=int, nargs="+", default=None,
                    help="要畫哪些源。省略 = best 檔裡的全部")
    ap.add_argument("--out", default=None)
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

    ids  = list(args.ids)
    ncol = 3
    nrow = int(np.ceil(len(ids) / ncol))

    # 一個方法一張圖。疊在一起看得出「換方法差多少」,分開才看得清「這個方法
    # 本身在每個源上發生什麼事」—— 尤其 SDSS 版的星系有 6 條候選,疊起來就糊了。
    for tag, lab, ttl in zip(tags, labels, titles):
        fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 3.1 * nrow),
                                 squeeze=False)
        for k, t in enumerate(ids):
            ax = axes[k // ncol][k % ncol]
            star, gal = per_template(tag, t)
            if star is None:
                ax.axis("off")
                continue

            xs_star = np.arange(23)
            # 星系候選排在恆星右邊,中間空一格 —— 兩者是不同性質的東西,
            # 緊貼著會被讀成同一個序列。
            xs_gal  = 24 + np.arange(len(gal))
            ax.plot(xs_star, star, "o", ms=3.5, color="#1f77b4",
                    label="stellar templates")
            if gal:
                ax.plot(xs_gal, [v for _, v in gal], "s", ms=4.5,
                        color="#d62728", label="galaxy")
                # 兩條水平線標出各自的最佳值,兩線的距離就是分類的裕度。
                ax.axhline(min(v for _, v in gal), color="#d62728", lw=1.0,
                           ls="--", alpha=0.8)
            ax.axhline(np.nanmin(star), color="#1f77b4", lw=1.0, ls="--",
                       alpha=0.8)

            vals = np.concatenate([star[np.isfinite(star)],
                                   np.array([v for _, v in gal])])
            ax.set_yscale("log")
            ax.set_ylim(np.nanmin(vals) * 0.85, np.nanmax(vals) * 1.2)
            ax.set_xticks(list(xs_star) + list(xs_gal))
            ax.set_xticklabels([STAR_TYPE[i] for i in range(23)]
                               + [n for n, _ in gal], fontsize=5.5, rotation=90)
            for lbl, xpos in zip(ax.get_xticklabels(),
                                 list(xs_star) + list(xs_gal)):
                if xpos >= 24:
                    lbl.set_color("#d62728")
            # log 軸但刻度寫成普通數字。預設的 6×10⁰ / 10⁵ 這種寫法要在腦裡換算
            # 一次才知道大小,而這張圖要比的就是大小。只留 1/2/5 三個位置,
            # 標籤才不會在小圖裡疊成一團。
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(
                base=10, subs=(1.0, 2.0, 5.0), numticks=8))
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{v:,.0f}" if v >= 1000 else f"{v:g}"))
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(alpha=0.3, which="both")
            pz = PROF_Z.get(t)
            win = ("STAR" if (not gal or np.nanmin(star) < min(v for _, v in gal))
                   else "GALAXY")
            ax.set_title(f"ID {t}   " + (f"gt z = {pz}" if pz
                                         else "gt: star")
                         + f"   ->  {win}", fontsize=9)
            ax.set_ylabel("best reduced chi2", fontsize=7)
            if k == 0:
                ax.legend(fontsize=7, loc="upper right")

        for k in range(len(ids), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")

        fig.suptitle(f"{ttl}   "
                     f"[{args.star_window[0]:.0f}-{args.star_window[1]:.0f} "
                     f"$\\AA$]\n"
                     "each point = that template's own best fit over the "
                     "redshift scan;  dashed lines = best star / best galaxy",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.955))
        stem = lab.replace(":", "").replace(" ", "_")
        out = (Path(args.out) if args.out
               else FIGURES / f"step4b_template_chi2_{stem}.png")
        fig.savefig(out, dpi=135, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
