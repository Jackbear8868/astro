"""天空連續譜係數 s 的空間形狀 —— 一顆 pointing 一張圖。

s 是什麼
--------
step5 對每個 spaxel 解 D(λ) = s·C_sky(λ) + Σₖ cₖ·Lₖ(λ)。s 是天空連續譜的振幅,
一格一個數。它理想上應該是平滑的(大氣輝光在幾十角秒的尺度上不會突然變),
所以 s 的空間圖是天空模型合不合理的直接檢查。

兩張圖各是什麼(都是 step5 已經寫好的檔,這支不重算)
--------------------------------------------------
    s_free   blank 逐格自由解。只有 blank 有值,源的位置是洞。
             它含求解雜訊,而且**源旁邊會被源的光墊高** —— 那正是 over-subtraction
             的入口:s 長高 -> 天空模型長高 -> 把源的光當天空扣掉。
    s_hat    擬合出來的場 mu + a(y) + b(x)。只用遠離源的格訓練,所以源旁邊那格
             說了不算,而且它伸得進洞裡(a(y) 是整列共用的參數,不是鄰域平均)。

step5 另外寫了一個 s_map(每格實際用掉的值),這裡不畫 —— 它和 s_hat 逐格相同。

    conda run -n astro python src/skymodel/evaluation/s_shape_map.py --work results/skymodel/p01
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, S_CMAP, diverging_range, load_field, pointing_dir  # noqa: E402
from utils import main_source_group  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="s 的空間形狀,一顆一張圖")
    ap.add_argument("--work", required=True, help="pointing 的工作區,例如 results/skymodel/p01")
    ap.add_argument("--run", default=None, help="step05 底下的 run 目錄名;預設取唯一的 *_sfield")
    args = ap.parse_args()

    W = ROOT / args.work
    runs = sorted((W / "step05").glob("*_sfield"))
    run = W / "step05" / args.run if args.run else runs[0]
    if not args.run and len(runs) != 1:
        raise SystemExit(f"★ {W/'step05'} 底下有 {len(runs)} 個 *_sfield,請用 --run 指定")

    seg, white, valid = load_field(W)
    main, ids, _ = main_source_group(seg, np.where(valid, white, np.nan), W / "step04")

    s_free = np.load(run / "s_free.npy").astype(float)
    s_hat  = np.load(run / "s_hat.npy").astype(float)
    for a in (s_free, s_hat):
        a[~valid] = np.nan

    # 色階兩張共用,否則「哪張比較高」看不出來。中心與範圍都由 s_free 決定 ——
    # 它是原始量測,s_hat 是它的擬合,拿擬合去定尺規會把自己的偏差藏起來。
    c, lo, hi = diverging_range(s_free)

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 6.2))
    for a, arr, ttl in zip(ax, (s_free, s_hat),
                           ("s solved per spaxel",
                            "s fitted = mu + a(y) + b(x)")):
        im = a.imshow(arr, origin="lower", cmap=S_CMAP, vmin=lo, vmax=hi)
        a.contour(seg > 0, levels=[0.5], colors="k", linewidths=0.4, alpha=.45)
        a.contour(main,    levels=[0.5], colors="k", linewidths=1.6)
        a.set_title(ttl, fontsize=12); a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046)

    d = (s_free - s_hat)[np.isfinite(s_free) & np.isfinite(s_hat)]
    fig.suptitle(W.name, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    o = pointing_dir(W.name) / "s_shape.png"
    fig.savefig(o, dpi=125, bbox_inches="tight")
    print(f"{W.name}  s_hat 中位 {np.nanmedian(s_hat[valid]):.4f}   "
          f"s_free-s_hat 中位 {np.median(d):+.4f}  散布 {np.std(d):.4f}   -> {o}")


if __name__ == "__main__":
    main()
