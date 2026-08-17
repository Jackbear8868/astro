"""天空連續譜係數 s 的空間形狀 —— 一顆 pointing 一張圖。

s 是什麼
--------
step5 對每個 spaxel 解 D(λ) = s·C_sky(λ) + Σₖ cₖ·Lₖ(λ)。s 是天空連續譜的振幅,
一格一個數。它理想上應該是平滑的(大氣輝光在幾十角秒的尺度上不會突然變),
所以 s 的空間圖是天空模型合不合理的直接檢查。

三張圖各是什麼(都是 step5 已經寫好的檔,這支不重算)
--------------------------------------------------
    s_free   blank 逐格自由解。只有 blank 有值,源的位置是洞。
             它含求解雜訊,而且**源旁邊會被源的光墊高** —— 那正是 over-subtraction
             的入口:s 長高 -> 天空模型長高 -> 把源的光當天空扣掉。
    s_hat    擬合出來的場 mu + a(y) + b(x)。只用遠離源的格訓練,所以源旁邊那格
             說了不算,而且它伸得進洞裡(a(y) 是整列共用的參數,不是鄰域平均)。
    s_map    每格實際用掉的值。

下排的兩條剖面是場的兩個成分:a(y) 沿 y、b(x) 沿 x。s_free 的逐列/逐行中位數
疊在上面,看場有沒有跟上資料。

    conda run -n astro python src/skymodel/experiments/s_shape_map.py --work results/skymodel/p01
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import main_source_group  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
# 圖寫中央、檔名帶 pointing —— 放各自的工作區裡的話,要比較幾顆就得開幾個目錄。
EVAL = ROOT / "results/skymodel/evaluation/s_field"


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

    seg   = fits.getdata(W / "step01/seg.fits").astype(int)
    white = np.asarray(fits.getdata(W / "step01/whitelight.fits"), float)
    valid = white != 0
    main, ids, _ = main_source_group(seg, np.where(valid, white, np.nan), W / "step04")

    s_free = np.load(run / "s_free.npy").astype(float)
    s_hat  = np.load(run / "s_hat.npy").astype(float)
    s_map  = np.load(run / "s_map.npy").astype(float)
    for a in (s_free, s_hat, s_map):
        a[~valid] = np.nan

    # 色階三張共用,否則「哪張比較高」看不出來。用穩健分位數,免得被少數離群格
    # 把整個色階拉平。
    v = s_free[np.isfinite(s_free)]
    lo, hi = np.percentile(v, [2, 98])

    fig, ax = plt.subplots(2, 3, figsize=(17, 10))
    for a, arr, ttl in zip(ax[0], (s_free, s_hat, s_map),
                           ("s_free   (blank, solved per spaxel)",
                            "s_hat   (fitted field  mu + a(y) + b(x))",
                            "s_map   (what step5 actually used)")):
        im = a.imshow(arr, origin="lower", cmap="viridis", vmin=lo, vmax=hi)
        a.contour(seg > 0, levels=[0.5], colors="w", linewidths=0.4, alpha=.55)
        a.contour(main,    levels=[0.5], colors="#e8272c", linewidths=1.4)
        a.set_title(ttl, fontsize=11); a.set_xticks([]); a.set_yticks([])
        fig.colorbar(im, ax=a, fraction=0.046)

    # 逐列/逐行中位數:s_free 用 nanmedian(有洞),s_hat 沒有洞
    yy = np.arange(seg.shape[0]); xx = np.arange(seg.shape[1])
    with np.errstate(all="ignore"):
        fy, hy = np.nanmedian(s_free, axis=1), np.nanmedian(s_hat, axis=1)
        fx, hx = np.nanmedian(s_free, axis=0), np.nanmedian(s_hat, axis=0)
    for a, u, f, h, lbl in ((ax[1, 0], yy, fy, hy, "y  (across slices)"),
                            (ax[1, 1], xx, fx, hx, "x  (along slice)")):
        a.plot(u, f, lw=0.8, color="#8a8a8a", label="s_free row/col median")
        a.plot(u, h, lw=1.6, color="#e8272c", label="s_hat")
        a.set_xlabel(lbl); a.set_ylabel("s"); a.grid(alpha=.25); a.legend(fontsize=8)

    d = (s_free - s_hat)[np.isfinite(s_free) & np.isfinite(s_hat)]
    ax[1, 2].hist(d, bins=120, color="#1f77b4", alpha=.85)
    ax[1, 2].axvline(0, color="k", lw=0.9)
    ax[1, 2].set_xlabel("s_free - s_hat"); ax[1, 2].set_ylabel("blank spaxels")
    ax[1, 2].grid(alpha=.25)
    ax[1, 2].set_title(f"median {np.median(d):+.4f}   scatter {np.std(d):.4f}", fontsize=10)

    p = json.loads((run / "meta.json").read_text())["s_field_params"]
    fig.suptitle(f"{W.name}   {run.name}   "
                 f"s_hat trained on spaxels > {p['r_far']:.0f} px from any source, "
                 f"> {p['r_far_haro']:.0f} px from Haro 11   "
                 f"(red = main source group, ids {ids})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    EVAL.mkdir(parents=True, exist_ok=True)
    o = EVAL / f"s_shape_{W.name}.png"
    fig.savefig(o, dpi=125, bbox_inches="tight")
    print(f"{W.name}  s_hat 中位 {np.nanmedian(s_hat[valid]):.4f}   "
          f"s_free-s_hat 中位 {np.median(d):+.4f}  散布 {np.std(d):.4f}   -> {o.name}")


if __name__ == "__main__":
    main()
