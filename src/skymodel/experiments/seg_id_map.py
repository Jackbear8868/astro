"""任一份 segmentation 的 source ID map —— 沿用 step4_catalog 的畫法。

step4_catalog.id_map() 需要 step4 的擬合目錄才跑得起來(rows 帶著 group、
紅移那些欄位)。但有時候我們只想看「這份遮罩把哪些東西圈成源」,和擬合無關 ——
例如比較教授給的 1 sigma 與 2 sigma 兩份 segmentation。

這支只做一件事:從一張 seg 圖組出 id_map() 要的 rows(id 與形心),
其餘的繪製完全交給 step4_catalog.id_map,不另外寫一套。同一張圖在專案裡
只能有一種畫法,否則兩份圖擺在一起會因為拉伸、配色、標號規則不同而看起來
像不同的資料。

    conda run -n astro python src/skymodel/experiments/seg_id_map.py \\
        --seg data/Haro11_NEpointing_seg1sigma.fits
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step4_catalog import id_map

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
FIGURES = ROOT / "results/skymodel/figures"


def main():
    ap = argparse.ArgumentParser(description="任一份 segmentation 的 source ID map")
    ap.add_argument("--seg", default=str(ROOT / "data/Haro11_NEpointing_seg1sigma.fits"),
                    help="要畫的 segmentation 圖")
    ap.add_argument("--min-area", type=int, default=1,
                    help="只畫面積 >= 這個值的源。1 sigma 有 116 個源,"
                         "小到幾個像素的標上去只會互相蓋住")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    white = fits.getdata(STEP01 / "whitelight.fits")
    seg   = fits.getdata(args.seg)
    if seg.shape != white.shape:
        raise SystemExit(f"seg {seg.shape} 與白光圖 {white.shape} 尺寸不同")

    ids, cnt = np.unique(seg[seg > 0], return_counts=True)
    rows = []
    for i, c in zip(ids, cnt):
        if c < args.min_area:
            continue
        y, x = np.nonzero(seg == i)
        # group 欄位 id_map 只在 by_group=True 時用到,這裡一律 by_group=False,
        # 但 rows 的欄位要齊全,否則 step4_catalog 那邊的 GROUP_COLOR 會 KeyError。
        rows.append(dict(id=int(i), x=float(x.mean()), y=float(y.mean()),
                         group="galaxy"))

    name = Path(args.seg).stem
    out = Path(args.out) if args.out else FIGURES / f"id_map_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    id_map(seg, white, rows, out, by_group=False)
    print(f"{name}: {len(ids)} 個源,畫出 {len(rows)} 個 (面積 >= {args.min_area} px)")
    print(f"  源 spaxel {int((seg > 0).sum()):,}  ({100 * (seg > 0).mean():.1f}% of field)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
