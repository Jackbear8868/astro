"""任一份 segmentation 的 source ID map —— 沿用 utils.id_map 的畫法。

utils.id_map() 的 rows 需要帶著 group、
紅移那些欄位)。但有時候我們只想看「這份遮罩把哪些東西圈成源」,和擬合無關 ——
例如比較教授給的 1 sigma 與 2 sigma 兩份 segmentation。

這支只做一件事:從一張 seg 圖組出 id_map() 要的 rows(id 與形心),
其餘的繪製完全交給 utils.id_map,不另外寫一套。同一張圖在專案裡
只能有一種畫法,否則兩份圖擺在一起會因為拉伸、配色、標號規則不同而看起來
像不同的資料。

    conda run -n astro python src/skymodel/evaluation/seg_id_map.py \\
        --work results/skymodel/p01
    conda run -n astro python src/skymodel/evaluation/seg_id_map.py \\
        --work results/skymodel/p01 --seg data/wsky_seg/DATACUBE_FINAL_1_seg.fits
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import EVAL, ROOT
from utils import id_map

FIGURES = EVAL / "masking"


def main():
    ap = argparse.ArgumentParser(description="任一份 segmentation 的 source ID map")
    ap.add_argument("--work", required=True,
                    help="pointing 的工作區,例如 results/skymodel/p01。"
                         "seg 與底圖預設都取自它的 step01/")
    ap.add_argument("--seg", default=None,
                    help="要畫的 segmentation 圖;預設為工作區的 step01/seg.fits")
    ap.add_argument("--white", default=None,
                    help="底圖;預設為工作區的 step01/whitelight.fits。"
                         "指到別份 seg 時底圖仍要是**那一顆自己的**白光 —— "
                         "配錯底圖會看到對不齊的錯覺,而那不是資料的問題")
    ap.add_argument("--min-area", type=int, default=1,
                    help="只畫面積 >= 這個值的源;"
                         "小到幾個像素的標上去只會互相蓋住")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    STEP01 = ROOT / args.work / "step01"
    white = fits.getdata(Path(args.white) if args.white
                         else STEP01 / "whitelight.fits")
    seg_path = Path(args.seg) if args.seg else STEP01 / "seg.fits"
    seg   = fits.getdata(seg_path)
    if seg.shape != white.shape:
        raise SystemExit(f"seg {seg.shape} 與白光圖 {white.shape} 尺寸不同")

    ids, cnt = np.unique(seg[seg > 0], return_counts=True)
    rows = []
    for i, c in zip(ids, cnt):
        if c < args.min_area:
            continue
        y, x = np.nonzero(seg == i)
        # group 欄位 id_map 只在 by_group=True 時用到,這裡一律 by_group=False,
        # 但 rows 的欄位要齊全,否則 utils 那邊的 GROUP_COLOR 會 KeyError。
        rows.append(dict(id=int(i), x=float(x.mean()), y=float(y.mean()),
                         group="galaxy"))

    # 檔名帶工作區名稱:14 顆的 step01/seg.fits 同名,只用 seg 的檔名會互相覆蓋。
    name = f"{Path(args.work).name}_{seg_path.stem}"
    out = Path(args.out) if args.out else FIGURES / f"id_map_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    id_map(seg, white, rows, out, by_group=False)
    print(f"{name}: {len(ids)} 個源,畫出 {len(rows)} 個 (面積 >= {args.min_area} px)")
    print(f"  源 spaxel {int((seg > 0).sum()):,}  ({100 * (seg > 0).mean():.1f}% of field)")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
