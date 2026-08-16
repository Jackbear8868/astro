"""把源遮罩長胖 —— 認標籤的膨脹,固定半徑。

動機:SExtractor 的足跡比源實際的延展緊,漏出去的光被當成 blank,會被天空
模型吸走(見 edge_oversubtraction.py)。把遮罩長胖就是把那圈 spaxel 從
blank 樣本裡拿掉。

**一定要認標籤,不能用普通的 binary dilation。** 相鄰的源之間可能只隔幾個像素,
普通的 dilation 一長就會把它們黏成一塊。這裡的做法是:

    每個 blank 像素問「離我最近的源是誰」,只歸給那一個源。
    兩個源之間會自然在中線停住,標籤永遠不會互相吞併。

用 distance_transform_edt(seg == 0, return_indices=True) 一次拿到距離與最近源的
座標,查 seg 就是歸屬。半徑是**固定值**、由使用者指定。

這支只產生遮罩與 source id map,不跑任何擬合。

    conda run -n astro python src/skymodel/experiments/dilate_seg.py --radii 2 4 8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import id_map

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/ne_pointing/step01"
EXPDIR  = ROOT / "results/skymodel/experiments/blank_geometry/seg_dilated"
FIGURES = ROOT / "results/skymodel/evaluation/masking/edge_oversub"


def dilate(seg, r):
    """認標籤的膨脹:每個 blank 像素歸給離它最近的那一個源,距離 <= r 才收。

    r 可以是一個數(所有源同一個半徑),也可以是 {源 ID: 半徑} 的字典
    (逐源不同,例如 ring_consistency.py 量出來的 r_stop)。字典裡沒有的源不長。
    """
    dist, (iy, ix) = ndimage.distance_transform_edt(seg == 0, return_indices=True)
    owner = seg[iy, ix]
    if isinstance(r, dict):
        rmap = np.zeros(int(seg.max()) + 1)
        for k, v in r.items():
            rmap[int(k)] = v
        lim = rmap[owner]
    else:
        lim = float(r)
    grow  = (seg == 0) & (owner > 0) & (dist <= lim)
    out   = seg.copy()
    out[grow] = owner[grow]
    return out, int(grow.sum())


def main():
    ap = argparse.ArgumentParser(description="認標籤的遮罩膨脹")
    ap.add_argument("--seg", default=str(STEP01 / "seg.fits"))
    ap.add_argument("--radii", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--rstop-json", default=None,
                    help="改用 ring_consistency.py 量出來的逐源 r_stop。"
                         "給了就忽略 --radii")
    ap.add_argument("--outdir", default=str(EXPDIR))
    ap.add_argument("--figdir", default=str(FIGURES))
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    fig = Path(args.figdir); fig.mkdir(parents=True, exist_ok=True)
    hdr   = fits.getheader(args.seg)
    seg   = fits.getdata(args.seg).astype(np.int32)
    white = fits.getdata(STEP01 / "whitelight.fits")
    valid = white != 0
    n_src0, n_blank0 = int((seg > 0).sum()), int(((seg == 0) & valid).sum())
    print(f"原始 {Path(args.seg).name}:源 {n_src0:,} spaxel,"
          f"blank {n_blank0:,}(有效視野內)")
    print(f"\n{'r':>3}{'新增':>10}{'源 spaxel':>12}{'佔視場':>9}"
          f"{'blank 剩':>11}{'blank 少了':>11}{'消失的源':>10}{'貼在一起':>10}")

    sh = ndimage.generate_binary_structure(2, 2)
    if args.rstop_json:
        j = json.loads(Path(args.rstop_json).read_text())
        radii = [{int(k): v["r_stop"] for k, v in j["sources"].items()}]
        print(f"逐源 r_stop 來自 {Path(args.rstop_json).name}:"
              f"{len(radii[0])} 個源,中位 "
              f"{np.median(list(radii[0].values())):.0f} px")
    else:
        radii = args.radii
    for r in radii:
        new, added = dilate(seg, r)
        n_src   = int((new > 0).sum())
        n_blank = int(((new == 0) & valid).sum())
        # 標籤守恆:認標籤成長不可能讓任何源消失。長胖後相鄰的源會在中線**貼在
        # 一起**,那是預期行為不是合併 —— 標籤仍然分開,step5 依 seg == rid 取
        # 區域完全不受影響。(用連通元件去數會把貼在一起的誤判成合併。)
        lost  = sorted(set(np.unique(seg[seg > 0]).tolist())
                       - set(np.unique(new[new > 0]).tolist()))
        touch = set()
        for i in np.unique(new[new > 0]):
            nb = np.unique(new[ndimage.binary_dilation(new == i, sh) & (new != i)])
            touch |= {tuple(sorted((int(i), int(j)))) for j in nb if j > 0}
        rlab = "rstop" if isinstance(r, dict) else str(r)
        print(f"{rlab:>5}{added:>10,}{n_src:>12,}{100 * n_src / seg.size:>8.1f}%"
              f"{n_blank:>11,}{100 * (1 - n_blank / n_blank0):>10.1f}%"
              f"{len(lost):>10}{len(touch):>10}")

        f = out / (f"seg_dil{r}.fits" if not isinstance(r, dict)
                   else "seg_rstop.fits")
        fits.writeto(f, new, hdr, overwrite=True)

        ids, cnt = np.unique(new[new > 0], return_counts=True)
        rows = []
        for i in ids:
            y, x = np.nonzero(new == i)
            rows.append(dict(id=int(i), x=float(x.mean()), y=float(y.mean()),
                             group="galaxy"))
        p = fig / (f"11_id_map_dil{r}.png" if not isinstance(r, dict)
                   else "17_id_map_rstop.png")
        id_map(new, white, rows, p, by_group=False)
        print(f"     -> {f.name}   {p.name}")


if __name__ == "__main__":
    main()
