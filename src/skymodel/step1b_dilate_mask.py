"""把 SExtractor 的源遮罩往外長胖 —— 認標籤的膨脹。

為什麼要長胖
------------
SExtractor 的足跡比源實際的延展緊,漏在外面的光被當成 blank。那些 spaxel 同時是
step3 學 basis 的樣本,所以污染是雙重的:天空模型學到了源的形狀,然後拿那個形狀
去把源扣掉。

**輸入的 seg.fits 一個位元都不動。** 這支讀它、寫出一份長胖的副本,下游用
`--seg` 顯式指過去。用不用膨脹版從指令就看得出來,而且兩種結果可以並存比較。

為什麼一定要認標籤
------------------
普通的 `binary_dilation` 作用在**二值**影像上,一開始就把標籤丟掉了 —— 相鄰的
源會黏成同一個連通塊,再也分不出哪個像素屬於誰。

這裡的做法:

    dist, (iy, ix) = distance_transform_edt(seg == 0, return_indices=True)
    owner = seg[iy, ix]

`distance_transform_edt` 對每個 True 的像素算「離最近的 False 像素多遠」。餵進
`seg == 0`,得到的就是每個 blank 像素離最近的源多遠;`return_indices` 另外給出
那個源像素的座標,拿它去查 seg 就是**歸屬**。

於是整個視場被切成一張 Voronoi 圖:每個 blank 像素只有**一個** owner。兩個源之間
的像素沿垂直平分線一刀分開,A 長到中線就停,B 也是,**永遠不重疊**。長胖後相鄰的源
會共用一段邊界,那是**貼在一起**不是合併 —— 標籤仍然分開,step5 用 `seg == rid`
取區域完全不受影響。

半徑怎麼選
----------
`r` 是歐氏距離,單位是像素。dist 只會出現 sqrt(a^2+b^2) 這些值,所以 r=1 只收
上下左右(1.000),不收斜對角(1.414)。長出來的是圓角的緩衝區,不是方形。

    r = 2 px = 0.40 角秒 = 0.49 x seeing FWHM
    r = 3 px = 0.60 角秒 = 0.74 x seeing FWHM
    r = 4 px = 0.80 角秒 = 0.99 x seeing FWHM   <- CLAUDE.md 檢查表的建議值

檢查表寫「dilation (safety margin) ≈ 1 x seeing FWHM ≈ 4 px」,物理意義是:即使源
本身是個點,大氣視寧度也會把它的光抹開約一個 FWHM,遮罩至少要留這麼寬才不會把
被抹出去的光算成天空。

    conda run -n astro python src/skymodel/step1b_dilate_mask.py -r 4
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage

ROOT   = Path(__file__).resolve().parents[2]
STEP01 = ROOT / "results/skymodel/step01"


def dilate(seg, r):
    """認標籤的膨脹。回傳 (新的 seg, 新增的像素數)。

    每個 blank 像素歸給離它最近的那一個源,距離 <= r 才收。見檔案上方說明。
    """
    dist, (iy, ix) = ndimage.distance_transform_edt(seg == 0, return_indices=True)
    owner = seg[iy, ix]
    grow  = (seg == 0) & (owner > 0) & (dist <= r)
    out   = seg.copy()
    out[grow] = owner[grow]
    return out, int(grow.sum())


def main():
    ap = argparse.ArgumentParser(description="認標籤地把源遮罩長胖")
    ap.add_argument("-r", "--radius", type=float, required=True,
                    help="膨脹半徑(像素,歐氏距離)")
    ap.add_argument("--seg", default=str(STEP01 / "seg.fits"),
                    help="輸入的 segmentation。預設是 SExtractor 的產物,不會被改")
    ap.add_argument("--out", default=None,
                    help="輸出路徑。預設 <輸入資料夾>/seg_dil{r}.fits")
    args = ap.parse_args()

    src   = Path(args.seg)
    seg   = fits.getdata(src).astype(np.int32)
    white = fits.getdata(STEP01 / "whitelight.fits")
    valid = white != 0

    new, added = dilate(seg, args.radius)

    n_blank0 = int(((seg == 0) & valid).sum())
    n_blank1 = int(((new == 0) & valid).sum())
    print(f"{src.name}  ->  膨脹 r = {args.radius:g} px "
          f"({args.radius * 0.2:.2f} 角秒,{args.radius / 4.06:.2f} x seeing FWHM)")
    print(f"  源 spaxel   {int((seg > 0).sum()):,} -> {int((new > 0).sum()):,}"
          f"   (+{added:,})")
    print(f"  blank       {n_blank0:,} -> {n_blank1:,}"
          f"   (少了 {100 * (1 - n_blank1 / n_blank0):.1f}%)")

    # 認標籤成長不可能讓任何源消失 —— 每個 blank 像素只歸給一個 owner。這一行是
    # 把那個保證真的驗一次,不是裝飾。
    lost = sorted(set(np.unique(seg[seg > 0]).tolist())
                  - set(np.unique(new[new > 0]).tolist()))
    print(f"  標籤守恆:消失的源 {len(lost)} 個" + (f" {lost}" if lost else " ✓"))

    out = Path(args.out) if args.out else src.with_name(
        f"seg_dil{args.radius:g}.fits")
    fits.writeto(out, new, fits.getheader(src), overwrite=True)
    print(f"saved -> {out}")
    print(f"\n下游用法(教授的 {src.name} 不受影響):")
    print(f"  step3: --seg {out}")
    print(f"  step5: --seg {out}")


if __name__ == "__main__":
    main()
