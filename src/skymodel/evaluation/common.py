"""evaluation 底下的腳本共用的東西 —— 路徑、讀檔、環的定義、顯示拉伸。

放進來的條件只有一個:**兩支以上的腳本在做同一件事**。同一件事各留一份的話,
改了其中一份、另一份沒跟著改,兩張圖就會在看不出來的地方不一致 —— 例如環的
半徑差 1 px、底圖的拉伸不同,而圖上看起來只像是資料變了。

只有一支腳本用得到的東西留在那支腳本裡。搬進來的話,讀那支腳本的人要跨兩個
檔案才看得懂它在做什麼。

檔名不能叫 utils.py:這裡的腳本用 sys.path 指到上一層去 import
`src/skymodel/utils.py`,同名的檔案會把它蓋掉。
"""
import re
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
# 圖與量測值一律寫中央,檔名帶 pointing —— 放各自的工作區裡的話,要比較幾顆
# 就得開幾個目錄,而且檔名相同排不到一起。
EVAL = ROOT / "results/skymodel/evaluation"

# 跨 spaxel 剔除壞 voxel 的門檻,沿用 step3_sky_basis.py 的 mean_sky。
CLIP_SIGMA = 30

# s 的空間圖一律用同一組色階。RdBu_r 是發散色階,中心點兩側分成兩個顏色 ——
# 「比典型值高還是低」一眼可分,而連續色階只能分出「深淺」。
S_CMAP = "RdBu_r"


def pointing_dir(name, *sub):
    """一顆 pointing 的圖放哪 —— evaluation/pNN/[子目錄…],並確保目錄存在。

    一顆一個目錄,而不是把 14 顆的圖混在同一層用檔名區分:看某一顆的時候,
    要的是那一顆的全部,不是在幾百個檔名裡挑出帶 pNN 的那些。
    """
    d = EVAL.joinpath(name, *sub)
    d.mkdir(parents=True, exist_ok=True)
    return d


# 疊在 magma 底圖上的輪廓顏色。magma 走「黑 -> 紫 -> 橘 -> 淡黃」,所以輪廓不能
# 用黑、橘、黃 —— 那些是色帶本身的顏色,線和暗處/亮處的資料分不開。亮綠落在色帶
# 之外,疊上去一定看得見。
SEG_COLOR = "#39ff14"


def asinh_bar(fig, im, ax, label, lo, hi):
    """asinh 拉伸的色階條 —— 刻度標在原本的物理值上,不是拉伸後的數字。

    影像畫的是 arcsinh(z),所以色階條的預設刻度是 0、2、4… 那些是壓縮過的數字,
    讀不出「這裡比天空亮多少」。把刻度放到 arcsinh(z) 的位置、標上 z 本身,
    色階就能直接當成 z 讀,而影像仍然享有 asinh 的動態範圍。

    lo/hi 是 z(不是 arcsinh(z))的範圍,用來決定哪些刻度落得進去。
    """
    ticks = [-3, -1, 0, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
    t = [v for v in ticks if lo <= v <= hi]
    cb = fig.colorbar(im, ax=ax, fraction=0.046,
                      ticks=[np.arcsinh(v) for v in t])
    cb.ax.set_yticklabels([f"{v:g}" for v in t])
    cb.set_label(label, fontsize=9)
    return cb


def slug(name):
    """區域名稱 -> 檔名。"src edge d=7px" -> "src_edge_d_7px"。

    只留字母數字,其餘一律換成底線再收斂重複的底線 —— 空白、'='、'#' 在 shell
    和路徑裡都要跳脫,而這些名稱是自動組出來的,遲早會出現。
    """
    return re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", name)).strip("_")


def diverging_range(a, centre=None, pct=2.0):
    """發散色階的 (中心, vmin, vmax) —— 對稱,而且對離群格穩健。

    範圍取分位數而不是最大最小值:少數幾個壞格就能把 min/max 拉到讓其餘部分
    全部擠成同一個顏色。中心不給的話用中位數,結構的對比最大。
    """
    v = a[np.isfinite(a)]
    c = float(np.median(v)) if centre is None else float(centre)
    r = float(max(abs(np.percentile(v, pct) - c), abs(np.percentile(v, 100 - pct) - c)))
    return c, c - r, c + r


def load_field(work):
    """一顆 pointing 的 step01 產物 —— 回傳 (seg, 白光圖, 有效視野遮罩)。

    白光圖轉成 float,後面拿它算分位數與 nanmean 時才不受原始 dtype 影響。
    視野外的 spaxel 在白光圖上是 0,valid 就是「這一格有資料」。
    """
    seg   = fits.getdata(work / "step01/seg.fits").astype(int)
    white = np.asarray(fits.getdata(work / "step01/whitelight.fits"), float)
    return seg, white, white != 0


def zones(seg, white, main):
    """四個環的遮罩,回傳 {名稱: 布林圖}。

        far          離所有源都遠。那裡沒有源,殘差的真值是 0。
        small 1-3    小源足跡外 1-3 px。
        main 1-3     主源足跡外 1-3 px。
        main 3-10    主源足跡外 3-10 px,延展的暈。

    四個環一律取 seg == 0 的格 —— 偵測不到的地方不代表沒有源的光,而那正是
    最容易被過度扣除的位置。視野邊緣 15 px 以內全部排除:那裡曝光次數少,量到
    的是拼接的邊界效應。

    main 由呼叫端給,不在這裡算 —— 要不要拿 step04 的紅移去篩成員,是呼叫端
    的決定。
    """
    edge   = ndimage.distance_transform_edt(white != 0)
    d_all  = ndimage.distance_transform_edt(seg == 0)
    d_main = ndimage.distance_transform_edt(~main)
    others = (seg > 0) & ~main
    # 沒有其他源時,「離其他源的距離」設成一個大到不會擋掉任何格的值
    d_oth  = (ndimage.distance_transform_edt(~others) if others.any()
              else np.full(seg.shape, 1e9))
    base = (seg == 0) & (white != 0) & (edge > 15)
    return {"far":       base & (d_all > 30) & (d_main > 110),
            "small 1-3": base & (d_oth > 1) & (d_oth <= 3) & (d_main > 30),
            "main 1-3":  base & (d_main > 1) & (d_main <= 3) & (d_oth > 6),
            "main 3-10": base & (d_main > 3) & (d_main <= 10) & (d_oth > 6)}


def arcsinh_stretch(img, valid=None, soft=0.02):
    """底圖的 asinh 拉伸 —— 回傳 (拉伸後的影像, vmax),vmin 一律用 0。

    白光的動態範圍跨好幾個量級(主星系本體 vs 暗源),線性顯示會讓本體以外的
    東西全黑。asinh 在亮處是對數、暗處是線性。軟閾值取 99.5 分位的 soft 倍,
    所以 vmax 恆為 arcsinh(1 / soft),換一份底圖仍然是同一組刻度。

    valid 給了的話,視野外的格填 NaN(畫出來是空白);不給就照原值畫。
    """
    m = np.isfinite(img) & (img != 0)
    v = np.nanpercentile(img[m], 99.5)
    a = img if valid is None else np.where(valid, img, np.nan)
    return np.arcsinh(a / (soft * v)), np.arcsinh(1 / soft)


def collapse(path, band, wl, seg):
    """把 cube 在指定波段壓成一張影像 —— 回傳 (影像, 剔除數, 參與剔除的元素數)。

    壞 voxel 先剔除再平均,做法與門檻沿用 step3_sky_basis.py 的 mean_sky:
    同一通道、跨 spaxel,中心取中位數、散布取 (p84 - p16) / 2(對離群值穩健),
    偏離超過 CLIP_SIGMA 倍的剔掉。這樣就不必另外挑一個絕對門檻。

    但**只剪 blank**。一條天光線在每個 spaxel 都亮,它的亮度就在該通道的中位數
    裡,不會被剪掉;源不是這樣 —— 源只在少數 spaxel 亮,跨 spaxel 看它本來就是
    離群值,用同一把尺會把源的 spaxel 整條剪光,nanmean 之後變成 NaN。
    """
    m = (wl >= band[0]) & (wl < band[1])
    with fits.open(path, memmap=True) as h:
        d = np.asarray(h[0].data[m] if h[0].data is not None
                       else h["DATA"].data[m], np.float32).copy()
    blank = seg == 0
    bl  = d[:, blank]
    p16, med, p84 = np.nanpercentile(bl, [16, 50, 84], axis=1)
    sg  = np.maximum((p84 - p16) / 2, 1e-6)
    bad = np.abs(bl - med[:, None]) > CLIP_SIGMA * sg[:, None]
    d[:, blank] = np.where(bad, np.nan, bl)
    return np.nanmean(d, axis=0), int(bad.sum()), int(bad.size)
