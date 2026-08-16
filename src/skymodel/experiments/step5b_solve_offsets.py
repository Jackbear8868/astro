"""求出三顆曝光相對 ESO 合併品的平移量 (ox, oy) —— 用現成的標準函式庫,不自己刻。

約定
----
    ref[y, x]  對應到  exp[y + oy, x + ox]
亦即重取樣時寫成 ``out[y, x] = exp[y + oy, x + ox]``。以下所有數字都是這個約定。

為什麼不用「白光圖互相關」
--------------------------
天空連續譜在空間上幾乎是平的,白光圖因此有一個巨大的正底盤。兩張處處非負的圖做
未正規化互相關,極大值會落在「重疊面積最大」的位移,而不是「結構對齊」的位移。
三顆曝光的視場形狀還不一樣 (306x315 vs 315x306),偏差量因此每顆都不同。先扣
平均、再除以重疊像素數也救不回來,因為缺值邊界本身就是結構,會被互相關當成訊號。

所以這裡改用兩個對「底盤」與「視場形狀」免疫的標準工具,而且要兩個都同意:

  階段 A(粗解,整數) skimage.registration.phase_cross_correlation 的 **遮罩版**
      給了 reference_mask / moving_mask 之後,skimage 走的是 Padfield (2012) 的
      masked normalized cross-correlation:它在每一個候選位移上「只用兩張圖都有效
      的那些像素」重新算一次平均與變異數,再做正規化相關。底盤被逐位移地扣掉,
      重疊面積被逐位移地除掉,所以上面那個偏差在數學上就不存在。
      代價是這個版本只給整數位移(skimage 不支援它的次像素上取樣)。

  階段 B(細解,次像素) 源的位置
      源的中心就是源的中心,和視場多大、邊緣缺多少完全無關。用兩個獨立的估計器:
        B1  photutils.DAOStarFinder 在兩張圖上各自找源,把「每一組 (ref源, exp源)
            配對所隱含的位移」丟進二維直方圖投票。真正的位移會讓十幾組配對同時
            投到同一格,雜訊配對則散得到處都是 —— 這就是 astroalign / ESO 的
            muse_exp_align 在做的事,而且完全不看像素值大小。
        B2  對每顆源擬合「二維高斯 + 平面背景」(step5b_align_check.fit_gauss),
            取所有源殘差的中位數。背景放進擬合而不是先扣掉:源附近的天空有梯度,
            用環狀中位先扣會把中心往梯度高的一側拉。

驗證(--synthetic)
-------------------
把參考圖平移一個「已知」的量、裁成不同長寬比、加上缺值邊框,再要求方法把那個已知
量還原回來。這一步同時抓兩種錯:符號/轉置的約定錯誤,以及視場形狀造成的偏差。
遮罩版與未遮罩的純相位相關 (normalization='phase') 並列輸出,兩者可直接對照。

skimage 回傳值的方向:phase_cross_correlation(ref, mov) 給的是「要把 mov 移多少才
對得上 ref」,和本檔約定差一個負號,即 (oy, ox) = -(raw[0], raw[1])。

用法
----
    conda run -n astro python src/skymodel/experiments/step5b_solve_offsets.py
    conda run -n astro python src/skymodel/experiments/step5b_solve_offsets.py --synthetic
    conda run -n astro python src/skymodel/experiments/step5b_solve_offsets.py --grid
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.registration import phase_cross_correlation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step5b_align_check import fit_gauss, med_image   # noqa: E402

ROOT   = Path(__file__).resolve().parents[3]
STEP01 = ROOT / "results/skymodel/ne_pointing/step01"
STEP03 = ROOT / "results/skymodel/ne_pointing/step03"
EXPS   = ROOT / "data/NEpointing_exps"
REF    = ROOT / "data/Haro11_NEpointing_wsky.fits"

PSF_FWHM = 4.06        # px,由 cube 裡的星量出來的(見 CLAUDE.md 的參數表)
CANVAS   = (512, 512)  # 共同畫布:比任一張圖都大,兩張都貼在原點,位移才有共同原點


# ---------------------------------------------------------------- 階段 A:粗解
def _pad(img, shape=CANVAS):
    """把圖貼到共同畫布的左上角。空白填 NaN,之後會變成遮罩的 False。"""
    out = np.full(shape, np.nan)
    out[:img.shape[0], :img.shape[1]] = img
    return out


def coarse_offset(ref, exp):
    """遮罩版正規化互相關,回傳整數 (ox, oy)。

    傳進去的像素值本身可以留著底盤 —— Padfield 的做法會在每個候選位移上自己扣掉
    重疊區的平均,所以先不先扣平均並不影響結果。遮罩才是重點:它告訴演算法哪些
    像素是「真的有觀測到」,缺值邊界因此不會被當成結構。
    """
    A, B = _pad(ref), _pad(exp)
    mA, mB = np.isfinite(A), np.isfinite(B)
    shift, _, _ = phase_cross_correlation(
        np.where(mA, A, 0.0), np.where(mB, B, 0.0),
        reference_mask=mA, moving_mask=mB)
    return -float(shift[1]), -float(shift[0])      # 負號:見模組說明


# ---------------------------------------------------------------- 階段 B:細解
def detect(img, nsig=3.0):
    """DAOStarFinder 找點源,回傳 (N, 2) 的 (y, x)。

    門檻用 sigma-clip 之後的 σ,不是全圖的 σ:全圖含源會把 σ 灌大。3σ 偏鬆,會混進
    假源,但投票法本來就靠「同一個位移被很多組配對同時投中」來過濾,假源只會均勻
    地散在整個位移平面上,不會堆成峰。
    """
    a = np.where(np.isfinite(img), img, np.nan)
    _, med, std = sigma_clipped_stats(a, sigma=3.0, maxiters=10)
    tbl = DAOStarFinder(fwhm=PSF_FWHM, threshold=nsig * std, exclude_border=True)(
        np.nan_to_num(a - med, nan=0.0), mask=~np.isfinite(a))
    if tbl is None:
        return np.empty((0, 2))
    return np.c_[np.asarray(tbl["ycentroid"]), np.asarray(tbl["xcentroid"])]


def vote_offset(ref_pts, exp_pts, guess, tol=1.5, iters=5):
    """源配對投票。guess 只用來圈出要收斂的那個峰,不影響最後的值。

    每一組 (ref 源, exp 源) 都隱含一個位移 exp_pos - ref_pos。真位移會被十幾組同時
    投中,所以在 guess 附近 tol 內取中位數、再以中位數為新中心重圈,幾輪就收斂。
    用中位數而不是平均:單一組錯誤配對不會把答案拉走。
    """
    dy = (exp_pts[:, 0][:, None] - ref_pts[:, 0][None, :]).ravel()
    dx = (exp_pts[:, 1][:, None] - ref_pts[:, 1][None, :]).ravel()
    cy, cx = guess[1], guess[0]
    m = np.zeros(dy.shape, bool)
    for _ in range(iters):
        m = (np.abs(dy - cy) < tol) & (np.abs(dx - cx) < tol)
        if m.sum() == 0:
            return np.nan, np.nan, 0, np.nan, np.nan
        cy, cx = float(np.median(dy[m])), float(np.median(dx[m]))
    return cx, cy, int(m.sum()), float(np.std(dx[m])), float(np.std(dy[m]))


def gauss_offset(ref_img, exp_img, probes, guess, iters=5):
    """對每顆源擬合高斯+平面背景,取殘差中位數修正 guess,迭代到收斂。

    要迭代的原因:擬合窗只有 +-7 px,起始猜測若偏了 3 px,源就不在窗中央,擬合會
    抓到雜訊或旁邊的源。先粗解一次把窗擺正,再解一次就穩了。
    """
    # 先在參考圖上定出每顆源的中心,這是之後所有殘差的基準
    ref_fit = {}
    for i, yc, xc in probes:
        f = fit_gauss(ref_img, yc, xc)
        if np.isfinite(f[0]) and 2.0 < f[2] < 12.0:
            ref_fit[i] = f

    gx, gy = guess
    dxs, sx, sy = [], np.nan, np.nan
    for _ in range(iters):
        dxs, dys = [], []
        for ry, rx, _, _ in ref_fit.values():
            f = fit_gauss(exp_img, ry + gy, rx + gx)     # 預期位置:ref -> exp
            if not np.isfinite(f[0]) or not (2.0 < f[2] < 12.0):
                continue
            dy, dx = f[0] - (ry + gy), f[1] - (rx + gx)
            if max(abs(dy), abs(dx)) > 5:                # 貼在窗邊 = 沒收斂到源上
                continue
            dxs.append(dx); dys.append(dy)
        if len(dxs) < 3:                     # 配到的源太少,這個起始猜測不可信
            return np.nan, np.nan, len(dxs), np.nan, np.nan, len(ref_fit)
        mx, my = float(np.median(dxs)), float(np.median(dys))
        sx, sy = float(np.std(dxs)), float(np.std(dys))
        gx, gy = gx + mx, gy + my
        if max(abs(mx), abs(my)) < 0.02:
            break
    return gx, gy, len(dxs), sx, sy, len(ref_fit)


# ---------------------------------------------------------------- 驗證工具
def synthetic_check():
    """把參考圖平移已知量再裁形狀,看方法能不能還原 —— 抓約定錯誤與形狀偏差。"""
    ref = _white_images(only_ref=True)["REF"]
    truths = [(0.0, -17.7), (-5.9, -10.6), (2.5, -1.2), (-3.25, 7.75), (-2.8, -1.2),
              (0.0, 0.0), (-6.0, -25.0), (4.0, -40.0), (-1.0, 3.0), (-8.4, -12.9)]
    print("合成測試:已知位移 -> 遮罩版互相關 / 未遮罩相位相關")
    print(f"{'真值(ox,oy)':>16} {'形狀':>9} {'masked':>20} {'phase(不採用)':>24}")
    e_m, e_p = [], []
    for oy, ox in truths:
        for shp in [(306, 315), (315, 306)]:
            a = np.where(np.isfinite(ref), ref, np.nanmedian(ref))
            # ndimage.shift(a, s) 給的是 out[i] = a[i-s],所以 out[y+oy, x+ox] = ref[y, x],
            # 正好就是本檔的約定 —— 平移量直接餵 (oy, ox) 即可
            F = ndimage.shift(a, (oy, ox), order=3, mode="constant", cval=np.nan)
            fake = np.full(shp, np.nan)
            ny, nx = min(shp[0], F.shape[0]), min(shp[1], F.shape[1])
            fake[:ny, :nx] = F[:ny, :nx]
            fake[:6] = fake[-6:] = np.nan
            fake[:, :6] = fake[:, -6:] = np.nan

            mx_, my_ = coarse_offset(ref, fake)
            A, B = _pad(ref), _pad(fake)
            mA, mB = np.isfinite(A), np.isfinite(B)
            sh, _, _ = phase_cross_correlation(
                np.where(mA, A - A[mA].mean(), 0.0), np.where(mB, B - B[mB].mean(), 0.0),
                upsample_factor=100, normalization="phase")
            px_, py_ = -float(sh[1]), -float(sh[0])
            em = max(abs(mx_ - ox), abs(my_ - oy)); e_m.append(em)
            ep = max(abs(px_ - ox), abs(py_ - oy)); e_p.append(ep)
            print(f"({ox:+7.2f},{oy:+6.2f}) {shp[0]}x{shp[1]} "
                  f"({mx_:+7.1f},{my_:+6.1f}) e={em:5.2f} "
                  f"({px_:+9.2f},{py_:+8.2f}) e={ep:7.2f}")
    print(f"\n  masked : 最大誤差 {max(e_m):.2f} px, >1px 的次數 {sum(e > 1 for e in e_m)}/{len(e_m)}")
    print(f"  phase  : 最大誤差 {max(e_p):.2f} px, >1px 的次數 {sum(e > 1 for e in e_p)}/{len(e_p)}")
    print("  -> 只有遮罩版通過;純相位相關會偶發大錯,不可單獨採信。")


def grid_check(imgs, answers, half=100, step=0.5, tol=1.0):
    """在 +-half px 的整個平面上數配對數,證明答案是唯一且突出的全域峰。"""
    ref_pts = detect(imgs["REF"])
    tree = cKDTree(ref_pts)
    g = np.arange(-half, half + step / 2, step)
    for name, (ox, oy) in answers.items():
        pts = detect(imgs[name])
        hits = []
        for gy in g:
            for gx in g:
                d, _ = tree.query(pts - np.array([gy, gx]), distance_upper_bound=tol)
                n = int(np.isfinite(d).sum())
                if n:
                    hits.append((n, gy, gx))
        hits.sort(key=lambda t: -t[0])
        peaks = []                                   # 只留彼此相距 >3 px 的獨立峰
        for n, gy, gx in hits:
            if all(max(abs(gy - p[1]), abs(gx - p[2])) > 3 for p in peaks):
                peaks.append((n, gy, gx))
            if len(peaks) == 4:
                break
        d, _ = tree.query(pts - np.array([oy, ox]), distance_upper_bound=tol)
        ok = np.isfinite(d)
        print(f"{name}: 解 ({ox:+.3f},{oy:+.3f}) 配到 {ok.sum()} 顆,"
              f"殘差中位 {np.median(d[ok]):.3f} px | 全域前四峰 "
              + ", ".join(f"n={n}@({gx:+.1f},{gy:+.1f})" for n, gy, gx in peaks))


# ---------------------------------------------------------------- 影像與主流程
def _white_images(lo=5000.0, hi=7000.0, only_ref=False):
    """線外通道的中位白光圖。用中位數是因為我們只要位置,宇宙線不該有發言權;
    只取線外通道是因為發射線的空間分布和連續譜不同,混進來會讓源的形狀失真。"""
    wl   = np.load(STEP03 / "wavelength.npy")
    lm   = np.load(STEP03 / "line_mask.npy")
    c0, c1 = int(np.searchsorted(wl, lo)), int(np.searchsorted(wl, hi))
    free = ~lm[c0:c1]
    paths = {"REF": REF}
    if not only_ref:
        for k in (1, 2, 3):
            paths[f"EXP{k}"] = EXPS / f"Haro11_NEpointing_EXP{k}_wsky.fits"
    return {k: med_image(str(p), c0, c1, free) for k, p in paths.items()}


def main():
    ap = argparse.ArgumentParser(description="求三顆曝光相對 ESO 合併品的平移量")
    ap.add_argument("--synthetic", action="store_true", help="只跑合成驗證")
    ap.add_argument("--grid", action="store_true", help="加跑全域配對數掃描")
    ap.add_argument("--lo", type=float, default=5000.0)
    ap.add_argument("--hi", type=float, default=7000.0)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    if args.synthetic:
        synthetic_check()
        return

    imgs = _white_images(args.lo, args.hi)

    # 用 seg 圖挑緊緻源當擬合探針:太小是雜訊,太大是延展源,高斯都擬不好
    seg = fits.getdata(STEP01 / "seg.fits")
    ids, cnt = np.unique(seg[seg > 0], return_counts=True)
    probes = []
    for i, c in zip(ids, cnt):
        if 30 <= c <= 400:
            y, x = np.nonzero(seg == i)
            probes.append((int(i), y.mean(), x.mean()))

    ref_pts = detect(imgs["REF"])
    print(f"參考圖 {args.lo:.0f}-{args.hi:.0f}A:DAOStarFinder {len(ref_pts)} 顆,"
          f"seg 探針 {len(probes)} 顆\n")
    print(f"{'cube':<6}{'A 粗解 (ox,oy)':>18}{'B1 投票 (ox,oy)':>22}{'n':>4}"
          f"{'B2 高斯 (ox,oy)':>22}{'n':>4}{'  散布 x/y':>14}")

    answers = {}
    for k in (1, 2, 3):
        name = f"EXP{k}"
        img = imgs[name]
        cx, cy = coarse_offset(imgs["REF"], img)                    # 階段 A
        vx, vy, vn, _, _ = vote_offset(ref_pts, detect(img), (cx, cy))   # 階段 B1
        gx, gy, gn, sx, sy, _ = gauss_offset(imgs["REF"], img, probes, (cx, cy))  # B2
        answers[name] = (gx, gy)
        print(f"{name:<6}({cx:+8.0f},{cy:+6.0f}) ({vx:+9.3f},{vy:+8.3f}){vn:>4}"
              f" ({gx:+9.3f},{gy:+8.3f}){gn:>4}   {sx:.3f}/{sy:.3f}"
              f"   +-({sx / max(gn, 1) ** .5:.2f},{sy / max(gn, 1) ** .5:.2f})")

    print("\n三個估計器彼此一致(<0.2 px)才算數;A 是整數,只用來把 B 帶到對的峰上。")
    if args.grid:
        print()
        grid_check(imgs, answers)


if __name__ == "__main__":
    main()
