"""曝光之間到底差多少 —— 用源的位置量,不用整幅影像的互相關。

白光圖互相關是「整幅對整幅」,結果會被視場裡最大的東西主導。天空連續譜在空間上
幾乎是平的,是一個巨大的正底盤;`white_image` 雖然扣了全場中位又截掉負值,截負
之後剩下的仍然是一張處處非負的圖,而未正規化的互相關會偏向「重疊面積最大」的
位移 —— 兩張圖尺寸不同、邊緣有缺值時,這個偏差還會隨圖的形狀而變。

源的位置沒有這個問題:對每一顆星做二維高斯擬合,中心就是中心,和視場多大、
邊緣缺多少都無關。做法:

    ① 在 ESO 合併品上,對每個緊緻的源擬合出中心 (y_ref, x_ref)
    ② 用粗略位移把該源在某顆曝光上的預期位置算出來,在那裡再擬合一次
    ③ 殘差 = 實測 − 預期,就是粗略位移還差多少
    ④ 取所有源的中位數 —— 單一顆源可能因為鄰居汙染而偏掉,中位數不會

擬合的是「高斯 + 平面背景」。背景放進擬合而不是先扣掉:源附近的天空有梯度,
先用環狀中位扣掉會在有梯度的地方把中心拉走。

    conda run -n astro python src/skymodel/experiments/step5b_align_check.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT   = Path(__file__).resolve().parents[3]
STEP01 = ROOT / "results/skymodel/ne_pointing/step01"
STEP03 = ROOT / "results/skymodel/ne_pointing/step03"
EXPS   = ROOT / "data/NEpointing_exps"

R_FIT = 7          # 擬合窗半邊長 [px] —— 約 3 個 seeing FWHM 的一半


def med_image(path, c0, c1, free, step=3):
    """線外通道的中位影像。中位數不受宇宙線影響,而我們只要位置。"""
    idx = np.flatnonzero(free)[::step] + c0
    with fits.open(path, memmap=True) as h:
        d = np.asarray(h["DATA"].data[idx.min():idx.max() + 1], np.float64)
        d = d[idx - idx.min()]
    with np.errstate(invalid="ignore"):
        return np.nanmedian(d, axis=0)


def fit_gauss(img, yc, xc):
    """在 (yc, xc) 附近擬合「二維高斯 + 平面背景」,回傳 (y, x, fwhm, amp)。

    背景用平面而不是常數:源附近的天空常有梯度,用常數的話梯度會被算進高斯裡,
    把中心往梯度高的一側拉。
    """
    y0, x0 = int(round(yc)) - R_FIT, int(round(xc)) - R_FIT
    if y0 < 0 or x0 < 0 or y0 + 2 * R_FIT + 1 > img.shape[0] \
            or x0 + 2 * R_FIT + 1 > img.shape[1]:
        return (np.nan,) * 4
    s = img[y0:y0 + 2 * R_FIT + 1, x0:x0 + 2 * R_FIT + 1]
    Y, X = np.mgrid[0:s.shape[0], 0:s.shape[1]]
    g = np.isfinite(s)
    if g.sum() < 0.7 * s.size:
        return (np.nan,) * 4

    lo = np.nanpercentile(s[g], 10)
    p0 = [float(np.nanmax(s[g]) - lo), float(R_FIT), float(R_FIT), 2.0,
          float(lo), 0.0, 0.0]

    def model(p):
        amp, my, mx, sg, b0, by, bx = p
        return (amp * np.exp(-((Y - my) ** 2 + (X - mx) ** 2) / (2 * sg ** 2))
                + b0 + by * (Y - R_FIT) + bx * (X - R_FIT))

    try:
        r = least_squares(lambda p: (model(p) - s)[g], p0,
                          bounds=([0, 1, 1, 0.5, -np.inf, -np.inf, -np.inf],
                                  [np.inf, 2 * R_FIT - 1, 2 * R_FIT - 1, 6,
                                   np.inf, np.inf, np.inf]),
                          max_nfev=2000)
    except Exception:
        return (np.nan,) * 4
    amp, my, mx, sg = r.x[:4]
    return y0 + my, x0 + mx, 2.3548 * sg, amp


def main():
    ap = argparse.ArgumentParser(description="用源的位置量曝光之間的位移")
    ap.add_argument("--ref", default=str(ROOT / "data/Haro11_NEpointing_wsky.fits"))
    ap.add_argument("--guess", nargs="+", default=[
        "EXP1=-17.885,0.030", "EXP2=-12.675,-6.263", "EXP3=-7.991,-5.350"],
        help="NAME=ox,oy 粗略位移")
    ap.add_argument("--also", nargs="+", default=[], help="NAME=PATH 另外要量的 cube")
    ap.add_argument("--iters", type=int, default=5, help="迭代幾輪")
    ap.add_argument("--lo", type=float, default=5000.0)
    ap.add_argument("--hi", type=float, default=7000.0)
    args = ap.parse_args()

    wl  = np.load(STEP03 / "wavelength.npy")
    lm  = np.load(STEP03 / "line_mask.npy")
    seg = fits.getdata(STEP01 / "seg.fits")
    ny, nx = seg.shape
    c0, c1 = int(np.searchsorted(wl, args.lo)), int(np.searchsorted(wl, args.hi))
    free = ~lm[c0:c1]

    ids, cnt = np.unique(seg[seg > 0], return_counts=True)
    probes = []
    for i, c in zip(ids, cnt):
        if not (30 <= c <= 400):          # 太小=雜訊,太大=延展源,高斯擬不好
            continue
        y, x = np.nonzero(seg == i)
        probes.append((int(i), y.mean(), x.mean()))

    R = med_image(args.ref, c0, c1, free)
    ref = {}
    for i, yc, xc in probes:
        f = fit_gauss(R, yc, xc)
        if np.isfinite(f[0]) and 2.0 < f[2] < 12.0:
            ref[i] = f
    print(f"參考 {Path(args.ref).name}:{len(ref)} 個源擬合成功")
    print(f"  FWHM 中位 {np.median([v[2] for v in ref.values()]):.3f} px\n")

    targets = [(n.split("=")[0], EXPS / f"Haro11_NEpointing_{n.split('=')[0]}_wsky.fits",
                *map(float, n.split("=")[1].split(",")))
               for n in args.guess]
    targets += [(a.split("=")[0], Path(a.split("=", 1)[1]), 0.0, 0.0)
                for a in args.also]

    print(f"{'cube':<12}{'n':>4}{'d_ox':>9}{'d_oy':>9}{'散布 x':>9}{'散布 y':>9}"
          f"{'FWHM':>8}{'  修正後的位移'}")
    for nm, path, gx, gy in targets:
        if not path.exists():
            print(f"{nm:<12} 跳過,檔案不存在"); continue
        img = med_image(path, c0, c1, free)
        # 迭代:擬合窗只有 +-7 px,起始猜測若偏了 3 px,窗就沒把源擺在中間,
        # 擬合會抓到雜訊或旁邊的源。先粗解一次、把窗移正、再解一次。
        for it in range(args.iters):
            dxs, dys, fw = [], [], []
            for i, (ry, rx, _, _) in ref.items():
                # 該源在這顆曝光上的預期位置:ref(y,x) <- cube(y+oy, x+ox)
                f = fit_gauss(img, ry + gy, rx + gx)
                if not np.isfinite(f[0]) or not (2.0 < f[2] < 12.0):
                    continue
                dy, dx = f[0] - (ry + gy), f[1] - (rx + gx)
                # 落在擬合窗邊緣的解不是真的收斂到源上,是被邊界擋住,丟掉
                if max(abs(dy), abs(dx)) > R_FIT - 2:
                    continue
                dys.append(dy); dxs.append(dx); fw.append(f[2])
            if len(dxs) < 3:
                break
            mx, my = float(np.median(dxs)), float(np.median(dys))
            gx, gy = gx + mx, gy + my
            if max(abs(mx), abs(my)) < 0.02:
                break
        if len(dxs) < 3:
            print(f"{nm:<12} 只有 {len(dxs)} 個源擬合成功,不可靠"); continue
        print(f"{nm:<12}{len(dxs):>4}{mx:>+9.3f}{my:>+9.3f}"
              f"{np.std(dxs):>9.3f}{np.std(dys):>9.3f}{np.median(fw):>8.3f}"
              f"   ({gx:+.3f}, {gy:+.3f})  [{it + 1} 輪]")


if __name__ == "__main__":
    main()
