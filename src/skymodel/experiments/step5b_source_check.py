"""合併有沒有傷到源 —— 銳利度、位置、流量、光譜,四項一起看。

step5b_bias_check 量的是「殘差乾不乾淨」,step5b_stat_check 量的是「雜訊對不對」。
兩者都不會告訴我們源有沒有被弄壞。取整位移的代價正是空間上的對位誤差,那個代價
必須直接量出來,不能只看殘差(殘差變好也可能是源被抹掉了)。

四項指標,都拿 ESO 官方合併品當基準:

    FWHM        銳利度。用背景扣除後的二階矩,FWHM = 2.3548·sqrt(a·b)
                (和 CLAUDE.md 的慣例一致)。內插會把源抹開 -> FWHM 變大。
    r50         半光半徑,對 PSF 翼區比二階矩不敏感,當作交叉檢查。
    centroid    源的位置。取整位移的誤差會直接寫在這裡。
    flux        固定孔徑內的總流量。抹開不該改變總流量,掉了就是真的丟了光。

外加逐波長的光譜比:把整個源區加總,ours/ESO 對波長畫出來。空間上的小差異
可能在光譜上完全看不見,也可能不是 —— 這一項才是我們真正在乎的東西。

    conda run -n astro python src/skymodel/experiments/step5b_source_check.py
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/ne_pointing/step01"
STEP03  = ROOT / "results/skymodel/ne_pointing/step03"
STEP05B = ROOT / "results/skymodel/ne_pointing/step05b"
FIGURES = ROOT / "results/skymodel/evaluation/subtraction_check"

R_STAMP = 10        # 量測用的方形半邊長 [px]
R_APER  = 8.0       # 孔徑半徑 [px] —— 約 2 個 seeing FWHM
R_BG    = (11, 14)  # 背景環 [px]


def white(path, c0, c1, free, chunk=200):
    """線外通道的平均影像。用平均而不是中位數 —— 這裡要的是流量,不是「哪裡亮」。"""
    with fits.open(path, memmap=True) as h:
        ny, nx = h["DATA"].data.shape[1:]
        s = np.zeros((ny, nx)); n = np.zeros((ny, nx))
        for j in range(c0, c1, chunk):
            k = min(j + chunk, c1)
            f = free[j - c0:k - c0]
            if not f.any():
                continue
            d = np.asarray(h["DATA"].data[j:k][f], np.float64)
            g = np.isfinite(d)
            s += np.where(g, d, 0.0).sum(axis=0); n += g.sum(axis=0)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def measure(img, yc, xc):
    """回傳 (fwhm, r50, dy, dx, flux)。dy/dx 是相對於給定中心的位移。"""
    y0, y1 = int(yc) - R_STAMP, int(yc) + R_STAMP + 1
    x0, x1 = int(xc) - R_STAMP, int(xc) + R_STAMP + 1
    st = img[y0:y1, x0:x1]
    Y, X = np.mgrid[y0:y1, x0:x1]

    # 局部背景:環內的中位數。天空連續譜是個大底盤,不扣掉的話二階矩會被
    # 整個方框主導,量到的是方框的大小而不是源的大小。
    r = np.hypot(Y - yc, X - xc)
    ann = np.isfinite(st) & (r >= R_BG[0]) & (r < R_BG[1])
    if ann.sum() < 20:
        return (np.nan,) * 5
    s = st - np.median(st[ann])

    m = np.isfinite(s) & (r < R_APER)
    w = np.where(m, np.maximum(s, 0.0), 0.0)     # 截負:負值的權重沒有意義
    tot = w.sum()
    if tot <= 0:
        return (np.nan,) * 5
    cy, cx = (w * Y).sum() / tot, (w * X).sum() / tot

    # 二階矩要以自己的形心為準,不是以輸入的中心 —— 否則位置偏移會被
    # 算進「變寬」裡,兩個指標就分不開了。
    a = (w * (Y - cy) ** 2).sum() / tot
    b = (w * (X - cx) ** 2).sum() / tot
    fwhm = 2.3548 * np.sqrt(np.sqrt(max(a, 0) * max(b, 0)))

    rr = np.hypot(Y - cy, X - cx)[m].ravel()
    vv = w[m].ravel()
    o = np.argsort(rr)
    cum = np.cumsum(vv[o])
    r50 = float(np.interp(0.5 * cum[-1], cum, rr[o]))
    return fwhm, r50, cy - yc, cx - xc, float(np.nansum(np.where(m, s, 0.0)))


def spectrum(path, mask, chunk=200):
    """源區加總的光譜。"""
    idx = np.flatnonzero(mask.ravel())
    with fits.open(path, memmap=True) as h:
        nz = h["DATA"].header["NAXIS3"]
        out = np.full(nz, np.nan)
        for j in range(0, nz, chunk):
            k = min(j + chunk, nz)
            d = np.asarray(h["DATA"].data[j:k], np.float64).reshape(k - j, -1)[:, idx]
            with np.errstate(invalid="ignore"):
                out[j:k] = np.nanmean(d, axis=1)
    return out


def main():
    ap = argparse.ArgumentParser(description="合併有沒有傷到源")
    ap.add_argument("--cubes", nargs="+", default=[
        f"ESO={ROOT}/data/Haro11_NEpointing_wsky.fits",
        f"bilinear={STEP05B}/merged_EXP123_wsky_bil_norej.fits",
        f"integer={STEP05B}/merged_EXP123_wsky_int_rej.fits",
    ], help="NAME=PATH,第一個當基準")
    ap.add_argument("--lo", type=float, default=5000.0)
    ap.add_argument("--hi", type=float, default=7000.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    wl  = np.load(STEP03 / "wavelength.npy")
    lm  = np.load(STEP03 / "line_mask.npy")
    seg = fits.getdata(STEP01 / "seg.fits")
    ny, nx = seg.shape
    c0, c1 = int(np.searchsorted(wl, args.lo)), int(np.searchsorted(wl, args.hi))
    free = ~lm[c0:c1]

    # 夠亮、夠緊緻、離邊界夠遠的源,拿來當 PSF 的探針
    ids, cnt = np.unique(seg[seg > 0], return_counts=True)
    probes = []
    for i, c in zip(ids, cnt):
        if not (40 <= c <= 400):
            continue
        y, x = np.nonzero(seg == i)
        yc, xc = y.mean(), x.mean()
        if min(yc, xc, ny - 1 - yc, nx - 1 - xc) < R_BG[1] + 2:
            continue
        probes.append((int(i), yc, xc))
    probes.sort(key=lambda t: -cnt[list(ids).index(t[0])])
    probes = probes[:8]
    print(f"探針源 {len(probes)} 個: " + ", ".join(f"#{i}" for i, _, _ in probes))
    print(f"通道 {c0}-{c1},無天光線 {int(free.sum())}\n")

    names = [c.split("=", 1)[0] for c in args.cubes]
    paths = [c.split("=", 1)[1] for c in args.cubes]
    imgs  = {}
    for nm, p in zip(names, paths):
        if not Path(p).exists():
            print(f"跳過 {nm}:{p} 不存在"); continue
        imgs[nm] = white(p, c0, c1, free)
        print(f"{nm:<10} 白光圖完成", flush=True)

    tab = {nm: {} for nm in imgs}
    for i, yc, xc in probes:
        for nm in imgs:
            tab[nm][i] = measure(imgs[nm], yc, xc)

    ref = names[0]
    print(f"\n{'':<12}" + "".join(f"{f'#{i}':>9}" for i, _, _ in probes) + f"{'中位':>9}")
    for lab, k, fmt in (("FWHM [px]", 0, "{:9.3f}"), ("r50 [px]", 1, "{:9.3f}"),
                        ("|shift| px", None, "{:9.3f}"), ("flux/ESO", 4, "{:9.4f}")):
        print(f"  {lab}")
        for nm in imgs:
            if k is None:
                v = [np.hypot(tab[nm][i][2], tab[nm][i][3]) for i, _, _ in probes]
            elif lab.startswith("flux"):
                v = [tab[nm][i][4] / tab[ref][i][4] for i, _, _ in probes]
            else:
                v = [tab[nm][i][k] for i, _, _ in probes]
            print(f"    {nm:<10}" + "".join(fmt.format(x) for x in v)
                  + fmt.format(np.nanmedian(v)))

    # 位置:相對基準的偏移(基準自己的形心當原點)
    print("  centroid shift vs ESO [px]")
    for nm in imgs:
        d = [np.hypot(tab[nm][i][2] - tab[ref][i][2], tab[nm][i][3] - tab[ref][i][3])
             for i, _, _ in probes]
        print(f"    {nm:<10}" + "".join(f"{x:9.3f}" for x in d)
              + f"{np.nanmedian(d):9.3f}")

    # ---- 光譜 ----
    specs = {}
    for nm, p in zip(names, paths):
        if nm not in imgs:
            continue
        specs[nm] = {t: spectrum(p, seg == t) for t in (1, 35)}
        print(f"{nm:<10} 光譜完成", flush=True)

    print("\n光譜比 ours/ESO (線外通道)")
    for t in (1, 35):
        for nm in specs:
            if nm == ref:
                continue
            g = np.zeros(len(wl), bool); g[c0:c1] = free
            r = specs[nm][t][g] / specs[ref][t][g]
            r = r[np.isfinite(r)]
            print(f"  #{t:<3} {nm:<10} 中位 {np.median(r):.5f}   "
                  f"16-84% [{np.percentile(r, 16):.5f}, {np.percentile(r, 84):.5f}]")

    # ---------------- 圖 ----------------
    col = {"ESO": "#000000", "bilinear": "#d62728", "integer": "#1f77b4"}
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))

    a = ax[0][0]
    xs = np.arange(len(probes))
    for nm in imgs:
        a.plot(xs, [tab[nm][i][0] for i, _, _ in probes], "o-",
               color=col.get(nm), label=nm, lw=1.2)
    a.set_xticks(xs); a.set_xticklabels([f"#{i}" for i, _, _ in probes])
    a.set_ylabel("FWHM [pix]"); a.set_title("sharpness (lower = better)", fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.25)

    a = ax[0][1]
    for nm in imgs:
        if nm == ref:
            continue
        a.plot(xs, [np.hypot(tab[nm][i][2] - tab[ref][i][2],
                             tab[nm][i][3] - tab[ref][i][3]) for i, _, _ in probes],
               "o-", color=col.get(nm), label=nm, lw=1.2)
    a.axhline(0, color="0.5", lw=0.8)
    a.set_xticks(xs); a.set_xticklabels([f"#{i}" for i, _, _ in probes])
    a.set_ylabel("|centroid shift| vs ESO [pix]")
    a.set_title("astrometry cost", fontsize=10); a.legend(fontsize=8); a.grid(alpha=0.25)

    for k, t in enumerate((1, 35)):
        a = ax[1][k]
        for nm in specs:
            if nm == ref:
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                r = specs[nm][t] / specs[ref][t]
            a.plot(wl, r, lw=0.5, color=col.get(nm), label=nm, alpha=0.8)
        a.axhline(1, color="k", lw=0.8)
        a.set_ylim(0.9, 1.1)
        a.set_xlabel("wavelength [$\\AA$]"); a.set_ylabel("flux ratio ours / ESO")
        a.set_title(f"summed spectrum of #{t}"
                    f"{' (Haro 11)' if t == 1 else ' (star)'}", fontsize=10)
        a.legend(fontsize=8); a.grid(alpha=0.25)

    fig.suptitle("is the source preserved by our merge?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = Path(args.out) if args.out else FIGURES / "step5b_source_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
