"""【實驗】形態判定:每個 seg 區域是點源(恆星/QSO)還是延展天體(星系)?

這是探索性的實驗,不是 reminder ① 的結案。

為什麼需要形態:早型星系的光譜本質上就是幾十億顆 F/G/K 恆星的疊加,所以
「一顆 G 星」和「一個 z≈0 的早型星系」在光譜上近乎簡併。chi2 分不出來,
形態分得出來。

用聚集度指數,不用 PSF 擬合:

    C = 半徑 2 px 內的流量 / 半徑 5 px 內的流量

理由 —— 這個場裡沒有已知的參考點源,所以任何需要「先知道 PSF」的方法(高斯
或 Moffat 擬合)都會被異常值帶偏。

C 不需要模型也不需要外部基準:
    點源的 C 由 PSF 決定,而且是「可能的最大值」—— 沒有東西比 PSF 更集中
    延展天體的 C 一定更小
所以點源會在「C vs 亮度」圖上聚成一條水平的上緣。

其他兩個處理:
    鄰近源的像素(屬於別的 seg ID)排除,避免汙染孔徑
    背景用 8–10 px 的環帶中位數扣掉,誤差由白光的雜訊圖傳遞
      (白光 = nz 個通道的平均 → Var = sum(Var) / nz²)

    conda run -n astro python src/skymodel/experiments/psf_morphology.py
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[3]

FIGURES = ROOT / "results/skymodel/evaluation/masking"
WSKY    = ROOT / "data/Haro11_NEpointing_wsky.fits"

R_IN, R_OUT = 2.0, 5.0      # 兩個孔徑半徑(px);PSF FWHM 約 4 px
BG_IN, BG_OUT = 8.0, 12.0   # 背景環帶,已在 PSF 的翼之外
HALF = 13                   # 小塊半寬,要能容下 BG_OUT


def concentration(img, err, own, cy, cx):
    """回傳 (C, C 的 1sigma 誤差, 峰值/雜訊)。own 標出屬於這個源的鄰域遮罩。"""
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    use = np.isfinite(img) & np.isfinite(err) & (err > 0) & own

    ring = use & (r >= BG_IN) & (r < BG_OUT)
    if ring.sum() < 15:
        return np.nan, np.nan, np.nan
    bg = float(np.median(img[ring]))
    y = img - bg

    a1, a2 = use & (r <= R_IN), use & (r <= R_OUT)
    if a1.sum() < 8 or a2.sum() < 40:
        return np.nan, np.nan, np.nan
    f1, f2 = float(y[a1].sum()), float(y[a2].sum())
    e1 = float(np.sqrt((err[a1] ** 2).sum()))
    e2 = float(np.sqrt((err[a2] ** 2).sum()))
    if f2 <= 0:
        return np.nan, np.nan, np.nan
    c = f1 / f2
    # f1 是 f2 的一部分,兩者相關;取保守的獨立近似即可(只用來排序與畫圖)
    dc = abs(c) * np.sqrt((e1 / f1) ** 2 + (e2 / f2) ** 2) if f1 != 0 else np.nan
    return c, dc, f2 / e2


def main():
    ap = argparse.ArgumentParser(description="【實驗】seg 區域的形態判定(聚集度)")
    ap.add_argument("--work", default="results/skymodel/p01",
                    help="這顆 pointing 的工作區(底下有 step01/step03/step04)")

    args = ap.parse_args()

    W = ROOT / args.work
    STEP01, STEP04 = W / "step01", W / "step04"

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    ny, nx = white.shape

    with fits.open(WSKY, memmap=True) as h:
        V = h["STAT"].data
        nz = V.shape[0]
        acc = np.zeros((ny, nx), np.float64)
        cnt = np.zeros((ny, nx), np.int64)
        for j in range(0, nz, 200):
            blk = np.asarray(V[j:j + 200], np.float64)
            m = np.isfinite(blk) & (blk > 0)
            acc += np.where(m, blk, 0).sum(axis=0)
            cnt += m.sum(axis=0)
    err = np.where(cnt > 0, np.sqrt(acc) / np.maximum(cnt, 1), np.nan)

    bf = sorted(STEP04.glob("best_*.npz"))
    if not bf:
        raise SystemExit(f"{STEP04} 裡沒有 best_*.npz,先跑 step4_fit_source.py")
    best = np.load(bf[0])
    info = {int(i): (str(g), float(z)) for i, g, z in
            zip(best["id"], best["group"], best["z"])}

    rows = []
    for t in np.unique(seg[seg > 0]):
        m = seg == t
        f = np.clip(white[m], 0, None)
        if f.sum() <= 0:
            continue
        ys, xs = np.nonzero(m)
        y0 = int(round((ys * f).sum() / f.sum()))
        x0 = int(round((xs * f).sum() / f.sum()))
        d = min(y0, ny - 1 - y0, x0, nx - 1 - x0, HALF)
        if d < BG_OUT:
            continue                        # 環帶取不完整
        sl = (slice(y0 - d, y0 + d + 1), slice(x0 - d, x0 + d + 1))
        sub = seg[sl]
        own = (sub == t) | (sub == 0)       # 排除鄰近的其他源
        c, dc, snr = concentration(white[sl], err[sl], own, d, d)
        g, z = info.get(int(t), ("?", np.nan))
        rows.append((int(t), int(m.sum()), g, z, c, dc, snr))

    ok = [r for r in rows if np.isfinite(r[4]) and r[6] > 10]
    cmax = max(r[4] for r in ok) if ok else np.nan

    print("【實驗】不是 reminder ① 的結案\n")
    print(f"孔徑 r <= {R_IN:.0f} px / r <= {R_OUT:.0f} px,背景環帶 {BG_IN:.0f}-{BG_OUT:.0f} px")
    print(f"擬合成功 {len([r for r in rows if np.isfinite(r[4])])}"
          f"/{len(np.unique(seg[seg>0]))} 個區域")
    print(f"最高的 C = {cmax:.3f} —— 沒有東西能比 PSF 更集中,所以這是點源的值\n")
    print(f"{'ID':>4}{'nspax':>7}{'group':>8}{'z':>9}{'孔徑 S/N':>10}"
          f"{'C':>8}{'誤差':>8}{'C/Cmax':>9}{'低於 sigma':>11}  判定")
    print("-" * 88)
    for t, n, g, z, c, dc, snr in sorted(rows, key=lambda r: (np.isnan(r[4]), -r[4])):
        if not np.isfinite(c):
            print(f"{t:>4}{n:>7}{g:>8}{z:>9.4f}{'—':>10}   (孔徑或環帶取不完整)")
            continue
        k = (cmax - c) / dc if np.isfinite(dc) and dc > 0 else np.nan
        v = ("點源" if k < 3 else ("延展" if k > 5 else "邊界")) if snr > 10 else "S/N 太低"
        print(f"{t:>4}{n:>7}{g:>8}{z:>9.4f}{snr:>10.1f}"
              f"{c:>8.3f}{dc:>8.3f}{c/cmax:>9.2f}{k:>11.1f}  {v}")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, a = plt.subplots(figsize=(10, 6.5))
    for g, col in [("galaxy", "#1f77b4"), ("qso", "#d62728"), ("star", "#2ca02c")]:
        s = [(r[6], r[4], r[5], r[0]) for r in rows if r[2] == g and np.isfinite(r[4])]
        if s:
            a.errorbar([x[0] for x in s], [x[1] for x in s], yerr=[x[2] for x in s],
                       fmt="o", c=col, label=g, ms=6, lw=0, elinewidth=1, zorder=3)
            for x, yv, _, t in s:
                a.annotate(str(t), (x, yv), fontsize=7, xytext=(5, 3),
                           textcoords="offset points")
    a.axhline(cmax, color="0.3", ls="--", lw=1.2,
              label=f"C_max = {cmax:.3f}  (point-source limit)")
    a.set_xscale("log")
    a.set_xlabel("aperture S/N in whitelight")
    a.set_ylabel(f"concentration  C = F(r<{R_IN:.0f}) / F(r<{R_OUT:.0f})")
    a.legend(fontsize=9, loc="lower right")
    a.set_title("morphology (experiment): point-like sits at C_max, "
                "extended sits below", fontsize=11)
    fig.tight_layout()
    out = FIGURES / "psf_morphology.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
