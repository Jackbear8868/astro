"""s(p) 能不能用一個平滑的二維曲面描述、並外推到源的底下?

源底下沒有天空可量,固定 s 只能靠周圍 blank 預測。所以真正要測的是
**預測能力**,不是擬合能力 —— 在 blank 區挖出和真實源一樣大的洞,
用洞外的資料擬合,再去預測洞裡的值。

比較幾種模型:
    常數        s = 全場中位數
    多項式 1-5  二維多項式曲面
    列中位數    每一列(IFU slice 方向)各自一個值

    conda run -n astro python src/skymodel/experiments/s_surface.py
"""
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP05  = ROOT / "results/skymodel/step05"
FIGURES = ROOT / "results/skymodel/figures"
TAG    = "svd_s_free"     # step05 的輸出 tag = {basis}_s_free 或 {basis}_s_{值}
SEED   = 0
N_HOLE = 60          # 挖幾個洞
R_HOLE = 5           # 洞的半徑,對應約 80 px 的源


def scale(a):
    """穩健的散布估計。s(p) 有少數被邊界夾住的極端值,np.std 會被拉高兩倍以上。"""
    a = a[np.isfinite(a)]
    return float((np.percentile(a, 84) - np.percentile(a, 16)) / 2)


def design(x, y, order):
    """二維多項式的設計矩陣,座標已正規化到 [-1, 1]。"""
    cols = [np.ones_like(x)]
    for o in range(1, order + 1):
        for i in range(o + 1):
            cols.append(x ** (o - i) * y ** i)
    return np.column_stack(cols)


def main():
    s     = np.load(STEP05 / f"s_map_{TAG}.npy").astype(np.float64)
    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    blank = (white != 0) & (seg == 0) & np.isfinite(s)
    ny, nx = s.shape

    yy, xx = np.mgrid[0:ny, 0:nx]
    xn = (xx - nx / 2) / (nx / 2)
    yn = (yy - ny / 2) / (ny / 2)

    # 挖洞:大小與真實源相當,位置隨機且完全落在 blank 內
    rng = np.random.default_rng(SEED)
    hole = np.zeros_like(blank)
    ys, xs = np.nonzero(blank)
    n = 0
    for i in rng.permutation(ys.size):
        cy, cx = int(ys[i]), int(xs[i])
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= R_HOLE ** 2
        if not blank[m].all() or hole[m].any():
            continue
        hole |= m
        n += 1
        if n == N_HOLE:
            break
    train = blank & ~hole
    test  = blank & hole
    print(f"blank {int(blank.sum()):,} px   訓練 {int(train.sum()):,}   "
          f"留洞 {int(test.sum()):,} ({n} 個洞, 半徑 {R_HOLE} px)")
    print(f"s 的整體 1σ = {(np.percentile(s[blank],84)-np.percentile(s[blank],16))/2:.4f}\n")

    print(f"{'模型':>12}{'參數':>6}{'訓練 RMS':>12}{'留洞 RMS':>12}{'相對常數':>10}")
    print("-" * 54)
    results = {}
    base_rms = None
    for name, order in [("常數", 0), ("平面 (1階)", 1), ("2 階", 2),
                        ("3 階", 3), ("4 階", 4), ("5 階", 5)]:
        M = design(xn[train], yn[train], order)
        coef, *_ = np.linalg.lstsq(M, s[train], rcond=None)
        pred_tr = M @ coef
        pred_te = design(xn[test], yn[test], order) @ coef
        rms_tr = scale(s[train] - pred_tr)
        rms_te = scale(s[test] - pred_te)
        if base_rms is None:
            base_rms = rms_te
        print(f"{name:>12}{M.shape[1]:>6}{rms_tr:>12.5f}{rms_te:>12.5f}"
              f"{rms_te/base_rms:>10.3f}")
        results[name] = (coef, order, rms_te)

    # 列中位數:IFU slice 沿 x 方向,天空的變化可能主要在 y
    row = np.full(ny, np.nan)
    for j in range(ny):
        m = train[j]
        if m.sum() > 20:
            row[j] = np.median(s[j][m])
    good = np.isfinite(row)
    row_f = np.interp(np.arange(ny), np.arange(ny)[good], row[good])
    rms_te_row = scale(s[test] - row_f[yy[test]])
    print(f"{'列中位數':>12}{ny:>6}{'—':>12}{rms_te_row:>12.5f}{rms_te_row/base_rms:>10.3f}")

    # 逐 spaxel 的雜訊下限:相鄰列差分,√2 修正
    d = np.diff(s, axis=0)
    m = blank[1:] & blank[:-1]
    noise = scale(d[m]) / np.sqrt(2)
    print(f"\n逐 spaxel 雜訊下限(相鄰列差分) ≈ {noise:.5f}")
    print("→ 留洞 RMS 若接近這個值,表示曲面已經抓住了全部可預測的結構。")

    # ---------------- 圖 ----------------
    best = min(("2 階", "3 階", "4 階", "5 階"), key=lambda k: results[k][2])
    coef, order, _ = results[best]
    surf = (design(xn.ravel(), yn.ravel(), order) @ coef).reshape(ny, nx)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    lo, hi = np.percentile(s[blank], [1, 99])
    im = ax[0].imshow(np.where(blank, s, np.nan), origin="lower", cmap="RdBu_r",
                      vmin=lo, vmax=hi)
    plt.colorbar(im, ax=ax[0]); ax[0].set_title("measured $s(p)$  (blank only)")
    im = ax[1].imshow(np.where(white != 0, surf, np.nan), origin="lower",
                      cmap="RdBu_r", vmin=lo, vmax=hi)
    ax[1].contour(seg > 0, levels=[.5], colors="k", linewidths=.4)
    plt.colorbar(im, ax=ax[1]); ax[1].set_title(f"fitted surface  ({best})")
    r = np.where(blank, s - surf, np.nan)
    im = ax[2].imshow(r, origin="lower", cmap="RdBu_r", vmin=-3*noise, vmax=3*noise)
    plt.colorbar(im, ax=ax[2]); ax[2].set_title("residual  (measured − surface)")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = FIGURES / "s_surface.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {out.name}")


if __name__ == "__main__":
    main()
