"""用固定的圓形孔徑抽源光譜,取代 SExtractor 的 segmentation footprint。

為什麼換
    step2 用的是 SExtractor 切出來的不規則形狀,形狀完全由偵測門檻決定 ——
    門檻動一點,抽出來的光譜就變。圓形孔徑是外加的、固定的:同一個半徑套在
    每個源上,和偵測門檻無關,可重現。

半徑 (--radius,預設 APERTURE_R)
    PSF 隨波長變窄,所以固定孔徑在不同波長收到的流量比例不同,會造成假的
    連續譜斜率:半徑越小,這個假斜率越大;半徑越大,孔徑收進的雜訊越多、
    S/N 越低。半徑是這兩者的折衷。

圓心 = 整數像素
    做法:背景扣除後的流量加權形心 → 四捨五入 → 用整數圓心重畫圓、再算一次
    形心 → 直到整數圓心不再移動。

    ① 背景一定要扣。白光背景的底座和源本身的淨流量同量級,不扣的話加權形心
       量到的主要是遮罩的形狀而不是源的分布。
    ② 迭代要在圓內做,不能只在 seg 遮罩內。只做後者的話圓心仍然繼承 seg 的
       形狀,而那正是我們想擺脫的東西。
    ③ 整數圓心的代價是最多 0.71 px(= √2 / 2)的捨入誤差;換到的是每個源
       完全相同的孔徑形狀與像素數,源之間直接可比。

輸出 results/skymodel/step02b/,格式和 step02 相同,step4b 只要換讀檔路徑:
    object_ids.npy / object_flux.npy / object_var.npy / object_nspax.npy
    aperture.npz    圓心、像素數、視場覆蓋率、是否有效
    aperture.txt    同一份表的純文字版

    conda run -n astro python src/skymodel/step2b_aperture.py
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP02B = ROOT / "results/skymodel/step02b"
WSKY    = ROOT / "data/Haro11_NEpointing_wsky.fits"

APERTURE_R = 6.0        # px。1 px = 0.2 arcsec,所以 6 px = 1.2 arcsec 半徑


def disc(shape, cy, cx, r):
    """以 (cy, cx) 為圓心、半徑 r 的圓內像素。

    判準是「像素中心到圓心的距離 ≤ r」,所以納入的一律是完整的像素 ——
    不會出現半個像素,不需要面積加權。
    """
    yy, xx = np.ogrid[0:shape[0], 0:shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def find_center(w, seg, t, r, max_iter=10):
    """整數圓心。回傳 (cy, cx, 迭代次數, 是否有效)。

    w 是已經扣掉背景、負值壓到 0 的白光影像 —— 權重必須非負,否則雜訊的負權重
    會把形心往任意方向拉。

    迭代到整數圓心不再移動為止。因為圓心被限制在整數格點上,這是一個有限狀態的
    迭代,收斂即是「不動點」,通常 1–2 輪就到;不收斂(在兩點之間跳)時取最後一次。
    """
    m = seg == t
    if not m.any():
        return None
    yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]

    # ---- 起點:seg 遮罩內的形心 ----
    a = w[m]
    if a.sum() > 0:
        cy, cx = (yy[m] * a).sum() / a.sum(), (xx[m] * a).sum() / a.sum()
    else:
        cy, cx = yy[m].mean(), xx[m].mean()      # 沒有訊號 → 幾何中心
    cy, cx = int(round(cy)), int(round(cx))

    # ---- 迭代:改用圓內的像素重算 ----
    for it in range(1, max_iter + 1):
        d = disc(seg.shape, cy, cx, r)
        a = w[d]
        if a.sum() <= 0:
            return cy, cx, it, False              # 圓內沒有任何訊號
        ny = int(round((yy[d] * a).sum() / a.sum()))
        nx = int(round((xx[d] * a).sum() / a.sum()))
        if (ny, nx) == (cy, cx):
            return cy, cx, it, True               # 不動點,收斂
        cy, cx = ny, nx
    return cy, cx, max_iter, True


def sum_spectra_by_mask(cube_path, masks, chunk=200):
    """對一組(可重疊的)布林遮罩各自加總光譜。

    不能用 step2 的標籤圖版本 —— 相鄰的源孔徑會重疊,標籤圖逼我們把重疊的
    像素判給其中一個,等於偷偷截掉另一個。

    flux、var、nspax 三者計數的必定是同一批 spaxel:先用 ok 遮罩把不可用的
    位置清成 0 再相加,而不是各自 nansum。
    """
    idx = [np.flatnonzero(m.ravel()) for m in masks]
    with fits.open(cube_path, memmap=True) as h:
        nz = h["DATA"].header["NAXIS3"]
        n  = masks[0].size
        flux  = np.zeros((len(masks), nz))
        var   = np.zeros((len(masks), nz))
        nspax = np.zeros((len(masks), nz))
        for j in range(0, nz, chunk):
            d = np.asarray(h["DATA"].data[j:j+chunk], np.float64).reshape(-1, n)
            v = np.asarray(h["STAT"].data[j:j+chunk], np.float64).reshape(-1, n)
            ok = np.isfinite(d) & np.isfinite(v) & (v > 0)
            d  = np.where(ok, d, 0.0)
            v  = np.where(ok, v, 0.0)
            for k, ii in enumerate(idx):
                flux[k,  j:j+chunk] = d[:, ii].sum(axis=1)
                var[k,   j:j+chunk] = v[:, ii].sum(axis=1)
                nspax[k, j:j+chunk] = ok[:, ii].sum(axis=1)
    return flux, var, nspax


def main():
    ap = argparse.ArgumentParser(description="圓形孔徑的源光譜抽取")
    ap.add_argument("-r", "--radius", type=float, default=APERTURE_R,
                    help="孔徑半徑 (px)。0.2 arcsec/px")
    args = ap.parse_args()
    STEP02B.mkdir(parents=True, exist_ok=True)

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits").astype(np.float64)
    fov   = white != 0
    seg   = np.where(fov, seg, 0)                 # 視場外一律歸 0

    # 背景取 blank 區(視場內、非源)的中位數。中位數而不是平均 —— 殘留的暗源
    # 會把平均拉高,中位數不受影響。
    bg = float(np.median(white[fov & (seg == 0)]))
    w  = np.maximum(white - bg, 0.0)              # 權重非負
    w[~fov] = 0.0

    ids = np.unique(seg[seg > 0])
    r   = args.radius
    n_full = int(disc((2 * int(r) + 3,) * 2, int(r) + 1, int(r) + 1, r).sum())
    print(f"{len(ids)} sources   半徑 {r:g} px = {r * 0.2:.2f} arcsec"
          f"   完整孔徑 {n_full} px   白光背景 {bg:.3f}\n")

    rows, masks = [], []
    for t in ids:
        cy, cx, nit, okc = find_center(w, seg, int(t), r)
        d   = disc(seg.shape, cy, cx, r)
        m   = d & fov                              # 只收視場內的像素
        cov = m.sum() / n_full
        rows.append(dict(id=int(t), cy=cy, cx=cx, n_pix=int(m.sum()),
                         n_iter=nit, covered=cov,
                         valid=bool(okc and cov > 0.999)))
        masks.append(m)

    print(f"{'ID':>4}{'圓心 y':>8}{'圓心 x':>8}{'像素':>7}{'覆蓋':>8}"
          f"{'迭代':>6}   狀態")
    print("-" * 56)
    for x in rows:
        why = ("OK" if x["valid"]
               else ("圓內無訊號" if x["n_iter"] and not x["covered"] > 0.999
                     and x["n_pix"] == 0 else
                     ("超出視場" if x["covered"] <= 0.999 else "圓內無訊號")))
        print(f"{x['id']:>4}{x['cy']:>8}{x['cx']:>8}{x['n_pix']:>7}"
              f"{100*x['covered']:>7.0f}%{x['n_iter']:>6}   {why}")

    flux, var, nspax = sum_spectra_by_mask(WSKY, masks)

    np.save(STEP02B / "object_ids.npy",   ids)
    np.save(STEP02B / "object_flux.npy",  flux)
    np.save(STEP02B / "object_var.npy",   var)
    np.save(STEP02B / "object_nspax.npy", nspax)
    np.savez(STEP02B / "aperture.npz", radius=r,
             **{k: np.array([x[k] for x in rows]) for k in
                ("id", "cy", "cx", "n_pix", "n_iter", "covered", "valid")})

    lines = [f"圓形孔徑抽取   半徑 {r:g} px = {r*0.2:.2f} arcsec   "
             f"完整孔徑 {n_full} px   白光背景 {bg:.3f}", "",
             f"{'ID':>4}{'cy':>6}{'cx':>6}{'n_pix':>7}{'covered':>9}"
             f"{'n_iter':>8}{'valid':>7}"]
    lines += [f"{x['id']:>4}{x['cy']:>6}{x['cx']:>6}{x['n_pix']:>7}"
              f"{100*x['covered']:>8.0f}%{x['n_iter']:>8}"
              f"{str(x['valid']):>7}" for x in rows]
    lines += ["", "cy, cx 是整數像素座標 —— 圓心就固定在這一格,圓由它畫出。",
              "covered = 孔徑落在視場內的比例;< 100% 表示圓有一部分在視場外。",
              "valid   = 圓內有訊號 且 孔徑完整落在視場內。"]
    (STEP02B / "aperture.txt").write_text("\n".join(lines) + "\n")

    aperture_map(white, seg, rows, r, STEP02B / "aperture_map.png")

    nv = sum(1 for x in rows if not x["valid"])
    print(f"\n有效 {len(rows) - nv} / {len(rows)}   "
          f"(無效的多半是孔徑超出視場)")
    print(f"saved -> {STEP02B}")


def aperture_map(white, seg, rows, r, out):
    """孔徑畫在白光上,圓心用十字標出來 —— 用來確認圓心沒有落在奇怪的地方。

    seg 的輪廓一起畫,才看得出「孔徑」和「原本的 segmentation」差多少。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    fig, ax = plt.subplots(figsize=(13, 12.5))
    v = np.nanpercentile(white[np.isfinite(white) & (white != 0)], 99.5)
    ax.imshow(np.arcsinh(white / (0.02 * v)), origin="lower", cmap="gray",
              vmin=0, vmax=np.arcsinh(1 / 0.02))
    ax.contour(seg > 0, levels=[0.5], colors="#7ec8ff", linewidths=0.7)
    for x in rows:
        c = "#ff7b7b" if x["valid"] else "#ffd60a"
        ax.add_patch(plt.Circle((x["cx"], x["cy"]), r, fill=False, color=c, lw=1.2))
        ax.plot(x["cx"], x["cy"], "+", color=c, ms=7, mew=1.4)
        ax.text(x["cx"], x["cy"] + r + 3, str(x["id"]), color="white", fontsize=9,
                fontweight="bold", ha="center", va="bottom",
                path_effects=[pe.withStroke(linewidth=2.4, foreground="black")])
    ax.set_title(f"circular apertures, r = {r:g} px = {r*0.2:.1f} arcsec\n"
                 "+ marks the integer pixel used as the centre;  "
                 "blue = SExtractor segmentation;  yellow = aperture not fully in FOV",
                 fontsize=12)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
