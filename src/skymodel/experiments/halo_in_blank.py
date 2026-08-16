"""Haro 11 的電離氣體暈,有多少落在我們的 blank 樣本裡?

blank 是白光影像上的 SExtractor 偵測界定的。但電離氣體暈是
「只有發射線、沒有連續譜」的結構,對寬波段白光幾乎沒有貢獻,在光譜裡卻完整
存在。若它落在 blank 裡,SVD 就可能學到帶有紅移 Ha 的成分,再從真正的源身上
減掉 —— 而殘差類的指標看不見這件事(源被扣掉時殘差反而變小)。

做法:直接量。從 ESO nosky 做一張連續譜相減的 Ha 窄帶影像 —— 用 ESO 而不是
我們自己的產物,是為了讓這個檢驗獨立於待檢驗的 pipeline。然後比較遮罩內外
的 Ha 流量。

Ha 在 z=0.0205 落在 6697 A,那裡有天光線,所以一定要用扣過天空的 cube。

    conda run -n astro python src/skymodel/experiments/halo_in_blank.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP03  = ROOT / "results/skymodel/step03"
STEP04  = ROOT / "results/skymodel/step04"
FIGURES = ROOT / "results/skymodel/figures"
ESO     = ROOT / "data/Haro11_NEpointing_esonosky.fits"
WSKY    = ROOT / "data/Haro11_NEpointing_wsky.fits"

CORE, GAP, BG = 3, 10, 35     # 線心 / 間隔 / 連續譜視窗 (px)


def main():
    ap = argparse.ArgumentParser(description="暈有多少落在 blank 裡")
    ap.add_argument("--line", type=float, default=6562.8, help="靜止波長 (A)")
    ap.add_argument("--snr", type=float, default=3.0)
    ap.add_argument("--tag", default="svd_K30_s_1.0", help="step05 的產物標籤")
    args = ap.parse_args()

    wl    = np.load(STEP03 / "wavelength.npy")
    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    best  = np.load(STEP04 / "best_svd_K30_s_1.0.npz")
    z = float(best["z"][int(np.flatnonzero(best["id"] == 1)[0])])
    lo = args.line * (1 + z)
    c  = int(np.argmin(np.abs(wl - lo)))
    print(f"Ha at z={z:.5f} → {lo:.1f} A → channel {c}")

    sl = slice(c - BG, c + BG + 1)
    with fits.open(ESO, memmap=True) as h:
        cube = np.asarray(h[1].data[sl], np.float64)
    with fits.open(WSKY, memmap=True) as h:
        var = np.asarray(h["STAT"].data[sl], np.float64)
    k = BG                                          # 線心在切片裡的位置
    sb = np.r_[np.arange(0, k - GAP), np.arange(k + GAP + 1, cube.shape[0])]
    cont = np.median(cube[sb], axis=0)
    nb   = np.nansum(cube[k - CORE:k + CORE + 1] - cont, axis=0)
    err  = np.sqrt(np.nansum(var[k - CORE:k + CORE + 1], axis=0))
    snr  = np.where(err > 0, nb / err, 0.0)

    valid  = (white != 0) & np.isfinite(nb) & np.isfinite(err) & (err > 0)
    src    = valid & (seg > 0)
    blank  = valid & (seg == 0)
    det    = blank & (snr > args.snr)

    print(f"\n有效 {int(valid.sum()):,}   遮罩內(源) {int(src.sum()):,}"
          f"   blank {int(blank.sum()):,}")
    print(f"\nHa 流量(連續譜已扣)")
    print(f"{'':>18}{'總流量':>16}{'佔全部':>10}{'spaxel 數':>11}")
    print("-" * 56)
    tot = np.nansum(nb[valid])
    for nm, m in [("遮罩內(源)", src), ("blank 全部", blank),
                  (f"blank 且 S/N>{args.snr:g}", det)]:
        f = np.nansum(nb[m])
        print(f"{nm:>18}{f:>16,.0f}{100*f/tot:>9.1f}%{int(m.sum()):>11,}")

    print(f"\nblank 裡 Ha 被偵測到(S/N>{args.snr:g})的 spaxel:"
          f"{int(det.sum()):,} / {int(blank.sum()):,} = {100*det.mean()/blank.mean():.2f}%")

    # ---- 徑向剖面:暈延伸到哪裡,遮罩到哪裡 ----
    yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
    w = np.where(src, np.maximum(nb, 0), 0)
    cy, cx = (w * yy).sum() / w.sum(), (w * xx).sum() / w.sum()
    r = np.hypot(yy - cy, xx - cx)
    print(f"\nHa 光度中心 (y,x) = ({cy:.1f}, {cx:.1f})")

    # ---- 端到端的檢驗:我們的 pipeline 有沒有把暈扣掉? ----
    # 不去猜 basis 學到了什麼,直接比同一條 Ha 的徑向剖面。ESO 的 basis 不是
    # 從這些 spaxel 學來的,所以它在這裡是獨立的參考。
    ours = ROOT / f"results/skymodel/step05/sky_subtracted_{args.tag}.fits"
    with fits.open(ours, memmap=True) as h:
        cu = np.asarray(h[0].data[sl] if h[0].data is not None else h[1].data[sl],
                        np.float64)
    nb_o = np.nansum(cu[k - CORE:k + CORE + 1] - np.median(cu[sb], axis=0), axis=0)
    ok2 = valid & np.isfinite(nb_o)

    print(f"{'半徑 (px)':>12}{'遮罩比例':>10}{'ESO Ha':>11}{'我們 Ha':>11}"
          f"{'差異':>10}{'誤差':>9}{'差/誤差':>10}")
    print("-" * 74)
    edges = [0, 20, 40, 60, 80, 110, 150, 200, 260]
    for a, b in zip(edges[:-1], edges[1:]):
        m = ok2 & (r >= a) & (r < b)
        if m.sum() < 30:
            continue
        e, o = np.nanmean(nb[m]), np.nanmean(nb_o[m])
        # 環帶平均的誤差:逐 spaxel 誤差求和後除以 spaxel 數
        se = np.sqrt(np.nansum(err[m] ** 2)) / m.sum()
        print(f"{f'{a}-{b}':>12}{100*np.mean(seg[m] > 0):>9.1f}%{e:>11.2f}{o:>11.2f}"
              f"{o-e:>10.2f}{se:>9.2f}{(o-e)/se:>10.1f}")
    print("  差/誤差 明顯為負 = 我們把暈扣掉了;~0 = 暈被保留")

    # ---- 圖 ----
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
    v = np.nanpercentile(nb[valid], 99.5)
    im = ax[0].imshow(np.where(valid, nb, np.nan), origin="lower",
                      vmin=-0.05 * v, vmax=0.15 * v, cmap="inferno")
    ax[0].contour(seg > 0, levels=[0.5], colors="cyan", linewidths=1.0)
    ax[0].set_title("continuum-subtracted Ha (ESO nosky)\ncyan = SExtractor "
                    "source mask", fontsize=11)
    fig.colorbar(im, ax=ax[0], shrink=0.8)

    m2 = np.where(valid & (seg == 0) & (snr > args.snr), nb, np.nan)
    im2 = ax[1].imshow(m2, origin="lower", vmin=0, vmax=0.15 * v, cmap="viridis")
    ax[1].contour(seg > 0, levels=[0.5], colors="red", linewidths=1.0)
    ax[1].set_title(f"Ha detected at S/N>{args.snr:g} INSIDE the blank sample\n"
                    "(these spaxels train the sky basis)", fontsize=11)
    fig.colorbar(im2, ax=ax[1], shrink=0.8)
    fig.tight_layout()
    out = FIGURES / "halo_in_blank.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
