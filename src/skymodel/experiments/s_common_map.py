"""14 顆 pointing 的 s 殘差疊成一張共同圖 —— 並先確認那個共同圖案不是星系的暈。

背景
----
s 場寫成 s_hat = M + r,M = mu + a(y) + b(x) 抓走條紋,r 是局部核回歸。核的支撐
只有 3 x sigma = 6 px,所以離訓練點超過 6 px 的地方 r 恰為 0 —— 被源遮住的大洞
裡沒有 r 可用。

但每一顆 pointing 的 Haro 11 在視場裡的位置不同,而 WCS 對齊之後各顆的殘差
**彼此相關**,代表其中有一部分是所有 pointing 共有的固定圖案。

那就可以疊:一顆的洞,在別顆那裡是 blank,量得到。

**但先要排除一個致命的可能**
--------------------------
共有的圖案不一定是儀器。**Haro 11 的延展暈在天上是固定位置**,所以在每一顆裡都
落在同一個天空座標,一樣會製造「共有的圖案」。如果把它疊進 s 場再拿去扣,那就是
把星系的暈當成天空扣掉 —— 正是我們一路在修的病,只是換了更隱蔽的入口。

    是儀器  ->  振幅和「離 Haro 11 的距離」無關,結構是條紋
    是暈    ->  離星系越近越強,結構是徑向遞減

這支先跑這個判別。不過關就不要往下做。

共同座標
--------
14 顆的 CRVAL 完全相同,只差 CRPIX,所以天空位置就是 (i − CRPIX1, j − CRPIX2)。
不需要真的做 WCS 重取樣,整數位移就對得上(位移量都是整數像素等級)。

    conda run -n astro python src/skymodel/experiments/s_common_map.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import main_source_group, rowcol_field, scale  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
FIG  = ROOT / "results/skymodel/figures/s_common"


def load_all(ns):
    """回傳每顆的 (殘差 R, blank 遮罩, 白光, CRPIX, 主源峰值)。"""
    out = {}
    for n in ns:
        W = ROOT / f"results/skymodel/p{n:02d}"
        runs = sorted((W / "step05").glob("*_sfield"))
        if not runs:
            continue
        s = np.load(runs[0] / "s_free.npy").astype(float)
        seg = fits.getdata(W / "step01/seg.fits").astype(int)
        white = np.asarray(fits.getdata(W / "step01/whitelight.fits"), float)
        valid = white != 0
        mg, _, pk = main_source_group(seg, np.where(valid, white, np.nan),
                                        W / "step04b")
        blank = valid & (seg == 0) & np.isfinite(s)
        M, _, _ = rowcol_field(s, blank)     # 各顆自己的條紋各自扣掉
        h = fits.getheader(ROOT / f"data/wshy/DATACUBE_FINAL_{n}.fits", 1)
        out[n] = dict(R=np.where(blank, s - M, np.nan), blank=blank, white=white,
                      crpix=(h["CRPIX1"], h["CRPIX2"]), peak=pk, main=mg)
    return out


def main():
    ap = argparse.ArgumentParser(description="共同殘差圖與「是不是暈」的判別")
    ap.add_argument("-n", type=int, nargs="+", default=list(range(1, 15)))
    ap.add_argument("--min-n", type=int, default=3,
                    help="共同圖上一格至少要有幾顆貢獻才採用")
    args = ap.parse_args()

    D = load_all(args.n)
    print(f"讀到 {len(D)} 顆")
    # 共同座標:天空位置 = (j − CRPIX2, i − CRPIX1)
    oy = {n: -int(round(d["crpix"][1])) for n, d in D.items()}
    ox = {n: -int(round(d["crpix"][0])) for n, d in D.items()}
    ylo = min(oy.values()); xlo = min(ox.values())
    yhi = max(oy[n] + D[n]["R"].shape[0] for n in D)
    xhi = max(ox[n] + D[n]["R"].shape[1] for n in D)
    H, Wd = yhi - ylo, xhi - xlo
    print(f"共同格網 {H} x {Wd}")

    stack = np.full((len(D), H, Wd), np.nan)
    wstack = np.full((len(D), H, Wd), np.nan)
    peaks = []
    for k, (n, d) in enumerate(sorted(D.items())):
        y0, x0 = oy[n] - ylo, ox[n] - xlo
        h_, w_ = d["R"].shape
        stack[k, y0:y0 + h_, x0:x0 + w_] = d["R"]
        wstack[k, y0:y0 + h_, x0:x0 + w_] = np.where(d["white"] != 0,
                                                     d["white"], np.nan)
        peaks.append((d["peak"][0] + y0, d["peak"][1] + x0))
    peaks = np.array(peaks)
    print(f"各顆主源峰值在共同座標上:y {peaks[:,0].min()}-{peaks[:,0].max()}  "
          f"x {peaks[:,1].min()}-{peaks[:,1].max()}   "
          f"散布 {peaks[:,0].std():.1f} / {peaks[:,1].std():.1f} px")
    py, px = np.median(peaks, axis=0)

    cnt = np.isfinite(stack).sum(axis=0)
    common = np.where(cnt >= args.min_n, np.nanmedian(stack, axis=0), np.nan)
    wl_common = np.where(cnt >= args.min_n, np.nanmedian(wstack, axis=0), np.nan)
    ok = np.isfinite(common)
    print(f"共同圖:{int(ok.sum()):,} 格有 >= {args.min_n} 顆貢獻   "
          f"振幅 穩健散布 {scale(common[ok]):.5f}")

    yy, xx = np.mgrid[0:H, 0:Wd]
    dist = np.hypot(yy - py, xx - px)
    print(f"\n=== 判別:共同圖案是儀器還是 Haro 11 的暈? ===")
    print(f"{'離 Haro 11':>14}{'格數':>8}{'共同圖 中位':>14}{'|共同圖| 中位':>14}")
    for lo, hi in ((0, 50), (50, 100), (100, 150), (150, 200), (200, 300),
                   (300, 500)):
        m = ok & (dist >= lo) & (dist < hi)
        if m.sum() < 200:
            continue
        print(f"{f'{lo}-{hi} px':>14}{int(m.sum()):>8,}{np.median(common[m]):>+14.5f}"
              f"{np.median(np.abs(common[m])):>14.5f}")
    r_dist = np.corrcoef(dist[ok], common[ok])[0, 1]
    r_absd = np.corrcoef(dist[ok], np.abs(common[ok]))[0, 1]
    g = ok & np.isfinite(wl_common)
    r_white = np.corrcoef(np.log10(np.maximum(wl_common[g], 1e-3)), common[g])[0, 1]
    print(f"\n  與距離的相關      {r_dist:+.4f}      與 |值| 的相關 {r_absd:+.4f}")
    print(f"  與 log(白光) 的相關 {r_white:+.4f}")
    print(f"  (暈的話:離越近值越高 -> 與距離負相關、與白光正相關)")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    v = 3 * scale(common[ok])
    im = ax[0].imshow(common, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    plt.colorbar(im, ax=ax[0], fraction=.046)
    ax[0].plot(px, py, "k+", ms=14, mew=2)
    ax[0].set_title(f"common residual map ({len(D)} pointings, median)", fontsize=10)
    im = ax[1].imshow(cnt, origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax[1], fraction=.046)
    ax[1].set_title("how many pointings contribute", fontsize=10)
    b = np.arange(0, 460, 20)
    idx = np.digitize(dist[ok], b)
    prof = [np.median(common[ok][idx == i]) for i in range(1, len(b))]
    npix = [int((idx == i).sum()) for i in range(1, len(b))]
    ax[2].plot(b[:-1] + 10, prof, "o-")
    ax[2].axhline(0, color="k", lw=.8)
    ax[2].set_xlabel("distance from Haro 11 [px]")
    ax[2].set_ylabel("median of common map"); ax[2].grid(alpha=.25)
    ax[2].set_title("flat = instrumental,  declining = halo", fontsize=10)
    fig.suptitle("is the common residual pattern instrumental, or Haro 11's halo?",
                 fontsize=12)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "common_map_check.png", dpi=140, bbox_inches="tight")
    np.save(FIG / "common_map.npy", common)
    np.save(FIG / "common_count.npy", cnt)
    print(f"\nsaved -> {FIG}/common_map_check.png")


if __name__ == "__main__":
    main()
