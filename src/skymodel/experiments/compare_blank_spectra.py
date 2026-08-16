"""隨機抽 blank spaxel,把我們扣完天空的光譜和 ESO nosky 畫在同一張上比較。

blank 區沒有源,兩者的真值都是 0,所以「誰比較貼近 0」就是誰扣得好。
每個 spaxel 一張圖,標題附上該 spaxel 的 rms 與 STAT 給出的光子雜訊下限
—— 兩者都貼近雜訊下限時,差距才是模型的差異而不是雜訊。

    conda run -n astro python src/skymodel/experiments/compare_blank_spectra.py \
        --tag svd_K25_s_1.0 --n 10
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import lsq_linear
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP03  = ROOT / "results/skymodel/step03"
STEP05  = ROOT / "results/skymodel/step05"
FIGURES = ROOT / "results/skymodel/figures"
WSKY    = ROOT / "data/Haro11_NEpointing_wsky.fits"
ESO     = ROOT / "data/Haro11_NEpointing_esonosky.fits"

# 固定的一組 blank spaxel。跨設定比較(換 K、換 basis)必須看同一批 spaxel,
# 否則量到的差異可能只是換了樣本 —— 用亂數種子不夠,因為候選池本身會隨
# 「這個 cube 哪些 spaxel 是有限值」而變,同一個種子會抽到不同的位置。
#
# 選法:在 25 的倍數格點中,取滿足「seg == 0、白光非零、我們與 ESO 兩個 cube 的
# 3801 個通道都有資料、離任何 seg > 0 至少 10 px」的點,再挑 10 個散佈全視場的。
# 座標全部取整是為了好讀好對照。
# (ESO 的 cube 要一起檢查:有的 spaxel 在我們的輸出乾淨,ESO 卻有 NaN 通道。)
FIXED_SPAXELS = [
    ( 50,  50), ( 50, 150), (100, 100), (150, 200), (200, 100),
    (200, 200), (250,  50), (250, 250), (300, 150), (300, 225),
]


def main():
    ap = argparse.ArgumentParser(description="blank spaxel:我們 vs ESO nosky")
    ap.add_argument("--tag", default="svd_K25_s_1.0", help="step05 的輸出 tag")
    ap.add_argument("--from-basis", nargs=2, metavar=("BASIS", "K"), default=None,
                    help="不讀 step05 的 cube,直接從 basis 算 blank 的殘差。"
                         "blank 區的擬合完全不用模板,所以結果和跑完 step4+step5 "
                         "再讀 cube 完全相同,但只要幾秒鐘。"
                         "例:--from-basis svd 35")
    ap.add_argument("--n", type=int, default=10,
                    help=f"畫幾個 spaxel(取 FIXED_SPAXELS 的前 n 個,最多 {len(FIXED_SPAXELS)})")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="固定 y 範圍;不給則用對稱的 99.5 百分位")
    ap.add_argument("--width", type=float, default=22.0, help="圖寬(吋)")
    ap.add_argument("--height", type=float, default=3.2, help="每個子圖的高(吋)")
    ap.add_argument("--panels", type=int, default=5,
                    help="把全波段切成幾段,每段一個子圖。3801 個通道畫在一張 16 吋寬、"
                         "140 dpi 的圖上是每像素 2.4 個通道 —— 通道解析不出來,兩條線"
                         "糊成一片。切 5 段、每段 22 吋 = 每個通道 4 px 才看得清楚。")
    ap.add_argument("--out-suffix", default="",
                    help="輸出資料夾加後綴,讓不同版面的圖並存不互相覆蓋")
    ap.add_argument("--with-wsky", action="store_true",
                    help="把原始的 wsky(未扣天空)疊在同一個面板。它的連續譜落在 "
                         "y 範圍內;天光線的峰值遠超出去,會被 y 範圍切掉。")
    ap.add_argument("--zoom", type=float, nargs=2, metavar=("L1", "L2"), default=None,
                    help="只畫這一段波長(覆蓋 --panels,用來細看某幾條天光線)")
    ap.add_argument("--smooth", type=int, default=0,
                    help="另外疊一條 N 通道移動平均。雜訊被壓成 1/sqrt(N),"
                         "零點偏移才浮得出來。0 = 不畫")
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    wl    = np.load(STEP03 / "wavelength.npy")

    if args.from_basis:
        b, k = args.from_basis[0], int(args.from_basis[1])
        # 檔名不帶 s 的設定 —— 這裡的 blank 一律自由解 s,
        # --s-fix / --s-free 只作用在 source 區。標上 s_1.0 會誤導。
        args.tag = f"{b}_K{k}_blank"
        sky = np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                         np.load(STEP03 / f"sky_basis_{b}_K{k}.npy")])
        with fits.open(WSKY, memmap=True) as h:
            S = np.asarray(h["DATA"].data, np.float32).copy()
        nzz, nyy, nxx = S.shape
        Sf = S.reshape(nzz, -1)
        want = [yy * nxx + xx for yy, xx in FIXED_SPAXELS]
        for jj in want:                       # 只算要畫的那幾個 spaxel
            d = Sf[:, jj].astype(np.float64)
            g = np.isfinite(d)
            if g.sum() <= sky.shape[0]:
                continue
            c = np.linalg.lstsq(sky[:, g].T, d[g], rcond=None)[0]
            if c[0] < 0:                      # s 不能是負的,和 fit_blank 一致
                lb = np.r_[0.0, np.full(sky.shape[0] - 1, -np.inf)]
                c = lsq_linear(sky[:, g].T, d[g],
                               bounds=(lb, np.full(sky.shape[0], np.inf)),
                               method="bvls").x
            Sf[:, jj] = d - sky.T @ c
        S = Sf.reshape(nzz, nyy, nxx)
    else:
        S = fits.getdata(STEP05 / f"sky_subtracted_{args.tag}.fits")
    E = fits.getdata(ESO)
    with fits.open(WSKY, memmap=True) as h:
        V = np.asarray(h["STAT"].data, np.float32)
        W = np.asarray(h["DATA"].data, np.float32) if args.with_wsky else None
    nz, ny, nx = S.shape
    S, E, V = S.reshape(nz, -1), E.reshape(nz, -1), V.reshape(nz, -1)
    if W is not None:
        W = W.reshape(nz, -1)

    blank = ((white != 0) & (seg == 0)).ravel()
    ok = blank & np.isfinite(S).all(0) & np.isfinite(E).all(0) & np.isfinite(V).all(0)
    pick = [y * nx + x for y, x in FIXED_SPAXELS[:args.n]]
    bad  = [(y, x) for (y, x), p in zip(FIXED_SPAXELS[:args.n], pick) if not ok[p]]
    if bad:
        raise SystemExit(f"固定樣本在這個 tag 下不可用(有 NaN 或不在 blank):{bad}")

    out = FIGURES / f"blank_compare_{args.tag}{args.out_suffix}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"固定樣本 {args.n} 個(blank 可用共 {int(ok.sum()):,})   tag={args.tag}\n")
    print(f"{'#':>3}{'(y, x)':>12}{'我們 rms':>11}{'ESO rms':>11}"
          f"{'雜訊下限':>11}{'我們偏移':>11}{'ESO 偏移':>11}{'勝者':>7}")
    print("-" * 78)

    for i, j in enumerate(pick, 1):
        y, x = divmod(int(j), nx)
        s, e, v = S[:, j].astype(np.float64), E[:, j].astype(np.float64), V[:, j].astype(np.float64)
        w = W[:, j].astype(np.float64) if W is not None else None
        rs, re_ = np.sqrt(np.mean(s ** 2)), np.sqrt(np.mean(e ** 2))
        nf = np.sqrt(np.mean(v))
        print(f"{i:>3}{f'({y},{x})':>12}{rs:>11.3f}{re_:>11.3f}{nf:>11.3f}"
              f"{s.mean():>11.4f}{e.mean():>11.4f}{'我們' if rs < re_ else 'ESO':>7}")

        # 波長切段。每段畫一個子圖,通道才解析得出來。
        if args.zoom:
            edges = np.array(args.zoom, float)
        else:
            edges = np.linspace(wl[0], wl[-1], args.panels + 1)
        npan = len(edges) - 1

        # y 範圍全部子圖共用,否則各段的縱向尺度不同,看起來像雜訊大小在變。
        lim = args.ylim or (lambda q: (-q, q))(np.percentile(np.abs(np.r_[s, e]), 99.5))

        fig, axes = plt.subplots(npan, 1,
                                 figsize=(args.width, args.height * npan + 0.4))
        axes = np.atleast_1d(axes)
        for k, a in enumerate(axes):
            m = (wl >= edges[k]) & (wl <= edges[k + 1])
            a.axhline(0, color="0.55", lw=0.6, zorder=1)
            # wsky 疊在同一個面板:它的連續譜落在 y 範圍內,可以直接看出「我們
            # 把那條線壓到 0」。天光線的峰值遠超出去,自然被 y 範圍切掉 ——
            # 那是刻意的,畫出來只會把殘差壓成一條線。
            if w is not None:
                a.plot(wl[m], w[m], lw=0.7, color="#7f7f7f", alpha=0.9, zorder=1.5,
                       label=f"wsky (before)   median {np.median(w):.1f}   "
                             f"peak {w.max():.0f}")
            # mean / sigma / rms 三個一起列:rms² = mean² + sigma²,分開看才知道
            # 優勢來自零點(mean)還是抖動(sigma)。blank 區真值是 0,三個都該接近 0。
            a.plot(wl[m], e[m], lw=0.7, color="#d62728", alpha=0.75, zorder=2,
                   label=f"ESO nosky      mean {e.mean():+.3f}   "
                         f"sigma {e.std():.3f}   rms {re_:.3f}")
            a.plot(wl[m], s[m], lw=0.7, color="#1f77b4", alpha=0.85, zorder=3,
                   label=f"ours                 mean {s.mean():+.3f}   "
                         f"sigma {s.std():.3f}   rms {rs:.3f}")
            if args.smooth > 1:
                kern = np.ones(args.smooth) / args.smooth   # 不可叫 w,那是 wsky
                for arr, col in ((e, "#d62728"), (s, "#1f77b4")):
                    a.plot(wl[m], np.convolve(arr, kern, "same")[m], lw=1.6, color=col,
                           alpha=0.95, zorder=4)
            a.set_xlim(edges[k], edges[k + 1])
            a.set_ylim(*lim)
            a.set_ylabel("flux  [$10^{-20}$ cgs]", fontsize=9)
            a.tick_params(labelsize=9)
            if k == 0:
                a.legend(fontsize=9, loc="upper right", framealpha=0.9)
                a.set_title(f"blank spaxel #{i}   (y={y}, x={x})   —   {args.tag}",
                            fontsize=11)
        axes[-1].set_xlabel("wavelength [$\\AA$]", fontsize=10)
        fig.tight_layout(h_pad=0.4)
        fig.savefig(out / f"spaxel_{i:02d}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    won = sum(np.sqrt(np.mean(S[:, j].astype(np.float64) ** 2))
              < np.sqrt(np.mean(E[:, j].astype(np.float64) ** 2)) for j in pick)
    print(f"\n這 {args.n} 個裡我們贏 {won} 個")
    print(f"saved {args.n} figures -> {out}")


if __name__ == "__main__":
    main()
