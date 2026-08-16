"""學 basis 之前,把有發射線的 spaxel 剔除掉

問題:blank 樣本裡有一部分 spaxel 含 Haro 11 電離氣體暈的發射線。暈只有發射線、
沒有連續譜,所以白光影像上的 SExtractor 偵測看不到它。於是 SVD 會學到一條含
紅移 [O III]/Ha 的成分,再從真正的源身上減掉。

要區分兩件事:
    源遮罩(白光 SExtractor)   界定「哪裡是源」,不動
    學 basis 的樣本純度                 界定「哪些 spaxel 乾淨到可以當天空的
                                        範本」,這是另一件事,而且只影響 step3

損失可以拆成兩塊(高對比成像文獻的標準拆法):
    一般性投影   源的光譜有一部分落在 K+1 維子空間裡
                 (注入的假源不在訓練集裡,所以只有這塊)
    自我扣除     源自己在訓練集裡,連 basis 都被它污染
剔除污染只能拿回第二塊。第一塊是線性投影的本質,拿不回來。

偵測用我們自己的扣天空產物,不用 ESO —— 這樣方法是自足的(未來可進 pipeline)。
代價是偏保守:現行 basis 已經扣掉一部分暈,所以會少偵測到一些。同時報 ESO 的
數字當作參考,看漏掉多少。

輸出 sky_basis_svdclean_K30.npy,可以直接餵給 step4/step5 的 --basis svdclean。

    conda run -n astro python src/skymodel/experiments/line_purity.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_dilation
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import estimate_continuum

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP03  = ROOT / "results/skymodel/step03"
STEP04  = ROOT / "results/skymodel/step04"
STEP05  = ROOT / "results/skymodel/step05"
WSKY    = ROOT / "data/Haro11_NEpointing_wsky.fits"
ESO     = ROOT / "data/Haro11_NEpointing_esonosky.fits"

SEED, WINDOW, THRESHOLDS, MAX_ITER, CLIP_SIGMA, TILE = 0, 300, (1, 2), 5, 30, 32
CORE, GAP, BG = 3, 10, 35

# Haro 11 的靜止空氣波長。只列夠亮、且不落在強天光線上的
GAL_LINES = [("Hb", 4861.3), ("[O III]4959", 4958.9), ("[O III]5007", 5006.8),
             ("Ha", 6562.8), ("[N II]6583", 6583.5),
             ("[S II]6716", 6716.4), ("[S II]6731", 6730.8)]


def line_snr(cube_path, var, wl, z, hdu):
    """逐 spaxel 對每條星系線做窄帶偵測,回傳各線 S/N 的逐 spaxel 最大值。"""
    nz = wl.size
    best = None
    with fits.open(cube_path, memmap=True) as h:
        d = h[hdu].data
        for nm, l0 in GAL_LINES:
            lo = l0 * (1 + z)
            c = int(np.argmin(np.abs(wl - lo)))
            if c - BG < 0 or c + BG + 1 > nz:
                continue
            sl = slice(c - BG, c + BG + 1)
            x = np.asarray(d[sl], np.float64)
            k = BG
            sb = np.r_[np.arange(0, k - GAP), np.arange(k + GAP + 1, x.shape[0])]
            nb = np.nansum(x[k-CORE:k+CORE+1] - np.median(x[sb], axis=0), axis=0)
            e = np.sqrt(np.nansum(var[sl][k-CORE:k+CORE+1], axis=0))
            s = np.where(e > 0, nb / e, 0.0)
            best = s if best is None else np.maximum(best, s)
    return best


def main():
    ap = argparse.ArgumentParser(description="剔除含發射線的 spaxel 再學 basis")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--snr", type=float, nargs="+", default=[5.0, 3.0, 2.0])
    ap.add_argument("--adopt", type=float, default=3.0, help="存檔採用的門檻")
    ap.add_argument("--dilate", type=int, default=1, help="空間膨脹 (px)")
    args = ap.parse_args()

    wl    = np.load(STEP03 / "wavelength.npy")
    IN    = np.load(STEP03 / "iter_line_mask.npy")[0]
    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    best  = np.load(STEP04 / "best_svd_K30_s_1.0.npz")
    z = float(best["z"][int(np.flatnonzero(best["id"] == 1)[0])])
    ny, nx = seg.shape

    with fits.open(WSKY, memmap=True) as h:
        nz = h["DATA"].header["NAXIS3"]
        var = np.asarray(h["STAT"].data, np.float32)
    print(f"z = {z:.5f}   檢查 {len(GAL_LINES)} 條星系線")

    snr_ours = line_snr(STEP05 / "sky_subtracted_svd_K30_s_1.0.fits", var, wl, z, 0)
    snr_eso  = line_snr(ESO, var, wl, z, 1)
    del var

    bm = (white != 0) & (seg == 0)
    print(f"\nblank {int(bm.sum()):,} spaxel")
    print(f"{'門檻':>8}{'我們的產物偵測到':>18}{'ESO 偵測到':>14}{'膨脹後剔除':>13}")
    print("-" * 56)
    masks = {}
    for t in args.snr:
        a = bm & (snr_ours > t)
        b = bm & (snr_eso > t)
        d = binary_dilation(a | b, iterations=args.dilate) if args.dilate else (a | b)
        masks[t] = bm & ~d
        print(f"{t:>8.1f}{int(a.sum()):>18,}{int(b.sum()):>14,}"
              f"{int((bm & d).sum()):>13,}")
    print("(取兩者的聯集再膨脹 —— 我們的產物已扣掉一部分暈,單獨用會偵測不足)")

    # ---- 載入 blank 光譜 ----
    with fits.open(WSKY, memmap=True) as h:
        D = np.empty((nz, int(bm.sum())), np.float32)
        for j in range(0, nz, 200):
            D[j:j+200] = np.asarray(h["DATA"].data[j:j+200], np.float32)[:, bm]
    ok = np.isfinite(D).all(axis=0)
    D = D[:, ok].astype(np.float64)

    yy, xx = np.mgrid[0:ny, 0:nx]
    tr = (((yy // TILE) + (xx // TILE)) % 2).astype(bool)[bm][ok]
    # test 端一律只用最嚴門檻下的乾淨 spaxel —— 各變體共用同一批,比較才公平
    clean_te = masks[min(args.snr)][bm][ok][~tr]

    def build(keep_spx):
        """keep_spx 是 (n_blank,) 的布林,指出哪些 train spaxel 拿來學 basis。"""
        X = D[:, tr & keep_spx]
        p16, med, p84 = np.percentile(X, [16, 50, 84], axis=1)
        sg = np.maximum((p84 - p16) / 2, 1e-6)
        kp = np.abs(X - med[:, None]) <= CLIP_SIGMA * sg[:, None]
        C = estimate_continuum((X * kp).sum(axis=1) / kp.sum(axis=1),
                               thresholds=THRESHOLDS, window=WINDOW,
                               max_iter=MAX_ITER)[0]
        R = np.where(kp, X - C[:, None], (med - C)[:, None])
        B = TruncatedSVD(n_components=args.K,
                         random_state=SEED).fit(R.T.astype(np.float32)).components_
        return C, B

    def oiii_strength(B):
        """basis 在 [O III]5007 的結構強度,以局部散布為單位(取絕對值最大者)。"""
        c = int(np.argmin(np.abs(wl - 5006.8 * (1 + z))))
        lo, hi = c - BG, c + BG + 1
        sb = np.r_[np.arange(lo, c - GAP), np.arange(c + GAP + 1, hi)]
        Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
        base = np.median(Bn[:, sb], axis=1)
        sc = np.maximum(1.4826*np.median(np.abs(Bn[:, sb]-base[:, None]), axis=1), 1e-12)
        core = Bn[:, c-CORE:c+CORE+1].sum(axis=1) - (2*CORE+1)*base
        return np.abs(core / (sc * np.sqrt(2*CORE+1))).max()

    print(f"\n{'訓練樣本':>22}{'spaxel 數':>11}{'test rms':>10}{'test 線內':>11}"
          f"{'test 線外':>11}{'vs 全部':>9}{'basis 的 [O III]':>17}")
    print("-" * 92)
    base_rms = None
    for nm, keep in [("全部 (現行)", np.ones(D.shape[1], bool))] + \
                    [(f"剔除 S/N>{t:g}", masks[t][bm][ok]) for t in args.snr]:
        C, B = build(keep)
        sky = np.vstack([C, B])
        Dte = D[:, ~tr][:, clean_te]
        Rte = Dte - sky.T @ (np.linalg.pinv(sky.T) @ Dte)
        rm = float(np.median(np.sqrt(np.mean(Rte ** 2, axis=0))))
        si = float(np.median(Rte[IN].std(axis=0)))
        so = float(np.median(Rte[~IN].std(axis=0)))
        rel = "" if base_rms is None else f"{100*(rm/base_rms-1):>8.2f}%"
        print(f"{nm:>22}{int((tr & keep).sum()):>11,}{rm:>10.4f}{si:>11.4f}"
              f"{so:>11.4f}{rel:>9}{oiii_strength(B):>17.1f}", flush=True)
        if base_rms is None:
            base_rms = rm

    # ---- 採用的門檻:用全部 train+test 的乾淨 spaxel 重學,寫檔給 step4/5 ----
    C, B = build(masks[args.adopt][bm][ok] & np.ones(D.shape[1], bool))
    keep_all = masks[args.adopt][bm][ok]
    X = D[:, keep_all]
    p16, med, p84 = np.percentile(X, [16, 50, 84], axis=1)
    sg = np.maximum((p84 - p16) / 2, 1e-6)
    kp = np.abs(X - med[:, None]) <= CLIP_SIGMA * sg[:, None]
    Cf = estimate_continuum((X * kp).sum(axis=1) / kp.sum(axis=1),
                            thresholds=THRESHOLDS, window=WINDOW, max_iter=MAX_ITER)[0]
    Rf = np.where(kp, X - Cf[:, None], (med - Cf)[:, None])
    Bf = TruncatedSVD(n_components=args.K,
                      random_state=SEED).fit(Rf.T.astype(np.float32)).components_
    out = STEP03 / f"sky_basis_svdclean_K{args.K}.npy"
    np.save(out, Bf)
    np.save(STEP03 / f"line_purity_mask_snr{args.adopt:g}.npy", masks[args.adopt])
    print(f"\n採用 S/N>{args.adopt:g}:{int(keep_all.sum()):,} 個 spaxel 學 basis")
    print(f"basis 的 [O III] 強度 {oiii_strength(Bf):.1f}"
          f"(現行 {oiii_strength(np.load(STEP03/f'sky_basis_svd_K{args.K}.npy')):.1f})")
    print(f"saved -> {out}")
    print(f"\n注意:C_sky 不隨這個標籤改變(step4/5 讀的是共用的 sky_continuum.npy),")
    print(f"      這是刻意的 —— 只換 basis,變因才單一。")


if __name__ == "__main__":
    main()
