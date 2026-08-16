"""剔除離群值之後,那個位置要填什麼?

剔除的判準已經定了(|x − med| / sg > 30),剩下的問題是被剔掉的位置填什麼值。
殘差矩陣 R = blank − C_sky 是 SVD 的輸入,填進去的值會被當成真實資料學習。

    填 0        線通道上的 R 很大,填 0 等於宣稱「這裡沒有天光線」——
                本身就是一個假訊號,只是比原本的離群值小
    填 med(R)   該通道的典型殘差。被剔掉的位置看起來像一個完全正常的 spaxel

兩個判準:
    blank 端   train/test 分半,basis 只用 train 學,殘差在 test 上算
    source 端  同一個源、同一條模板的 chi2_all(z) 動態範圍 —— 假性禁止有沒有消失

    conda run -n astro python src/skymodel/experiments/outlier_fill.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import lsq_linear
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import estimate_continuum
from templates import load_sdss_template, redshift_to_grid, air_to_vacuum

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP02  = ROOT / "results/skymodel/step02"
STEP03  = ROOT / "results/skymodel/step03"
TPL_DIR = ROOT / "data/sdss_templates"
WSKY    = ROOT / "data/Haro11_NEpointing_wsky.fits"

SEED, WINDOW, THRESHOLDS, MAX_ITER, CLIP_SIGMA = 0, 300, (1, 2), 5, 30


def n90(v):
    e = np.sort(v ** 2)[::-1]
    return int(np.searchsorted(np.cumsum(e), 0.9 * e.sum()) + 1)


def chi2_curve(flux, var, sky, spline, z_grid, lam_vac, s_fix):
    """照 step4.scan_object 的算法對單一模板掃 z(擬合用覆蓋通道,評分用全通道)。"""
    base = (np.isfinite(flux) & np.isfinite(var) & (var > 0)
            & np.all(np.isfinite(sky), axis=0))
    sig  = np.sqrt(np.where(var > 0, var, 1.0))
    sky_free, y = sky[1:], flux - s_fix * sky[0]
    skyw, yw = np.ascontiguousarray((sky_free / sig).T), y / sig
    p  = sky_free.shape[0] + 1
    lb = np.full(p, -np.inf); lb[0] = 0.0
    ub = np.full(p, np.inf)

    out = np.full(z_grid.size, np.nan)
    for i, z in enumerate(z_grid):
        T    = redshift_to_grid(spline, z, lam_vac)
        good = base & np.isfinite(T)
        if int(good.sum()) <= p:
            continue
        M = np.empty((int(good.sum()), p))
        M[:, 0], M[:, 1:] = T[good] / sig[good], skyw[good]
        th = lsq_linear(M, yw[good], bounds=(lb, ub), method="bvls").x
        m  = np.nan_to_num(T, nan=0.0) * th[0] + th[1:] @ sky_free
        out[i] = (((y - m) / sig) ** 2)[base].sum()
    return out


def main():
    ap = argparse.ArgumentParser(description="離群值位置的填值")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--id", type=int, default=14)
    ap.add_argument("--tpl", default="027")
    ap.add_argument("--zstep", type=float, default=1e-3)
    ap.add_argument("--s-fix", type=float, default=1.0)
    args = ap.parse_args()

    white = fits.getdata(STEP01 / "whitelight.fits")
    seg   = fits.getdata(STEP01 / "seg.fits")
    bm    = (white != 0) & ~((seg > 0) & (white != 0))

    with fits.open(WSKY, memmap=True) as h:
        hdr = h["DATA"].header
        nz  = hdr["NAXIS3"]
        wl  = hdr["CRVAL3"] + (np.arange(nz) + 1 - hdr["CRPIX3"]) * hdr["CD3_3"]
        blank = np.empty((nz, int(bm.sum())), np.float32)
        for j in range(0, nz, 200):
            blank[j:j+200] = np.asarray(h["DATA"].data[j:j+200], np.float32)[:, bm]
    blank = blank[:, np.isfinite(blank).all(axis=0)].astype(np.float64)
    print(f"blank spaxels {blank.shape[1]:,}   {nz} channels")

    # ---- C_sky 用 step3 的做法(sigma-clipped mean)算,各變體共用 ----
    p16, med, p84 = np.percentile(blank, [16, 50, 84], axis=1)
    sg   = np.maximum((p84 - p16) / 2, 1e-6)
    keep = np.abs(blank - med[:, None]) <= CLIP_SIGMA * sg[:, None]
    mean_sky = (blank * keep).sum(axis=1) / keep.sum(axis=1)
    C, _, lm, _ = estimate_continuum(mean_sky, thresholds=THRESHOLDS,
                                     window=WINDOW, max_iter=MAX_ITER)
    nbad = int((~keep).sum())
    print(f"clip {CLIP_SIGMA} sigma 剔除 {nbad:,} 個元素,"
          f"其中 {int((~keep)[lm].sum()):,} 個在線通道 "
          f"({100*(~keep)[lm].sum()/nbad:.0f}%,線通道佔波長的 {100*lm.mean():.1f}%)\n")

    R    = blank - C[:, None]
    medR = med - C                      # 每個通道的典型殘差

    VAR = [("A  不剔除",        R),
           ("B0 剔除,填 0",     np.where(keep, R, 0.0)),
           ("Bm 剔除,填 med(R)", np.where(keep, R, medR[:, None]))]

    # ---- blank 端:train 學 basis,test 算殘差 ----
    rng = np.random.default_rng(SEED)
    tr  = rng.random(blank.shape[1]) < 0.5
    clean = ~(~keep[:, ~tr]).any(axis=0)          # test 端排除含離群值的 spaxel
    print(f"train {int(tr.sum()):,}   test {int((~tr).sum()):,}"
          f"(統計用其中無離群值的 {int(clean.sum()):,} 個)\n")

    print(f"{'變體':>20}{'線內 sigma':>12}{'線外 sigma':>12}{'rms':>9}{'vs A':>9}"
          f"{'最集中 n90':>12}")
    print("-" * 76)
    BASES, base_rms = [], None
    for name, Rv in VAR:
        B = TruncatedSVD(n_components=args.K,
                         random_state=SEED).fit(Rv[:, tr].T).components_
        BASES.append(B)
        sky = np.vstack([C, B])
        res = (blank[:, ~tr] - sky.T @ (np.linalg.pinv(sky.T) @ blank[:, ~tr]))[:, clean]
        si, so = np.median(res[lm].std(0)), np.median(res[~lm].std(0))
        rm = np.median(np.sqrt((res ** 2).mean(0)))
        rel = "" if base_rms is None else f"{100*(rm/base_rms-1):>8.2f}%"
        print(f"{name:>20}{si:>12.4f}{so:>12.4f}{rm:>9.4f}{rel:>9}"
              f"{min(n90(b) for b in B):>12}")
        if base_rms is None:
            base_rms = rm

    # ---- source 端:chi2_all(z) 的動態範圍 ----
    ids   = np.load(STEP02 / "object_ids.npy")
    k     = int(np.flatnonzero(ids == args.id)[0])
    nspax = np.load(STEP02 / "object_nspax.npy")[k]
    f     = np.load(STEP02 / "object_flux.npy")[k] / nspax
    v     = np.load(STEP02 / "object_var.npy")[k]  / nspax ** 2
    spline = load_sdss_template(TPL_DIR / f"spDR2-{args.tpl}.fit")
    zg     = np.arange(0.0, 1.5 + args.zstep / 2, args.zstep)
    print(f"\nID {args.id} 模板 {args.tpl},{zg.size} 個 z...")

    print(f"\n{'變體':>20}{'chi2 最小':>13}{'最小處 z':>11}{'最大/最小':>12}"
          f"{'z>0.35 中位':>14}")
    print("-" * 72)
    for (name, _), B in zip(VAR, BASES):
        c = chi2_curve(f, v, np.vstack([C, B]), spline, zg, air_to_vacuum(wl),
                       args.s_fix)
        hi = zg > 0.35
        print(f"{name:>20}{np.nanmin(c):>13,.0f}{zg[np.nanargmin(c)]:>11.3f}"
              f"{np.nanmax(c)/np.nanmin(c):>11,.0f}x{np.nanmedian(c[hi]):>14,.0f}",
              flush=True)


if __name__ == "__main__":
    main()
