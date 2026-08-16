"""剔除離群值之後,chi2_all 在高 z 的暴增有沒有消失?

背景:通道 151(4788.4 A)有少數 blank spaxel 是極端的負值。
兩個後果串在一起:

    mean_sky 用 np.nanmean → 該通道的平均被拉離鄰居 → C_sky 也歪掉
    SVD 學到一條能量幾乎全集中在通道 151 的 basis → 它描述的是那些壞
    spaxel,不是天空

第二個才是真正傷人的。模板紅移到某個 z 之後藍端會蓋過 4788 A,那條 basis
的係數從此失去約束,chi2_all 跳一個數量級 —— 於是那個 z 之上被「假性禁止」,
不是因為物理,是因為一條壞掉的 basis。

三個變體,只差在離群值怎麼處理:

    A  現行            mean = nanmean,殘差原封不動
    B  截斷離群值      mean 不變,殘差裡偏離通道中位數 >N sigma 的元素設 0
    C  B + 穩健 mean   mean 改用中位數,C_sky 跟著重估

判準有二:那條被壞點佔走的 basis 是否消失(n90 變大),以及同一個源、同一條
模板的 chi2_all(z) 曲線上,那道懸崖是否被填平。

    conda run -n astro python src/skymodel/experiments/outlier_chi2_jump.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import lsq_linear
from sklearn.decomposition import TruncatedSVD
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import estimate_continuum
from templates import load_sdss_template, redshift_to_grid, air_to_vacuum

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/ne_pointing/step01"
STEP02  = ROOT / "results/skymodel/ne_pointing/step02"
STEP03  = ROOT / "results/skymodel/ne_pointing/step03"
FIGURES = ROOT / "results/skymodel/evaluation/sky_basis"
TPL_DIR = ROOT / "data/sdss_templates"
WSKY    = ROOT / "data/Haro11_NEpointing_wsky.fits"

SEED, WINDOW, THRESHOLDS, MAX_ITER = 0, 300, (1, 2), 5


def n90(v):
    """能量的 90% 落在幾個通道。個位數 = 這條 basis 被單一壞點佔走。"""
    e = np.sort(v ** 2)[::-1]
    return int(np.searchsorted(np.cumsum(e), 0.9 * e.sum()) + 1)


def chi2_curve(flux, var, sky, spline, z_grid, lam_vac, s_fix):
    """照 step4.scan_object 的算法,對單一模板掃 z,回傳 chi2_all(z)。

    擬合只用模板覆蓋到的通道,評分套回全部通道 —— 覆蓋少的 z 不能靠躲起來取勝。
    """
    base = (np.isfinite(flux) & np.isfinite(var) & (var > 0)
            & np.all(np.isfinite(sky), axis=0))
    sig  = np.sqrt(np.where(var > 0, var, 1.0))
    sky_free = sky[1:]                       # s 固定,天空連續譜先扣掉
    y        = flux - s_fix * sky[0]
    skyw     = np.ascontiguousarray((sky_free / sig).T)
    yw       = y / sig

    p  = sky_free.shape[0] + 1
    lb = np.full(p, -np.inf); lb[0] = 0.0    # 單一模板:A>=0 等價於源光譜>=0
    ub = np.full(p, np.inf)

    out = np.full(z_grid.size, np.nan)
    for i, z in enumerate(z_grid):
        T    = redshift_to_grid(spline, z, lam_vac)
        good = base & np.isfinite(T)
        n    = int(good.sum())
        if n <= p:
            continue
        M = np.empty((n, p))
        M[:, 0]  = T[good] / sig[good]
        M[:, 1:] = skyw[good]
        th = lsq_linear(M, yw[good], bounds=(lb, ub), method="bvls").x
        m  = np.nan_to_num(T, nan=0.0) * th[0] + th[1:] @ sky_free
        out[i] = (((y - m) / sig) ** 2)[base].sum()
    return out


def main():
    ap = argparse.ArgumentParser(description="離群值剔除對 chi2_all(z) 的效果")
    ap.add_argument("--id", type=int, default=14)
    ap.add_argument("--tpl", default="027")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--nsigma", type=float, default=30.0)
    ap.add_argument("--zstep", type=float, default=1e-3)
    ap.add_argument("--s-fix", type=float, default=1.0)
    args = ap.parse_args()

    # ---------- 照 step3 的方式取 blank ----------
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
    print(f"blank spaxels(完整光譜) {blank.shape[1]:,}   {nz} channels")

    c151 = int(np.argmin(np.abs(wl - 4788.4)))
    print(f"檢查通道 {c151} ({wl[c151]:.1f} A):"
          f"nanmean {blank[c151].mean():.3f}   median {np.median(blank[c151]):.3f}   "
          f"最小值 {blank[c151].min():,.0f}\n")

    # ---------- 2x2:mean_sky 的估計量 x 殘差要不要剔除離群值 ----------
    # 兩個改動都針對同一批壞 spaxel,但作用在不同地方 —— 一個決定 C_sky,
    # 一個決定 basis。分開量才知道功勞各是多少。
    #
    # 判準用「偏離該通道的中位數多少」而不是值本身。兩處是同一個不等式:
    # R = blank − C_sky 只差一個逐通道常數,常數同時平移 x 和 med,比值不變。
    med = np.median(blank, axis=1)
    sg  = np.maximum((np.percentile(blank, 84, axis=1)
                      - np.percentile(blank, 16, axis=1)) / 2, 1e-6)
    keep = np.abs(blank - med[:, None]) <= args.nsigma * sg[:, None]

    def build(clip_mean, reject):
        mean = ((blank * keep).sum(axis=1) / keep.sum(axis=1) if clip_mean
                else blank.mean(axis=1))
        C = estimate_continuum(mean, thresholds=THRESHOLDS,
                               window=WINDOW, max_iter=MAX_ITER)[0]
        R = blank - C[:, None]
        if reject:
            R = np.where(keep, R, med[:, None] - C[:, None])
        B = TruncatedSVD(n_components=args.K,
                         random_state=SEED).fit(R.T.astype(np.float32)).components_
        return C, B, 0 if not reject else int((~keep).sum())

    VAR = [("A   nanmean, 不剔除",  build(False, False)),
           ("A'  clipped, 不剔除",  build(True,  False)),
           ("B   nanmean, 剔除",    build(False, True)),
           ("C   clipped, 剔除",    build(True,  True))]

    print(f"{'變體':>18}{'C_sky[151]':>13}{'剔除元素':>10}{'最集中 n90':>12}"
          f"{'該 basis 在151的能量':>22}")
    print("-" * 76)
    for name, (C, B, nb) in VAR:
        k = int(np.argmin([n90(b) for b in B]))
        frac = B[k, c151] ** 2 / (B[k] ** 2).sum()
        print(f"{name:>18}{C[c151]:>13.3f}{nb:>10,}{n90(B[k]):>12}{100*frac:>21.1f}%")

    # ---------- 對同一個源、同一條模板掃 z ----------
    ids   = np.load(STEP02 / "object_ids.npy")
    k     = int(np.flatnonzero(ids == args.id)[0])
    nspax = np.load(STEP02 / "object_nspax.npy")[k]
    f     = np.load(STEP02 / "object_flux.npy")[k] / nspax
    v     = np.load(STEP02 / "object_var.npy")[k]  / nspax ** 2
    lam_vac = air_to_vacuum(wl)
    spline  = load_sdss_template(TPL_DIR / f"spDR2-{args.tpl}.fit")
    zg      = np.arange(0.0, 1.5 + args.zstep / 2, args.zstep)
    print(f"\nID {args.id} (nspax={int(np.median(nspax))})  模板 {args.tpl}  "
          f"{zg.size} 個 z 各擬合 3 次...")

    curves = []
    for name, (C, B, _) in VAR:
        curves.append(chi2_curve(f, v, np.vstack([C, B]), spline, zg,
                                 lam_vac, args.s_fix))
        print(f"  {name} 完成", flush=True)

    print(f"\n{'變體':>18}{'chi2_all 最小':>15}{'最小處 z':>11}"
          f"{'最大/最小':>12}{'z>0.35 的中位':>16}")
    print("-" * 74)
    for (name, _), c in zip(VAR, curves):
        hi = zg > 0.35
        print(f"{name:>18}{np.nanmin(c):>15,.0f}{zg[np.nanargmin(c)]:>11.3f}"
              f"{np.nanmax(c)/np.nanmin(c):>12,.0f}x"
              f"{np.nanmedian(c[hi]):>16,.0f}")

    # ---------- 圖 ----------
    FIGURES.mkdir(parents=True, exist_ok=True)
    # 圖例用 ASCII —— matplotlib 的預設字型沒有中文字符,會畫成方框
    LBL = ["A   plain mean, keep outliers  (current)",
           "A'  clipped mean, keep outliers",
           "B   plain mean, reject outliers",
           "C   clipped mean, reject outliers"]
    # 曲線可能兩兩重合,所以粗實線在下、細虛線在上,四條才都看得見。
    fig, ax = plt.subplots(figsize=(11, 6))
    for lbl, c, col, lw, ls in zip(LBL, curves,
                                   ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"],
                                   [3.0, 1.2, 3.0, 1.2], ["-", "--", "-", "--"]):
        ax.plot(zg, c, color=col, lw=lw, ls=ls, label=lbl, alpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("redshift z")
    ax.set_ylabel("chi2 over all channels")
    ax.set_title(f"ID {args.id}, template {args.tpl}: does outlier rejection "
                 f"remove the chi2 cliff?  (K={args.K})", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = FIGURES / f"outlier_chi2_jump_id{args.id}_tpl{args.tpl}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
