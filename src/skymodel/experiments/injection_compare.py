"""三種做法用同一把尺比較:注入已知強度的假源,看誰量得準。

    現行      源與天空同時擬合,全波長,s 自由
    教授      兩階段:先擬合天空 → 遮掉最亮的天光線 → 在剩下的窗口配源
    固定 s    源與天空同時擬合,全波長,s 由當地 blank 決定

判讀:
    A_true = 0 時量到的 A  = 假訊號底線,愈低愈好
    A_meas / A_true        = 回收率,愈接近 1 愈好
    散布                    = 同亮度下的不確定度,愈小愈好

教授的遮罩強度以「遮掉最亮的百分之幾通道」掃描,因為原話是
「mask the brightest sky lines」而非全部天光線。

    conda run -n astro python src/skymodel/experiments/injection_compare.py \\
        --work results/skymodel/p01
"""
import argparse
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import lsq_linear
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from templates import load_sdss_template, redshift_to_grid, air_to_vacuum

ROOT    = Path(__file__).resolve().parents[3]
TPL_DIR = ROOT / "data/sdss_templates"

BASIS    = "svd"
TEMPLATE = "027"
Z        = 0.0202
FWHM     = 4.06
RADIUS   = 4
N_SITE   = 40
SEED     = 0
AMPS     = np.array([0.0, 3.0, 10.0, 30.0, 100.0, 300.0])
MASK_PCT = [5, 20, 50]        # 教授版:遮掉最亮的百分之幾通道


def gaussian_stamp(radius, fwhm):
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    return np.exp(-(x ** 2 + y ** 2) / (2.0 * (fwhm / 2.3548) ** 2))


def pick_sites(blank, n, radius, seed):
    ok = blank & ~binary_dilation(~blank, iterations=radius + 1)
    ys, xs = np.nonzero(ok)
    rng = np.random.default_rng(seed)
    used = np.zeros_like(blank)
    sites = []
    for i in rng.permutation(ys.size):
        y, x = int(ys[i]), int(xs[i])
        sl = (slice(y - radius, y + radius + 1), slice(x - radius, x + radius + 1))
        if used[sl].any():
            continue
        used[sl] = True
        sites.append((y, x))
        if len(sites) == n:
            break
    return sites


def wls(design, y, sig, use, lb):
    """加權最小平方,受下界 lb 限制。use 指定使用哪些通道。"""
    return lsq_linear(design[:, use].T / sig[use][:, None], y[use] / sig[use],
                      bounds=(lb, np.full(design.shape[0], np.inf)),
                      method="bvls")


def fit_now(f, sig, sky, T, ok):
    """現行:源與天空同時擬合,全波長,s 自由。"""
    K = sky.shape[0]
    des = np.vstack([T, sky])
    r = wls(des, f, sig, ok, np.r_[0.0, 0.0, np.full(K - 1, -np.inf)])
    return r.x[0]


def fit_fixed(f, sig, sky, T, ok, s_fix, keep=None):
    """固定 s:s·C_sky 先扣掉,只解 A 與天光線係數。

    keep 給定時同時套用教授的天光線遮罩(兩種手段疊加)。
    """
    K = sky.shape[0]
    des = np.vstack([T, sky[1:]])
    use = ok if keep is None else (ok & keep)
    if use.sum() <= des.shape[0]:
        return np.nan
    r = wls(des, f - s_fix * sky[0], sig, use, np.r_[0.0, np.full(K - 1, -np.inf)])
    return r.x[0]


def fit_prof_2stage(f, sig, sky, T, ok, keep):
    """教授(讀法甲):先擬合天空(全波長)→ 扣掉 → 在未遮罩的窗口只配源模板。"""
    K = sky.shape[0]
    r = wls(sky, f, sig, ok, np.r_[0.0, np.full(K - 1, -np.inf)])
    resid = f - r.x @ sky
    use = ok & keep
    if use.sum() <= 2:
        return np.nan
    return wls(T[None, :], resid, sig, use, np.r_[0.0]).x[0]


def fit_prof_joint(f, sig, sky, T, ok, keep):
    """教授(讀法乙):源與天空同時擬合,但只用未遮罩的通道算 chi2。"""
    K = sky.shape[0]
    use = ok & keep
    des = np.vstack([T, sky])
    if use.sum() <= des.shape[0]:
        return np.nan
    return wls(des, f, sig, use, np.r_[0.0, 0.0, np.full(K - 1, -np.inf)]).x[0]


def main():
    ap = argparse.ArgumentParser(description="三種做法的注入-回收對照")
    ap.add_argument("--work", required=True,
                    help="pointing 的工作區,例如 results/skymodel/p01")
    ap.add_argument("--cube", default=None,
                    help="含天空的 cube;預設由 pNN 的編號推出 "
                         "data/wshy/DATACUBE_FINAL_N.fits")
    ap.add_argument("-K", type=int, default=30, help="天空 basis 條數,要和 step3 相同")
    args = ap.parse_args()

    # 變數名不能叫 W —— 底下的報表用 W 當欄寬。
    W_dir  = ROOT / args.work
    STEP01, STEP03 = W_dir / "step01", W_dir / "step03"
    CUBE = ROOT / (args.cube or f"data/wshy/DATACUBE_FINAL_{int(W_dir.name[1:])}.fits")

    seg    = fits.getdata(STEP01 / "seg.fits")
    white  = fits.getdata(STEP01 / "whitelight.fits")
    lam    = np.load(STEP03 / "wavelength.npy")
    wl_vac = air_to_vacuum(lam)
    sky    = np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                        np.load(STEP03 / f"sky_basis_{BASIS}_K{args.K}.npy")])
    mean_sky = np.load(STEP03 / "mean_sky.npy")
    K = sky.shape[0]

    with fits.open(CUBE, memmap=True) as hdul:
        D0 = np.asarray(hdul["DATA"].data, np.float32)
        V0 = np.asarray(hdul["STAT"].data, np.float32)

    blank = (white != 0) & (seg == 0)
    T  = redshift_to_grid(load_sdss_template(TPL_DIR / f"spDR2-{TEMPLATE}.fit"), Z, wl_vac)
    Tz = np.nan_to_num(T, nan=0.0)

    # 天光線的強度 = 平均天空 − 連續譜。教授的「最亮的天光線」以此排序。
    line_strength = mean_sky - sky[0]
    keeps = {}
    for pct in MASK_PCT:
        thr = np.percentile(line_strength, 100 - pct)
        keeps[pct] = line_strength < thr

    stamp = gaussian_stamp(RADIUS, FWHM)
    prof  = stamp.ravel()
    sites = pick_sites(blank, N_SITE, RADIUS, SEED)
    print(f"注入 tpl {TEMPLATE} @ z={Z}   孔徑 {stamp.size} px   {len(sites)} 個位置")
    for pct in MASK_PCT:
        print(f"  教授版遮掉最亮 {pct:>2}% 的通道 → 剩 {int(keeps[pct].sum()):,} / {lam.size:,}")

    # 每個位置的資料與固定用的 s(未注入,加權)
    patches, s_ref = [], []
    for (y, x) in sites:
        sl = (slice(y - RADIUS, y + RADIUS + 1), slice(x - RADIUS, x + RADIUS + 1))
        d = D0[:, sl[0], sl[1]].reshape(D0.shape[0], -1).astype(np.float64)
        v = V0[:, sl[0], sl[1]].reshape(V0.shape[0], -1).astype(np.float64)
        n = d.shape[1]
        with np.errstate(invalid="ignore", divide="ignore"):
            f0, v0 = np.nansum(d, 1) / n, np.nansum(v, 1) / n ** 2
        patches.append((f0, v0))
        ok = np.isfinite(f0) & np.isfinite(v0) & (v0 > 0) & np.all(np.isfinite(sky), 0)
        s_ref.append(float(wls(sky, f0, np.sqrt(np.where(v0 > 0, v0, 1.0)), ok,
                               np.r_[0.0, np.full(K - 1, -np.inf)]).x[0]))
    print(f"固定用的 s: 中位數 {np.median(s_ref):.4f}\n")

    cfgs = (["現行"]
            + [f"教授乙 {p}%" for p in MASK_PCT]
            + [f"教授甲 {p}%" for p in MASK_PCT]
            + ["固定 s"] + [f"固定s+遮{p}%" for p in MASK_PCT])
    W = 15
    print(f"{'A_true':>8}" + "".join(f"{c:>{W}}" for c in cfgs))
    print("-" * (8 + W * len(cfgs)))
    floors, rec100 = {}, {}
    for amp in AMPS:
        vals = {c: [] for c in cfgs}
        for (f0, v0), sf in zip(patches, s_ref):
            f = f0 + amp * (prof.sum() / prof.size) * Tz
            ok = np.isfinite(f) & np.isfinite(v0) & (v0 > 0) & np.all(np.isfinite(sky), 0) \
                 & np.isfinite(T)
            sig = np.sqrt(np.where(v0 > 0, v0, 1.0))
            vals["現行"].append(fit_now(f, sig, sky, Tz, ok))
            vals["固定 s"].append(fit_fixed(f, sig, sky, Tz, ok, sf))
            for p in MASK_PCT:
                vals[f"教授乙 {p}%"].append(fit_prof_joint(f, sig, sky, Tz, ok, keeps[p]))
                vals[f"教授甲 {p}%"].append(fit_prof_2stage(f, sig, sky, Tz, ok, keeps[p]))
                vals[f"固定s+遮{p}%"].append(fit_fixed(f, sig, sky, Tz, ok, sf, keeps[p]))
        true_tot = amp * prof.sum() / prof.size
        cells = []
        for c in cfgs:
            a = np.array(vals[c], float)
            if amp == 0:
                # A >= 0 被夾住,中位數恆為 0;用平均與 84% 分位描述假訊號的水位
                floors[c] = (float(np.nanmean(a)), float(np.nanpercentile(a, 84)))
                cells.append(f"{floors[c][0]:>7.2f}/{floors[c][1]:<6.2f}")
            else:
                if amp == 100.0:
                    rec100[c] = (float(np.nanmedian(a)/true_tot),
                                 float(np.nanstd(a)/true_tot))
                cells.append(f"{np.nanmedian(a)/true_tot:>7.2f}±{np.nanstd(a)/true_tot:<6.2f}")
        tag = "0 (空)" if amp == 0 else f"{amp:.0f}"
        print(f"{tag:>8}" + "".join(f"{x:>{W}}" for x in cells))

    print("\n第一列 = 假訊號的 平均/84%分位 (A>=0 被夾住,中位數恆為 0,不能用)")
    print("其餘   = A_meas / A_true ± 散布\n")
    base_contrast = (rec100["現行"][0] * 100.0 * prof.sum() / prof.size
                     / floors["現行"][0])
    print(f"{'做法':>14}{'假訊號(平均)':>14}{'相對現行':>10}"
          f"{'A=100 回收率':>16}{'散布':>10}{'偵測對比':>12}{'相對現行':>10}")
    print("-" * 88)
    for c in cfgs:
        rec, sc = rec100[c]
        true100 = 100.0 * prof.sum() / prof.size
        contrast = rec * true100 / floors[c][0]
        print(f"{c:>14}{floors[c][0]:>14.3f}"
              f"{floors[c][0]/floors['現行'][0]:>9.3f}x"
              f"{rec:>16.2f}{sc:>10.2f}{contrast:>12.1f}"
              f"{contrast/base_contrast:>10.1f}x")


if __name__ == "__main__":
    main()
