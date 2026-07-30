"""注入-回收(step4):固定 s 能不能讓模板搜尋找得更準?

在 blank 區域注入「已知模板、已知 z」的假源,跑完整的 step4 搜尋,
看能不能找回正確答案。兩種設定各跑一次:

    free   s 逐候選自由      = 現行 step4
    fixed  s 由當地 blank 決定,只解 A 與天光線係數

計分:
    模板正確率   best template == 注入的模板
    z 誤差       |z_best − z_true|
    假陽性       A_true = 0 時仍被判定為「有源」的比例

A_true = 0 那一組沒有正確答案可言 —— 它量的是搜尋在純天空上會找出什麼,
以及那個假答案的 dchi2 有多大(篩選門檻的依據)。

    conda run -n astro python src/skymodel/experiments/injection_step4.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.optimize import lsq_linear
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from templates import load_sdss_template, redshift_to_grid, air_to_vacuum

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP03  = ROOT / "results/skymodel/step03"
TPL_DIR = ROOT / "data/sdss_templates"
CUBE    = ROOT / "data/Haro11_NEpointing_wsky.fits"

BASIS    = "svd"
N_TPL    = 33
TPL_TRUE = "027"          # 注入用的模板與紅移(正確答案)
Z_TRUE   = 0.0202
FWHM     = 4.06
RADIUS   = 4
N_SITE   = 8
SEED     = 0
Z_GRID   = np.arange(0.0, 1.5 + 5e-5, 1e-4)      # 教授指定
Z_TOL    = 1e-3           # z 判定為正確的容差

AMPS = np.array([0.0, 30.0, 100.0, 300.0, 1000.0])

_S = {}


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


def solve(design, lb, X, y, sig, g):
    """加權最小平方,受 lb 下界限制;回傳 (係數, chi2)。"""
    fit = lsq_linear(design[:, g].T / sig[g][:, None], y[g] / sig[g],
                     bounds=(lb, np.full(design.shape[0], np.inf)), method="bvls")
    return fit.x, 2.0 * fit.cost


def scan(f, v, sky, templates, s_fix):
    """對一條光譜掃描模板與紅移。s_fix 為 None 時 s 自由。

    回傳 (最佳模板, 最佳 z, A, chi2_all, dchi2)。
    dchi2 以「同樣設定但不放模板」的 chi2_all 為基準。
    """
    K    = sky.shape[0]
    base = np.isfinite(f) & np.isfinite(v) & (v > 0) & np.all(np.isfinite(sky), axis=0)
    sig  = np.sqrt(np.where(v > 0, v, 1.0))
    free = s_fix is None

    # 基準:不放模板
    lb0 = np.r_[0.0, np.full(K - 1, -np.inf)] if free else np.full(K - 1, -np.inf)
    d0  = sky if free else sky[1:]
    y0  = f if free else f - s_fix * sky[0]
    th0, _ = solve(d0, lb0, None, y0, sig, base)
    m0 = (th0 @ sky) if free else (s_fix * sky[0] + th0 @ sky[1:])
    c_null = float((((f - m0) / sig) ** 2)[base].sum())

    lb = np.r_[0.0, 0.0, np.full(K - 1, -np.inf)] if free \
         else np.r_[0.0, np.full(K - 1, -np.inf)]
    best = (None, np.nan, np.nan, np.inf)
    for z in Z_GRID:
        for name, spl in templates.items():
            T = redshift_to_grid(spl, z, sky_lam)
            g = base & np.isfinite(T)
            des = np.vstack([T, sky]) if free else np.vstack([T, sky[1:]])
            if g.sum() <= des.shape[0]:
                continue
            y = f if free else f - s_fix * sky[0]
            th, _ = solve(des, lb, None, y, sig, g)
            Tz = np.nan_to_num(T, nan=0.0)
            m  = th[0] * Tz + (th[1:] @ sky if free else s_fix * sky[0] + th[1:] @ sky[1:])
            c  = float((((f - m) / sig) ** 2)[base].sum())
            if c < best[3]:
                best = (name, float(z), float(th[0]), c)
    return best + (c_null - best[3],)


def _run(task):
    i_site, amp, fixed = task
    d = _S["patch_d"][i_site] + amp * _S["prof"][None, :] * _S["Tz"][:, None]
    v = _S["patch_v"][i_site]
    n = d.shape[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.nansum(d, axis=1) / n
        vv = np.nansum(v, axis=1) / n ** 2
    s_fix = _S["s_ref"][i_site] if fixed else None
    t0 = time.time()
    r = scan(f, vv, _S["sky"], _S["templates"], s_fix)
    return task, r, time.time() - t0


def main():
    global sky_lam
    seg    = fits.getdata(STEP01 / "seg.fits")
    white  = fits.getdata(STEP01 / "whitelight.fits")
    sky_lam = air_to_vacuum(np.load(STEP03 / "wavelength.npy"))
    sky    = np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                        np.load(STEP03 / f"sky_basis_{BASIS}.npy")])
    K = sky.shape[0]

    with fits.open(CUBE, memmap=True) as hdul:
        D0 = np.asarray(hdul["DATA"].data, np.float32)
        V0 = np.asarray(hdul["STAT"].data, np.float32)

    blank = (white != 0) & (seg == 0)
    templates = {f"{i:03d}": load_sdss_template(TPL_DIR / f"spDR2-{i:03d}.fit")
                 for i in range(N_TPL)}
    Tz = np.nan_to_num(redshift_to_grid(templates[TPL_TRUE], Z_TRUE, sky_lam), nan=0.0)

    stamp = gaussian_stamp(RADIUS, FWHM)
    prof  = stamp.ravel()
    sites = pick_sites(blank, N_SITE, RADIUS, SEED)

    # 每個位置的固定 s:對未注入的平均光譜做一次加權擬合
    patch_d, patch_v, s_ref = [], [], []
    for (y, x) in sites:
        sl = (slice(y - RADIUS, y + RADIUS + 1), slice(x - RADIUS, x + RADIUS + 1))
        d = D0[:, sl[0], sl[1]].reshape(D0.shape[0], -1).astype(np.float64)
        v = V0[:, sl[0], sl[1]].reshape(V0.shape[0], -1).astype(np.float64)
        patch_d.append(d); patch_v.append(v)
        nn = d.shape[1]
        with np.errstate(invalid="ignore", divide="ignore"):
            f0, v0 = np.nansum(d, 1) / nn, np.nansum(v, 1) / nn ** 2
        g = np.isfinite(f0) & np.isfinite(v0) & (v0 > 0) & np.all(np.isfinite(sky), 0)
        th, _ = solve(sky, np.r_[0.0, np.full(K - 1, -np.inf)], None, f0,
                      np.sqrt(np.where(v0 > 0, v0, 1.0)), g)
        s_ref.append(float(th[0]))

    print(f"注入 tpl {TPL_TRUE} @ z={Z_TRUE}   孔徑 {stamp.size} px   {len(sites)} 個位置")
    print(f"固定用的 s(當地 blank,加權): 中位數 {np.median(s_ref):.4f}"
          f"  範圍 {min(s_ref):.4f}–{max(s_ref):.4f}")
    tasks = [(i, a, fx) for a in AMPS for fx in (False, True) for i in range(len(sites))]
    print(f"{len(tasks)} 次掃描 x {N_TPL * Z_GRID.size:,} 候選\n", flush=True)

    _S.update(patch_d=patch_d, patch_v=patch_v, s_ref=s_ref, prof=prof, Tz=Tz,
              sky=sky, templates=templates)

    out = {}
    n_workers = max(1, len(os.sched_getaffinity(0)) // 3)
    with Pool(n_workers) as pool:
        for task, r, dt in pool.imap_unordered(_run, tasks):
            out[task] = r
            print(f"  site {task[0]}  A={task[1]:>6.0f}  "
                  f"{'fixed' if task[2] else 'free ':>5}  →  tpl {r[0]} z={r[1]:.4f}"
                  f"  dchi2={r[4]:>10,.0f}   {dt:.0f}s", flush=True)

    print(f"\n{'A_true':>8}{'——— s 自由 ———':^34}{'——— s 固定 ———':^34}")
    print(f"{'':>8}{'模板正確':>10}{'z 正確':>10}{'dchi2 中位':>14}"
          f"{'模板正確':>12}{'z 正確':>10}{'dchi2 中位':>14}")
    print("-" * 78)
    for a in AMPS:
        row = []
        for fx in (False, True):
            rs = [out[(i, a, fx)] for i in range(len(sites))]
            okt = sum(r[0] == TPL_TRUE for r in rs)
            okz = sum(abs(r[1] - Z_TRUE) < Z_TOL for r in rs)
            row.append((okt, okz, np.median([r[4] for r in rs])))
        tag = "0 (空)" if a == 0 else f"{a:.0f}"
        print(f"{tag:>8}{row[0][0]:>7}/{len(sites)}{row[0][1]:>9}/{len(sites)}"
              f"{row[0][2]:>14,.0f}"
              f"{row[1][0]:>9}/{len(sites)}{row[1][1]:>9}/{len(sites)}"
              f"{row[1][2]:>14,.0f}")
    print("\nA_true = 0 沒有正確答案:那一列的 dchi2 就是純天空上搜尋能碰出的最大改善,"
          "\n也就是篩選門檻必須高過的水位。")


if __name__ == "__main__":
    main()
