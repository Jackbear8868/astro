"""逐 spaxel 擬合:固定 step4 定出的 (模板, z),對每個 spaxel 解線性係數。

    source (seg > 0)   D(p,λ) = A(p)·T(λ) + s(p)·C_sky(λ) + Σₖ cₖ(p)·Lₖ(λ)
    blank  (seg = 0)   D(p,λ) =             s(p)·C_sky(λ) + Σₖ cₖ(p)·Lₖ(λ)

T 是 step4 決定的模板(已紅移),形狀固定;逐 spaxel 只解係數的大小。

兩區的目標函數不同:source 最小化 chi2(以 1/sigma 加權),blank 最小化平方
誤差(不加權)。不加權時設計矩陣不隨 spaxel 改變,佔視場 84% 的 blank 可用
一次 pinv 全部解完;加權則必須逐 spaxel 求解。

輸出的天空模型不含模板項 —— 扣掉的只有天空,源要保留。
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from astropy.io import fits

from templates import load_sdss_template, redshift_to_grid, air_to_vacuum

ROOT    = Path(__file__).resolve().parents[2]
STEP01  = ROOT / "results/skymodel/step01"
STEP03  = ROOT / "results/skymodel/step03"
STEP04  = ROOT / "results/skymodel/step04"
STEP05  = ROOT / "results/skymodel/step05"
TPL_DIR = ROOT / "data/sdss_templates"
CUBE    = ROOT / "data/Haro11_NEpointing_wsky.fits"

def build_templates(best, lam_vac):
    """挑出要放模板的源,並把各自的模板紅移到 MUSE 波長格點。

    step4 已對 A 施加非負限制,A = 0 代表最佳解落在邊界上,
    也就是這條模板對該源沒有貢獻,這種源不放模板。

    Returns
    -------
    dict
        {segmentation ID: 已紅移到 lam_vac 的模板, shape (nz,)}
    """
    keep = best["A"] > 0
    out  = {}
    for i in np.flatnonzero(keep):
        spline = load_sdss_template(TPL_DIR / f"spDR2-{best['template'][i]}.fit")
        out[int(best["id"][i])] = redshift_to_grid(spline, float(best["z"][i]), lam_vac)
    return out

def fit_blank(D, sky):
    """blank spaxel 的係數:未加權最小平方,s 受非負限制。

    不加權時設計矩陣不隨 spaxel 改變,pinv 只算一次,乾淨的 spaxel
    用一次矩陣乘法全部解出。有壞通道的 spaxel 設計矩陣不同,逐一解。
    邊界事後才處理:無約束解若已滿足 s >= 0,它就是有約束解,不必重算;
    這樣才能保住一次解完的優勢,實際只有極少數 spaxel 需要重解。

    Parameters
    ----------
    D : ndarray, shape (nz, n)
        光譜,壞通道為 NaN。
    sky : ndarray, shape (K+1, nz)
        天空連續譜與天光線 basis。

    Returns
    -------
    ndarray, shape (K+1, n)
        每個 spaxel 的係數;無法求解者為 NaN。
    """
    coef  = np.full((sky.shape[0], D.shape[1]), np.nan)
    good  = np.isfinite(D)
    clean = good.all(axis=0)
    coef[:, clean] = np.linalg.pinv(sky.T) @ D[:, clean]
    for j in np.flatnonzero(~clean):
        g = good[:, j]
        if g.sum() <= sky.shape[0]:
            continue
        coef[:, j] = np.linalg.lstsq(sky[:, g].T, D[g, j], rcond=None)[0]
    
    lb = np.r_[0.0, np.full(sky.shape[0] - 1, -np.inf)]
    ub = np.full(sky.shape[0], np.inf)
    for j in np.flatnonzero(coef[0] < 0):
        g = good[:, j]
        coef[:, j] = lsq_linear(sky[:, g].T, D[g, j], bounds=(lb, ub), method="bvls").x
    return coef

def fit_source(D, var, sky, T, s_fix=None):
    """一批共用同一條模板的 source spaxel:最小化 chi2,逐 spaxel 求解。

    每個 spaxel 的 sigma 不同,設計矩陣隨 spaxel 改變,無法像 fit_blank
    那樣共用 pinv。A 與 s 受非負限制,與 step4 一致。

    s_fix 給定時,s·C_sky 先從資料扣掉,s 不再是自由參數。用意是切斷
    A·T 與 s·C_sky 的簡併 —— 兩者形狀幾乎相同,任其自由會讓模板吸走
    天空連續譜。blank 區保持自由,因為那正是量 s 的地方。

    Parameters
    ----------
    D, var : ndarray, shape (nz, n)
        光譜與變異數,壞通道為 NaN。
    sky : ndarray, shape (K+1, nz)
    T : ndarray or None
        已紅移到 MUSE 格點的模板 (nz,);None 表示這個區域不放模板。
    s_fix : float or None
        天空連續譜的固定係數;None 表示 s 為自由參數。

    Returns
    -------
    ndarray, shape (K+2, n)
        固定的排列 (A, s, c₁…c_K),與 T、s_fix 的有無無關。
        沒有模板時 A 為 NaN;無法求解者整欄為 NaN。
    """
    K   = sky.shape[0]                            # 1 條連續譜 + (K−1) 條天光線
    out = np.full((K + 1, D.shape[1]), np.nan)    # 固定排列 (A, s, c₁…c_{K−1})

    rows   = (([] if T is None else [T])
              + ([] if s_fix is not None else [sky[0]])
              + list(sky[1:]))
    design = np.vstack(rows)
    p      = design.shape[0]
    n_pos  = (0 if T is None else 1) + (0 if s_fix is not None else 1)
    lb = np.r_[np.zeros(n_pos), np.full(p - n_pos, -np.inf)]
    ub = np.full(p, np.inf)

    y = D if s_fix is None else D - s_fix * sky[0][:, None]
    good = (np.isfinite(y) & np.isfinite(var) & (var > 0)
            & np.all(np.isfinite(design), axis=0)[:, None])

    for j in range(D.shape[1]):
        g = good[:, j]
        if g.sum() <= p:
            continue
        sig = np.sqrt(var[g, j])
        th  = lsq_linear(design[:, g].T / sig[:, None], y[g, j] / sig,
                         bounds=(lb, ub), method="bvls").x
        i = 0
        if T is not None:
            out[0, j] = th[i]; i += 1
        out[1, j] = th[i] if s_fix is None else s_fix
        i += (s_fix is None)
        out[2:, j] = th[i:]
    return out

def main():
    ap = argparse.ArgumentParser(description="逐 spaxel 擬合天空與源模板")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("--s-fix", type=float, default=None,
                    help="源區域的天空連續譜係數固定值;不給則 s 為自由參數。blank 區一律保持自由。")
    args = ap.parse_args()
    tag = args.basis if args.s_fix is None else f"{args.basis}_sfix"

    STEP05.mkdir(parents=True, exist_ok=True)

    seg    = fits.getdata(STEP01 / "seg.fits")
    white  = fits.getdata(STEP01 / "whitelight.fits")
    wl_vac = air_to_vacuum(np.load(STEP03 / "wavelength.npy"))
    sky    = np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                        np.load(STEP03 / f"sky_basis_{args.basis}.npy")])
    # 模板必須來自「同一個模型」的 step4 —— s 的處理方式不同,選出的
    # (模板, z) 就不同(實測 12/37 個源會換模板),混用等於用錯的答案。
    best_file = STEP04 / f"best_{tag}.npz"
    if not best_file.exists():
        raise SystemExit(
            f"找不到 {best_file.name}。step4 必須先以相同的 s 設定跑過:\n"
            f"  conda run -n astro python src/skymodel/step4_find_template.py "
            f"--id all --basis {args.basis}"
            + (f" --s-fix {args.s_fix}" if args.s_fix is not None else ""))
    best = np.load(best_file)

    with fits.open(CUBE, memmap=True) as hdul:
        hdr = hdul["DATA"].header
        D   = np.asarray(hdul["DATA"].data, np.float32)
        V   = np.asarray(hdul["STAT"].data, np.float32)

    nz, ny, nx = D.shape
    D     = D.reshape(nz, -1)
    V     = V.reshape(nz, -1)
    seg_f = seg.reshape(-1)
    valid = (white != 0).reshape(-1)
    templates = build_templates(best, wl_vac)
    sky_model = np.full((nz, ny * nx), np.nan, np.float32)
    A_map     = np.full(ny * nx, np.nan, np.float32)
    s_map     = np.full(ny * nx, np.nan, np.float32)

    blank = valid & (seg_f == 0)
    c = fit_blank(D[:, blank], sky)
    sky_model[:, blank] = sky.T @ c
    s_map[blank]        = c[0]

    for rid in np.unique(seg_f[valid & (seg_f > 0)]):
        m = valid & (seg_f == rid)
        c = fit_source(D[:, m], V[:, m], sky, templates.get(int(rid)), s_fix=args.s_fix)
        sky_model[:, m] = sky.T @ c[1:]
        A_map[m]        = c[0]
        s_map[m]        = c[1]

    sub = D - sky_model
    cube = lambda x: x.reshape(nz, ny, nx)
    fits.writeto(STEP05 / f"sky_model_{tag}.fits",  cube(sky_model), hdr, overwrite=True)
    fits.writeto(STEP05 / f"sky_subtracted_{tag}.fits", cube(sub),   hdr, overwrite=True)
    np.save(STEP05 / f"A_map_{tag}.npy", A_map.reshape(ny, nx))
    np.save(STEP05 / f"s_map_{tag}.npy", s_map.reshape(ny, nx))

    print(f"blank {int(blank.sum()):,}  source {int((valid & (seg_f > 0)).sum()):,}"
          f"  放模板的區域 {len(templates)}")
    print(f"saved -> {STEP05}")

if __name__ == "__main__":
    main()