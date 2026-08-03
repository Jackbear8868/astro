"""逐 spaxel 擬合:固定 step4 定出的 (類別, 模型, z),對每個 spaxel 解線性係數。

    source (seg > 0)   D(p,λ) = Σⱼ aⱼ(p)·Tⱼ(λ) + s(p)·C_sky(λ) + Σₖ cₖ(p)·Lₖ(λ)
    blank  (seg = 0)   D(p,λ) =                   s(p)·C_sky(λ) + Σₖ cₖ(p)·Lₖ(λ)

Tⱼ 是 step4 決定的源模型(已紅移),形狀固定;逐 spaxel 只解係數的大小。
依 step4 定出的組別,源模型有兩種寬度:

    star        該條 SDSS 恆星模板,1 欄
    galaxy/qso  對應的本徵譜,4 欄

z 不逐 spaxel 重解 —— 它是天體的性質,而且加總光譜的 S/N 比單一 spaxel 高
兩個數量級(Haro 11 是 √13782 ≈ 117 倍),在單一 spaxel 上解 z 只會更差。

兩區的目標函數不同:source 最小化 chi2(以 1/sigma 加權),blank 預設最小化
平方誤差(不加權)。不加權時設計矩陣不隨 spaxel 改變,佔視場 84% 的 blank
可用一次 pinv 全部解完;加權則必須逐 spaxel 求解。

blank 的兩個選項互相正交,可組出 2x2 方便比較:
    --blank-chi2               改以 1/var 加權(最小化 chi2)
    --blank-region line1       只在第一輪 line mask 的通道上擬合
「第一輪」是 estimate_continuum 的 iteration 1,只抓到最強的線(35.6% 通道);
最後一輪的 79.4% 是自我強化迴圈撞到地板參數的結果,不是物理。
限縮範圍只改「在哪些通道解係數」,天空模型仍然在全部通道上求值 —— 我們
每個通道都要扣天空。

輸出的天空模型不含模板項 —— 扣掉的只有天空,源要保留。
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
from astropy.io import fits

from templates import (load_sdss_template, load_eigen_galaxy, load_eigen_qso,
                       redshift_to_grid, air_to_vacuum)

ROOT      = Path(__file__).resolve().parents[2]
STEP01    = ROOT / "results/skymodel/step01"
STEP03    = ROOT / "results/skymodel/step03"
STEP04    = ROOT / "results/skymodel/step04"
STEP05    = ROOT / "results/skymodel/step05"
TPL_DIR   = ROOT / "data/sdss_templates"
EIGEN_GAL = ROOT / "data/eigen_galaxy_Bolton2012.fits"
EIGEN_QSO = ROOT / "data/qso_eigen_linear_55732.dat"
CUBE      = ROOT / "data/Haro11_NEpointing_wsky.fits"

MIN_COVERAGE = 0.9      # spaxel 至少要有幾成波長通道有資料才納入擬合
N_SRC        = 4        # 源係數的固定欄數(本徵譜 4 條;恆星只用第 0 欄)

def build_templates(best, lam_vac):
    """挑出要放模型的源,並把各自的源模型紅移到 MUSE 波長格點。

    step4 已對第 1 個源係數施加非負限制,A[0] = 0 代表最佳解落在邊界上,
    也就是這個模型對該源沒有貢獻,這種源不放模型。

    模型依 step4 定出的組別而不同:恆星用該條 SDSS 模板(1 欄),
    星系/QSO 用對應的本徵譜(4 欄)。

    Returns
    -------
    dict
        {segmentation ID: 已紅移到 lam_vac 的模型, shape (nz, n_comp)}
    """
    eigen = {"galaxy": load_eigen_galaxy(EIGEN_GAL), "qso": load_eigen_qso(EIGEN_QSO)}
    out   = {}
    # 不能只看 A[:, 0] —— 本徵譜沒有非負約束,主導成分的係數可以是 0 或負的,
    # 源仍然由其餘三條扛著(Haro 11 實測 a₁ = −0.171,源光譜卻全正)。
    # 判準是「四個係數全部為 0」才算沒有模型;NaN 是恆星未用到的欄位。
    for i in np.flatnonzero(np.nansum(np.abs(best["A"]), axis=1) > 0):
        g = str(best["group"][i])
        spline = (eigen[g] if g in eigen
                  else load_sdss_template(TPL_DIR / f"spDR2-{best['template'][i]}.fit"))
        T = redshift_to_grid(spline, float(best["z"][i]), lam_vac)
        out[int(best["id"][i])] = T if T.ndim == 2 else T[:, None]
    return out

def fit_blank(D, sky, var=None, fit_mask=None):
    """blank spaxel 的係數,s 受非負限制。目標函數由 var 與 fit_mask 決定。

    var 為 None:最小化未加權平方誤差。設計矩陣不隨 spaxel 改變,pinv
    只算一次,乾淨的 spaxel 用一次矩陣乘法全部解出;只有帶壞通道的
    spaxel 設計矩陣不同,需要逐一解。
    var 給定:最小化 chi2。sigma 逐 spaxel 不同,設計矩陣跟著不同,
    pinv 無法共用,每個 spaxel 都得逐一解 —— 慢,但統計上是正確的權重。

    fit_mask 只縮小「解係數用哪些通道」,不影響回傳的係數怎麼被使用。
    它對所有 spaxel 都一樣,所以 pinv 捷徑仍然成立(共用的設計矩陣只是
    少了幾列)。限縮到天光線所在的通道反而讓問題更良態:線外的通道對
    天光線係數幾乎沒有貢獻(該區設計矩陣的條件數 3.4e6,接近奇異)。

    邊界事後才處理:無約束解若已滿足 s >= 0,它就是有約束解,不必重算;
    這樣才能保住一次解完的優勢,實際只有極少數 spaxel 需要重解。

    Parameters
    ----------
    D : ndarray, shape (nz, n)
        光譜,壞通道為 NaN。
    sky : ndarray, shape (K+1, nz)
        天空連續譜與天光線 basis。
    var : ndarray or None, shape (nz, n)
        給定則以 1/var 加權(最小化 chi2);None 則不加權。
    fit_mask : ndarray or None, shape (nz,)
        布林;只有 True 的通道進入擬合。None 表示全部通道。

    Returns
    -------
    ndarray, shape (K+1, n)
        每個 spaxel 的係數;無法求解者為 NaN。
    """
    K    = sky.shape[0]
    coef = np.full((K, D.shape[1]), np.nan)
    good = np.isfinite(D)
    if var is not None:
        good &= np.isfinite(var) & (var > 0)      # 加權時 var 也必須可用
    rows = np.ones(D.shape[0], bool) if fit_mask is None else fit_mask
    if fit_mask is not None:
        good &= fit_mask[:, None]                 # 廣播到所有 spaxel

    if var is None:
        # 「乾淨」= 擬合範圍內沒有壞通道;範圍外的通道本來就不參與,不算數
        clean = good[rows].all(axis=0)
        coef[:, clean] = np.linalg.pinv(sky[:, rows].T) @ D[rows][:, clean]
        rest = np.flatnonzero(~clean)             # 只有壞通道的 spaxel 要逐一解
    else:
        rest = np.arange(D.shape[1])              # 加權:每一個都要逐一解

    for j in rest:
        g = good[:, j]
        if g.sum() <= K:
            continue
        if var is None:
            coef[:, j] = np.linalg.lstsq(sky[:, g].T, D[g, j], rcond=None)[0]
        else:
            sig = np.sqrt(var[g, j])
            coef[:, j] = np.linalg.lstsq(sky[:, g].T / sig[:, None],
                                         D[g, j] / sig, rcond=None)[0]

    lb = np.r_[0.0, np.full(K - 1, -np.inf)]
    ub = np.full(K, np.inf)
    for j in np.flatnonzero(coef[0] < 0):         # NaN 比較為 False,不會誤抓
        g = good[:, j]
        if var is None:
            coef[:, j] = lsq_linear(sky[:, g].T, D[g, j],
                                    bounds=(lb, ub), method="bvls").x
        else:
            sig = np.sqrt(var[g, j])
            coef[:, j] = lsq_linear(sky[:, g].T / sig[:, None], D[g, j] / sig,
                                    bounds=(lb, ub), method="bvls").x
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
        已紅移到 MUSE 格點的源模型 (nz, n_comp);None 表示這個區域不放模型。
    s_fix : float or None
        天空連續譜的固定係數;None 表示 s 為自由參數。

    Returns
    -------
    ndarray, shape (N_SRC+K, n)
        固定的排列 (a₁…a₄, s, c₁…c_{K−1}),與 T 的欄數、s_fix 的有無無關。
        沒有模型時 a 全為 NaN;恆星只有 a₁ 有值;無法求解者整欄為 NaN。
    """
    K      = sky.shape[0]                         # 1 條連續譜 + (K−1) 條天光線
    n_comp = 0 if T is None else T.shape[1]
    out    = np.full((N_SRC + K, D.shape[1]), np.nan)

    rows   = (([] if T is None else list(T.T))
              + ([] if s_fix is not None else [sky[0]])
              + list(sky[1:]))
    design = np.vstack(rows)
    p      = design.shape[0]

    # 非負的參數不是「開頭連續的幾個」——順序是 [a₁…a_ncomp, (s), c₁…c_{K−1}],
    # s 自由時落在 index n_comp,前面隔著 n_comp−1 個源係數。
    # 單一模板(恆星)時 A ≥ 0 和「源光譜 ≥ 0」完全等價,保留;多成分基底則不等價,
    # 硬壓只會扭曲擬合(step4 實測 chi2_all 差 8.2%),與 step4 的處理一致。
    lb = np.full(p, -np.inf)
    if n_comp == 1:
        lb[0] = 0.0
    if s_fix is None:
        lb[n_comp] = 0.0
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
        out[:n_comp, j]    = th[:n_comp]
        out[N_SRC, j]      = th[n_comp] if s_fix is None else s_fix
        out[N_SRC + 1:, j] = th[n_comp + (s_fix is None):]
    return out

def main():
    ap = argparse.ArgumentParser(description="逐 spaxel 擬合天空與源模板")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=25,
                    help="天光線 basis 條數;必須和 step3/step4 用的 K 相同")
    ap.add_argument("--s-fix", type=float, default=1.0,
                    help="源區域的天空連續譜係數固定值(預設 1.0)。blank 區一律保持自由。")
    ap.add_argument("--s-free", action="store_true",
                    help="改讓源區域的 s 成為自由參數(教授原本的做法),覆蓋 --s-fix")
    ap.add_argument("--blank-chi2", action="store_true",
                    help="blank 區改用 chi2 加權;不給則最小化未加權平方誤差。")
    ap.add_argument("--blank-region", choices=["all", "line1"], default="all",
                    help="blank 區解係數使用的通道:all=全部,line1=第一輪 line mask。")
    args = ap.parse_args()
    s_fix = None if args.s_free else args.s_fix
    tag = (f"{args.basis}_K{args.K}_s_free" if s_fix is None
           else f"{args.basis}_K{args.K}_s_{s_fix}")
    # blank 的加權方式與擬合範圍只影響 step5 的輸出 —— step4 沒有 blank 區,
    # 所以讀 step4 產出時用 tag,寫自己的輸出時用 tag_out。
    tag_out = tag + ("_bchi2" if args.blank_chi2 else "") \
                  + ("" if args.blank_region == "all" else f"_{args.blank_region}")

    STEP05.mkdir(parents=True, exist_ok=True)

    seg    = fits.getdata(STEP01 / "seg.fits")
    white  = fits.getdata(STEP01 / "whitelight.fits")
    wl_vac = air_to_vacuum(np.load(STEP03 / "wavelength.npy"))
    sky    = np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                        np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")])
    # 模板必須來自「同一個模型」的 step4 —— s 的處理方式不同,選出的
    # (模板, z) 就不同(實測 12/37 個源會換模板),混用等於用錯的答案。
    best_file = STEP04 / f"best_{tag}.npz"
    if not best_file.exists():
        raise SystemExit(
            f"找不到 {best_file.name}。step4 必須先以相同的 s 設定跑過:\n"
            f"  conda run -n astro python src/skymodel/step4_find_template.py "
            f"--id all --basis {args.basis} -K {args.K}"
            + (" --s-free" if s_fix is None else f" --s-fix {s_fix}"))
    best = np.load(best_file)

    # 第一輪的 line mask:estimate_continuum 的 iteration 1,只抓到最強的線。
    # 之後每一輪門檻都會下移(遮掉線 → 連續譜與 sigma 下降),遮罩一路長到
    # 撞地板參數才停,所以最後一輪的範圍不是物理決定的。
    fit_mask = None
    if args.blank_region == "line1":
        f = STEP03 / "iter_line_mask.npy"
        if not f.exists():
            raise SystemExit(
                f"找不到 {f.name}。step3 必須以有存 history 的版本重跑一次:\n"
                f"  conda run -n astro python src/skymodel/step3_sky_basis.py "
                f"--methods {args.basis}")
        fit_mask = np.load(f)[0]

    with fits.open(CUBE, memmap=True) as hdul:
        hdr = hdul["DATA"].header
        D   = np.asarray(hdul["DATA"].data, np.float32)
        V   = np.asarray(hdul["STAT"].data, np.float32)

    nz, ny, nx = D.shape
    D     = D.reshape(nz, -1)
    V     = V.reshape(nz, -1)
    seg_f = seg.reshape(-1)
    # white != 0 只擋得掉「完全沒資料」;視野四周還有一圈 spaxel 是「幾乎沒資料」
    # —— 大氣色散(DAR)讓有效視野逐波長平移,邊界因此有約 1 px 寬的過渡帶,帶內
    # 的 spaxel 只在部分波長被覆蓋(實測有的只剩 46/3801 個通道)。這種 spaxel 的
    # 白光是個小負數(例 -5.97),不等於 0,所以一路被當成正常的 blank 帶到這裡,
    # 解出的係數可以爆到 1e6 量級(資料本身只有 1e2)。
    # step2/step3 靠加總與平均把它們稀釋掉了(mean_sky 中位僅差 0.012%),step5
    # 是唯一逐 spaxel 輸出的一步,沒有稀釋,必須自己擋。
    # 門檻 0.9:覆蓋率分佈是雙峰的(完整 84,053 個 vs 部分 1,052 個,中間幾乎沒有),
    # 0.5 到 0.95 之間結果相同。
    coverage = np.isfinite(D).sum(axis=0) / nz
    valid    = (white != 0).reshape(-1) & (coverage >= MIN_COVERAGE)
    templates = build_templates(best, wl_vac)
    sky_model = np.full((nz, ny * nx), np.nan, np.float32)
    A_map     = np.full((N_SRC, ny * nx), np.nan, np.float32)
    s_map     = np.full(ny * nx, np.nan, np.float32)

    blank = valid & (seg_f == 0)
    c = fit_blank(D[:, blank], sky, var=V[:, blank] if args.blank_chi2 else None,
                  fit_mask=fit_mask)
    sky_model[:, blank] = sky.T @ c
    s_map[blank]        = c[0]

    for rid in np.unique(seg_f[valid & (seg_f > 0)]):
        m = valid & (seg_f == rid)
        c = fit_source(D[:, m], V[:, m], sky, templates.get(int(rid)), s_fix=s_fix)
        sky_model[:, m] = sky.T @ c[N_SRC:]     # 只有天空進 sky_model,源要保留
        A_map[:, m]     = c[:N_SRC]
        s_map[m]        = c[N_SRC]

    sub = D - sky_model
    cube = lambda x: x.reshape(nz, ny, nx)
    fits.writeto(STEP05 / f"sky_model_{tag_out}.fits",  cube(sky_model), hdr, overwrite=True)
    fits.writeto(STEP05 / f"sky_subtracted_{tag_out}.fits", cube(sub),   hdr, overwrite=True)
    np.save(STEP05 / f"A_map_{tag_out}.npy", A_map.reshape(N_SRC, ny, nx))
    np.save(STEP05 / f"s_map_{tag_out}.npy", s_map.reshape(ny, nx))

    region = ("全部通道" if fit_mask is None
              else f"line1 {int(fit_mask.sum())}/{fit_mask.size} 通道")
    n_cut = int(((white != 0).reshape(-1) & (coverage < MIN_COVERAGE)).sum())
    print(f"blank {int(blank.sum()):,} ({'chi2 加權' if args.blank_chi2 else '未加權'},{region})"
          f"  source {int((valid & (seg_f > 0)).sum()):,}"
          f"  放模板的區域 {len(templates)}")
    print(f"覆蓋率 < {MIN_COVERAGE:.0%} 而剔除: {n_cut:,} spaxel (視野邊緣過渡帶) → 輸出為 NaN")
    print(f"saved -> {STEP05}")

if __name__ == "__main__":
    main()