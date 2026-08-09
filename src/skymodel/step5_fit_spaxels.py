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
import os

# BLAS 的執行緒數必須在 import numpy 之前設定 —— 函式庫載入時只讀一次。
# 這裡是逐 spaxel 呼叫 lsq_linear,每次的矩陣只有 3801x58,對 BLAS 來說太小:
# 分工同步的成本遠超過平行的好處,而 OpenMP 的執行緒預設忙碌等待,會一邊空轉
# 一邊燒 CPU。實測(K=54, 3801 通道)每個 spaxel 單執行緒 7.7 ms、8 執行緒
# 228.7 ms —— **慢 30 倍**。14,610 個源 spaxel 因此從 1.9 分變成 55.7 分。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import datetime
import json
import subprocess
import sys
import time
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

# 天空模型的訓練樣本 -> 資料夾名的第一段。命名的是「這個天空模型從哪些 spaxel
# 學來的」,不是「哪個 step 產生的」—— step 的編號會變,而且讀的人得先知道
# pipeline 才解得開;樣本組成則是自己解釋自己。
SKY_SAMPLE = {"step03": "blank", "step04d": "blank+src", "step04d_fake": "blank+fake"}


def _rel(p):
    """盡量存成相對於 repo 根的路徑;--sky-dir 傳進來的可能本來就是相對路徑,
    對絕對的 ROOT 做 relative_to 會拋例外,所以要兩邊都擋。"""
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT)
    except ValueError:
        return p


def run_name(args, sky_dir, n_templates, s_fix):
    """輸出資料夾名:{訓練樣本}_{分解}K{K}_src{源數}_{s 處理}[_{blank 非預設}]。

    只編進「會讓結果不同」的軸。blank 的兩個選項是預設值時不附加 —— 絕大多數
    run 都用預設,每個名字都掛著同樣的尾巴只會讓差異更難看出來。
    """
    s = ("sfree" if s_fix is None else
         f"s{s_fix:g}" if args.one_stage else f"s{s_fix:g}+free")
    return (f"{SKY_SAMPLE.get(sky_dir.name, sky_dir.name)}"
            f"_{args.basis}K{args.K}_src{n_templates}_{s}"
            + ("" if args.blank_s_fix is None else f"_bs{args.blank_s_fix:g}")
            + ("" if args.fake_region == "blank" else "_fakesrc")
            + ("_bchi2" if args.blank_chi2 else "")
            + ("" if args.blank_region == "all" else f"_{args.blank_region}"))


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

def fit_blank(D, sky, var=None, fit_mask=None, s_fix=None, _nonneg=True):
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
    if s_fix is not None:
        # s 固定:s·C_sky 先扣掉,只解天光線係數。這和 fit_source 的 s_fix 是
        # 同一個做法,但這裡必須關掉非負限制 —— 原本 lb[0] = 0 針對的是 s,
        # 扣掉之後第 0 個元素變成 c₁,而天光線係數的正負由 basis 的方向決定,
        # 硬壓成非負會扭曲擬合。
        c   = fit_blank(D - s_fix * sky[0][:, None], sky[1:], var, fit_mask,
                        _nonneg=False)
        out = np.full((sky.shape[0], D.shape[1]), np.nan)
        out[0], out[1:] = s_fix, c
        return out

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

    if not _nonneg:
        return coef
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

def fit_source(D, var, sky, T, s_fix=None, progress=False):
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
    progress : bool
        逐 spaxel 報進度。只在 spaxel 數 >= 500 的區域才印,小區域一瞬間就跑完,
        印了只是洗版。

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

    # 逐 spaxel 的進度。這裡是整個 step5 唯一昂貴的地方 —— Haro 11 一個區域就
    # 佔了 13,782 / 14,610 個源 spaxel,而且 seg ID 是升冪排序、它排第一,所以
    # 只在區域之間報進度的話,前幾十分鐘會完全沒有輸出。
    n   = D.shape[1]
    t0  = time.time()
    step = max(n // 20, 1)                # 最多 20 行,不洗版
    for j in range(n):
        if progress and n >= 500 and j and j % step == 0:
            el = time.time() - t0
            print(f"      {j:>6}/{n:,} ({100 * j / n:5.1f}%)  已用 {el:6.1f}s"
                  f"  預估剩餘 {el * (n - j) / j:6.1f}s", flush=True)
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
    ap.add_argument("--one-stage", action="store_true",
                    help="源區用單階段(s 固定解到底)。預設兩階段:先固定 s 解源的"
                         "振幅,扣掉源之後再讓 s 自由重解天空")
    ap.add_argument("--best", default=None,
                    help="源模型的來源檔。預設是舊 step4 的 best_{tag}.npz;"
                         "教授流程第 2 步要用 step4c 定案的分類,例如 "
                         "results/skymodel/step04b/classification_*.npz")
    ap.add_argument("--sky-dir", default=None,
                    help="天空模型(sky_continuum.npy 與 sky_basis_*.npy)的目錄。"
                         "預設 step03;要用精煉過的就指到 step04d")
    ap.add_argument("--blank-s-fix", type=float, default=None,
                    help="blank 區的天空連續譜係數也固定成這個值。預設 None = "
                         "自由 —— blank 正是量 s 的地方,那裡沒有源可以和 C_sky "
                         "簡併,所以本來不需要固定。設 1.0 是為了讓整個視場用"
                         "同一個 s,代價是天空連續譜的逐 spaxel 變化(實測 blank "
                         "的 s 中位 0.9995、16-84%% [0.985, 1.016])無處可去")
    ap.add_argument("--fake-region", choices=["blank", "source"], default="blank",
                    help="seg > 0 但不在源清單裡的區域(28 個假偵測)怎麼處理。"
                         "blank = 完全比照 blank(s 自由、依 --blank-chi2 加權),"
                         "預設,因為它們本來就是天空;source = 舊行為,走源的路徑"
                         "但不放模板(s 鎖在 s_fix、chi2 加權),留著做對照")
    ap.add_argument("--run", default=None,
                    help="輸出資料夾名。省略時由參數自動組出 —— "
                         "{訓練樣本}_{分解}K{K}_src{源數}_{s處理},例如 "
                         "blank+src_svdK54_src9_s1。自動組出的名字保證同樣的"
                         "設定永遠落在同一個資料夾,不會因為手打的字串不一致"
                         "而散成好幾份")
    args = ap.parse_args()
    s_fix = None if args.s_free else args.s_fix
    tag = (f"{args.basis}_K{args.K}_s_free" if s_fix is None
           else f"{args.basis}_K{args.K}_s_{s_fix}")
    # tag 只用來「讀」step3/step4d 的 basis 檔名;輸出改成一個 run 一個資料夾,
    # 名字由 run_name() 從參數組出來(見檔案上方)。
    STEP05.mkdir(parents=True, exist_ok=True)

    seg    = fits.getdata(STEP01 / "seg.fits")
    white  = fits.getdata(STEP01 / "whitelight.fits")
    wl_vac = air_to_vacuum(np.load(STEP03 / "wavelength.npy"))
    sky_dir = Path(args.sky_dir) if args.sky_dir else STEP03
    sky    = np.vstack([np.load(sky_dir / "sky_continuum.npy"),
                        np.load(sky_dir / f"sky_basis_{args.basis}_K{args.K}.npy")])
    print(f"天空模型來自 {sky_dir.name}")
    # 模板必須來自「同一個模型」的 step4 —— s 的處理方式不同,選出的
    # (模板, z) 就不同(實測 12/37 個源會換模板),混用等於用錯的答案。
    best_file = Path(args.best) if args.best else STEP04 / f"best_{tag}.npz"
    if not best_file.exists():
        raise SystemExit(f"找不到 {best_file}。" + ("" if args.best else
            f"step4 必須先以相同的 s 設定跑過:\n"
            f"  conda run -n astro python src/skymodel/step4_find_template.py "
            f"--id all --basis {args.basis} -K {args.K}"
            + (" --s-free" if s_fix is None else f" --s-fix {s_fix}")))
    best = np.load(best_file)
    print(f"源模型來自 {best_file.name}:{len(best['id'])} 個源")

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

    # 「假源」= seg > 0 但不在源清單裡的 28 個 SExtractor 偵測(雜訊尖峰、宇宙線、
    # 探測器條紋)。它們本來就是天空,所以預設完全比照 blank 處理。
    #
    # 舊版是 seg > 0 就走 fit_source 的路徑,於是假源落在一個說不通的第三類:
    # 沒有源模型(像 blank),但 s 被鎖在 s_fix、而且用 chi2 加權(像源)。
    # 那是判斷式的副作用不是設計 —— 天空模型在那 1,413 個 spaxel 上白白少了
    # 一個自由度(實測 blank 的 s 中位 0.9995,16-84% [0.985, 1.016])。
    src_ids = np.array(sorted(templates), dtype=seg_f.dtype)
    if args.fake_region == "blank":
        blank = valid & ~np.isin(seg_f, src_ids)
        rids  = src_ids
    else:
        blank = valid & (seg_f == 0)
        rids  = np.unique(seg_f[valid & (seg_f > 0)])
    n_src_tot = int((valid & np.isin(seg_f, rids)).sum())
    print(f"blank {int(blank.sum()):,} spaxel  (共用設計矩陣,一次解完)…",
          end="", flush=True)
    t0 = time.time()
    c = fit_blank(D[:, blank], sky, var=V[:, blank] if args.blank_chi2 else None,
                  fit_mask=fit_mask, s_fix=args.blank_s_fix)
    sky_model[:, blank] = sky.T @ c
    s_map[blank]        = c[0]
    print(f" {time.time() - t0:.1f}s", flush=True)

    # 進度以「已完成的 spaxel 數」計,不是「第幾個區域」—— 成本幾乎全部集中在
    # Haro 11(13,782 / 14,610 個源 spaxel),用區域數當進度會顯示成「36/37 很快
    # 跑完」然後卡在最後一個,看不出還要多久。
    print(f"source {n_src_tot:,} spaxel,{len(rids)} 個區域", flush=True)
    done, t0 = 0, time.time()
    for k, rid in enumerate(rids, 1):
        m = valid & (seg_f == rid)
        T = templates.get(int(rid))
        # 階段 1:源和天空一起解,s 固定 —— A·T 和 s·C_sky 都是平滑的連續譜,
        # 同時放自由的話天空會吸走源的光。
        c = fit_source(D[:, m], V[:, m], sky, T, s_fix=s_fix, progress=True)
        A_map[:, m] = c[:N_SRC]

        # 印在昂貴的那一步之後、分支之前 —— 分支底下有 continue,擺在迴圈尾端
        # 會漏掉一半的區域。剩下的矩陣乘法相對便宜,不影響進度的準確度。
        done += int(m.sum())
        el   = time.time() - t0
        print(f"  {k:>2}/{len(rids)}  ID {int(rid):>3}  "
              f"{'模板 ' + str(T.shape[1]) + ' 欄' if T is not None else '無模板  '}"
              f"  {int(m.sum()):>6} spaxel   累計 {done:>6,}/{n_src_tot:,}"
              f" ({100 * done / n_src_tot:5.1f}%)   已用 {el:6.1f}s"
              f"   預估剩餘 {el * (n_src_tot - done) / max(done, 1):6.1f}s",
              flush=True)

        if args.one_stage or T is None or s_fix is None:
            sky_model[:, m] = sky.T @ c[N_SRC:]   # 只有天空進 sky_model,源要保留
            s_map[m]        = c[N_SRC]
            continue
        # 階段 2:源已經扣掉,設計矩陣裡沒有 A·T,簡併不存在 —— s 沒有理由再
        # 被綁在 1.0。實測源區的殘差 mean 從 +1.3 掉到 0.00,那個偏移純粹是
        # 「連續譜的誤差在單階段裡無處可去」造成的。
        src = T @ np.nan_to_num(c[:T.shape[1]])
        src = np.where(np.all(np.isfinite(T), axis=1)[:, None], src, np.nan)
        c2  = fit_blank(D[:, m] - src, sky)
        sky_model[:, m] = sky.T @ c2
        s_map[m]        = c2[0]

    sub = D - sky_model
    cube = lambda x: x.reshape(nz, ny, nx)
    # 一個 run 一個資料夾,裡面檔名固定。平鋪的檔名塞不下六個會變動的軸
    # (天空模型的訓練樣本、分解方法、K、源清單、s 的處理、blank 的擬合選項),
    # 而且再長的檔名也記不住「--best 指向哪一份分類檔」—— 那寫進 meta.json。
    out = STEP05 / (args.run or run_name(args, sky_dir, len(templates), s_fix))
    out.mkdir(parents=True, exist_ok=True)
    fits.writeto(out / "sky_model.fits",      cube(sky_model), hdr, overwrite=True)
    fits.writeto(out / "sky_subtracted.fits", cube(sub),       hdr, overwrite=True)
    np.save(out / "A_map.npy", A_map.reshape(N_SRC, ny, nx))
    np.save(out / "s_map.npy", s_map.reshape(ny, nx))
    (out / "meta.json").write_text(json.dumps(dict(
        run=out.name,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        git_commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True,
                                  cwd=ROOT).stdout.strip(),
        cube=str(CUBE.relative_to(ROOT)), sky_dir=str(_rel(sky_dir)),
        best=args.best, basis=args.basis, K=args.K,
        s_fix=s_fix, one_stage=bool(args.one_stage),
        blank_weight="chi2" if args.blank_chi2 else "unweighted",
        blank_region=args.blank_region, fake_region=args.fake_region,
        blank_s_fix=args.blank_s_fix,
        n_blank=int(blank.sum()), n_source=int((valid & (seg_f > 0)).sum()),
        n_template_regions=len(templates),
        argv=sys.argv[1:]), ensure_ascii=False, indent=2))

    region = ("全部通道" if fit_mask is None
              else f"line1 {int(fit_mask.sum())}/{fit_mask.size} 通道")
    n_cut = int(((white != 0).reshape(-1) & (coverage < MIN_COVERAGE)).sum())
    print(f"blank {int(blank.sum()):,} ({'chi2 加權' if args.blank_chi2 else '未加權'},{region})"
          f"  source {int((valid & (seg_f > 0)).sum()):,}"
          f"  放模板的區域 {len(templates)}")
    print(f"覆蓋率 < {MIN_COVERAGE:.0%} 而剔除: {n_cut:,} spaxel (視野邊緣過渡帶) → 輸出為 NaN")
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()