"""逐 spaxel 擬合:固定 step4b 定出的 (類別, 模型, z),對每個 spaxel 解線性係數。

    source (seg > 0)   D(p,λ) = Σⱼ aⱼ(p)·Tⱼ(λ) + s(p)·C_sky(λ) + Σₖ cₖ(p)·Lₖ(λ)
    blank  (seg = 0)   D(p,λ) =                   s(p)·C_sky(λ) + Σₖ cₖ(p)·Lₖ(λ)

區域只分這兩類,而且只看 seg:**seg > 0 一律是源區域**,不論它有沒有進源清單、
也不論 SExtractor 抓到的是真源還是雜訊尖峰。清單只決定「有沒有模板可用」——
沒有模板的源區域,模型退化成 s·C_sky + Σcₖ·Lₖ,但 s 仍然鎖在 s_fix,天空長不高、
吃不到源的光。機制見下方 rids 那一段。

Tⱼ 是 step4b 決定的源模型(已紅移),形狀固定;逐 spaxel 只解係數的大小。
依 step4b 定出的組別,源模型有兩種寬度:

    star        該條 SDSS 恆星模板,1 欄
    galaxy/qso  對應的本徵譜,4 欄

z 不逐 spaxel 重解 —— 它是天體的性質,而且加總光譜的 S/N 比單一 spaxel 高
√N 倍(N = 該源的 spaxel 數),在單一 spaxel 上解 z 只會更差。

兩區的目標函數不同:source 最小化 chi2(以 1/sigma 加權),blank 預設最小化
平方誤差(不加權)。不加權時設計矩陣不隨 spaxel 改變,佔視場絕大部分的 blank
可用一次 pinv 全部解完;加權則必須逐 spaxel 求解。

blank 的兩個選項互相正交,可組出 2x2 方便比較:
    --blank-chi2               改以 1/var 加權(最小化 chi2)
    --blank-region line1       只在第一輪 line mask 的通道上擬合
「第一輪」是 estimate_continuum 的 iteration 1,只抓到最強的線。限縮範圍只改
「在哪些通道解係數」,天空模型仍然在全部通道上求值 —— 我們每個通道都要扣天空。

s 的空間場(--s-field)
---------------------
逐 spaxel 自由解 s 的話,源旁邊那格看到多出來的源光,唯一能解釋它的辦法就是把
自己的 s 抬高 —— **那個逐 spaxel 的自由度就是源光被天空模型吃掉的通道**。

--s-field 把 s 從「每格一個自由參數」換成「一個空間場在該位置的取值」:

    ① blank 先照原本的方式自由解一次        -> s_free（診斷用,會存檔）
    ② 只用**遠離所有源**的格建出場 s_hat    -> utils.build_s_field
    ③ blank 用 s_hat 重解(只解天光線係數)
    ④ 源區也用同一個 s_hat

一格的資料撼動不了那個場,源的光在天空模型裡就無處可去,只能留在殘差裡(= 被
保留)。

④ 不是可選的。若只在 blank 用場、源區仍鎖常數,兩邊用的是不同的 s 規則,邊界
上就會有一個階梯,乘上 C_sky 之後成為流量的跳變 —— 等於用新的階梯換掉舊的。
整個視場用同一個連續函數,s 這一項在邊界上才不會跳。

輸出的天空模型不含模板項 —— 扣掉的只有天空,源要保留。
"""
import os

# BLAS 的執行緒數必須在 import numpy 之前設定 —— 函式庫載入時只讀一次。
# 這裡是逐 spaxel 呼叫 lsq_linear,每次的矩陣對 BLAS 來說都太小:分工同步的
# 成本遠超過平行的好處,而 OpenMP 的執行緒預設忙碌等待,會一邊空轉一邊燒 CPU。
# 讓 BLAS 走單執行緒反而快得多。
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
from utils import DV_MAX, build_s_field, main_source_group

ROOT      = Path(__file__).resolve().parents[2]
WORK_DEFAULT = ROOT / "results/skymodel"
TPL_DIR   = ROOT / "data/sdss_templates"
EIGEN_GAL = ROOT / "data/eigen_galaxy_Bolton2012.fits"
EIGEN_QSO = ROOT / "data/qso_eigen_linear_55732.dat"
CUBE_DEFAULT = ROOT / "data/Haro11_NEpointing_wsky.fits"

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


def as_vector(s_fix, n):
    """把 s_fix 正規化成長度 n 的向量;None 原樣傳回。

    s_fix 可以是純量(全場同一個值)或一張逐 spaxel 的場。攤成向量之後,下游
    一律用 sv[j],不必到處寫 isscalar 判斷。純量走 broadcast_to,不會複製記憶體。
    """
    if s_fix is None:
        return None
    return np.broadcast_to(np.asarray(s_fix, float), (n,))


def run_name(args, sky_dir, n_templates, s_fix, seg_path):
    """輸出資料夾名:{訓練樣本}_{分解}K{K}_tpl{模板數}_{s 處理}[_{blank 非預設}][_seg]。

    只編進「會讓結果不同」的軸。blank 的兩個選項是預設值時不附加 —— 絕大多數
    run 都用預設,每個名字都掛著同樣的尾巴只會讓差異更難看出來。

    tpl 記的是「其中幾個源有模板」。源區域一律是「全部 seg > 0」,由 seg 決定、
    不由清單決定,寫進名字沒有資訊。
    """
    s = ("sfield" if args.s_field else
         "sfree" if s_fix is None else f"s{s_fix:g}")
    # 遮罩換了但名字沒變的話,兩個 run 會寫進同一個資料夾。預設的 seg.fits 不附加,
    # 非預設的把檔名去掉 "seg" 前綴接上去(seg_dil4.fits -> _dil4)。
    tag = ("" if seg_path.name == "seg.fits"
           else "_" + seg_path.stem.replace("seg_", ""))
    return (f"{SKY_SAMPLE.get(sky_dir.name, sky_dir.name)}"
            f"_{args.basis}K{args.K}_tpl{n_templates}_{s}"
            + ("" if args.blank_s_fix is None else f"_bs{args.blank_s_fix:g}")
            + ("_bchi2" if args.blank_chi2 else "")
            + ("" if args.blank_region == "all" else f"_{args.blank_region}")
            + tag)


def build_templates(best, lam_vac):
    """挑出要放模型的源,並把各自的源模型紅移到 MUSE 波長格點。

    step4b 已對第 1 個源係數施加非負限制,A[0] = 0 代表最佳解落在邊界上,
    也就是這個模型對該源沒有貢獻,這種源不放模型。

    模型依 step4b 定出的組別而不同:恆星用該條 SDSS 模板(1 欄),
    星系/QSO 用對應的本徵譜(4 欄)。

    Returns
    -------
    dict
        {segmentation ID: 已紅移到 lam_vac 的模型, shape (nz, n_comp)}
    """
    eigen = {"galaxy": load_eigen_galaxy(EIGEN_GAL), "qso": load_eigen_qso(EIGEN_QSO)}
    out   = {}
    # 不能只看 A[:, 0] —— 本徵譜沒有非負約束,主導成分的係數可以是 0 或負的,
    # 源仍然由其餘三條扛著。判準是「四個係數全部為 0」才算沒有模型;
    # NaN 是恆星未用到的欄位。
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
    天光線係數幾乎沒有貢獻,把它們放進來只會讓設計矩陣接近奇異。

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
    那樣共用 pinv。A 與 s 受非負限制,與 step4b 一致。

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
    s_fix : float, ndarray or None
        天空連續譜的固定係數。純量 = 全區同一個值;shape (n,) 的陣列 = 逐 spaxel
        給定(--s-field 走這條);None 表示 s 為自由參數。
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
    # 硬壓只會扭曲擬合,與 step4b 的處理一致。
    lb = np.full(p, -np.inf)
    if n_comp == 1:
        lb[0] = 0.0
    if s_fix is None:
        lb[n_comp] = 0.0
    ub = np.full(p, np.inf)

    # sv 是攤平成逐 spaxel 的版本。純量與場走同一條路 —— 廣播讓
    # sv * sky[0][:, None] 自動變成 (nz, n),每格用自己的 s。
    sv = as_vector(s_fix, D.shape[1])
    y = D if sv is None else D - sv * sky[0][:, None]
    good = (np.isfinite(y) & np.isfinite(var) & (var > 0)
            & np.all(np.isfinite(design), axis=0)[:, None])

    # 逐 spaxel 的進度。這裡是整個 step5 唯一昂貴的地方 —— 源 spaxel 絕大多數集中
    # 在最大的那個區域,只在區域之間報進度的話,那個區域跑多久就有多久沒有輸出。
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
        out[N_SRC, j]      = th[n_comp] if sv is None else sv[j]
        out[N_SRC + 1:, j] = th[n_comp + (sv is None):]
    return out

def main():
    ap = argparse.ArgumentParser(description="逐 spaxel 擬合天空與源模板")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=25,
                    help="天光線 basis 條數;必須和 step3/step4b 用的 K 相同")
    ap.add_argument("--s-fix", type=float, default=1.0,
                    help="源區域的天空連續譜係數固定值(預設 1.0)。blank 區一律保持自由。")
    ap.add_argument("--s-free", action="store_true",
                    help="改讓源區域的 s 成為自由參數,覆蓋 --s-fix")
    ap.add_argument("--blank-chi2", action="store_true",
                    help="blank 區改用 chi2 加權;不給則最小化未加權平方誤差。")
    ap.add_argument("--blank-region", choices=["all", "line1"], default="all",
                    help="blank 區解係數使用的通道:all=全部,line1=第一輪 line mask。")
    ap.add_argument("--s-field", action="store_true",
                    help="把 s 從「逐 spaxel 自由」換成「一個空間場」,blank 與源區"
                         "都用同一張 s_hat(x,y)。見檔頭說明。與 --s-free / "
                         "--blank-s-fix 互斥")
    ap.add_argument("--sf-r-far", type=float, default=15.0,
                    help="建場:訓練點必須離**任何**源這麼遠(px),避開源的 PSF 翼")
    ap.add_argument("--sf-r-far-haro", type=float, default=50.0,
                    help="建場:主源專用的加碼半徑。主源的延展暈比小源的 PSF 翼"
                         "遠得多,需要另外加大;設 0 表示不加碼")
    ap.add_argument("--sf-exclude-box", type=int, nargs=4, default=None,
                    metavar=("Y0", "Y1", "X0", "X1"),
                    help="建場:這個框裡的格不當訓練樣本(仍然會被扣天空)。"
                         "用途與 step3 的 --exclude-box 相同,而且兩者應該給同一個框")
    ap.add_argument("--sf-clip", type=float, default=8.0,
                    help="建場:|s − 中位| > clip x 穩健散布 的格不當訓練樣本")
    ap.add_argument("--main-dv-max", type=float, default=DV_MAX,
                    help="主源分組:相鄰成員的星系分支紅移離主源多少 km/s 之內"
                         "才算主星系的一部分")
    ap.add_argument("--best", required=True,
                    help="源模型的來源檔:step4c 定案的分類,例如 "
                         "{work}/step04b/classification_*.npz。"
                         "沒有預設值 —— 它決定每個源用哪個模板與紅移,"
                         "猜錯一個目錄就會安靜地套用另一次擬合的答案")
    ap.add_argument("--seg", default=None,
                    help="用哪一份 segmentation 劃分源與 blank。預設是 SExtractor 的 "
                         "step01/seg.fits;指到 experiments/dilate_seg.py 產生的 seg_dil{r}.fits 就是"
                         "膨脹版。**要和 step3 用同一份**,否則天空模型的訓練樣本"
                         "和擬合時的區域劃分會對不起來")
    ap.add_argument("--sky-dir", default=None,
                    help="天空模型(sky_continuum.npy 與 sky_basis_*.npy)的目錄。"
                         "預設 step03,也就是只從 blank 學出來的那一份;資料夾名"
                         "會編進輸出的 run 名稱(見 SKY_SAMPLE)")
    ap.add_argument("--blank-s-fix", type=float, default=None,
                    help="blank 區的天空連續譜係數也固定成這個值。預設 None = "
                         "自由 —— blank 正是量 s 的地方,那裡沒有源可以和 C_sky "
                         "簡併,所以本來不需要固定。設 1.0 是為了讓整個視場用"
                         "同一個 s,代價是天空連續譜真實的逐 spaxel 變化無處可去")
    ap.add_argument("--work", default=str(WORK_DEFAULT),
                    help="這顆 cube 的工作區(底下有 step01/step03/step04/step05)。"
                         "一個 pointing 一個工作區,結構相同、彼此不干擾")
    ap.add_argument("--cube", default=str(CUBE_DEFAULT),
                    help="要扣天空的 cube。**必須是含天空的 wsky**")
    ap.add_argument("--run", default=None,
                    help="輸出資料夾名。省略時由參數自動組出 —— "
                         "{訓練樣本}_{分解}K{K}_tpl{模板數}_{s處理},例如 "
                         "blank_svdK54_tpl68_s1_bs1。自動組出的名字保證同樣的"
                         "設定永遠落在同一個資料夾,不會因為手打的字串不一致"
                         "而散成好幾份")
    ap.add_argument("--sf-file", default=None,
                    help="改用外部算好的 s 場(.npy, shape = (ny, nx))取代 "
                         "build_s_field 的結果。用途:在 experiments/ 裡試新的建場"
                         "方法,不必把還沒驗證的做法寫進 pipeline")
    ap.add_argument("--save-stat", action="store_true",
                    help="sky_subtracted.fits 改成 MUSE 的多副檔格式(空的 primary + "
                         "DATA + STAT),STAT 直接沿用輸入 cube 的。**那份 STAT 不含"
                         "天空模型自己的不確定度** —— 我們沒有 basis 係數的完整共變異"
                         "數,硬加一個猜的數字比誠實地不加更糟。檔案會變成兩倍大。"
                         "省略 = 只寫 DATA 到 primary HDU")
    args = ap.parse_args()
    # 三個選項各自宣稱 s 該怎麼決定,同時給就是矛盾。默默讓其中一個贏會產生
    # 一個名字和內容對不起來的 run —— 直接擋掉。
    if args.s_field and (args.s_free or args.blank_s_fix is not None):
        raise SystemExit("--s-field 與 --s-free / --blank-s-fix 互斥:"
                         "前者說 s 由空間場決定,後者說 s 自由或固定成常數。")
    work = Path(args.work)
    STEP01, STEP03 = work / "step01", work / "step03"
    STEP05 = work / "step05"
    CUBE = Path(args.cube)
    print(f"工作區 {work}   cube {CUBE.name}")
    s_fix = None if args.s_free else args.s_fix
    tag = (f"{args.basis}_K{args.K}_s_free" if s_fix is None
           else f"{args.basis}_K{args.K}_s_{s_fix}")
    # tag 只用來「讀」step3 的 basis 檔名;輸出改成一個 run 一個資料夾,
    # 名字由 run_name() 從參數組出來(見檔案上方)。
    STEP05.mkdir(parents=True, exist_ok=True)

    seg_path = Path(args.seg) if args.seg else STEP01 / "seg.fits"
    seg    = fits.getdata(seg_path)
    white  = fits.getdata(STEP01 / "whitelight.fits")
    print(f"segmentation: {seg_path.name}  源 spaxel {int((seg > 0).sum()):,}")
    wl_vac = air_to_vacuum(np.load(STEP03 / "wavelength.npy"))
    sky_dir = Path(args.sky_dir) if args.sky_dir else STEP03
    sky    = np.vstack([np.load(sky_dir / "sky_continuum.npy"),
                        np.load(sky_dir / f"sky_basis_{args.basis}_K{args.K}.npy")])
    print(f"天空模型來自 {sky_dir.name}")
    # 模板必須來自「同一個模型」的 step4b —— s 的處理方式不同,選出的
    # (模板, z) 就不同,混用等於用錯的答案。
    best_file = Path(args.best)
    if not best_file.exists():
        raise SystemExit(f"找不到 {best_file}")
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
        hdr_stat = hdul["STAT"].header      # --save-stat 要原樣帶著它一起寫出去
        D   = np.asarray(hdul["DATA"].data, np.float32)
        V   = np.asarray(hdul["STAT"].data, np.float32)

    nz, ny, nx = D.shape
    D     = D.reshape(nz, -1)
    V     = V.reshape(nz, -1)
    seg_f = seg.reshape(-1)
    # white != 0 只擋得掉「完全沒資料」;視野四周還有一圈 spaxel 是「幾乎沒資料」
    # —— 大氣色散(DAR)讓有效視野逐波長平移,邊界因此有約 1 px 寬的過渡帶,帶內
    # 的 spaxel 只在部分波長被覆蓋。這種 spaxel 的白光是個小負數而不是 0,所以
    # 一路被當成正常的 blank 帶到這裡,而通道太少會讓解出的係數爆掉。
    # step2/step3 靠加總與平均把它們稀釋掉了,step5 是唯一逐 spaxel 輸出的一步,
    # 沒有稀釋,必須自己擋。門檻取 MIN_COVERAGE:覆蓋率的分佈是雙峰的(完整 vs
    # 只剩零星通道,中間幾乎沒有),所以門檻落在兩峰之間的哪裡並不敏感。
    coverage = np.isfinite(D).sum(axis=0) / nz
    valid    = (white != 0).reshape(-1) & (coverage >= MIN_COVERAGE)
    templates = build_templates(best, wl_vac)
    sky_model = np.full((nz, ny * nx), np.nan, np.float32)
    A_map     = np.full((N_SRC, ny * nx), np.nan, np.float32)
    s_map     = np.full(ny * nx, np.nan, np.float32)

    # 區域只分兩類,而且只看 seg:seg > 0 是源區域,seg = 0 是 blank。
    # 「哪些源進清單」不再影響這個劃分 —— 清單只決定「有沒有模板可用」。
    #
    # 為什麼不能讓沒模板的源掉進 blank:blank 的 s 是自由的,而 s·C_sky 和源的
    # 連續譜都是平滑的,最小平方分不開,於是 s 會長高去吞源的光,甚至過衝到把
    # 源扣成負的。
    #
    # 反過來,一個假偵測被當成源區域,代價只是該處局部的殘差稍差,而且各區域是
    # 分開解的,真源與 blank 逐位元不受影響。兩種錯誤極度不對稱,所以這裡不做
    # 取捨:全部當源。
    # 沒有模板的區域走 fit_source 但 T = None —— 沒有源模型,但 s 鎖在 s_fix,
    # 天空不會長高去吃它,源的光原封不動留在殘差裡。
    blank     = valid & (seg_f == 0)
    rids      = np.unique(seg_f[valid & (seg_f > 0)])
    n_src_tot = int((valid & (seg_f > 0)).sum())
    print(f"blank {int(blank.sum()):,} spaxel  (共用設計矩陣,一次解完)…",
          end="", flush=True)
    t0 = time.time()
    c = fit_blank(D[:, blank], sky, var=V[:, blank] if args.blank_chi2 else None,
                  fit_mask=fit_mask, s_fix=args.blank_s_fix)
    sky_model[:, blank] = sky.T @ c
    s_map[blank]        = c[0]
    print(f" {time.time() - t0:.1f}s", flush=True)

    # ---- s 的空間場 -------------------------------------------------------
    # 上面那次自由解已經把每個 blank spaxel 的 s 算出來了(c[0]),建場不需要
    # 再讀一次 cube、也不需要再解一次 —— 直接拿它當原料。第二次 fit_blank
    # 只是一次 pinv,幾秒鐘;step5 的成本全部在源區的逐格 lsq_linear。
    s_hat = None
    if args.s_field:
        s_free = np.full(ny * nx, np.nan)
        s_free[blank] = c[0]
        s2d = s_free.reshape(ny, nx)
        # 訓練樣本要排除解不出來的格 —— build_s_field 內部用 np.median 定中心,
        # 混進 NaN 會讓整個中心變成 NaN,連帶把 clip 全部作廢。
        ok2d = blank.reshape(ny, nx) & np.isfinite(s2d)
        t0 = time.time()
        sf_box = None
        if args.sf_exclude_box:
            by0, by1, bx0, bx1 = args.sf_exclude_box
            yy, xx = np.mgrid[0:ny, 0:nx]
            sf_box = (yy >= by0) & (yy <= by1) & (xx >= bx0) & (xx <= bx1)
        # 紅移取自 --best 那一份擬合的同一個目錄 —— 分類和紅移必須來自同一次
        # step4b,否則遮罩用的是 A 的紅移、模板用的是 B 的分類
        mg, mids, mk = main_source_group(seg, white, best_file.parent,
                                         args.main_dv_max)
        print(f"  主源(最亮像素 y={mk[0]}, x={mk[1]} 所在的整團):{len(mids)} 個 ID"
              f",共 {int(mg.sum()):,} px"
              f"(紅移相符 <= {args.main_dv_max:g} km/s)")
        # 用關鍵字傳 —— 位置參數裡夾一個 None 佔位,漏掉一個就會安靜地錯位
        s_hat, sf_train = build_s_field(
            s2d, seg, ok2d, args.sf_r_far, args.sf_r_far_haro or None,
            args.sf_clip, exclude=sf_box, main=mg)
        if args.sf_file:
            # 用外部算好的場取代。訓練遮罩仍然照上面算,因為 meta 要記錄它,
            # 而且外部的場理應是用同一組訓練點建的 —— 形狀對不上就直接停,
            # 錯用一張別顆 pointing 的場,結果看起來完全正常。
            ext = np.load(args.sf_file)
            if ext.shape != (ny, nx):
                raise SystemExit(f"★ --sf-file 的形狀 {ext.shape} 與這顆 cube "
                                 f"的 ({ny}, {nx}) 不符")
            print(f"  --sf-file:改用 {args.sf_file}"
                  f"(與內建場的中位差 {np.nanmedian(ext - s_hat):+.5f})")
            s_hat = ext
        s_hat = s_hat.ravel()
        print(f"s 空間場:訓練 {int(sf_train.sum()):,} 格 "
              f"(離源 > {args.sf_r_far:g} px" +
              (f",Haro 11 > {args.sf_r_far_haro:g} px" if args.sf_r_far_haro else "")
              + f",clip {args.sf_clip:g} sigma)   {time.time() - t0:.1f}s")
        print(f"  s_hat 中位 {np.nanmedian(s_hat[valid]):.5f}   "
              f"blank 自由解的中位 {np.nanmedian(c[0]):.5f}   "
              f"NaN {int((~np.isfinite(s_hat[valid])).sum())} 格")
        # blank 用場重解:s 鎖住,只剩 K−1 個天光線係數是自由的。
        t0 = time.time()
        c = fit_blank(D[:, blank], sky,
                      var=V[:, blank] if args.blank_chi2 else None,
                      fit_mask=fit_mask, s_fix=s_hat[blank])
        sky_model[:, blank] = sky.T @ c
        s_map[blank]        = c[0]
        print(f"  blank 以場重解完成  {time.time() - t0:.1f}s", flush=True)

    # 進度以「已完成的 spaxel 數」計,不是「第幾個區域」—— 區域的大小差好幾個
    # 數量級,用區域數當進度會顯示成「大部分很快跑完」然後卡在最大的那個,
    # 看不出還要多久。
    n_notpl = sum(1 for r in rids if int(r) not in templates)
    print(f"source {n_src_tot:,} spaxel,{len(rids)} 個區域"
          f"({len(rids) - n_notpl} 個有模板,{n_notpl} 個無模板但仍走源的路徑)",
          flush=True)
    done, t0 = 0, time.time()
    for k, rid in enumerate(rids, 1):
        m = valid & (seg_f == rid)
        T = templates.get(int(rid))
        # 源和天空一起解,s 固定 —— A·T 和 s·C_sky 都是平滑的連續譜,同時放自由
        # 的話天空會吸走源的光。--s-field 時 s 鎖的是該格的 s_hat 而不是常數,
        # 這樣邊界內外用的是同一個連續函數,s 這一項在邊界上不會跳。
        c = fit_source(D[:, m], V[:, m], sky, T,
                       s_fix=s_fix if s_hat is None else s_hat[m], progress=True)
        A_map[:, m] = c[:N_SRC]

        done += int(m.sum())
        el   = time.time() - t0
        print(f"  {k:>2}/{len(rids)}  ID {int(rid):>3}  "
              f"{'模板 ' + str(T.shape[1]) + ' 欄' if T is not None else '無模板  '}"
              f"  {int(m.sum()):>6} spaxel   累計 {done:>6,}/{n_src_tot:,}"
              f" ({100 * done / n_src_tot:5.1f}%)   已用 {el:6.1f}s"
              f"   預估剩餘 {el * (n_src_tot - done) / max(done, 1):6.1f}s",
              flush=True)

        sky_model[:, m] = sky.T @ c[N_SRC:]   # 只有天空進 sky_model,源要保留
        s_map[m]        = c[N_SRC]

    sub = D - sky_model
    cube = lambda x: x.reshape(nz, ny, nx)
    # 一個 run 一個資料夾,裡面檔名固定。平鋪的檔名塞不下六個會變動的軸
    # (天空模型的訓練樣本、分解方法、K、源清單、s 的處理、blank 的擬合選項),
    # 而且再長的檔名也記不住「--best 指向哪一份分類檔」—— 那寫進 meta.json。
    out = STEP05 / (args.run or run_name(args, sky_dir, len(templates), s_fix, seg_path))
    out.mkdir(parents=True, exist_ok=True)
    fits.writeto(out / "sky_model.fits", cube(sky_model), hdr, overwrite=True)
    if args.save_stat:
        # MUSE 的標準結構:空的 primary + DATA + STAT。下游用
        # `h["DATA"] if "DATA" in h else h[0]` 讀,兩種格式都吃得下。
        h2 = hdr_stat.copy()
        h2["HISTORY"] = ("STAT copied unchanged from the input cube; it does NOT "
                         "include the uncertainty of the sky model itself.")
        fits.HDUList([fits.PrimaryHDU(),
                      fits.ImageHDU(cube(sub), hdr, name="DATA"),
                      fits.ImageHDU(cube(V).astype(np.float32), h2, name="STAT")
                      ]).writeto(out / "sky_subtracted.fits", overwrite=True)
    else:
        fits.writeto(out / "sky_subtracted.fits", cube(sub), hdr, overwrite=True)
    np.save(out / "A_map.npy", A_map.reshape(N_SRC, ny, nx))
    np.save(out / "s_map.npy", s_map.reshape(ny, nx))   # 這個 run 實際用掉的 s
    if s_hat is not None:
        # 場本身,以及建場的原料(blank 的自由解)。兩個都存:s_map 已經等於場了,
        # 沒有 s_free 的話就再也回不去「場是從什麼東西建出來的」。
        np.save(out / "s_hat.npy",  s_hat.reshape(ny, nx).astype(np.float32))
        np.save(out / "s_free.npy", s_free.reshape(ny, nx).astype(np.float32))
    (out / "meta.json").write_text(json.dumps(dict(
        run=out.name,
        created=datetime.datetime.now().isoformat(timespec="seconds"),
        git_commit=subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                  capture_output=True, text=True,
                                  cwd=ROOT).stdout.strip(),
        cube=str(_rel(CUBE)), seg=str(_rel(seg_path)),
        sky_dir=str(_rel(sky_dir)),
        best=args.best, basis=args.basis, K=args.K,
        s_fix=s_fix,
        s_field=bool(args.s_field),
        s_field_params=(dict(r_far=args.sf_r_far, r_far_haro=args.sf_r_far_haro,
                             clip=args.sf_clip,
                             exclude_box=args.sf_exclude_box,
                             main_dv_max=args.main_dv_max,
                             main_ids=[int(i) for i in mids],
                             n_train=int(sf_train.sum()))
                        if args.s_field else None),
        blank_weight="chi2" if args.blank_chi2 else "unweighted",
        blank_region=args.blank_region, blank_s_fix=args.blank_s_fix,
        n_blank=int(blank.sum()), n_source=int((valid & (seg_f > 0)).sum()),
        n_source_regions=len(rids), n_template_regions=len(templates),
        save_stat=bool(args.save_stat),
        argv=sys.argv[1:]), ensure_ascii=False, indent=2))

    region = ("全部通道" if fit_mask is None
              else f"line1 {int(fit_mask.sum())}/{fit_mask.size} 通道")
    n_cut = int(((white != 0).reshape(-1) & (coverage < MIN_COVERAGE)).sum())
    print(f"blank {int(blank.sum()):,} ({'chi2 加權' if args.blank_chi2 else '未加權'},{region})"
          f"  source {n_src_tot:,}"
          f"  源區域 {len(rids)}(放模板 {len(rids) - n_notpl})")
    print(f"覆蓋率 < {MIN_COVERAGE:.0%} 而剔除: {n_cut:,} spaxel (視野邊緣過渡帶) → 輸出為 NaN")
    print(f"saved -> {out}")

if __name__ == "__main__":
    main()