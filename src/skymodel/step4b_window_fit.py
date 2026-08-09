"""源的模板擬合 —— 單階段,固定波長視窗,天光線通道排除在 chi2 之外。

和 step4_find_template.py 的差別

    step4    33 條模板(恆星+星系+QSO)一起比 chi2_all,誰低誰贏;chi2 用全部
             3801 個通道,包含天光線區。
    本檔     恆星模板與星系本徵譜在**同一組通道**上各自擬合,reduced chi2
             低的一組勝出,同時定出紅移。chi2 只算在指定視窗內、且不是天光線
             的通道上。

規格來自 reminder.txt:

    Fit source
    1. use line mask
    2. do not use line specturm
    3. 4800-6000 for star and non star
    4. 4800-7000 for galaxt to find redshift

為什麼排除天光線通道:那些通道的殘差被天空扣除的誤差主導,不是源的資訊。
把它們算進 chi2,等於讓「哪個模板比較能吸收天空殘差」去決定分類與紅移。
(blank 區的規則正好相反 —— 那裡只用線區,因為要學的就是天空。)

為什麼用固定視窗:step4 讓每個候選各自用它覆蓋到的通道,於是 n_good 隨 z 變,
chi2(z) 出現純粹來自通道數的階梯。固定視窗之後,星系本徵譜(靜止 1183–9840 A)
在 z = 0–1.5 全程蓋滿 4800–7000 A,所有候選的通道集合完全相同 —— 階梯消失,
chi2 之間可以直接相減。

為什麼分類不用絕對門檻:本檔原本是兩段式 —— 第 1 段用一個絕對門檻
(reduced chi2 < 2.0)問「像不像恆星」,通過就停,不通過才進第 2 段掃星系。
動機是讓「哪一組都不像」成為可以說出口的答案。但實測**沒有任何源通得過**
那個門檻:天光線殘差與流量刻度誤差讓所有源的 reduced chi2 都遠大於 2,
於是 group 永遠是 galaxy,而真正的判定一直是事後在 step4c 手算的
「同一組通道上誰的 reduced chi2 低」。現在把那個比較放回這裡,一次擬合、
直接比大小。門檻拿掉之後,分類的可信度改由**兩組冠軍的差距**來表達 ——
star_red_chi2 與 gal_red_chi2 都寫進輸出,差距小就代表這個分類不穩。

兩個視窗必須相同:reduced chi2 = chi2 / (n_good - n_param),分母裡的
n_good 由通道集合決定。視窗不同就不是同一個統計量,比大小沒有意義。
main() 會擋下不一致的設定。

    conda run -n astro python src/skymodel/step4b_window_fit.py --id all -K 54 \\
        --spec-dir results/skymodel/step02_eso --s-fix 0.0 \\
        --star-window 4700 8000 --gal-window 4700 8000 --line-mask-iter 1
"""
import os

# BLAS 的執行緒數必須在 import numpy 之前設定 —— 函式庫載入時只讀一次。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from templates import (load_sdss_template, load_eigen_galaxy, redshift_to_grid,
                       air_to_vacuum)
from utils import load_line_masks

ROOT      = Path(__file__).resolve().parents[2]
STEP02    = ROOT / "results/skymodel/step02"      # segmentation footprint
STEP02B   = ROOT / "results/skymodel/step02b"     # 圓形孔徑 (step2b_aperture.py)
STEP03    = ROOT / "results/skymodel/step03"
STEP04B   = ROOT / "results/skymodel/step04b"      # 不碰 step04/,兩套結果並存
TPL_DIR   = ROOT / "data/sdss_templates"
EIGEN_GAL = ROOT / "data/eigen_galaxy_Bolton2012.fits"

STAR_IDX = range(0, 23)     # spDR2-000..022 是恆星(docs/sdss-templates.md 第 2 節)
GAL_IDX  = range(23, 29)    # 023 早型、024-026、027 晚型、028 LRG,共 6 條星系模板
N_SRC    = 4                # A 欄位固定寬度:本徵譜 4 條,恆星只用第 0 欄

# 教授(2026-08-06)看過白光圖後指定:37 個 SExtractor 偵測裡只有這 11 個是真源,
# 其餘是雜訊尖峰、宇宙線、探測器條紋。在假源上算 chi2、排名、統計中位數都無效。
# 定義在這裡、由診斷程式一起 import —— 分成幾份寫的話,改了一邊就會有一支程式
# 還在畫假源,而那種錯誤從圖上看不出來。
KEEP_IDS = (1, 10, 12, 13, 14, 24, 27, 28, 30, 34, 35)

# 兩段各自的波長視窗 (A, 空氣波長)。改這裡就等於改預設值;
# 也可以在命令列用 --star-window / --gal-window 覆蓋,不必動程式。
# 視窗會編進輸出的 tag,所以不同視窗的結果各存各的,不互相覆蓋。
# 下限三者一致取 4600 —— MUSE 的第一個通道在 4599.7 A,所以 4600 就是「從頭開始」。
# 註:reminder.txt 記的是 4800;改成 4600 是把藍端那 160 個通道也放進來,
# 三個視窗共用同一個起點,彼此的差別才單純是「右邊延伸到哪」。
STAR_WINDOW = (4600.0, 6000.0)      # 恆星模板的擬合視窗
GAL_WINDOW  = (4600.0, 7000.0)      # 星系本徵譜的擬合視窗(須與上者相同)
FULL_RANGE  = (4600.0, 9400.0)      # MUSE 全波段,拿來當對照組用

_SHARED = {}


def make_tag(basis, K, s_fix, star_window, gal_window, sky_basis, line_iter,
             cumulative=True, aperture=False, suffix=""):
    """輸出檔名。凡是會改變結果的設定都編進去,重跑才不會靜靜蓋掉上一次。

    視窗與遮罩輪次都在裡面 —— 它們直接決定哪些通道進 chi2,不同設定的結果
    是不同的科學產物,必須並存。診斷程式也呼叫這個函式,兩邊的命名規則永遠
    一致;分成兩份寫的話,改了一邊忘了另一邊會變成「讀到錯的檔案」。
    """
    base = f"{basis}_K{K}" if sky_basis else "nobasis"
    return (f"{base}_s{'free' if s_fix is None else s_fix}"
            f"_{star_window[0]:.0f}-{star_window[1]:.0f}"
            f"_{gal_window[0]:.0f}-{gal_window[1]:.0f}"
            f"_L{line_iter}{'cum' if cumulative else 'raw'}"
            + ("_ap" if aperture else "") + suffix)


def scan_object(flux, var, sky, jobs, lam_muse, fit, s_fix=None,
                allow_partial=False):
    """在指定的通道集合 fit 上,對一條加總光譜掃描模板與紅移。

    fit 是布林陣列,長度等於通道數 —— 波長視窗與天光線遮罩都已經合併進去。
    擬合與評分用同一個集合,所以 chi2 之間可以直接比較(step4 是擬合用覆蓋
    通道、評分用全部通道,那是為了處理覆蓋不同的候選;固定視窗之後沒這問題)。

    A 與 s 的非負限制、s_fix 的意義都和 step4 相同,不重複說明。
    """
    base = (fit & np.isfinite(flux) & np.isfinite(var) & (var > 0)
            & np.all(np.isfinite(sky), axis=0))
    sig  = np.sqrt(np.where(var > 0, var, 1.0))
    n_full = int(base.sum())            # 資料端可用的通道數,所有候選的共同上限

    if s_fix is None:
        sky_free, y, s_free = sky, flux, True
    else:
        sky_free, y, s_free = sky[1:], flux - s_fix * sky[0], False

    skyw = np.ascontiguousarray((sky_free / sig).T)
    yw   = y / sig

    results = []
    for group, name, spline, z_grid in jobs:
        n_comp = 1 if spline.c.ndim == 1 else spline.c.shape[1]
        p      = sky_free.shape[0] + n_comp

        lb = np.full(p, -np.inf)
        if n_comp == 1:
            lb[0] = 0.0                 # 單一模板恆正,A ≥ 0 等價於「源不發負的光」
        if s_free:
            lb[n_comp] = 0.0
        ub = np.full(p, np.inf)

        for z in z_grid:
            T = redshift_to_grid(spline, z, lam_muse)
            if T.ndim == 1:
                T = T[:, None]
            good = base & np.all(np.isfinite(T), axis=1)
            n    = int(good.sum())
            if n <= p:                  # 至少留 1 個自由度,否則 reduced chi2 無定義
                continue
            # 覆蓋不滿整個視窗的候選直接丟掉。chi2 是「加總」,通道少的候選天生
            # 就小 —— 不擋的話,掃描會跑去模板剛好只覆蓋到幾個通道的那個 z,
            # 得到 chi2/dof = 0.07 這種看似完美其實沒有資料的解。
            # (SDSS 星系模板靜止只到 3500 A,z > 0.343 就蓋不住 4700-8000。)
            if not allow_partial and n < n_full:
                continue

            M = np.empty((n, p))
            M[:, :n_comp] = T[good] / sig[good][:, None]
            M[:, n_comp:] = skyw[good]

            fitres = lsq_linear(M, yw[good], bounds=(lb, ub), method="bvls")
            theta  = fitres.x
            chi2   = 2.0 * fitres.cost

            # 源光譜要看整段(不只視窗內)有沒有跑到負的 —— 負流量是模型的物理
            # 問題,不會因為我們沒把那段算進 chi2 就不存在。
            src      = np.nan_to_num(T, nan=0.0) @ theta[:n_comp]
            m_all    = src + theta[n_comp:] @ sky_free
            chi2_all = float((((y - m_all) / sig) ** 2)[base].sum())
            ok       = np.isfinite(flux) & np.all(np.isfinite(T), axis=1)

            results.append(dict(group=group, template=name, z=float(z),
                    A=theta[:n_comp], s=theta[n_comp] if s_free else s_fix,
                    chi2=chi2, chi2_all=chi2_all, red_chi2=chi2 / (n - p),
                    n_good=n, src_min=float(src[ok].min())))

    return sorted(results, key=lambda r: r["chi2"])


def _save_scan(path, results):
    """把一段掃描的完整結果寫成 npz,欄位和 step4 相同,診斷程式可以直接讀。"""
    A = np.full((len(results), N_SRC), np.nan)
    for i, x in enumerate(results):
        A[i, :len(x["A"])] = x["A"]
    np.savez(path, A=A,
             group=np.array([x["group"] for x in results]),
             template=np.array([x["template"] for x in results]),
             z=np.array([x["z"] for x in results]),
             s=np.array([x["s"] for x in results]),
             chi2=np.array([x["chi2"] for x in results]),
             chi2_all=np.array([x["chi2_all"] for x in results]),
             red_chi2=np.array([x["red_chi2"] for x in results]),
             n_good=np.array([x["n_good"] for x in results]),
             src_min=np.array([x["src_min"] for x in results]))


def _scan_one(t):
    """單階段擬合單一源:恆星模板與星系本徵譜在同一組通道上競爭。

    共用資料由 fork 繼承,不經過 pickle。

    為什麼不再分兩段
        舊版第 1 段用一個**絕對門檻**問「像不像恆星」(reduced chi2 < 2.0),
        通過就停,不通過才進第 2 段掃星系。實測**沒有任何源通得過那個門檻**
        —— 天光線殘差與流量刻度誤差讓所有源的 reduced chi2 都遠大於 2 ——
        於是輸出的 group 永遠是 galaxy,第 1 段等於只是在挑「最像的恆星模板」。
        真正的判定一直是事後在 step4c 手算「同一組通道上誰的 reduced chi2 低」。
        這裡把那個比較放回它該在的地方:兩組都掃、直接比,不需要任何絕對門檻。

    為什麼 reduced chi2 可以直接比
        reduced chi2 = chi2 / (n_good - n_param),分母已經把「星系本徵譜有 4 個
        成分、恆星模板只有 1 個」的自由度差算進去。前提是兩者用**同一組通道**
        (n_good 相同),所以 main() 會檢查兩個視窗一致。
    """
    S = _SHARED
    k = int(np.flatnonzero(S["ids"] == t)[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        f = S["flux"][k] / S["nspax"][k]
        v = S["var"][k]  / S["nspax"][k] ** 2

    r1 = scan_object(f, v, S["sky"], S["star_jobs"], S["wl_vac"],
                     S["fit_star"], s_fix=S["s_fix"],
                     allow_partial=S["allow_partial"])
    r2 = scan_object(f, v, S["sky"], S["gal_jobs"], S["wl_vac"],
                     S["fit_gal"], s_fix=S["s_fix"],
                     allow_partial=S["allow_partial"])
    if not r1 and not r2:
        return t, None
    if r1:
        _save_scan(STEP04B / f"scan1_id{t}_{S['tag']}.npz", r1)
    if r2:
        _save_scan(STEP04B / f"scan2_id{t}_{S['tag']}.npz", r2)

    # 兩組各自的冠軍對決。掃描結果已依 chi2 排序,所以 [0] 就是各組最佳。
    best = min([x[0] for x in (r1, r2) if x], key=lambda d: d["red_chi2"])

    A = np.full(N_SRC, np.nan)
    A[:len(best["A"])] = best["A"]
    # 兩組的冠軍值都留著 —— 分類是由這兩個數字的大小決定的,不記下來的話
    # 下游無法檢查「贏多少」,也就無法判斷這個分類穩不穩。
    return t, dict(id=t, nspax=int(np.median(S["nspax"][k])),
                   star_red_chi2=r1[0]["red_chi2"] if r1 else np.nan,
                   star_tpl=r1[0]["template"] if r1 else "",
                   gal_red_chi2=r2[0]["red_chi2"] if r2 else np.nan,
                   gal_tpl=r2[0]["template"] if r2 else "",
                   **{**best, "A": A})


def main():
    ap = argparse.ArgumentParser(description="單階段源模板擬合(固定視窗 + 天光線遮罩)")
    ap.add_argument("--id",    default="all",           help="segmentation ID,或 all")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30, help="天光線 basis 條數,須與 step3 相同")
    ap.add_argument("--star-window", type=float, nargs=2, default=STAR_WINDOW,
                    metavar=("LO", "HI"), help="恆星模板的擬合視窗 (A, air);須與 --gal-window 相同")
    ap.add_argument("--gal-window", type=float, nargs=2, default=GAL_WINDOW,
                    metavar=("LO", "HI"), help="星系本徵譜的擬合視窗 (A, air);須與 --star-window 相同")
    ap.add_argument("--full-range", action="store_true",
                    help=f"兩段都改用 MUSE 全波段 {FULL_RANGE[0]:.0f}-{FULL_RANGE[1]:.0f} A "
                         "當對照組。注意:恆星模板靜止只到約 9200 A,全波段下不同模板"
                         "覆蓋到的通道數不再相同,chi2 之間的比較會被通道數影響。")
    ap.add_argument("--line-mask-iter", type=int, nargs="+", default=[1, 2, 3, 4],
                    help="用 step3 的第幾輪天光線遮罩,可給多個,每一輪各存一份結果。"
                         "1 最鬆(只標最強的線,35.5%% 通道),越後面涵蓋越多,"
                         "第 4 輪已達 79.6%%。預設四輪全跑。")
    ap.add_argument("--sky-basis", action="store_true",
                    help="把天光線 basis 也放進源的擬合(step4 的做法)。預設不放:"
                         "天光線通道已經排除在 chi2 之外,basis 在剩下的通道上只有"
                         "約 10%% 的能量,那 K 個弱約束的參數只會吸走源訊號。"
                         "不放的話源的模型只有 1 個自由參數 A。")
    ap.add_argument("--zmin",  type=float, default=0.0)
    ap.add_argument("--zmax",  type=float, default=1.5)
    ap.add_argument("--zstep", type=float, default=1e-4)
    ap.add_argument("--star-dz", type=float, default=0.005,
                    help="恆星的 z 掃描半寬(±1500 km/s)。能被解析成點源的星必定是"
                         "銀河系前景星,沒有哈伯流,只有本動速度。")
    ap.add_argument("--aperture", action="store_true",
                    help="改讀 step02b/ 的圓形孔徑光譜(step2b_aperture.py 產生),"
                         "而不是 step02/ 的 segmentation footprint")
    ap.add_argument("--allow-partial", action="store_true",
                    help="允許模板只覆蓋部分視窗的候選進入比較。預設不允許 ——"
                         "chi2 是加總,通道少的候選天生就小,掃描會被推向模板"
                         "剛好蓋不住的那個 z。只有在明確知道自己在做什麼時才開。")
    ap.add_argument("--gal-model", choices=["eigen", "sdss"], default="eigen",
                    help="用哪種星系模型。eigen = Bolton 2012 的 4 條本徵譜"
                         "(教授指定,在星系族群裡連續內插);sdss = spDR2-023..028 "
                         "那 6 條星系模板(各自是一條完整光譜,挑最像的一條,"
                         "而且可以施加 A >= 0)")
    ap.add_argument("--spec-dir", default=None,
                    help="源光譜的目錄,覆蓋 --aperture。分類要用扣過天空的版本,"
                         "例如 results/skymodel/step02_ours(我們自己扣的)或 "
                         "step02_eso(ESO 扣的)。目錄名會編進輸出的 tag,"
                         "不同來源的結果各存各的")
    ap.add_argument("--raw-mask", action="store_true",
                    help="四輪遮罩各自獨立使用,不累加。預設累加 —— step3 的原始"
                         "四列不是嚴格巢狀的(有通道會掉出),累加後才是乾淨的"
                         "「遮得越來越多」序列。")
    ap.add_argument("--s-fix", type=float, default=1.0)
    ap.add_argument("--s-free", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()
    s_fix = None if args.s_free else args.s_fix
    if args.full_range:
        args.star_window = args.gal_window = FULL_RANGE

    STEP04B.mkdir(parents=True, exist_ok=True)

    src = (Path(args.spec_dir) if args.spec_dir
           else (STEP02B if args.aperture else STEP02))
    # 光譜來源不同 = 不同的科學產物,tag 必須分得開,否則會靜靜蓋掉上一次。
    suffix = (f"_{src.name.replace('step02', '')}" if args.spec_dir else "") \
             + ("_galtpl" if args.gal_model == "sdss" else "")
    ids   = np.load(src / "object_ids.npy")
    flux  = np.load(src / "object_flux.npy")
    var   = np.load(src / "object_var.npy")
    nspax = np.load(src / "object_nspax.npy")

    wl_air = np.load(STEP03 / "wavelength.npy")
    wl_vac = air_to_vacuum(wl_air)
    C_sky  = np.load(STEP03 / "sky_continuum.npy")
    B      = np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")
    sky    = np.vstack([C_sky, B]) if args.sky_basis else C_sky[None, :]

    # 天光線遮罩。iter_line_mask 的第 i 列是 step3 第 i+1 輪的結果 ——
    # 遮罩是在空氣波長上定義的,所以視窗也用空氣波長切,兩者一致。
    line_masks = load_line_masks(STEP03 / "iter_line_mask.npy",
                                 cumulative=not args.raw_mask)
    win_star = (wl_air >= args.star_window[0]) & (wl_air < args.star_window[1])
    win_gal  = (wl_air >= args.gal_window[0])  & (wl_air < args.gal_window[1])

    z_exg  = np.arange(args.zmin, args.zmax + args.zstep / 2, args.zstep)
    z_star = np.arange(-args.star_dz, args.star_dz + args.zstep / 2, args.zstep)
    star_jobs = [("star", f"{i:03d}",
                  load_sdss_template(TPL_DIR / f"spDR2-{i:03d}.fit"), z_star)
                 for i in STAR_IDX]
    eigen_gal = load_eigen_galaxy(EIGEN_GAL)
    # 星系側的候選。本徵譜是「一條 4 成分的模型」;SDSS 星系模板是「6 條各自
    # 獨立的候選」,所以後者要列成 6 個 job,掃描時各自求解、取最低。
    gal_jobs = ([("galaxy", "eigen", eigen_gal, z_exg)] if args.gal_model == "eigen"
                else [("galaxy", f"{i:03d}",
                       load_sdss_template(TPL_DIR / f"spDR2-{i:03d}.fit"), z_exg)
                      for i in GAL_IDX])

    targets = ids.tolist() if args.id == "all" else [int(args.id)]
    if args.id != "all" and targets[0] not in ids:
        raise SystemExit(f"ID {targets[0]} 不存在。可用:{ids.min()}–{ids.max()}")

    n_workers = args.num_workers or max(1, len(os.sched_getaffinity(0)) // 3)
    n_workers = min(n_workers, len(targets))

    # 兩組模型的 reduced chi2 要能直接比大小,就必須算在同一組通道上。
    # 視窗不同時 n_good 不同,reduced chi2 的分母不同,比出來的大小沒有意義。
    if tuple(args.star_window) != tuple(args.gal_window):
        raise SystemExit(
            f"恆星視窗 {args.star_window} 與星系視窗 {args.gal_window} 不同。"
            "單階段擬合是直接比兩者的 reduced chi2,通道集合必須一致 —— "
            "請把 --star-window 與 --gal-window 設成相同的範圍。")

    print(f"恆星  {args.star_window[0]:.0f}-{args.star_window[1]:.0f} A  "
          f"視窗 {int(win_star.sum())} 通道   {len(star_jobs)} 條恆星模板 × "
          f"{z_star.size} 個 z")
    print(f"星系  {args.gal_window[0]:.0f}-{args.gal_window[1]:.0f} A  "
          f"視窗 {int(win_gal.sum())} 通道   星系本徵譜 × {z_exg.size} 個 z")
    print("分類 = 同一組通道上 reduced chi2 較低者(無絕對門檻)")
    print("s 為自由參數" if s_fix is None else f"天空連續譜固定為 {s_fix} × C_sky,先扣掉")
    print("源的模型 = A × 模板" + ("  + 天光線 basis" if args.sky_basis
                                   else "   (1 個自由參數)"))
    print(f"光譜來源 {src.name}"
          + ("  (圓形孔徑 r=6 px)" if args.aperture else "  (segmentation footprint)"))
    print(f"{len(targets)} object(s)   {n_workers} workers   "
          f"遮罩輪次 {args.line_mask_iter}")

    KEYS = ("id", "nspax", "group", "template", "z", "A", "s", "chi2", "chi2_all",
            "red_chi2", "n_good", "src_min", "star_red_chi2", "star_tpl",
            "gal_red_chi2", "gal_tpl")
    outs = []

    # 每一輪遮罩是一組獨立的結果 —— 通道集合不同,chi2 就不同,不能混在一起。
    # 靜態資料(模板、光譜、z 網格)只準備一次,迴圈裡只換遮罩。
    for it in args.line_mask_iter:
        line = line_masks[it - 1]
        fit_star, fit_gal = win_star & ~line, win_gal & ~line
        tag = make_tag(args.basis, args.K, s_fix, args.star_window,
                       args.gal_window, args.sky_basis, it, not args.raw_mask,
                       args.aperture, suffix)

        print(f"\n{'=' * 112}")
        print(f"遮罩 iter{it}{'(累加)' if not args.raw_mask else '(獨立)'}:全波段標了 {int(line.sum()):,} / {line.size} 通道"
              f" ({100 * line.mean():.1f}%)   "
              f"擬合用的乾淨通道 {int(fit_star.sum())}")
        print(f"{'=' * 112}")
        print(f"{'ID':>5}{'nspax':>8}{'group':>8}{'tpl':>7}{'z':>10}{'A':>12}"
              f"{'n':>7}{'chi2':>14}{'chi2/dof':>10}{'star chi2/dof':>15}"
              f"{'gal chi2/dof':>14}"
              f"{'src_min':>10}")
        print("-" * 112)

        _SHARED.update(ids=ids, flux=flux, var=var, nspax=nspax, sky=sky,
                       star_jobs=star_jobs, gal_jobs=gal_jobs, wl_vac=wl_vac,
                       fit_star=fit_star, fit_gal=fit_gal, z_exg=z_exg,
                       tag=tag, s_fix=s_fix,
                       allow_partial=args.allow_partial)

        summary = []
        with Pool(n_workers) as pool:
            for t, row in pool.imap(_scan_one, targets):
                if row is None:
                    print(f"{t:>5}   (全部擬合失敗,跳過)")
                    continue
                summary.append(row)
                print(f"{t:>5}{row['nspax']:>8}{row['group']:>8}{row['template']:>7}"
                      f"{row['z']:>10.5f}{row['A'][0]:>12.4g}{row['n_good']:>7}"
                      f"{row['chi2']:>14,.0f}{row['red_chi2']:>10.2f}"
                      f"{row['star_red_chi2']:>15.2f}{row['gal_red_chi2']:>14.2f}"
                      f"{row['src_min']:>10.2f}",
                      flush=True)

        new = {k: np.array([x[k] for x in summary]) for k in KEYS}

        # 併入既有結果,不覆寫 —— 重跑單一 ID 只該更新那一列。
        out = STEP04B / f"best_{tag}.npz"
        if out.exists():
            old = np.load(out, allow_pickle=False)
            if set(old.files) == set(KEYS):
                keep = ~np.isin(old["id"], new["id"])
                if keep.any():
                    new = {k: np.concatenate([old[k][keep], new[k]]) for k in KEYS}
                    print(f"併入既有的 {int(keep.sum())} 個源")
        o = np.argsort(new["id"])
        np.savez(out, **{k: v[o] for k, v in new.items()})
        outs.append((it, out, summary))

    print(f"\n{'=' * 60}\n各輪比較")
    print(f"{'iter':>6}{'乾淨通道':>10}{'恆星':>7}{'星系':>7}"
          f"{'star chi2/dof 中位':>20}{'負流量源數':>12}")
    print("-" * 62)
    for it, out, summary in outs:
        ns = sum(1 for r in summary if r["group"] == "star")
        med = float(np.median([r["star_red_chi2"] for r in summary]))
        neg = sum(1 for r in summary if r["src_min"] < 0)
        print(f"{it:>6}{int((win_star & ~line_masks[it-1]).sum()):>10}"
              f"{ns:>7}{len(summary) - ns:>7}{med:>20.2f}{neg:>12}")
    print("\n" + "\n".join(f"saved -> {o}" for _, o, _ in outs))


if __name__ == "__main__":
    main()
