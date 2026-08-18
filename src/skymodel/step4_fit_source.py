"""源的模板擬合 —— 單階段,固定波長視窗,天光線通道排除在 chi2 之外。

分類的方式

    恆星模板與星系本徵譜在**同一組通道**上各自擬合,reduced chi2 低的一組
    勝出,同時定出紅移。chi2 只算在指定視窗內、且不是天光線的通道上。

    另一種做法是把 33 條模板(恆星+星系+QSO)全部丟在一起比 chi2,通道用全部
    3801 個 —— 下面兩段說明為什麼不那樣做。

規格來自 reminder.txt:用 line mask、不用線區的通道、恆星與星系各自在一個
固定的波長視窗裡擬合。視窗的實際值見 --star-window / --gal-window 的預設值,
不在這裡重複寫 —— 兩個地方寫同一組數字遲早會不同步。

為什麼排除天光線通道:那些通道的殘差被天空扣除的誤差主導,不是源的資訊。
把它們算進 chi2,等於讓「哪個模板比較能吸收天空殘差」去決定分類與紅移。
(blank 區的規則正好相反 —— 那裡只用線區,因為要學的就是天空。)

為什麼用固定視窗:讓每個候選各自用它覆蓋到的通道的話,n_good 會隨 z 變,
chi2(z) 出現純粹來自通道數的階梯。固定視窗之後,只要視窗落在星系本徵譜
(靜止 1183–9840 A)在整個掃描 z 範圍內都蓋得住的區間,所有候選的通道集合
完全相同 —— 階梯消失,chi2 之間可以直接相減。

為什麼分類不用絕對門檻:一個「像不像恆星」的絕對門檻要能用,前提是 reduced
chi2 的絕對值可信;但天光線殘差與流量刻度誤差會把所有源的 reduced chi2 一起
抬高,門檻不是太鬆就是全部不過。改成直接比兩組冠軍的大小,分類的可信度則由
**兩組冠軍的差距**來表達 —— star_red_chi2 與 gal_red_chi2 都寫進輸出,
差距小就代表這個分類不穩。

兩個視窗必須相同:reduced chi2 = chi2 / (n_good - n_param),分母裡的
n_good 由通道集合決定。視窗不同就不是同一個統計量,比大小沒有意義。
main() 會擋下不一致的設定。

    conda run -n astro python src/skymodel/step4_fit_source.py --id all -K 54 \\
        --spec-dir results/skymodel/ne_pointing/step02_eso --s-fix 0.0 \\
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
# 這三個必須是模組層級的全域 —— multiprocessing 的 worker 由 fork 產生,
# 只看得到 fork 當下的模組全域,看不到 main() 的區域變數。
# main() 依 --work 賦值,在開 Pool 之前。
STEP02B = STEP03 = STEP04 = None
TPL_DIR   = ROOT / "data/sdss_templates"
EIGEN_GAL = ROOT / "data/eigen_galaxy_Bolton2012.fits"

STAR_IDX = range(0, 23)     # spDR2-000..022 是恆星(docs/sdss-templates.md 第 2 節)
GAL_IDX  = range(23, 29)    # 023 早型、024-026、027 晚型、028 LRG,共 6 條星系模板
N_SRC    = 4                # A 欄位固定寬度:本徵譜 4 條,恆星只用第 0 欄

# 兩段各自的波長視窗 (A, 空氣波長)。改這裡就等於改預設值;
# 也可以在命令列用 --star-window / --gal-window 覆蓋,不必動程式。
# 視窗會編進輸出的 tag,所以不同視窗的結果各存各的,不互相覆蓋。
# 下限三者取同一個值,三個視窗的差別才單純是「右邊延伸到哪」。
# 4600 不等於「從頭開始」—— 13 顆的第一個通道在 4599.6-4600.3 A,但 p14 是
# 4749.83 A(只有 3681 個通道)。實際起點以各 cube 的第一個通道為準;同一顆
# 內部的 reduced chi2 比較不受影響,因為兩條分支用同一組通道。
# 兩個視窗必須相同(見上面「兩個視窗必須相同」那一段),所以 reminder.txt 那組
# 4600-6000 / 4600-7000 不能照字面用 —— 分母的 n_good 會不同,reduced chi2 之間
# 就不可比。取兩者的聯集上界 8000,下界維持 reminder.txt 的 4600。
STAR_WINDOW = (4600.0, 8000.0)      # 恆星模板的擬合視窗
GAL_WINDOW  = (4600.0, 8000.0)      # 星系本徵譜的擬合視窗(須與上者相同)
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
    擬合與評分用同一個集合,所以 chi2 之間可以直接比較。

    jobs 是 (組別, 名稱, 樣條, z 網格) 的清單。z 網格屬於各個候選而不是全域
    共用 —— 恆星只需要掃 ±0.005(銀河系內的本動速度),星系要掃 0 到 1.5。

    A 與 s 受非負限制:源的振幅和天空連續譜的係數在物理上不能是負的。
    天光線係數不設限,因為 basis 學自殘差,本來就有正有負。

    s_fix 給定時,s·C_sky 先從資料扣掉,s 不再是自由參數(參數少一個)。
    用意是切斷 A·T 與 s·C_sky 的簡併 —— 資料無法區分兩者,任其自由會讓模板
    吸走天空連續譜。
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
            # 得到一個看似完美其實沒有資料的解。模板的靜止波長範圍有限,z 大到
            # 一定程度就蓋不住視窗,這種候選必須排除而不是讓它贏。
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
    """把一段掃描的完整結果寫成 npz,診斷程式可以直接讀。"""
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

    兩組都掃、直接比大小,不需要任何絕對門檻。

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
        _save_scan(STEP04 / f"scan1_id{t}_{S['tag']}.npz", r1)
    if r2:
        _save_scan(STEP04 / f"scan2_id{t}_{S['tag']}.npz", r2)

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


def write_classification(out_dir, tag, best, ids=None, over=None):
    """把擬合結果收斂成 step5 讀的那份清單。

    分類本身已經由上面的掃描決定 —— 恆星模板與星系本徵譜在同一組通道上競爭,
    reduced chi2 低的勝出。這裡不重算:同樣的判定寫兩份,改了一邊就會靜靜地
    不一致,而那種錯誤從輸出看不出來。

    ids  只收錄這些 seg ID;None = best 檔裡的全部。少寫一個源,那個源在 step5
         就沒有模板可扣,沒有別的好處,所以預設不篩選。
    over {id: z} 把某個源的紅移改成指定值,只用來做敏感度測試。振幅會在該 z 上
         重新取最佳解 —— 模板形狀隨 z 變,振幅不通用。
    """
    over = over or {}
    idx  = {int(i): k for k, i in enumerate(best["id"])}
    ids  = ids if ids else [int(i) for i in best["id"]]

    rows = []
    print(f"\n{'ID':>4}{'class':>8}{'template':>10}{'z':>10}"
          f"{'star X2':>10}{'gal X2':>10}{'margin':>9}")
    print("-" * 61)
    for t in ids:
        if t not in idx:
            print(f"{t:>4}   source not found in best file, skipping")
            continue
        k = idx[t]
        group, tpl = str(best["group"][k]), str(best["template"][k])
        z, A = float(best["z"][k]), np.asarray(best["A"][k], float)
        r1, r2 = float(best["star_red_chi2"][k]), float(best["gal_red_chi2"][k])

        if t in over:
            s2 = np.load(out_dir / f"scan2_id{t}_{tag}.npz")
            j  = int(np.argmin(np.abs(s2["z"] - over[t])))
            group, tpl = "galaxy", str(s2["template"][j])
            z, A = float(s2["z"][j]), np.asarray(s2["A"][j], float)

        a = np.full(N_SRC, np.nan)
        a[:len(A)] = A
        rows.append(dict(id=t, group=group, template=tpl, z=z, A=a))
        mark = "  <- overridden" if t in over else ""
        print(f"{t:>4}{group:>8}{tpl:>10}{z:>10.4f}{r1:>10.2f}{r2:>10.2f}"
              f"{max(r1, r2) / min(r1, r2):>8.2f}x{mark}")

    if not rows:
        raise SystemExit("no sources found; classification file not written")

    out = out_dir / f"classification_{tag}.npz"
    np.savez(out,
             id=np.array([r["id"] for r in rows]),
             group=np.array([r["group"] for r in rows]),
             template=np.array([r["template"] for r in rows]),
             z=np.array([r["z"] for r in rows]),
             A=np.vstack([r["A"] for r in rows]))
    ns = sum(1 for r in rows if r["group"] == "star")
    print(f"\n{len(rows)} sources: {ns} stars / {len(rows) - ns} galaxies")
    print("margin = ratio of the two models' reduced chi2; closer to 1 means less classification confidence")
    print(f"saved -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="single-stage source template fitting (fixed window + sky-line mask)")
    ap.add_argument("--id",    default="all",           help="segmentation ID, or all")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, required=True,
                    help="number of sky-line basis vectors. Required -- all three steps must use the same K; separate defaults would silently read a different basis set")
    ap.add_argument("--star-window", type=float, nargs=2, default=STAR_WINDOW,
                    metavar=("LO", "HI"), help="stellar template fitting window (A, air); must match --gal-window")
    ap.add_argument("--gal-window", type=float, nargs=2, default=GAL_WINDOW,
                    metavar=("LO", "HI"), help="galaxy eigenspectrum fitting window (A, air); must match --star-window")
    ap.add_argument("--full-range", action="store_true",
                    help=f"use full MUSE range {FULL_RANGE[0]:.0f}-{FULL_RANGE[1]:.0f} A "
                         "as a control. Note: stellar templates extend to ~9200 A at rest; "
                         "under full range different templates cover different channel counts, "
                         "so chi2 comparisons are affected by channel count.")
    ap.add_argument("--line-mask-iter", type=int, nargs="+", default=[1, 2, 3, 4],
                    help="which step3 sky-line mask iteration(s) to use; can specify "
                         "multiple, each produces a separate result. Iter 1 is the loosest "
                         "(only the strongest lines); higher iterations mask more channels. "
                         "Default: all four.")
    ap.add_argument("--sky-basis", action="store_true",
                    help="include sky-line basis in the source fit. Off by default: "
                         "sky-line channels are already excluded from chi2, the basis has "
                         "almost no power in the remaining channels, and those K weakly "
                         "constrained parameters only absorb source signal. Without it the "
                         "source model has only 1 free parameter A.")
    ap.add_argument("--zmin",  type=float, default=0.0)
    ap.add_argument("--zmax",  type=float, default=1.5)
    ap.add_argument("--zstep", type=float, default=1e-4)
    ap.add_argument("--star-dz", type=float, default=0.005,
                    help="half-width of z scan for stars (+-1500 km/s). Resolved point "
                         "sources must be Milky Way foreground stars with no Hubble flow, "
                         "only peculiar velocity.")
    ap.add_argument("--aperture", action="store_true",
                    help="read circular aperture spectra from step02b/ (produced by "
                         "experiments/step2b_aperture.py) instead of segmentation footprint "
                         "from step02/")
    ap.add_argument("--allow-partial", action="store_true",
                    help="allow candidates where the template covers only part of the "
                         "window. Off by default -- chi2 is a sum, so candidates with fewer "
                         "channels are inherently smaller, biasing the scan toward z values "
                         "where the template barely covers the window. Enable only when you "
                         "know what you are doing.")
    ap.add_argument("--gal-model", choices=["eigen", "sdss"], default="eigen",
                    help="which galaxy model to use. eigen = Bolton 2012 4 eigenspectra "
                         "(continuous interpolation across galaxy populations); "
                         "sdss = 6 galaxy templates spDR2-023..028 (each a full spectrum, "
                         "picks the best match, and can enforce A >= 0)")
    ap.add_argument("--spec-dir", default=None,
                    help="directory of source spectra, overrides --aperture. Classification "
                         "requires sky-subtracted spectra, e.g. .../step02_ours (our subtraction) "
                         "or step02_eso (ESO subtraction). Directory name is encoded in the "
                         "output tag so different sources are stored separately")
    ap.add_argument("--raw-mask", action="store_true",
                    help="use each mask iteration independently without accumulation. "
                         "Default is cumulative -- step3's raw iterations are not strictly "
                         "nested (some channels drop out); cumulative mode produces a clean "
                         "'progressively more masked' sequence.")
    ap.add_argument("--s-fix", type=float, default=1.0)
    ap.add_argument("--s-free", action="store_true")
    ap.add_argument("--ids", type=int, nargs="+", default=None,
                    help="classification file includes only these seg IDs. Omit = all fitted sources")
    ap.add_argument("--z-override", nargs="*", default=[], metavar="ID=Z",
                    help="override a source's redshift to a specified value, for sensitivity "
                         "testing only -- production records should not contain manually set "
                         "values. Amplitude is re-solved at the overridden z, not carried over")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--work", required=True,
                    help="working directory for this cube (contains step02/step03/step04)")
    args = ap.parse_args()
    over = {int(k): float(v) for k, v in (x.split("=") for x in args.z_override)}
    work    = Path(args.work)
    # 這三個必須是模組層級的全域 —— _scan_one 在 multiprocessing 的 worker
    # 行程裡執行,看不到 main() 的區域變數(worker 由 fork 產生,看得到的是 fork 當下的模組全域)。
    global STEP02B, STEP03, STEP04
    STEP02B = work / "step02b"
    STEP03  = work / "step03"
    STEP04 = work / "step04"
    print(f"workspace {work}")
    s_fix = None if args.s_free else args.s_fix
    if args.full_range:
        args.star_window = args.gal_window = FULL_RANGE

    STEP04.mkdir(parents=True, exist_ok=True)

    # 源光譜的來源必須明講。沒有可用的預設 —— 用含天空的光譜去分類,結果看起來
    # 完全正常,只是每個源的模板和紅移都是錯的。
    if not args.spec_dir and not args.aperture:
        raise SystemExit(f"requires --spec-dir (e.g. {work}/step02_eso) or --aperture")
    src = Path(args.spec_dir) if args.spec_dir else STEP02B
    # 光譜來源不同 = 不同的科學產物,tag 必須分得開,否則會靜靜蓋掉上一次。
    # 光譜來源編進 tag —— 同一個工作區裡若有多種來源(例如 ne_pointing 的
    # step02_eso 與 step02_ours),不編進檔名就會靜靜蓋掉上一次。
    # 預設來源 step02 不加後綴:後綴標記的是「偏離預設」,預設本身不必標。
    suffix = ("" if src.name == "step02" else f"_{src.name.replace('step02', '')}") \
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
        raise SystemExit(f"ID {targets[0]} does not exist. Available: {ids.min()}-{ids.max()}")

    n_workers = args.num_workers or max(1, len(os.sched_getaffinity(0)) // 3)
    n_workers = min(n_workers, len(targets))

    # 兩組模型的 reduced chi2 要能直接比大小,就必須算在同一組通道上。
    # 視窗不同時 n_good 不同,reduced chi2 的分母不同,比出來的大小沒有意義。
    if tuple(args.star_window) != tuple(args.gal_window):
        raise SystemExit(
            f"star window {args.star_window} and galaxy window {args.gal_window} differ. "
            "Single-stage fitting directly compares their reduced chi2, so the channel set "
            "must be identical -- set --star-window and --gal-window to the same range.")

    print(f"star  {args.star_window[0]:.0f}-{args.star_window[1]:.0f} A  "
          f"window {int(win_star.sum())} channels   {len(star_jobs)} stellar templates x "
          f"{z_star.size} z values")
    print(f"galaxy  {args.gal_window[0]:.0f}-{args.gal_window[1]:.0f} A  "
          f"window {int(win_gal.sum())} channels   galaxy eigenspectra x {z_exg.size} z values")
    print("classification = lower reduced chi2 on the same channel set (no absolute threshold)")
    print("s is a free parameter" if s_fix is None else f"sky continuum fixed to {s_fix} x C_sky, subtracted first")
    print("source model = A x template" + ("  + sky-line basis" if args.sky_basis
                                          else "   (1 free parameter)"))
    print(f"spectra from {src.name}"
          + ("  (circular aperture r=6 px)" if args.aperture else "  (segmentation footprint)"))
    print(f"{len(targets)} object(s)   {n_workers} workers   "
          f"mask iterations {args.line_mask_iter}")

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
        print(f"mask iter{it}{'(cumulative)' if not args.raw_mask else '(independent)'}: flagged {int(line.sum()):,} / {line.size} channels"
              f" ({100 * line.mean():.1f}%)   "
              f"clean channels for fitting {int(fit_star.sum())}")
        print(f"{'=' * 112}")
        print(f"{'ID':>5}{'nspax':>8}{'group':>8}{'tpl':>7}{'z':>10}{'A':>12}"
              f"{'n':>7}{'chi2':>14}{'chi2/dof':>10}{'star chi2/dof':>15}"
              f"{'gal chi2/dof':>14}"
              f"{'src_min':>10}")
        print("-" * 112)

        _SHARED.update(ids=ids, flux=flux, var=var, nspax=nspax, sky=sky,
                       star_jobs=star_jobs, gal_jobs=gal_jobs, wl_vac=wl_vac,
                       fit_star=fit_star, fit_gal=fit_gal,
                       tag=tag, s_fix=s_fix,
                       allow_partial=args.allow_partial)

        summary = []
        with Pool(n_workers) as pool:
            for t, row in pool.imap(_scan_one, targets):
                if row is None:
                    print(f"{t:>5}   (all fits failed, skipping)")
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
        out = STEP04 / f"best_{tag}.npz"
        if out.exists():
            old = np.load(out, allow_pickle=False)
            if set(old.files) != set(KEYS):
                print(f"  * {out.name} fields differ from current format, discarding entire file."
                      f"\n    extra {sorted(set(old.files) - set(KEYS))}"
                      f"  missing {sorted(set(KEYS) - set(old.files))}")
            if set(old.files) == set(KEYS):
                keep = ~np.isin(old["id"], new["id"])
                if keep.any():
                    new = {k: np.concatenate([old[k][keep], new[k]]) for k in KEYS}
                    print(f"merged {int(keep.sum())} existing sources")
        o = np.argsort(new["id"])
        np.savez(out, **{k: v[o] for k, v in new.items()})
        write_classification(STEP04, tag, np.load(out), args.ids, over)
        outs.append((it, out, summary))

    print(f"\n{'=' * 60}\ncross-iteration comparison")
    print(f"{'iter':>6}{'clean ch':>10}{'stars':>7}{'galaxies':>9}"
          f"{'star chi2/dof med':>20}{'neg-flux src':>13}")
    print("-" * 65)
    for it, out, summary in outs:
        ns = sum(1 for r in summary if r["group"] == "star")
        med = float(np.median([r["star_red_chi2"] for r in summary]))
        neg = sum(1 for r in summary if r["src_min"] < 0)
        print(f"{it:>6}{int((win_star & ~line_masks[it-1]).sum()):>10}"
              f"{ns:>7}{len(summary) - ns:>9}{med:>20.2f}{neg:>13}")
    print("\n" + "\n".join(f"saved -> {o}" for _, o, _ in outs))


if __name__ == "__main__":
    main()
