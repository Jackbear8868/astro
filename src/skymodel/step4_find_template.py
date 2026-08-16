import os

# BLAS 的執行緒數必須在 import numpy 之前設定 —— 函式庫載入時只讀一次。
# 平行化時每個 process 各自開滿執行緒會超額訂閱,反而比序列慢。
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from templates import (load_sdss_template, load_eigen_galaxy, load_eigen_qso,
                       redshift_to_grid, air_to_vacuum)

ROOT      = Path(__file__).resolve().parents[2]
STEP02    = ROOT / "results/skymodel/step02"
STEP03    = ROOT / "results/skymodel/step03"
STEP04    = ROOT / "results/skymodel/step04"
TPL_DIR   = ROOT / "data/sdss_templates"
EIGEN_GAL = ROOT / "data/eigen_galaxy_Bolton2012.fits"
EIGEN_QSO = ROOT / "data/qso_eigen_linear_55732.dat"

# SDSS spDR2 模板的類別歸屬(docs/sdss-templates.md 第 2 節)。
# 三類都只用「1 條模板 + 1 個振幅」,自由參數數目相同,chi2 的比較才公平。
SDSS_GROUPS = {"star": range(0, 23), "galaxy": range(23, 29), "qso": range(29, 33)}
N_SRC       = 4         # 源係數欄位的固定寬度(本徵譜 4 條;恆星只用第 0 欄)

_SHARED = {}

def scan_object(flux, var, sky, jobs, lam_muse, n_min=0, s_fix=None):
    """對一條加總光譜掃描模板與紅移,每個候選各自用它覆蓋到的通道。

    jobs 是 (名稱, 樣條, z 網格) 的清單 —— z 網格屬於各個候選,不是全域共用的,
    因為恆星只需要掃 ±0.005(銀河系內的本動速度),星系與 QSO 要掃 0 到 1.5。

    模板紅移後蓋不到的通道從該候選的擬合中剔除,因此 n 隨候選變動。
    擬合只用該候選覆蓋到的通道;評分則把解出的係數套回全部通道算 chi2_all,
    讓覆蓋少的候選無法靠躲開難擬合的區域取勝。

    A 與 s 受非負限制:源的振幅和天空連續譜的係數在物理上不能是負的。
    天光線係數不設限,因為 basis 學自殘差,本來就有正有負。

    s_fix 給定時,s·C_sky 先從資料扣掉,s 不再是自由參數(參數少一個),
    用意是切斷 A·T 與 s·C_sky 的簡併，資料無法區分,任其自由會讓模板吸走天空連續譜。

    回傳 results,依 chi2_all 由小到大排序,
    每筆為 dict(template, z, A, s, chi2, chi2_all, red_chi2, n_good)。
    """
    base = (np.isfinite(flux) & np.isfinite(var) & (var > 0) & np.all(np.isfinite(sky), axis=0))
    sig  = np.sqrt(np.where(var > 0, var, 1.0))

    if s_fix is None:
        sky_free, y, s_free = sky, flux, True      # A 與 s 都是自由且非負的參數
    else:
        sky_free, y, s_free = sky[1:], flux - s_fix * sky[0], False   # 天空連續譜先扣掉

    skyw = np.ascontiguousarray((sky_free / sig).T)
    yw   = y / sig

    results = []
    for group, name, spline, z_grid in jobs:
        # 樣條的係數陣列 c 保留了建構時 y 的尾端形狀:單一模板是 (n,),
        # 本徵譜是 (n, 4)。從它推出源區塊佔幾欄,不必另外傳一個參數進來
        # (多傳一個就多一個和實際內容不一致的機會)。
        n_comp = 1 if spline.c.ndim == 1 else spline.c.shape[1]
        p      = sky_free.shape[0] + n_comp

        # 非負的參數不是「開頭連續的幾個」——參數順序是 [a₁…a_ncomp, (s), c₁…c_K],
        # s 自由時落在 index n_comp,前面隔著 n_comp−1 個源係數。
        #
        # 單一模板時 A ≥ 0 和「源的光譜 ≥ 0」完全等價(模板本身恆正),那是嚴格的
        # 物理陳述,保留。多成分基底則沒有任何單一係數的約束等價於那件事 —— 成分
        # 有正有負,壓住 a₁ 只是扭曲擬合而不保證光譜為正;要真正約束得對每一個通道
        # 各下一條不等式,太貴。改成不設限、記下 src_min 事後檢查。
        lb = np.full(p, -np.inf)
        if n_comp == 1:
            lb[0] = 0.0
        if s_free:
            lb[n_comp] = 0.0
        ub = np.full(p, np.inf)

        for z in z_grid:
            T = redshift_to_grid(spline, z, lam_muse)
            if T.ndim == 1:
                T = T[:, None]      # 單一模板也當成 1 欄的矩陣,底下只要一種寫法
            good = base & np.all(np.isfinite(T), axis=1)
            n    = int(good.sum())
            if n <= n_min + p:          # 至少要留 1 個自由度,否則 reduced chi2 無定義
                continue

            M = np.empty((n, p))
            M[:, :n_comp] = T[good] / sig[good][:, None]
            M[:, n_comp:] = skyw[good]

            # bvls(主動集法)適合這種稠密、少量邊界的小問題;它和預設的 trf
            # 求的是同一個凸問題的同一個最佳解。
            fit   = lsq_linear(M, yw[good], bounds=(lb, ub), method="bvls")
            theta = fit.x
            chi2  = 2.0 * fit.cost

            src      = np.nan_to_num(T, nan=0.0) @ theta[:n_comp]
            m_all    = src + theta[n_comp:] @ sky_free
            chi2_all = float((((y - m_all) / sig) ** 2)[base].sum())

            results.append(dict(group=group, template=name, z=float(z),
                    A=theta[:n_comp], s=theta[n_comp] if s_free else s_fix,
                    chi2=chi2, chi2_all=chi2_all, red_chi2=chi2 / (n - p), n_good=n,
                    src_min=float(src[base].min())))   # 重建的源光譜有沒有跑到負的

    return sorted(results, key=lambda r: r["chi2_all"])
    
def _save_scan(path, results):
    """把一段掃描的完整結果寫成 npz。

    第 1 段(33 條離散模板)與第 2 段(本徵譜)共用同一組欄位,這樣兩段的診斷
    程式只要寫一套讀檔邏輯。A 固定 N_SRC 欄:單一模板只有第 0 欄有值,
    本徵譜四欄都有。
    """
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
    """在 worker process 裡掃描單一源。

    共用資料由 fork 繼承,不經過 pickle。回傳值只有摘要一列,完整掃描結果
    (若需要)由 worker 自己寫檔,避免把整份候選表傳回主行程。
    """
    S = _SHARED
    k = int(np.flatnonzero(S["ids"] == t)[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        f = S["flux"][k] / S["nspax"][k]
        v = S["var"][k]  / S["nspax"][k] ** 2

    # ---- 第 1 段:33 條離散模板分類 ----
    # 三組都是「1 條模板 + 1 個自由振幅」,參數數目相同,chi2 的比較是公平的。
    # 本徵譜不放在這一段,否則星系/QSO 會多 3 個自由參數,天生佔便宜。
    results = scan_object(f, v, S["sky"], S["jobs"], S["wl_vac"], s_fix=S["s_fix"])
    if not results:
        return t, None

    if S["save_scan"]:
        _save_scan(STEP04 / f"scan_id{t}_{S['basis']}.npz", results)

    # 每一組各自的最佳解(results 已依 chi2_all 排序,第一次遇到該組就是該組最好的)。
    # 最佳組與次佳組的差要留下來:分類是由這兩個 chi2_all 的大小決定的,差距小
    # 或每一組的擬合品質都差的案例必須看得見,不能被 argmin 靜靜吞掉。
    top = {}
    for r in results:
        top.setdefault(r["group"], r)
    ranked = sorted(top.values(), key=lambda r: r["chi2_all"])
    best   = ranked[0]
    group  = best["group"]
    dchi2  = float(ranked[1]["chi2_all"] - best["chi2_all"]) if len(ranked) > 1 else np.inf

    # ---- 第 2 段:星系/QSO 換成本徵譜重新掃 z ----
    # 離散模板只有 6 條星系、4 條 QSO,是很粗的取樣;本徵譜的四條成分能連續
    # 內插出中間的型態。掃全域而不是只在第 1 段的 z 附近微調 —— 沒必要假設
    # 第 1 段用不同模型定出的 z 對這個模型也是最佳的。恆星沒有本徵譜,
    # 維持第 1 段的結果。
    if group in S["eigen"]:
        ref = scan_object(f, v, S["sky"],
                          [(group, "eigen", S["eigen"][group], S["z_exg"])],
                          S["wl_vac"], s_fix=S["s_fix"])
        if ref:
            best = ref[0]
            if S["save_scan"]:
                _save_scan(STEP04 / f"scan2_id{t}_{S['basis']}.npz", ref)

    A = np.full(N_SRC, np.nan)          # 固定 4 欄,恆星只有第 0 欄有值
    A[:len(best["A"])] = best["A"]
    return t, dict(id=t, nspax=int(np.median(S["nspax"][k])), dchi2=dchi2,
                   **{**best, "A": A})


def main():
    ap = argparse.ArgumentParser(description="對 segmentation ID 掃描 SDSS 模板與紅移")
    ap.add_argument("--id",    default="1",             help="segmentation ID,或 all")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("--zmin",  type=float, default=0.0, help="星系/QSO 的 z 下限")
    ap.add_argument("--zmax",  type=float, default=1.5, help="星系/QSO 的 z 上限")
    ap.add_argument("--zstep", type=float, default=1e-4)
    ap.add_argument("--star-dz", type=float, default=0.005,
                    help="恆星的 z 掃描半寬(預設 0.005 = ±1500 km/s)。能被解析成"
                         "點源的星必定是銀河系前景星 —— 沒有哈伯流,只有本動速度"
                         "(盤星 ~100、暈星 ~300、全天最極端的超高速星約 1000 km/s)。"
                         "但也不能硬鎖 0:MUSE 在 5000 A 是 75 km/s/像素,"
                         "一顆 300 km/s 的暈星會位移 4 個像素。")
    ap.add_argument("--save-scan", action="store_true", help="每個源都存完整掃描結果(all 模式下每個源一個檔)")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="平行 process 數,0 = 自動(可用核數的 1/3)。這類工作受記憶體"
                         "頻寬限制,開滿核數不會等比變快")
    ap.add_argument("-K", type=int, default=25,
                    help="天光線 basis 條數;必須和 step3 用的 K 相同")
    ap.add_argument("--s-fix", type=float, default=1.0,
                    help="天空連續譜係數的固定值(預設 1.0 —— 天空不會因為底下有源"
                         "就變弱)")
    ap.add_argument("--s-free", action="store_true",
                    help="改讓 s 成為自由參數,覆蓋 --s-fix")
    args = ap.parse_args()
    s_fix = None if args.s_free else args.s_fix

    STEP04.mkdir(parents=True, exist_ok=True)

    # ---------- 只做一次的準備 ----------
    ids   = np.load(STEP02 / "object_ids.npy")
    flux  = np.load(STEP02 / "object_flux.npy")        # 整批載入,不切單一源
    var   = np.load(STEP02 / "object_var.npy")
    nspax = np.load(STEP02 / "object_nspax.npy")

    wl_air = np.load(STEP03 / "wavelength.npy")
    wl_vac = air_to_vacuum(wl_air)
    C_sky  = np.load(STEP03 / "sky_continuum.npy")
    B      = np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")
    sky    = np.vstack([C_sky, B])

    # 各組自己的 z 網格。恆星只掃 ±star_dz,是物理決定的:銀河系前景星沒有
    # 哈伯流。掃得太寬,恆星模板就有機會靠位移去湊星系的譜線而搶走分類。
    z_exg  = np.arange(args.zmin, args.zmax + args.zstep / 2, args.zstep)
    z_star = np.arange(-args.star_dz, args.star_dz + args.zstep / 2, args.zstep)
    grids  = {"star": z_star, "galaxy": z_exg, "qso": z_exg}
    jobs = [(g, f"{i:03d}", load_sdss_template(TPL_DIR / f"spDR2-{i:03d}.fit"), grids[g])
            for g, idxs in SDSS_GROUPS.items() for i in idxs]
    n_cand = sum(len(j[3]) for j in jobs)          # j[3] 是該候選自己的 z 網格

    # 第 2 段用的本徵譜。批次樣條:一次求值回傳全部 4 條成分,成本和拿 1 條一樣
    # (二分搜尋找區間那一步和成分數無關),所以換成 4 成分不會在這裡付出代價。
    eigen = {"galaxy": load_eigen_galaxy(EIGEN_GAL), "qso": load_eigen_qso(EIGEN_QSO)}

    if args.id == "all":
        targets = ids.tolist()
    else:
        t = int(args.id)
        if t not in ids:
            raise SystemExit(f"ID {t} 不存在。可用:{ids.min()}–{ids.max()},共 {ids.size} 個")
        targets = [t]


    # 輸出檔名帶上 basis 與 s 的設定,不同設定的結果並存不互相覆蓋。
    tag = (f"{args.basis}_K{args.K}_s_free" if s_fix is None
           else f"{args.basis}_K{args.K}_s_{s_fix}")

    print(f"basis={args.basis}   {len(jobs)} templates, {n_cand:,} candidates"
          f"   {wl_air.size} channels ({wl_air.min():.1f}-{wl_air.max():.1f} A air)")
    print("s 為自由參數" if s_fix is None
          else f"s 固定為 {s_fix} —— 天空連續譜先從資料扣掉,不列入自由參數")
    n_workers = args.num_workers or max(1, len(os.sched_getaffinity(0)) // 3)
    n_workers = min(n_workers, len(targets))
    print(f"{len(targets)} object(s),每個 {n_cand:,} fits"
          f"   {n_workers} workers\n")

    print(f"{'ID':>5}{'nspax':>8}{'group':>8}{'tpl':>7}{'z':>10}{'A':>12}{'s':>8}"
          f"{'n_good':>8}{'chi2_all':>16}{'d(次佳組)':>16}")
    print("-" * 94)

    _SHARED.update(ids=ids, flux=flux, var=var, nspax=nspax,
                   sky=sky, jobs=jobs, wl_vac=wl_vac, eigen=eigen, z_exg=z_exg,
                   basis=tag, s_fix=s_fix,
                   save_scan=args.save_scan or len(targets) == 1)
    # ---------- 逐源掃描 ----------
    summary = []
    with Pool(n_workers) as pool:
        for t, row in pool.imap(_scan_one, targets):
            if row is None:
                print(f"{t:>5}   (全部擬合失敗,跳過)")
                continue
            summary.append(row)
            print(f"{t:>5}{row['nspax']:>8}{row['group']:>8}{row['template']:>7}"
                  f"{row['z']:>10.5f}{row['A'][0]:>12.4g}{row['s']:>8.4f}"
                  f"{row['n_good']:>8}{row['chi2_all']:>16,.0f}"
                  f"{row['dchi2']:>16,.0f}", flush=True)

    KEYS = ("id", "nspax", "group", "template", "z", "A", "s",
            "chi2", "chi2_all", "red_chi2", "n_good", "dchi2", "src_min")
    new = {k: np.array([x[k] for x in summary]) for k in KEYS}

    # 併入既有結果,不要覆寫。tag 已經編碼了 basis/K/s,所以同一個檔裡的列
    # 都出自相同的設定;重跑單一個 ID 應該只更新那一列,不該把其他源的結果毀掉。
    out = STEP04 / f"best_{tag}.npz"
    if out.exists():
        old = np.load(out, allow_pickle=False)
        if set(old.files) == set(KEYS):
            keep = ~np.isin(old["id"], new["id"])
            if keep.any():
                new = {k: np.concatenate([old[k][keep], new[k]]) for k in KEYS}
                print(f"\n併入既有的 {int(keep.sum())} 個源(本次更新 {len(summary)} 個)")
        else:
            print(f"\n注意:既有的 {out.name} 欄位不同(舊格式),直接覆寫")
    o = np.argsort(new["id"])
    np.savez(out, **{k: v[o] for k, v in new.items()})

    # 多成分基底沒有施加非負約束(見 scan_object),所以重建的源光譜原則上可能
    # 跑到負的。那不是物理的解,要看得見。
    neg = [r for r in summary if r["src_min"] < 0]
    if neg:
        print(f"\n注意:{len(neg)} 個源的重建光譜有負值 —— "
              + ", ".join(f"ID {r['id']} ({r['group']}, 最小 {r['src_min']:.3g})" for r in neg))
    print(f"\nsaved -> {out}")
    

if __name__ == "__main__":
    main()