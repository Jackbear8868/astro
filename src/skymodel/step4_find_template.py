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

from templates import load_sdss_template, redshift_to_grid, air_to_vacuum

ROOT    = Path(__file__).resolve().parents[2]
STEP02  = ROOT / "results/skymodel/step02"
STEP03  = ROOT / "results/skymodel/step03"
STEP04  = ROOT / "results/skymodel/step04"
TPL_DIR = ROOT / "data/sdss_templates"
N_TPL   = 33            # SDSS spDR2 模板 spDR2-000 … spDR2-032

_SHARED = {}

def scan_object(flux, var, sky, templates, z_grid, lam_muse, n_min=0, s_fix=None):
    """對一條加總光譜掃描模板與紅移,每個候選各自用它覆蓋到的通道。

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
        sky_free, y, n_pos = sky, flux, 2          # A 與 s 都是自由且非負的參數
    else:
        sky_free, y, n_pos = sky[1:], flux - s_fix * sky[0], 1   # 天空連續譜先扣掉

    skyw = np.ascontiguousarray((sky_free / sig).T)
    yw   = y / sig
    p    = sky_free.shape[0] + 1

    lb = np.r_[np.zeros(n_pos), np.full(p - n_pos, -np.inf)]
    ub = np.full(p, np.inf)
    results = []
    for z in z_grid:
        for name, spline in templates.items():
            T    = redshift_to_grid(spline, z, lam_muse)
            good = base & np.isfinite(T)
            n    = int(good.sum())
            if n <= n_min + p:          # 至少要留 1 個自由度,否則 reduced chi2 無定義
                continue

            M = np.empty((n, p))
            M[:, 0]  = T[good] / sig[good]
            M[:, 1:] = skyw[good]

            # bvls(主動集法)適合這種稠密、少量邊界的小問題,比預設的 trf 快 2.3 倍;
            # 兩者求同一個凸問題的同一個最佳解。
            fit   = lsq_linear(M, yw[good], bounds=(lb, ub), method="bvls")
            theta = fit.x
            chi2  = 2.0 * fit.cost

            m_all    = theta[0] * np.nan_to_num(T, nan=0.0) + theta[1:] @ sky_free
            chi2_all = float((((y - m_all) / sig) ** 2)[base].sum())

            results.append(dict(template=name, z=float(z), A=theta[0], s=theta[1] if s_fix is None else s_fix,
                    chi2=chi2, chi2_all=chi2_all, red_chi2=chi2 / (n - p), n_good=n))

    return sorted(results, key=lambda r: r["chi2_all"])
    
def _scan_one(t):
    """在 worker process 裡掃描單一源。

    共用資料由 fork 繼承,不經過 pickle。回傳值只有摘要一列,
    完整掃描結果(若需要)由 worker 自己寫檔,避免把 168028 筆傳回主行程。
    """
    S = _SHARED
    k = int(np.flatnonzero(S["ids"] == t)[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        f = S["flux"][k] / S["nspax"][k]
        v = S["var"][k]  / S["nspax"][k] ** 2

    results = scan_object(f, v, S["sky"], S["templates"], S["z_grid"], S["wl_vac"], s_fix=S["s_fix"])
    if not results:
        return t, None

    if S["save_scan"]:
        np.savez(STEP04 / f"scan_id{t}_{S['basis']}.npz",
                template=np.array([x["template"] for x in results]),
                z=np.array([x["z"] for x in results]),
                A=np.array([x["A"] for x in results]),
                s=np.array([x["s"] for x in results]),
                chi2=np.array([x["chi2"] for x in results]),
                chi2_all=np.array([x["chi2_all"] for x in results]),
                red_chi2=np.array([x["red_chi2"] for x in results]),
                n_good=np.array([x["n_good"] for x in results]))

    return t, dict(id=t, nspax=int(np.median(S["nspax"][k])), **results[0])


def main():
    ap = argparse.ArgumentParser(description="對 segmentation ID 掃描 SDSS 模板與紅移")
    ap.add_argument("--id",    default="1",             help="segmentation ID,或 all")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("--zmin",  type=float, default=0.0)
    ap.add_argument("--zmax",  type=float, default=1.5)
    ap.add_argument("--zstep", type=float, default=1e-4)
    ap.add_argument("--save-scan", action="store_true", help="每個源都存完整掃描結果(all 模式下每個源一個檔)")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="平行 process 數,0 = 自動(可用核數的 1/3);超過 1/3 後記憶體頻寬成為瓶頸")
    ap.add_argument("--s-fix", type=float, default=None,
    help="固定天空連續譜係數 s;不給則 s 為自由參數。")
    args = ap.parse_args()

    STEP04.mkdir(parents=True, exist_ok=True)

    # ---------- 只做一次的準備 ----------
    ids   = np.load(STEP02 / "object_ids.npy")
    flux  = np.load(STEP02 / "object_flux.npy")        # 整批載入,不切單一源
    var   = np.load(STEP02 / "object_var.npy")
    nspax = np.load(STEP02 / "object_nspax.npy")

    wl_air = np.load(STEP03 / "wavelength.npy")
    wl_vac = air_to_vacuum(wl_air)
    C_sky  = np.load(STEP03 / "sky_continuum.npy")
    B      = np.load(STEP03 / f"sky_basis_{args.basis}.npy")
    sky    = np.vstack([C_sky, B])

    templates = {f"{i:03d}": load_sdss_template(TPL_DIR / f"spDR2-{i:03d}.fit") for i in range(N_TPL)}
    z_grid = np.arange(args.zmin, args.zmax + args.zstep / 2, args.zstep)

    if args.id == "all":
        targets = ids.tolist()
    else:
        t = int(args.id)
        if t not in ids:
            raise SystemExit(f"ID {t} 不存在。可用:{ids.min()}–{ids.max()},共 {ids.size} 個")
        targets = [t]


    # 輸出檔名帶上 basis 與 s 的設定,不同設定的結果並存不互相覆蓋。
    tag = f"{args.basis}_s_free" if args.s_fix is None else f"{args.basis}_s_{args.s_fix}"

    print(f"basis={args.basis}   {len(templates)} templates x {z_grid.size} z"
          f"   {wl_air.size} channels ({wl_air.min():.1f}-{wl_air.max():.1f} A air)")
    print("s 為自由參數" if args.s_fix is None
          else f"s 固定為 {args.s_fix} —— 天空連續譜先從資料扣掉,不列入自由參數")
    n_workers = args.num_workers or max(1, len(os.sched_getaffinity(0)) // 3)
    n_workers = min(n_workers, len(targets))
    print(f"{len(targets)} object(s),每個 {len(templates)*z_grid.size} fits"
          f"   {n_workers} workers\n")

    print(f"{'ID':>5}{'nspax':>8}{'tpl':>6}{'z':>10}{'A':>12}{'s':>8}"
          f"{'n_good':>8}{'chi2_all':>16}")
    print("-" * 73)

    _SHARED.update(ids=ids, flux=flux, var=var, nspax=nspax,
                   sky=sky, templates=templates, z_grid=z_grid, wl_vac=wl_vac,
                   basis=tag, s_fix=args.s_fix,
                   save_scan=args.save_scan or len(targets) == 1)
    # ---------- 逐源掃描 ----------
    summary = []
    with Pool(n_workers) as pool:
        for t, row in pool.imap(_scan_one, targets):
            if row is None:
                print(f"{t:>5}   (全部擬合失敗,跳過)")
                continue
            summary.append(row)
            print(f"{t:>5}{row['nspax']:>8}{row['template']:>6}{row['z']:>10.5f}"
                  f"{row['A']:>12.4g}{row['s']:>8.4f}{row['n_good']:>8}"
                  f"{row['chi2_all']:>16,.0f}", flush=True)

    out = STEP04 / f"best_{tag}.npz"
    np.savez(out, **{key: np.array([x[key] for x in summary])
                    for key in ("id", "nspax", "template", "z", "A", "s",
                                "chi2", "chi2_all", "red_chi2", "n_good")})
    print(f"\nsaved -> {out}")
    

if __name__ == "__main__":
    main()