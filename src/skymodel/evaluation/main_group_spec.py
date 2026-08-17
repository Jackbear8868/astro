"""相鄰整團的每個成員,用光譜判斷它是不是主星系的一部分。

主源分組的問題是:SExtractor 的 deblender 會把主星系拆成數塊,而「貼在一起」
這個條件不夠 —— 疊在星系上的別的天體同樣貼著。這支腳本把判斷需要的光譜證據
列出來,一顆 pointing 一張表。

每個成員印四組數字:

  star / gal rchi2   同一個源的兩條分支各自的最佳 reduced chi2。
                     只能在同一列之內比大小 —— 亮源的光子雜訊小,模型的一點點
                     不完美除以很小的 sigma 就是很大的 chi2,跨列比沒有意義。
  z_gal, dv          星系分支的最佳紅移,以及它離主源的速度差。
                     主源取「最亮像素所在的那個成員」的星系分支紅移。
  n_z                星系分支裡 rchi2 落在最佳值 +1% 以內的格點數。
                     這是紅移被釘得多緊 —— 數字大代表紅移沒被約束住。
  forced             把這個成員強迫放在主源的紅移上,rchi2 會變差幾 %。

    conda run -n astro python src/skymodel/evaluation/main_group_spec.py
    conda run -n astro python src/skymodel/evaluation/main_group_spec.py -n 1 4 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, load_field  # noqa: E402
from utils import DV_MAX  # noqa: E402

C_KMS = 299792.458


def blob_members(seg, white):
    """最亮像素所在的連通塊,以及塊裡所有的 seg ID(不做任何面積篩選)。"""
    k = np.unravel_index(np.nanargmax(np.where(np.isfinite(white), white, -np.inf)),
                         white.shape)
    lab, _ = ndimage.label(seg > 0)
    blob = lab == lab[k]
    ids = [int(i) for i in np.unique(seg[blob]) if i > 0]
    return ids, blob, k


def area_keep(seg, ids, min_frac):
    """舊的面積判準,只留在這裡當對照 —— utils 已經改用紅移判準。"""
    area = {i: int((seg == i).sum()) for i in ids}
    top = max(area.values())
    return [i for i in ids if area[i] >= min_frac * top]


def branch_best(work, sid):
    """一個 seg ID 的兩條分支各自的最佳解。

    step4b 把兩條分支分開存:scan1 是恆星(z 是視向速度),scan2 是星系。
    回傳 (star 最佳 rchi2, galaxy 最佳 rchi2, galaxy 最佳 z, galaxy 的 z 與 rchi2 陣列)。
    """
    f1 = sorted((work/"step04").glob(f"scan1_id{sid}_*.npz"))
    f2 = sorted((work/"step04").glob(f"scan2_id{sid}_*.npz"))
    if not f1 or not f2:
        return None
    r_star = float(np.load(f1[0], allow_pickle=True)["red_chi2"].min())
    d2 = np.load(f2[0], allow_pickle=True)
    z, r = d2["z"], d2["red_chi2"]
    j = int(np.argmin(r))
    return r_star, float(r[j]), float(z[j]), z, r


def main():
    ap = argparse.ArgumentParser(description="相鄰成員的光譜證據")
    ap.add_argument("-n", type=int, nargs="+", default=list(range(1, 15)))
    ap.add_argument("--min-frac", type=float, default=0.05,
                    help="拿來對照的面積判準門檻")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="算 n_z 時,rchi2 相對於最佳值的容忍比例")
    ap.add_argument("--dv-max", type=float, default=DV_MAX,
                    help="光譜判準:離主源多少 km/s 之內算主星系的一部分")
    args = ap.parse_args()

    rows = []
    for n in args.n:
        W = ROOT / f"results/skymodel/p{n:02d}"
        seg, white, valid = load_field(W)
        wn = np.where(valid, white, np.nan)
        ids, _, k = blob_members(seg, wn)
        keep_area = area_keep(seg, ids, args.min_frac)

        # 主源的紅移取「最亮像素所在的那個成員」—— 面積最大不一定是核心
        peak_id = int(seg[k])
        b = branch_best(W, peak_id)
        if b is None:
            print(f"p{n:02d}: step04 沒有 id{peak_id} 的 scan 檔,跳過")
            continue
        z_main = b[2]

        print(f"\n=== p{n:02d}   主源紅移 z = {z_main:.4f}"
              f"(取最亮像素所在的 id {peak_id}) ===")
        print(f"{'id':>5} {'area':>8} {'star rchi2':>12} {'gal rchi2':>12} {'gal/star':>9}"
              f" {'z_gal':>8} {'dv km/s':>10} {'n_z':>6} {'forced':>8}"
              f"  面積判準  光譜判準")
        print("-" * 112)
        for i in ids:
            b = branch_best(W, i)
            if b is None:
                print(f"{i:>5}  —— step04 沒有這個 ID 的 scan 檔 ——")
                continue
            r_star, r_gal, z_gal, z, r = b
            area = int((seg == i).sum())
            dv = C_KMS * (z_gal - z_main) / (1 + z_main)
            n_z = int((r <= r_gal * (1 + args.tol)).sum())
            # 強迫放在主源紅移:取格點裡離 z_main 最近的那一格
            jf = int(np.argmin(np.abs(z - z_main)))
            forced = 100 * (r[jf] - min(r_gal, r_star)) / min(r_gal, r_star)
            verdict = "留" if i in keep_area else "剔"
            v_spec = "留" if abs(dv) <= args.dv_max else "剔"
            flag = "" if verdict == v_spec else "   <<< 兩個判準不同"
            print(f"{i:>5} {area:>8,} {r_star:>12,.2f} {r_gal:>12,.2f}"
                  f" {r_gal/r_star:>9.3f} {z_gal:>8.4f} {dv:>+10,.0f} {n_z:>6,}"
                  f" {forced:>+7.1f}%  {verdict:>6}  {v_spec:>6}{flag}")
            rows.append((n, i, area, r_star, r_gal, z_gal, dv, n_z, forced,
                         i in keep_area))

    print(f"\n\n=== 全部 {len(rows)} 個相鄰成員 ===")
    dv = np.array([r[6] for r in rows])
    nz = np.array([r[7] for r in rows])
    ratio = np.array([r[4]/r[3] for r in rows])
    kept = np.array([r[9] for r in rows], bool)
    print(f"面積判準留下 {int(kept.sum())} 個,剔掉 {int((~kept).sum())} 個")
    print(f"|dv| 分布:  <100 km/s {int((np.abs(dv) < 100).sum())} 個,"
          f"  100-1000 {int(((np.abs(dv) >= 100) & (np.abs(dv) < 1000)).sum())} 個,"
          f"  >1000 {int((np.abs(dv) >= 1000).sum())} 個")
    print(f"n_z 分布:   =1 {int((nz == 1).sum())} 個,"
          f"  2-10 {int(((nz > 1) & (nz <= 10)).sum())} 個,"
          f"  >10 {int((nz > 10).sum())} 個")
    print(f"gal/star:   <1（星系比較好）{int((ratio < 1).sum())} 個,"
          f"  >=1（恆星比較好）{int((ratio >= 1).sum())} 個")

    print(f"\n兩個判準給出不同答案的成員（dv_max = {args.dv_max:,.0f} km/s）:")
    bad = [r for r in rows if r[9] != (abs(r[6]) <= args.dv_max)]
    for r in bad:
        print(f"  p{r[0]:02d} id{r[1]:<4d} area {r[2]:>6,} px   dv {r[6]:>+9,.0f} km/s"
              f"   n_z {r[7]:>3}   gal/star {r[4]/r[3]:.3f}   "
              f"面積 {'留' if r[9] else '剔'} / 光譜 "
              f"{'留' if abs(r[6]) <= args.dv_max else '剔'}")
    if not bad:
        print("  （沒有）")

    # 門檻掃描:讓「這個判準對門檻有多敏感」直接看得見
    print(f"\n{'dv_max':>12} | {'留下':>6} | {'剔除':>6} | 剔除的成員")
    print("-" * 70)
    for v in (100, 300, 1000, 1468, 3000, 10000, 100000):
        drop = [f"p{r[0]:02d}id{r[1]}" for r in rows if abs(r[6]) > v]
        print(f"{v:>9,} km/s | {len(rows)-len(drop):>6} | {len(drop):>6} | "
              f"{', '.join(drop) if drop else '（無）'}")


if __name__ == "__main__":
    main()
