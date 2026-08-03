"""blank 區的兩階段擬合:線外定 s,線內定 cₖ。

動機:天空的兩個成分在兩個區域各自可測,但現行做法把它們混在一起解。

    line1 外 (2446 通道)   連續譜佔流量 94.6%,天光線 basis 幾乎不可分辨
                           (該區設計矩陣 cond 3.4e6)  → 量 s 的最佳位置
    line1 內 (1355 通道)   天光線佔流量 76.3%,basis 99.9% 的能量都在這裡
                           → 量 cₖ 的最佳位置

先前測過「只用 line1 解全部 11 個係數」:天光線沒有變好(rms 0.0737 → 0.0735),
但 s 散布惡化 47%(0.0156 → 0.0230) —— 丟掉線外通道等於丟掉 s 的證據。
兩階段的想法是兩區都用,但各解各的:

    stage 1   線外通道解  d ≈ s·C_sky              1 個參數,極良態
    stage 2   線內通道解  d − s·C_sky ≈ Σcₖ·Lₖ     10 個參數

兩個設計矩陣都不隨 spaxel 改變,各做一次 pinv 就能解完所有乾淨的 spaxel。

比較對象是現行做法(全通道、未加權、11 個參數一起解)。判準分區報:
    line1 內   天光線扣得乾不乾淨
    line1 外   連續譜扣得準不準(系統性偏移在這裡最明顯)

    conda run -n astro python src/skymodel/experiments/blank_two_stage.py --basis svd
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT   = Path(__file__).resolve().parents[3]
STEP01 = ROOT / "results/skymodel/step01"
STEP03 = ROOT / "results/skymodel/step03"
CUBE   = ROOT / "data/Haro11_NEpointing_wsky.fits"

MIN_COVERAGE = 0.9        # 與 step5 一致:排除視野邊緣的部分覆蓋 spaxel


def scale(a):
    """穩健的散布估計;s 有少數被邊界夾住的極端值,np.std 會被拉高。"""
    a = a[np.isfinite(a)]
    return float((np.percentile(a, 84) - np.percentile(a, 16)) / 2)


def report(name, resid, lm1):
    """resid: (nz, n) 的殘差;分線內/線外報 rms 與系統性偏移。"""
    m = np.nanmean(resid, axis=1)                 # 平均殘差光譜
    return dict(
        name=name,
        rms_in=np.sqrt(np.nanmean(m[lm1] ** 2)),  off_in=np.nanmean(m[lm1]),
        rms_out=np.sqrt(np.nanmean(m[~lm1] ** 2)), off_out=np.nanmean(m[~lm1]),
        rms_all=np.sqrt(np.nanmean(m ** 2)),       off_all=np.nanmean(m))


def main():
    ap = argparse.ArgumentParser(description="blank 兩階段擬合 vs 現行聯合擬合")
    ap.add_argument("--basis", default="svd")
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    C     = np.load(STEP03 / "sky_continuum.npy")
    B     = np.load(STEP03 / f"sky_basis_{args.basis}.npy").astype(np.float64)
    lm1   = np.load(STEP03 / "iter_line_mask.npy")[0]
    sky   = np.vstack([C, B])

    with fits.open(CUBE, memmap=True) as hdul:
        D = np.asarray(hdul["DATA"].data, np.float32)
    nz = D.shape[0]
    D  = D.reshape(nz, -1)

    cov   = np.isfinite(D).sum(axis=0) / nz
    valid = (white != 0).reshape(-1) & (cov >= MIN_COVERAGE)
    blank = valid & (seg.reshape(-1) == 0)
    Db    = D[:, blank].astype(np.float64)
    del D
    clean = np.isfinite(Db).all(axis=0)            # 只用完好的 spaxel,不混入壞通道效應
    Db    = Db[:, clean]

    print(f"basis={args.basis}   blank {int(blank.sum()):,} 個,其中完好 {int(clean.sum()):,} 個")
    print(f"line1 {lm1.sum()}/{lm1.size} 通道 ({100*lm1.mean():.1f}%)\n")

    # ---------- 現行做法:全通道、11 個參數一起解 ----------
    coef_j = np.linalg.pinv(sky.T) @ Db
    resid_j = Db - sky.T @ coef_j
    s_j = coef_j[0]

    # ---------- 兩階段 ----------
    # stage 1 用哪個區域定 s,是關鍵。line1 只遮掉最強的線(35.6%),它的補集裡
    # 還藏著第 2-4 輪才找到的弱線 —— 那些正的流量會把 s 推高。真正無線的區域
    # 要用「最後一輪遮罩」的補集(遮 79.4%,只剩 20.6% 乾淨)。
    M      = np.load(STEP03 / "iter_line_mask.npy")
    lmlast = M[-1]
    variants = [("兩階段 (stage1: ~line1)",   ~lm1),
                ("兩階段 (stage1: ~最後一輪)", ~lmlast)]

    results, s_all = [], [("現行(聯合)", s_j)]
    for lab, free in variants:
        A1  = C[free][:, None]                     # (n_free, 1)
        s_t = (np.linalg.pinv(A1) @ Db[free])[0]   # (n_clean,)
        A2  = B[:, lm1].T                          # (n_in, K)
        c_t = np.linalg.pinv(A2) @ (Db[lm1] - np.outer(C[lm1], s_t))
        results.append((lab, Db - (np.outer(C, s_t) + B.T @ c_t), int(free.sum())))
        s_all.append((lab, s_t))

    # ---------- 報告 ----------
    print("s(p) 的分佈:")
    print(f"{'':26}{'中位數':>12}{'1sigma 散布':>14}{'最小':>10}{'最大':>10}")
    for lab, s in s_all:
        print(f"{lab:26}{np.median(s):>12.4f}{scale(s):>14.4f}{s.min():>10.4f}{s.max():>10.4f}")
    for lab, s in s_all[1:]:
        print(f"  vs 現行:  相關 {np.corrcoef(s_j, s)[0,1]:.4f}   "
              f"中位差 {np.median(s - s_j):+.4f}   ({lab})")

    print(f"\n殘差(blank 平均光譜):")
    print(f"{'':26}{'rms 線內':>12}{'偏移 線內':>12}{'rms 線外':>12}"
          f"{'偏移 線外':>12}{'rms 全部':>12}{'偏移 全部':>12}")
    rows = [report("現行(聯合)", resid_j, lm1)]
    rows += [report(lab, r, lm1) for lab, r, _ in results]
    for r in rows:
        print(f"{r['name']:26}{r['rms_in']:>12.4f}{r['off_in']:>12.4f}"
              f"{r['rms_out']:>12.4f}{r['off_out']:>12.4f}"
              f"{r['rms_all']:>12.4f}{r['off_all']:>12.4f}")

    print(f"\nstage1 可用的無線通道數: "
          + "   ".join(f"{lab.split(': ')[1][:-1]} {n}" for lab, _, n in results))
    print(f"條件數:  聯合(全通道) {np.linalg.cond(sky.T):.1f}   "
          f"stage2(線內,{B.shape[0]} 參數) {np.linalg.cond(B[:, lm1].T):.1f}")


if __name__ == "__main__":
    main()
