"""不同的訓練遮罩 -> 不同的天空模型 -> 效果差多少。

比較的五個遮罩
--------------
    A  全部 blank              沒有任何限制(最多樣本,最髒)
    B  x < 170                 現行設定
    C  x < 170 + 小源 15 px    再排除其他源的 PSF 翼
    D  C + Haro 11 50 px       距離從**整個 seg==1** 算(含 CLEAN 併進來的碎塊)
    E  C + Haro 11 50 px       距離只從 seg==1 的**主連通塊**算

D 和 E 的差別:SExtractor 的 `CLEAN Y` 會把獨立天體的像素併進 ID 1,所以
seg==1 其實不是單一連通塊。用整個 seg==1 算距離的話,那些散在外面的碎塊會
把 50 px 的排除區撐大。

怎麼評分(三組指標,缺一不可)
----------------------------
① 天空扣得乾不乾淨 —— 在**遠場**(只有天空的地方)量殘差。
   線內/線外分開報:basis 的工作是天光線,連續譜是 C_sky 的工作,混在一起看不出誰的問題。
   **所有選項都用同一組通道**(取自 A 的 line_mask),否則比的是不同的題目。

② 對 ESO 的外部基準 —— rms(我們的殘差 − ESO 的殘差)。
   這是唯一不受我們自己的假設影響的判準。blank 殘差本身不能當判準:
   訓練樣本越多、越貼近評估樣本,殘差必然越小,那個排序毫無資訊。

③ 源保留 —— 在源外圈量連續譜留下多少。真值是
   P = <環的原始資料> − <遠場的原始資料>,完全沒有經過擬合。

**評估樣本對五個選項完全相同**,只有訓練樣本不同。

    conda run -n astro python src/skymodel/experiments/mask_compare_basis.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import estimate_continuum
from step3_sky_basis import learn_sky_basis, CLIP_SIGMA, WINDOW, THRESHOLDS, MAX_ITER

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/ne_pointing/step01"
STEP03  = ROOT / "results/skymodel/ne_pointing/step03"
WSKY    = ROOT / "data/Haro11_NEpointing_wsky.fits"
ESO     = ROOT / "data/Haro11_NEpointing_esonosky.fits"
FIGURES = ROOT / "results/skymodel/figures/s_field"


def read_blank(cube, idx, nz, ny, nx):
    out = np.empty((nz, idx.size), np.float32)
    with fits.open(cube, memmap=True) as h:
        for j in range(0, nz, 200):
            out[j:j+200] = np.asarray(h["DATA"].data[j:j+200],
                                      np.float32).reshape(-1, ny*nx)[:, idx]
    return out


def learn(D, train, K, method, verbose=""):
    """step3 的流程,原封搬過來:sigma-clip 平均 -> 連續譜 -> 殘差 -> 分解。"""
    X = D[:, train]
    p16, med, p84 = np.percentile(X, [16, 50, 84], axis=1)
    sg   = np.maximum((p84 - p16) / 2, 1e-6)
    keep = np.abs(X - med[:, None]) <= CLIP_SIGMA * sg[:, None]
    mean_sky = (X * keep).sum(axis=1, dtype=np.float64) / keep.sum(axis=1)
    C, sigma, lm, hist = estimate_continuum(mean_sky, thresholds=THRESHOLDS,
                                            window=WINDOW, max_iter=MAX_ITER)
    resid = np.where(keep, X - C[:, None], (med - C)[:, None])
    t0 = time.time()
    basis, _ = learn_sky_basis(resid, K=K, method=method)
    print(f"    {verbose} basis {basis.shape}  {time.time()-t0:.0f}s  "
          f"line_mask {100*lm.mean():.1f}%", flush=True)
    return np.vstack([C, basis]), C, lm


def fit_all(D, sky):
    """所有 blank spaxel 用同一個設計矩陣一次解完(未加權,s 自由)——與 step5 一致。"""
    c = np.linalg.pinv(sky.T) @ D
    return D - sky.T @ c, c


def rms(a):
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if a.size else np.nan


def main():
    ap = argparse.ArgumentParser(description="不同訓練遮罩的天空模型效果比較")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--xcut", type=int, default=170)
    ap.add_argument("--r-oth", type=float, default=15.0)
    ap.add_argument("--r-haro", type=float, default=50.0)
    ap.add_argument("--figdir", default=str(FIGURES))
    args = ap.parse_args()

    fig_dir = Path(args.figdir); fig_dir.mkdir(parents=True, exist_ok=True)
    seg   = fits.getdata(STEP01 / "seg.fits").astype(int)
    white = fits.getdata(STEP01 / "whitelight.fits")
    comp  = fits.getdata(STEP01 / "complete.fits").astype(bool)
    wl    = np.load(STEP03 / "wavelength.npy")
    ny, nx = seg.shape
    nz = wl.size

    valid = white != 0
    blank = valid & (seg == 0) & comp
    idx   = np.flatnonzero(blank.ravel())
    print(f"blank 且光譜完整:{idx.size:,}", flush=True)

    # Haro 11:整個 seg==1 vs 只取主連通塊
    lab, n = ndimage.label(seg == 1)
    sz = np.array(ndimage.sum(seg == 1, lab, range(1, n + 1)))
    main_blob = lab == (int(np.argmax(sz)) + 1)
    print(f"seg==1 連通塊 {n} 個,主體 {int(main_blob.sum()):,} px", flush=True)

    d_all  = ndimage.distance_transform_edt(seg == 0)
    d_oth  = ndimage.distance_transform_edt(~((seg > 0) & (seg != 1)))
    d_har  = ndimage.distance_transform_edt(seg != 1)          # 全部 seg==1
    d_main = ndimage.distance_transform_edt(~main_blob)        # 只有主體
    edge   = ndimage.distance_transform_edt(valid)
    xx     = np.mgrid[0:ny, 0:nx][1]

    f = lambda m: m.ravel()[idx]
    MASKS = [
        ("A  all blank",            np.ones(idx.size, bool)),
        (f"B  x<{args.xcut}",        f(xx < args.xcut)),
        (f"C  B + {args.r_oth:g}px compact",
         f((xx < args.xcut) & (d_oth > args.r_oth))),
        (f"D  C + {args.r_haro:g}px Haro (all seg==1)",
         f((xx < args.xcut) & (d_oth > args.r_oth) & (d_har > args.r_haro))),
        (f"E  C + {args.r_haro:g}px Haro (main blob)",
         f((xx < args.xcut) & (d_oth > args.r_oth) & (d_main > args.r_haro))),
    ]

    print("\n讀 wsky…", flush=True)
    D = read_blank(WSKY, idx, nz, ny, nx)
    ok = np.isfinite(D).all(axis=0)
    D, idx = D[:, ok], idx[ok]
    MASKS = [(t, m[ok]) for t, m in MASKS]
    print(f"  {D.shape}  ({D.nbytes/2**30:.2f} GiB)", flush=True)
    print("讀 ESO nosky…", flush=True)
    E = read_blank(ESO, idx, nz, ny, nx)
    print(f"  {E.shape}", flush=True)

    # ---- 評估區域:五個選項完全共用 --------------------------------------
    zones = {
        "far field":      f((d_all > 30) & (d_main > 110) & (edge > 15))[ok],
        "small src 1-3":  f((d_oth > 1) & (d_oth <= 3) & (d_main > 30) & (edge > 15))[ok],
        "Haro 11 1-3":    f((d_main > 1) & (d_main <= 3) & (d_oth > 6) & (edge > 15))[ok],
        "Haro 11 3-10":   f((d_main > 3) & (d_main <= 10) & (d_oth > 6) & (edge > 15))[ok],
    }
    for k, v in zones.items():
        print(f"  評估區 {k:<15} {int(v.sum()):>7,} spaxel", flush=True)
    far = zones["far field"]
    Dfar = D[:, far].mean(axis=1)

    results, spectra = {}, {}
    ref_lm = None
    for title, m in MASKS:
        print(f"\n=== {title}   訓練 {int(m.sum()):,}", flush=True)
        sky, C, lm = learn(D, m, args.K, args.basis, verbose=title[:1])
        if ref_lm is None:
            ref_lm = lm            # 所有選項共用 A 的通道劃分,否則比的是不同題目
        R, coef = fit_all(D, sky)
        row = dict(n_train=int(m.sum()))
        for zn, zm in zones.items():
            r = R[:, zm].mean(axis=1)
            e = E[:, zm].mean(axis=1)
            row[zn] = dict(
                rms_line=rms(r[ref_lm]), rms_cont=rms(r[~ref_lm]),
                bias=float(np.median(r)), vs_eso=rms(r - e),
                eso_rms=rms(e))
            spectra[(title, zn)] = (r, e, D[:, zm].mean(axis=1) - Dfar)
        results[title] = row
        del R
        print(f"    遠場 rms(線內) {row['far field']['rms_line']:.4f}  "
              f"rms(線外) {row['far field']['rms_cont']:.4f}  "
              f"對 ESO {row['far field']['vs_eso']:.4f}", flush=True)

    # ---- 報表 -------------------------------------------------------------
    print("\n\n" + "=" * 100)
    print("① 天空扣得乾不乾淨 —— 遠場殘差（只有天空的地方，越小越好）")
    print(f"{'遮罩':<34}{'訓練樣本':>10}{'線內 rms':>11}{'線外 rms':>11}"
          f"{'偏移':>10}{'對 ESO':>10}")
    print("-" * 100)
    for t, r in results.items():
        z = r["far field"]
        print(f"{t:<34}{r['n_train']:>10,}{z['rms_line']:>11.4f}"
              f"{z['rms_cont']:>11.4f}{z['bias']:>+10.4f}{z['vs_eso']:>10.4f}")
    print(f"{'(ESO 自己的殘差 rms)':<34}{'':>10}{'':>11}{'':>11}{'':>10}"
          f"{list(results.values())[0]['far field']['eso_rms']:>10.4f}")

    print("\n② 源保留 —— 源外圈的連續譜（真值 P 在最後一列，越接近越好）")
    win = [(5250, 5450), (5600, 5750), (6050, 6200)]
    wm = np.zeros(nz, bool)
    for a, b in win:
        wm |= (wl >= a) & (wl < b)
    for zn in ("small src 1-3", "Haro 11 1-3", "Haro 11 3-10"):
        print(f"\n   {zn}:")
        P = spectra[(MASKS[0][0], zn)][2]
        for t, _ in MASKS:
            r = spectra[(t, zn)][0]
            print(f"      {t:<34} 殘差連續譜中位 {np.median(r[wm]):>8.4f}"
                  f"   保留 {np.median(r[wm])/np.median(P[wm]):>6.2f}")
        print(f"      {'真值 P（未經擬合）':<34}{np.median(P[wm]):>19.4f}")

    print("\n③ 對 ESO 的差距（各區域，越小代表越接近參考流程）")
    print(f"{'遮罩':<34}" + "".join(f"{z:>16}" for z in zones))
    print("-" * 100)
    for t, r in results.items():
        print(f"{t:<34}" + "".join(f"{r[z]['vs_eso']:>16.4f}" for z in zones))

    # ---- 圖 ---------------------------------------------------------------
    cols = ["#7f7f7f", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    sm = lambda y, k=9: np.convolve(y, np.ones(k)/k, mode="same")
    show = ["far field", "Haro 11 1-3", "small src 1-3"]
    fig, ax = plt.subplots(len(show), 2, figsize=(17, 3.7*len(show)),
                           gridspec_kw=dict(width_ratios=[2.2, 1]))
    for row, zn in enumerate(show):
        P = spectra[(MASKS[0][0], zn)][2]
        eso = spectra[(MASKS[0][0], zn)][1]
        for col, (lo, hi) in enumerate([(4750, 9300), (5540, 5620)]):
            a = ax[row][col]
            mm = (wl >= lo) & (wl < hi)
            a.axhline(0, color="0.6", lw=0.8)
            if zn != "far field":
                a.plot(wl[mm], sm(P)[mm], lw=1.3, color="k", alpha=0.6,
                       label="$P$ = truth (no fitting)")
            a.plot(wl[mm], sm(eso)[mm], lw=0.9, color="#9467bd", ls="--",
                   alpha=0.8, label="ESO nosky")
            for (t, _), c in zip(MASKS, cols):
                a.plot(wl[mm], sm(spectra[(t, zn)][0])[mm], lw=1.0, color=c,
                       label=t)
            a.set_xlabel("wavelength [A]")
            if col == 0:
                a.set_ylabel(f"{zn}\n({int(zones[zn].sum())} spaxels)", fontsize=9)
                if row == 0:
                    a.legend(fontsize=7, ncol=2)
            else:
                a.set_title("zoom: O I 5577 (brightest sky line)", fontsize=9)
            a.grid(alpha=0.25)
    fig.suptitle("what each training mask gives you   —   residual spectrum per zone "
                 "(smoothed 9 px for display)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = fig_dir / "12_mask_compare.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)

    (fig_dir / "mask_compare_basis.json").write_text(json.dumps(
        dict(basis=args.basis, K=args.K, xcut=args.xcut, r_oth=args.r_oth,
             r_haro=args.r_haro, results=results), ensure_ascii=False, indent=2))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
