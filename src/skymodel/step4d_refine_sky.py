"""把源扣掉,讓源區也變成天空樣本,重新推導 C_sky 與天光線 basis。

step3 只從 blank spaxel 學天空(84,052 個),源區的 14,610 個一直被排除在外 ——
而其中 Haro 11 本體就佔 13,726 個,正好是視場中央、天空最需要扣準的地方。
源被模型扛住之後,那些 spaxel 也可以當天空樣本用:

    ①  9 個源的區域逐 spaxel 擬合(模板形狀與 z 固定,只解係數)
            D(p,λ) = Σⱼ aⱼ(p)·Tⱼ(λ) + s·C_sky(λ) + Σₖ cₖ(p)·Lₖ(λ)
    ②  只扣掉源那一項,天空留著
            sky_only(p,λ) = D(p,λ) − Σⱼ aⱼ(p)·Tⱼ(λ)
    ③  其餘 spaxel 原樣保留 —— blank、26 個假源、教授移除的 #27/#30,
        它們本來就是天空
    ④  用全視場重新推導 C_sky 與 basis

和 step5 的關係:同一個擬合,相反的輸出。step5 扣天空留源,這裡扣源留天空。

沒有加任何 spaxel 層級的門檻。實測(step4d_spaxel_quality)扣完源之後,8/9 個源
的逐 spaxel reduced chi2 中位落在 1.71–2.57,blank 是 1.95 —— 統計上分不出來,
沒有證據需要剔除。真正的壞值(宇宙線、壞像素)由逐通道的 CLIP_SIGMA 處理,
那個機制已經存在而且對門檻不敏感(10~100 結果相同)。

源的振幅是用**現有的**(step3)天空模型解出來的 —— 這是必然的,那是唯一存在的
天空模型。所以這是單向的一次修正,不迭代;教授 2026-08-07 指示迭代只用於
天空連續譜與天光線的區域劃分,不重做分類。

    conda run -n astro python src/skymodel/step4d_refine_sky.py \\
        --basis svd -K 30 --methods svd \\
        --best results/skymodel/step04b/classification_nobasis_s0.0_4700-8000_4700-8000_L1cum__eso.npz
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
import time
from pathlib import Path

import numpy as np
from astropy.io import fits

from step3_sky_basis import (learn_sky_basis, CLIP_SIGMA, WINDOW, THRESHOLDS,
                             MAX_ITER)
from step5_fit_spaxels import build_templates, fit_source, N_SRC, MIN_COVERAGE
from templates import air_to_vacuum
from utils import estimate_continuum

ROOT   = Path(__file__).resolve().parents[2]
STEP01 = ROOT / "results/skymodel/step01"
STEP03 = ROOT / "results/skymodel/step03"
STEP04D = ROOT / "results/skymodel/step04d"
WSKY   = ROOT / "data/Haro11_NEpointing_wsky.fits"


def main():
    ap = argparse.ArgumentParser(description="扣掉源之後用全視場重推天空模型")
    ap.add_argument("--basis", default="svd",
                    help="解源振幅時使用的現有 basis")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--methods", nargs="+", default=["svd"],
                    help="新 basis 要用哪些分解方法")
    ap.add_argument("--s-fix", type=float, default=1.0,
                    help="解源振幅時天空連續譜的固定係數。預設 1.0,和 step5 一致 —— "
                         "A·T 和 s·C_sky 的形狀都平滑,讓 s 自由會使天空吸走源的光,"
                         "源就會被扣得不夠")
    ap.add_argument("--best", required=True)
    ap.add_argument("--sources", choices=["subtract", "exclude"], default="subtract",
                    help="真源的區域怎麼處理。subtract = 扣掉源模型之後也當天空樣本"
                         "(預設);exclude = 整個排除,只把 28 個假源與教授移除的"
                         "#27/#30 當成 blank 併入樣本。後者是隔離變因用的對照組 —— "
                         "假源本來就是天空,不需要扣任何模型,樣本裡就不會混進源模型"
                         "的誤差")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # 兩種樣本產出的是不同的天空模型,必須分開存,否則後跑的會靜靜蓋掉前一個
    out = Path(args.out) if args.out else (
        STEP04D if args.sources == "subtract" else STEP04D.with_name("step04d_fake"))
    out.mkdir(parents=True, exist_ok=True)

    white = fits.getdata(STEP01 / "whitelight.fits")
    seg   = fits.getdata(STEP01 / "seg.fits")
    wl    = np.load(STEP03 / "wavelength.npy")
    sky   = np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                       np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")])
    best  = np.load(args.best)
    fit_ids = [int(i) for i in best["id"]]

    with fits.open(WSKY, memmap=True) as hdul:
        nz = hdul["DATA"].header["NAXIS3"]
        D  = np.asarray(hdul["DATA"].data, np.float32).reshape(nz, -1)
        V  = np.asarray(hdul["STAT"].data, np.float32).reshape(nz, -1)

    seg_f = seg.reshape(-1)
    # 和 step3 一致:只收光譜完整的 spaxel。視野邊緣的部分覆蓋 spaxel 會被
    # nan_to_num 填 0,SVD 會認真去擬合那些捏造的零。
    valid = (white != 0).reshape(-1) & np.isfinite(D).all(axis=0)
    print(f"完整且在視野內的 spaxel:{int(valid.sum()):,}")

    # ---- 源區:扣掉源,天空留著 ----
    templates = build_templates(best, air_to_vacuum(wl))
    n_sub = 0
    if args.sources == "exclude":
        # 隔離變因用的對照組。假源(以及教授移除的 #27/#30)本來就是天空,
        # 不需要扣任何模型,所以樣本裡不會混進源模型的誤差;真源整個排除。
        # 若這一組就有改善,代表改善來自「樣本變多」;若沒有,代表原版那點
        # 改善全部來自真源的 spaxel —— 而那正是最可能把源吃掉的地方。
        n_real = int((valid & np.isin(seg_f, fit_ids)).sum())
        valid &= ~np.isin(seg_f, fit_ids)
        print(f"--sources exclude:排除 {len(fit_ids)} 個真源的 {n_real:,} 個 spaxel,"
              f"只把假源當成 blank 併入樣本")
        fit_ids = []                      # 下面的扣源迴圈自然不執行

    for t in fit_ids:
        m = valid & (seg_f == t)
        if not m.any():
            continue
        T = templates.get(t)
        c = fit_source(D[:, m], V[:, m], sky, T, s_fix=args.s_fix)
        # T 在模板覆蓋不到的通道是 NaN —— 恆星模板靜止只到 9192 A,#35 的觀測
        # 紅端(9350 A → 靜止 9362 A)就超出去了。那些通道的源沒有被扣掉,
        # 不是乾淨的天空,所以標成 NaN 讓後面的統計逐元素排除,而不是把整條
        # spaxel 丟掉(那會白白損失 231 個 spaxel 的 3670 個好通道)。
        src = T @ np.nan_to_num(c[:T.shape[1]])
        src = np.where(np.all(np.isfinite(T), axis=1)[:, None], src, np.nan)
        # 係數解不出來的 spaxel(有效通道太少)整欄是 NaN,src 會是 0,
        # 等於沒扣 —— 那種 spaxel 不該進樣本,直接從 valid 拿掉。
        okc = np.isfinite(c[:T.shape[1]]).all(axis=0)
        D[:, np.flatnonzero(m)[okc]] -= src[:, okc].astype(np.float32)
        drop = np.flatnonzero(m)[~okc]
        valid[drop] = False
        n_sub += int(okc.sum())
        nch = int((~np.all(np.isfinite(T), axis=1)).sum())
        print(f"  ID {t:>3}  {int(m.sum()):>6} spaxel  扣掉源 {int(okc.sum()):>6}"
              + (f"  (解不出來 {drop.size} 個,剔除)" if drop.size else "")
              + (f"  模板未覆蓋 {nch} 個通道,標為缺值" if nch else ""),
              flush=True)

    n_blank = int((valid & (seg_f == 0)).sum())
    n_other = int(valid.sum()) - n_blank - n_sub
    print(f"\n天空樣本 {int(valid.sum()):,} 個 = blank {n_blank:,}"
          f" + 扣過源的 {n_sub:,} + 假源與移除的源 {n_other:,}"
          f"   (step3 只用 blank 的 {n_blank:,},增加 {100*(n_sub+n_other)/n_blank:.1f}%)")

    S = D[:, valid]

    # ---- 以下和 step3 完全相同的處理,只是樣本換成全視場 ----
    # 用 nan-aware 的百分位,並把缺值一併算進 keep —— 上面標成 NaN 的是
    # 「模板沒覆蓋、源沒扣掉」的元素,它們必須逐元素排除在統計之外。
    p16, med, p84 = np.nanpercentile(S, [16, 50, 84], axis=1)
    sg   = np.maximum((p84 - p16) / 2, 1e-6)
    keep = np.isfinite(S) & (np.abs(S - med[:, None]) <= CLIP_SIGMA * sg[:, None])
    mean_sky = (np.nan_to_num(S) * keep).sum(axis=1, dtype=np.float64) \
        / np.maximum(keep.sum(axis=1), 1)
    print(f"mean_sky: sigma-clip {CLIP_SIGMA} sigma 剔除 {int((~keep).sum()):,} / "
          f"{keep.size:,} 個元素 ({100*(~keep).mean():.6f}%)")

    C_sky, sigma, line_mask, history = estimate_continuum(
        mean_sky, thresholds=THRESHOLDS, window=WINDOW, max_iter=MAX_ITER)
    print(f"line_mask: {100*line_mask.mean():.1f}% of channels  "
          f"({len(history)} iterations: "
          f"{' -> '.join(f'{100*h[2].mean():.1f}%' for h in history)})")

    np.save(out / "wavelength.npy",    wl)
    np.save(out / "mean_sky.npy",      mean_sky)
    np.save(out / "sky_continuum.npy", C_sky)
    np.save(out / "sky_sigma.npy",     sigma)
    np.save(out / "line_mask.npy",     line_mask)
    np.save(out / "iter_continuum.npy", np.array([h[0] for h in history]))
    np.save(out / "iter_sigma.npy",     np.array([h[1] for h in history]))
    np.save(out / "iter_line_mask.npy", np.array([h[2] for h in history]))
    np.save(out / "sample_mask.npy",    valid.reshape(seg.shape))

    residual = np.where(keep, S - C_sky[:, None], (med - C_sky)[:, None])
    for method in args.methods:
        t0 = time.time()
        basis, _ = learn_sky_basis(residual, K=args.K, method=method)
        np.save(out / f"sky_basis_{method}_K{args.K}.npy", basis)
        print(f"{method:8s} basis {basis.shape}  {time.time() - t0:6.1f}s",
              flush=True)

    # 和舊版比:連續譜差多少?第一條 basis 有多像?
    old_c = np.load(STEP03 / "sky_continuum.npy")
    d = C_sky - old_c
    print(f"\nC_sky 相對 step3:中位差 {np.median(d):+.4f}"
          f"  ({100*np.median(d)/np.median(old_c):+.2f}%)"
          f"   最大 |差| {np.abs(d).max():.3f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
