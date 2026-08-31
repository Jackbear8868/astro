"""自我一致地修正 C_sky 的形狀,再端到端驗收。

C_sky 是用 running median + spline 從平均天空譜估的,而迭代到最後只剩一部分通道沒
被遮、分布還不均勻,spline 在約束不均的地方會留下平滑的形狀偏差。s 固定之後模型裡
沒有自由的平滑成分,這種偏差只能倒進源模板,成為源流量的加法偏差。診斷見
sky_continuum_dof 與 flux_bias_map。

修法:
    1. 用現在的模型解 blank,取殘差
    2. 只用非天光線的通道算逐通道中位 —— 修的是連續譜,不是線
    3. 平滑成 delta_C(λ),外推到全部通道
    4. C_sky <- C_sky + delta_C

樣本完全沒變,還是只有 blank。把源扣掉、拿源區當天空樣本的話,源就有機會漏進天空
模型;這裡不碰源區,只修 C_sky 的形狀和 blank 格實際需要的不一致。

事先訂好的判準:主源旁保留率進步 >= 0.02、遠場殘差留在 ±0.05、blank 上的加法偏差
明顯往 0 靠。

    conda run -n astro python src/skymodel/experiments/ccorr_continuum.py
"""
import argparse
import json
import shutil
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.interpolate import UnivariateSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from products import fit_dirs, sky_amplitude_params  # noqa: E402
from utils import (build_amplitude_field, fit_blank, main_source_group,  # noqa: E402
                   running_median, robust_spread)

ROOT = Path(__file__).resolve().parents[3]
# 圖與量測值一律寫中央,檔名帶 pointing,否則要比較幾顆就得開幾個目錄。
# SKYMODEL_EVAL 可以把圖改寫到別處,讓 checkout 之外的結果也有地方放;
# 不設就是 repo 裡的老位置。
EVAL = Path(os.environ.get(
    "SKYMODEL_EVAL", ROOT / "results/skymodel/evaluation")).expanduser() / "sky_basis"


def main():
    ap = argparse.ArgumentParser(description="修正 C_sky 的形狀")
    ap.add_argument("--work", default="results/skymodel/p01")
    ap.add_argument("--run", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="修正後的天空模型寫到哪。省略 = "
                         "results/skymodel/experiments/ccorr/{pointing}。"
                         "不寫進工作區 —— pNN/ 底下只放 pipeline.py 的產物")
    ap.add_argument("--window", type=int, default=151,
                    help="平滑 delta_C 的 running median 寬度(通道)。逐通道中位"
                         "本身還帶著雜訊,不平滑的話會把雜訊寫進天空模型")
    ap.add_argument("--smooth", type=float, default=0.02,
                    help="spline 的平滑度(相對於資料的變異)")
    args = ap.parse_args()

    W = ROOT / args.work
    run, _ = fit_dirs(W, args.run)
    meta = json.loads((run / "meta.json").read_text())
    p = sky_amplitude_params(meta)
    sky_dir = ROOT / meta["sky_dir"]

    seg   = fits.getdata(W / "step01/segmentation_input.fits").astype(int)
    white = np.asarray(fits.getdata(W / "step01/whitelight_nosky.fits"), float)
    wl    = np.load(sky_dir / "wavelength.npy")
    C_sky = np.load(sky_dir / "sky_continuum.npy")
    basis = np.load(sky_dir / f"sky_line_basis_{meta['basis']}_K{meta['K']}.npy")
    lmask = np.load(sky_dir / "continuum_iterations.npz")["line_mask"][-1]
    sky   = np.vstack([C_sky, basis])
    s_free = np.load(run / "sky_continuum_amplitude_per_spaxel.npy").astype(float)
    valid = white != 0
    mg, _, _ = main_source_group(seg, np.where(valid, white, np.nan),
                                        W / "step04")
    blank2d = valid & (seg == 0) & np.isfinite(s_free)
    s_hat, _ = build_amplitude_field(s_free, seg, blank2d, p["min_source_distance"], p["min_main_source_distance"],
                                p["train_clip_sigma"],
                                main=mg)

    ys, xs = np.nonzero(blank2d)
    with fits.open(ROOT / meta["cube"], memmap=True) as h:
        D = np.asarray(h["DATA"].data, np.float32).reshape(len(wl), -1)
        D = D[:, ys * seg.shape[1] + xs]
    print(f"blank {ys.size:,} 格")
    c = fit_blank(D, sky, s_fix=s_hat[ys, xs])
    R = D - sky.T @ c
    del D

    med = np.nanmedian(R, axis=1)
    se  = 1.253 * robust_spread(R[np.isfinite(R)].ravel()[:200000]) / np.sqrt(ys.size)
    cont = ~lmask & np.isfinite(med)
    print(f"逐通道中位的標準誤 ≈ {se:.4f}   連續譜通道 {int(cont.sum()):,} / {len(wl):,}")
    print(f"平均殘差(連續譜通道):中位 {np.median(med[cont]):+.4f}   "
          f"範圍 {med[cont].min():+.4f} ~ {med[cont].max():+.4f}")

    # 只用非天光線通道決定形狀再外推:天光線通道的殘差被線的擬合誤差主導。
    x = np.flatnonzero(cont).astype(float)
    y = running_median(med[cont], args.window)
    spl = UnivariateSpline(x, y, k=3, s=len(x) * (args.smooth * robust_spread(y)) ** 2, ext=3)
    dC = spl(np.arange(len(wl), dtype=float))
    print(f"delta_C:中位 {np.median(dC):+.4f}   "
          f"藍端 {dC[:400].mean():+.4f}  紅端 {dC[-400:].mean():+.4f}   "
          f"跨度 {dC.max()-dC.min():.4f}")
    print(f"  相對於 C_sky 的比例 中位 {np.median(dC/C_sky)*100:+.3f}%")

    out = (Path(args.out_dir) if args.out_dir
           else ROOT / "results/skymodel/experiments/ccorr" / W.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(sky_dir, out)
    np.save(out / "sky_continuum.npy", C_sky + dC)
    np.save(out / "delta_continuum.npy", dC)
    print(f"\nsaved -> {out}  (sky_continuum.npy 已更新,basis 原樣沿用)")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    ax[0].plot(wl[cont], med[cont], lw=.4, color="0.6", label="median residual")
    ax[0].plot(wl, dC, lw=1.4, color="#d62728", label="$\\Delta C$ (smoothed)")
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set_xlabel("wavelength [$\\AA$]"); ax[0].set_ylabel("flux")
    ax[0].grid(alpha=.25); ax[0].legend(fontsize=8)
    ax[0].set_title("shape error of $C_{sky}$, measured on blank spaxels", fontsize=10)
    ax[1].plot(wl, 100 * dC / C_sky, lw=1.0, color="#1f77b4")
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_xlabel("wavelength [$\\AA$]")
    ax[1].set_ylabel("$\\Delta C / C_{sky}$  [%]"); ax[1].grid(alpha=.25)
    ax[1].set_title("as a fraction of the continuum", fontsize=10)
    fig.suptitle(f"self-consistent correction to $C_{{sky}}$   {args.work}",
                 fontsize=12)
    fig.tight_layout()
    o = EVAL / f"ccorr_continuum_{W.name}.png"
    o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(o, dpi=140, bbox_inches="tight")
    print(f"saved -> {o}")


if __name__ == "__main__":
    main()
