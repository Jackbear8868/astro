"""比較 step3 產出的各種 sky basis,在 blank region 的天空重建品質。

量的是「天空重建得多乾淨」。這是**篩選**而不是**裁決**:
自由度更高的方法一定能把天空重建得更好,但也更可能在 step4 吃掉源。
blank 區的殘差 rms 這個指標會**獎勵過度減除**,所以殘差比較大不等於比較差。
最終判準是 step4 的 Δχ²(源有沒有被保住)。

blank spaxel 的模型(和 step4 的天空項完全相同,只是少了源):

    blank(p, λ) = s(p)·C_sky(λ) + Σₖ cₖ(p)·Lₖ(λ)

設計矩陣 = [C_sky, L₁ … L_K],共 K+1 個自由參數。所有方法的 K 相同,
所以自由度一致,chi-square 可以直接比較。

評測使用**全部** blank spaxel。

用法:  python src/skymodel/step3_compare_basis.py
"""
import time

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# 搬到子目錄之後,同層的 templates / utils 不再自動可見。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import fit_chi2_coefficients, plot_compare, spectrum_stats

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP03  = ROOT / "results/skymodel/step03"
COMPARE = STEP03 / "compare"
WSKY    = ROOT / "data/Haro11_NEpointing_wsky.fits"
NOSKY   = ROOT / "data/Haro11_NEpointing_esonosky.fits"


def read_columns(cube_path, ext, cols, npix, nz, chunk=200):
    """從 cube 分批讀出指定的一維空間索引,回傳 (nz, len(cols))。"""
    out = np.empty((nz, cols.size), np.float32)
    with fits.open(cube_path, memmap=True) as hdul:
        for j in range(0, nz, chunk):
            plane = np.asarray(hdul[ext].data[j:j + chunk], np.float32)
            out[j:j + chunk] = plane.reshape(-1, npix)[:, cols]
    return out


def evaluate(d, v, design):
    """對每個 spaxel 解係數、重建、回傳 (平均殘差光譜, 每 spaxel reduced chi2)。"""
    model = np.empty_like(d)
    for j in range(d.shape[1]):
        coeff = fit_chi2_coefficients(d[:, j], v[:, j], design)
        model[:, j] = coeff @ design

    resid = d - model
    del model
    good = np.isfinite(resid) & np.isfinite(v) & (v > 0)

    # chi2 在 float32 下計算、以 float64 累加,避免產生 2.2 GB 的 float64 中間陣列。
    chi2  = np.where(good, resid ** 2 / np.where(good, v, 1), 0).sum(axis=0, dtype=np.float64)
    dof   = good.sum(axis=0) - design.shape[0]
    rchi2 = chi2 / np.maximum(dof, 1)

    total = np.where(good, resid, 0).sum(axis=1, dtype=np.float64)
    count = good.sum(axis=1)
    mean_resid = total / np.maximum(count, 1)
    return mean_resid, rchi2


def main():
    COMPARE.mkdir(parents=True, exist_ok=True)

    wl        = np.load(STEP03 / "wavelength.npy")
    C_sky     = np.load(STEP03 / "sky_continuum.npy")
    line_mask = np.load(STEP03 / "line_mask.npy")
    nz        = wl.size

    bases = {f.stem.replace("sky_basis_", ""): np.load(f)
             for f in sorted(STEP03.glob("sky_basis_*.npy"))}
    print("methods:", list(bases))

    white = fits.getdata(STEP01 / "whitelight.fits")
    seg   = fits.getdata(STEP01 / "seg.fits")
    valid = white != 0
    blank = valid & ~((seg > 0) & valid)
    npix  = valid.size

    cols = np.flatnonzero(blank.ravel())
    print(f"evaluating on all {cols.size} blank spaxels")

    t0 = time.time()
    d = read_columns(WSKY,  "DATA", cols, npix, nz)
    v = read_columns(WSKY,  "STAT", cols, npix, nz)
    mean_nosky = np.nanmean(read_columns(NOSKY, "DATA", cols, npix, nz), axis=1)
    print(f"cube read in {time.time() - t0:.1f}s")

    results = {}
    for name, B in bases.items():
        design = np.vstack([C_sky, B]).astype(np.float64)      # (K+1, nz)
        t0 = time.time()
        results[name] = evaluate(d, v, design)
        print(f"  {name:14s} K+1={design.shape[0]:3d}  {time.time() - t0:6.1f}s", flush=True)

    def row(name, mr, rc=None):
        """三個主欄位沿用 utils.spectrum_stats 的定義,與 test.py 的圖表可直接對照。

        mean          殘差的直流偏移(整體多減/少減了多少)
        sigma         = np.std,殘差隨波長變動的部分(扣掉 mean 之後剩下的)
        rms_from_zero = sqrt(mean^2 + sigma^2),離零的總距離
        """
        st = spectrum_stats(mr)
        chi = f"{np.median(rc):>10.4f}" if rc is not None else f"{'':>10}"
        return (f"{name:<14}{st['mean']:>9.4f}{st['sigma']:>10.4f}{st['rms_from_zero']:>16.4f}"
                f"{np.sqrt(np.nanmean(mr[line_mask]**2)):>11.4f}"
                f"{np.sqrt(np.nanmean(mr[~line_mask]**2)):>11.4f}{chi}")

    print()
    print("=" * 82)
    print("blank region 平均殘差光譜")
    print("  mean  = 直流偏移   sigma = 波長變動   rms_from_zero = 兩者合起來")
    print("  注意:數字小不一定好 —— 過度減除也會讓殘差變小。最終判準是 step4 的 dchi2。")
    print("=" * 82)
    print(f"{'method':<14}{'mean':>9}{'sigma':>10}{'rms_from_zero':>16}"
          f"{'rms(line)':>11}{'rms(free)':>11}{'red_chi2':>10}")
    print("-" * 82)
    for name, (mr, rc) in sorted(results.items(), key=lambda kv: spectrum_stats(kv[1][0])["sigma"]):
        print(row(name, mr, rc))
    print("-" * 82)
    print(row("ESO nosky", mean_nosky))

    # reduced chi2 分布(理想 = 1)
    plt.figure(figsize=(9, 5))
    hi = max(np.percentile(rc, 99) for _, rc in results.values())
    for name, (_, rc) in results.items():
        plt.hist(rc, bins=np.linspace(0, hi, 80), histtype="step", lw=1.1, label=name)
    plt.axvline(1.0, color="k", ls="--", lw=1, label="ideal = 1")
    plt.xlabel("reduced chi2"); plt.ylabel("N spaxels")
    plt.title("blank-region goodness of fit")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(COMPARE / "cmp_reduced_chi2.png", dpi=150); plt.close()

    # 每個方法各自跟 ESO nosky 對照
    for name, (mr, _) in results.items():
        plot_compare(wl, mr, mean_nosky, COMPARE / f"cmp_{name}.png",
                     label=name, label_compare="ESO nosky",
                     title=f"sky subtracted with {name} basis")

    print(f"\nfigures -> {COMPARE}")


if __name__ == "__main__":
    main()
