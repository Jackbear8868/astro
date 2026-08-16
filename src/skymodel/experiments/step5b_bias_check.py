"""我們的合併品和官方合併品之間,有沒有系統性偏差?

step5b 的 --compare 只印一個全域中位數。那不夠 —— 偏差如果是隨波長變、或者
只出現在某些空間位置(視野邊緣、dither 覆蓋不同的地方),全域中位數會把它
平均掉,看起來乾乾淨淨。

這裡把同一個量沿三個方向攤開:

    逐通道   bias(λ) = ⟨(ours − ESO) / σ_ESO⟩ 對所有 spaxel 平均
             會抓到「藍端偏高、紅端偏低」這種波長相依的問題,例如流量刻度
             或波長對齊差了一點點(差一點點就會在天光線兩側形成正負對稱的尖峰)

    逐 spaxel bias(x,y) = ⟨(ours − ESO) / σ_ESO⟩ 對所有通道平均
             會抓到「只有邊緣壞掉」「某一顆曝光的位移量錯了」這種空間圖樣

    全體分布 (ours − ESO)/σ_ESO 的直方圖
             中心不在 0 就是有整體偏移;比高斯胖就是有離群的壞區

除以 σ_ESO 而不是除以流量:我們要問的是「差多少個雜訊」,不是「差百分之幾」。
天空亮的地方差 1% 可能還在雜訊裡,暗的地方差 1% 可能是 10 sigma。

注意 σ_ESO 只是尺度,不是「誰對」的判準 —— 兩邊的雜訊高度相關(同一批光子),
所以這個比值會比真正獨立的兩次觀測小很多。它有意義的地方是「有沒有結構」,
不是它的絕對大小。

    conda run -n astro python src/skymodel/experiments/step5b_bias_check.py \\
        --ours results/skymodel/ne_pointing/step05b/merged_EXP123_wsky.fits \\
        --ref  data/Haro11_NEpointing_wsky.fits
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[3]
STEP03  = ROOT / "results/skymodel/ne_pointing/step03"
FIGURES = ROOT / "results/skymodel/figures"


def main():
    ap = argparse.ArgumentParser(description="合併品的系統性偏差檢查")
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    wl = np.load(STEP03 / "wavelength.npy")

    with fits.open(args.ours, memmap=True) as ho, fits.open(args.ref, memmap=True) as hr:
        nz, ny, nx = ho["DATA"].data.shape
        nexp = np.asarray(ho["NEXP"].data) if "NEXP" in ho else None
        # 只比「滿曝光」的 voxel。邊緣只有 1-2 顆曝光的地方,兩邊的加權本來
        # 就不同,拿去比會把「覆蓋不同」誤判成「合併錯了」。
        full = (nexp == nexp.max()) if nexp is not None else np.ones((nz, ny, nx), bool)

        # 累加器:逐通道、逐 spaxel 各一組(和 sum-of-squares 一起,才算得出散布)
        c_s = np.zeros(nz); c_s2 = np.zeros(nz); c_n = np.zeros(nz)
        p_s = np.zeros(ny * nx); p_s2 = np.zeros(ny * nx); p_n = np.zeros(ny * nx)
        rat_s = np.zeros(nz); rat_n = np.zeros(nz)          # STAT 比,逐通道
        wo = np.zeros((ny, nx)); wr = np.zeros((ny, nx)); wn = np.zeros((ny, nx))
        rng = np.random.default_rng(0)
        samp_d, samp_f = [], []                             # 抽樣,用來畫直方圖

        for j in range(0, nz, args.chunk):
            k = min(j + args.chunk, nz)
            do = np.asarray(ho["DATA"].data[j:k], np.float64)
            vo = np.asarray(ho["STAT"].data[j:k], np.float64)
            dr = np.asarray(hr["DATA"].data[j:k], np.float64)
            vr = np.asarray(hr["STAT"].data[j:k], np.float64)
            m  = (np.isfinite(do) & np.isfinite(dr) & np.isfinite(vr) & (vr > 0)
                  & np.isfinite(vo) & (vo > 0) & full[j:k])

            z = np.where(m, (do - dr) / np.sqrt(np.where(vr > 0, vr, 1.0)), 0.0)
            c_s[j:k]  += z.sum(axis=(1, 2));  c_s2[j:k] += (z ** 2).sum(axis=(1, 2))
            c_n[j:k]  += m.sum(axis=(1, 2))
            p_s  += z.sum(axis=0).ravel();    p_s2 += (z ** 2).sum(axis=0).ravel()
            p_n  += m.sum(axis=0).ravel()

            rv = np.where(m, vo / np.where(vr > 0, vr, 1.0), 0.0)
            rat_s[j:k] += rv.sum(axis=(1, 2)); rat_n[j:k] += m.sum(axis=(1, 2))

            wo += np.where(m, do, 0.0).sum(axis=0); wr += np.where(m, dr, 0.0).sum(axis=0)
            wn += m.sum(axis=0)

            sel = m & (rng.random(m.shape) < 0.002)
            samp_d.append(z[sel])
            g = sel & (np.abs(dr) > 1e-3)
            samp_f.append((do[g] / dr[g]))
            print(f"   {k:>5}/{nz}", end="\r", flush=True)

    zc = np.where(c_n > 0, c_s / np.maximum(c_n, 1), np.nan)          # 逐通道均值
    zp = np.where(p_n > 0, p_s / np.maximum(p_n, 1), np.nan)          # 逐 spaxel 均值
    zc_sd = np.sqrt(np.maximum(c_s2 / np.maximum(c_n, 1) - zc ** 2, 0))
    rc = np.where(rat_n > 0, rat_s / np.maximum(rat_n, 1), np.nan)
    d_all = np.concatenate(samp_d); f_all = np.concatenate(samp_f)
    wo = np.where(wn > 0, wo / np.maximum(wn, 1), np.nan)
    wr = np.where(wn > 0, wr / np.maximum(wn, 1), np.nan)

    n_tot = float(c_n.sum())
    print(f"\n比對的 voxel {n_tot:,.0f}  ({100 * n_tot / (nz * ny * nx):.1f}% 的 cube)")
    print(f"\n(ours - ESO)/sigma_ESO")
    print(f"  全體       均值 {np.average(zc[c_n > 0], weights=c_n[c_n > 0]):+.4f}"
          f"   中位 {np.median(d_all):+.4f}   散布 {np.std(d_all):.4f}")
    print(f"  逐通道均值 中位 {np.nanmedian(zc):+.4f}"
          f"   16-84% [{np.nanpercentile(zc, 16):+.4f}, {np.nanpercentile(zc, 84):+.4f}]"
          f"   最大|.| {np.nanmax(np.abs(zc)):.4f} @ {wl[np.nanargmax(np.abs(zc))]:.1f} A")
    print(f"  逐spaxel均值 中位 {np.nanmedian(zp):+.4f}"
          f"   16-84% [{np.nanpercentile(zp, 16):+.4f}, {np.nanpercentile(zp, 84):+.4f}]"
          f"   最大|.| {np.nanmax(np.abs(zp)):.4f}")
    print(f"\n流量比 ours/ESO   中位 {np.median(f_all):.5f}"
          f"   16-84% [{np.percentile(f_all, 16):.5f}, {np.percentile(f_all, 84):.5f}]")
    print(f"STAT 比 ours/ESO  中位 {np.nanmedian(rc):.4f}"
          f"   波長端點 {rc[10]:.4f} (藍) -> {rc[-10]:.4f} (紅)")
    g = np.isfinite(wo) & np.isfinite(wr)
    print(f"白光圖相關係數 {np.corrcoef(wo[g], wr[g])[0, 1]:.5f}")

    # ---------------- 圖 ----------------
    fig, ax = plt.subplots(2, 2, figsize=(15, 8))

    a = ax[0][0]
    a.plot(wl, zc, lw=0.5, color="#1f77b4")
    a.axhline(0, color="k", lw=0.7)
    a.set_xlabel("wavelength [$\\AA$]"); a.set_ylabel("mean (ours-ESO)/$\\sigma_{ESO}$")
    a.set_title("bias vs wavelength", fontsize=10); a.grid(alpha=0.25)
    v = float(np.nanpercentile(np.abs(zc), 99.5)) * 1.5
    a.set_ylim(-v, v)

    a = ax[0][1]
    v = float(np.nanpercentile(np.abs(zp), 99)) or 1e-3
    im = a.imshow(zp.reshape(ny, nx), origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    plt.colorbar(im, ax=a, fraction=0.046)
    a.set_title("bias vs position", fontsize=10)
    a.set_xlabel("x [pix]"); a.set_ylabel("y [pix]")

    a = ax[1][0]
    lim = float(np.percentile(np.abs(d_all), 99.5))
    a.hist(d_all, bins=200, range=(-lim, lim), color="#1f77b4", histtype="step")
    a.axvline(0, color="k", lw=0.7)
    a.set_xlabel("(ours-ESO)/$\\sigma_{ESO}$"); a.set_ylabel("voxels")
    a.set_title(f"distribution   median {np.median(d_all):+.4f}", fontsize=10)
    a.set_yscale("log"); a.grid(alpha=0.25)

    a = ax[1][1]
    a.plot(wl, rc, lw=0.5, color="#d62728")
    a.axhline(1, color="k", lw=0.7)
    a.set_xlabel("wavelength [$\\AA$]"); a.set_ylabel("STAT ours / STAT ESO")
    a.set_title("variance ratio vs wavelength", fontsize=10); a.grid(alpha=0.25)
    a.set_ylim(0, 1.2)

    fig.suptitle(f"systematic bias: {Path(args.ours).name} vs {Path(args.ref).name}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.out) if args.out else FIGURES / "step5b_bias_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
