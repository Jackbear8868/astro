"""reminder ③:把相鄰 spaxel 的係數平滑掉,天空扣得會不會更好?

想法是:天空在空間上是連續變化的,所以相鄰格子的係數應該相近;把係數圖平滑
掉可以壓掉雜訊。這支程式畫兩種圖來檢驗:

  (a) 係數的空間分布圖
      平滑的對象就是這些圖。平滑前後並排,再加上「差值」——
      如果差值看起來像純雜訊,那平滑掉的就是雜訊(好事);
      如果差值裡有結構,那被抹掉的是真實的空間變化(壞事)。

  (b) 固定 10 個 blank spaxel 的殘差光譜
      版面和 blank_compare 一致:一個選項一個資料夾,各自和 ESO 比。

平滑的變體:
    未平滑        基準
    只平滑 s      只動天空連續譜的係數
    只平滑線      只動 K 條天光線的係數
    全部平滑      兩者都動

鄰域用八鄰格、權重 ∝ 1/距離²(上下左右 1、對角 1/2),這樣核才是等向的。

    conda run -n astro python src/skymodel/experiments/compare_blank_smooth.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_blank_spectra import FIXED_SPAXELS

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP03  = ROOT / "results/skymodel/step03"
FIGURES = ROOT / "results/skymodel/figures"
WSKY    = ROOT / "data/Haro11_NEpointing_wsky.fits"
ESO     = ROOT / "data/Haro11_NEpointing_esonosky.fits"

MIN_COVERAGE = 0.9
C_OURS, C_ESO = "#0d47a1", "#d62728"
VARIANTS = ["unsmoothed", "smooth s only", "smooth lines only", "smooth all"]


def shift(a, dy, dx, fill=np.nan):
    out = np.full_like(a, fill)
    ys_s = slice(max(0, -dy), a.shape[-2] - max(0, dy))
    ys_d = slice(max(0, dy),  a.shape[-2] - max(0, -dy))
    xs_s = slice(max(0, -dx), a.shape[-1] - max(0, dx))
    xs_d = slice(max(0, dx),  a.shape[-1] - max(0, -dx))
    out[..., ys_d, xs_d] = a[..., ys_s, xs_s]
    return out


def smooth(C, mask):
    """八鄰格 + 中心的加權平均,權重 ∝ 1/距離²。只在 mask 內取值。"""
    Cn = np.where(mask, C, np.nan)
    acc = np.zeros_like(C)
    wsum = np.zeros_like(C)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            w = 1.0 if (dy, dx) == (0, 0) else 1.0 / (dy * dy + dx * dx)
            v = shift(Cn, dy, dx)
            ok = np.isfinite(v)
            acc += np.where(ok, v, 0.0) * w
            wsum += ok * w
    with np.errstate(invalid="ignore"):
        return np.where(wsum > 0, acc / np.maximum(wsum, 1e-12), np.nan)


def main():
    ap = argparse.ArgumentParser(description="reminder ③:係數空間平滑的比較")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--ylim", type=float, nargs=2, default=(-20.0, 40.0))
    ap.add_argument("--panels", type=int, default=5)
    ap.add_argument("--width", type=float, default=22.0)
    ap.add_argument("--height", type=float, default=3.2)
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    wl    = np.load(STEP03 / "wavelength.npy")
    lm    = np.load(STEP03 / "iter_line_mask.npy")[0]
    sky   = np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                       np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")])
    K = sky.shape[0]

    with fits.open(WSKY, memmap=True) as h:
        D = np.asarray(h["DATA"].data, np.float32)
    nz, ny, nx = D.shape
    D = D.reshape(nz, -1)
    cov = np.isfinite(D).sum(axis=0) / nz
    E = fits.getdata(ESO).reshape(nz, -1)
    # ESO 也要一起要求有限 —— 它有少數通道是 NaN,不擋掉的話下面拿它當基準的
    # 統計整列都會變成 NaN。
    blank = ((white != 0).ravel() & (seg.ravel() == 0) & (cov >= MIN_COVERAGE)
             & np.isfinite(D).all(axis=0) & np.isfinite(E).all(axis=0))
    idx = np.flatnonzero(blank)
    De = D[:, idx].astype(np.float64)
    del D

    # 一次 pinv 解完所有 blank 的係數(未加權、全部通道,和 fit_blank 的快路徑相同)
    C0 = np.linalg.pinv(sky.T) @ De
    maps = np.full((K, ny * nx), np.nan)
    maps[:, idx] = C0
    maps = maps.reshape(K, ny, nx)
    mask = np.zeros(ny * nx, bool); mask[idx] = True; mask = mask.reshape(ny, nx)
    print(f"basis={args.basis} K={args.K}   blank {idx.size:,} 個 spaxel\n")

    Sm = smooth(maps, mask)                      # 全部係數都平滑一份,之後挑著用
    coefs = {}
    for v in VARIANTS:
        M = maps.copy()
        if v in ("smooth s only", "smooth all"):
            M[0] = Sm[0]
        if v in ("smooth lines only", "smooth all"):
            M[1:] = Sm[1:]
        coefs[v] = M

    # 固定樣本在 blank 陣列裡的位置
    want = [y * nx + x for y, x in FIXED_SPAXELS[:args.n]]
    pos = {p: int(np.flatnonzero(idx == p)[0]) for p in want if p in set(idx.tolist())}

    R = {}
    for v in VARIANTS:
        c = coefs[v].reshape(K, -1)[:, idx]
        R[v] = De - sky.T @ c

    print(f"{'設定':>20}{'|mean|':>10}{'線內 sigma':>12}{'線外 sigma':>12}"
          f"{'rms':>9}{'s 散布':>10}")
    print("-" * 74)
    Ee = E[:, idx].astype(np.float64)
    print(f"{'ESO nosky':>20}{np.median(np.abs(Ee.mean(0))):>10.4f}"
          f"{np.median(Ee[lm].std(0)):>12.4f}{np.median(Ee[~lm].std(0)):>12.4f}"
          f"{np.median(np.sqrt((Ee**2).mean(0))):>9.4f}")
    print("-" * 74)
    for v in VARIANTS:
        X = R[v]
        s = coefs[v][0][mask]
        print(f"{v:>20}{np.median(np.abs(X.mean(0))):>10.4f}"
              f"{np.median(X[lm].std(0)):>12.4f}{np.median(X[~lm].std(0)):>12.4f}"
              f"{np.median(np.sqrt((X**2).mean(0))):>9.4f}"
              f"{(np.percentile(s,84)-np.percentile(s,16))/2:>10.5f}")

    # ---------- (a) 係數的空間分布圖 ----------
    out = FIGURES / f"blank_smooth_{args.basis}_K{args.K}"
    out.mkdir(parents=True, exist_ok=True)
    rows = [(0, "s  (sky continuum)")] + [(k, f"c{k}  (sky line basis)")
                                          for k in (1, 2, 5)]
    fig, axes = plt.subplots(len(rows), 3, figsize=(15, 4.2 * len(rows)))
    for r, (k, lab) in enumerate(rows):
        a0 = maps[k]; a1 = Sm[k]; d = a1 - a0
        v = np.nanpercentile(np.abs(a0[mask]), 98)
        dv = np.nanpercentile(np.abs(d[mask]), 98)
        for c, (img, ttl, lim) in enumerate(
                [(a0, "before", v), (a1, "after smoothing", v),
                 (d, "difference (after - before)", dv)]):
            ax = axes[r, c]
            m = ax.imshow(np.where(mask, img, np.nan), origin="lower", cmap="RdBu_r",
                          vmin=np.nanmedian(a0[mask]) - lim if c < 2 else -lim,
                          vmax=np.nanmedian(a0[mask]) + lim if c < 2 else dv)
            ax.set_title(f"{lab}   —   {ttl}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(m, ax=ax, fraction=0.046)
    fig.suptitle(f"coefficient maps, {args.basis} K={args.K}   —   "
                 f"if the difference looks like pure noise, smoothing removed noise; "
                 f"if it has structure, real spatial variation was erased", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out / "coefficient_maps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---------- (b) 固定樣本的殘差光譜 ----------
    edges = np.linspace(wl[0], wl[-1], args.panels + 1)
    nfig = 0
    for v in VARIANTS:
        sub = out / v.replace(" ", "_")
        sub.mkdir(parents=True, exist_ok=True)
        for j, p in enumerate(want, 1):
            if p not in pos:
                continue
            y0, x0 = divmod(p, nx)
            vv = R[v][:, pos[p]]
            ee = E[:, p].astype(np.float64)
            fig, axes = plt.subplots(args.panels, 1,
                                     figsize=(args.width,
                                              args.height * args.panels + 0.6))
            axes = np.atleast_1d(axes)
            for b, a in enumerate(axes):
                m = (wl >= edges[b]) & (wl <= edges[b + 1])
                a.axhline(0, color="0.55", lw=0.6, zorder=1)
                a.plot(wl[m], ee[m], lw=0.7, color=C_ESO, alpha=0.85, zorder=2,
                       label=f"ESO nosky          mean {ee.mean():+.3f}   "
                             f"sigma {ee.std():.3f}   "
                             f"rms {np.sqrt((ee**2).mean()):.3f}")
                a.plot(wl[m], vv[m], lw=0.7, color=C_OURS, alpha=0.9, zorder=3,
                       label=f"{v:<18} mean {vv.mean():+.3f}   "
                             f"sigma {vv.std():.3f}   "
                             f"rms {np.sqrt((vv**2).mean()):.3f}")
                a.set_xlim(edges[b], edges[b + 1]); a.set_ylim(*args.ylim)
                a.set_ylabel("flux  [$10^{-20}$ cgs]", fontsize=9)
                a.tick_params(labelsize=9)
                if b == 0:
                    a.legend(fontsize=9, loc="upper right", framealpha=0.9)
                    a.set_title(f"blank spaxel #{j}   (y={y0}, x={x0})   "
                                f"{args.basis} K={args.K}   —   {v}", fontsize=11)
            axes[-1].set_xlabel("wavelength [$\\AA$]", fontsize=10)
            fig.tight_layout(h_pad=0.4)
            fig.savefig(sub / f"spaxel_{j:02d}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)
            nfig += 1
    print(f"\nsaved coefficient_maps.png + {nfig} spectra -> {out}")


if __name__ == "__main__":
    main()
