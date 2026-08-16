"""精煉前後,天空扣得更好還是更差 —— 逐 spaxel 畫成一張地圖。

抽樣幾個點或幾個方框,結論會隨抽樣位置而變。整張地圖沒有這個問題:每個
spaxel 一個像素,好壞由顏色直接呈現,不必挑代表。

殘差的定義和其他 step4d 的圖一致:

    blank   residual = sky_subtracted                    (源模型 = 0)
    源區    residual = sky_subtracted − 源模型            兩者都扣掉,理想值 0

以 1/sigma 為權重取 rms,所以地圖的單位是「該 spaxel 的雜訊的幾倍」——
不同位置的雜訊差好幾倍,用流量單位畫會變成「哪裡比較亮」的地圖。

    conda run -n astro python src/skymodel/experiments/step4d_improve_map.py \\
        --basis svd -K 30 \\
        --best results/skymodel/step04b/classification_nobasis_s0.0_4700-8000_4700-8000_L1cum__eso.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step5_fit_spaxels import build_templates, CUBE, STEP01, STEP03, STEP05, N_SRC
from templates import air_to_vacuum

ROOT    = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "results/skymodel/figures"


def rms_map(sub_path, amap_path, seg, T_all, V, chunk=200):
    """逐 spaxel 的加權殘差 rms。sky_subtracted 已經扣過天空,這裡再扣源。"""
    A = np.load(amap_path).reshape(N_SRC, -1)
    with fits.open(sub_path, memmap=True) as h:
        nz = h["DATA"].header["NAXIS3"]
        acc = np.zeros(seg.size)
        cnt = np.zeros(seg.size)
        for j in range(0, nz, chunk):
            S = np.asarray(h["DATA"].data[j:j + chunk], np.float64).reshape(
                -1, seg.size)
            # 源模型:該 spaxel 所屬 ID 的模板 × 它自己的振幅
            for rid, T in T_all.items():
                m = np.flatnonzero(seg == rid)
                if m.size == 0:
                    continue
                S[:, m] -= T[j:j + chunk] @ np.nan_to_num(A[:T.shape[1], m])
            v = V[j:j + chunk]
            g = np.isfinite(S) & np.isfinite(v) & (v > 0)
            acc += np.where(g, S ** 2 / np.where(v > 0, v, 1.0), 0.0).sum(axis=0)
            cnt += g.sum(axis=0)
    return np.where(cnt > 0, np.sqrt(acc / np.maximum(cnt, 1)), np.nan)


def main():
    ap = argparse.ArgumentParser(description="精煉前後的逐 spaxel 改善地圖")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--s-fix", type=float, default=1.0)
    ap.add_argument("--best", required=True)
    ap.add_argument("--tags", nargs=2, default=["before_refine", "after_refine"],
                    metavar=("BEFORE", "AFTER"),
                    help="要比較的兩個 step5 產物的字尾。s 的處理必須相同,"
                         "否則量到的是「換天空模型」和「換 s 處理」的混合")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = f"{args.basis}_K{args.K}_s_{args.s_fix}"
    seg2 = fits.getdata(STEP01 / "seg.fits")
    ny, nx = seg2.shape
    seg  = seg2.reshape(-1)
    wl   = np.load(STEP03 / "wavelength.npy")
    T_all = build_templates(np.load(args.best), air_to_vacuum(wl))

    with fits.open(CUBE, memmap=True) as h:
        V = np.asarray(h["STAT"].data, np.float64).reshape(len(wl), -1)

    out = {}
    for nm in args.tags:
        out[nm] = rms_map(STEP05 / f"sky_subtracted_{tag}_{nm}.fits",
                          STEP05 / f"A_map_{tag}_{nm}.npy", seg, T_all, V)
        print(f"{nm}: 中位 rms {np.nanmedian(out[nm]):.4f}", flush=True)

    b, a = out[args.tags[0]], out[args.tags[1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        chg = 100 * (a / b - 1)
    good = np.isfinite(chg)
    print(f"\n改善的 spaxel {100 * np.mean(chg[good] < 0):.1f}%   "
          f"變差的 {100 * np.mean(chg[good] > 0):.1f}%")
    print(f"變化的中位 {np.median(chg[good]):+.2f}%   "
          f"16-84 百分位 [{np.percentile(chg[good], 16):+.2f}, "
          f"{np.percentile(chg[good], 84):+.2f}]%")
    for nm, m in (("Haro 11 (seg=1)", seg == 1), ("其他 8 個源", np.isin(seg, [
            int(i) for i in np.load(args.best)["id"] if i != 1])),
            ("blank", seg == 0)):
        g = good & m
        print(f"  {nm:<16} {int(g.sum()):>7} spaxel   中位 {np.median(chg[g]):+.2f}%"
              f"   改善的佔 {100 * np.mean(chg[g] < 0):.0f}%")

    fig, ax = plt.subplots(1, 3, figsize=(19, 6))
    q = np.nanpercentile(b[good], [2, 98])
    for k, (nm, m) in enumerate(((args.tags[0], b), (args.tags[1], a))):
        im = ax[k].imshow(m.reshape(ny, nx), origin="lower", cmap="viridis",
                          vmin=q[0], vmax=q[1])
        ax[k].set_title(f"{nm}   weighted residual rms", fontsize=11)
        plt.colorbar(im, ax=ax[k], fraction=0.046)
    # 發散色階、以 0 為中心:好壞是有方向的量,順序色階讀不出「哪邊是零」
    v = float(np.nanpercentile(np.abs(chg[good]), 98))
    im = ax[2].imshow(chg.reshape(ny, nx), origin="lower", cmap="RdBu_r",
                      vmin=-v, vmax=v)
    ax[2].contour((seg2 > 0).astype(float), levels=[0.5], colors="k",
                  linewidths=0.5)
    ax[2].set_title("change [%]   blue = better after refine", fontsize=11)
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    for a_ in ax:
        a_.set_xlabel("x [pix]")
    ax[0].set_ylabel("y [pix]")
    fig.suptitle("does the refined sky model subtract better?   "
                 f"[{args.basis} K={args.K}]", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = Path(args.out) if args.out else FIGURES / "step4d_improve_map.png"
    fig.savefig(p, dpi=135, bbox_inches="tight")
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
