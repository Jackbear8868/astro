"""源區域的殘差:wsky 扣掉我們預測的天空**和**源之後還剩下什麼。

和拿 nosky 當基準不同 —— nosky 裡面有 ESO 自己扣天空的誤差,拿它當「真值」
會把兩套流程的誤差混在一起。這裡完全在自己的產物內部閉合:

    residual = sky_subtracted - A x T
             = wsky - sky_model - source_model

模型如果完整,殘差就只剩雜訊,平均值在 0、沒有波長結構。任何殘留的結構都是
模型沒解釋掉的東西,不需要外部參考就能判讀。

一列一個 run,右邊是該列自己的統計量。chi 那一欄是殘差除以 cube 自帶的 STAT
的標準差 —— STAT 直接採用,不做任何修正,模型完美時應該是 1。

    conda run -n astro python src/skymodel/experiments/star_library_residual.py \
        --work results/skymodel/p01 --ids 22 50 \
        --run default:"pipeline" alt:"alternative run"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fitting import N_SRC, build_templates  # noqa: E402
from templates import air_to_vacuum  # noqa: E402
from products import fit_dirs, spectrum_stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
COLORS = ["#ff7f0e", "#1f77b4", "#2ca02c", "#d62728"]


def source_residual(cube_dir, seg, sid, lam_vac):
    """一個源的平均殘差、平均 sigma,以及模板蓋不到的通道數。

    A_map 的第 j 列是第 j 條模板成分的係數,超過該源成分數的列是 NaN(step6 用
    固定寬度 N_SRC 存),所以要照模板實際的欄數去切。

    模板的靜止範圍蓋不到的通道,源模型是 **0** 而不是 NaN —— 那才是交付出去的
    產物實際的樣子:step6 在那些通道沒有源模型,源的流量原封不動留在資料裡。
    設成 NaN 會讓那一段從統計裡消失,等於把「模型蓋不到」偽裝成「沒有問題」。
    """
    meta = json.loads((cube_dir / "meta.json").read_text())
    # "classification" is the key; "best" is what step5 wrote before the
    # parameter was renamed, and products made then are still on disk.
    best = np.load(ROOT / (meta.get("classification") or meta["best"]))
    T = build_templates(best, lam_vac).get(int(sid))

    mask = seg == sid
    yy, xx = np.nonzero(mask)
    y0, y1, x0, x1 = yy.min(), yy.max(), xx.min(), xx.max()
    sub_box = mask[y0:y1 + 1, x0:x1 + 1]

    with fits.open(cube_dir / "sky_subtracted.fits", memmap=True) as h:
        sub = np.asarray(h["DATA"].data[:, y0:y1 + 1, x0:x1 + 1], np.float32)[:, sub_box]
        var = np.asarray(h["STAT"].data[:, y0:y1 + 1, x0:x1 + 1], np.float32)[:, sub_box]

    A = np.load(cube_dir / "A_map.npy")[:, y0:y1 + 1, x0:x1 + 1][:, sub_box]
    n_uncovered = len(lam_vac)
    if T is not None:
        n_uncovered = int((~np.all(np.isfinite(T), axis=1)).sum())
        sub = sub - np.nan_to_num(T) @ np.nan_to_num(A[:T.shape[1]])

    with np.errstate(invalid="ignore"):
        r = np.nanmean(sub, axis=1)
        # 平均之後的 sigma:獨立 spaxel 的平均,變異數是各自變異數和除以 N 平方
        n = np.sum(np.isfinite(sub), axis=1)
        s = np.sqrt(np.nansum(var, axis=1)) / np.maximum(n, 1)
    return r, s, n_uncovered, int(mask.sum())


def main():
    ap = argparse.ArgumentParser(description="源區域殘差:wsky 扣掉預測的天空與源")
    ap.add_argument("--work", required=True)
    ap.add_argument("--ids", type=int, nargs="+", required=True)
    ap.add_argument("--run", nargs="+", default=["default:pipeline"],
                    help="要比的 run,寫成 名稱:標籤;default = pipeline 自己的 step05/step06")
    ap.add_argument("--band", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="統計量只算這一段;預設整個 cube")
    ap.add_argument("--smooth", type=int, default=1, help="畫圖用的移動平均寬度(通道)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    W = ROOT / args.work
    out = Path(args.out) if args.out else \
        ROOT / f"results/skymodel/evaluation/{W.name}/star_library"
    out.mkdir(parents=True, exist_ok=True)

    seg = fits.getdata(W / "step01/seg.fits")
    wl  = np.load(W / "step03/wavelength.npy")
    lam_vac = air_to_vacuum(wl)
    band = np.ones(len(wl), bool) if args.band is None else \
        (wl >= args.band[0]) & (wl <= args.band[1])

    runs = []
    for spec in args.run:
        name, _, label = spec.partition(":")
        runs.append((name, label or name, fit_dirs(W, None if name == "default" else name)[1]))

    def smooth(a):
        if args.smooth <= 1:
            return a
        k = np.ones(args.smooth) / args.smooth
        return np.convolve(np.nan_to_num(a), k, mode="same") / \
            np.convolve(np.isfinite(a).astype(float), k, mode="same")

    for sid in args.ids:
        data = []
        for name, label, d in runs:
            r, s, n_unc, npix = source_residual(d, seg, sid, lam_vac)
            data.append((label, r, s, n_unc, npix))
            print(f"  ID {sid}  {label}: {npix} spaxels, "
                  f"source template missing on {n_unc} channels")

        allv = np.concatenate([r[band][np.isfinite(r[band])] for _, r, _, _, _ in data])
        lim = np.percentile(np.abs(allv), 99.5) * 1.6

        fig = plt.figure(figsize=(16, 3.1 * len(data) + 0.7))
        gs = fig.add_gridspec(len(data), 2, width_ratios=[6, 1.15],
                              hspace=0.12, wspace=0.02, top=0.90, bottom=0.09)
        for row, (label, r, s, n_unc, npix) in enumerate(data):
            ax = fig.add_subplot(gs[row, 0])
            ax.axhline(0, color="0.55", lw=0.8)
            ax.plot(wl, smooth(r), lw=0.5, color=COLORS[row % len(COLORS)])
            ax.set_ylim(-lim, lim)
            ax.set_xlim(wl[0], wl[-1])
            ax.set_ylabel("residual")
            ax.set_title(f"{label}"
                         + ("" if n_unc == 0 else
                            f"   -- no source template on {n_unc} channels,"
                            f" the source flux stays in the residual there"),
                         fontsize=11, loc="left")
            if row < len(data) - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("wavelength [A, air]")

            st = spectrum_stats(r[band])
            with np.errstate(invalid="ignore", divide="ignore"):
                chi = r[band] / s[band]
            chi = chi[np.isfinite(chi)]
            n_ok = int(np.isfinite(r[band]).sum())
            # 通道數要一起報:模板蓋不到的地方 T 是 NaN,殘差也就沒有定義,
            # 兩個 run 的統計量若建立在不同數量的通道上就不能直接比。
            lines = [f"{'channels':<14}{n_ok:>10,}",
                     f"{'of':<14}{int(band.sum()):>10,}",
                     f"{'no template':<14}{n_unc:>10,}",
                     "",
                     f"{'mean':<14}{st['mean']:>10.4f}",
                     f"{'sigma':<14}{st['sigma']:>10.4f}",
                     f"{'rms from 0':<14}{st['rms_from_zero']:>10.4f}",
                     f"{'skewness':<14}{st['skewness']:>10.4f}",
                     f"{'kurtosis':<14}{st['kurtosis']:>10.4f}",
                     "",
                     f"{'chi mean':<14}{chi.mean():>10.4f}",
                     f"{'chi sigma':<14}{chi.std():>10.4f}"]
            sa = fig.add_subplot(gs[row, 1]); sa.axis("off")
            sa.text(0, 0.98, "\n".join(lines), va="top", family="monospace",
                    fontsize=9, color=COLORS[row % len(COLORS)],
                    transform=sa.transAxes)

        rng = "whole cube" if args.band is None else f"{args.band[0]:.0f}-{args.band[1]:.0f} A"
        fig.suptitle(f"{W.name}  ID {sid}   {data[0][4]} spaxels   "
                     f"residual = wsky - sky model - source model"
                     f"   (statistics over {rng})", fontsize=13)
        o = out / f"residual_id{sid:03d}.png"
        fig.savefig(o, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved -> {o}")


if __name__ == "__main__":
    main()
