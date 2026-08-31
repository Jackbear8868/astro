"""源邊界的 over-subtraction —— 遮罩外面那一圈被扣掉多少?

若 SExtractor 的足跡抓得比源實際的延展緊,足跡外那一圈仍帶著源的光卻被當成 blank
擬合;那些光被吸進天空模型的話,扣完之後那一圈會比周圍低。殘差指標看不到這件事
(過度扣除讓殘差更平),只能直接去量那一圈。

  01 環形殘差光譜:以足跡為界往外一圈一圈量平均殘差,和遠場比。負 = 被吃掉。
  02 甜甜圈影像:某個波段的殘差沿波長壓成一張圖,疊上足跡輪廓。
  03 徑向剖面:01 依波段收成「殘差 vs 離足跡距離」,多個 run 疊著比,判讀主圖。
  04 同一個版面,畫的是天空模型自己的超出量。

環一定要認標籤:相鄰的源可能只隔幾個像素,普通 dilation 一長就把它們黏成一塊。
這裡用最近標籤(Voronoi)長,每個 blank 像素只歸給最近的源,兩源停在中線。

    conda run -n astro python src/skymodel/experiments/edge_oversubtraction.py \\
        --work results/skymodel/p01
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from products import fit_dirs                             # noqa: E402

ROOT    = Path(__file__).resolve().parents[3]
OUTDIR  = ROOT / "results/skymodel/evaluation/masking/edge_oversub"

# 環的分界(px)。內圈窄、外圈寬:效應集中在貼著足跡的頭幾個像素,外圈只是對照。
RINGS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 20), (20, 40)]

# 報表用的波段。
BANDS = [(4700, 5000), (5000, 6000), (6000, 7000), (7000, 8000), (8000, 9300)]

# 一個極端壞 voxel 就主宰一整格波段的平均。剔的是 voxel,不是整個通道。
WILD = 1e3


def rings_by_label(seg, max_dist):
    """回傳 (dist, owner):每個像素到最近源足跡的距離,以及那個源的 ID。

    distance_transform_edt 算到零元素的距離,餵 seg == 0 就是每個 blank 像素離最近
    的源多遠;return_indices 給出那個源像素的座標,查 seg 便知屬於誰。
    """
    dist, (iy, ix) = ndimage.distance_transform_edt(seg == 0, return_indices=True)
    owner = seg[iy, ix]
    owner[dist > max_dist] = 0          # 太遠的不算任何人的,留給遠場
    owner[seg > 0] = seg[seg > 0]       # 足跡本身歸自己,距離 0
    return dist, owner


def band_mean(wl, spec, lo, hi):
    m = (wl >= lo) & (wl < hi)
    return float(np.nanmean(spec[m]))


def make_groups(seg, white, owner, main_id):
    """把源分組。主源自己一組,其餘依白光總流量的中位數切成亮、暗兩半。

    亮/暗這一刀是對照組:超出量若來自源的光漏出去就必須隨亮度而變,若只是天空的
    空間梯度則兩組一樣。白光獨立於擬合,分組不會用到結果。
    """
    others = np.array([i for i in np.unique(seg[seg > 0]) if i != main_id])
    flux   = np.array([np.nansum(white[seg == i]) for i in others])
    cut    = np.median(flux)
    bright, faint = others[flux >= cut], others[flux < cut]
    return {f"#{main_id}": owner == main_id,
            f"other bright x{bright.size}": np.isin(owner, bright),
            f"other faint x{faint.size}":   np.isin(owner, faint)}


def ring_spectra(D, dist, groups, far):
    """回傳 {區域名: (平均光譜, spaxel 數)}。D 可以是殘差,也可以是天空模型。

    極端壞 voxel 先換成 NaN 再平均。源足跡內部本來就有大訊號,所以門檻 WILD 取在
    遠高於天空殘差的量級。
    """
    def avg(m):
        sub = D[:, m]
        return np.nanmean(np.where(np.abs(sub) > WILD, np.nan, sub), axis=1)

    out = {"far field": (avg(far), int(far.sum()))}
    for gname, gmask in groups.items():
        for d0, d1 in RINGS:
            m = gmask & (dist >= d0) & (dist < d1)
            if m.sum() < 20:
                continue
            tag = "footprint" if d0 == 0 else f"{d0}-{d1}px"
            out[f"{gname} {tag}"] = (avg(m), int(m.sum()))
    return out


def main():
    ap = argparse.ArgumentParser(description="源邊界 over-subtraction 診斷")
    ap.add_argument("--work", required=True,
                    help="pointing 的工作區,例如 results/skymodel/p01")
    ap.add_argument("--run", nargs="+", default=None,
                    help="step05 底下要比的替代 run 目錄名(可給多個,一個一欄);\n預設用 pipeline 自己的 step06")
    ap.add_argument("--seg", default=None,
                    help="segmentation;預設為工作區的 step01/segmentation_input.fits")
    ap.add_argument("--main-id", type=int, default=1, help="單獨拿出來看的源")
    ap.add_argument("--band", type=float, nargs=2, default=(5000, 6000),
                    help="甜甜圈影像用哪個波段")
    ap.add_argument("--far", type=float, default=40.0,
                    help="離所有源這麼遠以外算遠場對照")
    ap.add_argument("--outdir", default=str(OUTDIR))
    args = ap.parse_args()

    W = ROOT / args.work
    STEP01, STEP03 = W / "step01", W / "step03"
    # 名字同時是圖上的欄標題,所以 run -> 目錄的對應要留著,不能只留目錄。
    cubedir = {(r or "step06"): fit_dirs(W, r)[1] for r in (args.run or [None])}
    for r, d in cubedir.items():
        if not (d / "sky_subtracted.fits").exists():
            raise SystemExit(f"★ 找不到 {d / 'sky_subtracted.fits'}")
    args.run = list(cubedir)

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    seg   = fits.getdata(Path(args.seg) if args.seg
                         else STEP01 / "segmentation_input.fits").astype(int)
    white = fits.getdata(STEP01 / "whitelight_nosky.fits")
    wl    = np.load(STEP03 / "wavelength.npy")
    dist, owner = rings_by_label(seg, args.far)
    far    = (dist > args.far) & (seg == 0)
    groups = make_groups(seg, white, owner, args.main_id)
    gnames = list(groups)
    print(f"seg {seg.shape}  {np.unique(seg[seg>0]).size} 個源   "
          f"遠場(> {args.far:g} px) {int(far.sum()):,} spaxel")
    print("分組: " + " / ".join(gnames))

    allspec = {}
    for run in args.run:
        D = fits.getdata(cubedir[run] / "sky_subtracted.fits").astype(np.float32)
        spec = ring_spectra(D, dist, groups, far)
        # 殘差看「留下多少」,天空模型看「扣掉多少」;blank 的 s 固定成 1 時 s·C_sky
        # 處處相同,環上的超出量只能來自係數 cₖ 吸進了源的光。
        S = fits.getdata(cubedir[run] / "sky_model.fits").astype(np.float32)
        sspec = ring_spectra(S, dist, groups, far)
        del S
        print(f"\n=== {run}")
        print(f"{'region':<24}{'spaxel':>8}"
              + "".join(f"{f'{a}-{b}':>11}" for a, b in BANDS))
        ffar = spec["far field"][0]
        for k, (s, n) in spec.items():
            print(f"{k:<24}{n:>8,}"
                  + "".join(f"{band_mean(wl, s, a, b):>11.4f}" for a, b in BANDS))
        # 環上多出來的光只有兩個去處:留在殘差裡,或進了天空模型;兩者相加不該隨
        # 擬合方式改變,所以這一欄也是自我檢查。
        sfar = sspec["far field"][0]
        print("\n環上多出來的光去了哪裡(5000-6000 A,皆已減掉遠場)")
        print(f"{'group':<20}{'ring':>8}{'留在殘差':>10}{'進天空模型':>12}"
              f"{'合計':>9}{'被吸走':>9}")
        for g in gnames:
            for d0, d1 in RINGS[1:]:
                k = f"{g} {d0}-{d1}px"
                if k not in spec:
                    continue
                r = band_mean(wl, spec[k][0] - ffar, 5000, 6000)
                s = band_mean(wl, sspec[k][0] - sfar, 5000, 6000)
                print(f"{g:<20}{f'{d0}-{d1}px':>8}{r:>10.4f}{s:>12.4f}"
                      f"{r + s:>9.4f}{100 * s / (r + s):>8.1f}%")
        allspec[run] = (spec, sspec)

        # ---- 01 環形殘差光譜 --------------------------------------------
        nb = 40                              # 每 40 通道(50 A)平一次,不然看不出趨勢
        cut = len(wl) // nb * nb
        wb  = wl[:cut].reshape(-1, nb).mean(axis=1)
        def binned(s):
            return np.nanmean(s[:cut].reshape(-1, nb), axis=1)

        fig, ax = plt.subplots(len(gnames), 1, figsize=(14, 4.4 * len(gnames)),
                               sharex=True)
        for a, g in zip(np.atleast_1d(ax), gnames):
            keys = [k for k in spec if k.startswith(g) and "footprint" not in k]
            for i, k in enumerate(keys):
                a.plot(wb, binned(spec[k][0]), lw=1.0,
                       label=f"{k.replace(g + ' ', '')}  ({spec[k][1]:,})",
                       color=plt.cm.plasma(i / max(len(keys) - 1, 1)))
            a.plot(wb, binned(spec["far field"][0]), lw=1.5, color="k", ls="--",
                   label=f"far field > {args.far:g} px  ({spec['far field'][1]:,})")
            a.axhline(0, color="0.6", lw=0.8)
            a.axvspan(*args.band, color="0.85", zorder=0)
            a.set_ylabel("mean residual")
            a.set_title(f"{g}   (footprint itself excluded)", fontsize=10)
            a.legend(fontsize=7, ncol=2)
            a.grid(alpha=0.25)
        np.atleast_1d(ax)[-1].set_xlabel("wavelength [$\\AA$]")
        fig.suptitle(f"residual just outside the source footprint — {run}", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out / f"01_rings_{W.name}_{run}.png", dpi=135, bbox_inches="tight")
        plt.close(fig)

        # ---- 02 甜甜圈影像 ----------------------------------------------
        bm  = (wl >= args.band[0]) & (wl < args.band[1])
        img = np.nanmedian(D[bm], axis=0)      # 中位數:躲開殘留天光線與壞 voxel 的尖峰
        fig, ax = plt.subplots(1, 2, figsize=(15, 6.5))
        v = float(np.nanpercentile(np.abs(img[far]), 99)) * 2.0
        for a, (title, box) in zip(ax, [("full field", None), (f"#{args.main_id}", args.main_id)]):
            im = a.imshow(img, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
            a.contour(seg > 0, levels=[0.5], colors="k", linewidths=0.5)
            if box is not None:
                y, x = np.nonzero(seg == box)
                a.set_xlim(x.min() - 35, x.max() + 35)
                a.set_ylim(y.min() - 35, y.max() + 35)
            a.set_title(f"{title}   median residual "
                        f"{args.band[0]:.0f}-{args.band[1]:.0f} A", fontsize=10)
            plt.colorbar(im, ax=a, fraction=0.046, pad=0.02)
        fig.suptitle(run, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out / f"02_donut_{W.name}_{run}.png", dpi=135, bbox_inches="tight")
        plt.close(fig)
        del D

    # ---- 03 徑向剖面(列 = 分組,行 = run,同一列共用 y 軸才比得動) ---------
    # x 用等距序號而不是真實距離:環寬本來就不等,真實距離會把內圈幾個點擠成一團,
    # log 軸則會冒出一堆小刻度蓋掉環的標籤。
    xr = np.arange(len(RINGS) - 1)
    fig, ax = plt.subplots(len(gnames), len(args.run), squeeze=False,
                           figsize=(5.2 * len(args.run), 4.6 * len(gnames)))
    for row, g in enumerate(gnames):
        lo_y, hi_y = np.inf, -np.inf
        for col, run in enumerate(args.run):
            a, (spec, _) = ax[row][col], allspec[run]
            for i, (lo, hi) in enumerate(BANDS):
                y = [band_mean(wl, spec[f"{g} {d0}-{d1}px"][0], lo, hi)
                     if f"{g} {d0}-{d1}px" in spec else np.nan for d0, d1 in RINGS[1:]]
                a.plot(xr, y, marker="o", ms=3.5, lw=1.3,
                       color=plt.cm.viridis(i / (len(BANDS) - 1)), label=f"{lo}-{hi} A")
                lo_y, hi_y = min(lo_y, np.nanmin(y)), max(hi_y, np.nanmax(y))
            a.axhline(band_mean(wl, spec["far field"][0], 4700, 9300),
                      color="k", ls="--", lw=1.0, label="far field (all wl)")
            a.axhline(0, color="0.6", lw=0.8)
            a.set_xticks(xr)
            a.set_xticklabels([f"{d0}-{d1}" for d0, d1 in RINGS[1:]], fontsize=8)
            a.set_xlabel("distance outside footprint [px]")
            a.set_title(f"{g}\n{run}", fontsize=9)
            a.grid(alpha=0.25)
            if col == 0:
                a.set_ylabel("mean residual")
                a.legend(fontsize=7)
        pad = 0.08 * (hi_y - lo_y)
        for col in range(len(args.run)):
            ax[row][col].set_ylim(lo_y - pad, hi_y + pad)
    fig.suptitle("radial profile of the residual outside the source footprint "
                 "(same y-scale within each row)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / f"03_radial_profile_{W.name}.png", dpi=135, bbox_inches="tight")
    plt.close(fig)

    # ---- 04 天空模型的超出量 ----------------------------------------------
    # 和 03 同版面,但畫天空模型減掉遠場的天空模型。真正的天空不知道附近有沒有源,
    # 任何隨「離足跡多遠」變化的趨勢都是污染。
    fig, ax = plt.subplots(len(gnames), len(args.run), squeeze=False,
                           figsize=(5.2 * len(args.run), 4.6 * len(gnames)))
    for row, g in enumerate(gnames):
        lo_y, hi_y = np.inf, -np.inf
        for col, run in enumerate(args.run):
            a, (_, sspec) = ax[row][col], allspec[run]
            sfar = sspec["far field"][0]
            for i, (lo, hi) in enumerate(BANDS):
                y = [band_mean(wl, sspec[f"{g} {d0}-{d1}px"][0] - sfar, lo, hi)
                     if f"{g} {d0}-{d1}px" in sspec else np.nan for d0, d1 in RINGS[1:]]
                a.plot(xr, y, marker="o", ms=3.5, lw=1.3,
                       color=plt.cm.viridis(i / (len(BANDS) - 1)), label=f"{lo}-{hi} A")
                lo_y, hi_y = min(lo_y, np.nanmin(y)), max(hi_y, np.nanmax(y))
            a.axhline(0, color="0.6", lw=0.8)
            a.set_xticks(xr)
            a.set_xticklabels([f"{d0}-{d1}" for d0, d1 in RINGS[1:]], fontsize=8)
            a.set_xlabel("distance outside footprint [px]")
            a.set_title(f"{g}\n{run}", fontsize=9)
            a.grid(alpha=0.25)
            if col == 0:
                a.set_ylabel("sky model  (ring - far field)")
                a.legend(fontsize=7)
        pad = 0.08 * (hi_y - lo_y)
        for col in range(len(args.run)):
            ax[row][col].set_ylim(lo_y - pad, hi_y + pad)
    fig.suptitle("how much the SKY MODEL itself is raised near sources "
                 "(same y-scale within each row)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out / f"04_skymodel_excess_{W.name}.png", dpi=135, bbox_inches="tight")
    plt.close(fig)

    # ---- 膨脹餘裕 ---------------------------------------------------------
    print("\n膨脹餘裕(認標籤 vs 不認標籤)")
    print(f"{'r px':>5}{'新增 spaxel':>13}{'佔 blank':>10}"
          f"{'普通 dilation 黏住的源':>24}")
    blank0 = int((seg == 0).sum())
    conn   = ndimage.generate_binary_structure(2, 2)
    for r in (1, 2, 3, 4, 6, 8):
        add   = int(((dist > 0) & (dist <= r)).sum())
        grown = ndimage.binary_dilation(seg > 0, conn, iterations=r)
        lab, n = ndimage.label(grown, conn)
        merged = sum(max(np.unique(seg[(lab == L) & (seg > 0)]).size - 1, 0)
                     for L in range(1, n + 1))
        print(f"{r:>5}{add:>13,}{100 * add / blank0:>9.1f}%{merged:>24}")

    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
