"""37 個源的總表 + 一張標了編號的位置圖。

step4 的 best_*.npz 已經有擬合結果,但少了兩件事:源在天上的樣子(多大、在哪、
多亮),以及第 1 段自己的 z —— 那被第 2 段覆寫掉了。要判斷一個解可不可信,
這些必須擺在一起看:一個只有 12 個 spaxel 的暗源,擬合再漂亮也只是雜訊。

產出(和逐源的圖放在同一個資料夾 figures/step4_{tag}/)
    source_id_map.png              哪一個點是哪一個源。底圖是白光影像,
                                   每個源依組別上色並標上編號。
    catalog.txt                    完整總表(對齊的純文字)。
    catalog.csv                    同一份表,給試算表用。

欄位的意思
    nspax / 面積     源遮罩的 spaxel 數。0.2"/px,所以 1 spaxel = 0.04 arcsec²。
    r_eq            等效半徑 √(nspax/π),拿來和 PSF(FWHM ≈ 4 px)比大小。
    edge            遮罩最靠近視場邊界的距離 (px)。視場邊緣的曝光覆蓋掉下來、
                    雜訊抬高,SExtractor 會在那裡偵測到假源 —— 小的 edge 值
                    是「這可能不是真的源」的警訊。
    白光淨          遮罩內白光中位數 − blank 區白光中位數,單位同白光影像。
                    不用 object_flux 算面亮度:那是從含天空的 cube 抽的,
                    中位數量到的是天空(全部落在 29–32),不是源。
    z1 / z2         第 1 段(離散模板)與第 2 段(本徵譜)各自定出的紅移。
                    兩者差很多 = 換一個模型就換一個答案,這個 z 不穩。
    dchi2           最佳組與次佳組的 chi2_all 差。小 = 分類幾乎是擲骰子。
    src_min         重建的源光譜最小值。負 = 模型宣稱源發出負的光,非物理。
    ok%             整個 z 掃描範圍內,src_min ≥ 0 的比例。
                    0% = 這個模型在任何紅移下都無法描述這個源而不出現負流量。

    conda run -n astro python src/skymodel/experiments/step4_catalog.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import distance_transform_edt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP02  = ROOT / "results/skymodel/step02"
STEP04  = ROOT / "results/skymodel/step04"
FIGURES = ROOT / "results/skymodel/figures"

GROUP_COLOR = {"star": "#2ca02c", "galaxy": "#1f77b4", "qso": "#d62728"}
PLAIN_COLOR = "#ff7b7b"     # 不分組時的統一顏色。淡紅在灰階底圖上夠明顯,
                            # 而且明確不屬於灰階,不會被誤讀成影像本身。
PIXSCALE = 0.20                       # arcsec/px,來自 header CD1_1


def id_map(seg, white, rows, out, by_group=False):
    """白光底圖 + 源的範圍 + 編號。

    底圖用 asinh 拉伸:白光的動態範圍跨好幾個量級(Haro 11 本體 vs 暗源),
    線性顯示會讓除了本體以外的東西全黑。asinh 在亮處是對數、暗處是線性,
    是影像顯示的標準做法。

    by_group=False(預設)只畫「範圍 + 編號」,不上組別顏色。分類是 step4 的
    「結論」,把結論畫進定位圖裡,看圖的人會不自覺地把它當成既定事實 ——
    而我們已經量到 15/37 個源的分類裕度小於 100。定位圖的職責只是回答
    「哪個點是哪個源」。
    """
    fig, ax = plt.subplots(figsize=(13, 12.5))
    v = np.nanpercentile(white[np.isfinite(white) & (white != 0)], 99.5)
    ax.imshow(np.arcsinh(white / (0.02 * v)), origin="lower", cmap="gray",
              vmin=0, vmax=np.arcsinh(1 / 0.02))

    # 每個源填一層半透明的顏色,再描一圈輪廓 —— 填色看得出範圍,
    # 輪廓在小源上仍然看得見。
    for r in rows:
        m = seg == r["id"]
        c = GROUP_COLOR[r["group"]] if by_group else PLAIN_COLOR
        rgba = np.zeros(seg.shape + (4,))
        rgba[m] = list(matplotlib.colors.to_rgb(c)) + [0.45]
        ax.imshow(rgba, origin="lower")
        ax.contour(m, levels=[0.5], colors=c, linewidths=0.9)
        ax.text(r["x"], r["y"], str(r["id"]), color="white", fontsize=11,
                fontweight="bold", ha="center", va="center",
                path_effects=[pe.withStroke(linewidth=2.6, foreground="black")])

    ax.set_title("source ID map" + ("\nwhitelight (asinh) + SExtractor "
                 "segmentation, colour = the group step4 assigned"
                 if by_group else ""), fontsize=14)
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")
    if by_group:
        # 圖例放在座標軸外面 —— 右上角有 ID 32/36/11 三個源,壓在上面看不見。
        ax.legend(handles=[plt.Line2D([], [], color=c, lw=6, label=g)
                           for g, c in GROUP_COLOR.items()],
                  loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3,
                  frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="37 個源的總表與位置圖")
    ap.add_argument("--tag", default="svd_K30_s_1.0")
    args = ap.parse_args()

    best  = np.load(STEP04 / f"best_{args.tag}.npz")
    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits").astype(np.float64)
    ids   = np.load(STEP02 / "object_ids.npy")
    nspx  = np.load(STEP02 / "object_nspax.npy")

    # 視場外是 0。distance_transform_edt 給每個 FOV 內的格點到最近的非 FOV
    # 格點的距離,取遮罩內的最小值 = 這個源最靠近邊界到什麼程度。
    fov  = white != 0
    dist = distance_transform_edt(fov)
    sky_level = float(np.median(white[fov & (seg == 0)]))

    yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
    rows = []
    for j, t in enumerate(best["id"]):
        t = int(t)
        m = seg == t
        k = int(np.flatnonzero(ids == t)[0])
        n = int(np.median(nspx[k]))
        wnet = float(np.median(white[m]) - sky_level)
        edge = float(dist[m].min())

        # 第 1 段自己的 z:best 存的是第 2 段覆寫後的結果,要回頭讀掃描檔。
        z1, tpl1 = np.nan, "-"
        f1 = STEP04 / f"scan_id{t}_{args.tag}.npz"
        if f1.exists():
            s1 = np.load(f1)
            grp1 = s1["group"] == str(best["group"][j])
            if grp1.any():
                i1 = int(np.flatnonzero(grp1)[np.argmin(s1["chi2_all"][grp1])])
                z1, tpl1 = float(s1["z"][i1]), str(s1["template"][i1])

        # ok%:整個掃描範圍裡有多少比例的 z 給出非負的源光譜。
        # 星系/QSO 看第 2 段(scan2),恆星沒有第 2 段,退回第 1 段同組的候選。
        f2 = STEP04 / f"scan2_id{t}_{args.tag}.npz"
        if f2.exists():
            sm = np.load(f2)["src_min"]
        elif f1.exists():
            s1 = np.load(f1)
            sm = s1["src_min"][s1["group"] == str(best["group"][j])]
        else:
            sm = np.array([np.nan])
        okpc = 100.0 * float(np.mean(sm >= 0))

        rows.append(dict(
            id=t, nspax=n, area=n * PIXSCALE ** 2, r_eq=np.sqrt(n / np.pi),
            y=float(yy[m].mean()), x=float(xx[m].mean()), wnet=wnet, edge=edge,
            group=str(best["group"][j]), tpl1=tpl1, tpl2=str(best["template"][j]),
            z1=z1, z2=float(best["z"][j]), s=float(best["s"][j]),
            A=best["A"][j], chi2_all=float(best["chi2_all"][j]),
            red_chi2=float(best["red_chi2"][j]), n_good=int(best["n_good"][j]),
            dchi2=float(best["dchi2"][j]), src_min=float(best["src_min"][j]),
            okpc=okpc))

    rows.sort(key=lambda r: -r["nspax"])

    hdr = (f"{'ID':>4}{'nspax':>7}{'面積':>8}{'r_eq':>6}{'y':>6}{'x':>6}{'edge':>6}"
           f"{'白光淨':>8}{'group':>8}{'tpl1':>6}{'z1':>8}{'z2':>8}{'|Δz|':>8}"
           f"{'A1':>10}{'χ²/dof':>9}{'dchi2':>13}{'src_min':>9}{'ok%':>7}")
    lines = [f"step4 源總表   tag = {args.tag}   依 nspax 排序", "",
             f"{'':>19}[arcsec²]{'[px]':>5}{'':>12}[px]", hdr, "-" * len(hdr)]
    for r in rows:
        dz = abs(r["z2"] - r["z1"])
        lines.append(
            f"{r['id']:>4}{r['nspax']:>7}{r['area']:>8.2f}{r['r_eq']:>6.1f}"
            f"{r['y']:>6.0f}{r['x']:>6.0f}{r['edge']:>6.1f}{r['wnet']:>8.2f}"
            f"{r['group']:>8}{r['tpl1']:>6}{r['z1']:>8.4f}{r['z2']:>8.4f}{dz:>8.4f}"
            f"{r['A'][0]:>10.4g}{r['red_chi2']:>9.2f}{r['dchi2']:>13,.0f}"
            f"{r['src_min']:>9.2f}{r['okpc']:>6.1f}%")

    lines += ["", "面積 = nspax × 0.04 arcsec² ;  r_eq = √(nspax/π) px,PSF FWHM ≈ 4 px",
              "edge = 遮罩最靠近視場邊界的距離;小 = 可能是邊緣的假偵測",
              "白光淨 = 遮罩內白光中位數 − blank 區白光中位數",
              "z1   = 第 1 段(離散模板)的 z ;  z2 = 第 2 段(本徵譜)的 z,也是報出的值",
              "dchi2= 次佳組 − 最佳組的 chi2_all(分類裕度);  src_min < 0 = 解不合物理",
              "ok%  = 整個 z 掃描範圍內 src_min ≥ 0 的比例"]

    out = FIGURES / f"step4_{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    txt = out / "catalog.txt"
    txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    csv = out / "catalog.csv"
    with csv.open("w") as fh:
        fh.write("id,nspax,area_arcsec2,r_eq_px,y,x,edge_px,white_net,group,"
                 "tpl1,tpl2,z1,z2,dz,"
                 "A1,A2,A3,A4,s,chi2_all,red_chi2,n_good,dchi2,src_min,ok_pct\n")
        for r in rows:
            a = ",".join("" if not np.isfinite(v) else f"{v:.6g}" for v in r["A"])
            fh.write(f"{r['id']},{r['nspax']},{r['area']:.3f},{r['r_eq']:.2f},"
                     f"{r['y']:.1f},{r['x']:.1f},{r['edge']:.1f},"
                     f"{r['wnet']:.4g},{r['group']},"
                     f"{r['tpl1']},{r['tpl2']},{r['z1']:.6f},{r['z2']:.6f},"
                     f"{abs(r['z2']-r['z1']):.6f},{a},{r['s']:.4f},"
                     f"{r['chi2_all']:.6g},{r['red_chi2']:.4f},{r['n_good']},"
                     f"{r['dchi2']:.6g},{r['src_min']:.4f},{r['okpc']:.1f}\n")

    png  = out / "source_id_map.png"          # 依組別上色
    png2 = out / "source_id_map_plain.png"    # 只有範圍與編號
    id_map(seg, white, rows, png,  by_group=True)
    id_map(seg, white, rows, png2, by_group=False)
    print(f"\nsaved -> {txt}\n         {csv}\n         {png}\n         {png2}")


if __name__ == "__main__":
    main()
