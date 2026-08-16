"""把 step4 的擬合過程攤開來看:每個源的候選排名、chi2 地形、光譜對照。

step4 只留下贏家一列,中間的過程全部丟掉了。這支程式讀 `--save-scan` 存下的
完整掃描結果,重建前幾名的模型光譜,產出:

    landscape.png          chi2_all vs z,一條曲線一個模板,依組別上色。
                           整個擬合地形一眼看完:哪些模板有明確的谷、
                           哪些是平的、贏家贏多少。
    top10_summary.png      整體前 10 名的「源模型」疊在一起 + 數值表。
                           源模型是平滑的,疊起來看得清楚。
    top10/rank_NN.png      前 10 名各一張:觀測光譜、完整模型、源的部分,
                           5 段波長,版面和其他圖一致。
    per_template/tpl_X.png 每個模板一張:它自己的 chi2(z) 曲線(標出前 10 個 z)
                           + 它最佳解的光譜。
    ranking.txt            前 10 名與逐模板最佳解的數值表。

再接著呼叫 step4_stages,補上兩段各自的拆解與對照:

    stage1/                第 1 段(33 條離散模板 × 1 個非負振幅)的地形與組別比較。
    stage2/                第 2 段(4 條本徵譜 × 可正可負係數)的地形與最佳光譜。
    compare/               兩段疊在一起:地形、光譜、殘差、自由度。

模型是重算的,不是存下來的 —— 給定 (組別, 模板, z) 重解一次很便宜,
存下所有候選的光譜則要天文數字的空間。

    conda run -n astro python src/skymodel/experiments/step4_diagnostics.py --id 14
    conda run -n astro python src/skymodel/experiments/step4_diagnostics.py --id all
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step4_stages as stages
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from templates import (load_sdss_template, load_eigen_galaxy, load_eigen_qso,
                       redshift_to_grid, air_to_vacuum)

ROOT      = Path(__file__).resolve().parents[3]
STEP02    = ROOT / "results/skymodel/step02"
STEP03    = ROOT / "results/skymodel/step03"
STEP04    = ROOT / "results/skymodel/step04"
FIGURES   = ROOT / "results/skymodel/figures"
TPL_DIR   = ROOT / "data/sdss_templates"
EIGEN_GAL = ROOT / "data/eigen_galaxy_Bolton2012.fits"
EIGEN_QSO = ROOT / "data/qso_eigen_linear_55732.dat"

GROUP_COLOR = {"star": "#2ca02c", "galaxy": "#1f77b4", "qso": "#d62728"}
SDSS_GROUPS = {"star": range(0, 23), "galaxy": range(23, 29), "qso": range(29, 33)}
GROUP_OF = {f"{i:03d}": g for g, r in SDSS_GROUPS.items() for i in r}


def _parser():
    ap = argparse.ArgumentParser(description="step4 的完整診斷:候選排名 + 兩段拆解")
    ap.add_argument("--id", default="all", help="源編號,或 all 跑全部")
    ap.add_argument("--tag", default="svd_K30_s_1.0")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--s-fix", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--dedup-dz", type=float, default=0.01,
                    help="前 N 名去重:同一條模板且 z 差距小於這個值的視為同一個解。"
                         "z 網格是 1e-4,不去重的話前 10 名會全是同一條模板的相鄰 z。"
                         "設 0 則不去重。")
    ap.add_argument("--panels", type=int, default=5)
    ap.add_argument("--width", type=float, default=22.0)
    ap.add_argument("--height", type=float, default=3.2)
    ap.add_argument("--no-stages", action="store_true",
                    help="只畫本檔的圖,不接著跑 stage1/stage2/compare")
    return ap


def one_source(args):
    scan_f = STEP04 / f"scan_id{args.id}_{args.tag}.npz"
    if not scan_f.exists():
        raise SystemExit(f"找不到 {scan_f.name}。step4 要加 --save-scan 跑過:\n"
                         f"  conda run -n astro python src/skymodel/step4_find_template.py "
                         f"--id {args.id} --basis {args.basis} -K {args.K}")
    sc = np.load(scan_f)
    has_group = "group" in sc
    grp = (sc["group"] if has_group
           else np.array([GROUP_OF.get(t, "?") for t in sc["template"]]))

    ids   = np.load(STEP02 / "object_ids.npy")
    k     = int(np.flatnonzero(ids == args.id)[0])
    nspax = np.load(STEP02 / "object_nspax.npy")[k]
    with np.errstate(invalid="ignore", divide="ignore"):
        flux = np.load(STEP02 / "object_flux.npy")[k] / nspax
        var  = np.load(STEP02 / "object_var.npy")[k] / nspax ** 2

    wl_air = np.load(STEP03 / "wavelength.npy")
    wl     = air_to_vacuum(wl_air)
    C_sky  = np.load(STEP03 / "sky_continuum.npy")
    B      = np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")
    sky    = np.vstack([C_sky, B])

    base = (np.isfinite(flux) & np.isfinite(var) & (var > 0)
            & np.all(np.isfinite(sky), axis=0))
    sig  = np.sqrt(np.where(var > 0, var, 1.0))
    s_fix = args.s_fix
    sky_free, y = sky[1:], flux - s_fix * sky[0]
    skyw = np.ascontiguousarray((sky_free / sig).T)

    eig = {"galaxy": load_eigen_galaxy(EIGEN_GAL), "qso": load_eigen_qso(EIGEN_QSO)}
    spl_cache = {}

    def spline_of(name, group):
        if name == "eigen":
            return eig[group]
        if name not in spl_cache:
            spl_cache[name] = load_sdss_template(TPL_DIR / f"spDR2-{name}.fit")
        return spl_cache[name]

    def solve(group, name, z):
        """重解一個候選,回傳 (源的部分, 天空的部分, chi2_all, n_good)。"""
        T = redshift_to_grid(spline_of(name, group), float(z), wl)
        if T.ndim == 1:
            T = T[:, None]
        nc = T.shape[1]
        good = base & np.all(np.isfinite(T), axis=1)
        p = sky_free.shape[0] + nc
        if good.sum() <= p + 1:
            return None
        M = np.empty((int(good.sum()), p))
        M[:, :nc] = T[good] / sig[good][:, None]
        M[:, nc:] = skyw[good]
        lb = np.full(p, -np.inf)
        if nc == 1:
            lb[0] = 0.0
        th = lsq_linear(M, (y / sig)[good], bounds=(lb, np.full(p, np.inf)),
                        method="bvls").x
        src = np.nan_to_num(T, nan=0.0) @ th[:nc]
        skym = th[nc:] @ sky_free + s_fix * sky[0]
        c2 = float((((y - (src + th[nc:] @ sky_free)) / sig) ** 2)[base].sum())
        return src, skym, c2, int(good.sum()), th[:nc]

    out = FIGURES / f"step4_{args.tag}" / f"id{args.id:02d}"
    (out / "top10").mkdir(parents=True, exist_ok=True)
    (out / "per_template").mkdir(parents=True, exist_ok=True)
    edges = np.linspace(wl_air[0], wl_air[-1], args.panels + 1)

    # ---------------- landscape:chi2_all vs z,一條線一個模板 ----------------
    names = np.unique(sc["template"])
    fig, ax = plt.subplots(figsize=(14, 7))
    for nm in names:
        m = sc["template"] == nm
        o = np.argsort(sc["z"][m])
        g = grp[m][0]
        ax.plot(sc["z"][m][o], sc["chi2_all"][m][o], lw=0.6, alpha=0.55,
                color=GROUP_COLOR.get(str(g), "0.5"))
    for g, c in GROUP_COLOR.items():
        ax.plot([], [], color=c, lw=1.5, label=g)
    b = int(np.argmin(sc["chi2_all"]))
    ax.plot(sc["z"][b], sc["chi2_all"][b], "k*", ms=16, zorder=5,
            label=f"best: {grp[b]} {sc['template'][b]} z={sc['z'][b]:.4f}")
    ax.set_yscale("log"); ax.set_xlabel("redshift z"); ax.set_ylabel("chi2_all")
    ax.set_title(f"ID {args.id}   chi2 landscape   "
                 f"{len(names)} templates x {sc['z'][sc['template']==names[0]].size} z"
                 f"   ({args.tag})", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "landscape.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---------------- 整體前 N 名 ----------------
    # 去重:貪婪地依 chi2 取,跳過「同模板且 z 太近」的重複解
    srt = np.argsort(sc["chi2_all"])
    order, taken = [], []
    for i in srt:
        nm, z = str(sc["template"][i]), float(sc["z"][i])
        if any(nm == n0 and abs(z - z0) < args.dedup_dz for n0, z0 in taken):
            continue
        taken.append((nm, z)); order.append(i)
        if len(order) >= args.top:
            break
    order = np.array(order)
    lines = [f"ID {args.id}   tag {args.tag}   nspax {int(nspax.max())}",
             "", f"整體前 {args.top} 名(第 1 段,33 條離散模板;"
                 f"同模板且 |dz| < {args.dedup_dz} 視為同一個解)",
             f"{'#':>3}{'group':>8}{'tpl':>6}{'z':>10}{'A':>12}{'chi2_all':>16}"
             f"{'n_good':>8}{'vs 第1名':>12}"]
    best_c2 = sc["chi2_all"][order[0]]
    tops = []
    for r, i in enumerate(order, 1):
        g, nm, z = str(grp[i]), str(sc["template"][i]), float(sc["z"][i])
        res = solve(g, nm, z)
        if res is None:
            continue
        src, skym, c2, ng, coef = res
        tops.append((r, g, nm, z, src, skym, c2, coef))
        lines.append(f"{r:>3}{g:>8}{nm:>6}{z:>10.4f}{coef[0]:>12.4g}"
                     f"{sc['chi2_all'][i]:>16,.0f}{sc['n_good'][i]:>8}"
                     f"{sc['chi2_all'][i]-best_c2:>12,.0f}")

    # 每個模板的最佳解
    lines += ["", "逐模板的最佳解(依 chi2_all 排序)",
              f"{'#':>3}{'group':>8}{'tpl':>6}{'z':>10}{'chi2_all':>16}{'vs 全域最佳':>14}"]
    per = []
    for nm in names:
        m = sc["template"] == nm
        j = np.flatnonzero(m)[int(np.argmin(sc["chi2_all"][m]))]
        per.append((str(grp[j]), str(nm), float(sc["z"][j]), float(sc["chi2_all"][j])))
    per.sort(key=lambda t: t[3])
    for r, (g, nm, z, c2) in enumerate(per, 1):
        lines.append(f"{r:>3}{g:>8}{nm:>6}{z:>10.4f}{c2:>16,.0f}{c2-best_c2:>14,.0f}")
    (out / "ranking.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:16]))

    # top10_summary:源模型疊圖
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(wl_air, flux - C_sky, lw=0.5, color="0.75", zorder=1,
            label="observed - C_sky (for scale)")
    for r, g, nm, z, src, skym, c2, coef in tops:
        ax.plot(wl_air, src, lw=1.0, alpha=0.9,
                label=f"#{r} {g} {nm} z={z:.4f}  chi2 {c2:,.0f}")
    ax.set_xlabel("wavelength [$\\AA$]"); ax.set_ylabel("flux  [$10^{-20}$ cgs]")
    ax.set_title(f"ID {args.id}   source model of the top {len(tops)} candidates",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "top10_summary.png", dpi=140,
                                    bbox_inches="tight")
    plt.close(fig)

    # 每一名一張:觀測 / 完整模型 / 源
    def spectra_fig(path, title, src, skym):
        """每個波段兩列:上看擬合,下看源與殘差。

        不能共用一個 y 軸 —— 天光線比源亮好幾個數量級,擠在同一個軸上源會被
        壓成貼著 0 的直線。
        """
        full = src + skym
        res = flux - full
        fig, axes = plt.subplots(2 * args.panels, 1,
                                 figsize=(args.width, args.height * args.panels * 1.5
                                          + 0.6),
                                 gridspec_kw={"height_ratios": [1.3, 1] * args.panels})
        for bi in range(args.panels):
            m = (wl_air >= edges[bi]) & (wl_air <= edges[bi + 1])
            a1, a2 = axes[2 * bi], axes[2 * bi + 1]

            a1.plot(wl_air[m], flux[m], lw=0.8, color="0.35", label="observed")
            a1.plot(wl_air[m], full[m], lw=0.8, color="#d62728", alpha=0.85,
                    label="full model (source + sky)")
            a1.set_xlim(edges[bi], edges[bi + 1])
            a1.set_ylabel("flux", fontsize=9)
            a1.tick_params(labelsize=9, labelbottom=False)

            a2.axhline(0, color="0.55", lw=0.6)
            a2.plot(wl_air[m], res[m], lw=0.7, color="0.55", alpha=0.8,
                    label="residual (observed - model)")
            a2.plot(wl_air[m], src[m], lw=1.3, color="#0d47a1", label="source only")
            a2.set_xlim(edges[bi], edges[bi + 1])
            # 源與殘差各自的尺度,不受天光線影響
            hi = max(np.nanpercentile(np.abs(res[m]), 99),
                     np.nanmax(np.abs(src[m])) if np.isfinite(src[m]).any() else 0, 1)
            a2.set_ylim(-1.2 * hi, 1.2 * hi)
            a2.set_ylabel("source / resid", fontsize=9)
            a2.tick_params(labelsize=9)
            if bi == 0:
                a1.legend(fontsize=9, loc="upper right", framealpha=0.9)
                a2.legend(fontsize=9, loc="upper right", framealpha=0.9)
                a1.set_title(title, fontsize=11)
        axes[-1].set_xlabel("wavelength [$\\AA$]", fontsize=10)
        fig.tight_layout(h_pad=0.25)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    for r, g, nm, z, src, skym, c2, coef in tops:
        ax.plot(wl_air, src, lw=1.0, alpha=0.9,
                label=f"#{r} {g} {nm} z={z:.4f}  chi2 {c2:,.0f}")
    ax.set_xlabel("wavelength [$\\AA$]"); ax.set_ylabel("flux  [$10^{-20}$ cgs]")
    ax.set_title(f"ID {args.id}   source model of the top {len(tops)} candidates",
                 fontsize=11)
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / "top10_summary.png", dpi=140,
                                    bbox_inches="tight")
    plt.close(fig)

    # 每一名一張:觀測 / 完整模型 / 源
    def spectra_fig(path, title, src, skym):
        fig, axes = plt.subplots(args.panels, 1,
                                 figsize=(args.width, args.height * args.panels + 0.6))
        axes = np.atleast_1d(axes)
        for bi, a in enumerate(axes):
            m = (wl_air >= edges[bi]) & (wl_air <= edges[bi + 1])
            a.plot(wl_air[m], flux[m], lw=0.7, color="0.35", label="observed")
            a.plot(wl_air[m], (src + skym)[m], lw=0.7, color="#d62728", alpha=0.85,
                   label="full model (source + sky)")
            a.plot(wl_air[m], src[m], lw=1.1, color="#0d47a1", label="source only")
            a.set_xlim(edges[bi], edges[bi + 1])
            a.set_ylabel("flux  [$10^{-20}$ cgs]", fontsize=9)
            a.tick_params(labelsize=9)
            if bi == 0:
                a.legend(fontsize=9, loc="upper right", framealpha=0.9)
                a.set_title(title, fontsize=11)
        axes[-1].set_xlabel("wavelength [$\\AA$]", fontsize=10)
        fig.tight_layout(h_pad=0.4)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    for r, g, nm, z, src, skym, c2, coef in tops:
        spectra_fig(out / "top10" / f"rank_{r:02d}.png",
                    f"ID {args.id}   rank #{r}   {g} {nm}   z={z:.4f}   "
                    f"A={coef[0]:.4g}   chi2_all={c2:,.0f}", src, skym)

    # ---------------- 每個模板一張 ----------------
    for nm in names:
        m = sc["template"] == nm
        z_, c_ = sc["z"][m], sc["chi2_all"][m]
        o = np.argsort(z_)
        g = str(grp[m][0])
        j = int(np.argmin(c_))
        res = solve(g, str(nm), float(z_[j]))
        if res is None:
            continue
        src, skym, c2, ng, coef = res
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(14, 9),
                                     gridspec_kw={"height_ratios": [1, 1.3]})
        a1.plot(z_[o], c_[o], lw=0.8, color=GROUP_COLOR.get(g, "0.4"))
        t10 = np.argsort(c_)[:10]
        a1.plot(z_[t10], c_[t10], "o", ms=5, color="k", label="top 10 z")
        a1.axhline(best_c2, color="0.5", ls=":", label="global best chi2")
        a1.set_yscale("log"); a1.set_xlabel("redshift z"); a1.set_ylabel("chi2_all")
        a1.legend(fontsize=9); a1.grid(alpha=0.3)
        a1.set_title(f"ID {args.id}   template {nm} ({g})   "
                     f"best z={z_[j]:.4f}  chi2_all={c_[j]:,.0f}  "
                     f"(+{c_[j]-best_c2:,.0f} vs global best)", fontsize=11)
        a2.plot(wl_air, flux, lw=0.5, color="0.35", label="observed")
        a2.plot(wl_air, src + skym, lw=0.5, color="#d62728", alpha=0.8,
                label="full model")
        a2.plot(wl_air, src, lw=1.0, color="#0d47a1", label="source only")
        a2.set_xlabel("wavelength [$\\AA$]"); a2.set_ylabel("flux  [$10^{-20}$ cgs]")
        a2.legend(fontsize=9); a2.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "per_template" / f"tpl_{nm}.png", dpi=130,
                    bbox_inches="tight")
        plt.close(fig)

    print(f"\nsaved -> {out}")
    print(f"  landscape.png / top10_summary.png / ranking.txt")
    print(f"  top10/rank_01..{len(tops):02d}.png")
    print(f"  per_template/tpl_*.png  ({len(names)} 張)")


def main(argv=None):
    """本檔的圖 + step4_stages 的 stage1/stage2/compare,一個指令產出全部。

    兩支程式讀同一批 scan 檔、寫同一個資料夾,分開跑只是讓人少拿到一半的圖。
    step4_stages 當模組呼叫(而不是複製它的程式碼),兩邊各自仍可單獨執行。
    """
    args = _parser().parse_args(argv)
    ids = ([int(x) for x in np.load(STEP02 / "object_ids.npy")]
           if str(args.id).lower() == "all" else [int(args.id)])
    for n, sid in enumerate(ids, 1):
        print(f"\n{'=' * 70}\n[{n}/{len(ids)}]  ID {sid}\n{'=' * 70}")
        one = copy.copy(args)          # 淺複製即可:只改 id,其餘都是純量
        one.id = sid
        one_source(one)
        if not args.no_stages:
            stages.main(["--id", str(sid), "--basis", args.basis,
                         "-K", str(args.K), "--s-fix", str(args.s_fix),
                         "--top", str(args.top), "--dedup-dz", str(args.dedup_dz)])


if __name__ == "__main__":
    main()
