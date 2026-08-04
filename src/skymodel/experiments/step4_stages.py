"""把 step4 的兩段拆開看:第 1 段、第 2 段、以及兩者的比較

step4_diagnostics.py 畫的全部是第 1 段(33 條離散模板),但報出去的 z 是第 2 段
(本徵譜)算的 —— 也就是說,決定答案的那個模型從來沒有被畫出來過。

這支補上三塊:

  A  第 1 段還缺的
     ① 恆星那 23 條模板只掃 z ∈ [−0.005, 0.005],畫在 0–1.5 的軸上是原點的一個
        點 —— 等於隱形。給它自己的放大面板。
     ② n_good(z):不同 z 用不同數量的通道擬合,而 chi2_all 是套回全部通道評分的。
        兩者疊看才分得出「這個谷是物理」還是「覆蓋率變化造成的」(4788 Å 的假象
        就是這樣來的)。
     ③ src_min(z):重建的源光譜最小值。負值 = 該 z 的解非物理。掃描檔裡本來就有。
     ④ 三組各自的最佳 chi2(z),並列。

  B  第 2 段(完全不存在)
     chi2(z)、4 個本徵譜係數 vs z、src_min(z)、前 N 名、最佳模型對照觀測。
     step4 沒有存第 2 段的掃描,所以這裡重算。

  C  兩段的比較
     兩條 Δchi2(z)(各自對自己的最小值歸一化 —— 絕對值不可比,因為自由度不同:
     第 1 段是 1 個非負振幅 + K,第 2 段是 4 個可正可負的係數 + K)、
     兩段最佳模型疊在同一張觀測光譜上、殘差對照、自由度表。

    conda run -n astro python src/skymodel/experiments/step4_stages.py --id 14
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# 判斷紅移對不對,最後靠的是「這幾條線有沒有落在模型說的位置」。真空靜止波長 ——
# 模板與 z 都定義在真空,所以先在真空空間定位到通道,再畫在空氣波長軸上,
# 避免多做一次轉換公式引入誤差。
LINES = [("[O II]",  3728.48), ("Mg II",  2799.94), ("Hβ",     4862.68),
         ("[O III]", 4960.30), ("[O III]", 5008.24), ("[N II]", 6549.86),
         ("Hα",      6564.61), ("[N II]", 6585.27)]


def solve(f, v, sky, spline, z, lam_vac, s_fix):
    """單一 (模板, z) 的解,設定與 step4.scan_object 完全相同。

    回傳 dict:chi2_all(套回全通道)、chi2(只用覆蓋通道)、n_good、
    src(源光譜)、model(全模型)、coef(源的係數)、src_min。
    """
    T = redshift_to_grid(spline, z, lam_vac)
    if T.ndim == 1:
        T = T[:, None]
    nc = T.shape[1]
    base = (np.isfinite(f) & np.isfinite(v) & (v > 0)
            & np.all(np.isfinite(sky), axis=0))
    sig = np.sqrt(np.where(v > 0, v, 1.0))
    sky_free, y = sky[1:], f - s_fix * sky[0]
    good = base & np.all(np.isfinite(T), axis=1)
    p = sky_free.shape[0] + nc
    if int(good.sum()) <= p:
        return None
    M = np.empty((int(good.sum()), p))
    M[:, :nc] = T[good] / sig[good][:, None]
    M[:, nc:] = (sky_free[:, good] / sig[good]).T
    lb = np.full(p, -np.inf)
    if nc == 1:                       # 單一模板時 A>=0 等價於「源光譜>=0」
        lb[0] = 0.0
    fit = lsq_linear(M, (y / sig)[good], bounds=(lb, np.full(p, np.inf)),
                     method="bvls")
    th = fit.x
    src = np.nan_to_num(T, nan=0.0) @ th[:nc]
    m = src + th[nc:] @ sky_free
    return dict(chi2_all=float((((y - m) / sig) ** 2)[base].sum()),
                chi2=float(2 * fit.cost), n_good=int(good.sum()), p=p,
                src=src, model=m + s_fix * sky[0], coef=th[:nc],
                src_min=float(src[base].min()), base=base, sig=sig)


def local_minima(y, x, top, dz):
    """chi2 曲線的局部極小值,依深度排序並依 |Δz| 去重。"""
    lo = np.flatnonzero((y[1:-1] <= y[:-2]) & (y[1:-1] <= y[2:])) + 1
    lo = lo[np.argsort(y[lo])]
    keep = []
    for i in lo:
        if all(abs(x[i] - x[j]) > dz for j in keep):
            keep.append(int(i))
        if len(keep) == top:
            break
    return keep


def line_channels(z, wl_vac):
    """擬合出的 z 下,各條診斷譜線落在哪個通道。回傳 [(標籤, 通道)]。"""
    out = []
    for nm, l0 in LINES:
        lo = l0 * (1 + z)
        if wl_vac[0] <= lo <= wl_vac[-1]:
            out.append((nm, int(np.argmin(np.abs(wl_vac - lo)))))
    return out


def spectra_panels(path, title, wl, obs, curves, marks=(), npan=5):
    """5 段波長的光譜對照圖。

    curves = [(標籤, 陣列, 顏色, 線寬)];marks = [(譜線標籤, 通道)]。
    譜線位置標上去之後,「這個 z 對不對」用眼睛兩秒就能判斷 —— 線在不在那裡。
    """
    edges = np.linspace(0, wl.size, npan + 1).astype(int)
    fig, axes = plt.subplots(npan, 1, figsize=(20, 3.0 * npan))
    for k, a in enumerate(np.atleast_1d(axes)):
        sl = slice(edges[k], edges[k + 1])
        a.plot(wl[sl], obs[sl], lw=0.6, color="0.55", label="observed")
        for nm, y, c, w in curves:
            a.plot(wl[sl], y[sl], lw=w, color=c, label=nm)
        a.set_xlim(wl[sl][0], wl[sl][-1])
        for nm, ch in marks:
            if edges[k] <= ch < edges[k + 1]:
                a.axvline(wl[ch], color="#7f7f7f", ls="--", lw=1.0, alpha=0.8)
                a.annotate(nm, (wl[ch], 1.0), xycoords=("data", "axes fraction"),
                           xytext=(2, -12), textcoords="offset points",
                           fontsize=9, color="#404040", rotation=90,
                           va="top", ha="left")
        a.grid(alpha=0.3)
        if k == 0:
            a.set_title(title, fontsize=12)
            a.legend(fontsize=9, ncol=len(curves) + 1, loc="lower right")
    np.atleast_1d(axes)[-1].set_xlabel("wavelength (A)")
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    """argv 留成參數,是為了讓 step4_diagnostics 能把這支當模組直接呼叫。"""
    ap = argparse.ArgumentParser(description="step4 兩段的拆解與比較")
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--s-fix", type=float, default=1.0)
    ap.add_argument("--zstep", type=float, default=5e-4,
                    help="第 2 段重算地形的 z 步長")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--dedup-dz", type=float, default=0.01)
    args = ap.parse_args(argv)
    tag = f"{args.basis}_K{args.K}_s_{args.s_fix}"

    wl_air = np.load(STEP03 / "wavelength.npy")
    wl_vac = air_to_vacuum(wl_air)
    sky = np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                     np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")])
    best = np.load(STEP04 / f"best_{tag}.npz")
    j = int(np.flatnonzero(best["id"] == args.id)[0])
    grp, z2 = str(best["group"][j]), float(best["z"][j])

    ids  = np.load(STEP02 / "object_ids.npy")
    k    = int(np.flatnonzero(ids == args.id)[0])
    nspx = np.load(STEP02 / "object_nspax.npy")[k]
    f_   = np.load(STEP02 / "object_flux.npy")[k] / nspx
    v_   = np.load(STEP02 / "object_var.npy")[k] / nspx ** 2

    out = FIGURES / f"step4_{tag}" / f"id{args.id:02d}"
    for d in ("stage1", "stage2", "compare"):
        (out / d).mkdir(parents=True, exist_ok=True)
    print(f"ID {args.id}   group={grp}   報出的 z={z2:.4f}   nspax={int(np.median(nspx))}")

    # ================= A. 第 1 段 =================
    sc = np.load(STEP04 / f"scan_id{args.id}_{tag}.npz")
    g1, t1, z1a = sc["group"].astype(str), sc["template"].astype(str), sc["z"]
    ca1, ng1, sm1 = sc["chi2_all"], sc["n_good"], sc["src_min"]

    def envelope(mask, zgrid):
        """每個 z 上取該子集裡最好的候選,回傳 (chi2, n_good, src_min, 模板)。"""
        idx = np.round((z1a[mask] - zgrid[0]) / (zgrid[1] - zgrid[0])).astype(int)
        ok = (idx >= 0) & (idx < zgrid.size)
        out = np.full((4, zgrid.size), np.nan, object)
        chi = np.full(zgrid.size, np.inf)
        cm, nm_, sm_, tm = chi.copy(), np.zeros(zgrid.size), np.zeros(zgrid.size), \
            np.empty(zgrid.size, object)
        for i, c, n, s, t in zip(idx[ok], ca1[mask][ok], ng1[mask][ok],
                                 sm1[mask][ok], t1[mask][ok]):
            if c < cm[i]:
                cm[i], nm_[i], sm_[i], tm[i] = c, n, s, t
        return cm, nm_, sm_, tm

    zex = np.unique(z1a[g1 != "star"])
    zst = np.unique(z1a[g1 == "star"])
    ce, ne, se, te = envelope(g1 != "star", zex)
    cs, ns, ss, ts = envelope(g1 == "star", zst)

    def three_panel(path, zg, c, n, s, title, xlabel):
        fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                               gridspec_kw=dict(height_ratios=[2, 1, 1]))
        ax[0].plot(zg, c, lw=0.8, color="#1f77b4")
        i = int(np.nanargmin(c))
        ax[0].axvline(zg[i], color="#d62728", ls="--", lw=1.2,
                      label=f"best z = {zg[i]:.4f}")
        ax[0].set_yscale("log"); ax[0].set_ylabel("chi2_all (all channels)")
        ax[0].set_title(title, fontsize=12); ax[0].legend(fontsize=9)
        ax[1].plot(zg, n, lw=0.8, color="#ff7f0e")
        ax[1].set_ylabel("channels the fit could use\n(template coverage)")
        ax[1].annotate("a step here = the fit changed how much data it used;\n"
                       "a chi2 change at the same z is an artefact, not physics",
                       (0.01, 0.06), xycoords="axes fraction", fontsize=9,
                       color="#8c4a00")
        ax[2].plot(zg, s, lw=0.8, color="#2ca02c")
        ax[2].axhline(0, color="0.4", lw=1.0)
        ax[2].fill_between(zg, np.minimum(s, 0), 0, color="#d62728", alpha=0.35,
                           label="source spectrum goes negative (unphysical)")
        ax[2].set_ylabel("faintest point of the\nreconstructed source spectrum")
        ax[2].annotate("below 0 = the model says the source emits negative "
                       "light = not physical",
                       (0.01, 0.08), xycoords="axes fraction", fontsize=9,
                       color="#8c1a1a")
        ax[2].legend(fontsize=9, loc="lower right")
        ax[2].set_xlabel(xlabel)
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)

    three_panel(out / "stage1" / "landscape_extragalactic.png", zex, ce, ne, se,
                f"ID {args.id}  stage 1, galaxy+QSO templates — is the valley "
                f"physical, or an artefact of changing coverage?", "redshift z")
    three_panel(out / "stage1" / "landscape_star.png", zst, cs, ns, ss,
                f"ID {args.id}  stage 1, the 23 stellar templates "
                f"(invisible on a 0–1.5 axis)", "redshift z (stellar range)")

    # 恆星只掃 z ∈ [-0.005, 0.005],和 0–1.5 放同一根軸會縮成一個點。
    # 但要比較的是「哪一組贏」,不是「恆星隨 z 怎麼變」—— 所以把恆星的最佳值
    # 畫成一條水平線,三組就能直接比高低。
    fig, ax = plt.subplots(figsize=(14, 6))
    for gname in ("galaxy", "qso"):
        c, _, _, _ = envelope(g1 == gname, zex)
        ax.plot(zex, c, lw=0.8, color=GROUP_COLOR[gname], label=gname)
    cs_best = float(np.nanmin(cs))
    ax.axhline(cs_best, color=GROUP_COLOR["star"], ls="--", lw=1.6,
               label=f"star, best over its own z range ({cs_best:,.0f})")
    ax.annotate(f"star best = {cs_best:,.0f}\n(template {ts[int(np.nanargmin(cs))]},"
                f" z = {zst[int(np.nanargmin(cs))]:+.4f})",
                (zex[-1], cs_best), xytext=(-8, 8), textcoords="offset points",
                ha="right", fontsize=9, color=GROUP_COLOR["star"])
    for gname in ("galaxy", "qso"):
        c, _, _, _ = envelope(g1 == gname, zex)
        i = int(np.nanargmin(c))
        ax.plot(zex[i], c[i], "o", color=GROUP_COLOR[gname], ms=8, zorder=4)
        ax.annotate(f"{gname} best = {c[i]:,.0f}\nat z = {zex[i]:.4f}",
                    (zex[i], c[i]), xytext=(6, -18), textcoords="offset points",
                    fontsize=9, color=GROUP_COLOR[gname])
    ax.set_yscale("log"); ax.set_xlabel("redshift z")
    ax.set_ylabel("best chi2_all within the group (lower = better)")
    ax.set_title(f"ID {args.id}  which group wins, and by how much\n"
                 f"all three on one axis — the vertical gap between the curves "
                 f"IS the margin", fontsize=12)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(out / "stage1" / "per_group.png", dpi=140,
                                    bbox_inches="tight"); plt.close(fig)

    print(f"\n第 1 段  各組最佳")
    print(f"{'group':>8}{'template':>10}{'z':>10}{'chi2_all':>14}{'n_good':>9}"
          f"{'src_min':>10}")
    print("-" * 62)
    s1_best = {}
    for gname in ("star", "galaxy", "qso"):
        m = g1 == gname
        i = int(np.argmin(ca1[m]))
        s1_best[gname] = dict(tpl=t1[m][i], z=float(z1a[m][i]),
                              chi2=float(ca1[m][i]), n=int(ng1[m][i]),
                              sm=float(sm1[m][i]))
        b = s1_best[gname]
        print(f"{gname:>8}{b['tpl']:>10}{b['z']:>10.4f}{b['chi2']:>14,.0f}"
              f"{b['n']:>9}{b['sm']:>10.3f}")

    # ================= B. 第 2 段 =================
    eigen = {"galaxy": load_eigen_galaxy(EIGEN_GAL), "qso": load_eigen_qso(EIGEN_QSO)}
    S2 = {}

    # step4 現在會把「贏家那一組」的第 2 段掃描存成 scan2_*.npz —— 那正是 pipeline
    # 實際跑過的東西,直接讀比重算誠實(格點、設定都保證一致)。另一組沒有存,
    # 因為 step4 不會去跑它,只能自己重算。
    s2f = STEP04 / f"scan2_id{args.id}_{tag}.npz"
    if s2f.exists():
        d = np.load(s2f)
        o = np.argsort(d["z"])
        S2[grp] = dict(z=d["z"][o], chi2=d["chi2_all"][o], smin=d["src_min"][o],
                       ng=d["n_good"][o], coef=d["A"][o])
        print(f"第 2 段 {grp}:讀入 step4 存的掃描 ({o.size} 個 z)", flush=True)

    for gname, spl in eigen.items():
        if gname in S2:
            continue
        zz = np.arange(0.0, 1.5 + args.zstep / 2, args.zstep)
        res = [solve(f_, v_, sky, spl, z, wl_vac, args.s_fix) for z in zz]
        S2[gname] = dict(
            z=zz,
            chi2=np.array([r["chi2_all"] if r else np.nan for r in res]),
            smin=np.array([r["src_min"] if r else np.nan for r in res]),
            ng=np.array([r["n_good"] if r else 0 for r in res]),
            coef=np.array([r["coef"] if r else [np.nan] * 4 for r in res]))
        print(f"第 2 段 {gname}:重算 ({zz.size} 個 z)", flush=True)

    for gname in eigen:
        d = S2[gname]
        zg2 = d["z"]
        fig, ax = plt.subplots(3, 1, figsize=(14, 11), sharex=True,
                               gridspec_kw=dict(height_ratios=[2, 1.4, 1]))
        ax[0].plot(zg2, d["chi2"], lw=0.8, color=GROUP_COLOR[gname])
        i = int(np.nanargmin(d["chi2"]))
        ax[0].axvline(zg2[i], color="#d62728", ls="--", lw=1.2,
                      label=f"best z = {zg2[i]:.4f}")
        if gname == grp:
            ax[0].axvline(z2, color="k", ls=":", lw=1.2,
                          label=f"z reported by step4 = {z2:.4f}")
        ax[0].set_yscale("log"); ax[0].set_ylabel("chi2_all")
        ax[0].set_title(f"ID {args.id}  stage 2 — the {gname} eigenspectra, "
                        f"the model that actually sets the reported z", fontsize=12)
        ax[0].legend(fontsize=9)
        for c in range(4):
            ax[1].plot(zg2, d["coef"][:, c], lw=0.8, label=f"a{c+1}")
        ax[1].axhline(0, color="0.4", lw=1.0)
        ax[1].set_ylabel("eigen coefficients\n(unconstrained in sign)")
        ax[1].legend(fontsize=8, ncol=4)
        ax[2].plot(zg2, d["smin"], lw=0.8, color="#2ca02c")
        ax[2].axhline(0, color="0.4", lw=1.0)
        ax[2].fill_between(zg2, np.minimum(d["smin"], 0), 0, color="#d62728",
                           alpha=0.35, label="source spectrum negative (unphysical)")
        ax[2].set_ylabel("faintest point of the\nreconstructed source spectrum")
        ax[2].annotate("below 0 = the model says the source emits negative "
                       "light = not physical",
                       (0.01, 0.08), xycoords="axes fraction", fontsize=9,
                       color="#8c1a1a")
        ax[2].set_xlabel("redshift z"); ax[2].legend(fontsize=9, loc="lower right")
        for a in ax:
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "stage2" / f"landscape_{gname}.png", dpi=140,
                    bbox_inches="tight"); plt.close(fig)

    if grp not in S2:
        # 恆星沒有本徵譜,第 2 段根本不存在 —— 第 1 段的結果就是最終答案。
        # (svd K30 目前 37 個源都不是恆星,但 svdclean 有兩個,所以要擋。)
        print(f"\n{grp} 沒有本徵譜,第 2 段不適用;只產出 stage1/。")
        print(f"saved -> {out}")
        return
    d = S2[grp]
    zg2 = d["z"]
    tops = local_minima(d["chi2"], zg2, args.top, args.dedup_dz)
    print(f"\n第 2 段({grp} eigen)前 {len(tops)} 個局部極小值")
    print(f"{'#':>3}{'z':>10}{'chi2_all':>14}{'vs 第1名':>12}{'src_min':>10}"
          f"{'a1':>11}")
    print("-" * 60)
    for r, i in enumerate(tops, 1):
        print(f"{r:>3}{zg2[i]:>10.4f}{d['chi2'][i]:>14,.0f}"
              f"{d['chi2'][i]-d['chi2'][tops[0]]:>12,.0f}{d['smin'][i]:>10.3f}"
              f"{d['coef'][i,0]:>11.4g}")

    r2 = solve(f_, v_, sky, eigen[grp], z2, wl_vac, args.s_fix)
    spectra_panels(out / "stage2" / "best_spectrum.png",
                   f"ID {args.id}  stage 2 best: {grp} eigenspectra at z = {z2:.4f}",
                   wl_air, f_, [("full model", r2["model"], "#d62728", 0.9),
                                ("source only", r2["src"], "#1f77b4", 1.1)])

    # ================= C. 兩段的比較 =================
    b1 = s1_best[grp]
    spl1 = load_sdss_template(TPL_DIR / f"spDR2-{b1['tpl']}.fit")
    r1 = solve(f_, v_, sky, spl1, b1["z"], wl_vac, args.s_fix)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(zex, ce - np.nanmin(ce), lw=0.8, color="#1f77b4",
            label=f"stage 1 (discrete templates, p = 1 + {args.K})")
    ax.plot(d["z"], d["chi2"] - np.nanmin(d["chi2"]), lw=0.8, color="#d62728",
            label=f"stage 2 ({grp} eigenspectra, p = 4 + {args.K})")
    ax.axvline(b1["z"], color="#1f77b4", ls="--", lw=1.1)
    ax.axvline(z2, color="#d62728", ls="--", lw=1.1)
    ax.set_yscale("log"); ax.set_ylim(bottom=1)
    ax.set_xlabel("redshift z"); ax.set_ylabel("chi2_all − its own minimum")
    ax.set_title(f"ID {args.id}  the two stages, each normalised to its own "
                 f"minimum\nabsolute chi2 is NOT comparable — the two models have "
                 f"different numbers of free parameters", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(out / "compare" / "landscape_normalised.png",
                                    dpi=140, bbox_inches="tight"); plt.close(fig)

    spectra_panels(out / "compare" / "spectra.png",
                   f"ID {args.id}  stage 1 (tpl {b1['tpl']}, z={b1['z']:.4f}) vs "
                   f"stage 2 ({grp} eigen, z={z2:.4f})", wl_air, f_,
                   [("stage 1 model", r1["model"], "#1f77b4", 0.9),
                    ("stage 2 model", r2["model"], "#d62728", 0.9)])
    spectra_panels(out / "compare" / "residual.png",
                   f"ID {args.id}  residual / sigma — where does each stage fail?",
                   wl_air, np.zeros_like(f_),
                   [("stage 1", (f_ - r1["model"]) / r1["sig"], "#1f77b4", 0.7),
                    ("stage 2", (f_ - r2["model"]) / r2["sig"], "#d62728", 0.7)])

    lines = []
    lines.append(f"ID {args.id}   group = {grp}\n")
    lines.append(f"{'':>10}{'model':>26}{'p':>5}{'n_good':>9}{'chi2_all':>14}"
                 f"{'chi2/dof':>11}{'z':>10}{'src_min':>10}")
    lines.append("-" * 96)
    for nm, r, zz, mdl in [("stage 1", r1, b1["z"], f"template {b1['tpl']}"),
                           ("stage 2", r2, z2, f"{grp} eigenspectra x4")]:
        lines.append(f"{nm:>10}{mdl:>26}{r['p']:>5}{r['n_good']:>9}"
                     f"{r['chi2_all']:>14,.0f}"
                     f"{r['chi2_all']/(r['n_good']-r['p']):>11.3f}{zz:>10.4f}"
                     f"{r['src_min']:>10.3f}")
    lines.append("")
    lines.append("第 2 段的 chi2 必定較低,因為它多了 3 個自由參數而且係數可正可負;")
    lines.append("直接比較兩者的 chi2_all 是無效的。上面的 chi2/dof 已做自由度修正,")
    lines.append("但它仍不是模型選擇的判準 —— 兩個模型並非巢狀關係。")
    (out / "compare" / "dof.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
