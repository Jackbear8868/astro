"""把 s 換成空間場之後,源旁邊的 over-subtraction 有沒有改善 —— 直接看殘差。

前面所有的實驗量的都是「s_hat 準不準」。那不是我們要的答案。真正的問題是
**科學上有沒有變好**:源外圍那圈延展光,有沒有從天空模型手裡救回來。

比較的兩個模型(只動 blank 區)
------------------------------
    A 現行   D = s·C_sky + Σcₖ·Lₖ        s 自由,整個擬合結果都當天空扣掉
    B 新的   D = s_hat(p)·C_sky + Σcₖ·Lₖ  s **鎖在空間場的值**,只解天光線係數

差別只有一處:A 的 s 可以逐格自由長高去吞源的光,B 不行 —— 它的值是從遠場
決定的,源旁邊那格說了不算。

為什麼只看 blank 就夠:你要看的環在**源足跡外面**,那裡本來就是 blank。而且
blank 的設計矩陣不隨 spaxel 改變,一次 pinv 就解完所有乾淨的 spaxel,不必等
逐 spaxel 的源擬合。源足跡**內部**這支不處理。

判讀方向(很重要,別看反了)
--------------------------
blank 區的殘差理想上應該是 0 + 雜訊。但源旁邊那圈**本來就有源的光**,所以:

    殘差 ≈ 0     -> 源的光被天空模型吃掉了   ← 這是病
    殘差 > 0     -> 源的光留在殘差裡         ← 這是我們要的

所以 B 的徑向剖面若**高於** A,就是改善。差值 B − A 就是救回來的光。

天光線係數 cₖ 在兩邊都是自由的,它們也能吸走一部分連續譜(30 條 basis 對 3801
個通道),所以 B 不會把光 100% 救回來。這支量的是實際救回多少,不是理論上限。

    conda run -n astro python src/skymodel/experiments/s_field_residual.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import scale, build_s_field  # noqa: E402

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP03  = ROOT / "results/skymodel/step03"
STEP05  = ROOT / "results/skymodel/step05"
CUBE    = ROOT / "data/Haro11_NEpointing_wsky.fits"
FIGURES = ROOT / "results/skymodel/figures/s_field"

MIN_COVERAGE = 0.9      # 與 step5 一致:排除視野邊緣的部分覆蓋 spaxel
CLIP_SIGMA   = 30       # 與 step3 / oversub_whitelight 一致的壞 voxel 剔除門檻


def profile(img, dist, usable, rmax):
    """逐環中位數。中位數而非平均:源附近有零星亮結構,平均會被它們拉走。"""
    med = np.full(rmax, np.nan)
    cnt = np.zeros(rmax, int)
    for r in range(rmax):
        m = usable & (dist > r) & (dist <= r + 1)
        cnt[r] = int(m.sum())
        if cnt[r] >= 20:
            med[r] = float(np.median(img[m]))
    return med, cnt


def main():
    ap = argparse.ArgumentParser(description="s 換成空間場後,源旁殘差有沒有改善")
    ap.add_argument("--run", default="blank_svdK30_tpl68_s1",
                    help="從哪個 step5 run 讀 s_map.npy 當建場的原料")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--band", type=float, nargs=2, default=(5000, 6000),
                    help="壓成影像用的波段。預設 5000-6000:這一段天光線最少")
    ap.add_argument("--r-far", type=float, default=15.0)
    ap.add_argument("--r-far-haro", type=float, default=50.0)
    ap.add_argument("--clip", type=float, default=8.0)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--rmax", type=int, default=25)
    ap.add_argument("--min-area", type=int, default=25)
    ap.add_argument("--edge-margin", type=float, default=15.0)
    ap.add_argument("--n-null", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figdir", default=str(FIGURES))
    args = ap.parse_args()

    fig_dir = Path(args.figdir); fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    seg   = fits.getdata(STEP01 / "seg.fits").astype(int)
    white = fits.getdata(STEP01 / "whitelight.fits")
    s_map = np.load(STEP05 / args.run / "s_map.npy").astype(np.float64)
    wl    = np.load(STEP03 / "wavelength.npy")
    sky   = np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                       np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")])
    ny, nx = seg.shape
    nz = sky.shape[1]
    print(f"天空模型 {sky.shape}  ({args.basis} K={args.K})")

    # ---- 讀 cube 的 blank 欄 ---------------------------------------------
    with fits.open(CUBE, memmap=True) as h:
        hdr = h["DATA"].header
        cov = np.zeros(ny * nx)
        for j in range(0, nz, 400):
            cov += np.isfinite(np.asarray(h["DATA"].data[j:j+400],
                                          np.float32)).sum(axis=0).ravel()
        cov /= nz
        valid = (white != 0).ravel() & (cov >= MIN_COVERAGE)
        blank = valid & (seg.ravel() == 0)
        idx   = np.flatnonzero(blank)
        D = np.empty((nz, idx.size), np.float32)
        for j in range(0, nz, 200):
            D[j:j+200] = np.asarray(h["DATA"].data[j:j+200],
                                    np.float32).reshape(-1, ny*nx)[:, idx]
    blank2 = blank.reshape(ny, nx)
    print(f"blank {idx.size:,} 個 spaxel 讀入 ({D.nbytes/2**30:.2f} GiB)")

    # 一次 pinv 只對「光譜完整」的 spaxel 成立;其餘留 NaN。
    # 這是診斷,不是 pipeline —— step5 會逐一解那些,這裡直接跳過並報數量。
    clean = np.isfinite(D).all(axis=0)
    print(f"光譜完整 {int(clean.sum()):,} / {idx.size:,} "
          f"({100*clean.mean():.1f}%),其餘不納入本圖")

    # ---- 建空間場 ---------------------------------------------------------
    s_hat_full, train, ridge = build_s_field(s_map, seg, blank2, args.r_far,
                                             args.r_far_haro, args.clip, args.sigma)
    print(f"空間場:訓練 {int(train.sum()):,} 格 "
          f"(r_far {args.r_far:g}, Haro {args.r_far_haro:g}, ridge {ridge:.4f})")
    s_hat = s_hat_full.ravel()[idx]

    # ---- 兩個模型 ---------------------------------------------------------
    Dc = D[:, clean]
    # A:現行做法,s 自由。設計矩陣不隨 spaxel 改變,一次 pinv 解完。
    cA = np.linalg.pinv(sky.T) @ Dc
    # B:s 鎖在 s_hat。先把 s_hat·C_sky 扣掉,剩下只解 K−1 個天光線係數。
    #    這裡不能保留 s ≥ 0 的非負限制 —— 扣掉之後第 0 個參數已經是 c₁,
    #    天光線係數的正負由 basis 的方向決定(與 step5 fit_blank 的處理一致)。
    yB = Dc - s_hat[clean][None, :].astype(np.float32) * sky[0][:, None].astype(np.float32)
    cB = np.linalg.pinv(sky[1:].T) @ yB
    print(f"A: s 中位 {np.median(cA[0]):.5f}   B: s_hat 中位 {np.median(s_hat[clean]):.5f}")

    bm = (wl >= args.band[0]) & (wl < args.band[1])
    print(f"波段 {args.band[0]:.0f}-{args.band[1]:.0f} A:{int(bm.sum())} 個通道")
    rA = Dc[bm] - sky[:, bm].T @ cA
    rB = yB[bm] - sky[1:, bm].T @ cB
    del D, Dc, yB

    # 壞 voxel 剔除:同一通道、跨 spaxel、穩健散布、30 sigma。做法與門檻沿用
    # step3_sky_basis.py 的 mean_sky,只作用在 blank(這裡本來就只有 blank)。
    def collapse(r):
        p16, med, p84 = np.percentile(r, [16, 50, 84], axis=1)
        sg  = np.maximum((p84 - p16) / 2, 1e-6)
        bad = np.abs(r - med[:, None]) > CLIP_SIGMA * sg[:, None]
        return np.nanmean(np.where(bad, np.nan, r), axis=0), int(bad.sum())

    vA, nbA = collapse(rA)
    vB, nbB = collapse(rB)
    print(f"sigma-clip 剔除 A {nbA:,} / B {nbB:,} 個元素")
    del rA, rB

    imgA = np.full(ny * nx, np.nan); imgA[idx[clean]] = vA; imgA = imgA.reshape(ny, nx)
    imgB = np.full(ny * nx, np.nan); imgB[idx[clean]] = vB; imgB = imgB.reshape(ny, nx)

    # ---- 徑向剖面 ---------------------------------------------------------
    edge  = ndimage.distance_transform_edt(white != 0)
    d_all = ndimage.distance_transform_edt(seg == 0)
    d_har = ndimage.distance_transform_edt(seg != 1)
    ids = [int(i) for i in np.unique(seg[seg > 0])
           if i != 1 and (seg == i).sum() >= args.min_area
           and edge[seg == i].min() > args.edge_margin]
    other = np.isin(seg, ids)
    d_oth = ndimage.distance_transform_edt(~other)
    usable = np.isfinite(imgA) & np.isfinite(imgB) & (edge > args.edge_margin)
    print(f"\n源(面積 >= {args.min_area},離視場邊界夠遠,排除 Haro 11):{len(ids)} 個")

    far = usable & (d_all > 30)
    bA, bB = float(np.median(imgA[far])), float(np.median(imgB[far]))
    sg_px  = scale(imgA[far])
    print(f"遠場基線:A {bA:+.4f}   B {bB:+.4f}   逐格散布 {sg_px:.4f}")

    prof = {}
    for nm, d, extra in (("other", d_oth, d_har > 30), ("haro", d_har, ~other)):
        u = usable & extra
        prof[f"{nm}|A"], prof[f"{nm}|n"] = profile(imgA, d, u, args.rmax)
        prof[f"{nm}|B"], _               = profile(imgB, d, u, args.rmax)

    # 虛無:每個源各自平移到空白處,同樣算剖面。它回答「這條剖面的起伏有多少
    # 只是雜訊」—— 沒有它就無法判斷 B − A 的差是不是真的。
    stamps = [(lambda yx: (yx[0] - yx[0].min(), yx[1] - yx[1].min()))(np.nonzero(seg == i))
              for i in ids]
    free = usable & (d_all > 30)
    u_null = usable & ~other & (d_all > 30)
    nulls = {"A": [], "B": []}
    for _ in range(args.n_null):
        stamp = np.zeros_like(other); occ = ~free
        for fy, fx in stamps:
            for _t in range(200):
                dy = int(rng.integers(0, ny - fy.max()))
                dx = int(rng.integers(0, nx - fx.max()))
                yy, xx = fy + dy, fx + dx
                if np.any(occ[yy, xx]):
                    continue
                stamp[yy, xx] = True
                occ[max(yy.min()-3, 0):yy.max()+4, max(xx.min()-3, 0):xx.max()+4] = True
                break
        dn = ndimage.distance_transform_edt(~stamp)
        nulls["A"].append(profile(imgA, dn, u_null, args.rmax)[0])
        nulls["B"].append(profile(imgB, dn, u_null, args.rmax)[0])
    nul = {k: np.array(v) for k, v in nulls.items()}
    print(f"虛無對照:{args.n_null} 次(每個源各自平移)")

    # ---- 報表 -------------------------------------------------------------
    print(f"\n【其他源】源足跡外的殘差(已扣各自的遠場基線)"
          f"   {args.band[0]:.0f}-{args.band[1]:.0f} A\n")
    print(f"{'r [px]':>7}{'n':>7}{'A 現行':>10}{'B 場':>10}{'B−A':>10}"
          f"{'B−A 虛無 16-84%':>20}{'救回':>8}")
    print("-" * 74)
    gain = []
    for r in range(min(args.rmax, 14)):
        a_, b_ = prof["other|A"][r] - bA, prof["other|B"][r] - bB
        if not (np.isfinite(a_) and np.isfinite(b_)):
            continue
        dn = nul["B"][:, r] - bB - (nul["A"][:, r] - bA)
        dn = dn[np.isfinite(dn)]
        lo, hi = (np.percentile(dn, [16, 84]) if dn.size >= 10 else (np.nan, np.nan))
        pct = 100 * (b_ - a_) / abs(a_) if abs(a_) > 1e-9 else np.nan
        gain.append((r, b_ - a_, lo, hi))
        print(f"{r:>7}{prof['other|n'][r]:>7}{a_:>10.4f}{b_:>10.4f}{b_-a_:>10.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>20}{pct:>7.0f}%")

    print(f"\n【Haro 11】\n")
    print(f"{'r [px]':>7}{'n':>7}{'A 現行':>10}{'B 場':>10}{'B−A':>10}{'救回':>8}")
    print("-" * 54)
    for r in range(min(args.rmax, 25)):
        a_, b_ = prof["haro|A"][r] - bA, prof["haro|B"][r] - bB
        if not (np.isfinite(a_) and np.isfinite(b_)):
            continue
        pct = 100 * (b_ - a_) / abs(a_) if abs(a_) > 1e-9 else np.nan
        print(f"{r:>7}{prof['haro|n'][r]:>7}{a_:>10.4f}{b_:>10.4f}"
              f"{b_-a_:>10.4f}{pct:>7.0f}%")

    # 一個總結數字:1-3 px 環內的總流量差
    ring = usable & (d_all > 1) & (d_all <= 3)
    fA = float(np.nansum(imgA[ring] - bA))
    fB = float(np.nansum(imgB[ring] - bB))
    print(f"\n源足跡外 1-3 px 環({int(ring.sum()):,} 格)的總流量:")
    print(f"  A 現行 {fA:>10.1f}    B 場 {fB:>10.1f}    差 {fB-fA:>+10.1f}"
          f"  ({100*(fB-fA)/abs(fA) if abs(fA)>1e-9 else float('nan'):+.1f}%)")

    # ---- 圖 ---------------------------------------------------------------
    v = 4 * sg_px
    fig, ax = plt.subplots(1, 3, figsize=(19, 6.2))
    for a_, (img, t) in zip(ax, [
            (imgA - bA, "A — current:  $s$ free per spaxel"),
            (imgB - bB, "B — new:  $s$ locked to the spatial field $\\hat{s}(x,y)$"),
            (imgB - bB - (imgA - bA), "B − A  =  light recovered from the sky model")]):
        im = a_.imshow(img, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
        a_.contour(seg > 0, levels=[0.5], colors="k", linewidths=0.4)
        a_.set_title(t, fontsize=10); a_.set_xticks([]); a_.set_yticks([])
        plt.colorbar(im, ax=a_, fraction=0.046, pad=0.02)
    fig.suptitle(f"sky-subtracted residual in the blank region   "
                 f"{args.band[0]:.0f}-{args.band[1]:.0f} A   "
                 f"colour scale $\\pm{v:.3f}$ (4x the blank scatter)   "
                 f"blue = over-subtracted", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p1 = fig_dir / "04_residual_compare.png"
    fig.savefig(p1, dpi=140, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
    rr = np.arange(args.rmax) + 0.5
    for a_, nm, ttl in ((ax[0], "other", f"other sources ({len(ids)})"),
                        (ax[1], "haro", "Haro 11")):
        if nm == "other" and nul["A"].size:
            lo = np.nanpercentile(nul["A"] - bA, 2.5, axis=0)
            hi = np.nanpercentile(nul["A"] - bA, 97.5, axis=0)
            a_.fill_between(rr, lo, hi, color="0.85", zorder=0,
                            label="null 2.5–97.5%")
        a_.axhline(0, color="0.6", lw=0.8)
        a_.plot(rr, prof[f"{nm}|A"] - bA, "o-", ms=4, lw=1.6, color="#1f77b4",
                label="A — $s$ free (current)")
        a_.plot(rr, prof[f"{nm}|B"] - bB, "s-", ms=4, lw=1.6, color="#d62728",
                label="B — $s$ from the field")
        a_.set_xlabel("distance from the source footprint [px]")
        a_.set_ylabel(f"median residual, {args.band[0]:.0f}-{args.band[1]:.0f} A")
        a_.set_title(ttl, fontsize=10); a_.legend(fontsize=8); a_.grid(alpha=0.3)
    fig.suptitle("higher = more of the source's extended light survives "
                 "the sky subtraction", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p2 = fig_dir / "05_residual_profile.png"
    fig.savefig(p2, dpi=140, bbox_inches="tight"); plt.close(fig)

    # 放大看甜甜圈:程式挑源,不用眼睛挑(用眼睛挑等於先看到答案再選證據)
    ringm = (d_all >= 1) & (d_all < 3)
    score = {}
    for i in ids:
        m = ringm & (ndimage.distance_transform_edt(seg != i) < 6) & usable
        if m.sum() >= 20:
            score[i] = float(np.nanmean(imgA[m] - bA))
    pick = sorted(score, key=score.get)[:5]
    print(f"\n環最負(被扣最兇)的源:" + "  ".join(f"#{i}({score[i]:+.3f})" for i in pick))
    if pick:
        fig, ax = plt.subplots(len(pick), 3, squeeze=False,
                               figsize=(9.6, 3.1 * len(pick)))
        for row, i in enumerate(pick):
            y, x = np.nonzero(seg == i)
            y0, y1 = max(y.min()-14, 0), min(y.max()+15, ny)
            x0, x1 = max(x.min()-14, 0), min(x.max()+15, nx)
            for col, (img, t) in enumerate([(imgA-bA, "A current"), (imgB-bB, "B field"),
                                            (imgB-bB-(imgA-bA), "B − A")]):
                a_ = ax[row][col]
                a_.imshow(img[y0:y1, x0:x1], origin="lower", cmap="RdBu_r",
                          vmin=-v, vmax=v)
                a_.contour(seg[y0:y1, x0:x1] == i, levels=[0.5], colors="k",
                           linewidths=0.8)
                a_.set_xticks([]); a_.set_yticks([])
                if row == 0:
                    a_.set_title(t, fontsize=10)
                if col == 0:
                    a_.set_ylabel(f"#{i}  {int((seg==i).sum())} px", fontsize=9)
        fig.suptitle(f"the donut, zoomed   {args.band[0]:.0f}-{args.band[1]:.0f} A",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        p3 = fig_dir / "06_donut_compare.png"
        fig.savefig(p3, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"saved -> {p3}")

    (fig_dir / "s_field_residual.json").write_text(json.dumps(dict(
        run=args.run, band=list(args.band), basis=args.basis, K=args.K,
        r_far=args.r_far, r_far_haro=args.r_far_haro, sigma=args.sigma,
        baseline_A=bA, baseline_B=bB, sigma_pixel=sg_px, n_sources=len(ids),
        ring_flux=dict(A=fA, B=fB, diff=fB - fA),
        profiles={k: [None if (isinstance(x, float) and x != x) else float(x)
                      for x in np.asarray(v).tolist()] for k, v in prof.items()}),
        ensure_ascii=False, indent=2))
    print(f"saved -> {p1}\n         {p2}")


if __name__ == "__main__":
    main()
