"""源旁邊那圈的殘差**光譜** —— s 換成空間場之後,被吃掉的是什麼、救回的是什麼。

為什麼光譜比影像更能定案
------------------------
白光影像只告訴你「多了多少光」。光譜告訴你**那是什麼光**,而這裡有一個天然的
對照組:

    紅移的發射線(Haro 11 z=0.0204:[O III] 5007 -> 5109 A、Ha 6563 -> 6697 A)
    **不是天光線**。天空模型 = C_sky + K 條從 blank 學來的天光線 basis,
    它沒有任何成分長得像 5109 A 的一條線,所以**無論哪個模型都扣不掉它**。

    源的**連續譜**則和 C_sky 幾乎同形,兩者最小平方分不開 —— 這正是病灶。

所以預期是:兩個模型的殘差都保留發射線,差別全在連續譜。若圖真的長這樣,就
證明差異來自我們宣稱的機制,而不是別的東西。

三條線一起看
------------
    P(λ)  = <環的原始資料> − <遠場的原始資料>      **完全沒有經過擬合**
            這是「那裡到底多了什麼光」的模型無關答案,是真值的最佳代表。
    resid_A = 現行做法(s 逐格自由)扣完天空剩下的
    resid_B = s 鎖在空間場扣完天空剩下的

    resid ≈ 0    -> 那段光被天空模型吃掉了
    resid ≈ P    -> 那段光完整留下來

不必算整個殘差 cube
-------------------
模型是線性的,所以一塊區域的平均殘差 = 平均資料 − 天空模型代入平均係數:

    <resid_A> = <D> − skyᵀ · <c_A>
    <resid_B> = <D> − <s_hat>·C_sky − sky[1:]ᵀ · <c_B>

只要區域平均,不必展開 3801 x 78,000 的殘差矩陣。

    conda run -n astro python src/skymodel/experiments/s_field_spectrum.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s_field_residual import MIN_COVERAGE
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import scale, build_s_field  # noqa: E402

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/ne_pointing/step01"
STEP03  = ROOT / "results/skymodel/ne_pointing/step03"
STEP05  = ROOT / "results/skymodel/ne_pointing/step05"
CUBE    = ROOT / "data/Haro11_NEpointing_wsky.fits"
FIGURES = ROOT / "results/skymodel/evaluation/s_field"

Z_HARO = 0.0204
LINES  = [("[O II] 3727",  3727.0), ("H-beta 4861", 4861.3),
          ("[O III] 4959", 4958.9), ("[O III] 5007", 5006.8),
          ("H-alpha 6563", 6562.8), ("[S II] 6716",  6716.4)]


def main():
    ap = argparse.ArgumentParser(description="源旁那圈的殘差光譜:A vs B")
    ap.add_argument("--run", default="blank_svdK30_tpl68_s1")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--r-far", type=float, default=15.0)
    ap.add_argument("--r-far-haro", type=float, default=50.0)
    ap.add_argument("--clip", type=float, default=8.0)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--min-area", type=int, default=25)
    ap.add_argument("--edge-margin", type=float, default=15.0)
    ap.add_argument("--smooth", type=int, default=9,
                    help="畫圖時的移動平均寬度(通道)。只是為了看得清楚,"
                         "所有數字都沒有平滑過")
    ap.add_argument("--figdir", default=str(FIGURES))
    args = ap.parse_args()

    fig_dir = Path(args.figdir); fig_dir.mkdir(parents=True, exist_ok=True)
    seg   = fits.getdata(STEP01 / "seg.fits").astype(int)
    white = fits.getdata(STEP01 / "whitelight.fits")
    s_map = np.load(STEP05 / args.run / "s_map.npy").astype(np.float64)
    wl    = np.load(STEP03 / "wavelength.npy")
    sky   = np.vstack([np.load(STEP03 / "sky_continuum.npy"),
                       np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")])
    ny, nx = seg.shape
    nz = sky.shape[1]

    with fits.open(CUBE, memmap=True) as h:
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
    clean = np.isfinite(D).all(axis=0)
    D = D[:, clean]
    idx = idx[clean]
    print(f"blank 且光譜完整:{idx.size:,} 個 spaxel")

    s_hat_full, train = build_s_field(s_map, seg, blank.reshape(ny, nx),
                                             args.r_far, args.r_far_haro,
                                             args.clip)
    s_hat = s_hat_full.ravel()[idx]
    print(f"空間場訓練 {int(train.sum()):,} 格 "
          f"(r_far {args.r_far:g}, Haro {args.r_far_haro:g})")

    cA = np.linalg.pinv(sky.T) @ D
    cB = np.linalg.pinv(sky[1:].T) @ (D - s_hat[None, :].astype(np.float32)
                                      * sky[0][:, None].astype(np.float32))

    # ---- 區域 -------------------------------------------------------------
    edge  = ndimage.distance_transform_edt(white != 0)
    d_all = ndimage.distance_transform_edt(seg == 0)
    d_har = ndimage.distance_transform_edt(seg != 1)
    ids = [int(i) for i in np.unique(seg[seg > 0])
           if i != 1 and (seg == i).sum() >= args.min_area
           and edge[seg == i].min() > args.edge_margin]
    d_oth = ndimage.distance_transform_edt(~np.isin(seg, ids))
    base_ok = (edge > args.edge_margin).ravel()[idx]
    print(f"其他源 {len(ids)} 個")

    def sel(d, lo, hi, extra=None):
        m = base_ok & (d.ravel()[idx] > lo) & (d.ravel()[idx] <= hi)
        return m if extra is None else m & extra.ravel()[idx]

    far = sel(d_all, 30, 1e9)
    regions = {
        "other 1-3 px":  sel(d_oth, 1, 3,  d_har > 30),
        "other 3-6 px":  sel(d_oth, 3, 6,  d_har > 30),
        "Haro 11 1-3 px":  sel(d_har, 1, 3,  d_oth > 6),
        "Haro 11 3-10 px": sel(d_har, 3, 10, d_oth > 6),
        "Haro 11 10-25 px": sel(d_har, 10, 25, d_oth > 6),
    }

    # <resid> = <D> − 模型代入平均係數。模型線性,所以可以只算區域平均。
    Dfar = D[:, far].mean(axis=1)
    out = {}
    print(f"\n{'區域':>18}{'spaxel':>8}   殘差 rms (5000-6000 A)")
    bm = (wl >= 5000) & (wl < 6000)
    for nm, m in regions.items():
        n = int(m.sum())
        if n < 30:
            print(f"{nm:>18}{n:>8}   格數不足,跳過"); continue
        Dm = D[:, m].mean(axis=1)
        rA = Dm - sky.T @ cA[:, m].mean(axis=1)
        rB = Dm - s_hat[m].mean() * sky[0] - sky[1:].T @ cB[:, m].mean(axis=1)
        P  = Dm - Dfar                       # 模型無關的「多出來的光」
        out[nm] = dict(n=n, P=P, A=rA, B=rB)
        print(f"{nm:>18}{n:>8}   A {np.sqrt(np.mean(rA[bm]**2)):7.3f}"
              f"   B {np.sqrt(np.mean(rB[bm]**2)):7.3f}"
              f"   P {np.sqrt(np.mean(P[bm]**2)):7.3f}")

    # 遠場當零點對照:兩個模型在那裡都該貼著 0
    rAf = Dfar - sky.T @ cA[:, far].mean(axis=1)
    rBf = Dfar - s_hat[far].mean() * sky[0] - sky[1:].T @ cB[:, far].mean(axis=1)
    print(f"\n遠場(>30 px, {int(far.sum()):,} 格)的殘差 —— 零點對照:")
    print(f"  A 中位 {np.median(rAf[bm]):+.4f}  rms {np.sqrt(np.mean(rAf[bm]**2)):.4f}")
    print(f"  B 中位 {np.median(rBf[bm]):+.4f}  rms {np.sqrt(np.mean(rBf[bm]**2)):.4f}")

    # ---- 保留了多少:連續譜 vs 發射線 -------------------------------------
    # 連續譜的量:在沒有強天光線也沒有源發射線的乾淨窗口取中位數
    win = [(5250, 5450), (5600, 5750), (6050, 6200)]
    wmask = np.zeros(nz, bool)
    for a, b in win:
        wmask |= (wl >= a) & (wl < b)
    print(f"\n連續譜保留率(在 {win} 這幾個乾淨窗口取中位數):")
    print(f"{'區域':>18}{'P (真值)':>10}{'A 現行':>10}{'B 場':>10}"
          f"{'A/P':>8}{'B/P':>8}")
    for nm, v in out.items():
        p, a, b = (np.median(v["P"][wmask]), np.median(v["A"][wmask]),
                   np.median(v["B"][wmask]))
        print(f"{nm:>18}{p:>10.3f}{a:>10.3f}{b:>10.3f}"
              f"{a/p if abs(p)>1e-6 else np.nan:>8.2f}"
              f"{b/p if abs(p)>1e-6 else np.nan:>8.2f}")

    # 發射線的量:[O III] 5007 紅移後扣掉鄰近連續譜
    lam = 5006.8 * (1 + Z_HARO)
    core = (wl > lam - 4) & (wl < lam + 4)
    side = ((wl > lam - 30) & (wl < lam - 10)) | ((wl > lam + 10) & (wl < lam + 30))
    print(f"\n[O III] 5007 紅移到 {lam:.1f} A 的線強度(扣鄰近連續譜):")
    print(f"{'區域':>18}{'P (真值)':>10}{'A 現行':>10}{'B 場':>10}"
          f"{'A/P':>8}{'B/P':>8}")
    for nm, v in out.items():
        f = lambda x: float(x[core].mean() - np.median(x[side]))
        p, a, b = f(v["P"]), f(v["A"]), f(v["B"])
        print(f"{nm:>18}{p:>10.3f}{a:>10.3f}{b:>10.3f}"
              f"{a/p if abs(p)>1e-6 else np.nan:>8.2f}"
              f"{b/p if abs(p)>1e-6 else np.nan:>8.2f}")
    print("  預期:兩者都接近 1.00 —— 天空模型沒有任何成分長得像這條線,扣不掉它。")
    print("  對照組成立的話,連續譜那張表的差異就確實來自 s 的處理方式。")

    # ---- 圖 ---------------------------------------------------------------
    def sm(y):
        k = args.smooth
        return y if k <= 1 else np.convolve(y, np.ones(k) / k, mode="same")

    keys = [k for k in ("other 1-3 px", "Haro 11 1-3 px", "Haro 11 10-25 px")
            if k in out]
    fig, ax = plt.subplots(len(keys), 2, squeeze=False,
                           figsize=(17, 3.6 * len(keys)),
                           gridspec_kw=dict(width_ratios=[2.2, 1]))
    for row, nm in enumerate(keys):
        v = out[nm]
        for col, (lo, hi) in enumerate([(4750, 9300), (5050, 5200)]):
            a_ = ax[row][col]
            m = (wl >= lo) & (wl < hi)
            a_.axhline(0, color="0.6", lw=0.8)
            a_.plot(wl[m], sm(v["P"])[m], lw=1.1, color="0.35",
                    label="$P$ = raw ring − raw far field  (no fitting)")
            a_.plot(wl[m], sm(v["A"])[m], lw=1.0, color="#1f77b4",
                    label="residual A — $s$ free (current)")
            a_.plot(wl[m], sm(v["B"])[m], lw=1.0, color="#d62728",
                    label="residual B — $s$ from the field")
            for lab, l0 in LINES:
                lz = l0 * (1 + Z_HARO)
                if lo <= lz < hi:
                    a_.axvline(lz, color="#2ca02c", ls=":", lw=0.9)
                    a_.text(lz, a_.get_ylim()[1], lab, rotation=90, fontsize=6,
                            va="top", ha="right", color="#2ca02c")
            a_.set_xlabel("wavelength [A]")
            if col == 0:
                a_.set_ylabel(f"{nm}\n({v['n']} spaxels)", fontsize=9)
                if row == 0:
                    a_.legend(fontsize=7.5, loc="upper left")
            else:
                a_.set_title("zoom: redshifted [O III] 5007", fontsize=9)
            a_.grid(alpha=0.25)
    fig.suptitle("residual spectrum in the ring around sources   —   "
                 "grey = what is really there,  blue = current,  red = $s$ from "
                 f"the spatial field   (smoothed {args.smooth} px for display)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = fig_dir / "07_residual_spectrum.png"
    fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {p}")

    # ---- 遠場:對照組 ------------------------------------------------------
    # 那裡只有天空,所以殘差**應該**貼著 0。這是「殘差小 = 好」唯一成立的地方,
    # 也是判斷「B 有沒有把天空扣壞」的唯一依據。
    skyA = sky.T @ cA[:, far].mean(axis=1)
    skyB = s_hat[far].mean() * sky[0] + sky[1:].T @ cB[:, far].mean(axis=1)
    print(f"\n遠場逐波段(>30 px,{int(far.sum()):,} 格):")
    print(f"{'波段':>12}{'原始資料':>10}{'A 殘差中位':>12}{'A 殘差 rms':>12}"
          f"{'B 殘差中位':>12}{'B 殘差 rms':>12}{'扣掉了':>9}")
    print("-" * 79)
    for lo in range(4700, 9300, 500):
        m = (wl >= lo) & (wl < lo + 500)
        if m.sum() < 20:
            continue
        print(f"{f'{lo}-{lo+500}':>12}{np.median(Dfar[m]):>10.1f}"
              f"{np.median(rAf[m]):>12.4f}{np.sqrt(np.mean(rAf[m]**2)):>12.4f}"
              f"{np.median(rBf[m]):>12.4f}{np.sqrt(np.mean(rBf[m]**2)):>12.4f}"
              f"{100*(1 - np.sqrt(np.mean(rBf[m]**2))/np.sqrt(np.mean(Dfar[m]**2))):>8.2f}%")

    href = out.get("Haro 11 1-3 px")
    fig, ax = plt.subplots(2, 2, figsize=(17, 8.4))

    a_ = ax[0][0]
    a_.semilogy(wl, np.maximum(Dfar, 1e-2), lw=0.6, color="0.25",
                label="raw data (sky included)")
    a_.semilogy(wl, np.maximum(skyB, 1e-2), lw=0.6, color="#d62728", alpha=0.7,
                label="sky model B")
    a_.set_ylabel("flux (log)"); a_.set_xlabel("wavelength [A]")
    a_.set_title("what we subtract in the far field — the OH forest dominates "
                 "beyond ~7500 A", fontsize=10)
    a_.legend(fontsize=8); a_.grid(alpha=0.25)

    # 右上:和 Haro 11 環那張圖**同一個 y 尺度**,好直接比較「有多平」
    a_ = ax[0][1]
    a_.axhline(0, color="0.6", lw=0.8)
    if href is not None:
        a_.plot(wl, sm(href["P"]), lw=0.9, color="0.55",
                label="(for scale) $P$ in the Haro 11 1-3 px ring")
    a_.plot(wl, sm(rAf), lw=1.0, color="#1f77b4", label="far-field residual A")
    a_.plot(wl, sm(rBf), lw=1.0, color="#d62728", label="far-field residual B")
    if href is not None:
        a_.set_ylim(-2.5, 7.5)
    a_.set_xlabel("wavelength [A]"); a_.set_ylabel("residual")
    a_.set_title("same y-scale as the source panels — the control is flat",
                 fontsize=10)
    a_.legend(fontsize=8); a_.grid(alpha=0.25)

    a_ = ax[1][0]
    a_.axhline(0, color="0.6", lw=0.8)
    a_.plot(wl, sm(rAf), lw=0.9, color="#1f77b4", label="residual A")
    a_.plot(wl, sm(rBf), lw=0.9, color="#d62728", label="residual B")
    a_.set_ylim(-0.6, 0.6)
    a_.set_xlabel("wavelength [A]"); a_.set_ylabel("residual")
    a_.set_title("zoomed in 10x — what is actually left", fontsize=10)
    a_.legend(fontsize=8); a_.grid(alpha=0.25)

    a_ = ax[1][1]
    m = (wl >= 8700) & (wl < 8900)
    a_.axhline(0, color="0.6", lw=0.8)
    a_.plot(wl[m], rAf[m], lw=0.9, color="#1f77b4", label="residual A (unsmoothed)")
    a_.plot(wl[m], rBf[m], lw=0.9, color="#d62728", label="residual B (unsmoothed)")
    b_ = a_.twinx()
    b_.plot(wl[m], Dfar[m], lw=0.7, color="0.6", alpha=0.6)
    b_.set_ylabel("raw sky (grey)", color="0.5", fontsize=8)
    b_.tick_params(axis="y", labelcolor="0.5", labelsize=7)
    a_.set_xlabel("wavelength [A]"); a_.set_ylabel("residual")
    a_.set_title("hardest place: the OH forest at 8700-8900 A", fontsize=10)
    a_.legend(fontsize=8); a_.grid(alpha=0.25)

    fig.suptitle(f"the control — far field (> 30 px from any source, "
                 f"{int(far.sum()):,} spaxels).  Only sky is there, so the "
                 f"residual should sit on zero.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p2 = fig_dir / "08_farfield_spectrum.png"
    fig.savefig(p2, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"saved -> {p2}")


if __name__ == "__main__":
    main()
