"""STAT 有沒有低估雜訊 —— 用 blank 區的空間分格實測。

「我們的 STAT 比 ESO 小」不等於「我們的 STAT 錯了」。雙線性內插是平滑,
它**真的**會讓單一像素的變異數變小:

    Var(out) = Σ wₖ²·Var(in),   Σ wₖ = 1  =>  Σ wₖ² <= 1

代價是輸出的相鄰像素變得相關,大尺度上雜訊一點也沒少。所以該量的不是
「STAT 對不對」,而是**雜訊隨空間分格怎麼縮**。

做法:
    ① 只用 blank spaxel、只用沒有天光線的通道。
    ② 殘差取相鄰兩個通道的差
           r(λⱼ) = [D(λⱼ) − D(λⱼ₊₁)] / √2
       這一步自動消掉所有「空間上有結構、但波長上平滑」的東西 —— 天空連續譜、
       平場誤差、照明梯度。不這樣做的話那些東西會偽裝成相關性。
       它的預期變異數 pred = [V(λⱼ) + V(λⱼ₊₁)] / 2。
    ③ 分成 N×N 方格。若像素獨立,格內和的變異數就是格內 pred 的和,所以
           ratio(N) = ⟨(Σ_格 r)²⟩ / ⟨Σ_格 pred⟩
       獨立  -> ratio ≡ 1 對所有 N
       正相關 -> ratio 隨 N 上升,在相關長度之外趨於平台
    ④ 同時直接量空間自相關 ρ(Δ) = ⟨z(x)·z(x+Δ)⟩ / ⟨z²⟩,z = r/√pred。

判讀:
    ratio(1)              STAT 對「單一像素」準不準
    ratio(平台)           大尺度上的真實雜訊是 STAT 說的幾倍(這才是 chi2 要用的)
    ρ(1)                  相鄰像素的相關係數,內插的直接指紋

理論預期(雙線性、位移小數 fx):1 維權重 (1−fx, fx),
    Var 縮 (1−fx)² + fx²,  ρ(1) = fx(1−fx) / [(1−fx)² + fx²],
    大尺度不變 -> ratio(平台) / ratio(1) = 1 / {[(1−fx)²+fx²]·[(1−fy)²+fy²]}

    conda run -n astro python src/skymodel/experiments/step5b_stat_check.py
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[3]
STEP01  = ROOT / "results/skymodel/step01"
STEP03  = ROOT / "results/skymodel/step03"
FIGURES = ROOT / "results/skymodel/figures"

BINS = (1, 2, 3, 4, 6, 8)
LAGS = (1, 2, 3, 4)


def block_sum(a, N):
    """把 (ny, nx) 切成 N×N 方格求和,回傳 (ny//N, nx//N)。邊緣的餘數丟掉。"""
    ny, nx = a.shape[-2] // N * N, a.shape[-1] // N * N
    return a[..., :ny, :nx].reshape(a.shape[:-2] + (ny // N, N, nx // N, N)) \
            .sum(axis=(-3, -1))


def run_cube(path, blank, c0, c1, free, chunk=200):
    """回傳 ratio[N]、rho[lag] 與比對用的計數。

    blank : (ny, nx) 布林,這個 cube 自己格點上的 blank 遮罩
    c0,c1 : 使用的通道範圍
    free  : (c1-c0,) 布林,該通道是不是沒有天光線
    """
    acc_r2 = {N: 0.0 for N in BINS}      # Σ (格內 r 的和)²
    acc_pv = {N: 0.0 for N in BINS}      # Σ (格內 pred 的和)
    acc_n  = {N: 0 for N in BINS}
    cx = {d: 0.0 for d in (0,) + LAGS}   # Σ z(x)·z(x+d),沿 x
    cy = {d: 0.0 for d in (0,) + LAGS}
    nx_ = {d: 0 for d in (0,) + LAGS}
    ny_ = {d: 0 for d in (0,) + LAGS}

    with fits.open(path, memmap=True) as h:
        for j in range(c0, c1 - 1, chunk):
            k = min(j + chunk + 1, c1)                # +1:跨 chunk 的配對要用到
            D = np.asarray(h["DATA"].data[j:k], np.float64)
            V = np.asarray(h["STAT"].data[j:k], np.float64)
            for i in range(k - j - 1):
                if not (free[j - c0 + i] and free[j - c0 + i + 1]):
                    continue
                d0, d1 = D[i], D[i + 1]
                v0, v1 = V[i], V[i + 1]
                ok = (blank & np.isfinite(d0) & np.isfinite(d1)
                      & np.isfinite(v0) & np.isfinite(v1) & (v0 > 0) & (v1 > 0))
                if ok.sum() < 500:
                    continue
                r    = (d0 - d1) / np.sqrt(2.0)
                pred = (v0 + v1) / 2.0
                # 扣掉全場中位:天空連續譜有斜率,相鄰通道的差會留一個所有
                # spaxel 共同的小偏移,那不是雜訊。
                r = r - np.median(r[ok])
                r    = np.where(ok, r, 0.0)
                pred = np.where(ok, pred, 0.0)
                M    = ok.astype(np.float64)

                for N in BINS:
                    rs, ps, ms = block_sum(r, N), block_sum(pred, N), block_sum(M, N)
                    g = ms == N * N                   # 只要整格都有效的方格
                    acc_r2[N] += float((rs[g] ** 2).sum())
                    acc_pv[N] += float(ps[g].sum())
                    acc_n[N]  += int(g.sum())

                z = np.where(ok, r / np.sqrt(np.where(pred > 0, pred, 1.0)), 0.0)
                for d in (0,) + LAGS:
                    a, b = (z, z) if d == 0 else (z[:, :-d], z[:, d:])
                    m = (ok, ok) if d == 0 else (ok[:, :-d], ok[:, d:])
                    gg = m[0] & m[1]
                    cx[d] += float((a * b)[gg].sum()); nx_[d] += int(gg.sum())
                    a, b = (z, z) if d == 0 else (z[:-d], z[d:])
                    m = (ok, ok) if d == 0 else (ok[:-d], ok[d:])
                    gg = m[0] & m[1]
                    cy[d] += float((a * b)[gg].sum()); ny_[d] += int(gg.sum())

    ratio = {N: acc_r2[N] / acc_pv[N] if acc_pv[N] > 0 else np.nan for N in BINS}
    c0x = cx[0] / max(nx_[0], 1)
    rho_x = {d: (cx[d] / max(nx_[d], 1)) / c0x for d in LAGS}
    rho_y = {d: (cy[d] / max(ny_[d], 1)) / (cy[0] / max(ny_[0], 1)) for d in LAGS}
    return ratio, rho_x, rho_y, acc_n


def main():
    ap = argparse.ArgumentParser(description="STAT 的空間分格實測")
    ap.add_argument("--lo", type=float, default=5000.0)
    ap.add_argument("--hi", type=float, default=7000.0)
    ap.add_argument("--cubes", nargs="+", default=None,
                    help="NAME=PATH。省略時用預設的四個(ESO 合併品、我們的兩個"
                         "版本、單顆曝光)。單顆曝光要寫成 NAME=PATH=EXP 才會"
                         "套用 blank 遮罩的位移")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    wl   = np.load(STEP03 / "wavelength.npy")
    lm   = np.load(STEP03 / "line_mask.npy")
    seg  = fits.getdata(STEP01 / "seg.fits")
    wht  = fits.getdata(STEP01 / "whitelight.fits")
    blank = (seg == 0) & (wht != 0)

    c0 = int(np.searchsorted(wl, args.lo))
    c1 = int(np.searchsorted(wl, args.hi))
    free = ~lm[c0:c1]
    npair = int((free[:-1] & free[1:]).sum())
    print(f"通道 {c0}-{c1} ({args.lo:.0f}-{args.hi:.0f} A)   無天光線 {int(free.sum())}"
          f"   相鄰配對 {npair}")
    print(f"blank spaxel {int(blank.sum()):,}\n")

    # EXP1 在自己的格點上 —— 用量到的整數位移把 blank 遮罩搬過去。
    # merged(x,y) <- EXP1(x+ox, y+oy),所以 EXP1(X,Y) <- merged(X-ox, Y-oy)。
    OX, OY = -18, 0
    ny1, nx1 = fits.getheader(ROOT / "data/NEpointing_exps/"
                              "Haro11_NEpointing_EXP1_wsky.fits",
                              "DATA")["NAXIS2"], 315
    b1 = np.zeros((ny1, nx1), bool)
    Y, X = np.mgrid[0:ny1, 0:nx1]
    yy, xx = Y - OY, X - OX
    g = (yy >= 0) & (yy < blank.shape[0]) & (xx >= 0) & (xx < blank.shape[1])
    b1[g] = blank[yy[g], xx[g]]

    S5B = ROOT / "results/skymodel/step05b"
    if args.cubes:
        cubes = []
        for c in args.cubes:
            parts = c.split("=")
            cubes.append((parts[0], Path(parts[1]),
                          b1 if len(parts) > 2 and parts[2] == "EXP" else blank))
    else:
        cubes = [
            ("ESO merged",  ROOT / "data/Haro11_NEpointing_wsky.fits", blank),
            ("bilinear",    S5B / "merged_EXP123_wsky_bil_norej.fits", blank),
            ("integer",     S5B / "merged_EXP123_wsky_int_rej.fits", blank),
            ("EXP1 single", ROOT / "data/NEpointing_exps/"
                            "Haro11_NEpointing_EXP1_wsky.fits", b1),
        ]

    res = {}
    for name, path, bl in cubes:
        if not Path(path).exists():
            print(f"跳過 {name}:{path} 不存在\n"); continue
        print(f"--- {name} ---", flush=True)
        ratio, rx, ry, n = run_cube(path, bl, c0, c1, free)
        res[name] = (ratio, rx, ry)
        print("  N      ratio   noise scale   方格數")
        for N in BINS:
            print(f"  {N:<3}  {ratio[N]:8.4f}   x{np.sqrt(ratio[N]):6.3f}"
                  f"   {n[N]:>10,}")
        print("  rho_x " + "  ".join(f"{d}:{rx[d]:+.3f}" for d in LAGS))
        print("  rho_y " + "  ".join(f"{d}:{ry[d]:+.3f}" for d in LAGS))
        print(f"  平台/單像素 = {ratio[BINS[-1]] / ratio[1]:.3f}\n", flush=True)

    # ---------------- 圖 ----------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    col = {"ESO merged": "#000000", "bilinear": "#d62728", "integer": "#1f77b4",
           "EXP1 single": "#7f7f7f"}

    a = ax[0]
    for nm, (ratio, _, _) in res.items():
        a.plot(BINS, [ratio[N] for N in BINS], "o-", color=col.get(nm), label=nm, lw=1.2)
    a.axhline(1, color="0.5", lw=0.8, ls="--")
    a.set_xlabel("bin size N (N x N spaxels)")
    a.set_ylabel("observed var / STAT-predicted var")
    a.set_title("does the noise scale as STAT says?", fontsize=10)
    a.legend(fontsize=8); a.grid(alpha=0.25)

    for k, (lab, idx) in enumerate((("$\\rho$ along x", 1), ("$\\rho$ along y", 2))):
        a = ax[idx]
        for nm, (_, rx, ry) in res.items():
            rr = rx if idx == 1 else ry
            a.plot(LAGS, [rr[d] for d in LAGS], "o-", color=col.get(nm), label=nm, lw=1.2)
        a.axhline(0, color="0.5", lw=0.8, ls="--")
        a.set_xlabel("lag [pix]"); a.set_ylabel("correlation")
        a.set_title(f"spatial correlation of the noise, {lab[-1]}", fontsize=10)
        a.legend(fontsize=8); a.grid(alpha=0.25)

    fig.suptitle("STAT vs the actual noise, measured in blank spaxels "
                 f"({args.lo:.0f}-{args.hi:.0f} $\\AA$, line-free channels)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.out) if args.out else FIGURES / "step5b_stat_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=135, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
