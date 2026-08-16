"""天空連續譜真的只有一個自由度嗎?

現在的模型寫成 `s(x,y) · C_sky(λ)` —— **一個逐格的振幅,乘上一個全場共用的固定
形狀**。如果真實的天空連續譜連**形狀**都隨位置變,那個變化在模型裡無處可去。

s_flux_bias 查的正是這件事的徵候:blank 格扣完整套天空模型之後,殘差不是零,
而是一個沿波長的傾斜。而 s 被固定之後,模型裡唯一還能描述平滑形狀的就只剩源
模板 —— 於是那個傾斜被倒進源模板,變成源流量的加法偏差。

這支就問:那個傾斜是**天空的第二個自由度**嗎?

做法
----
    ① 用正式流程的做法解 blank(s 固定成場的值,只解天光線係數)
    ② 殘差 r = D − s·C_sky − Sum c_k L_k
    ③ **只留非天光線的通道** —— 我們問的是連續譜,不是線
    ④ 對 r 做 SVD,看有沒有一條有結構的成分

先驗證再相信
------------
一條 PCA 成分一定畫得出來,問題是它是不是雜訊。兩個獨立的檢查:

    形狀   把 blank 格隨機分兩半,各自做 SVD,比較第一條成分。雜訊的話兩半
           會完全不像。
    係數   把通道分奇偶兩半,各自算每一格的係數,比較。真的空間結構在兩半
           之間會一致;雜訊不會。

兩個都通過,才有資格說「天空連續譜有第二個自由度」。

    conda run -n astro python src/skymodel/experiments/sky_continuum_dof.py
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
from step5_fit_spaxels import fit_blank                   # noqa: E402
from utils import build_s_field, main_source_group, scale  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def first_mode(R, ncomp=4):
    """對 (n_chan, n_spaxel) 的殘差做 SVD,回傳 (成分, 每格的係數, 能量佔比)。"""
    U, sv, Vt = np.linalg.svd(R, full_matrices=False)
    frac = sv[:ncomp] ** 2 / (sv ** 2).sum()
    return U[:, :ncomp], (sv[:ncomp, None] * Vt[:ncomp]), frac


def main():
    ap = argparse.ArgumentParser(description="天空連續譜的自由度")
    ap.add_argument("--work", default="results/skymodel/p01")
    ap.add_argument("--run", default=None)
    ap.add_argument("--n-sample", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    W = ROOT / args.work
    run = (W / "step05" / args.run if args.run
           else sorted((W / "step05").glob("*_sfield"))[0])
    meta = json.loads((run / "meta.json").read_text())
    p = meta["s_field_params"]
    seg   = fits.getdata(W / "step01/seg.fits").astype(int)
    white = np.asarray(fits.getdata(W / "step01/whitelight.fits"), float)
    wl    = np.load(W / "step03/wavelength.npy")
    C_sky = np.load(W / "step03/sky_continuum.npy")
    basis = np.load(W / f"step03/sky_basis_{meta['basis']}_K{meta['K']}.npy")
    lmask = np.load(W / "step03/line_mask.npy")
    sky   = np.vstack([C_sky, basis])
    s_free = np.load(run / "s_free.npy").astype(float)
    valid = white != 0
    mg, _, _ = main_source_group(seg, np.where(valid, white, np.nan),
                                        W / "step04")
    blank = valid & (seg == 0) & np.isfinite(s_free)
    s_hat, train = build_s_field(s_free, seg, blank, p["r_far"],
                                    p["r_far_haro"], p["clip"],
                                main=mg)

    ys, xs = np.nonzero(blank)
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(ys.size, min(args.n_sample, ys.size), replace=False)
    ys, xs = ys[sel], xs[sel]
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    with fits.open(ROOT / meta["cube"], memmap=True) as h:
        D = np.asarray(h["DATA"].data[:, y0:y1, x0:x1], np.float32)[:, ys - y0, xs - x0]
    print(f"blank 抽樣 {ys.size:,} 格")

    # 正式流程的做法:s 固定成場的值,只解天光線係數
    c = fit_blank(D, sky, s_fix=s_hat[ys, xs])
    R = D - sky.T @ c
    cont = ~lmask & np.isfinite(C_sky)          # 只留非天光線的通道
    Rc = np.where(np.isfinite(R[cont]), R[cont], 0.0).astype(np.float64)
    print(f"連續譜通道 {int(cont.sum()):,} / {len(wl):,}"
          f"   殘差在這些通道的中位 {np.median(Rc):+.4f}"
          f"   逐格散布 {scale(Rc.ravel()):.4f}\n")

    U, A, frac = first_mode(Rc)
    print("殘差(只算連續譜通道)的 SVD:")
    for i in range(4):
        print(f"  成分 {i+1}: 佔能量 {100*frac[i]:5.1f}%")

    # --- 驗證 1:形狀 ---
    h1 = rng.permutation(ys.size)
    a_, b_ = h1[:ys.size // 2], h1[ys.size // 2:]
    U1, _, _ = first_mode(Rc[:, a_], 2)
    U2, _, _ = first_mode(Rc[:, b_], 2)
    r_shape = abs(float(np.dot(U1[:, 0], U2[:, 0])))
    print(f"\n驗證 1(形狀):把格分兩半各自做 SVD,第一條成分的內積 = {r_shape:.4f}")
    print(f"  (1 = 完全一致,0 = 毫無關係)")

    # --- 驗證 2:係數 ---
    ch = np.flatnonzero(cont)
    e, o = ch[0::2], ch[1::2]
    ce = np.linalg.lstsq(U[np.isin(ch, e), :1], R[e], rcond=None)[0][0]
    co = np.linalg.lstsq(U[np.isin(ch, o), :1], R[o], rcond=None)[0][0]
    g = np.isfinite(ce) & np.isfinite(co)
    r_coef = float(np.corrcoef(ce[g], co[g])[0, 1])
    print(f"驗證 2(係數):通道分奇偶各自算係數,兩半的相關 = {r_coef:+.4f}")
    print(f"  -> 係數的變異裡約 {100*max(r_coef,0):.0f}% 是真的空間結構")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for i in range(3):
        ax[0].plot(wl[cont], U[:, i] * np.sign(U[np.argmax(np.abs(U[:, i])), i]),
                   lw=0.7, label=f"comp {i+1}  ({100*frac[i]:.1f}%)")
    ax[0].plot(wl, C_sky / np.linalg.norm(C_sky), lw=1.2, color="k", ls="--",
               label="$C_{sky}$ (normalised)")
    ax[0].set_xlabel("wavelength [$\\AA$]"); ax[0].set_ylabel("shape")
    ax[0].grid(alpha=.25); ax[0].legend(fontsize=8)
    ax[0].set_title("SVD of the blank residual, continuum channels only", fontsize=10)

    cm = np.full(seg.shape, np.nan)
    cm[ys, xs] = A[0]
    v = 3 * scale(A[0])
    im = ax[1].imshow(cm, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v)
    plt.colorbar(im, ax=ax[1], fraction=.046)
    ax[1].contour(seg > 0, levels=[.5], colors="k", linewidths=.4)
    ax[1].set_title("coefficient of component 1", fontsize=10)

    ax[2].plot(ce[g], co[g], ".", ms=1.5, alpha=.3)
    lim = np.percentile(np.abs(ce[g]), 99)
    ax[2].plot([-lim, lim], [-lim, lim], "k-", lw=.8)
    ax[2].set_xlim(-lim, lim); ax[2].set_ylim(-lim, lim)
    ax[2].set_xlabel("coefficient, even channels")
    ax[2].set_ylabel("coefficient, odd channels")
    ax[2].set_title(f"split-half of the coefficient   r = {r_coef:+.3f}", fontsize=10)
    ax[2].grid(alpha=.25)
    fig.suptitle(f"does the sky continuum have a second degree of freedom?   "
                 f"{args.work}", fontsize=12)
    fig.tight_layout()
    o_ = W / "figures/sky_continuum_dof.png"
    o_.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(o_, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {o_}")


if __name__ == "__main__":
    main()
