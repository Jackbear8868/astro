"""遠場殘差的完整量測 —— 連續譜、天光線、分波段,並且和光子雜訊底比較。

為什麼要有這支
--------------
`check_pointing` 的 `far` 只量三個窄窗口(5250-5450、5600-5750、6050-6200)的
連續譜。兩個問題:

    ① 不具代表性。三個窄窗口涵蓋不了整條光譜,殘差在波長上若有結構就量不到。
    ② **完全沒有量天光線。** 而天光線正是天空扣除最難的部分。

指標
----
遠場的格沒有源,所以殘差的理想值是 0,剩下的只該是光子雜訊。因此除了 rms 之外,
**一律再除以該通道的期望雜訊**:

    期望雜訊(λ) = sqrt( <STAT(λ)> / N )        N 個遠場格取平均之後的雜訊

    散布 / 期望雜訊  ->  1   已經到雜訊底,模型沒有留下結構
                     ->  >1  模型還有系統性誤差

散布用**穩健**的 (p84 − p16)/2,不用 rms —— 光譜兩端與少數壞通道會有極端值,
rms 會被它們綁架。

這個歸一化很重要:天光線通道的光子雜訊本來就大得多,不歸一化的話,兩個波段的
rms 根本不能比。

    conda run -n astro python src/skymodel/experiments/far_field_spectrum.py \\
        --runs blank_svdK30_tpl68_sfield zz_rfar15
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
from utils import main_source_group, scale  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
BANDS = ((4600, 5500), (5500, 6500), (6500, 7500), (7500, 9350))


def far_mask(W, edge_margin=15, d_all=30, d_main=110):
    seg = fits.getdata(W / "step01/seg.fits").astype(int)
    white = np.asarray(fits.getdata(W / "step01/whitelight.fits"), float)
    valid = white != 0
    mg, _, _ = main_source_group(seg, np.where(valid, white, np.nan),
                                        W / "step04")
    return ((seg == 0) & valid
            & (ndimage.distance_transform_edt(valid) > edge_margin)
            & (ndimage.distance_transform_edt(seg == 0) > d_all)
            & (ndimage.distance_transform_edt(~mg) > d_main))


def mean_spec(path, mask, nz, hdu_name=None):
    """遠場格的平均光譜。分塊讀,不要把整顆 cube 拉進記憶體。"""
    ys, xs = np.nonzero(mask)
    acc = np.zeros(nz); cnt = np.zeros(nz)
    with fits.open(path, memmap=True) as h:
        d = h[hdu_name] if hdu_name and hdu_name in h else h[0]
        if d.data is None:
            d = h["DATA"]
        for y0 in range(ys.min(), ys.max() + 1, 40):
            m = (ys >= y0) & (ys < y0 + 40)
            if not m.any():
                continue
            blk = np.asarray(d.data[:, y0:y0 + 40, :], np.float32)
            v = blk[:, ys[m] - y0, xs[m]]
            acc += np.nansum(v, axis=1); cnt += np.isfinite(v).sum(axis=1)
    return acc / np.maximum(cnt, 1)


def main():
    ap = argparse.ArgumentParser(description="遠場殘差:連續譜 + 天光線 + 分波段")
    ap.add_argument("--work", default="results/skymodel/p01")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--cube", default="data/wshy/DATACUBE_FINAL_1.fits")
    args = ap.parse_args()

    W = ROOT / args.work
    wl = np.load(W / "step03/wavelength.npy")
    lmask = np.load(W / "step03/line_mask.npy")
    far = far_mask(W)
    nz = len(wl)
    print(f"遠場 {int(far.sum()):,} 格   天光線通道 {int(lmask.sum()):,} / {nz:,}")

    # 期望雜訊:N 個格取平均之後,每個通道的 sigma
    ys, xs = np.nonzero(far)
    with fits.open(ROOT / args.cube, memmap=True) as h:
        acc = np.zeros(nz); cnt = np.zeros(nz)
        for y0 in range(ys.min(), ys.max() + 1, 40):
            m = (ys >= y0) & (ys < y0 + 40)
            if not m.any():
                continue
            v = np.asarray(h["STAT"].data[:, y0:y0 + 40, :],
                           np.float32)[:, ys[m] - y0, xs[m]]
            acc += np.nansum(v, axis=1); cnt += np.isfinite(v).sum(axis=1)
    sig = np.sqrt(acc / np.maximum(cnt, 1) ** 2)      # sqrt(sum var)/N

    sp = {r: mean_spec(W / f"step05/{r}/sky_subtracted.fits", far, nz)
          for r in args.runs}
    cl = ~lmask

    print(f"\n{'run':<30}{'連續譜 偏差':>13}{'連續譜 散布':>13}{'/雜訊':>8}"
          f"{'天光線 散布':>13}{'/雜訊':>8}")
    print("-" * 84)
    for r in args.runs:
        s = sp[r]
        g = np.isfinite(s)
        cc, ll = cl & g, lmask & g
        print(f"{r:<30}{np.median(s[cc]):>+13.4f}"
              f"{scale(s[cc]):>12.4f}"
              f"{scale(s[cc])/np.median(sig[cc]):>8.2f}"
              f"{scale(s[ll]):>12.4f}"
              f"{scale(s[ll])/np.median(sig[ll]):>8.2f}")

    print(f"\n分波段(穩健散布 / 期望雜訊):")
    print(f"{'run':<30}" + "".join(f"{f'{a}-{b}':>22}" for a, b in BANDS))
    print(f"{'':<30}" + "".join(f"{'連續譜':>11}{'天光線':>11}" for _ in BANDS))
    for r in args.runs:
        s = sp[r]; row = ""
        for a, b in BANDS:
            m = (wl >= a) & (wl < b) & np.isfinite(s)
            for sel in (m & cl, m & lmask):
                row += (f"{scale(s[sel])/np.median(sig[sel]):>11.2f}"
                        if sel.sum() > 10 else f"{'—':>11}")
        print(f"{r:<30}{row}")

    fig, ax = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    for r in args.runs:
        ax[0].plot(wl, sp[r], lw=.45, alpha=.85, label=r)
    ax[0].plot(wl, sig, lw=.8, color="k", ls="--", label="expected noise")
    ax[0].plot(wl, -sig, lw=.8, color="k", ls="--")
    ax[0].axhline(0, color="k", lw=.8); ax[0].set_ylim(-.6, .6)
    ax[0].set_ylabel("far-field residual"); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.25)
    ax[0].set_title("all channels", fontsize=10)
    for r in args.runs:
        s = np.where(cl, sp[r], np.nan)
        ax[1].plot(wl, s, lw=.6, alpha=.85, label=r)
    ax[1].axhline(0, color="k", lw=.8); ax[1].set_ylim(-.2, .2)
    ax[1].set_xlabel("wavelength [$\\AA$]"); ax[1].set_ylabel("residual")
    ax[1].grid(alpha=.25); ax[1].legend(fontsize=8)
    ax[1].set_title("continuum channels only (sky lines masked out)", fontsize=10)
    fig.suptitle(f"far-field residual   {args.work}   {int(far.sum()):,} spaxels",
                 fontsize=12)
    fig.tight_layout()
    o = W / "figures/far_field_spectrum.png"
    o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(o, dpi=140, bbox_inches="tight")
    print(f"\nsaved -> {o}")


if __name__ == "__main__":
    main()
