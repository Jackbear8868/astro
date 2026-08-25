"""sky basis 裡有沒有混進 Haro 11 自己的發射線?

blank 是用「白光影像上的 SExtractor 偵測」界定的。但 Haro 11 的電離氣體暈是
「只有發射線、沒有連續譜」的 —— 它對寬波段白光幾乎沒有貢獻,在光譜裡卻完整
存在,所以有可能整片落在 blank 裡。

若暈落在 blank 裡,SVD 會學到一條帶有紅移 Ha/[N II]/[O III] 的成分,然後從
真正的源身上減掉它。這是科學錯誤,不是精度問題,而且對殘差類的指標完全隱形
—— 殘差會變小,因為源被扣掉了。

判準:在 Haro 11 的發射線波長上,basis 有沒有超出局部散布的結構。
對照組是隨機挑的「既沒有天光線也沒有星系線」的波長 —— 沒有污染的話,
兩者的分布應該一樣。

    conda run -n astro python src/skymodel/experiments/basis_contamination.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT    = Path(__file__).resolve().parents[3]

FIGURES = ROOT / "results/skymodel/evaluation/sky_basis"

# Haro 11 的靜止空氣波長 (A)
GAL_LINES = [("Hb", 4861.3), ("[O III]4959", 4958.9), ("[O III]5007", 5006.8),
             ("[N II]6548", 6548.0), ("Ha", 6562.8), ("[N II]6583", 6583.5),
             ("[S II]6716", 6716.4), ("[S II]6731", 6730.8),
             ("[S III]9069", 9068.6)]

CORE, GAP, BG = 3, 8, 30      # 線心半寬 / 間隔 / 背景視窗半寬 (px)


def line_strength(B, c):
    """每條 basis 在通道 c 的線強度,以該處局部散布為單位。

    線心取 ±CORE 的和,背景取兩側 GAP..BG 的中位數;除以背景的散布,
    所以回傳值是「幾個 sigma」,不同 basis、不同波長之間可比。
    """
    nz = B.shape[1]
    lo, hi = max(0, c - BG), min(nz, c + BG + 1)
    sb = np.r_[np.arange(lo, c - GAP), np.arange(c + GAP + 1, hi)]
    if sb.size < 10 or c - CORE < 0 or c + CORE + 1 > nz:
        return np.zeros(B.shape[0])
    base = np.median(B[:, sb], axis=1)
    scat = np.maximum(1.4826 * np.median(np.abs(B[:, sb] - base[:, None]), axis=1), 1e-12)
    core = B[:, c - CORE:c + CORE + 1].sum(axis=1) - (2 * CORE + 1) * base
    return core / (scat * np.sqrt(2 * CORE + 1))


def main():
    ap = argparse.ArgumentParser(description="basis 是否含 Haro 11 的發射線")
    ap.add_argument("--work", default="results/skymodel/p01",
                    help="這顆 pointing 的工作區(底下有 step01/step03/step04)")
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--id", type=int, default=1, help="Haro 11 的 segmentation ID")
    args = ap.parse_args()

    W = ROOT / args.work
    STEP01, STEP03, STEP04 = W / "step01", W / "step03", W / "step04"

    wl = np.load(STEP03 / "wavelength.npy")
    IN = np.load(STEP03 / "iter_line_mask.npy")[0]
    B  = np.load(STEP03 / f"sky_basis_{args.basis}_K{args.K}.npy")
    B  = B / np.linalg.norm(B, axis=1, keepdims=True)
    nz = wl.size

    # 檔名裡的 tag 由 step4 的設定組成,與這支無關 —— 直接找目錄裡的 best
    bf = sorted(STEP04.glob("best_*.npz"))
    if not bf:
        raise SystemExit(f"{STEP04} 裡沒有 best_*.npz,先跑 step4_classify_sources.py")
    best = np.load(bf[0])
    j = int(np.flatnonzero(best["id"] == args.id)[0])
    z = float(best["z"][j])
    print(f"Haro 11 (ID {args.id}) 的紅移 z = {z:.5f}"
          f"   (cz = {z*299792.458:,.0f} km/s)")

    seg = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    tot = int((white != 0).sum())
    blank = int(((white != 0) & (seg == 0)).sum())
    print(f"有效 spaxel {tot:,}   blank {blank:,}   被遮罩的源 {tot-blank:,}"
          f"   ({100*(tot-blank)/tot:.1f}%)\n")

    # ---- 對照組:既非天光線、也非星系線的波長 ----
    gal_ch = []
    for nm, l0 in GAL_LINES:
        lo = l0 * (1 + z)
        if wl[0] + BG * 1.25 < lo < wl[-1] - BG * 1.25:
            gal_ch.append((nm, lo, int(np.argmin(np.abs(wl - lo)))))
    forbid = np.zeros(nz, bool)
    for _, _, c in gal_ch:
        forbid[max(0, c - BG):c + BG + 1] = True
    ctrl = np.flatnonzero(~IN & ~forbid)
    ctrl = ctrl[(ctrl > BG) & (ctrl < nz - BG)]
    rng = np.random.default_rng(0)
    ctrl = rng.choice(ctrl, min(400, ctrl.size), replace=False)
    S_ctrl = np.abs(np.array([line_strength(B, c) for c in ctrl]))
    thr = np.percentile(S_ctrl, 99.9)
    print(f"對照組({ctrl.size} 個無線通道 x {args.K} 條 basis)的 |強度| 分布:"
          f"中位 {np.median(S_ctrl):.2f}   99% {np.percentile(S_ctrl,99):.2f}"
          f"   99.9% {thr:.2f}   最大 {S_ctrl.max():.2f}\n")

    print(f"{'線':>12}{'觀測波長':>10}{'ch':>6}{'在天光線遮罩內?':>16}"
          f"{'最強 basis':>11}{'強度(sigma)':>13}{'判定':>10}")
    print("-" * 82)
    flagged = []
    for nm, lo, c in gal_ch:
        s = line_strength(B, c)
        k = int(np.argmax(np.abs(s)))
        hit = abs(s[k]) > thr
        if hit:
            flagged.append((nm, lo, c, k, s[k]))
        print(f"{nm:>12}{lo:>10.1f}{c:>6}{('是' if IN[c] else '否'):>16}"
              f"{k:>11}{s[k]:>13.2f}{('⚠ 超出對照' if hit else 'ok'):>10}")

    print(f"\n對照組的 99.9% 門檻 = {thr:.2f} sigma。超出者代表 basis 在那個波長上"
          f"有\n非隨機的結構,而該波長沒有天光線 —— 那結構只可能來自源。")

    # ---- 圖:最可疑的那條 basis 在星系線附近的樣子 ----
    if flagged:
        nm, lo, c, k, s = max(flagged, key=lambda r: abs(r[4]))
        FIGURES.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(11, 5))
        w = slice(max(0, c - 60), min(nz, c + 61))
        ax.plot(wl[w], B[k][w], lw=1.2, color="#1f77b4", label=f"basis #{k}")
        ax.axvline(lo, color="#d62728", ls="--", lw=1.2,
                   label=f"{nm} at z={z:.4f}")
        for l2, n2 in [(l * (1 + z), n) for n, l in GAL_LINES]:
            if wl[w][0] < l2 < wl[w][-1] and l2 != lo:
                ax.axvline(l2, color="#d62728", ls=":", lw=0.9, alpha=0.7)
        ax.fill_between(wl[w], -1, 1, where=IN[w], color="0.85", alpha=0.5,
                        transform=ax.get_xaxis_transform(), label="sky-line mask")
        ax.set_xlabel("wavelength (A)"); ax.set_ylabel("basis amplitude")
        ax.set_title(f"sky basis #{k} near Haro 11's {nm} — "
                     f"structure here cannot be airglow", fontsize=11)
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        fig.tight_layout()
        out = FIGURES / f"basis_contamination_{args.basis}_K{args.K}.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        print(f"\n{len(flagged)} 條線超出對照組門檻。saved -> {out}")
    else:
        print("\n沒有任何星系線超出對照組門檻 —— basis 未偵測到源污染。")


if __name__ == "__main__":
    main()
