"""從 step4 的掃描結果畫 chi2(z),判斷紅移的不確定度、有無別名。

用法:  python src/skymodel/step4_plot_scan.py --id 1 --basis svd
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path(__file__).resolve().parents[3]
STEP04 = ROOT / "results/skymodel/step04"

Z_KNOWN = 0.0206          # Haro 11 的文獻紅移
N_PARAM = 12              # 1 模板 + 1 連續譜 + 10 條天光線 basis


def main():
    ap = argparse.ArgumentParser(description="畫 step4 掃描的 chi2(z) 曲線")
    ap.add_argument("--id",    type=int, default=1)
    ap.add_argument("--basis", default="svd")
    ap.add_argument("--top",   type=int, default=6, help="畫出最好的幾條模板")
    args = ap.parse_args()

    d = np.load(STEP04 / f"scan_id{args.id}_{args.basis}.npz")
    tpl, z, chi2 = d["template"], d["z"], d["chi2"]
    n_good = int(d["n_good"])

    # --- 全域最佳解 ---
    i        = int(np.argmin(chi2))
    chi2_min = chi2[i]
    z_best   = z[i]
    tpl_best = tpl[i]

    dof    = n_good - N_PARAM
    red    = chi2_min / dof
    thresh = chi2_min + red            # 誤差放大後的 1-sigma 門檻

    print(f"best:  template {tpl_best}   z = {z_best:.5f}   chi2 = {chi2_min:.1f}")
    print(f"reduced chi2 = {red:.1f}   (dof = {dof})")

    # --- 最佳模板那條曲線,依 z 重新排序 ---
    m = tpl == tpl_best
    o = np.argsort(z[m])
    zb, cb = z[m][o], chi2[m][o]

    within = zb[cb <= thresh]
    half   = (within.max() - within.min()) / 2
    print(f"1-sigma z: {within.min():.5f} ~ {within.max():.5f}   "
          f"半寬 {half:.5f} = {half * 299792:.0f} km/s")
    print(f"文獻值 z = {Z_KNOWN},相差 {abs(z_best - Z_KNOWN) * 299792:.0f} km/s")

    # --- 挑出最好的幾條模板來畫 ---
    names = np.unique(tpl)
    show  = sorted(names, key=lambda n: chi2[tpl == n].min())[:args.top]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for name in show:
        m = tpl == name
        o = np.argsort(z[m])
        for ax in (a1, a2):
            ax.plot(z[m][o], chi2[m][o], lw=1.0, label=name)

    for ax, title in [(a1, "full scan"), (a2, "zoom")]:
        ax.axvline(Z_KNOWN, color="k", ls="--", lw=1, label="known z")
        ax.set_xlabel("z"); ax.set_ylabel("chi2"); ax.set_title(title)
    a1.legend(fontsize=8)

    a2.axhline(thresh, color="r", ls=":", lw=1, label="1-sigma")
    a2.set_xlim(z_best - 0.006, z_best + 0.006)
    a2.set_ylim(chi2_min - 0.2 * red * 30, chi2_min + red * 30)
    a2.legend(fontsize=8)

    fig.tight_layout()
    out = STEP04 / f"chi2_z_id{args.id}_{args.basis}.png"
    fig.savefig(out, dpi=140)
    print(f"saved {out}")


if __name__ == "__main__":
    main()