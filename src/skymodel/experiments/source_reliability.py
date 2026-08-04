"""37 個源裡,哪些的分類與紅移是可信的?

四個互相獨立的判準:

    連續譜 S/N   源本身夠不夠亮(從 step5 的 sky_subtracted 逐區域加總量)
    譜線 S/N     預期的發射線在不在。這是最強的判準,因為它獨立於擬合 ——
                 chi2 在任何 z 都找得到極小值,但只有線真的存在 S/N 才會高
    d(次佳組)    最佳組別贏次佳組多少。小 = 連「是星系還是 QSO」都分不出來
    d(遠處的 z)  離最佳解 0.05 以外,還有沒有差不多好的解。
                 小 = chi2(z) 的地形是平的,最小值落在哪裡是隨機的

判讀:
    譜線 S/N >= 5           紅移被譜線釘住,可信
    只有連續譜、沒有譜線     紅移只能靠平滑的連續譜形狀猜,不可信
                            (實測:換個 z 網格答案就變)

    conda run -n astro python src/skymodel/experiments/source_reliability.py
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
STEP04  = ROOT / "results/skymodel/step04"
STEP05  = ROOT / "results/skymodel/step05"
FIGURES = ROOT / "results/skymodel/figures"
WSKY    = ROOT / "data/Haro11_NEpointing_wsky.fits"

GROUP_COLOR = {"star": "#2ca02c", "galaxy": "#1f77b4", "qso": "#d62728"}
LINES = {"galaxy": [("[O II]", 3728.5), ("Hb", 4862.7),
                    ("[O III]", 5008.2), ("Ha", 6564.6)],
         "qso":    [("MgII", 2799.1), ("CIII]", 1908.7), ("Hb", 4862.7)],
         "star":   []}


def vac_to_air(l):
    s = (1e4 / l) ** 2
    return l / (1 + 8.336624212083e-5 + 2.408926869968e-2 / (130.1065924522 - s)
                + 1.599740894897e-4 / (38.92568793293 - s))


def main():
    ap = argparse.ArgumentParser(description="源的可信度排名")
    ap.add_argument("--tag", default="svd_K30_s_1.0")
    args = ap.parse_args()

    seg = fits.getdata(STEP01 / "seg.fits").ravel()
    wl  = np.load(STEP03 / "wavelength.npy")
    lm  = np.load(STEP03 / "iter_line_mask.npy")[0]
    b   = np.load(STEP04 / f"best_{args.tag}.npz")
    S   = fits.getdata(STEP05 / f"sky_subtracted_{args.tag}.fits")
    nz  = S.shape[0]
    S   = S.reshape(nz, -1)
    with fits.open(WSKY, memmap=True) as h:
        V = np.asarray(h["STAT"].data, np.float32).reshape(nz, -1)

    rows = []
    for i, t in enumerate(b["id"]):
        m = seg == t
        if m.sum() == 0:
            continue
        f = np.nansum(S[:, m], axis=1)
        var = np.nansum(V[:, m], axis=1)
        cont_sn = f[~lm].sum() / np.sqrt(var[~lm].sum())

        g, z = str(b["group"][i]), float(b["z"][i])
        line_sn, line_nm = 0.0, "-"
        for nm, l0 in LINES[g]:
            lo = vac_to_air(l0) * (1 + z)
            if not (wl[0] + 20 < lo < wl[-1] - 20):
                continue
            c = int(np.argmin(np.abs(wl - lo)))
            w = slice(c - 3, c + 4)
            sb = np.r_[np.arange(c - 30, c - 8), np.arange(c + 9, c + 31)]
            sb = sb[(sb >= 0) & (sb < nz)]
            e = np.sqrt(var[w].sum())
            if e > 0 and (f[w] - np.median(f[sb])).sum() / e > line_sn:
                line_sn = (f[w] - np.median(f[sb])).sum() / e
                line_nm = nm

        # 離最佳解 0.05 以外,最好的競爭解差多少
        sf = STEP04 / f"scan_id{t}_{args.tag}.npz"
        dz_far = np.nan
        if sf.exists():
            sc = np.load(sf)
            j = int(np.argmin(sc["chi2_all"]))
            far = np.abs(sc["z"] - sc["z"][j]) > 0.05
            if far.any():
                dz_far = float(sc["chi2_all"][far].min() - sc["chi2_all"][j])

        rows.append(dict(id=int(t), n=int(b["nspax"][i]), g=g, z=z,
                         cont=float(cont_sn), line=float(line_sn), lnm=line_nm,
                         dg=float(b["dchi2"][i]), dz=dz_far))

    rows.sort(key=lambda r: -r["line"])
    print(f"tag={args.tag}   {len(rows)} 個源\n")
    print(f"{'ID':>4}{'nspax':>7}{'group':>8}{'z':>9}{'連續譜S/N':>11}"
          f"{'最強線':>9}{'線S/N':>9}{'d(次佳組)':>12}{'d(遠處z)':>11}  判定")
    print("-" * 96)
    for r in rows:
        dz = "—" if not np.isfinite(r["dz"]) else f"{r['dz']:,.0f}"
        if r["line"] >= 5:
            v = "紅移由譜線釘住,可信"
        elif r["g"] == "star":
            v = "恆星,本來就沒有發射線"
        elif np.isfinite(r["dz"]) and r["dz"] > 1000:
            v = "無線但 chi2 地形陡,待查"
        else:
            v = "只有連續譜 → 紅移不可信"
        print(f"{r['id']:>4}{r['n']:>7}{r['g']:>8}{r['z']:>9.4f}{r['cont']:>11.1f}"
              f"{r['lnm']:>9}{r['line']:>9.1f}{r['dg']:>12,.0f}{dz:>11}  {v}")

    ok = sum(1 for r in rows if r["line"] >= 5)
    print(f"\n譜線 S/N >= 5 的有 {ok}/{len(rows)} 個 —— 只有這些的紅移可以拿去用。")

    # ---- 圖:譜線 S/N vs d(次佳組),兩個判準是否一致 ----
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    for g, c in GROUP_COLOR.items():
        s = [r for r in rows if r["g"] == g]
        if s:
            ax.scatter([max(r["dg"], 1) for r in s], [max(r["line"], 0.1) for r in s],
                       c=c, s=50, label=g, zorder=3)
            for r in s:
                ax.annotate(str(r["id"]), (max(r["dg"], 1), max(r["line"], 0.1)),
                            fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axhline(5, color="0.3", ls="--", lw=1.2, label="line S/N = 5")
    ax.axvline(1000, color="0.6", ls=":", lw=1.2, label="d(runner-up group) = 1000")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("chi2 margin over the runner-up group")
    ax.set_ylabel("S/N of the strongest expected emission line")
    ax.set_title("which sources have a trustworthy classification and redshift?",
                 fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = FIGURES / f"source_reliability_{args.tag}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
