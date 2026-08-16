"""step4b 第 1 段的文字總表:每個源 × 每一輪遮罩的 template / z / A。

圖看得出趨勢,看不出「ID 35 在 iter2 換成哪一條模板」這種要逐格對照的事。
這支把結果攤成純文字,方便搜尋、貼進信件、和之後的版本 diff。

兩張表
    ① 逐源逐輪的 tpl / z / A          最主要的一張
    ② 穩定度摘要                       模板、型別、z、A 在四輪之間變多少

「型別」和「模板編號」要分開看:008 和 009 都是 G 型星,編號換了但結論沒變;
013(M5)換成 022(K subdwarf)才是結論真的變了。所以穩定度分別統計兩者。

輸出是英文的 —— 這份表會直接貼進給教授的信裡。

    conda run -n astro python src/skymodel/experiments/step4b_overview.py
    conda run -n astro python src/skymodel/experiments/step4b_overview.py --aperture
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from step4_fit_source import STAR_WINDOW, GAL_WINDOW, make_tag

ROOT    = Path(__file__).resolve().parents[3]
STEP02  = ROOT / "results/skymodel/step02"
STEP02B = ROOT / "results/skymodel/step02b"
STEP04B = ROOT / "results/skymodel/step04b"

STAR_TYPE = {
    0: "O", 1: "O/B", 2: "B", 3: "A", 4: "A", 5: "F/A", 6: "F", 7: "F",
    8: "G", 9: "G", 10: "K", 11: "M1", 12: "M3", 13: "M5", 14: "M8", 15: "L1",
    16: "magWD", 17: "C", 18: "C", 19: "C", 20: "WD", 21: "BWD", 22: "Ksd",
}


def main():
    ap = argparse.ArgumentParser(description="step4b 第 1 段的文字總表")
    ap.add_argument("--star-window", type=float, nargs=2, default=STAR_WINDOW)
    ap.add_argument("--gal-window", type=float, nargs=2, default=GAL_WINDOW)
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--sky-basis", action="store_true")
    ap.add_argument("--aperture", action="store_true")
    ap.add_argument("--raw-mask", action="store_true")
    ap.add_argument("--iters", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()

    D = {}
    for it in args.iters:
        tag = make_tag(args.basis, args.K, 1.0, args.star_window, args.gal_window,
                       args.sky_basis, it, not args.raw_mask, args.aperture)
        f = STEP04B / f"best_{tag}.npz"
        if not f.exists():
            print(f"iter{it}: 找不到 {f.name},跳過")
            continue
        D[it] = np.load(f)
    if not D:
        raise SystemExit("沒有任何結果檔。先跑 step4_fit_source.py")
    its = sorted(D)

    src   = STEP02B if args.aperture else STEP02
    nspax = np.load(src / "object_nspax.npy")
    sids  = np.load(src / "object_ids.npy")
    ids   = [int(i) for i in D[its[0]]["id"]]

    def get(it, t, key):
        d = D[it]
        j = int(np.flatnonzero(d["id"] == t)[0])
        return d[key][j]

    L = [f"step4b stage 1 — best stellar template per source, per sky-line mask "
         f"iteration",
         f"window {args.star_window[0]:.0f}-{args.star_window[1]:.0f} A (air), "
         f"{'circular aperture r = 6 px' if args.aperture else 'SExtractor segmentation footprint'}, "
         f"{'cumulative' if not args.raw_mask else 'independent'} masks",
         "",
         "Model:  flux(lambda) - 1.0 * C_sky(lambda)  =  A * T(lambda / (1+z)),  A >= 0",
         "One free parameter (A). Sky-line channels are excluded from chi2.",
         ""]

    # ---------- ① 逐源逐輪 ----------
    # 一個源一個區塊、一輪一行。各輪並排成一列會寬到必須換行,換行之後就
    # 完全讀不出哪個數字屬於哪一輪。
    L += ["[1] template / z / A for every source and every mask iteration", ""]
    for t in ids:
        n = int(np.median(nspax[int(np.flatnonzero(sids == t)[0])]))
        L.append(f"ID {t}   nspax {n}")
        for i in its:
            tp = int(get(i, t, "star_tpl"))
            L.append(f"  iter{i}  {tp:03d} {STAR_TYPE[tp]:<6}"
                     f" z= {float(get(i, t, 'z')):>8.4f}"
                     f"   A= {float(get(i, t, 'A')[0]):<11.5g}")
        L.append("")

    # ---------- ② 穩定度 ----------
    L += ["", "",
          "[2] stability across the mask iterations", "",
          "  tpl   : is the template NUMBER the same in all iterations?",
          "  type  : is the spectral TYPE the same? (008 and 009 are both G,",
          "          so a 008 -> 009 switch counts as stable)",
          "  z span: max(z) - min(z) over the iterations",
          "  A span: (max(A) - min(A)) / min(A)", ""]
    hdr = (f"{'ID':>4}{'nspax':>7}{'tpl':>6}{'type':>6}{'types seen':>24}"
           f"{'z span':>9}{'km/s':>8}{'A span':>10}")
    L += [hdr, "-" * len(hdr)]
    stab = []
    for t in ids:
        tps = [int(get(i, t, "star_tpl")) for i in its]
        tys = [STAR_TYPE[x] for x in tps]
        zs  = np.array([float(get(i, t, "z")) for i in its])
        As  = np.array([float(get(i, t, "A")[0]) for i in its])
        st, sy = len(set(tps)) == 1, len(set(tys)) == 1
        dA = (As.max() - As.min()) / As.min() if As.min() > 0 else np.inf
        zsp = float(np.ptp(zs))
        stab.append((t, st, sy, zsp, dA))
        n = int(np.median(nspax[int(np.flatnonzero(sids == t)[0])]))
        L.append(f"{t:>4}{n:>7}{'same' if st else 'VARY':>6}"
                 f"{'same' if sy else 'VARY':>6}"
                 f"{'/'.join(dict.fromkeys(tys)):>24}"
                 f"{zsp:>9.4f}{3e5*zsp:>8.0f}"
                 f"{'inf' if not np.isfinite(dA) else f'{100*dA:.0f}%':>10}")
    ns, nt = sum(s[1] for s in stab), sum(s[2] for s in stab)
    fin = [s[4] for s in stab if np.isfinite(s[4])]
    L += ["",
          f"  same template number in all {len(its)} iterations : {ns} / {len(ids)}",
          f"  same spectral type   in all {len(its)} iterations : {nt} / {len(ids)}",
          f"  median z span : {np.median([s[3] for s in stab]):.4f}"
          f"  ({3e5*np.median([s[3] for s in stab]):.0f} km/s, out of a "
          f"3000 km/s scan range)",
          f"  median A span : {100*np.median(fin):.0f}%"]

    tagbase = make_tag(args.basis, args.K, 1.0, args.star_window, args.gal_window,
                       args.sky_basis, 0, not args.raw_mask,
                       args.aperture).replace("_L0cum", "").replace("_L0raw", "")
    out = STEP04B / f"overview{tagbase.replace('nobasis_s1.0', '')}.txt"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
