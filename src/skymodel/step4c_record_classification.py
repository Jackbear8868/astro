"""把分類結果定案成一份檔案 —— 教授流程第 2 步的輸入。

分類本身已經由 step4b 決定 —— 它在同一組通道上讓恆星模板與星系本徵譜競爭,
reduced chi2 低的勝出,結果連同兩組的冠軍值寫在 best_*.npz 裡。這支不重算,
只讀那份檔案。

    (歷史:step4b 原本是兩段式,第 1 段用絕對門檻 reduced chi2 < 2.0 問
    「像不像恆星」,而沒有任何源通得過,於是 group 永遠是 galaxy;真正的判定
    只好在這裡事後手算。2026-08-08 把那個比較搬回 step4b,本檔的重算就移除了
    —— 同樣的判定寫兩份,改了一邊就會靜靜地不一致。)

所以本檔現在只做一件事:**決定哪些源進清單**,並把結果定案成 step5 的輸入。

**檔案裡的每一個數字都來自擬合,沒有任何手動指定的值。**唯一寫死的外部
決定是「哪些源進清單」(FIT_IDS),那和 KEEP_IDS 一樣是取捨,不是量出來的
東西 —— 把擬合出來的參數手動改掉,會讓下游的產物再也無法從 pipeline 重現。

    FIT_IDS 的由來
    教授 2026-08-06 從 37 個 SExtractor 偵測裡指出 11 個真源;2026-08-07 再
    指示把 #27 和 #30 移除 —— 前者在 4700-8000 內沒有可判別的特徵(強線都在
    8000 A 之外),後者光譜太吵,兩個的紅移都沒有被資料決定,放進源模型只會
    把錯的形狀帶進天空的擬合。

    #12 的注意事項
    它有兩個幾乎等高的解(z = 0.1546 與 1.033,reduced chi2 差 3%),教授認為
    後者才對、只是這份資料的 S/N 不足以排除前者。這裡照擬合結果寫 0.1546。
    它只有 29 個 spaxel(全視場的 0.03%),對整個 cube 的天空模型影響可忽略;
    要確認的話,用 --z-override 12=1.033 再跑一次,比較兩份天空模型即可。

輸出的欄位名和 step4 的 best_*.npz 完全一致(id / group / template / z / A),
所以 step5_fit_spaxels.py 只要換讀檔路徑就能直接吃。

    conda run -n astro python src/skymodel/step4c_record_classification.py \\
        --spec-dir results/skymodel/step02_eso --s-fix 0 \\
        --star-window 4700 8000 --gal-window 4700 8000 --iter 1
"""
import argparse
from pathlib import Path

import numpy as np

from step4b_window_fit import (STAR_WINDOW, GAL_WINDOW, KEEP_IDS, N_SRC,
                               make_tag, STEP04B)

# 進入源模型的源。KEEP_IDS 的 11 個扣掉教授 2026-08-07 指示移除的兩個。
FIT_IDS = tuple(i for i in KEEP_IDS if i not in (27, 30))


def main():
    ap = argparse.ArgumentParser(description="把分類結果定案成一份檔案")
    ap.add_argument("--star-window", type=float, nargs=2, default=STAR_WINDOW)
    ap.add_argument("--gal-window", type=float, nargs=2, default=GAL_WINDOW)
    ap.add_argument("--basis", default="svd")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--sky-basis", action="store_true")
    ap.add_argument("--aperture", action="store_true")
    ap.add_argument("--raw-mask", action="store_true")
    ap.add_argument("--iter", type=int, default=1)
    ap.add_argument("--s-fix", type=float, default=0.0)
    ap.add_argument("--spec-dir", default=None)
    ap.add_argument("--gal-model", choices=["eigen", "sdss"], default="eigen")
    ap.add_argument("--ids", type=int, nargs="+", default=list(FIT_IDS),
                    help="要放源模型的 ID,預設 FIT_IDS")
    ap.add_argument("--z-override", nargs="*", default=[], metavar="ID=Z",
                    help="把某個源的紅移改成指定值,只用來做敏感度測試 —— "
                         "正式的記錄檔不該有手動指定的數字。振幅會在該 z 上"
                         "重新取最佳解,不會沿用原本的")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    over = {int(k): float(v) for k, v in (s.split("=") for s in args.z_override)}

    suffix = (f"_{Path(args.spec_dir).name.replace('step02', '')}"
              if args.spec_dir else "") \
             + ("_galtpl" if args.gal_model == "sdss" else "")
    tag = make_tag(args.basis, args.K, args.s_fix, args.star_window,
                   args.gal_window, args.sky_basis, args.iter,
                   not args.raw_mask, args.aperture, suffix)

    # 分類本身由 step4b 決定 —— 它在同一組通道上比兩組模型的 reduced chi2,
    # 結果連同兩組的冠軍值一起寫在 best_*.npz 裡。這裡不再重算一次:
    # 同樣的判定寫兩份,改了一邊就會靜靜地不一致,而那種錯誤從輸出看不出來。
    # 本檔剩下的工作只有一件 —— 決定哪些源進清單(args.ids)。
    bpath = STEP04B / f"best_{tag}.npz"
    if not bpath.exists():
        raise SystemExit(f"找不到 {bpath.name},請先跑 step4b")
    best = np.load(bpath)
    idx  = {int(i): k for k, i in enumerate(best["id"])}

    rows = []
    print(f"{'ID':>4}{'class':>8}{'template':>10}{'z':>10}"
          f"{'star X2':>10}{'gal X2':>10}{'margin':>9}")
    print("-" * 61)
    for t in args.ids:
        if t not in idx:
            print(f"{t:>4}   best 檔裡沒有這個源,跳過")
            continue
        k = idx[t]
        group, tpl = str(best["group"][k]), str(best["template"][k])
        z, A = float(best["z"][k]), np.asarray(best["A"][k], float)
        r1, r2 = float(best["star_red_chi2"][k]), float(best["gal_red_chi2"][k])

        if t in over:
            # 覆寫紅移時,振幅要取「在那個 z 上的最佳解」,不能沿用原本的 ——
            # 模板形狀隨 z 變,振幅不通用。掃描已經在每個 z 解過,查表即可。
            s2 = np.load(STEP04B / f"scan2_id{t}_{tag}.npz")
            j  = int(np.argmin(np.abs(s2["z"] - over[t])))
            group, tpl = "galaxy", str(s2["template"][j])
            z, A = float(s2["z"][j]), np.asarray(s2["A"][j], float)

        a = np.full(N_SRC, np.nan)
        a[:len(A)] = A
        rows.append(dict(id=t, group=group, template=tpl, z=z, A=a))
        mark = "  <- overridden" if t in over else ""
        print(f"{t:>4}{group:>8}{tpl:>10}{z:>10.4f}{r1:>10.2f}{r2:>10.2f}"
              f"{max(r1, r2) / min(r1, r2):>8.2f}x{mark}")

    if not rows:
        raise SystemExit("沒有任何源,不寫檔")

    out = Path(args.out) if args.out else STEP04B / f"classification_{tag}.npz"
    np.savez(out,
             id=np.array([r["id"] for r in rows]),
             group=np.array([r["group"] for r in rows]),
             template=np.array([r["template"] for r in rows]),
             z=np.array([r["z"] for r in rows]),
             A=np.vstack([r["A"] for r in rows]))
    ns = sum(1 for r in rows if r["group"] == "star")
    print(f"\n{len(rows)} 個源:{ns} 恆星 / {len(rows) - ns} 星系"
          f"   (KEEP_IDS 的 {len(KEEP_IDS)} 個扣掉 "
          f"{', '.join(f'#{i}' for i in KEEP_IDS if i not in args.ids)})")
    print("margin = 兩個模型的 reduced chi2 比值,越接近 1 代表分類越沒有裕度")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
