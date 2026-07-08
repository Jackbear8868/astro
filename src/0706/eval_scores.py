"""
eval_scores — 印出「四方數值」比較表（nosky/wsky × raw/ZAP）。
  輸入： results/zap/npz/summ_nosky.npz, summ_wsky.npz, _cache.npz
  輸出： 終端機表格（不存檔）
  跑法： conda run -n astro python src/0706/eval_scores.py

  四欄 = nosky raw / nosky+ZAP / wsky raw / wsky+ZAP。
  三種指標：
    (1) 天空線殘餘：空白區中位光譜在 5577/6300/8400Å 的值 → 越接近 0 越乾淨。
    (2) 空白區雜訊：逐波長 std 的中位數 → 越小越好。
    (3) 源 Hα 通量：最亮 spaxel 的 Hα 積分通量 → 看源有沒有被吃掉。
  判讀基準：nosky raw 是 MUSE 真值；wsky+ZAP 應同時「天空線→0」且「源 Hα ≈ 真值」。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings
import numpy as np

def scores():
    wl = np.load(str(settings.CACHE_NPZ_PATH))["wl"]
    npz = {cube_name: np.load(str(settings.summary_npz_path(cube_name))) for cube_name in settings.CUBE_NAMES}
    at = lambda arr, lam: float(arr[int(np.argmin(abs(wl - lam)))])

    def haflux(spec):                                      # Hα 積分通量：線內 減 兩側連續譜
        (a1, b1), (a2, b2) = settings.HALPHA_FLUX_CONTINUUM
        li = (wl > settings.HALPHA_LINE_WINDOW[0]) & (wl < settings.HALPHA_LINE_WINDOW[1])
        cb = ((wl > a1) & (wl < b1)) | ((wl > a2) & (wl < b2))
        return float(np.nansum(spec[li] - np.nanmean(spec[cb])))

    # 收集存在的欄位：(欄名, med, std, srcspec)
    cols = []
    for cube_name in settings.CUBE_NAMES:
        for kind, lbl in (("raw", "raw"), ("zap", "+ZAP")):
            if f"{kind}_med" in npz[cube_name]:
                cols.append((f"{cube_name} {lbl}", npz[cube_name][f"{kind}_med"],
                             npz[cube_name][f"{kind}_std"], npz[cube_name][f"{kind}_srcspec"]))

    names = [c[0] for c in cols]
    print("\n===== 四方數值比較 (基準: nosky raw = MUSE 真值) =====")
    header = "  " + " ".join(f"{n:>12s}" for n in names)
    print(f"{'指標':<20s}{header}")
    for lam in settings.SKY_LINE_WAVELENGTHS:                    # (1) 天空線殘餘
        row = " ".join(f"{at(med, lam):>12.2f}" for _, med, _, _ in cols)
        print(f"{'sky '+str(lam)+'Å 殘餘':<20s}  {row}")
    row = " ".join(f"{float(np.median(std)):>12.2f}" for _, _, std, _ in cols)
    print(f"{'空白區中位 std':<20s}  {row}")             # (2) 雜訊
    row = " ".join(f"{haflux(src):>12.0f}" for _, _, _, src in cols)
    print(f"{'源 Hα 通量':<20s}  {row}")                 # (3) 源保真

    # 重點結論：wsky+ZAP 的 Hα 相對真值保留率
    truth = dict((c[0], c) for c in cols).get("nosky raw")
    wz = dict((c[0], c) for c in cols).get("wsky +ZAP")
    if truth and wz:
        keep = 100 * haflux(wz[3]) / haflux(truth[3])
        print(f"\n  → wsky+ZAP 源 Hα 保留率 = {keep:.1f}% (相對 nosky raw 真值)")

if __name__ == "__main__":
    scores()
