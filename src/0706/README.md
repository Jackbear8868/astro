# src/0706 — ZAP 扣天空 pipeline（處理 / 評估 分離版）

一條線 + 一個設定檔。**處理(process)** 產出 fits，**評估(evaluate)** 拿 fits 畫圖/算分數，兩半不混。
所有路徑與物理參數只在 `settings.py` 定義一次；每支程式都 import 它，改參數只改那一份。

> 一律**從專案根目錄**執行，用 conda env `astro`；zap 那步會自動把 `libs/zap` 加進 path。

## 資料流

```
── 處理 process（產出大 fits）──────────────────────────────────
  data/Haro11_{cube}.fits            {cube} = nosky / wsky；整張全視場 499×559，不空間裁切

  step1_mask  <from_cube> [method]   建 source mask（method 預設 sep，另一個 claude）
     └►  results/zap/masks/<method>_from-<from_cube>/mask.fits   (1=源, 0=可用天空；餵 step2)
         results/zap/masks/<method>_from-<from_cube>/blanks.npz  (亮源座標 sy,sx；供 M3)
         每個 (方法 × cube) 各自一個資料夾，彼此不覆蓋。

  step2_zap   <target> <mask_from>   ZAP 對象 × mask 來源（2×2 = 4 次）
     └►  results/zap/cubes/<target>_maskfrom-<mask_from>/{zap,sky,var}.fits

── 評估 evaluate（讀 fits，逐波長對「所有 blank spaxel」統計）─────
  eval_common.py                     共用：blank = valid(FoV內) & ~source；逐波長 mean/median
     │                               （不取樣、用全部 blank；結果快取 results/zap/blankstats/）
     ├─ eval_m1_muse  <target> <mask_from> ─►  <run>/fig_M1_muse.png
     └─ eval_m1_zap   <target> <mask_from> ─►  <run>/fig_M1_zap_mean.png（±40）+ _median.png（±8）
```

## 怎麼跑

**處理階段**（產出 fits；step2 貴，約 30 分鐘/次）：
```bash
# step1：建 source mask（從 nosky 與 wsky 各建一張，便宜；預設 sep，加 claude 換方法）
conda run -n astro python src/0706/step1_mask.py nosky
conda run -n astro python src/0706/step1_mask.py wsky
# conda run -n astro python src/0706/step1_mask.py nosky claude   # 對照方法（robust-MAD）

# step2：ZAP，2×2 四次（<target> <mask_from>，可加 --ncpu N）
conda run -n astro python src/0706/step2_zap.py wsky  nosky --ncpu 16
conda run -n astro python src/0706/step2_zap.py wsky  wsky  --ncpu 16
conda run -n astro python src/0706/step2_zap.py nosky nosky --ncpu 16
conda run -n astro python src/0706/step2_zap.py nosky wsky  --ncpu 16
```

**評估階段 · M1 天空殘餘圖**（讀 fits，第一次跑會建 blank 統計快取，約 1–2 分鐘；之後瞬間）：
```bash
# MUSE convention（W20 Fig 15）：median 殘餘 vs ±1/5/10% 天空包絡 + 5577/OH zoom
conda run -n astro python src/0706/eval_m1_muse.py wsky nosky

# ZAP convention（Soto+2016 Fig 1）：標準 vs ZAP 雙 panel；分兩張圖 mean / median
conda run -n astro python src/0706/eval_m1_zap.py  wsky nosky
```
> 把上面兩行的 `wsky nosky` 換成任一個 run（如 `wsky wsky`、`nosky nosky`）即可畫該 run 的 M1。

## 每支程式的輸入 / 輸出

| 檔案 | 階段 | 讀 | 寫 |
|---|---|---|---|
| `settings.py` | 設定 | — | （只定義路徑 + 物理參數，不執行） |
| `step1_mask.py` | 處理 | `data/<from_cube>` `[sep\|claude]` | `<method>_from-<from_cube>/mask.fits`, `.../blanks.npz` |
| `step2_zap.py` | 處理 | `data/<target>` + `sep_from-<mask_from>/mask.fits` | `<run>/{zap,sky,var}.fits` |
| `eval_mask_compare.py` | 評估(實驗) | nosky+wsky cube | `<method>_from-<cube>/mask.fits`（×4）, `fig_mask_compare_2x2.png` |
| `eval_common.py` | 評估(共用) | `<run>/zap.fits`, nosky/wsky raw, mask | `blankstats/*.npz`（快取，被下面兩支共用） |
| `eval_m1_muse.py` | 評估 | 同上 | `<run>/fig_M1_muse.png` |
| `eval_m1_zap.py` | 評估 | 同上 | `<run>/fig_M1_zap_mean.png`（y 固定 ±40）, `<run>/fig_M1_zap_median.png`（±8）；右軸天空 0–1500 |

`<run>` = `results/zap/cubes/<target>_maskfrom-<mask_from>/`。

## M1 的統計是怎麼算的

- **逐波長、對所有 blank spaxel 收斂成一個值** → 得到一條殘餘光譜（y vs λ）。
- **blank = valid & ~source**：只取遮罩外（非源）**且**在視場內的 spaxel。
  ⚠ FoV 外的 spaxel 在 raw cube 是 NaN，但 ZAP 會把它們填成有限值（≈0），
  必須用 valid（raw 全波長 nansum≠0）排除，否則約 2.8 萬個邊緣 spaxel 會把中位數往 0 拉偏。
- **不取樣**：用全部 ~15 萬個 valid blank，曲線更精準、可重現（step1 也不再存取樣，blanks.npz 只留亮源座標供 M3）。
- **mean vs median**：median 對離群 spaxel 穩健（W20 用 median）；mean 是 Soto 原文統計量，
  但在大遮罩上被離群值主導、較雜。ZAP 圖 mean/median 各畫一張（±40 / ±8）分開看；
  nosky 已被 MUSE 扣過天空，mean≈median（僅差 ~0.1），差異主要在 ZAP 側。

## nosky vs wsky

zap / 評估的**程式完全相同**（只差吃哪個 cube / 哪張 mask），用參數化，不重複寫。
四個 run = `{nosky, wsky}(target) × {nosky, wsky}(mask_from)`；判讀基準是 **nosky raw = MUSE 標準處理**
（在 M1 圖裡就是左 panel / 藍線）。ZAP 正確用法是餵還含天空的 **wsky**；nosky 也跑，兩顆同等對待、一起比較。

> 物理參數的更動＝科學決策，請先確認 result-preserving（見專案根 `CLAUDE.md` 原則 2）。
