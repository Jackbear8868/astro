# src/zap — ZAP 扣天空 pipeline

一條線 + 一個設定檔。**mask**（建/後製源遮罩）→ **zap**（扣天空）→ **eval_spectrum**（任選區域看殘餘 spectrum）。
所有路徑與物理參數只在 `settings.py` 定義一次；每支程式都 import 它，改參數只改那一份。

> 一律**從專案根目錄**執行，用 conda env `astro`；`zap.py` 會自動把 `libs/zap` 加進 path。
> 每支程式的用法（跑法、參數、輸入輸出）都寫在該檔案自己的 docstring 裡，這裡只列整體架構。

## 檔案

| 檔案 | 角色 |
|---|---|
| `settings.py` | 唯一設定檔：所有路徑 + 物理參數；re-export `cube_config` 的 `CubeConfig`/`DetectParams`/`get_cube_config` |
| `cube_config.py` | per-cube 偵測參數（PSF 量測、`bw`/`kernel`/`min_area`/`dilate`），lazy registry |
| `mask.py` | mask 後製（讀既有 `mask.fits` 加工，存成新 method 名，不覆蓋原檔）+ 畫圖函式 |
| `zap.py` | 跑 ZAP 扣天空 |
| `eval_spectrum.py` | 評估函式庫（任選空間區域 + 任選波段畫 spectrum、算摘要統計） |

## 資料流

```
data/Haro11_{nosky,wsky}.fits / Haro11_NEpointing_{wsky,esonosky}.fits   （唯讀，永不刪）
        ▼
results/zap/masks/<method>_from-<cube>/mask.fits         source mask
        ▼  可選：mask.py 後製 → 存成新 method 名
        ▼
zap.py                                                    扣天空
        └► results/zap/cubes/<target>_maskfrom-[<method>-]<mask_from>/{zap,sky,var}.fits
        ▼
eval_spectrum.py（import 用）                             任選區域/波段 → spectrum 圖 + 摘要統計
```

## 怎麼跑

三支程式的參數都用明確的 `--tag`（不是位置參數），照打就知道每個值是什麼：

```bash
# mask 後製（讀既有 mask.fits，存成新 method，不覆蓋原檔）
conda run -n astro python src/zap/mask.py --from-cube NEnosky --method seg1sigma

# ZAP 扣天空
conda run -n astro python src/zap/zap.py --target NEwsky --mask-from NEnosky --mask-method seg1sigma_brq --ncpu 16

# eval_spectrum：CLI 永遠跑 whole/blank，--box 可重複加自訂區域，--y-range 固定 y 軸；
# 存 png + spectra_report.json 到 --out-dir
conda run -n astro python src/zap/eval_spectrum.py \
    --fits-path results/zap/cubes/NEwsky_maskfrom-seg1sigma_brq-NEnosky/zap.fits \
    --mask-from NEnosky --mask-method seg1sigma_brq --y-range -20 20 \
    --box spec2 40 120 20 100
```
> `eval_spectrum.py` 也可以純 import 當函式庫用（自訂更複雜的分析時），見檔案開頭 docstring。
> 各程式完整參數/用法（包含 `eval_spectrum.py` 其他函式）見各自檔案開頭的 docstring。

## nosky / wsky / NE pointing

`settings.CUBE_NAMES = ("nosky", "wsky", "NEwsky", "NEnosky")`：nosky/wsky 是主場（已扣天空 /
含天空），NEnosky/NEwsky 是 NE pointing 的對應版本。`zap.py`/`eval_spectrum.py` 的程式對四顆
cube 完全相同，只差吃哪個 cube / 哪張 mask，用參數化，不重複寫。

> 物理參數的更動＝科學決策，請先確認 result-preserving（見專案根 `CLAUDE.md` 原則 2）。
