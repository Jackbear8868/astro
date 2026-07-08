# src/legacy — pre-0706 的舊 ZAP pipeline（已封存，不再維護）

這裡是 **`src/0706/` 之前**的舊版 ZAP 扣天空 / 評估腳本。功能已被 `src/0706/` 全面取代，
保留僅供參考與歷史對照，**不建議再執行**。

| 舊腳本 | 現在請改用 |
|---|---|
| `run_zap_compare.py` | `src/0706/step1_mask.py` + `step2_zap.py` + `eval_*` |
| `fig1_wsky_effect.py` | `src/0706/eval_figures.py`（fig1） |
| `fig2_nosky_effect.py` | `src/0706/eval_figures.py`（fig2） |
| `fig3_source_halpha.py` | `src/0706/eval_figures.py`（fig3） |
| `fig4_source_mask.py` | `src/0706/eval_figures.py`（fig4） |
| `tune_nevals.py` | （ZAP nevals 調參實驗，未移植到 0706） |

⚠️ 這些腳本讀的 `results/zap/fits/source_mask.fits` 已於整併時移除。現在遮罩改為**自帶出處**的檔名：
- `source_mask_sep_nosky.fits` — 正式餵給 ZAP 的（= 0706 的 `settings.SOURCE_MASK_PATH`，方法 B / sep）
- `source_mask_robust_nosky.fits` — 舊 robust 法（方法 A）的產物，內容等同被移除的 `source_mask.fits`

若真要重跑舊腳本，需自行把路徑改指到上面新檔名。現行 pipeline 一律使用 `src/0706/`（見該資料夾 README）。
