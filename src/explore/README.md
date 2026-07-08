# src/explore — SEP 探索 / 學習 / 診斷小工具

跟正式 pipeline（`src/0706/`）無關的獨立 SEP 練習與診斷腳本，保留當參考。

| 腳本 | 做什麼 |
|---|---|
| `sextract.py` | SEP 源萃取最小範例（SExtractor 的 Python 版） |
| `segment_background.py` | 用 SEP 把 source / background 區域分開（最簡版） |
| `main.py` | 手把手：用 SEP 提取 source / background 並畫 segmentation 圖 |
| `bkg_rms_map.py` | 畫 SEP 估計的背景 RMS（2D 雜訊地圖） |

正式的源遮罩偵測（SEP + matched filter + 2σ）已整合進 `src/0706/step1_mask.py`。
