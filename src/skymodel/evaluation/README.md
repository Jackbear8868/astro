# 驗收用的程式

這裡放的是**看現行 pipeline 跑出來的結果好不好**的程式：吃 `results/skymodel/pNN/`
的產物，畫成圖或算成數字。每顆 pointing 都適用，跑完 `run_pointing.sh` 之後例行地跑。

隔壁 `experiments/` 放的是另一種東西：**問「該不該改成另一種做法」**的一次性實驗。
兩者的差別是問句不同 —— 這裡問「現在的結果如何」，那裡問「換一種做法會不會更好」。
一支程式如果比較的是兩個候選方案，它屬於 `experiments/`。

輸出一律寫到 `results/skymodel/evaluation/`，檔名帶 pointing。放各自的工作區裡的話，
要比較幾顆就得開幾個目錄，而且檔名相同排不到一起。

## 檔案

| 檔案 | 回答什麼 |
|---|---|
| `check_pointing.py` | 驗收：天空扣乾淨了嗎、源有沒有被扣掉 |
| `box_spectra.py` | 方框裡的平均光譜，和 ESO nosky 並排 |
| `zone_spectra.py` | 同一個環上，ESO 與我們各扣出什麼 |
| `s_shape_map.py` | 天空連續譜係數 s 的空間形狀 |
| `oversub_whitelight.py` | over-subtraction 畫成看得見的白光圖 |
| `main_group_map.py` | 主源怎麼從被拆散的 seg ID 拼回來 |
| `main_group_spec.py` | 相鄰整團的每個成員，用光譜判斷是不是主星系的一部分 |
| `compare_runs.py` | 幾個 step5 的 run 並排成白光圖 |
| `prof_seg_maps.py` | 教授的 14 份 segmentation 長什麼樣 |
| `seg_id_map.py` | 任一份 segmentation 的 source ID 對照圖 |

`ROOT` 與 `import utils` 都靠 `Path(__file__).resolve().parents[N]`，而這個目錄和
`experiments/` 在同一層，所以搬動不需要改路徑。
