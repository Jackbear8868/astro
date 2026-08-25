# 驗收用的程式

這裡放的是**看現行 pipeline 跑出來的結果好不好**的程式：吃 `results/skymodel/pNN/`
的產物，畫成圖或算成數字。每顆 pointing 都適用，跑完 `run_pipeline.py` 之後例行地跑。

隔壁 `experiments/` 放的是另一種東西：**問「該不該改成另一種做法」**的一次性實驗。
兩者的差別是問句不同 —— 這裡問「現在的結果如何」，那裡問「換一種做法會不會更好」。
一支程式如果比較的是兩個候選方案，它屬於 `experiments/`。

底下的 `poster/` 是第三種：**同樣的資料、同樣的曲線，換一套印刷排版**。它不問任何新
問題，只把線加粗、字放大、比例指定。科學內容要改，改的是這一層，不是 `poster/`。

輸出一律寫到 `results/skymodel/evaluation/`，不寫進 `pNN/` 工作區 —— `pNN/` 底下的每
一個檔案都是 `run_pipeline.py` 寫的，這條規則讓「刪掉 pNN 重跑」永遠安全。

```
results/skymodel/evaluation/
  p01/                      一顆 pointing 的全部驗收圖
    s_shape.png             s 的空間形狀
    main_group.png          主源分組
    sky_region.png          兩個階段各自用了哪些 spaxel 學天空
    segmentation_map.png    這顆的 segmentation
    basis/                  step3 學到的 K 條天空 basis，一條一張，加 overview / topN
    box/                    一個方框一張圖 + map.png 標出位置
    halo/                   Haro 11 延展光的分層光譜、源外環的比較
    masking/                halo_sources.png（延展光上偵測到什麼）
    sfield/                 這顆的 s 場，用跨 pointing 的共同色階
    sky/                    blank 區扣完之後剩下什麼、那些殘留是不是雜訊
    template_fit/           step4 的紅移掃描，一個源一張
    whitelight/             wsky.png（輸入）、compare.png（ESO vs 我們）
                            非預設波段編進檔名，例如 compare_5000-6000.png
  p02/  …
  halo/                     14 顆的延展光疊在同一張
  masking/                  教授的 seg、ID 對照、投影片版源圖
  sky_basis/                14 顆的天空連續譜、天空線基底的學習輸入
  sfield/                   14 顆的 s 場並排
  templates/                step4 擬合用的源模板長什麼樣
  template_fit/             跨 pointing 的擬合診斷
  poster_cache/             poster/ 的 cube 平均快取，不是圖
  subtraction_check/        跨 pointing、或不屬於任何一顆的驗收圖
  talk/  attic/
```

一顆一個目錄，而不是把 14 顆混在同一層用檔名區分：看某一顆的時候，要的是那一顆
的全部，不是在幾百個檔名裡挑出帶 `pNN` 的那些。路徑一律用 `common.pointing_dir()`
組出來，不要在各支腳本裡各拼各的。

## 檔案

分成四組，照「問的是 pipeline 的哪一段」排。

### 一 遮罩與源：這顆的源在哪裡、主星系是哪些 ID

| 檔案 | 回答什麼 | 寫到 |
|---|---|---|
| `prof_seg_maps.py` | 教授的 14 份 segmentation 長什麼樣 | `pNN/` |
| `seg_id_map.py` | 任一份 segmentation 的 source ID 對照圖 | `masking/` |
| `seg_slide_map.py` | 同一張源圖，去掉周邊資訊，投影片用 | `masking/` |
| `main_group_map.py` | 主源怎麼從被拆散的 seg ID 拼回來 | `pNN/` |
| ↑ | step5 每跑一顆就已經畫同一張到 `pNN/step05/main_group.png`(同一個 `plotting.plot_main_group`,逐像素相同)。這支多的是：一次跑多顆、印出被剔除的 ID 與源流量佔比 | |
| `main_group_spec.py` | 相鄰整團的每個成員，用光譜判斷是不是主星系的一部分 | 只印數字 |
| `halo_sources.py` | 放大看主星系：延展光上與周圍偵測到了什麼 | `pNN/masking/` |

### 二 天空模型本身：step3 與 step5 學到了什麼

| 檔案 | 回答什麼 | 寫到 |
|---|---|---|
| `plot_basis.py` | step3 學到的每一條天空 basis 長什麼樣 | `pNN/basis/` |
| `sky_line_residual.py` | 天空線基底是從什麼學來的：平均天空減掉天空連續譜 | `sky_basis/` |
| `continuum_compare.py` | 14 顆的天空連續譜疊在同一張，形狀差多少、水準差多少 | `sky_basis/` |
| `s_shape_map.py` | 天空連續譜係數 s 的空間形狀 | `pNN/` |
| `s_compare.py` | 14 顆的 s 場並排，用同一組色階 | `sfield/`、`pNN/sfield/` |
| `sky_region_map.py` | 天空是從哪些 spaxel 學的 —— 兩個階段各自用了哪些 | `pNN/` |

### 三 扣完之後：天空扣乾淨了嗎、源有沒有被扣掉

這一組每一支都同時看兩件事。只看殘留大小沒有用 —— 過度扣除同樣會讓殘留變平。

| 檔案 | 回答什麼 | 寫到 |
|---|---|---|
| `whitelight_wsky.py` | 輸入 cube（含天空）的白光影像 | `pNN/whitelight/` |
| `whitelight_compare.py` | 扣完天空的白光影像，ESO 與我們並排 | `pNN/whitelight/` |
| `box_spectra.py` | 方框裡的平均光譜，和 ESO nosky 並排 | `pNN/box/` |
| `blank_compare.py` | blank 區扣完之後剩下什麼，我們對 ESO | `pNN/sky/` |
| `blank_noise_floor.py` | 那些殘留是雜訊還是錯誤 —— 逐通道，兩個 pipeline 都算 | `pNN/sky/` |
| `halo_spectra.py` | Haro 11 延展光的光譜，一層一層往外 | `pNN/halo/` |
| `halo_compare.py` | 同樣的分層，14 顆疊在一起 | `halo/` |
| `outside_compare.py` | 源邊界外的環，我們扣完剩什麼、ESO 扣完剩什麼 | `pNN/halo/` |

`halo_spectra.py` 定義了分層與環（主源分組、等量白光亮度層、往外的距離環），
`halo_compare.py` 和 `outside_compare.py` 直接 import 它而不是各自重寫一份：第二套
「outside」的定義會讓兩張圖對不起來，而原因跟 pipeline 無關。

### 四 源的擬合：step4 用什麼模板、擬得如何

| 檔案 | 回答什麼 | 寫到 |
|---|---|---|
| `plot_eigen.py` | step4 擬合用的源模板長什麼樣（星系 / QSO eigenspectra、恆星庫） | `templates/` |
| `chi2_scan.py` | 單一源的紅移掃描：reduced χ² 對 z，恆星與星系分開 | `pNN/template_fit/` |

`plot_eigen.py` 是用擬合時同一組 spline 求值畫出來的，不是重讀檔案，而且只畫每條
spline 自己的定義域 —— 所以畫出來的就是擬合看得到的，兩端補的常數與檔案裡填零的空
隙都不會出現在圖上。

## 其他

`common.py` 放共用的東西：`ROOT`、`EVAL`、`pointing_dir()`、`load_field()`、`slug()`、
`qualitative()`（相鄰源不會混淆的配色）、`asinh_bar()`。

已刪除：`check_pointing.py`（驗收：天空扣乾淨了嗎、源有沒有被扣掉）、
`zone_spectra.py`（同一個環上 ESO 與我們各扣出什麼）。功能被 `box_spectra.py`
的逐方框光譜取代。`pNN/point/` 與 `pNN/zone/` 是它們留下來的舊輸出，現在沒有程式會寫。

`ROOT` 與 `import utils` 都靠 `Path(__file__).resolve().parents[N]`，而這個目錄和
`experiments/` 在同一層，所以搬動不需要改路徑。`poster/` 在下一層，它的兩行
`sys.path.insert` 用的是 `parents[1]` 與 `parents[2]`。
