# 驗收用的程式

這裡放的是**看現行 pipeline 跑出來的結果好不好**的程式：吃 `results/skymodel/pNN/`
的產物，畫成圖或算成數字。每顆 pointing 都適用，跑完 `pipeline.py` 之後例行地跑。

隔壁 `experiments/` 放的是另一種東西：**問「該不該改成另一種做法」**的一次性實驗。
兩者的差別是問句不同 —— 這裡問「現在的結果如何」，那裡問「換一種做法會不會更好」。
一支程式如果比較的是兩個候選方案，它屬於 `experiments/`。

底下的 `poster/` 是第三種：**同樣的資料、同樣的曲線，換一套印刷排版**。它不問任何新
問題，只把線加粗、字放大、比例指定。科學內容要改，改的是這一層，不是 `poster/`。

印刷版的檔名跟它所依據的螢幕版一樣，所以它們寫到 `evaluation/poster/` 底下自成一層
（`poster/pNN/basis/top5.png`、跨 pointing 的則是 `poster/sky_basis/`），否則會直接
覆蓋掉原本那張。屬於單一 pointing 的用 `Run.figdir(..., poster=True)`。

輸出一律寫到 `results/skymodel/evaluation/`，不寫進 `pNN/` 工作區 —— `pNN/` 底下的每
一個檔案都是 `pipeline.py` 寫的，這條規則讓「刪掉 pNN 重跑」永遠安全。

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

## 怎麼用

**平常只需要這一行。**

```bash
python src/skymodel/evaluation/run.py --work results/skymodel/p01          # 這一顆的全部
python src/skymodel/evaluation/run.py --work results/skymodel/p01 --all    # 再加上跨 pointing 的
python src/skymodel/evaluation/run.py --work results/skymodel/p01 --only sky
python src/skymodel/evaluation/run.py --work results/skymodel/p01 --list
```

`run.py` 裡有一張表，一列一組圖：叫什麼、屬於哪一組、哪支程式畫、寫到哪裡。要多一張
圖就多一列，不必改邏輯；要知道「這個專案到底量了什麼」，讀那張表就好，不必讀 22 支
程式。

| 選項 | 做什麼 |
|---|---|
| `--only sky` | 只跑一組（`masking` / `sky` / `subtraction` / `fit`），或直接點名 `--only halo outside` |
| `--all` | 連跨 pointing 的那幾張一起（沒給 `--work` 時本來就只有這些） |
| `--everything` | 連「不隨這顆 run 改變」的也跑：教授的 seg、step4 的模板庫、投影片版。它們重跑會得到同一個檔，所以預設不跑 |
| `--list` | 什麼都不畫，只報告哪些畫過、哪些**比它所描述的產物還舊** |
| `--dry-run` | 印出指令而不執行，要單獨手跑某一支時用 |

`--list` 是重點。一張比產物舊的圖，從外觀看不出任何問題 —— 它只是一張「已經不存在的
那次 run」的照片。這個專案裡已經抓到過兩次（`s_hat.png`、`outside_raw_vs_signal`），
而一顆 pointing 有一百多個檔，手動比日期沒有人做得到。

底下的表是同一份資料的說明版；每支程式仍然可以單獨執行，`run.py` 只是把它們排好。

## 檔案

分成四組，照「問的是 pipeline 的哪一段」排。

### 一 遮罩與源：這顆的源在哪裡、主星系是哪些 ID

| 檔案 | 回答什麼 | 寫到 |
|---|---|---|
| `prof_seg_maps.py` | 教授的 14 份 segmentation 長什麼樣 | `pNN/` |
| `seg_id_map.py` | 任一份 segmentation 的 source ID 對照圖 | `masking/` |
| `seg_slide_map.py` | 同一張源圖，去掉周邊資訊，投影片用 | `masking/` |
| `main_group_map.py` | 主源怎麼從被拆散的 seg ID 拼回來 | `pNN/` |
| ↑ | step5 每跑一顆就已經畫同一張到 `pNN/step05/main_source_group.png`(同一個 `utils.plot_main_group`,逐像素相同)。這支多的是：一次跑多顆、印出被剔除的 ID 與源流量佔比 | |
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
| `zone_spectra.py` | 每個 zone 的平均光譜：星系的亮度層、源外的環，一個 cube 或好幾個並排 | `pNN/halo/` |
| `halo_compare.py` | 同樣的分層，14 顆疊在一起 | `halo/` |

`zone_spectra.py` 是 `halo_spectra.py` 與 `outside_compare.py` 合併來的 —— 它們本來
就是同一支程式：同一個 zone 構造、同一種面板、同樣的譜線標記、同樣「哪些通道跑出面板」
的回報。一個畫全部 zone 各一條曲線，另一個畫外圈 zone 各兩條。合併之後就是兩個開關：

```bash
zone_spectra.py --work results/skymodel/p01                                  # 全部 zone，我們的 cube
zone_spectra.py --work results/skymodel/p01 --zones outside --cubes ours eso # 源外的環，我們 vs ESO
zone_spectra.py --work results/skymodel/p01 --zones galaxy  --cubes ours eso # 星系分層，我們 vs ESO
```

最後那一張是舊的兩支都畫不出來的:`halo_spectra` 不能比較兩個 cube，`outside_compare`
只能畫源外的環。`--cubes` 認得 `ours`（step06）、`eso`、`wsky`（輸入）、`model`（扣掉的
天空）、`run:GLOB`，或直接給路徑。

**顏色永遠承載「變動的那個維度」。** 只有一個 cube 時，面板之間差的是 zone，所以顏色由
內而外走 viridis，zone 地圖也用同一組顏色；有好幾個 cube 時，顏色代表 cube 且每一格都
一樣 —— 要讀的是它們之間的差別，那個差別必須在每一格都是同一個意思。

zone 的定義在 `zones.py`，`halo_compare.py` 和 `poster/` 都 import 它：第二套「outside」
的定義會讓兩張圖對不起來，而原因跟 pipeline 無關。

### 四 源的擬合：step4 用什麼模板、擬得如何

| 檔案 | 回答什麼 | 寫到 |
|---|---|---|
| `plot_eigen.py` | step4 擬合用的源模板長什麼樣（星系 / QSO eigenspectra、恆星庫） | `templates/` |
| `chi2_scan.py` | 單一源的紅移掃描：reduced χ² 對 z，恆星與星系分開 | `pNN/template_fit/` |

`plot_eigen.py` 是用擬合時同一組 spline 求值畫出來的，不是重讀檔案，而且只畫每條
spline 自己的定義域 —— 所以畫出來的就是擬合看得到的，兩端補的常數與檔案裡填零的空
隙都不會出現在圖上。

`chi2_scan.py` 是唯一需要完整掃描曲線的一支，它讀 `pNN/step04/scans/`。那個目錄只有
在 `configs/pNN.yaml` 的 `source_fit.keep_scans` 打開時才會寫出來，預設是 `false`；
要畫這張圖，就把那顆的 `keep_scans` 改成 `true` 再跑一次。其餘的程式都只讀
`step04/source_fits.npz` 與 `classification.npz`，掃描檔在不在都一樣。

## 其他

一個 run 的產物統一由 `products.Run` 讀。它問的是「產物在哪」，不是「誰跑的」——
`pipeline.py` 和 `standalone/` 寫出來的是同一棵目錄樹（`standalone/check_mirror.py`
就是在保證這件事），所以同一個物件兩邊都適用。每個欄位第一次被碰到才讀、讀完留著，
cube 則只給路徑不給陣列，讓呼叫端自己 memmap。

```python
run = Run("results/skymodel/p01")
run.wl        run.seg      run.white    run.valid
run.continuum run.mean_sky run.line_mask run.basis("svd", 30)
run.s_field   run.cube     run.nosky    run.step04
run.meta(3)   run.figdir("halo")
```

共用的東西分成四個模組，照「這是關於什麼的」分：

| 模組 | 放什麼 |
|---|---|
| `products.Run` | 一個 run 的產物在哪、怎麼讀 |
| `zones.py` | 哪些 spaxel：`zone_labels()`（亮度層＋距離環）、`blank_mask()`（某次 run 的 blank）、`zone_means()`（每個 zone 的平均光譜） |
| `spectra.py` | 光譜圖共用的字彙：`Z_HARO`、`LINES`、曲線顏色、`despiked_range()` / `panel_ylim()` / `robust_range()` |
| `common.py` | 影像那一層：`EVAL`、`POSTER`、`slug()`、`qualitative()`、`asinh_bar()`、`diverging_range()`、`collapse()`、`data_hdu()`、`band_tag()`、`seg_and_background()`、`map_name()`、`sigma_image()`、`seg_outline()`、`s_panel()` |

**畫圖的腳本不再被當成 library。** 以前 `blank_compare` 匯出 `our_cube`、`halo_spectra`
匯出 `zone_labels`、`outside_compare` 匯出 `despiked_range`（後兩支現已合併為
`zone_spectra.py`），別的腳本 import 它們只為了
拿一個函式，卻連帶執行整個模組層。現在這些定義都在上面四個模組裡，腳本之間互不 import。

被合掉的重複：`Z_HARO` 原本定義在三個檔、`zone_means` 有三份、`panel_ylim` 與
`despiked_range` 函式體一字不差（只差最後補的邊距）、`our_cube` 與 `s_dir` 是同一個
形狀（現在是 `products.latest_run`）。

### 成對的圖：共用畫法，但各自是一支程式

有三對圖畫的是同一種東西：

| 一對 | 差在哪 | 共用什麼 |
|---|---|---|
| `whitelight_wsky` / `whitelight_compare` | 一格 vs 兩格；含天空的要先扣掉底座 | `sigma_image()`、`seg_outline()`、`band_tag()` |
| `seg_id_map` / `seg_slide_map` | 工作用的定位圖 vs 投影片版 | `seg_and_background()`、`map_name()` |
| `s_shape_map` / `s_compare` | 一顆各自色階 vs 多顆同一色階 | `s_panel()` |

它們**沒有**被合併成一支加模式旗標的程式。每一對問的是不同的問題、印的是不同的診斷
數字（`whitelight_compare` 印零點與主源中位數，對單格圖沒有意義），合起來只會讓檔案更
難讀。重複的是「怎麼畫」，抽掉的就是那一段 —— 這樣同一個欄位的兩張圖不可能因為 stretch
或輪廓畫法不同而看起來像不同的資料，而每支程式仍然只回答它自己那一個問題。

已刪除：`check_pointing.py`（驗收：天空扣乾淨了嗎、源有沒有被扣掉）、
`zone_spectra.py`（同一個環上 ESO 與我們各扣出什麼）。功能被 `box_spectra.py`
的逐方框光譜取代。`pNN/point/` 與 `pNN/zone/` 是它們留下來的舊輸出，現在沒有程式會寫。

`ROOT` 與 `import utils`、`import products` 都靠 `Path(__file__).resolve().parents[N]`，而這個目錄和
`experiments/` 在同一層，所以搬動不需要改路徑。`poster/` 在下一層，它的兩行
`sys.path.insert` 用的是 `parents[1]` 與 `parents[2]`。

`sigma_image()` 和 `seg_outline()` 是兩個小函式而不是一個「畫一格」的大函式，因為兩支
白光程式的順序不同（`compare` 的 colorbar 在畫輪廓之前建立），而 colorbar 會從軸上挖走
空間，順序一換版面就差幾個像素。拆成兩個，各自照自己的順序呼叫，圖才會逐位元一樣。

`sigma_image()` 的 `hi` 要傳原本算出來的值，不要先 `float()`：`np.arcsinh(float32)` 和
`np.arcsinh(float64)` 差在第 8 位，而色階上限就是從它來的。
