# SExtractor 安裝與使用說明

> SExtractor（Source Extractor, Bertin & Arnouts 1996）：對 2D 天文影像做**源偵測與測光**的
> 標準工具。輸入一張 FITS 影像，輸出源目錄（catalog）與檢查影像——我們要的是
> **SEGMENTATION map**（每像素標「屬於第幾號源、0=背景」），它就是 skymodel 的 source mask 原料。
> 本資料夾的 `default.*` 四個檔是教授提供的設定檔（**權威基準，保持原封**，CLAUDE.md Principle 2）。
> 工作流就是在本資料夾直接下 `sex` 指令，不經任何包裝。

---

## 1. 安裝

已裝進本專案的 `astro` conda 環境（conda-forge 打包）：

```bash
conda install -n astro -c conda-forge astromatic-source-extractor
conda activate sex
sex --version        # SExtractor version 2.28.2（本機已裝）
```

`sex` 與 `source-extractor` 兩個指令等價。
（別和 Python 套件 `sep` 混淆：`sep` 是同演算法的 Python 移植，0706 pipeline 用它；
本資料夾跑的是原版程式。）

## 2. 四個設定檔各是什麼

| 檔案 | 角色 |
|---|---|
| `default.sex` | **主設定檔**：偵測門檻（1.0σ）、MINAREA（10）、背景網格（64）、濾波核、輸出設定 |
| `default.param` | catalog 要輸出哪些欄位（NUMBER、X/Y_IMAGE、FLUX_ISO、ISOAREA、FLAGS、CLASS_STAR） |
| `default.conv` | 偵測前的卷積濾波核（3×3、FWHM 2px） |
| `default.nnw` | 星系/恆星分類（CLASS_STAR）的類神經網路權重——照用即可 |

## 3. 使用方式（標準工作流）

SExtractor 吃 **2D 影像**（cube 要先壓成影像，見 §5）。在本資料夾直接跑：

```bash
cd src/skymodel/SExtractor
sex det_ha.fits -c default.sex
```

產出（檔名由 `default.sex` 決定，落在本資料夾）：

| 檔案 | 內容 |
|---|---|
| `test.cat` | 源目錄（ASCII，欄位見 `default.param`） |
| `seg.fits` | **segmentation map**：每像素 = 源編號，0 = 背景 → source mask 原料 |
| `nosky.fits` | 扣掉估計背景後的影像（CHECKIMAGE `-BACKGROUND`） |

實跑紀錄（main 全場，whitelight 偵測）：**306 objects**，seg>0 覆蓋視場 **29.2%**，
星系核心＋內圈暈成一塊連通區。

> **偵測影像 = whitelight（全譜 nanmean），教授指示**：whitelight 對所有連續譜源
> （恆星、背景星系、暈的連續光）靈敏，遮罩才完備；Hα 窄帶只對發射線源靈敏、會漏源。
> 批次腳本 `run_on_subcubes.py` 偵測 `det_white.fits`，輸出至 `results/skymodel/sextractor_sub/`。

任何參數想試不同值時，不要改 `default.sex`，用命令列覆寫（例：`-DETECT_THRESH 2.0`）；
覆寫屬於實驗，正式產物一律出自原設定檔。

## 4. 參數的物理意義（理解用參考；權威值 = default.sex）

各參數在「量什麼」的層面上的物理對應，幫助讀懂設定檔（依 CLAUDE.md Principle 2：
此表僅供理解與討論，**不是**用來修改教授設定的依據）：

| 參數 | default.sex 值 | 物理意義 |
|---|---|---|
| `DETECT_THRESH` | 1.0 σ | 幾倍背景雜訊以上算訊號（門檻越低偵測越深、假陽性越多） |
| `DETECT_MINAREA` | 10 px | 連通區至少幾個像素才算源（過濾雜訊點；可對照 PSF 面積） |
| `FILTER_NAME` | 3×3 FWHM 2px | 偵測前的 matched filter（核越接近 PSF，點源 S/N 增益越大） |
| `BACK_SIZE` | 64 px | 背景網格尺寸（決定「多大尺度以上的光算背景」——與延展暈的關係可用 `CHECKIMAGE_TYPE BACKGROUND` 直接檢視） |
| `DEBLEND_*` | 32 / 0.005 | 相黏的源怎麼拆開 |
| `CLEAN` | Y | 剔除鄰近亮源翅膀造成的假偵測 |

## 5. 偵測影像（SExtractor 的輸入）的來源

`det_ha.fits`／`det_white.fits` 由 `data/Haro11_nosky.fits` 壓成
（舊產物已移除，需要時可依下式重壓）：

- `det_ha.fits`：Hα 窄帶 = mean(6692–6708 Å) − mean(連續譜窗 6605–6645 / 6760–6795 Å)
  （nosky 與 wsky 同曝光同網格，偵測用較乾淨的 nosky）
- `det_white.fits`：全譜 nanmean（白光）
- 無效像素（NaN／視場外）已填 0——SExtractor 不吃 NaN

## 6. 常用 CHECKIMAGE_TYPE 速查

| 值 | 內容 |
|---|---|
| `SEGMENTATION` | 每像素屬於第幾號源（0=背景）→ mask 原料 |
| `-BACKGROUND` | 原圖減估計背景 |
| `BACKGROUND` / `BACKGROUND_RMS` | 背景圖 / 背景雜訊圖（檢視背景估計把什麼吸進去了） |
| `OBJECTS` / `-OBJECTS` | 只留源 / 挖掉源 |
| `APERTURES` | 測光孔徑疊圖 |

官方手冊：https://sextractor.readthedocs.io
