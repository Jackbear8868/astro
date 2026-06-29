# E. 源偵測 / 測光（對應 src/ 程式）

> 用到的工具：**SEP**（Source Extraction and Photometry，經典軟體 **SExtractor / Source Extractor** 的 Python 版）。
> 你 repo 的程式：`src/main.py`(手把手)、`src/segment_background.py`、`src/bkg_rms_map.py`、`src/sextract.py`。
> 真實數字皆跑自 `data/Haro11_nosky.fits`，波段 6500–6700 Å（含 Hα 發射線）。

## 這組在做什麼（一句話）
**讓電腦自動在影像裡找出一個個天體、框出它們、量它們的亮度。** 流程：
```
cube → 收成 2D 偵測影像 → 估背景並扣掉 → 用門檻找源 → 分割標記 → 孔徑測光 → 產出星表(catalog)
```
- **2D 偵測影像**：cube 沿波長加總成一張圖。全部加總 = 白光影像 (white-light)；只加一個波段 = 波段影像（訊號集中、較乾淨，如 Hα 波段）。
- 這組把 C 組（SNR/雜訊）、D 組（背景）整個用上：**門檻是用雜訊(σ)當尺、背景要先扣掉**。

---

## 27. Detection threshold（偵測門檻）

- 問題：扣完背景後，每個像素都有個值。哪些是**真的源**、哪些只是**雜訊抖動**？
- 規則：像素值 **> THRESH × 背景RMS** 才算源候選。`THRESH` 的單位是 **σ（sigma，= 雜訊大小）**。
  - 例：`THRESH=5` ＝ 要比雜訊亮 5 倍才算 → 這就是 C 組的 **SNR ≥ 5**。
- 還要過 `MINAREA`：至少要有 N 個**相連**像素都超過門檻（單一熱點不算），程式用 `MINAREA=5`。
- 取捨（真實證據，同一張圖只改 THRESH）：

  | THRESH | 偵測到源數 |
  |---|---|
  | 2σ | 222 |
  | 3σ | 108 |
  | 5σ | 54 |
  | 10σ | 25 |
  | 30σ | 11 |

  → **門檻低**：抓到暗源，但也把雜訊尖峰當成源（假源多）。**門檻高**：只留有把握的，但漏掉暗源。
  `src/main.py`、`segment_background.py` 用 5σ，`sextract.py` 用 4σ。

---

## 28. Background RMS map（背景起伏圖）— `bkg_rms_map.py`

- **background（背景）**：扣完天空後仍殘留的平滑底盤。真實值 `globalback ≈ 47`。
- **RMS（root-mean-square，均方根）= 雜訊大小**：像素在背景上下抖動的幅度。真實值 `globalrms ≈ 43`。
- 關鍵：**雜訊不是全圖一樣！** 亮星系附近 Poisson 雜訊更大（C 組：變異數=訊號）。所以雜訊是一張**隨位置變化的 2D 地圖**（`bkg.rms()`）。
  - 真實證據：RMS map 範圍 **3.1（空白天空）~ 308.5（星系旁）**。
- 為什麼重要：偵測門檻應該用**當地**雜訊（THRESH × 局部RMS），不是全圖一個值——星系亮處雜訊大，門檻的絕對高度也要跟著高，才不會把星系的雜訊誤判成一堆假源。
- `bkg_rms_map.py` 的眉角：估背景前要先**遮掉補零區與亮源**（`mask`），否則 RMS 網格在星系陡邊會被樣條內插「過衝」而出現負值。

---

## 29. Segmentation（分割圖）— `segment_background.py` / `main.py`

- **segmentation map（分割圖 segmap）**：一張和影像同尺寸的「標籤影像」：
  - 值 `0` = 背景；值 `1,2,3…` = 第 1,2,3… 個源所佔的像素。
- 用途：把影像切成兩區 → `source 區 = segmap>0`、`background 區 = valid 且 segmap==0`。也讓你知道「哪些像素屬於第幾號源」，後續才能分別量它們。
- 真實證據（THRESH=5）：54 個源；source 像素 20791、background 像素 229395。
- SEP 取得方式：`sep.extract(..., segmentation_map=True)` 會多回傳這張 segmap。
- 成果圖：`results/segmentation.png`（四格：扣背景影像 / 分割圖 / 源區 / 背景區）。

---

## 30. Aperture / Deblending（測光孔徑 / 分離重疊源）— `sextract.py`

### 孔徑測光 (aperture photometry)
- 量一個源的亮度 = 把它周圍**一個形狀（圓孔）內的通量加總**。`sep.sum_circle(影像, x, y, 半徑, err=雜訊)`。
- **孔徑大小的取捨**（真實證據，最亮源不同半徑）：

  | 半徑 | flux(總亮度) | err | SNR |
  |---|---|---|---|
  | 2px | 7.9e6 | 154 | 51500 |
  | 3px | 1.5e7 | 231 | 65926 |
  | 5px | 2.9e7 | 384 | **76465** |
  | 8px | 4.7e7 | 615 | 75883 |

  → 孔徑**太小**漏掉外圍的光；**太大**多收進雜訊（SNR 在 r=5 達頂、r=8 反而略降）。要選在「收進大部分源光、又不過量加噪」之間。

### Deblending（分離重疊源）
- 兩個靠很近的源，超過門檻後會連成一塊 → 被當成「一個」。**deblending 把它們拆回各自獨立的源**。
- `deblend_cont` 控制拆分積極度：**越小越愛拆**；`1.0` = 關閉拆分。
- 真實證據（同 THRESH=5）：

  | deblend_cont | 源數 |
  |---|---|
  | 1.0（關閉）| 49 |
  | 0.05 | 50 |
  | 0.005 | 54 |
  | 0.0001 | 54 |

  → 開啟拆分後多認出幾個原本黏在一起的源（此波段源較分離，效果溫和）。
- 產物：`sextract.py` 輸出星表 `results/sextract_catalog.txt`（id, x, y, flux, flux_err）與框源圖。

---

## 本組小結

- 工具 = **SEP（SExtractor 的 Python 版）**；流程：cube→2D 影像→扣背景→門檻偵測→分割→孔徑測光→星表。
- **偵測門檻**：值 > THRESH×RMS 才算源（THRESH 即 σ＝C 組 SNR）；低門檻多假源、高門檻漏暗源（2σ→222、5σ→54、30σ→11）。
- **背景 RMS map**：雜訊隨位置變（3→308），門檻要用**局部**雜訊；估背景前先遮亮源避免過衝負值。
- **Segmentation**：標籤圖（0=背景、1,2,…=各源），用來切源區/背景區並指認每個源的像素。
- **孔徑測光**：圓孔加總通量，半徑要折衷（太小漏光、太大加噪，本例 r≈5 最佳）；**deblending** 拆開黏在一起的源。
</content>
