# MUSE 光譜資料學習總覽 (Learning Objectives)

這份清單整理了研究 MUSE (Multi Unit Spectroscopic Explorer) 積分視場光譜資料時，
應該理解的望遠鏡/儀器設定、資料結構、雜訊統計與分析概念。

我們會逐組 (A–F) 深入，每組各有一份獨立筆記檔，並用 `data/` 裡的真實 FITS 檔對照說明。

學習順序：**B（資料結構）→ C（雜訊統計）→ A（觀測設定）→ D（天空扣除）→ E（源偵測）→ F（光譜分析）**

進度標記：`[ ]` 未開始 · `[~]` 進行中 · `[x]` 已理解

---

## A. 望遠鏡 / 儀器的觀測設定（檔名裡藏的資訊） — 筆記：`A-observing-setup.md` ✅

- [x] 1. WFM / NFM — 視野模式（廣視野 1′×1′ vs 窄視野 7.5″×7.5″）
- [x] 2. AO / NOAO — 是否開啟調適光學（雷射導星修正大氣擾動）
- [x] 3. E / N (Extended / Nominal) — 光譜的波長涵蓋範圍
- [x] 4. Spatial sampling / spaxel scale — 每個空間格子代表天上多大
- [x] 5. Spectral resolution (R) — 能把多接近的兩條譜線分開
- [x] 6. Spectral sampling — 每個波長像素間隔多少 Å（MUSE 約 1.25 Å）
- [x] 7. Seeing / PSF / FWHM — 點光源被抹開的程度（影像清晰度）
- [x] 8. Sodium notch（鈉雷射缺口）— AO 模式下 ~5800–6000 Å 被擋掉
- [x] 9. Exposure time / Airmass — 曝光時間、天體離天頂多遠

## B. 資料的結構（IFU / FITS 檔怎麼組成） — 筆記：`B-data-structure.md` ✅

- [x] 10. Datacube（資料立方體）— (x, y, 波長) 三維陣列
- [x] 11. Pixel / Spaxel / Voxel — 三個常被混用但不同的東西
- [x] 12. HDU / Extension（DATA、STAT、DQ）— FITS 檔內部分層
- [x] 13. WCS（World Coordinate System）— 像素座標 ↔ 天球/波長
- [x] 14. Flux units（通量單位）— MUSE 常見 1e-20 erg/s/cm²/Å

## C. 像素層級的測量、雜訊與統計 — 筆記：`C-noise-statistics.md` ✅

- [x] 15. Pixel（像素）— 在此的物理意義
- [x] 16. Counts / ADU / electrons / Gain — 偵測器數字與光子的關係
- [x] 17. Signal vs Noise — 你要的東西 vs 干擾
- [x] 18. Error / Uncertainty（誤差）— 一次測量的不確定度（標準差）
- [x] 19. Variance（變異數）— 誤差的平方；STAT 存的就是這個
- [x] 20. Poisson noise（光子散粒雜訊）— 變異數 = 訊號量
- [x] 21. Gaussian noise（讀出雜訊）— 固定大小、與訊號無關
- [x] 22. SNR（訊雜比）— 訊號 / 雜訊

## D. 天空與背景及其扣除（對應 ZAP） — 筆記：`D-sky-background.md` ✅

- [x] 23. Sky lines / Sky continuum — 大氣自己發的光
- [x] 24. Sky subtraction — 扣天空（檔名 nosky / wsky）
- [x] 25. ZAP — 用 PCA 清除殘餘天光
- [x] 26. Background — 天文背景 vs 偵測器背景

## E. 源偵測 / 測光（對應 src/ 程式） — 筆記：`E-source-extraction.md` ✅

- [x] 27. Detection threshold（偵測門檻）
- [x] 28. Background RMS map（背景起伏圖）— `bkg_rms_map.py`
- [x] 29. Segmentation（分割圖）— `segment_background.py`
- [x] 30. Aperture / Deblending（測光孔徑 / 分離重疊源）

## F. 光譜分析（specutils） — 筆記：`F-spectral-analysis.md` ✅

- [x] 31. Emission / Absorption line（發射線 / 吸收線）
- [x] 32. Continuum（連續譜）
- [x] 33. Redshift / Rest-frame（紅移 / 靜止座標系）

---

## 對照用的資料檔（data/）

| 檔名 | 模式 | 說明 |
|------|------|------|
| `Haro11_WFM_MUSE_archive.fits` | WFM | 廣視野，完整星系結構 |
| `Haro11_NFM_ESO_nosky.fits` | NFM | 窄視野，高解析核心；已扣天空 |
| `Haro11_wsky.fits` | — | 含天空 (with sky) |
| `Haro11_nosky.fits` | — | 已扣天空 (no sky) |

> Haro 11 是一個著名的藍緻密矮星系 (blue compact dwarf galaxy)，有強烈恆星形成。
