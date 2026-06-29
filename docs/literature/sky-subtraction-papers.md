# 天空扣除文獻分類：預測 sky vs 預測 residual

> 由 5 個平行搜尋 agent 彙整（物理 sky 模型 / 經驗 sky / PCA-residual / ML / 廣掃）。
> 分類定義：
> - **預測 sky（策略一）**：建模天空本身（連續譜+天光線，物理或經驗/資料驅動），再 `data − sky`。
> - **預測 residual（策略二）**：先粗扣，再對「扣剩的殘餘」建模並移除（多為 PCA/資料驅動）。
> - **混合**：pipeline 同時用兩策略。**其他**：相關但非上述（線表、telluric、ML 去噪、觀測策略、回顧）。
> - 所有條目皆經 agent 在實際搜尋結果中確認存在；末段列出待驗證者。

---

## 類別一：預測 SKY（策略一）

### 1a. 經驗 / 觀測法（傳統）
- **Kelson 2003** — Optimal Techniques in 2D Spectroscopy: Background Subtraction. `astro-ph/0303507` · PASP 115,688。直接建 2D 天空模型、抽取前先扣；「抽取前扣天空」的奠基作。
- **Glazebrook & Bland-Hawthorn 2001** — Microslit Nod-Shuffle Spectroscopy. `astro-ph/0011104` · PASP 113,197。望遠鏡 nod + CCD 電荷搬移，**直接量到天空**(~0.04%)；屬「直接取得 sky 再減」。
- **Davies 2007** — A method to remove residual OH emission from NIR spectra. `astro-ph/0612257` · MNRAS 375,1099。把參考天空的 OH 線群依物理分組重標到科學幀再減（標題雖含 residual，核心是**重標 sky**）。SINFONI/KMOS 採用。
- **Noll et al. 2014 — Skycorr**. `1405.3679` · A&A 567,A25。Davies 的通用化：分離線/連續譜、依變動性分組 OH/O₂，把參考天空縮放擬合到科學光譜。儀器無關標準工具。
- **Law et al. 2016 — MaNGA DRP**. `1607.08619` · AJ 152,83。~92 根 sky fiber 併成超取樣天空，逐 fiber 波長格估算+縮放再減（<8500Å 近 Poisson 極限）。
- **Sánchez et al. 2016 — CALIFA DR3**. `1604.02289` · A&A 594,A36。用 PPak 最暗 sky fiber 平均建天空再減（殘餘~1–5%）。
- **Streicher et al. 2011 — MUSE pipeline sky subtraction**. ADS 2011ASPC..442..257S（無 arXiv）。MUSE 從最暗 spaxel 估天空、含 sky-line LSF/超取樣。
- **Rodrigues et al. 2010** — new algorithm for sky extraction for multi-fiber. `1009.0554`。由 sky fiber 重建天空的空間起伏再減。
- **Han, Song & Zhao 2023** — Sky subtraction of LAMOST at bright night. MNRAS 526,5520（無 arXiv）。加權趨勢面逐 fiber 重建天空，處理月光色梯度。
- （延伸）**"Accurate Sky Continuum Subtraction with Fibre-fed Spectrographs"** `1302.3620`。

### 1b. 資料驅動 / 低秩（傳統；用 NMF/低秩建 sky）
- **Zhang, Zhang & Ye 2016** — NMF with Sparsity sky model (LAMOST). PASA 33,e058。NMF+稀疏建天空，與 B-spline/PCA 對比。
- **Kolganov, Chilingarian & Grishin 2023** — NMF approach to sky subtraction. `2312.06761`。用 NMF 取代 PCA 得非負天空基底(~10× 有效成分)，無需 offset sky。

### 1c. 機器學習（ML）★ 唯一的 DL 預測 sky
- **Zhang et al. 2025 — Sky Background Building via Mutual Information Network (SMI)**. `2508.19875` · RAA。雙網路（波長校正 + 互資訊最大化）用**全部** fiber 預測每個物件位置的天空再減。LAMOST，藍端改善明顯。

### 1d. 物理 / 合成天空模型（傳統；直接「產生」天空光譜 → 屬預測 sky）
> 註：這些是「天空模型/資料庫」，本身非扣除演算法，但生成可直接相減的 sky。
- **Noll et al. 2012 — Cerro Paranal Advanced Sky Model (optical)**. `1205.2003` · A&A 543,A92。物理建模散射月光/星光、黃道光、氣輝線+連續譜（LBLRTM 輻射轉移）。
- **Jones et al. 2013 — advanced scattered moonlight model**. `1310.7030` · A&A 560,A91。月光散射成分。
- **Noll et al. 2025 — PALACE v1.0 (airglow model)**. `2504.10683` · GMD 18,4353。9 物種、26541 條氣輝線、3 連續譜成分，含太陽週期/季節氣候學。
- **Patat 2008** — The dancing sky: 6 yr at Cerro Paranal. `0801.2270` · A&A 481,575。氣輝變動的經驗特徵化（Noll 2012 的基礎）。
- **Krisciunas & Schaefer 1991** — Model of the Brightness of Moonlight. PASP 103,1033。經典月光亮度解析模型。
- **Yoachim et al. 2016** — optical–IR sky brightness model for LSST. SPIE 9910。由 SkyCalc 模板建 Rubin 天空光譜庫。
- **ESO SkyCalc**（工具）— Cerro Paranal 模型的 web/CLI 實作（引用 Noll 2012 / Jones 2013）。

---

## 類別二：預測 RESIDUAL（策略二）

### 2a. PCA / SVD（傳統）— 主力
- **Kurtz & Mink 2000 — Eigenvector Sky Subtraction**. `astro-ph/0003112` · ApJ 533,L183。迭代用 SVD 推得的本徵天空/殘餘模型相減；PCA-殘餘譜系的源頭。
- **Wild & Hewett 2005 — Peering through the OH-forest**. `astro-ph/0501460` · MNRAS 358,1083。對「已扣天空」的 OH 殘餘做 PCA 建本徵譜再逐譜扣。fiber 巡天的奠基 predict-residual。
  - 資料釋出：**Wild & Hewett 2010**（SDSS DR7 殘餘已扣版）`1010.2500`。
- **Sharp & Parkinson 2010 — Sky subtraction at the Poisson limit**. `1007.0648` · MNRAS 408,2495。指出 fiber 扣天空留系統殘餘，提出 PCA 殘餘程序（長曝光優於 nod-and-shuffle）。亦含策略比較。
- **Soto et al. 2016 — ZAP (Zurich Atmosphere Purge)**. `1602.08037` · MNRAS 458,3210 · github.com/musevlt/zap。★ **MUSE/IFU 的標準殘餘移除工具**：首扣後對 cube 做過濾+分段 PCA，建「消毒過」本徵譜只抓天空殘餘、保留源。
- **Marchetti et al. 2017 — VIPERS PCA cleaning/reconstruction**. `1612.01825` · A&A 600,A54。觀測座標 PCA 標出 sky-line 殘餘、再以靜止座標 PCA 重建修補(~9 萬譜)。
- **Hart 2019 — Sky Residual Correction**. AJ 157,213 · DOI 10.3847/1538-3881/ab1a35（無 arXiv）。以已扣天空的 sky fiber 為訓練集建 ~20 個 PCA 成分扣殘餘（SDSS/BOSS/APOGEE）。
- **Husemann et al. 2022 — CARS / CubePCA**. `2111.10417` · A&A 659,A124。ZAP 的簡化版 PCA sky-line 殘餘抑制器（參數更少、對源內容更穩健），MUSE 用。

### 2b. 貝氏 / 其他資料驅動（borderline ML）
- **Uzsoy et al. 2025 — Bayesian Component Separation for DESI LAE**. `2504.06870`。以資料驅動先驗，**聯合推論天空殘餘成分 + LAE 訊號**，對殘餘做邊際化而非硬減。概念上是 predict-residual，但用貝氏非神經網路。

### 2c. 機器學習（ML）
- **（空缺）** — 五個 agent 皆未找到「用深度學習**專門預測扣天空後殘餘**」的論文。現況最接近者：傳統 PCA（ZAP / Wild&Hewett）+ 上述貝氏法（2b）。**這是一個明確的研究空缺**（你若做 ML predict-residual，幾乎是空白地帶）。

---

## 類別三：混合（pipeline 同時用兩策略）
- **Weilbacher et al. 2020 — MUSE DRP**. `2006.08638` · A&A 641,A28。pipeline 內建天空模型(連續譜+氣輝線含 LSF)相減（預測 sky），並建議後接 **ZAP**（預測 residual）。
- **Guy et al. 2023 — DESI spectro pipeline**. `2209.14482` · AJ 165,144。先 per-petal sky fiber 前向建模(spectro-perfectionism, 預測 sky)，再對強天光線做 PCA 殘餘修正(預測 residual)。
- **Croom et al. 2021 — SAMI DR3**. `2101.12224` · MNRAS 505,991。先扣 master sky，再用最暗 ~10% fiber 取少數主成分最小化 sky-line 殘餘。
- **Bai et al. 2017 — Sky Subtraction for LAMOST**. `1705.02079` · RAA 17,91。B-spline 超取樣 master sky（預測 sky）+ PCA 再修正紅端 OH 殘餘 ~25%（預測 residual）。

---

## 類別四：其他（相關但非 sky/residual 預測）

### 4a. 天光發射線表 / atlas（建模的輸入資料）
- **Hanuschik 2003** — UVES 光學天光發射 atlas. A&A 407,1157（2808 線, R~45000）。
- **Rousselot et al. 2000** — NIR OH 線表. A&A 354,1134（4732 條 OH, 1.0–2.25µm）。
- **Cosby et al. 2006** — UVES/VLT nightglow 線辨識. JGR 111,A12307。
- **Oliva et al. 2015** — GIANO-TNG NIR 天光線+連續譜. `1506.09004` · A&A 581,A47（揭露 "hot-OH"）。
- **Viuho, Fynbo & Andersen 2025** — NIR airglow continuum conundrum. `2506.02102`（FeO 主導連續譜）。

### 4b. Telluric 吸收校正（姊妹問題，非天空發射）
- **Smette et al. 2015 — Molecfit I**. `1501.07239` · A&A 576,A77（輻射轉移建大氣穿透）。（II：Kausch et al. 2015 `1501.07237`）
- **Sedaghat et al. 2023 — Stellar Karaoke**. `2301.00313` · MNRAS。深度 autoencoder 盲分離大氣成分（~25 萬 HARPS 譜）；ML 最接近「從資料學大氣成分」的類比。
- **Telluric autoencoder 2021** — Unsupervised spectral unmixing for telluric correction. `2111.09081`（作者待確認）。

### 4c. ML 光譜去噪 / 對殘餘 robust / 分類（ML，皆屬其他）
- **Melchior et al. 2023 — SPENDER (Autoencoding Galaxy Spectra I)**. `2211.07890` · AJ 166,74。卷積 autoencoder，刻意設計成**對 skyline 殘餘 robust**。（II：`2302.02496`）
- **Camilleri et al. 2025** — Emergent Denoising of SDSS Galaxy Spectra (unsupervised AE). `2510.08411`。
- **Denoising medium-res stellar spectra with U-Net 2025**. `2504.02523`。
- **Mukae et al. 2026** — CNN for Lyα Emitter ID in HETDEX. `2604.12414`。分辨真 LAE vs「artifact 與 sky 殘餘」（非移除）。
- **MaNGA anomaly-detection autoencoder 2026**. `2603.03734`。
- **CNN–Transformer denoiser for low-S/N galaxy spectra 2026**. `2605.04434`。
- **Physics-informed super-resolution of galaxy spectra 2026**. `2603.18357`。
- **Vision Transformers for spectral analysis 2025**. `2506.00294`。
- **DESI DR2 pipeline QA with AI 2026**. `2606.21035`（評估扣天空品質，非執行）。

### 4d. 觀測策略比較 / 回顧
- **Rodrigues et al. 2012/2016** — On-sky tests of sky-subtraction methods (FLAMES). `1609.06142` · SPIE 8450（cross-beam-switching/dual-stare <1%）。
- （Sharp & Parkinson 2010 亦含策略比較，見 2a。）

---

## 待驗證 / 存疑（引用前再查）
- **Subaru PFS sky subtraction** — SPIE 13096,130962M (2024)，無 arXiv；2D-PSF 前向建模(預測 sky)。DOI 10.1117/12.3015628 待查。
- **HETDEX/VIRUS** — local amplifier-level sky model(預測 sky)；儀器文 `2110.03843`、DR `2606.04208`(2026，暫定)。
- **2026 年 ML arXiv（4c 多篇）** — 標題+ID 經查存在，但**作者列表暫定**，引用前確認。
- **Bai 2008 — PCA sky-subtraction**. ChA&A 32,109（2008ChA&A..32..109B）。作者全名未確認（疑為同一 Z.-R. Bai）。
- **LAMOST 2D sky-background (PASA AS11071)** 與 **KICA-based LAMOST (IEEE 8564351)** — 文章存在，作者/年代未完全確認。

---

## 給本專案的重點結論
1. **預測 sky** 陣營成熟且多元：經驗(sky fiber/nod-shuffle/Kelson)、資料驅動(NMF)、物理模型(Cerro Paranal/skycorr)、近年 1 篇 ML(SMI 2025)。
2. **預測 residual** 幾乎等同 **PCA 家族**：源頭 Kurtz&Mink 2000 → Wild&Hewett 2005 → **ZAP 2016(IFU 標準)**；變體 CubePCA、Hart、VIPERS；DESI/LAMOST/SAMI 把它當 pipeline 內的第二刀。
3. **明確空缺**：沒有「深度學習專門預測 residual」的論文（最近者為貝氏 Uzsoy 2025）。
4. 老師要走「預測 sky」→ 主力參考 = **skycorr(Noll 2014) + MUSE DRP(Weilbacher 2020) + MaNGA(Law 2016) + Kelson 2003**，物理模型可用 **Noll 2012 / SkyCalc** 當天空先驗。
</content>
