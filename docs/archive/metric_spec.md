> **封存中,不是現行的規格。**
> 這份文件定義的是評估 ZAP 扣天空的五個指標。M1 做完了,M2/M3/M5 只有規格沒有實作,
> M4 當初就標著暫緩。目前專案的主線是 sky reconstruction(`src/skymodel/`),不是 ZAP
> 對照;要恢復比較時再回來看這份。
>
> **恢復前必須知道:它依據的 ZAP 產物已經不在磁碟上了。**
> 第 3 節指名的 `results/zap/cubes/{target}_maskfrom-{masksrc}/{zap,sky,var}.fits`
> 四個全視場 run 已刪除。遮罩(`results/zap/masks/`)與輸入 cube 都還在,所以可以用
> `src/zap/zap.py` 重跑,一個 run 約 65 分鐘、記憶體峰值 43.7 GB。

# Metric Spec — ZAP 扣天空評估指標（Haro11 / MUSE）

本檔是評估指標的**規格**。每個指標一經定案即以最終形式寫入；本檔不保留草稿、前後比較或待選項。

---

## 1. 評估原則

扣天空的評估必須同時證明兩件事，缺一不可：

- **Removal**：天空被乾淨扣除。
- **Preservation**：源沒有被過度扣除吃掉。

只看殘餘不足以判斷：過度扣除會**同時**壓平殘餘與削掉源訊號，因此每一個 removal 指標都必須有對應的 preservation 指標並列。

---

## 2. 參考文獻

| 代號 | 文獻 | arXiv |
|---|---|---|
| ZAP | Soto et al. 2016, MNRAS 458, 3210 | 1602.08037 |
| W20 | Weilbacher et al. 2020, A&A 641, A28（MUSE pipeline） | 2006.08638 |
| Wis16 | Wisotzki et al. 2016, A&A 587, A98 | 1509.05143 |
| Lec17 | Leclercq et al. 2017, A&A 608, A8 | 1710.10271 |
| WH05 | Wild & Hewett 2005, MNRAS | astro-ph/0501460 |
| SP10 | Sharp & Parkinson 2010 | 1007.0648 |

---

## 3. 資料與對照基準

- **原始 cube**（唯讀）：`data/Haro11_{nosky,wsky}.fits`，含 `DATA` + `STAT`（STAT = 逐 voxel 變異，MUSE pipeline 傳播的）。
- **ZAP 產物**：`results/zap/cubes/{target}_maskfrom-{masksrc}/{zap,sky,var}.fits`（`zap` 的 STAT 為原始照抄，未重算）。
- **對照基準（真值）**：`nosky` raw = MUSE 官方扣天空的結果。
- **天空線**（removal 用，與源無關）：[OI] 5577.339 Å、[OI] 6300.304 Å、OH 帶（W20 zoom 在 OH 7-3 帶 ~8760 Å）。
- **源線**（preservation 用，Haro11 觀測系 z=0.0206）：Hα 6698 Å、[NII] 6683 / 6719 Å。
- **STAT 校正因子**：MUSE 的 STAT 低估孔徑量的真實雜訊，measured/expected ≈ **1.5**（範圍 1.2–2.5），因 cube 重取樣造成鄰近像素相關性（Wis16 §3.2.5）。凡以 √STAT 當雜訊底，均乘 1.5。

---

## 4. 指標

### M1 — Sky-subtraction residuals（殘餘天空譜）

**量**：空白（無源）spaxel 扣天空後的殘餘 flux，逐波長，對波長。

**物理意義**：空白區扣天空後應只剩雜訊、逼近 0。殘餘偏離 0 表示天空未扣淨；殘餘轉**負**表示過度扣除。

**blank 的定義 = valid & ~source**：只取遮罩外（非源）**且**在視場（FoV）內的 spaxel。FoV 外的 spaxel 在 raw cube 是 NaN、但 ZAP 會填成有限值（≈0），必須以 valid（raw 全波長 nansum≠0）排除，否則約 2.8 萬個邊緣 spaxel 會把統計往 0 拉偏。**用全部 valid blank（不取樣）**，逐波長對 spaxel 收斂成一個值 → 一條殘餘光譜（比取樣更精準、可重現）。

**統計量**：MUSE 慣例用 **median**（W20，對離群穩健）；ZAP 慣例 **mean 與 median 各畫一張**（mean 是 Soto 原文統計量，但在大遮罩上被離群 spaxel 主導、天空線處尖刺；median 乾淨）。

**輸出兩種慣例的圖**，畫同一個量，各只疊該慣例的判準（不交叉）：

#### 圖 A — MUSE 慣例（依 W20 Fig 15）
- x 軸：波長 [Å]；另附 5577 與 OH 帶（~8760 Å）的 zoom 子圖。
- y 軸：殘餘 flux [10⁻²⁰ erg s⁻¹ cm⁻² Å⁻¹]，**linear、以 0 為中心、共用 ±5 尺度**。
- 曲線：`{target}+ZAP`（median）、`nosky` raw（MUSE 標準，對照）。
- 判準帶：原始天空（`wsky` raw 的 mean）的 **±1%（黑）/ ±5% / ±10%** 包絡。
- 及格：連續譜殘餘落在 **±1%** 內、強天空線以外落在 **±5%** 內（目標 2%）。

#### 圖 B — ZAP 慣例（依 ZAP Fig 1）
- 版面：**兩 panel** —— 左 `Standard processing (MUSE pipeline) = nosky raw`、右 `ZAP = {target}+ZAP`（＝ Soto Fig 1「標準 vs ZAP」對照）。
- x 軸：波長 [Å]。
- y 軸：**雙軸** —— 左軸殘餘 flux、右軸原始天空 flux（灰線，`wsky` raw）。
- **無誤差帶**：ZAP Fig 1 與 W20 Fig 15 皆未畫任何雜訊/Poisson 帶；圖上的灰線是**原始天空譜本身**，非誤差包絡。
- 及格：殘餘遠小於右軸天空、且不轉負（不過扣）。

**尺度規定**：殘餘一律 **linear**（殘餘有正負號，log/symlog 無法顯示負值，會遮蔽過度扣除）；y 軸**固定範圍**（不自動縮放，四個 run 可直接互比）：MUSE 圖 ±5、ZAP-mean 圖 ±40、ZAP-median 圖 ±8、右軸天空 0–1500。log 僅用於畫**原始天空譜本身**（恆正、跨數量級）。

**計算**（`eval_common.py`，結果快取 `results/zap/blankstats/`）：對所有 valid blank，逐波長取 mean 與 median。
- 殘餘譜 = `zap` cube；標準/真值譜 = `nosky` raw；原始天空譜 = `wsky` raw（＝該視場的天空，% 包絡基準對所有 run 一致，不因 target 退化）。
- valid 由 raw(mask 來源 cube) 全波長 nansum≠0 決定。
- （不再用 `blanks.npz` 的 `(by,bx)` 取樣；`blanks.npz` 只保留亮源座標 `(sy,sx)` 供 M3。）

**命名**（run 由資料夾承擔，檔名帶指標+慣例）：
- `results/zap/cubes/{target}_maskfrom-{masksrc}/fig_M1_muse.png`
- `.../fig_M1_zap_mean.png`、`.../fig_M1_zap_median.png`

**範圍**：每個 run 各自出圖（四個 run 皆可）。

---

### M2 — Noise spectrum（雜訊譜）

**量**：空白 spaxel 殘餘的**散布（robust rms）**，逐波長，對波長。抓 M1（中位/偏移）看不到的失敗：ZAP 灌入或吸走雜訊。

**物理意義**：扣天空不應改變雜訊；殘餘散布應落在統計雜訊底。
- rms ≈ 底 → 理想（只剩統計雜訊）。
- rms > 底 → 欠扣（殘留天空結構）。
- rms < 底（尤其強天空線波長）→ 過度扣除 / 去噪，源訊號被吸走（ZAP §5）。這是 preservation 警訊，只有看散布才抓得到。

**對照對象**：`{target}` raw（扣前） vs `{target}+ZAP`（扣後）。

**依 WH05 Fig 4** 複製（唯一畫 rms-vs-波長且扣前/後對照的圖；SDSS 光纖，概念移到 MUSE 空白 spaxel。ZAP 與 W20 皆無此圖）。**兩個堆疊 panel**：

#### 上 panel
- x 軸：波長 [Å]（MUSE 全段 4750–9350），linear。
- y 軸：robust rms [10⁻²⁰ erg s⁻¹ cm⁻² Å⁻¹]，linear。
- 曲線：`{target}` raw、`{target}+ZAP`。

#### 下 panel
- x 軸：同上。
- y 軸：rms ÷ (√STAT × 1.5)，無因次，linear。
- 及格：**≈ 1 且平坦**。> 1 = 欠扣（殘留 OH 凸起）；< 1 = 過扣（凹陷、去噪）。

**rms 定義**（WH05 §2.2.1 robust）：每波長取空白 spaxel 的 `|flux − median|` 的 **67 百分位**（≈ 1σ，對離群穩健）。

**尺度規定**：rms 恆正，y 用 **linear**（照 WH05）。

**計算**：對所有 valid blank（＝ valid & ~source，同 M1，用 `eval_common`）。
- rms_raw(λ) = 67pct( |`{target}` raw − median| ) across blanks。
- rms_zap(λ) = 67pct( |`{target}+ZAP` − median| ) across blanks。
- 雜訊底(λ) = √( `{target}` STAT 在 blanks 的逐波長 median ) × 1.5。
- 下 panel = rms_raw / 底、rms_zap / 底。

**命名**（照 M1）：
- 檔案：`results/zap/cubes/{target}_maskfrom-{masksrc}/fig_noise-spectrum.png`。
- 標題（兩行）：主 `Noise spectrum`；副（小字）`target = {target} · mask from {masksrc}`。

**範圍**：先做單一 run。

**依據**：WH05 Fig 4（astro-ph/0501460）；雜訊底概念 ZAP Fig 1、SP10；STAT×1.5 見 §3。

### M3 — Source Hα fidelity（源亮核保真）

**量**：源亮核的 **1″ 孔徑積分光譜**，扣天空前後疊放；並量 Hα 線強度（EW）是否保留。測**亮核**（延展/faint 由 M4 負責）。

**物理意義**：扣天空不應吃掉源的發射線。`{target}+ZAP` 的光譜應貼合真值；Hα EW 應 ≈ 真值。EW 明顯 < 真值 = 源被過度扣除。

**對照對象**：`nosky` raw（MUSE 真值） vs `{target}+ZAP` vs 原始天空（僅 `wsky` 有）。

**依 ZAP Fig 6 複製**（源保真的公認圖；原圖是 5 源 gallery，我們只有一個星系 → 一個孔徑）：
- 版面：單一光譜 panel。
- x 軸：波長 [Å]，**全段 4750–9350，無 zoom**，linear。
- y 軸：flux [10⁻²⁰ erg s⁻¹ cm⁻² Å⁻¹]，linear。
- 孔徑：**1″ 直徑圓**（半徑 2.5 px），置於最亮點 (237, 315) = Haro11 亮核。
- 曲線：`nosky` raw、`{target}+ZAP`、原始天空（僅 `wsky`）。

**定量（依 WH05 Fig 12 + Table 1 的 EW 不變性）**：量 Hα 的 equivalent width，比 `{target}+ZAP` 對真值。
- EW(Hα) = Σ_line ( F − F_cont ) / F_cont · Δλ，line window = 6692–6708 Å（`HALPHA_LINE_WINDOW`，僅 Hα，避開 [NII]）。
- 連續譜 F_cont：兩側各一窗，**皆避開 [NII]6548=6682.9 與 [NII]6583=6719.1**：左 **6655–6678**、右 **6730–6758**。
  （＝修正舊 `HALPHA_FLUX_CONTINUUM` 左窗 6660–6688 切到 [NII]6548 的問題。）
- 報告 **EW_zap / EW_truth**（≈ 1 = 保住）：標在圖上一角，並進純量摘要表。

**計算**：對 `zap` 與各 raw cube，取以 (237,315) 為圓心、半徑 2.5 px 內 spaxel 的 flux 加總 → 孔徑光譜；各自量 EW(Hα)。

**命名**（照 M1/M2）：
- 檔案：`results/zap/cubes/{target}_maskfrom-{masksrc}/fig_source-halpha.png`。
- 標題（兩行）：主 `Source Hα fidelity`；副（小字）`target = {target} · mask from {masksrc}`。

**範圍**：先做單一 run。

**依據**：ZAP Fig 6（arXiv:1602.08037，1″ 孔徑、full range、linear、pipeline/ZAP/sky 三曲線）；定量 WH05 Fig 12 + Table 1（EW 不變性）。

### M4 — Extended Hα surface-brightness profile（延展暈徑向剖面）〔⏸ 暫緩 · 之後討論〕

> **狀態：暫緩，尚未定案。** 以下為已查證的調查紀錄（Wisotzki Fig 4 圖法 + PSF 參數文獻），恢復討論時直接沿用；PSF 取法（擬合前景星 vs 固定 β=2.8）與是否納入 M4 待定。

**量**：以源為心的**方位平均 Hα 表面亮度**，對半徑。測**延展暈（faint）**是否被 ZAP 保住 —— PCA 扣天空最容易過扣延展 faint 訊號之處（M3 管亮核，M4 管延展）。

**物理意義**：`{target}+ZAP` 的 SB(r) 應貼合 `nosky` 真值；在大半徑仍高於 PSF（= 真延展、非點源翼）與 1σ 極限（= 真偵測）。ZAP 的 SB(r) 掉到真值以下 = 延展暈被過扣。

**對照對象**：`nosky` raw（真值）、`{target}+ZAP`、PSF 剖面、1σ 偵測極限。

**依 Wisotzki 2016 Fig 4 複製**（方位平均徑向 SB 剖面；方法與譜線無關，低紅移 Hα CGM 有先例 Dutta 2024、Chung/Dey 2019）：
- x 軸：半徑 [arcsec]，linear。
- y 軸：Hα 表面亮度 [erg s⁻¹ cm⁻² arcsec⁻²]，**log**（負值標三角形）。
- 環：同心圓、**0.2″ = 1 spaxel 寬**、方位平均（排除遮罩/壞點）、以 Haro11 核 (237,315) 為心。
- 曲線：`nosky` raw、`{target}+ZAP`、PSF 剖面、1σ 極限。

**1σ 偵測極限**（Wisotzki 經驗法）：在 ~100 個空白位置跑同樣的環抽取，每半徑 bin 取 (Q3−Q1)/1.35 = σ_eff。**不用 STAT cube**（MUSE STAT 低估相關性雜訊）。

**PSF**（circular Moffat）：
- **β**：視場有孤立未飽和前景星 → 擬合（β 自由）；否則**固定 β=2.8**（WFM-NOAO 標準，B17/Leclercq）。
- **FWHM(λ) = a + b·λ**（線性、往紅端變小）：有星則擬合；否則以 header QC seeing 錨定（`wsky` 的 `EXPCOMB FWHM MEDIAN`=1.24″；⚠️ `nosky` 該值未填 =0，且 wsky 值跨曝光 0.70–1.92″，故僅當錨點、不整段套單值）。
- **敏感度**：延展暈 vs PSF 的結論須跑 **β ∈ [2.5, 3.0]** 敏感度並回報（β 控制 PSF 翼，直接影響「延展 vs 點源翼」判定）。

> **PSF 參數（文獻紀錄，供參）**：seeing-limited MUSE WFM 的 Moffat β 群集 **2.5–2.8**（B17=2.8 固定、HDFS Bacon+2015=2.6 擬合、一般 seeing-limited ~2.5；AO 的 Fusco+2020 擬合 2.3–2.7）。β 隨波長視為固定；FWHM 隨波長線性下降（UDF 0.71″→0.57″）。B17 的 2.8 本身即來自 **seeing-limited WFM-NOAO**（MUSE UDF 2014–2016，AO 未上線），與 Haro11 同 regime，故可用。取 PSF 的標準做法：有星就擬合 Moffat（PampelMuse/mpdaf），無星則固定 β=2.8 只擬合 FWHM(λ)（B17）。

**計算**：對 `zap` / `nosky raw` cube 做連續譜扣除的 Hα 窄帶影像 → 以核為心方位平均 → SB(r)；1σ 用空白區同法估；PSF 用 header/擬合建 Moffat 剖面。

**命名**（照 M1–M3）：
- 檔案：`results/zap/cubes/{target}_maskfrom-{masksrc}/fig_radial-halpha.png`。
- 標題（兩行）：主 `Extended Hα surface-brightness profile`；副（小字）`target = {target} · mask from {masksrc}`。

**範圍**：先做單一 run。對到既有 CGM Hα 分析的 fig7（此為其正式規格版）。

**依據**：Wisotzki 2016 Fig 4（arXiv:1509.05143）、Leclercq 2017（1710.10271）；PSF：Bacon+2017（1710.03002, β=2.8）、Bacon+2015（1411.7667, β=2.6）；Hα CGM 先例 Dutta 2024（2410.05392）、Chung/Dey 2019（1904.07874）。

### M5 — Source mask diagnostic（源遮罩診斷）

**角色**：**診斷 / 文件**，非驗證 metric，**無及格線**。交代「哪些 spaxel 被遮、以及用不同 cube 建差多少」。

**對照對象**：`sep_from-nosky/mask.fits`（≈41%） vs `sep_from-wsky/mask.fits`（≈36%），疊在同一影像上比。

**物理意義**：遮罩是 ZAP 成敗關鍵（遮太小 → 源被當天空學走、掉 70%）。展示遮罩涵蓋範圍與 mask 來源差異。

**版面（一張圖、左右兩 panel，共用 `nosky` 底圖）**：
- 左 panel：**Hα 窄帶影像**（nosky）為底 —— 看遮罩貼不貼合 Hα 發射（遮罩即用 Hα 偵測建的）。
- 右 panel：**白光影像**（nosky，整段 nansum）為底 —— 看遮罩相對恆星連續光（Hα 暈延伸超過星光，遮罩會超出一圈）。
- 兩 panel 都疊**兩條遮罩輪廓**：`nosky`-built（一色）、`wsky`-built（另一色）。灰階底圖，輪廓用對比色。

**標註**：各遮罩覆蓋率（nosky 41% / wsky 36%）、最遠半徑。

**命名**（屬遮罩，放 masks/）：
- 檔案：`results/zap/masks/fig_source-mask.png`。
- 標題（單行）：`Source mask (sep) — built from nosky vs wsky`。

**計算**：Hα 窄帶 = 線內 − 連續譜（共用 settings `halpha_narrowband_image`）；白光 = 整段 nansum；輪廓 = 兩張 mask 的 0.5 等值線；底圖皆取自 `nosky`（最乾淨、可公平共用）。

**依據**：無文獻圖（ZAP 無遮罩圖）；純為可重現性交代，屬診斷，不列 removal/preservation 判準。

### 純量摘要表（Scalar summary table）

**用途**：把 M1–M3 的關鍵數字濃縮成一張終端機/報告表；不看圖也能判「扣乾淨 + 源沒被吃」。**removal 與 preservation 並列**（原則1）。

**版面**：欄 = 四個 run（2×2，`{target}_maskfrom-{masksrc}`）；基準 = `nosky` raw。

| 列 | 定義 | 來自 | 及格 |
|---|---|---|---|
| sky 5577 殘餘（% of sky） | 空白區中位殘餘 ÷ 原始天空 @5577.339 Å | M1 | < 5%（目標 2%） |
| sky 6300 殘餘（% of sky） | 同上 @6300.304 Å | M1 | < 5% |
| sky OH 殘餘（% of sky） | 同上 @OH 帶（波長見 M1；現行 settings 用 8400） | M1 | < 5% |
| 連續譜殘餘（% of sky） | line-free 波長的中位 | M1 | < 1% |
| 殘餘 RMS ÷ (√STAT×1.5) | line-free 波長的中位 | M2 | ≈ 1 |
| Hα EW 保留率 | EW_zap ÷ EW_truth（1″ 孔徑、亮核） | M3 | ≈ 1（100%） |

**M4（延展暈）暫緩** → 徑向剖面相關的數字（暈偵測半徑、大半徑 SB 保留率等）**待 M4 定案後再補**。

**判讀**：`wsky+ZAP` 應同時 removal 及格（前 5 列）+ preservation ≈ 100%（末列）。
