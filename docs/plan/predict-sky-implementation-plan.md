# 策略一「預測 sky」可動手實作計畫（基於 SMI 與 NMF 兩篇）

> 目標：對 MUSE cube 建出天空模型 `sky_model`，做 `object = data − sky_model`（策略一）。
> 兩個範本：**Kolganov 2023 (NMF 低秩建天空)** 與 **Zhang 2025 (SMI 深度學習、逐位置、共同 vs 專屬發射線)**。
> 摘要見 [`../literature/predict-sky-summaries.md`](../literature/predict-sky-summaries.md)。

---

## 0. ⚠️ 重要更正：`wsky − nosky` 不是 ground truth
**header 實證**（muse_scipost）：`wsky: skymethod=none`（沒扣天空）、`nosky: skymethod=model`（用 MUSE
內建 model 法、最暗 10% spaxel 扣天空）。所以：
```
wsky − nosky = MUSE pipeline「model」法估出並扣掉的天空模型  ≠ 真天空
```
- 它是**「現有策略一方法的產物」**，帶著該方法的假設與誤差（正是 ZAP 要清的殘餘）。
- ❌ **不要**把它當監督學習目標——那只是**複製 MUSE model 法（連同誤差）**，無法超越它。
- ✅ 它的正確角色：**要打敗/對照的基準（baseline）**；`wsky`=未扣天空的原始輸入。

### 真值/評估要怎麼來
- **真 ground truth → 用模擬**：自造 mock cube（已知合成天空＋已知星系＋雜訊），這才有真值可訓練/量誤差。
- **offset/blank sky 曝光**：近似參考（不同時間/方向）。
- **truth-free 殘餘指標（實務標準）**：扣完後空白區應是**平坦雜訊、無天光線殘餘** + **源通量保真**。
  → 評估扣得好不好**根本不需要真天空**（ZAP 等都這樣評）。

---

## 概念對照：光纖巡天 → MUSE IFU（把論文翻譯到你的資料）
| 論文(光纖) | 你的 IFU(MUSE) |
|---|---|
| sky fiber | 空白天空 spaxel（白光最暗的一群）|
| plate 上所有 fiber | 整片 spaxel 網格 (492×500 等) |
| Super Sky（平均 sky fiber）| 空白 spaxel 的平均/中位光譜 |
| 逐 fiber 估天空 | 逐 spaxel 估天空 |
| 光纖效率差 H | spaxel 間 throughput 差（用亮天光線歸一）|

---

## 評估指標（先定義，全程共用）— 放在 `src/skymodel/metrics.py`
對任何方法產生的 `sky_pred`：
```
residual = (wsky − sky_pred) − nosky      ← 與 pipeline 結果的差
```
1. **整體殘餘 RMS**（在空白區 vs 源區分開算）。
2. **天光線處殘餘**（5577 / 6300 / OH 8400 等；發射線是難點）。
3. **源通量保真度**：源區 `(wsky − sky_pred)` 不應比 nosky 系統性偏低（過扣）。
4. **藍端表現**（SMI 強調藍端）。
5. **SNR 回復**：用 STAT/實測雜訊算 SNR。
- **對照基準**：① pipeline 的 `nosky`；② **ZAP（策略二）** 跑一遍當對照，凸顯「預測 sky vs 預測 residual」差異。
- ⚠️ 用**實測雜訊**（空白區 DATA 標準差）而非純 STAT 當分母——記得 STAT 因相關雜訊低估 ~1.8×（見 C 組）。

---

## Phase 1 — 基線 Super Sky（要打敗的對象）｜0.5–1 天
**做法**（LAMOST/MaNGA 的標準法）：
1. 白光影像選最暗 ~30–50% 的有效 spaxel = 空白天空遮罩。
2. `super_sky(λ)` = 這些 spaxel 的中位光譜。
3. 逐 spaxel **縮放**：用一條強天光線（NOAO 用 5577；AO 檔用 OH）的強度比，修 throughput 差（skycorr/Han 的精神）。
4. `sky_pred = scale(x,y) × super_sky`，相減。
- **產出**：`src/skymodel/baseline_supersky.py` + 指標基準數字。
- **驗收**：殘餘 RMS、天光線殘餘——這是後面所有方法要超越的底線。

## Phase 2 — NMF 低秩天空模型（Kolganov 範本）｜2–4 天
**核心**：天空不是「一條」而是「少數幾個非負基底的線性組合」，更能吸收變動。
1. **連續譜 / 發射線分離**：對每條空白 spaxel，用大窗中位濾波分出連續譜與發射線（沿用我們 D 組做法）。
2. **建基底**：把空白 spaxel 的（發射線）光譜疊成矩陣 `A (Nspaxel×Nwl)`，用 `sklearn.decomposition.NMF` 取 k≈10–20 個非負成分 `C`。
3. **逐 spaxel 擬合**：對每個 spaxel（含源區），解非負最小二乘 `s ≈ Cᵀ·x`，**用 1/STAT 當權重**（inverse-variance，見 C 組），並**遮掉源的發射線波長**避免污染基底擬合。
4. `sky_pred = 連續譜模型 + Cᵀ·x`，相減。
- IFU 版的「2D 分離」：Kolganov 用沿縫平移消平坦天空；你用**空白 spaxel 建基底、源區只擬合天空成分**達成同樣的「天空 vs 源」分離（呼應我們驗證的「天空均勻、源局部」）。
- **產出**：`src/skymodel/nmf_sky.py`。**驗收**：殘餘應低於 Phase 1，尤其天光線處。
- **可選**：同時做 PCA 版對照（即 ZAP/Kurtz 的基底），比較 NMF（非負、~10× 成分）是否更好。

## Phase 3 — 逐位置空間建模（SMI 精神，但先免 DL）｜2–3 天
**動機**：天空有空間梯度（月光），單一 super_sky 抓不到。用 trend surface 捕捉梯度（Han 2023）。
1. 把天空拆成 **共同成分**（連續譜 + 全場共同發射線 `Ssm`）與 **位置專屬成分**（`So`）。
2. 對每個 NMF 成分的**權重 x**，用**只在空白 spaxel** 擬一個平滑 2D 曲面（如低階多項式/樣條）`x_k(y,x)`。
3. 在**源 spaxel** 用該曲面**內插**出當地天空權重 → 得逐位置 `sky_pred(y,x)`。
- **產出**：`src/skymodel/trend_surface_sky.py`。**驗收**：源區殘餘與梯度區是否優於 Phase 2。

## Phase 4 — 深度學習（SMI 範本；研究貢獻所在）｜2–4 週
> ⭐ 文獻空缺：目前**沒有 DL 預測 sky 用在 IFU/MUSE**——這就是題目空間。你又有 ground truth，能做監督式（比 SMI 更強）。

**路線 A（務實、推薦先做）：監督式 spaxel→sky（⚠️ 真值要用模擬）**
- ❗ 監督目標**不能**用 `wsky − nosky`（那是 MUSE model 法的估計，學它=複製它）。
- **正解**：在**模擬 mock cube** 上監督——輸入 = 含天空的 mock spaxel，目標 = **注入的已知合成天空**（真值）。
- 模型：1D-CNN 或 U-Net（譜方向卷積）。損失：源遮罩外算 MSE/Huber，**用 1/STAT 加權**。
- 資料切分：**按空間分塊**做 train/val/test（避免相鄰 spaxel 因相關雜訊洩漏，見 C 組）。
- 借 SMI 的 **calibration 模組**：對齊卷積後位移的發射線特徵。
- 在**真實資料**（wsky）上則用 truth-free 殘餘指標評估，並與 `nosky`(MUSE model) 對照看能否做得更乾淨。

**路線 B（忠於 SMI）：互資訊分離 shared/unique**
- 階段一最大化不同 spaxel 表徵的 MI → 共同天空 `Ssm`；階段二最小化 → 專屬 `So`。MI 估計用 MINE（Belghazi 2018）。
- 適合若你想主打「方法新穎性」；但路線 A 有 ground truth、較易上手且可能更準。

**MUSE 專屬注意**：天光線要隨**每個位置的 LSF** 變（A 組）；模型輸入可加入波長軸的 LSF 資訊，或先把譜歸一到共同 LSF。AO 檔有鈉缺口(5800–6000=0)要遮掉。
- **產出**：`src/skymodel/dl_sky/`（資料集、模型、訓練、推論）。**驗收**：贏過 Phase 1–3 與 ZAP，尤其藍端與天光線。

## Phase 5 — 比較與報告｜2–3 天
- 同一套指標跑：Phase1 / 2 / 3 / 4 / ZAP，對照 `nosky` 與 `sky_truth`。
- 出圖：殘餘譜、殘餘影像、天光線放大、SNR 回復、源通量保真。
- 結論：哪種「預測 sky」最佳、是否勝過「預測 residual(ZAP)」。

---

## 工具與環境（用 uv 安裝）
- **MPDAF**（MUSE 專用，data+var 成對搬運，自動誤差傳遞）— `uv pip install mpdaf`
- numpy / scipy / **scikit-learn**（NMF, PCA）/ matplotlib
- **pytorch**（Phase 4）
- 已有：astropy、sep、specutils、pymupdf

## 風險與決策點
1. **資料量**：DL 跨條件泛化需要更多 cube；目前僅夠 within-cube 驗證。→ 決定要不要去要更多 MUSE 曝光。
2. **相關雜訊**：評估與資料切分都要避開相鄰 spaxel 洩漏（C 組）。
3. **LSF 隨位置變**：減天光線準不準的關鍵；Phase 2+ 要處理。
4. **AO 鈉缺口**：用 AO 資料時要遮 5800–6000Å。
5. **過扣源**：源區擬合天空時務必遮源發射線 + 用 var 加權。

## 建議順序（crawl → walk → run）
**Phase 1（基線）→ Phase 2（NMF）→ Phase 3（trend surface）→ Phase 4（DL）→ Phase 5（比較）**。
每個 Phase 都產出可量化的殘餘數字，逐步逼近並超越 pipeline 與 ZAP。先把 1–3 做穩（傳統、快、可解釋），再投入 Phase 4 的研究貢獻。
</content>
