# SExtractor 偵測參數參考(本專案遮罩流程)

> 本專案的源偵測/遮罩流程:**SExtractor 對白光影像偵測**(教授工作流)。
> 工作夾:`src/skymodel/SExtractor/`(用法見其 README);下游:`src/skymodel/step3_sky_basis.py`(界定 blank)與 `src/skymodel/step5_fit_spaxels.py`(界定源區)。
>
> **分工**:
> - 這份文件 = 各偵測參數的**通用物理意義**與「如何從任一 cube 的資料/header 推出參考值」。
> - Haro11 這批資料「為什麼這樣設」的物理推理,見
>   [`docs/segmentation-parameters-explained.md`](./segmentation-parameters-explained.md)。
> - 核心原則見 [`CLAUDE.md`](../CLAUDE.md) Principle 2 與「Operational Checklist」表。
>
> **基線與參考值的位階(Principle 2)**:`default.sex` 是**教授提供的權威基線,原封使用**;
> 任何參數想試不同值,用命令列覆寫(例:`-DETECT_THRESH 2.0`),覆寫屬於實驗。
> 本文的「自推參考值」欄**僅供理解物理意義與討論之用,不是對基線的糾正**。

---

## 1. 偵測參數與物理意義

背景估計原理(SExtractor `BACK_*` 家族):把影像切成 `BACK_SIZE` 的方格,每格 sigma-clip
排掉源後估當地背景與 RMS,再以 `BACK_FILTERSIZE` 格的中值濾波平滑格點、內插回每像素。
**它會自動排源估雜訊**,所以估出的 RMS 是「乾淨天空」的 σ——這正是偵測門檻該用的 σ
(若不排源、直接對整張含源影像估 σ,σ 會被源拉高、門檻跟著失真)。

| 關鍵字 | 物理意義 | 基線(`default.sex`) | 調大 / 調小的效果 | 自推參考值(討論用) |
|---|---|---|---|---|
| `DETECT_THRESH` | 偵測門檻,單位 = 背景 RMS 的倍數(σ);像素要超過 `thresh × σ` 才參與偵測 | **1.0** | 高→只抓亮源、漏暗源;低→抓到更暗但假陽性增(見 §3) | 2σ(假陽性 2.3%) |
| `DETECT_MINAREA` | 一個偵測至少要有幾個相連超門檻像素才算源 | **10** | 高→濾掉小雜點但可能漏小源;低→撿到雜訊碎點 | ≈1 個 PSF 面積 = π(FWHM/2)² ≈ 13 px |
| `FILTER` / `FILTER_NAME` | 偵測前的平滑核(matched filter,見 §2);`default.conv` = 高斯 FWHM≈2 px | **Y / default.conv** | 核寬大→壓雜訊、利暗延展源,但糊小結構;核寬小→保細節但壓噪不足 | 高斯 FWHM ≈ seeing ≈ 4 px |
| `BACK_SIZE` | 背景方格邊長(px) | **64** | 大→背景平滑、不吃延展源,對背景梯度反應鈍;**小→把大於格子的延展源當背景減掉**(見 §4) | > 最大要保留的物件(暈 Ø≈226 → ≥256 或全域) |
| `BACK_FILTERSIZE` | 背景格點中值濾波視窗(格) | **3** | 大→壓掉被亮源污染的格點,背景更糊 | 3(標準) |
| `DEBLEND_NTHRESH` | deblend 在 [門檻,峰值] 間切幾層找子峰 | **32** | 多→更能分開靠近的源 | 32(標準) |
| `DEBLEND_MINCONT` | 子峰占母源通量的最小比例才算獨立源;1.0 = 關閉 deblend | **0.005** | 高→傾向不拆;低→積極拆 | 0.005(標準) |
| `CLEAN` / `CLEAN_PARAM` | 清除亮源翼/雜訊造成的假偵測 | **Y / 1.0** | 大→清得更兇 | 預設 |

事後膨脹(遮罩安全邊界)不屬 SExtractor 本體,由下游以 `scipy.ndimage.binary_dilation` 做;
參考尺度 ≈ 1×seeing(≈4 px)。

---

## 2. matched filter:為何核 FWHM ≈ seeing(與對延展暈的限制)

- **原理**:偵測「已知形狀的訊號 + 白雜訊」時,S/N 最大化的濾波器就是**與訊號同形狀的核**
  (matched filter 定理)。天文點源被大氣糊成 PSF,故點源偵測的最佳核 ≈ **PSF(高斯,FWHM = seeing)**。
- **為何是 seeing 這個尺度**:seeing FWHM 是影像裡「真實結構的最小尺度」——比它更細的變化不可能是
  真天體。在這個尺度平滑,最大程度壓雜訊、又不糊掉任何真結構。核太小壓噪不足;核太大糊掉位置與形狀。
- **對延展暈的限制**:matched filter 對**點源**最佳;Haro11 的 Hα 暈是延展、低表面亮度結構,尺度遠大於
  seeing。核 = seeing 能把暈的每像素 S/N 抬高,但**不是大尺度結構的嚴格最佳核**。抓暗暈的正解是
  「matched filter 抬 S/N + 夠大的背景框(§4)」,而不是把門檻壓到雜訊以下(§3)。

---

## 3. 門檻的統計意義

門檻對應「純雜訊像素被誤判成源」的假陽性率(高斯單尾):

| 門檻 | 純雜訊超過的機率 |
|---|---|
| 0.75σ | ≈ 23% |
| 1σ | ≈ 16% |
| 1.5σ | ≈ 6.7% |
| 2σ | ≈ 2.3%(天文常用 2–5σ) |

低門檻換到更暗的偵測極限,代價是假陽性;對「抓暗延展暈」而言,降門檻不是首選槓桿——
先用 matched filter 抬 S/N、配足夠大的背景框,才是既定做法(§2、§4)。
現行基線 1.0σ 與雙向門檻實驗(1σ/2σ)是**教授指定、探索中的工作值**;本表僅提供統計背景,
不構成對工作值的評判。

---

## 4. `BACK_SIZE` 的失敗模式:背景框必須大於要保留的物件

- **失敗模式**:`BACK_SIZE` 若**小於**延展物件,背景方格整個泡在物件裡,會把**物件本身當成背景**
  估出來並減掉 → 物件消失、**無論門檻多低都偵測不到**。
- **規則**:`BACK_SIZE` 要大於你想保留的最大天體直徑;巨大延展暈就用全域背景或 `≥ 物件直徑`。
  Haro11 暈 Ø≈226 px → 參考值 256(此點與 CLAUDE.md 檢查表一致,基線 64 與參考值的差異
  屬檢查表既載的「與教授討論」事項)。

---

## 5. 逐檔推導配方(任一新 cube 如何從 header/資料推出參考值)

| 參數 | 由什麼推 | 公式 | Haro11 值 |
|---|---|---|---|
| pixel scale | header `CD1_1` | `√(CD1_1²+CD2_1²)×3600`(deg→arcsec) | 0.20″/px |
| seeing FWHM(px) | **實測 cube 內星點 PSF**(見下) | `median(2.3548·√(a·b))` over 星點 | **≈4.06 px ≈ 0.81″** |
| 平滑核 FWHM | = seeing FWHM | — | ≈4 px |
| 偵測門檻 | 統計標準(§3) | — | 2σ(參考) |
| MINAREA | 1 個 PSF 面積 | `π(FWHM/2)²` | ≈13 px |
| 膨脹(下游) | 1×seeing | `round(FWHM)` | ≈4 px |
| BACK_SIZE | > 最大物件(§4) | 取 ≥ 物件直徑的 2 的次方 | 256(或全域) |

**seeing 怎麼來(優先序)**:

1. **首選——直接從 cube 內的星點量 PSF FWHM**:做去發射線的連續譜白光影像,抽源後對每源以
   二階矩算 `FWHM = 2.3548·√(a·b)`,只留緊緻(FWHM<8 px)、圓(b/a>0.6)、夠亮的星,取中位數。
   本批 Haro11 實測 **≈4.06 px = 0.81″**(10 顆星,16–84% 範圍 3.58–4.77 px)。
2. **無星可用時的退路——header 代理**:`ESO OCS SGS AG FWHMX/Y MED`(自動導星)≈0.89″≈4.4 px、
   `ESO TEL AMBI FWHM`(DIMM)≈0.94–0.96″≈4.7 px;與實測一致指向 ≈4 px。
3. ⚠️ **不可盲用**:`ESO QC EXPCOMB FWHM MEDIAN` 在本批資料(`Haro11_nosky.fits`、
   `Haro11_NEpointing_esonosky.fits`)**= 0.0(未填)**,直接讀會得到 0-px 核;
   `ESO OCS SGS FWHM *` 同樣為 0.0。任何自動讀取都要加 `fwhm > 0` 的防呆。

---

## 6. 交叉引用與參考文獻

- [`docs/segmentation-parameters-explained.md`](./segmentation-parameters-explained.md) — Haro11 這批資料的參數物理推理。
- [`CLAUDE.md`](../CLAUDE.md) — Principle 2 與 Operational Checklist(參數位階:教授基線 > 自推參考)。
- `src/skymodel/SExtractor/` — 現行工作夾(`default.sex` 基線、det 影像重壓公式、批次腳本)。

**權威來源**:

- Bertin, E. & Arnouts, S. (1996). *SExtractor: Software for source extraction.* A&AS, 117, 393–404.
  doi:10.1051/aas:1996164 —— 原始演算法(背景、分割、deblend、CLEAN)。
- SExtractor 參數手冊:https://sextractor.readthedocs.io/ 。
- 舊版文件(sep 完整 API 對照,sep 流程已不用)存於 git 歷史(`docs/sep-sextractor-parameters.md`)。
