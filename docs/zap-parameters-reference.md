# ZAP 參數權威參考(Haro11 MUSE cube)

> 本檔是 ZAP(Zurich Atmosphere Purge, Soto+2016)在本專案的**權威參數手冊**:
> 解釋演算法、逐一定義 `zap.process` / `zap.SVDoutput` 的每個關鍵字、列出本專案目前實際生效的預設值,
> 並依 `CLAUDE.md` Principle 2 標示哪些預設需要科學驗證,最後給出「對任意新 cube 調參」的可重現流程。
>
> **來源真值(source of truth)**:本專案 vendored 的 ZAP 原始碼 `libs/zap/`。
> 版本 **2.1**(`zap.__version__ == '2.1'`;`git describe` = `2.1-6-g974231e`,commit `974231e`,CHANGELOG 最新條目為 `2.2.dev`)。
> 凡本檔陳述的「預設值 / 行為」皆逐一核對 `libs/zap/zap/zap.py`;與 readthedocs 或 Soto+2016 論文有出入者,**以 vendored 2.1 為準**並註記分歧。
>
> 相關文件:實務結論見 `docs/zap-experiment-log.md`;源遮罩(mask)參數的物理推導見 `docs/segmentation-parameters-explained.md`;核心原則見 `CLAUDE.md`。
> 實際呼叫點:`src/0706/step2_zap.py`;nevals 掃描診斷:`src/legacy/tune_nevals.py`。

---

## 0. 一句話定位

ZAP 用**重建天空**的方式扣天空:先做逐波長中位數扣掉 ~99% 的天空,再對「殘餘天空」在**源遮罩之外的純天空 spaxel** 上做 PCA/SVD,學出殘餘天空的 eigenspectra,選取適當數目的成分重建並扣除。ZAP 就是 `CLAUDE.md` Principle 1「sky reconstruction」的引擎:**天空基底只從 source-free spaxel 學**,亮源不參與。因此評估必須永遠同時看兩件事——**天空線殘餘下降 + 源通量保留**(Principle 1)。

---

## 1. ZAP 演算法(讓每個參數都有依據)

ZAP 對每根光譜(spaxel)依序做以下步驟(`Zap._run` → `_prepare` → `_msvd` → `optimize`/`chooseevals` → `reconstruct` → `remold`):

1. **NaN 清理(`clean`)** — 逐 spaxel 檢查 NaN。NaN 佔比 >25% 的 spaxel 整根剔除不處理;其餘 NaN 用 3×3×3 鄰域內插補上,處理完再把 NaN 塞回輸出 cube(`_nanclean`,`boxsz=1`、`rejectratio=0.25`)。

2. **零階天空扣除 zlevel(`zlevel`)** — 對每個單色層(monochromatic layer)取**中位數**(或 sigma-clip 平均)並扣掉。這一步就移除了約 99% 的天空訊號;剩下的是與儀器有關的殘餘(LSF、波長標定不連續、flat 誤差),它們在空間上呈現為 spaxel 間的起伏。

3. **連續譜濾除 continuum filter(`cftype` / `cfwidth`)** — 在扣 zlevel 後的 stack 上,對每根光譜沿波長做**中位數濾波**(視窗寬 `cfwidth` 個波長像素;實作為 `uniform_filter(width=3)` 後接 `median_filter(cfwidth)`)。目的是把「天體的連續譜」壓平、只留下窄的天空殘餘,好讓後續 PCA 學到的是**天空**而不是**源的連續譜**。濾掉的連續譜存在 `contarray`,殘餘存在 `normstack`。

4. **變異數正規化** — 每個 segment 內,把 `normstack` 除以其逐 spaxel 變異數,使 SVD 不被個別亮 spaxel 主導。

5. **切 sky segment(`SKYSEG`)** — 把波長軸切成數段,每段各自做 PCA。分段的物理理由:天空發射線由 OH、O₂、O I、Na I 等**成群的躍遷**產生,同一段波長內的天空殘餘在時間/空間上高度相關;而資料處理的瑕疵也「在短波長區間內同調」。**vendored 2.1 的預設是單一 segment**(見 §4.1)。

6. **PCA / SVD(每段)** — 把該段的 `normstack`(維度 ≈ 波長像素 n × spaxel 數 m)做奇異值分解。列(spaxel)是樣本、行(波長)是特徵,得到一組**天空 eigenspectra**(沿波長的譜形)。因為 masked 的源 spaxel 已被設成 NaN、在 `_extract` 時被剔除,**這組基底只由純天空 spaxel 張成**(sklearn `PCA`)。

7. **選成分數(`nevals` / `optimize`)** — 決定用前幾個 eigenspectra 重建。變異曲線(explained variance vs 成分數)有兩段:**前段陡降**(頭 ~10 個成分在扣真正的天空殘餘)、**後段近線性**(高階成分開始吃掉源與雜訊)。ZAP 自動法找兩段的轉折(`optimize`:取前 25% 成分的一階導數,找它落回 `mean − 5σ` 的交點;這是論文「變異曲線二階導數趨零的拐點」的實作)。`nevals=[]` 時走自動法;給定數值則固定。

8. **重建並扣除(`reconstruct` / `remold`)** — 用選定的成分把殘餘天空投影/反投影重建成 `recon`,再從 stack 扣掉,塞回 cube 成 `cleancube`。

**為什麼成分數是成敗關鍵**:天空 eigenspectra 只在天空線波長被約束,在別處(源的發射線/連續譜)並不受約束。成分數一多,高階 eigenspectra 就開始擬合並扣掉**真正的源通量與線形** → 過扣、吃源。太少則天空殘餘扣不乾淨。這正是 `CLAUDE.md` Principle 1「不能只看殘餘」的根源:過扣會同時壓低殘餘**又**吃掉源。

**mask 的角色(sky reconstruction 的核心)**:`mask` 標 ≥1 的源 spaxel 會在建 SVD 前被設成 NaN、排除在天空基底之外(`_applymask`)。遮罩不足 → 源(尤其 Haro11 的延展 Hα 暈)混進天空基底 → 被當天空扣掉。本專案實測:白光 2σ 只遮 8% 造成 **70% 源流量損失**;改用 Hα 窄帶偵測 + 膨脹遮到 44% 後源保留 124%(見 `docs/zap-experiment-log.md` §2–3)。遮罩本身的參數推導見 `docs/segmentation-parameters-explained.md`。

---

## 2. `zap.process` 完整參數參考

實際簽章(逐字核對 `libs/zap/zap/zap.py`,ZAP 2.1):

```python
zap.process(cubefits, outcubefits='DATACUBE_ZAP.fits', clean=True,
            zlevel='median', cftype='median', cfwidthSVD=300, cfwidthSP=300,
            nevals=[], extSVD=None, skycubefits=None, mask=None,
            interactive=False, ncpu=None, pca_class=None, n_components=None,
            overwrite=False, varcurvefits=None)
```

| 參數 | 型別 / 單位 | 預設(2.1) | 意義與改變它的效果 | 物理 / 實務指引 |
|---|---|---|---|---|
| `cubefits` | str | (必填) | 輸入 cube 檔名。MUSE 讀第 1 個 extension(DATA)。 | **注意分歧**:論文/舊 readthedocs 稱 `incubefits`,vendored 2.1 的位置參數名為 `cubefits`。只接受單一檔名字串。 |
| `outcubefits` | str | `'DATACUBE_ZAP.fits'` | 輸出扣天空後 cube。用 `mergefits` 寫回原檔結構,**保留所有 extension(含 STAT)**。 | STAT 為**原封照抄**、未經 ZAP 傳遞(見 §5)。本專案設為 `.../zap.fits`。 |
| `clean` | bool | `True` | NaN 清理(即論文所稱 nanclean)。>25% NaN 的 spaxel 剔除,其餘內插;輸出時 NaN 塞回。 | 保持 `True`。MUSE cube 邊緣因大氣折射有大量 NaN;關掉會讓含任一 NaN 的 spaxel 完全不處理。 |
| `zlevel` | str | `'median'` | 零階天空扣除法:`'none'` / `'sigclip'`(3σ 截尾平均)/ `'median'`。 | `median` 穩健,是標準選擇。`sigclip` 更貴、遇強源較保險但一般非必要。`none` 只在天空已由外部扣除時使用。 |
| `cftype` | str | `'median'` | 連續譜濾波法:`'median'` / `'fit'`(deg-5 多項式)/ `'none'`。 | `median` 為 2.1 預設且最穩。`'fit'` 為 MUSE 特化(排除 pixel>3600 的紅端與 notch 區),**在紅端易失控**,非必要不用。`'none'` 不濾連續譜,僅適合無連續譜的場。 |
| `cfwidthSVD` | int(**波長像素**) | `300` | 建立 **SVD 基底**時的連續譜濾波視窗。 | 300 px。本 cube `CD3_3=1.25 Å/px` → **300 px = 375 Å**。此值大、對天空連續譜較穩。 |
| `cfwidthSP` | int(**波長像素**) | `300` | 計算**每根光譜特徵值/投影**時的連續譜濾波視窗。 | **⚠ 科學關鍵、`CLAUDE.md` 點名**。ZAP docstring 自述最佳範圍 **20–50 px**(=25–62.5 Å)「較能追蹤源」;預設 300 px 偏大。詳見 §3、§4.2。 |
| `nevals` | list / int | `[]` | 每段用幾個 eigenspectra 重建。`[]`→自動 `optimize()`;單值→所有段同值;長度=段數的 list→逐段指定。 | **⚠ 科學關鍵**。過多→過扣吃源;過少→天空殘餘。自動法通常合理,但**必須用雙指標驗證**(§4.3)。docstring 說「11 值的 list」是舊 11-段設計的殘留;2.1 單段時給單值即可。 |
| `extSVD` | Zap 物件 | `None` | 改用 `zap.SVDoutput(...)` 在**別的 cube**(offset sky / 別次曝光)算好的 SVD 基底。 | 多曝光 / filled field 用。**與 `mask` 互斥**(同時給會 raise `ValueError`:要用 mask 就得重算 SVD)。見 §4.4。 |
| `skycubefits` | str | `None` | 額外輸出「被扣掉的天空」= 輸入 − 輸出 cube(`writeskycube`)。 | 診斷用。本專案設為 `.../sky.fits`。 |
| `mask` | str | `None` | 2D FITS 遮罩:源標 **≥1**、天空標 **0**。源 spaxel 設 NaN、排除於天空基底外。 | sky reconstruction 的核心。遮罩品質直接決定成敗(§1、`segmentation-parameters-explained.md`)。 |
| `interactive` | bool | `False` | `True` 時回傳 `Zap` 物件、**不寫檔**,可用 `reprocess(nevals=...)` 快掃成分數。 | nevals 掃描用(見 `src/legacy/tune_nevals.py`)。 |
| `ncpu` | int | `None` | 平行處理程序數。`None`→`cpu_count()`(全部核心)。 | 依可用核心設定。本專案設 16;整張視場實測峰值記憶體 ~43.7 GB(`zap-experiment-log.md` 階段 H)。 |
| `pca_class` | class | `None` | 替換 PCA 實作類別。`None`→sklearn `PCA`。 | 進階;一般不動。 |
| `n_components` | float | `None` | 計算多少個 PCA 成分(非重建用的 nevals)。給值時 `ncomp = max(nwave_seg × n_components, 60)`。 | 進階;`None`→計算完整 PCA。改它只影響「算多少成分」的上限與速度,不等於重建成分數。 |
| `overwrite` | bool | `False` | 是否覆寫既有輸出檔。 | 本專案設 `True`。 |
| `varcurvefits` | str | `None` | 額外輸出各段 `explained_variance_` 曲線成 FITS table(`writevarcurve`)。 | **強烈建議開**:配合表頭 `ZAPNEV*` 檢視自動選的成分數是否合理。本專案設為 `.../var.fits`。 |

**寫入輸出表頭的 ZAP 關鍵字**(`_newheader`):`ZAPvers`、`ZAPzlvl`、`ZAPclean`、`ZAPcftyp`、`ZAPcfwid`、`ZAPnseg`、以及逐段 `ZAPseg{i}`(段的像素範圍)/`ZAPnev{i}`(該段用的成分數)。**跑完務必檢查 `ZAPnev*` 與 `ZAPnseg`。**

### 2.1 與 readthedocs / 論文的分歧(以 vendored 2.1 為準)

- **無 `pevals`、無 `optimizeType`**:ZAP 1.0 曾有這兩個關鍵字(百分比選成分、選擇模式)。2.x 已移除;自動選成分由 `optimize()` 方法(變異曲線導數法)取代。若外部教學提到 `pevals`/`optimizeType`,對本 vendored 版**不適用**。
- **無 `Zap.getzcube`**:task 提及的 `getzcube` 在 2.1 不存在。相關方法為 `make_cube_from_stack`、`make_contcube`、`writecube`、`writeskycube`、`writevarcurve`、`mergefits`、`reprocess`、`optimize`。
- **SKYSEG 預設為單段**:論文(1.0)描述 11 段;2.0 起預設**單段**(見 §4.1)。
- **`cfwidth` 預設 300**:論文描述 100 px(建基底)+ 20–50 px(算特徵值);2.0 起把預設併為 300(舊的 100/50 太小、造成紅端背景震盪,見 CHANGELOG 2.0)。

---

## 3. 相關頂層函式

| 函式 | 簽章(2.1) | 用途 |
|---|---|---|
| `zap.SVDoutput` | `(cubefits, clean=True, zlevel='median', cftype='median', cfwidth=300, mask=None, ncpu=None, pca_class=None, n_components=None)` | 在某 cube 上算好 SVD 基底,回傳可餵給 `process(extSVD=...)` 的 `Zap` 物件。多曝光/offset sky 用。注意這裡只有單一 `cfwidth`(對應 `process` 的 `cfwidthSVD`)。 |
| `zap.nancleanfits` | `(cubefits, outfn='NANCLEAN_CUBE.fits', rejectratio=0.25, boxsz=1, overwrite=False)` | 獨立跑 NaN 內插並寫檔。 |
| `zap.contsubfits` | `(cubefits, outfits='CONTSUB_CUBE.fits', ncpu=None, cftype='median', cfwidth=300, clean_nan=True, zlevel='median', overwrite=False)` | 獨立輸出連續譜扣除後的 cube(診斷連續譜濾波)。 |
| `zap.mask_nan_edges` | `(cube, outfile=None, plot=False, threshold=50, extname='DATA')` | 遮掉邊緣 NaN 過多(>threshold %)的 spaxel,避免這些未被扣天空的 spaxel 在輸出留下高殘餘。 |

`process` 內部流程:若給了 `mask`(或 `cfwidthSVD != cfwidthSP`),先呼叫 `SVDoutput(cfwidth=cfwidthSVD, mask=mask)` 用**遮罩後的 cube**建基底;再建 `Zap` 物件、以 `cfwidthSP` 做逐光譜連續譜濾波、沿用該基底投影重建。**因此 `mask` 只作用在 SVD 基底那一步**(zlevel 也在遮罩下算),而 `cfwidthSP` 決定實際扣天空時對每根光譜追蹤源的細緻度。

---

## 4. 本專案目前生效的預設,以及是否站得住腳

`src/0706/step2_zap.py` 只傳 `mask`、`ncpu=16`、`overwrite=True`(以及三個輸出路徑),其餘全走 ZAP 2.1 預設:

| 旋鈕 | 目前生效值 | 是否物理站得住腳(Principle 2) |
|---|---|---|
| `zlevel` | `'median'`(預設) | ✅ 標準穩健,無須改。 |
| `cftype` | `'median'`(預設) | ✅ 2.1 預設、最穩。不建議改 `'fit'`(紅端易失控)。 |
| `cfwidthSVD` | `300 px = 375 Å`(預設) | ✅ 建基底用大視窗合理。 |
| `cfwidthSP` | `300 px = 375 Å`(預設) | ⚠ **需科學驗證**。ZAP 自述最佳 20–50 px;300 px 對「亮緊緻核 + 延展 Hα 暈」偏粗。見 §4.2。 |
| `SKYSEG` | `[]` → **單段** 4750–9348 Å(預設) | ✅ 這正是 2.x 推薦預設(§4.1)。改成多段是科學決定,需驗證。 |
| `nevals` | `[]` → **自動** `optimize()`(整張視場實測選 53) | ⚠ **需雙指標驗證**。自動法在本資料運作正常(源保留 124%),但仍須看源保真而非只看殘餘。見 §4.3。 |
| `extSVD` | `None` → 用同一 cube 自建 SVD(單曝光重建) | ✅ 目前只有 1 個含天空 cube;多曝光時改用 extSVD(§4.4)。 |
| `clean` | `True`(預設) | ✅ 保持。 |
| `ncpu` | `16`(明給) | ✅ 依機器核心;調它不影響科學結果,只影響速度/記憶體。 |

> **`CLAUDE.md` Principle 2 警示(以下為既定指引,非草稿)**:
> `cfwidthSP` 與 `SKYSEG` 被 `CLAUDE.md`「Other」節點名為科學關鍵。其中 `SKYSEG` 目前值(單段)本身就是推薦預設、站得住腳;**`cfwidthSP=300`(預設)偏大,可能欠追蹤源的連續譜**,建議依 §4.2 評估 20–50 px。任何偏離目前值的更動(不論 `cfwidthSP`、`SKYSEG` 或固定 `nevals`)都**改變科學結果**,必須先確認是否 result-preserving;若否,視為科學決定,用 §4.3 的雙指標驗證後才採用。

---

### 4.1 SKYSEG(sky segment 邊界)

- **vendored 2.1 預設 = 單段**:`SKYSEG = []`,`Zap` 依 cube 的 λ min/max 取單一區段(本 cube 整段 4750–9348 Å,`ZAPnseg=1`)。
- **物理理由**:單段讓**整個波長範圍的天空線相關性**都被 PCA 利用,扣天空更乾淨;且大幅**降低殺掉發射線的風險**(多段會造成連續譜震盪、且逐段選成分數極敏感)。這是 2.0 起改單段的原因(CHANGELOG 2.0)。
- **舊 11 段邊界(僅供參考)**:`[0, 5400, 5850, 6440, 6750, 7200, 7700, 8265, 8602, 8731, 9275, 10000]` Å——按 OH / O₂ / Na I / [O I] 的天空線族與儀器響應斷點分組。
- **何時改**:只有在單段明顯扣不乾淨、且診斷指出不同波長區的天空殘餘行為差異大時才考慮。**改法**:`from zap.zap import SKYSEG; SKYSEG[:] = [...]`(就地改 list,不能重新賦值)。
- **驗證**:改段數會改變每段的成分選擇與連續譜行為,**不是 result-preserving**。必須用 §4.3 雙指標比較單段 vs 多段:天空線殘餘要降、源 Hα 通量與線形要保留、line-free 區不得灌入雜訊。

### 4.2 cfwidth / cfwidthSP(連續譜濾波視窗)

- **單位是波長像素,不是 Å**。本 cube `CD3_3 = 1.25 Å/px`,故換算:`300 px = 375 Å`;`20–50 px = 25–62.5 Å`。
- **權衡**:
  - 太小 → 濾波跟著源的連續譜/發射線起伏跑,把**真正的源連續譜當殘餘扣掉**(吃源)。
  - 太大 → 濾波過於平滑,**欠追蹤源**,源的連續譜洩漏進 `normstack`,可能被 PCA 學進天空基底。
- **兩個視窗分工**:`cfwidthSVD`(建基底)可用大視窗求穩(預設 300 合理);`cfwidthSP`(逐光譜投影)ZAP 自述最佳 **20–50 px**,因為這一步要細緻追蹤源、避免把源當天空。
- **本專案現況與建議**:`cfwidthSP` 目前吃預設 300 px(=375 Å),對 Haro11 這種「亮緊緻核 + 低表面亮度 Hα 暈」偏粗。**建議在 20–50 px 範圍(≈25–62.5 Å)實測**;但因 `CLAUDE.md` 點名此參數為科學關鍵,**更動屬科學決定**,須用 §4.3 雙指標確認源保留不變差、天空殘餘不變糟才採用。
- 若只是想單獨看連續譜濾波的效果,可用 `zap.contsubfits(cfwidth=...)` 輸出連續譜扣除 cube 檢查。

### 4.3 nevals / 成分數 —— 用雙指標驗證,不給魔術數字

- `nevals=[]` 走自動 `optimize()`;整張視場自動選 **53**,源保留 124%、天空線壓到 ~0–1.3(`zap-experiment-log.md` §3),自動法在本資料**運作正常**。
- **但成分數是源保真 vs 天空乾淨的直接權衡**:太多→過扣吃源;太少→天空殘餘。**`CLAUDE.md` Principle 1:絕不能只看殘餘**——過扣會同時壓低殘餘又吃源。
- **既有結論(勿重蹈)**:對已扣天空的 `nosky` 跑 ZAP 時,成分數從 3 掃到 55 **都救不了**過扣——因為問題在**輸入 cube 選錯**(nosky 沒天空可學),不是調 nevals(`zap-experiment-log.md` §2 階段 E、§4)。**先確認餵的是含天空的 `wsky`**,再談成分數。
- **驗證流程(可重現)**——用 `interactive` 一次算 SVD、`reprocess()` 快掃(範本:`src/legacy/tune_nevals.py`):

  ```python
  import zap
  zobj = zap.process("<wsky_cube>.fits", mask="<source_mask>.fits",
                     interactive=True, overwrite=True)   # 只算一次 SVD
  print("auto nevals =", zobj.nevals)
  for N in [3,5,8,10,12,15,20,25,30,40, int(zobj.nevals[0])]:
      zobj.reprocess(nevals=[N])
      # 對 zobj.cleancube 量三個指標(見下)
  ```

  在 source-free 的 blank spaxel 與源 spaxel 上同時量**三個指標**:
  1. **天空線 spatial std**(如 5577/6300/8400 Å):要**下降**到接近 MUSE `nosky` 真值。
  2. **line-free 純雜訊**(如 7000–7120 Å 的逐-spaxel RMS):**不得明顯上升**(門檻經驗值 ≤ 1.5× raw)。
  3. **源 Hα 積分通量**:**保留 ≥ ~98%**(相對 raw),線形不變。

  選能同時滿足三者、且成分數盡量高(天空扣最乾淨)的 N。三者的定義與門檻見 `src/legacy/tune_nevals.py` 與 `docs/metric_spec.md`。

### 4.4 ncpu 與 extSVD(多曝光)

- **`ncpu`**:設為可用核心數(不影響科學結果)。整張 499×559×3679 cube 記憶體峰值約 43.7 GB,請確認 RAM 足夠。
- **`extSVD`(多曝光 / filled field)**:目前本專案只有單一含天空 cube(單曝光重建)。**若日後有多次曝光或 offset sky frame**,在一個(遮好源的)frame 上算 SVD 再套到各 frame:

  ```python
  extSVD = zap.SVDoutput("offset_or_expo1.fits", mask="mask.fits", cfwidth=300)
  zap.process("science_expo.fits", outcubefits="out.fits", extSVD=extSVD)
  ```

  注意 **`extSVD` 與 `mask` 不能同時給**(要遮罩就得重算 SVD);offset frame 只需 2–3 分鐘短曝光即可。

---

## 5. 輸出與 STAT 的處理(重要)

- `outcubefits` 由 `mergefits` 寫成:**開原始輸入檔、只替換 DATA(第 1 extension)為 `cleancube`,其餘 extension(含 STAT)原封保留**。
- 因此 **STAT(逐 voxel 變異)是原始照抄、未經 ZAP 傳遞**。ZAP 的扣天空並未更新變異;若後續分析需要正確的誤差傳遞,須自行處理(此為已知限制,見 `zap-experiment-log.md` §7 與 step2 docstring)。
- `skycubefits` = 輸入 − 輸出(被扣掉的天空);`varcurvefits` = 各段 explained variance 曲線(診斷用)。

---

## 6. 交叉參考

- `CLAUDE.md` — Principle 1(sky reconstruction、雙指標)、Principle 2(參數須物理可辯護)、「Other」節點名 `SKYSEG`/`cfwidthSP` 為科學關鍵。
- `docs/zap-experiment-log.md` — 實測結論:餵對 cube(wsky)是關鍵、遮罩不足會吃源、成分數掃描救不了輸入錯誤、整張視場數字。
- `docs/segmentation-parameters-explained.md` — `mask` 的偵測參數(threshold 2σ、matched filter 核=seeing、bw>暈、minarea、dilation)的完整物理推導。
- `docs/metric_spec.md` — 評估指標定義。
- `src/0706/step2_zap.py` — 實際呼叫點。
- `src/legacy/tune_nevals.py` — nevals 掃描 + 雙指標驗證範本。
- ZAP 論文:Soto, Lilly, Bacon, Richard & Conseil (2016), MNRAS 458, 3210。vendored 原始碼:`libs/zap/`(2.1)。

---

## 7. 未能完全查證 / 存疑處

- ZAP 論文 PDF 的細節(如 100 px vs 20–50 px 的原始建議、11 段邊界的天空線族歸屬)取自論文 HTML(ar5iv)與 readthedocs 摘要;演算法**逐步行為以 vendored 2.1 原始碼為準**,論文數字僅作背景。凡兩者不一致(段數、cfwidth 預設、pevals/optimizeType、getzcube),本檔已在 §2.1 註明並以 2.1 為準。
