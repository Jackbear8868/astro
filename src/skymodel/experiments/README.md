# skymodel 實驗

> 問「**該不該改成另一種做法**」的一次性實驗。隔壁 `evaluation/` 問的是另一種問句
> ——「現行 pipeline 跑出來的結果如何」。一支程式如果比較的是兩個候選方案,它屬於
> 這裡;如果它只是把現行結果畫出來驗收,它屬於 `evaluation/`。

---

## 工作區的指定方式

所有讀 pipeline 產物的程式都吃 `--work`,指到一顆 pointing 的工作區
(`results/skymodel/pNN/`,底下是 `step01`…`step05`):

```bash
conda run -n astro python src/skymodel/experiments/choose_K.py --work results/skymodel/p01
```

需要 cube 的另外吃 `--cube`,省略時由 `pNN` 的編號推出
`data/wsky/DATACUBE_FINAL_N.fits`。

輸出一律寫到 `results/skymodel/evaluation/` 底下,**檔名或目錄名帶 `pNN`**。
14 顆用的是同一組設定,產物名字彼此完全相同,不帶 `pNN` 的話後跑的那顆會無聲蓋掉
前一顆。唯一的例外是 `step2b_aperture.py`:它產的是**源光譜**而不是圖,寫在工作區的
`{work}/step02b/`。目前 pipeline 沒有任何一步會去讀那個目錄 —— `run_pointing` 把
`classify_sources` 的 `spec_dir` 一律指到 `{work}/step02`。

---

## 檔案

### 遮罩與學天光的範圍

| 檔案 | 問什麼 |
|---|---|
| `prof_seg_range.py` | 教授的遮罩之外還剩多少 Haro 11 的光 —— 學天光的範圍該切在哪 |
| `sky_region_visual.py` | 把星系的暈畫到眼睛能判讀,由人自己讀出界線 |
| `sky_region_bound.py` | 用殘留星系光自動定一條界線 |
| `dilate_seg.py` | 認標籤地把源遮罩長胖,產生 `seg_dil{r}.fits` |
| `edge_oversubtraction.py` | 遮罩外面那一圈被扣掉多少(甜甜圈與徑向剖面) |
| `ring_consistency.py` | 逐圈問「這一圈還是不是同一個天體」,定逐源的 `r_stop` |
| `curve_of_growth.py` | 累積 Δchi2 的成長曲線 —— 峰值就是最佳半徑,不需要門檻 |

`configs/pNN.yaml` 的 `sky_region` 是使用者看著 `sky_region_visual.py` 的輸出
(`evaluation/masking/prof_seg/visual_pNN.png`)親自定的,不是推導出來的值。

要用 `dilate_seg.py` 產的遮罩跑 pipeline,把該顆 `configs/pNN.yaml` 的 `input.seg`
指到那份 `seg_dil{r}.fits`:`pipeline.place_segmentation` 會把它複製成
`step01/segmentation_input.fits`,而讀遮罩的四步 —— step2(抽源光譜)、step3(界定 blank)、
step5 與 step6(界定源區)—— 都是從那裡讀的。要和原本的結果並存的話,`output`
也要一起改:產物的檔名不帶遮罩,同一個 `output` 會被後跑的那次蓋掉。這條路徑的
好處是「這次跑用的是哪一份遮罩」留在 config 檔裡,命令列旗標留不住。

### 天空 basis

| 檔案 | 問什麼 |
|---|---|
| `choose_K.py` | K 該取多少 —— ZAP 判準 / 交叉驗證 / 雜訊平台,三個一起看 |
| `basis_contamination.py` | basis 裡有沒有混進 Haro 11 自己的發射線 |
| `plot_sky_basis.py` | K 條成分一次看完(影像 + 前幾條的線形) |
| `plot_linemask_iters.py` | `estimate_continuum` 每一輪的連續譜、門檻與線遮罩 |
| `ccorr_continuum.py` | 自我一致地修正 `C_sky` 的形狀,再端到端驗收 |
| `exp02_basis_solver_matrix.py` | ZAP 邏輯的參考版本(見下) |

### 天空連續譜係數 s 的空間場

| 檔案 | 問什麼 |
|---|---|
| `s_prior_holetest.py` | 把 s 的先驗放進擬合,能不能贏過現行的事後平滑 |
| `s_lowrank.py` | 低秩補全 vs 現行的 row+col 場 |
| `s_common_map.py` | 14 顆的 s 殘差疊成共同圖 —— 並先確認那不是星系的暈 |
| `s_flux_bias.py` | 源流量的加法偏差是誰造成的(競爭 vs 殘差本身) |
| `flux_bias_map.py` | 那個偏差在全場是不是常數 |

### 源與模板

| 檔案 | 問什麼 |
|---|---|
| `step2b_aperture.py` | 用固定圓形孔徑抽源光譜,取代 segmentation footprint |
| `star_library_residual.py` | 源模型有沒有把源解釋完 —— 逐源畫 `wsky − sky_model − source_model` |

`star_library_residual.py` 一列一個 run,右邊是該列自己的統計量,所以不同 run 的
源區殘差可以擺在同一張圖上讀。

---

## exp02 — basis × solver 全因子矩陣

在 **blank region** 上比較天光線模型的所有組合。不碰 source 區,所有 blank spaxel
全數納入,不做訓練／評測切分。

**它跑過的工作區(單 pointing 時代的 `ne_pointing/`)已經刪除**,所以這支不能直接
重跑;要重跑得先另建一個工作區並把 `STEP01` 指過去。下面這一節就是它的結果紀錄。

### 實驗設計:2 × 3 × 4 = 24 組

**連續譜處理(2)** — 決定「量天光線振幅時,用哪一種殘差去擬合」

| | 說明 |
|---|---|
| `shared` | 所有 spaxel 共用一條 mean sky 連續譜 |
| `own` | 每條 spaxel 自己的連續譜(ZAP 的做法) |

**basis 分解方法(3)**

| | 非負限制 | 是否需要 clip |
|---|---|---|
| `NMF` | 兩個矩陣都非負 | 需要(負值 clip 成 0 → 系統性正偏移) |
| `semiNMF` | 只有 basis 非負 | 不需要 |
| `PCA` | 都可正可負(ZAP 用的) | 不需要 |

**振幅解法(4)** — 每條 spaxel 各自解

| | 加權 | 非負 |
|---|---|---|
| `unw+nn` | 無 | NNLS |
| `chi2+nn` | 1/STAT | NNLS |
| `unw+free` | 無 | 無約束 |
| `chi2+free` | 1/STAT | 無約束 |

這個 2x2 是 `exp02_basis_solver_matrix.py` 自帶的實作,不經過 step5,所以仍然跑得動。
但要知道 **pipeline 那邊只剩 `unw` 這一半**:step5 的加權路徑在 2026-08-18 比較後
移除(見 `docs/tried-not-adopted.md` 第二節),而被標成「教授的方式(字面上)」的
`chi2+free` 在 pipeline 裡已無對應選項。

### 記帳原則

最後扣除的一律是「共用天光連續譜 + 重建的天光線」:

    resid = spectra − continuum_shared − A @ W

`own` 連續譜**只作為擬合工具,不永久扣除**。理由:逐 spaxel 連續譜搬到 source 區時,
會把 source 自己的連續譜一起吃掉。因此 `shared`/`own` 這一軸比較的是
**擬合殘差的來源**,不是**最終扣除的對象**。

### 評測指標

- blank region 平均殘差光譜的 mean / rms,並拆成「全波長 / 天光線通道 / 無線通道」三段
- 每條 spaxel 的 **reduced χ²**(理想值 = 1;> 1 擬合不足,< 1 過度擬合即吃掉訊號)
- 對照基準:ESO `nosky` cube 的同一組數字

### 參數

定義在 `exp02_basis_solver_matrix.py` 檔頭:

| 參數 | 值 | 意義 |
|---|---|---|
| `K` | 10 | 天光線基底條數 |
| `WINDOW` | 300 | 連續譜 running median 視窗(spectral pixels) |
| `THRESHOLDS` | (1, 2) | 線偵測門檻(正, 負) |
| `MAX_ITER` | 20 | `estimate_continuum` 迭代上限(只作用在 mean sky 上) |
| `SEMI_ITER` | 300 | semi-NMF 乘法更新次數 |
| `CHUNK` | 8000 | 分塊大小,只控制記憶體與速度 |

### 已知重複

`exp02_basis_solver_matrix.py` 內含 `semi_NMF`、`spectrum_stats`、`plot_compare`、
`per_spaxel_continuum` 的本地副本。這些函式已收納進 `src/skymodel/products.py`;
封存檔保留自己的副本,以確保重跑時得到與 2026-07-24 完全相同的結果。

---

## 這些實驗已經定案的事

| 項目 | 決定 | 依據 |
|---|---|---|
| 線偵測的 σ 估計 | **mean-spectrum σ**:先把所有 blank spaxel 平均成一條 mean sky,再在其上量 σ | 教授指示(2026-07-21) |
| 線偵測門檻 | 雙向 `(1σ, 2σ)` | 教授指定 |
| 天空連續譜 | **shared 形狀**(由大量 blank spaxel 合併內插取得),**振幅逐 spaxel 自由縮放** | 教授指示(2026-07-26) |
| segmentation | 用教授交付的那一份,不自己跑 SExtractor | `configs/pNN.yaml` |
| 分解方法 | 只留 `pca` / `svd`;NMF 與 RPCA 已退役 | `step3_sky_basis.METHODS` |

天空連續譜的形式:

    SkyC(p, λ) = s(p) · C_sky(λ)

形狀共用以抵抗單 spaxel 噪聲;只有振幅 `s(p)` 逐 spaxel 自由,用以吸收 throughput
與照明的空間變化。exp02 中的 `shared` 相當於把 `s(p)` 固定為 1,尚未含這個自由參數。

已經測過並否決的做法記在 `docs/rejected-approaches.md`。
