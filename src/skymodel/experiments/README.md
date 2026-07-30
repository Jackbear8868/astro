# skymodel 實驗封存

> 已完成、不再繼續開發的比較實驗。留存目的是保留**對照組數據**：
> 現行 pipeline 選了某個做法，這裡記錄「其他做法差多少」，供論文與後續討論引用。
> 現役開發檔在上一層 `src/skymodel/`。

---

## 檔案

| 檔案 | 原名 | 狀態 |
|---|---|---|
| `exp02_basis_solver_matrix.py` | `test_zap_style.py` | 已跑完（2026-07-24），結果保留 |

---

## exp02 — basis × solver 全因子矩陣

在 **blank region** 上比較天光線模型的所有組合。不碰 source 區，所有 blank spaxel 全數納入，
不做訓練／評測切分。

### 實驗設計：2 × 3 × 4 = 24 組

**連續譜處理（2）** — 決定「量天光線振幅時，用哪一種殘差去擬合」

| | 說明 |
|---|---|
| `shared` | 所有 spaxel 共用一條 mean sky 連續譜 |
| `own` | 每條 spaxel 自己的連續譜（ZAP 的做法） |

**basis 分解方法（3）**

| | 非負限制 | 是否需要 clip |
|---|---|---|
| `NMF` | 兩個矩陣都非負 | 需要（負值 clip 成 0 → 系統性正偏移） |
| `semiNMF` | 只有 basis 非負 | 不需要 |
| `PCA` | 都可正可負（ZAP 用的） | 不需要 |

**振幅解法（4）** — 每條 spaxel 各自解

| | 加權 | 非負 |
|---|---|---|
| `unw+nn` | 無 | NNLS |
| `chi2+nn` | 1/STAT | NNLS |
| `unw+free` | 無 | 無約束 |
| `chi2+free` | 1/STAT | 無約束 |

### 記帳原則

最後扣除的一律是「共用天光連續譜 + 重建的天光線」：

    resid = spectra − continuum_shared − A @ W

`own` 連續譜**只作為擬合工具，不永久扣除**。理由：逐 spaxel 連續譜搬到 source 區時，
會把 source 自己的連續譜一起吃掉。因此 `shared`/`own` 這一軸比較的是
**擬合殘差的來源**，不是**最終扣除的對象**。

### 評測指標

- blank region 平均殘差光譜的 mean / rms，並拆成「全波長 / 天光線通道 / 無線通道」三段
- 每條 spaxel 的 **reduced χ²**（理想值 = 1；> 1 擬合不足，< 1 過度擬合即吃掉訊號）
- 對照基準：ESO `nosky` cube 的同一組數字

### 輸出

`results/skymodel/step01/zap_style/`

| 檔名樣式 | 內容 |
|---|---|
| `cmp_basis_{cont}_{solver}.png` | 固定連續譜與解法，比三種 basis |
| `cmp_solver_{cont}_{basis}.png` | 固定連續譜與 basis，比四種解法 |
| `basis_compare_{cont}.png` | K=10 條 basis 模板的長相對照 |
| `reduced_chi2_hist.png` | 24 組的 reduced χ² 分布 |
| `current_{basis}.png` | 當時 `test.py` 兩條路徑與 nosky 的對照 |

**彙整表只輸出到 stdout，未存檔。** 重跑時需自行導向檔案：

```bash
conda run -n astro python src/skymodel/experiments/exp02_basis_solver_matrix.py \
    2>&1 | tee results/skymodel/step01/zap_style/summary.txt
```

---

## 這些實驗已經定案的事

| 項目 | 決定 | 依據 |
|---|---|---|
| 線偵測的 σ 估計 | **mean-spectrum σ**：先把所有 blank spaxel 平均成一條 mean sky，再在其上量 σ | 教授指示（2026-07-21） |
| 線偵測門檻 | 雙向 `(1σ, 2σ)` | 教授指定 |
| 天空連續譜 | **shared 形狀**（由大量 blank spaxel 合併內插取得），**振幅逐 spaxel 自由縮放** | 教授指示（2026-07-26） |

天空連續譜的形式：

    SkyC(p, λ) = s(p) · C_sky(λ)

形狀共用以抵抗單 spaxel 噪聲；只有振幅 `s(p)` 逐 spaxel 自由，用以吸收 throughput
與照明的空間變化。exp02 中的 `shared` 相當於把 `s(p)` 固定為 1，尚未含這個自由參數。

---

## 參數

定義在 `exp02_basis_solver_matrix.py` 檔頭：

| 參數 | 值 | 意義 |
|---|---|---|
| `K` | 10 | 天光線基底條數 |
| `WINDOW` | 300 | 連續譜 running median 視窗（spectral pixels） |
| `THRESHOLDS` | (1, 2) | 線偵測門檻（正, 負） |
| `MAX_ITER` | 20 | `estimate_continuum` 迭代上限（只作用在 mean sky 上） |
| `SEMI_ITER` | 300 | semi-NMF 乘法更新次數 |
| `CHUNK` | 8000 | 分塊大小，只控制記憶體與速度 |

---

## 已知重複

`exp02_basis_solver_matrix.py` 內含 `semi_NMF`、`spectrum_stats`、`plot_compare`、
`per_spaxel_continuum` 的本地副本。這些函式已收納進 `src/skymodel/utils.py`；
封存檔保留自己的副本，以確保重跑時得到與 2026-07-24 完全相同的結果。
