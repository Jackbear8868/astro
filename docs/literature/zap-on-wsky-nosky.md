# 把 ZAP 跑在 wsky / nosky 上：方法、論文呈現、你的呈現

> 依據：Soto et al. 2016（`docs/paper/predict-residual/Soto2016_ZAP.pdf`）+ 本地 `libs/zap`(v2.2.dev，本環境可 import)。

## ZAP 是什麼、工作流程
ZAP = **後處理的 PCA 殘餘天空移除**（策略二）。流程：
1. **（可選）空間遮罩源**：把白光 >2σ 的源 spaxel 排除，避免它們污染本徵譜。
2. **zlevel（零階天空）**：對每個波長層取**中位數**減掉。對 with-sky 移掉 ~99% 天空；對已扣天空的 cube 則移掉殘餘中位。→ **所以 ZAP 對 wsky 與 nosky 都能跑**。
3. **連續譜濾波**（加權中位）：留窄的天光殘餘、壓掉源連續譜。
4. **分段 SVD**：把波長切成 11 段（依 OH/O₂ 線群，見論文 Table 1），各段做 SVD 得本徵譜。
5. **自動選 neval**：用變異數曲線轉折點決定用幾個本徵譜（太多會吃到源）。
6. **重建殘餘並相減** → 乾淨 cube。

## 論文怎麼呈現結果（Soto 2016 的圖）
| 圖 | 內容 |
|---|---|
| Fig 1 | MUSE pipeline 殘留的天空殘餘長相（動機）|
| Fig 2 | 某段的**本徵譜矩陣** + 前 10 條本徵譜疊圖 |
| Fig 3 | 發射線處本徵譜特寫（如何修正 over/under-subtraction）|
| Fig 4 | **變異數 vs 使用本徵譜數**曲線 → 決定 neval（轉折點）|
| Fig 5 | 用 1/10/20/30/60 個本徵譜對一條星系譜的效果（殘餘下降，但太多會傷源）|
| **Fig 6** | ⭐ **ZAP(紅) vs MUSE pipeline(藍) 光譜疊圖**（HDFS 稀疏場）= 招牌 before/after |
- 核心訊息：**天光殘餘下降、同時保住源的通量與線型**。
- 量化主軸：殘餘/變異數降低 + 源通量不被吃掉。

## 怎麼跑（兩個 run 各代表什麼）
- **Run A：ZAP on `nosky`**（ZAP 的本職：殘餘清除）。
  `nosky − nosky_zap` = ZAP 清掉的**殘餘天光**（應集中在天光線波長、量小）。
- **Run B：ZAP on `wsky`**（讓 ZAP 當完整扣天空器）。
  `wsky − wsky_zap` = ZAP 扣掉的**整個天空**；再比 `wsky_zap` vs `nosky`(pipeline 結果)。
- 關鍵參數：
  - `mask=`：源遮罩 FITS（白光 >2σ）。Haro 11 視野大半是空白（稀疏場），可先不遮、再加遮對照。
  - `skycubefits=`：**把 ZAP 扣掉的天空另存成 cube**（呈現用）。
  - `cfwidthSVD/SP`(預設300)、`nevals`(空=自動)、`zlevel='median'`。

```python
import sys; sys.path.insert(0, "libs/zap")
import zap
# Run A: 在已扣天空的 cube 上清殘餘
zap.process("data/Haro11_nosky.fits", outcubefits="results/zap/nosky_zap.fits",
            skycubefits="results/zap/nosky_skyremoved.fits",
            mask="results/zap/source_mask.fits", overwrite=True)
# Run B: 在含天空的 cube 上直接當扣天空器
zap.process("data/Haro11_wsky.fits", outcubefits="results/zap/wsky_zap.fits",
            skycubefits="results/zap/wsky_skyremoved.fits",
            mask="results/zap/source_mask.fits", overwrite=True)
```

## 你該怎麼呈現（針對 Haro 11，對齊論文又用上你的 ground truth）
1. **空白 spaxel 光譜 before/after 疊圖**（仿 Fig 6）：`nosky`(藍) vs `nosky_zap`(紅)，放大 5577/6300/OH → 殘餘尖峰縮小。
2. **逐波長殘餘 std**（空白區）before vs after → ZAP 曲線在天光線處更低。
3. **單波長影像**（取一條天光線，如 6300Å）before vs after → 殘餘條紋/圖樣消失。
4. **源保真檢查**：亮源 spaxel before/after，**Hα 等發射線不變** → ZAP 沒吃掉源（論文最在意這點）。
5. **ZAP 扣掉的天空 cube**（skycubefits）：畫出來，應是天光線殘餘圖樣。
6. **變異數曲線 + neval**（`varcurvefits=`）：仿 Fig 4。
7. **量化表**：天光線殘餘 RMS（before/after）、殘餘直方圖（越集中 0 越好）。

### ⭐ 你比論文多一張王牌：ground truth
你有 `sky_truth = wsky − nosky`，所以可額外呈現：
- **Run B 驗證**：把 ZAP 扣的天空 `wsky − wsky_zap` 直接和 `sky_truth` 比 → 看 ZAP 當「完整扣天空器」準不準（論文做不到，因為它沒 ground truth）。
- 這正好把「策略二(ZAP)」放到和你要做的「策略一」同一把尺上比較。

## 注意事項
- 本地 zap 在 `libs/zap`，用 `sys.path.insert(0,"libs/zap")` 或 `PYTHONPATH=libs/zap`。
- 跑全 cube 需數分鐘 + 數 GB 記憶體（cube 3679×499×559）。
- AO 檔有鈉缺口(5800–6000=0)；這裡用的 wsky/nosky 是 WFM-NOAO，無缺口，較單純。
- 評估殘餘時分母用**實測雜訊**（STAT 低估 ~1.8×，見 C 組）。
</content>
