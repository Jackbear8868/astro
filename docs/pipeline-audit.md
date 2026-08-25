# pipeline 待決事項（審查於 2026-08-18）

對 `run_pipeline.py`、`step1`–`step6`、`utils.py`、`templates.py` 與
`src/skymodel/evaluation/` 的一次完整審查找出 25 項問題。**24 項已修並驗證**，
每一項的理由寫在程式的註解裡。

已修的條目不留在這裡 —— 完整原文（證據、行號、後果、修法）在

```
git show 665d7a9:docs/pipeline-audit.md
```

這份文件只留還沒收掉的：一項軟體做完、科學待決的，加上一組本來就不屬於軟體審查的
參數問題。

---

## 一 軟體做完、科學待決

### step3 與 step5 對「哪裡可以學天空」認定不一致 · 已驗證

`configs/pNN.yaml` 的 `sky_region` 預設 `apply_to: [basis]`,**只**傳給 step3。step5 沒有收到；
它的 s 場訓練樣本由 `utils.build_s_field` 決定，只看「離 seg 多遠」（`--sf-r-far`
15 px、`--sf-r-far-haro` 50 px）。

落在 step3 禁區裡的 s 場訓練點：

```
  p09  --xlim 0 100      訓練 14,150   禁區內 6,123  (43.3%)
  p11  --xlim 0 ...      訓練 25,436   禁區內 9,682  (38.1%)
  p12                    訓練 24,291   禁區內 7,935  (32.7%)
  p01  --xlim 0 165      訓練 39,286   禁區內 12,412 (31.6%)
  p03  --ylim 170 9999   訓練 27,322   禁區內 6,681  (24.5%)   該處 s 中位高 0.8%
  p14  --exclude-box     訓練 38,290   禁區內 1,630  ( 4.3%)
```

s 場乘上 `C_sky` 之後扣在**每一個** spaxel 上（含源區）。這條路徑正是「限制 sky basis
學習範圍」要擋的 over-subtraction —— 從 step3 擋掉之後，從 step5 的側門走回來。

**軟體上已經做完的**：step5 有 `--sf-xlim/--sf-ylim`，含法（含 LO 不含 HI）與 step3 的
`--xlim/--ylim` 逐字相同，並和 `--sf-exclude-box` 合成同一張 exclude 遮罩交給
`build_s_field`；兩個範圍都進 `meta.json` 的 `s_field_params`。驗證：p01 給
`--sf-xlim 0 165` 之後訓練點 39,286 → 26,874，正好等於上表的 39,286 − 12,412。

**沒有做的**：`apply_to` 刻意不含 `s_field`,沒把範圍轉傳給 step5。那一步會改變科學結果。
現在「兩邊不一致」仍是預設值，差別在於它現在可以被寫出來、被 `meta.json` 記下來。

要試就是在 step5 的指令加上和 `REGION` 同一組數字，並用 `--run` 給它自己的資料夾：

```
  ... step5_fit_s_field.py ... --sf-xlim 0 165 --out results/skymodel/p01/step05_sfregion
```

---

## 二 交給教授決定的科學問題

這些**不是**軟體缺陷，這份文件不對它們提出建議。

1. **s 場的訓練區域是否應等於 step3 的天空學習區域**（上一節）。證據已量化。
2. 參數本身：`-K = 30`、`--s-fix`、`--sf-r-far 15` / `--sf-r-far-haro 50` /
   `--sf-clip 8`、`MIN_COVERAGE = 0.9`、`CLIP_SIGMA = 30`、
   `--star-window/--gal-window 4600 8000`、`--line-mask-iter 1`、以及
   `configs/pNN.yaml` 那 14 組 `sky_region`。
3. `DV_MAX = 1468.0`（`utils.py`）**單獨列出來，因為它的性質和上面那些不同**。
   查證過（`git log -S "DV_MAX" --all`）：它在 `b147631`（2026-08-16，主源分組改用
   紅移判準）第一次出現，在那之前 repo 裡沒有 `1468` 也沒有 `DV_MAX`，**不是從別處
   搬來的，也不是教授交付的值**。同一個 commit 的訊息自己寫著「門檻從 300 到
   100,000 km/s 結果相同」—— 寫下它的人當時就知道這個精確數值不影響結果。
   沒有任何程式、註解或文件記下這四位有效數字怎麼來的。

   所以要問的不是「1468 的物理意義是什麼」，而是「我們自己產生的門檻要不要換成一個
   誠實表達『這只是安全上界』的整數」。
4. step2 是否改用自己扣過天空的 cube。目前用 ESO 的 nosky；換成我們的
   `step05/sky_subtracted.fits` 會構成 5 → 2 → 4 → 5 的迴圈。

---

## 三 這次審查沒有涵蓋的範圍

`src/skymodel/experiments/`、`src/zap/`、`libs/`；擬合的物理正確性；`evaluation/` 的
繪圖細節；效能與記憶體。審查過程沒有重跑任何一個 step，第一節的數字是用存下來的
`s_free.npy` 重建 `build_s_field` 的訓練遮罩得到的（`n_train` 逐位元符合各顆的
`meta.json`）。
