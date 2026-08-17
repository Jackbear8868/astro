# pipeline 設計審查（2026-08-18）

對 `run_pointing.sh`、`step1`–`step5`、`utils.py`、`templates.py` 與
`src/skymodel/evaluation/` 的一次完整審查。記錄的是**軟體缺陷**：程式在什麼情況下會
給出錯誤或誤導的結果。**不含任何科學參數的建議** —— 那些列在最後一節，交給教授決定。

每一項的格式是：是什麼 → 證據（`檔案:行`）→ 會造成什麼後果 → 最小修法。
標「已驗證」的是實際跑過或逐項核對過的；其餘來自逐行閱讀。

---

## 甲 會產生錯誤結果而不報錯

### 甲1 step3 與 step5 對「哪裡可以學天空」認定不一致 · 已驗證

`run_pointing.sh:37-53` 的 `--xlim/--ylim/--exclude-box` **只**傳給 step3（`:82`）。
step5（`:92-94`）沒有收到；它的 s 場訓練樣本由 `utils.py:453-460` 決定，只看
「離 seg 多遠」（`--sf-r-far` 15 px、`--sf-r-far-haro` 50 px）。

`step5_fit_spaxels.py:352` 的 help 自己寫著 `--sf-exclude-box`「用途與 step3 的
`--exclude-box` 相同，**而且兩者應該給同一個框**」—— 程式承認兩邊該一致，
`run_pointing.sh` 卻沒有接線，而 `--xlim/--ylim` 連對應的選項都不存在。

落在 step3 禁區裡的 s 場訓練點：

```
  p09  --xlim 0 100      訓練 14,150   禁區內 6,123  (43.3%)
  p11  --xlim 0 ...      訓練 25,436   禁區內 9,682  (38.1%)
  p12                    訓練 24,291   禁區內 7,935  (32.7%)
  p01  --xlim 0 165      訓練 39,286   禁區內 12,412 (31.6%)
  p03  --ylim 170 9999   訓練 27,322   禁區內 6,681  (24.5%)   該處 s 中位高 0.8%
  p14  --exclude-box     訓練 38,290   禁區內 1,630  ( 4.3%)
```

s 場乘上 `C_sky` 之後扣在**每一個** spaxel 上（含源區，`step5:566`）。這條路徑正是
「限制 sky basis 學習範圍」要擋的 over-subtraction —— 從 step3 擋掉之後，從 step5
的側門走回來。終端輸出只印訓練點總數，不印它們在哪，所以看不出徵兆。

**修法（軟體部分）**：給 step5 加 `--sf-xlim/--sf-ylim`，`run_pointing.sh` 把同一組
`REGION` 一起傳給兩個 step。**要不要讓兩者一致是科學決定**（見最後一節）；軟體上
該做的是讓這件事可以被表達、被記錄。

### 甲2 step03 的產物沒有 provenance，重跑會靜默覆蓋

`step3_sky_basis.py:268-291` 寫出 8 個檔，檔名只編進 `method` 與 `K`。
`--xlim/--ylim/--exclude-box/--seg/--cube` 都不進檔名，`step03/` 底下也沒有
`meta.json`（全 repo 只有 step05 寫 meta）。

兩個方向都會出事：

- 改了 REGION 只重跑 step3 → basis 被靜默取代，step05 那份舊結果從此對應不到磁碟上
  的 step03，而它的 `meta.json` 只寫 `sky_dir: results/skymodel/pNN/step03`。
- `step5 --seg` 的 help（`:366-367`）寫「要和 step3 用同一份，否則天空模型的訓練樣本
  和擬合時的區域劃分會對不起來」—— **沒有任何程式碼檢查這件事**。

`run_pointing.sh:66` 的 `cp "$PROF_SEG" "$W/step01/seg.fits"` 同理：教授換一版 seg，
所有下游產物瞬間過期而毫無標記。

**修法**：step3 照 step5 的作法寫 `out_dir/meta.json`（`cube / seg / xlim / ylim /
exclude_box / K / methods / n_blank / git_commit / argv`）；step5 啟動時比對
`step03/meta.json` 的 `seg` 與自己的 `seg_path`，不同就報錯。

### 甲3 `galaxy_redshifts` 用 `glob(...)[0]` 取紅移

`utils.py:309-313` 用 `sorted(Path(step04).glob(f"scan2_id{i}_*.npz"))[0]`，tag 不參與
比對。呼叫端 `step5_fit_spaxels.py:512-515` 的註解卻寫「紅移取自 `--best` 那一份擬合的
同一個目錄 —— 分類和紅移必須來自同一次 step4b」。**程式只保證同一個目錄，不保證同一次
擬合。**

step4 的 `--line-mask-iter` 預設是 `[1,2,3,4]`（`step4_fit_source.py:312`），跑一次預設值
就會留下 L1/L2/L3/L4 四組 `scan2_*`。紅移決定 `main_source_group` 收哪些 seg ID 進主源，
主源足跡決定 `--sf-r-far-haro` 從哪裡起算 —— 最後回到甲1那條路徑。

這條路徑有 6 個呼叫端：`step5:514`、`check_pointing.py:84`、`whitelight_compare.py:57`、
`main_group_map.py:46`、`s_shape_map.py:48`、`box_spectra.py:101`。

今天安全是因為 `run_pointing.sh:87` 傳了 `--line-mask-iter 1`，每顆只有一個 tag。

**修法**：`galaxy_redshifts(step04, ids, tag=None)` 多收 tag，由呼叫端從
`classification_{tag}.npz` 剝出來；沒有 tag 時命中 >1 就報錯，不取 `[0]`。

### 甲4 `run_pointing.sh` 手抄 step4 的 tag · 已驗證

`run_pointing.sh:91`：

```bash
BEST=$W/step04/classification_nobasis_s0.0_4600-8000_4600-8000_L1cum.npz
```

這是 `step4_fit_source.py:81-94` `make_tag()` 的第二份實作，用手抄的。`make_tag` 的
docstring 自己寫著「分成兩份寫的話，改了一邊忘了另一邊會變成讀到錯的檔案」。

把 `:86` 的 `--s-fix 0.0` 改成 `1.0` 再重跑：step4 產生新檔，`BEST` 仍指向舊的 →
step5 不報錯，安靜地用上一次設定的分類與紅移。

**修法**：把 step4 的可變參數提成 shell 變數（`SFIX`、`WIN`、`LITER`），`BEST` 用同一組
變數組出來。

### 甲5 s 場的參數不進 run 名字，`--sf-file` 不進 meta 的結構化欄位

`--sf-r-far / --sf-r-far-haro / --sf-clip / --sf-exclude-box / --main-dv-max / --sf-file`
全都改變結果，全都不在 `run_name()`（`step5:111-131`）裡。`:587` 是
`mkdir(exist_ok=True)`、`:588,600` 是 `overwrite=True` —— 不同設定寫進同一個資料夾並
**靜默覆蓋**，連同 meta.json 這唯一的紀錄一起。

`--sf-file`（`:523-533`）把整張 s 場換成外部檔案，但 meta.json（`:619-625`）仍記著
`s_field_params`，那組參數描述的是被丟掉的那張場；`sf_file` 只出現在 `argv` 陣列末端。

**修法**：寫檔前若 `out/meta.json` 已存在且「會改變結果的欄位」與本次不同，就
`raise SystemExit` 要求 `--run` 明講。meta 加 `sf_file` 欄位。

### 甲6 `--s-field` 與 `--s-fix` 沒有互斥 · 已驗證

`step5:401-403` 只擋 `--s-field` + `--s-free` / `--blank-s-fix`，`--s-fix` 不在裡面。
而 `:566` 是 `s_fix=s_fix if s_hat is None else s_hat[m]` —— 開了 `--s-field` 之後
`s_fix` 完全用不到。

p01 的 meta.json 同時寫著 `"s_field": true` 和 `"s_fix": 1.0`。那個 1.0 從未被使用，
但 meta 把它記成一個生效的設定。跑 `--s-field --s-fix 0.0` 會得到與 `1.0` **逐位元
相同**的 cube，而 meta 宣稱 `s_fix: 0.0`。

**修法**：`:617` 改成 `s_fix=None if args.s_field else s_fix`；`--s-fix` 的 default
改成 `None`，才分得出「沒給」與「給了預設值」。

### 甲7 evaluation 兩處 `[0]`：多個 run 並存時靜默挑一個

| 檔案:行 | 行為 |
|---|---|
| `whitelight_compare.py:51-52` | `runs[0]`，沒有數量檢查 |
| `zone_spectra.py:100-104` | `hit[0]`，預設 pattern 是 `blank_*` |

對照組 `check_pointing.py:98-101` 有檢查，註解寫「多個 run 並存時不能默默挑一個 ——
挑錯了表格看起來完全正常」。同一個 repo、同一個判斷，兩支做了兩支沒做。

`zone_spectra` 特別諷刺：它的預設 pattern `blank_*` 就是為了匹配多個 run 而設計的比較
工具，命中多個時卻只取第一個，而且因為 `hits` 長度是 1，標籤還會寫成 `ours`。字典序
最小往往是舊的（`tpl44 < tpl60`）。

**修法**：兩處加長度檢查；`zone_spectra` 應該把 `hit` 全部加入。

### 甲8 同一個 K，三個 step 三個預設值

```
  step3_sky_basis.py:17    K = 25
  step4_fit_source.py:303  K = 30   （help：須與 step3 相同）
  step5_fit_spaxels.py:330 K = 25   （help：必須和 step3/step4b 相同）
```

`run_pointing.sh` 三處都顯式傳 `-K 30`，pipeline 本身安全。危險在手動重跑：step3 忘了
給 → 寫出 `sky_basis_svd_K25.npy` 與 K30 並存（`:291` 不覆蓋也不清舊檔）→ step5 忘了給
→ 讀到 K25 的 basis，不報錯。run 名字變成 `blank_svdK25_…`，於是 `step05/` 出現兩個
`*_sfield`，接上甲7。

**修法**：三個 `-K` 都改成 `required=True`，比照本專案對 `--work/--cube/--best/--spec-dir`
已經採用的同一帖藥。

---

## 乙 註解與程式不符

> 這一節在本 repo 特別重要：CLAUDE.md 原則 0 要求每一行都能被解釋，而註解就是那個解釋。
> 註解錯了 = 缺陷。

### 乙1 p14 的波長不是從 4600 開始 · 已驗證

`step4_fit_source.py:69-70` 的註解寫「MUSE 的第一個通道在 4599.7 A，所以 4600 就是
從頭開始」。實測：

```
  p01  CRVAL3 = 4599.66   3801 通道
  p13  CRVAL3 = 4599.72   3801 通道
  p14  CRVAL3 = 4749.83   3681 通道   ← 差 150 A
```

對 p14 而言 4600 不是「從頭開始」，而 tag 仍寫 `4600-8000`，14 顆看起來用了同一個視窗。
同一顆內部的 reduced chi2 比較不受影響，所以是誤導而非錯誤結果。

**修法**：註解改成「下限取 4600 是為了讓三個視窗的起點一致；實際起點以各 cube 的第一個
通道為準」。

### 乙2 `step5_fit_spaxels.py:410-412` 是死碼配假註解 · 已驗證

```python
tag = (f"{args.basis}_K{args.K}_s_free" if s_fix is None else ...)
# tag 只用來「讀」step3 的 basis 檔名;…
```

basis 檔名在 `:423` 是直接用 `args.basis`/`args.K` 組的。`tag` 在 `:412` 之後全檔不再
出現。**修法**：刪掉三行。

### 乙3 對 multiprocessing 機制的兩段註解互相矛盾

`step4_fit_source.py:56-57` 與 `:362-363` 寫「worker 是重新 import 這個模組的」，
`:196` 寫「共用資料由 fork 繼承，不經過 pickle」。Linux 上 `Pool` 走 fork，第二句對、
第一句是 spawn 的行為。結論碰巧仍成立，但理由是錯的。

兩處都寫「這四個必須是模組層級的全域」，實際只有三個（`STEP02B, STEP03, STEP04`），
其中真正被 worker 用到的只有 `STEP04`。

### 乙4 step1 把 WCS 丟掉，使 seg 對齊檢查無法做到位 · 已驗證

`run_pointing.sh:67-68` 的註解宣稱「seg 和白光必須是同一個像素格點。對不上的話下游不會
報錯，只會安靜地把遮罩套錯位置，所以在這裡擋掉」。`:72` 實際只比 `s.shape != w.shape`。

而這個檢查再也做不到更好：`step1_whitelight.py:30` 的 `fits.writeto(...)` 沒有帶
header，白光圖被寫成一張沒有 WCS 的裸陣列。教授的 seg 有完整 WCS
（`CRVAL1 = 9.221972`），cube 也有 —— 資訊在源頭都在，是 step1 丟的。

**修法**：step1 寫檔時帶上 `WCS(hdr).celestial.to_header()`；檢查加比
`CRVAL1/CRVAL2/CD1_1/CD2_2`。

### 乙5 `build_templates` 的 docstring 描述已被取代的判準

`step5_fit_spaxels.py:135-137` 的 docstring 說「`A[0] = 0` 代表最佳解落在邊界上…這種源
不放模型」，但 `:150-153` 的行內註解與程式用的是「四個係數全部為 0 才算沒有模型」，而且
明講「不能只看 `A[:, 0]`」。同一個函式裡兩段文字互相否定。**修法**：刪掉 docstring 那句。

### 乙6 三處 help 描述不存在的預設值

- `step2_object_spectra.py:79-81`：`--cube` help 說「預設是含天光的原始 cube」——
  它是 `required=True`。
- `step3_sky_basis.py:154-155`：「預設的工作區與 cube」—— 兩者皆 `required=True`。
- `step5_fit_spaxels.py:359`：`--best` help 說「step4c 定案的分類」——
  `step4c_record_classification.py` 已不存在。

### 乙7 `zone_spectra` 與其他呼叫端對「主源」的定義不同 · 已驗證

```
  zone_spectra.py:77     main_source_group(seg, ...)              ← 沒傳 step04
  check_pointing.py:84   main_source_group(seg, white, W/"step04")
```

少傳 = 跳過紅移篩選（`utils.py:349`），主源足跡變大 → `zones()` 算出的 `main 1-3`、
`main 3-10` 是**不同的一圈 spaxel**。兩張輸出都叫 `main 3-10`，卻不保證在講同一塊天空。
`zone_spectra:89` 自己又去讀了 step04 拿紅移，看起來是遺漏而非決定。

**修法**：`zone_spectra.py:77` 補上 `W / "step04"`。

### 乙8 `step4_fit_source.py:495-501` 的靜默丟棄分支

key set 不相符時整份舊結果被丟掉且不印訊息，而 `:493` 的註解寫「併入既有結果，不覆寫」。
**修法**：`else` 分支印一行警告。

---

## 丙 整潔

- **丙1** `step3_sky_basis.py` 的 `robust_pca`（`:31-87`）、`soft_threshold`（`:26-28`）、
  `zap_k`（`:89-119`）都不被 `learn_sky_basis` 呼叫，`--methods` 的 `choices` 也只允許
  `pca`/`svd`。唯一使用者是 `experiments/choose_K.py`、`hyper_search.py`。建議搬去
  `experiments/`，而不是留在 pipeline 主檔裡讓人以為 step3 會跑 RPCA。
- **丙2** qso 分支永遠到不了 · 已驗證。14 顆的 group 集合是 `{star, galaxy}`，但
  `step5:148` 的 `eigen` dict、`templates.py:58` 的 `load_eigen_qso`、`utils.py:467` 的
  `GROUP_COLOR["qso"]` 都在描述一個分類器產不出來的類別。
- **丙3** `step1:10`、`step2:74`、`step3:153` 定義了不使用的 `ROOT`。
- **丙4** `s_shape_map.py:42-45` 的守衛寫在被守衛的那一行之後 · 已驗證：`runs` 為空時
  得到 IndexError 而不是友善訊息。兩行對調即可。
- **丙5** `box_spectra.py:279-280`、`seg_id_map.py:27`、`step4_fit_source.py:34-36,341-342`
  的預設路徑仍指向已刪的 `ne_pointing/`。
- **丙6** `check_pointing.py:137-139` 的 `CUBES` 與 `:150` 的 fallback 完全等價。
- **丙7** 小型死碼：`step2:90` 的 `else` 到不了；`step4:474` 放進 `_SHARED` 的 `z_exg`
  從未被讀取；`run_pointing.sh:93` 的 `--sky-dir` 傳的就是預設值。
- **丙8** `run_pointing.sh` 用 `>/dev/null` 與 `| grep` 把唯一的執行紀錄丟掉。step3 的
  空間限制統計、step4 的逐源分類表（含 margin 欄，那是分類穩不穩的唯一指標）都不落地。
  配合甲2，這些數字跑完就永久消失。建議 `| tee "$W/stepNN.log"` 再 grep。
- **丙9** `--num-workers 16`（`run_pointing.sh:88`）是機器相關的硬編碼，繞過了
  `step4:421` 依 CPU 數自動決定的預設。

---

## 耦合與順序：沒有問題

依賴圖是 `1 → {2, 3}`、`{2,3} → 4`、`{1,3,4} → 5`。編號是這張圖的合法拓撲序，沒有循環。

唯一一條循環的邊只存在於文件裡：`step2_object_spectra.py:80-81` 建議 `--cube` 用 step05
的 `sky_subtracted.fits`，那會構成 5 → 2 → 4 → 5。目前 `run_pointing.sh:77` 用 ESO 的
nosky cube 打斷了它。

兩個可能的靜默錯配經查**不是問題**：14 顆 cube 的空間尺寸互不相同，所以 `--work` 與
`--cube` 指到不同 pointing 幾乎必定觸發形狀錯誤；每顆的 wsky 與 nosky 波長格點逐項相同，
所以 step4 拿 step03 的波長軸配 step02 的流量是對齊的。

---

## 交給教授決定的科學問題

以下**不是**軟體缺陷，這份文件不對它們提出建議：

1. **s 場的訓練區域是否應等於 step3 的天空學習區域**（甲1）。證據已量化：4.3%–43.3% 的
   訓練點落在 step3 的禁區，p03 該處的 s 中位高 0.8%。
2. `-K = 30`、`--s-fix`、`--sf-r-far 15` / `--sf-r-far-haro 50` / `--sf-clip 8`、
   `DV_MAX = 1468.0`（`utils.py:298`，這個四位有效數字的來源在 repo 裡查不到）、
   `MIN_COVERAGE = 0.9`、`CLIP_SIGMA = 30`、`--star-window/--gal-window 4600 8000`、
   `--line-mask-iter 1`、以及 `run_pointing.sh:37-53` 那 14 組 REGION。
3. step2 是否改用自己扣過天空的 cube（打開 5→2 那條迴圈）。

---

## 這次審查沒有涵蓋的範圍

`src/skymodel/experiments/`、`src/zap/`、`libs/`；擬合的物理正確性；`evaluation/` 的繪圖
細節；效能與記憶體。審查過程沒有重跑任何一個 step，甲1 的數字是用存下來的 `s_free.npy`
重建 `build_s_field` 的訓練遮罩得到的（`n_train` 逐位元符合各顆的 meta.json）。
