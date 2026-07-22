## Sky substraction on CGM


### Environment (conda)

```bash
# create the `astro` env (python 3.12 + all deps, incl. editable libs/zap)
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ZAP=2.1 conda env create -f environment.yml
conda activate astro
```

The `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ZAP` variable is required because
`libs/zap` versions itself with setuptools-scm, but its git metadata is a
dangling submodule link and is unavailable here, so the version (2.1) must be
supplied manually.


### Data

Haro11_NFM_ESO_nosky.fits -> NFM-AO-N
Haro11_nosky.fits -> WFM-NOAO-E
Haro11_WFM_MUSE_archive.fits -> WFM-NOAO-E
Haro11_wsky.fits -> WFM-NOAO-E

WFM (Wide Field Mode，廣視野模式)： 視野比較大（通常是 $1 \times 1$ 角分），可以看到比較完整的星系結構。
NFM (Narrow Field Mode，窄視野模式)： 視野非常小（通常是 $7.5 \times 7.5$ 角秒），但解析度極高，用來放大看星系核心的細節。
AO (Adaptive Optics，調適光學)： 有開啟雷射導星系統來修正地球大氣層擾動，拍出來的影像會非常清晰。
NOAO (No Adaptive Optics，無調適光學)： 沒有開啟大氣修正。
E / N (Extended / Nominal，波長範圍)： * N (Nominal) 代表標準波長範圍（大約 480-930 nm）。

下載資料 (Google Drive, 需共用權限)：
```bash
cd data && gdown <file_id>   # 4 個 Haro11 cube, 每個約 7-8 GB
```


### 專案結構（現行）

- `src/skymodel/` — **現行主線**：sky reconstruction pipeline（白光影像 + SExtractor 建遮罩
  → 從 blank spaxel 學天空模型；規格見 `docs/plan/joint-sky-factorization-spec.md`），
  產出在 `results/skymodel/`。
- `src/zap/` — ZAP 扣天空**對照組**（mask → zap → eval_spectrum，見 `src/zap/README.md`），
  產出在 `results/zap/`。
- `src/cgm_halpha.py` — CGM Hα 延展結構分析（見下節）。
- 更早期的腳本（pre-0706 舊版、0706 版、sep 探索工具）已移除，存於 git 歷史。


### ZAP 扣天空對照（`src/zap/`）

ZAP 是「扣天空 + 去殘差」工具，**正確輸入是還含天空的 cube (`wsky`)**。跑法：

```bash
conda run -n astro python src/zap/mask.py --from-cube NEnosky --method seg2sigma
conda run -n astro python src/zap/zap.py --target NEwsky --mask-from NEnosky --mask-method seg2sigma_brq --ncpu 16
```
（完整用法與參數見 `src/zap/README.md` 與 `docs/zap-parameters-reference.md`）

**關鍵結論：**
- **源遮罩是成敗關鍵**。Haro11 的電離氣 (CGM) 延展到視場 30-44%，遮罩太小會讓 ZAP 把
  Hα 當天空學起來、把源吃掉 (~70%)；需以「偵測 + 膨脹邊界」的大遮罩涵蓋整個延展氣。
- **`wsky` + ZAP 有效**：天空線從 ~250-460 扣到 ~0-1, 與 MUSE 的 `nosky` 真值相符, 且保留源
  (見 `results/zap/fig1_wsky_effect.png` 與 `fig3_source_halpha.png`)。
- **`nosky` + ZAP 是 null test**：`nosky` 已被 MUSE 扣過天空 (空白譜已平), ZAP 沒天空可學、只會
  灌雜訊──這不是 ZAP 壞掉, 是餵錯 cube。


### CGM Hα 延展結構分析 (`src/cgm_halpha.py`)

```bash
PYTHONPATH=libs/zap python3 src/cgm_halpha.py
# -> fig6 (Hα 表面亮度圖), fig7 (方位平均徑向剖面 + 偵測極限)
```

比較 MUSE(`nosky`) 與 ZAP(`wsky`) 兩種扣天空法對外圍 CGM 的影響：
- 兩者都還原出 Haro11 的延展 Hα 暈 (核心 + 到 ~20-30" 的halo)。
- **MUSE `nosky` 背景較乾淨** (1σ/spaxel ≈ 14 vs ZAP ≈ 36), 但在大半徑 (>35") 略微**過扣** (中位轉負)。
- **ZAP `wsky` 在大半徑維持正值**, 但其正偏移落在天空曝光的矩形足跡內 (fig6), **較可能是扣天空殘差
  而非真實 CGM**, 解讀需謹慎。
- 對 Haro11 faint CGM 而言, `nosky` 較適合 (更乾淨); ZAP 的價值在於可獨立、不依賴 pipeline 重現扣天空。

