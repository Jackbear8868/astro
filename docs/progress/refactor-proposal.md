# 程式碼與輸出檔案重整提案（src/ + results/）

本文是一份**提案**：只建議、不執行；不動任何 code / data / cube。
目標回應三個需求：(1) 輸出檔案更整齊、可擴充、能區分「小而可重現」與「大而可再生」；
(2) code 更彈性，偵測參數改為**每個 cube 一組**（並預留 STAT 變異圖當噪音輸入）；
(3) 評估「複製一份舊資料夾再改」這個做法。

所有結論都以現況實測為依據（header、git 狀態、目錄實況），下面先講事實，再給設計。

---

## 一、現況評估

### 1.1 該保留的（做得對，別動）

- **process / evaluate 分離、單一 `settings.py`**：`step1_mask → step2_zap → eval_*`，每支程式 docstring 都寫清楚「讀什麼 / 寫什麼」，參數只定義一次。這是好骨架，重整要**沿用這個精神**。
- **產物「自帶出處」的命名意圖**：`masks/<method>_from-<cube>/`、`cubes/<target>_maskfrom-<from>/` 讓每個實驗自成一夾、彼此不覆蓋 —— 想法對，只是編碼要修（見 1.2）。
- **`blankstats/` 快取模式**：把幾 GB 大 cube 縮成幾十 KB 的逐波長曲線、多張圖共用 —— 保留。
- **`.gitignore` 的粗分層正確**：`data/`、`results/zap/`、`*.fits`、`*.npz`、`*.pdf` 都排除，等於「只追蹤小的 code+docs、不追蹤大 cube」。這是對的基線，重整只需把「小產物中的 provenance」補回追蹤。
- **物理參數集中且有推導依據**（CLAUDE.md 操作清單）—— 保留，但要從「全域常數」升級成「每 cube 一組」。

### 1.2 該修的（點名具體檔案 / 目錄）

**痛點排序（前三名是最該先處理的）：**

1. **現行 pipeline `src/0706/` 完全沒被 git 追蹤。** `git ls-files src` 追蹤的是 `src/main.py`、`src/sextract.py`… 這些**舊檔（現在顯示為 deleted）**；真正在跑的 `src/0706/*.py`、`scripts/`、多份 `docs/*.md` 全是 untracked。等於**正式程式沒版本控管**，弄丟就沒了。這是第一優先。

2. **日期命名的目錄 `src/0706/`。** 用「寫它的那天」命名資料夾不具自我說明性，而且它本身就是「複製一份再改」的化石 —— 下次再改就變 `src/0709`、`src/0714`，正是需求 3 要避免的分岔。

3. **三支壞掉的 eval 腳本躺在活腳本旁邊。** `eval_figures.py`、`eval_scores.py`、`eval_summarize.py` 仍引用 `settings.CACHE_NPZ_PATH`、`settings.SOURCE_MASK_PATH`、`settings.summary_npz_path`、`results/zap/npz/` —— 這些符號**現在的 `settings.py` 根本沒有**（實測 grep 不到），一跑就 crash。它們和能跑的 `eval_common.py` / `eval_m1_muse.py` / `eval_m1_zap.py` / `eval_mask_compare.py` 混在同一夾，難分死活。

4. **偵測參數是全域常數，無法表達 per-cube。** `MASK_BACKGROUND_BOX_PIXELS=256`、`MASK_KERNEL_FWHM_PIXELS=6.0`… 是模組層級常數。實測 header：主場 `NAXIS=559×499`、NE 指向 `NAXIS=332×320`。`bw=256` 對主場（暈 Ø≈256px）合理，但對 320px 的 NE 幾乎等於全域 —— 不同 cube 本來就該用不同值，全域常數表達不了。

5. **run / mask 命名把多個軸擠進一條字串，還藏預設值。** `_run_suffix` 讓 `sep` 方法**隱藏前綴**：`nosky_maskfrom-nosky`（sep 被藏）對上 `nosky_maskfrom-claude-nosky`、`nosky_maskfrom-box_sep-nosky`。而 `box_` 又借用 method 欄位表達「ZAP 在盒內學」。於是「ZAP 對象 × mask 方法 × mask 來源 × box 變體」四個正交軸被壓成一條有隱藏預設的連字號字串，難解析、易撞名。實測還留下一個**空的殘夾** `cubes/wsky_maskfrom-box_claude-wsky/`（跑一半的失敗 run）。

6. **`results/zap/` 根目錄是傾倒場。** 根目錄散落約 18 張 PNG（`fig1-4`、`fig6-7`、`fig_depth_grid`、`fig_exposure_depth`、`fig_offset_spatial`、`fig_whitelight_*`、`cmp_nosky_*`）和結構化子夾（`masks/`、`cubes/`、`blankstats/`、`compare/`、`figs_archive/`）平放；檔名不帶 cube 出處；`figs_archive/` 又和根目錄的 `fig1-3/6-7` 重複。

7. **`data/` 有垃圾檔。** `ls data` 有 5 個 0-byte 檔 `=1.4.0`、`=2.4.6`、`=3.11.0`、`=5.0.2`、`=8.0.0` —— 是某次 `pip install >=x` 被 shell 導向誤建的空檔，應刪。另有 `src/astro.egg-info/`、多個 `__pycache__/` 混在追蹤範圍邊緣。

8. **`docs/` 也散。** 頂層一堆散落 `.md`（`metric_spec.md`、`segmentation-parameters-explained.md`、`sep-sextractor-parameters.md`、`zap-parameters-reference.md`、`qfitsview-remote-access.md`、`zap_2x2_run_plan.txt`）和結構化子夾（`plan/`、`progress/`、`literature/`、`paper/`、`learning-objects/`）並存。此提案聚焦 code/results，`docs/` 僅建議把「參數參考類」歸一個 `docs/reference/`。

### 1.3 一個必須向你示警的物理事實（CLAUDE.md 原則 2）

實測兩顆 cube 的 PRIMARY header：**`ESO QC EXPCOMB FWHM MEDIAN = 0.0`** —— CLAUDE.md 指定用來取 seeing 的那個關鍵字，在這批檔案裡是空的（0.0）。可用的 seeing 代理是 AG / ambient FWHM ≈ 0.89–1.0″。也就是說，現行「kernel 6px ≈ 1.24″ seeing 從 header 來」實際上是**假定的常數，不是量到的**。而且 NE 的 PRIMARY header 這些 QC 欄位和主場**逐位元相同**（同一 OB 繼承來的），所以真正能可靠 per-cube 區分的 header 量只有 `NAXIS1/NAXIS2`（場大小）。

**結論給設計**：config 要做「能從 header 推的就推（場大小 → bw），推不到的（seeing）就用**明確、有註記出處的 per-cube 常數**，並在關鍵字不可用（=0.0）時**主動印警告**」。這正好強化了「per-cube config 物件 + 記錄 provenance」的設計，而不是盲目 header 自動推導。

另一個好消息：兩顆 cube 都有 `STAT` 延伸（variance），所以「用 STAT 當 per-pixel 噪音圖」的後續功能**資料上可行**。

---

## 二、建議的目標結構

### 2.1 `src/` — 用「功能」命名的套件，取代日期夾

```
src/
  skysub/                    # 正式套件（取代 src/0706/；名字講功能，不講日期）
    __init__.py
    config.py                # CubeConfig dataclass + CUBES registry + from_header()  ← 本提案核心
    io.py                    # 開 cube、wavelength_axis、載入 STAT/variance 的共用 helper
    imaging.py               # halpha_narrowband_image、白光、gauss matched-filter kernel
    mask.py                  # step1：build_mask(cfg) → 寫 mask.fits + meta.json（原 step1_mask）
    zap_run.py               # step2：跑 ZAP（原 step2_zap）
    eval/
      common.py              # blank_residual + valid + 快取（原 eval_common）
      m1_muse.py             # 原 eval_m1_muse
      m1_zap.py              # 原 eval_m1_zap
      mask_compare.py        # 原 eval_mask_compare
    cli.py                   # 統一入口：python -m skysub.cli mask NE-nosky / zap NE-nosky NE-nosky / eval-m1 ...
  archive/                   # 把 legacy/ + explore/ 收進來，明確標示「不維護、僅參考」
    legacy/                  # 原 src/legacy
    explore/                 # 原 src/explore
```

一行理由：
- `skysub/` 是**可 import 的套件**，用功能命名 → 不會每次改就 fork 出新日期夾；`cli.py` 保留 `python -m skysub.cli mask NE-nosky` 的肌肉記憶。
- `config.py` 獨立成檔，容納 per-cube 設定（見第三節）。
- `io.py` / `imaging.py` 抽出純計算 helper（現在散在 `settings.py` 尾與各 step），讓 STAT 噪音載入等新輸入有明確落點。
- **刪掉三支壞的 eval**（`eval_figures/scores/summarize`），或把還要的 fig1-4 移進 `eval/`；不要讓死碼混在活碼旁。
- `archive/` 把 `legacy/` + `explore/` 收攏，一眼分辨「正式 vs 歷史」。

### 2.2 `results/` — 依「小/大、可重現/可再生」分層

```
results/
  masks/                     # 【小・可重現・該保存】每 cube 每方法一夾
    <cube>/<mask_id>/        # 例：NE-nosky/sep/、main-nosky/claude/、main-nosky/sep-box/
      mask.fits              # 2D uint8（幾百 KB）
      blanks.npz            # 亮源座標 + 波長軸（供 M3）
      meta.json             # provenance：cube, method, 完整 DetectParams, git sha, 時間  ← 新增
  runs/                      # 【大・可再生】每個 ZAP 實驗一夾
    <run_id>/               # run_id = zap-<target>__mask-<cube>-<mask_id>（雙底線分軸，無隱藏預設）
      config.json           # 這次的完整 CubeConfig + mask_id + ncpu + git sha（小・該保存）
      zap.fits              # 巨大・gitignore・本機 scratch・鏡像到 Drive
      sky.fits              # 巨大・同上
      var.fits              # 小・診斷曲線・可保存
      figs/                 # 本 run 的 eval 圖（fig_M1_*）
  diagnostics/               # 跨 run / 探索性圖，用 cube 分名空間（取代根目錄那堆散圖）
    <cube>/fig_whitelight_*.png, fig_depth_*.png, ...
  cache/                     # 【可再生・gitignore】blankstats 搬來這
    <cube>/valid.npy, <tag>.npz
  MANIFEST.md                # 舊夾名 ↔ 新 run_id 對照表（遷移時建，保證 Drive 上的 cube 不孤兒）
```

**分層原則（直接回應需求 1）：**

| 類別 | 檔案 | 大小 | 該怎麼待它 |
|---|---|---|---|
| **該保存（tiny, 可重現）** | `masks/*/mask.fits`+`meta.json`+`blanks.npz`、`runs/*/config.json`、`runs/*/figs/`、`runs/*/var.fits`、`MANIFEST.md` | KB–MB | git 追蹤 provenance（`.json`/`.md`）；mask/fig 可上 Drive 備份 |
| **scratch（huge, 可再生）** | `runs/*/zap.fits`、`runs/*/sky.fits`、`cache/*` | GB | `.gitignore`；本機暫存；`zap/sky.fits` 鏡像到 `gdrive:astro/…` |

`run_id` 用 `zap-<target>__mask-<cube>-<mask_id>`：**每個軸都寫出來、不藏預設**（不再有「sep 被隱藏」），雙底線分隔主軸、單連字號分子項，glob / 解析都直覺，也不會撞名。`meta.json` / `config.json` 是關鍵新增：讓每個產物**自我說明**（記下當時用的 per-cube 參數與 git sha），一舉解掉「輸出檔看起來很亂」與「參數改了卻不知哪個 run 用哪組」。

---

## 三、建議的 config 設計（需求 2 的核心）

**一句話推薦：用一個 `CubeConfig` frozen dataclass、每顆 cube 一個實例、集中在 `config.py` 的 `CUBES: dict[str, CubeConfig]` registry；並提供 `from_header()` 類方法，能從 header 推的就推、推不到的用明確 override。**

**為什麼是它（而不是 YAML/TOML，也不是 per-folder 複製）：**
- 它是 **Python 原生**：不引進新 parser 相依，有型別檢查、單一 import，且能**從 header 計算**參數（場大小 → bw）—— 靜態 YAML 做不到「計算」，只能把數字手抄一遍，抄了就會 drift（正是要避免的）。
- provenance 留在 code 裡被 review，又能**原封序列化**成每個產物的 `meta.json`/`config.json`。
- YAML/TOML 會把「真相來源」從消費它的 code 拆走、且無法 derive-from-header；全域常數無法 per-cube；per-folder 複製會 drift（見第五節）。三者皆劣。

### 3.1 `config.py` 草圖（示意，非最終碼）

```python
from dataclasses import dataclass, field
from pathlib import Path
from astropy.io import fits
import warnings

PIXEL_SCALE_ARCSEC = 0.20   # header CD1_1，本批 MUSE WFM cube 皆同

@dataclass(frozen=True)
class DetectParams:
    """偵測旋鈕。預設值是『規則』，實際值每 cube 由資料尺度推得（見 CLAUDE.md 操作清單）。"""
    bkg_box_px:      int            # > 最大物體；場太小就 (近) 全域
    kernel_fwhm_px:  float          # ≈ seeing FWHM
    threshold_sigma: float = 2.0    # ≥2σ，永不下探（CLAUDE.md 原則 2）
    min_area_px:     int   = 30     # ≈1 PSF 面積
    dilate_px:       int   = 6      # ≈1×seeing 安全邊界
    use_stat_noise:  bool  = False  # 新增：用 cube STAT 延伸當 per-pixel err map

@dataclass(frozen=True)
class CubeConfig:
    name:      str
    path:      Path
    ny: int; nx: int                # 場大小，來自 header NAXIS2 / NAXIS1
    seeing_fwhm_arcsec: float       # 有註記出處的常數（header QC 關鍵字這裡是 0.0，故明確給）
    reference: tuple[str, str]      # (standard=已扣天空, sky=含天空) 供 M1
    detect:    DetectParams

    @property
    def seeing_fwhm_px(self) -> float:
        return self.seeing_fwhm_arcsec / PIXEL_SCALE_ARCSEC

    @classmethod
    def from_header(cls, name, path, reference, *, seeing_fwhm_arcsec,
                    halo_diam_px=None, bkg_box_px=None, use_stat_noise=False):
        h = fits.getheader(path, "DATA")
        ny, nx = h["NAXIS2"], h["NAXIS1"]
        # 原則 2 示警：nominal seeing 關鍵字不可用時，明講「這是假定值不是量到的」
        qc = fits.getheader(path, 0).get("ESO QC EXPCOMB FWHM MEDIAN", 0.0)
        if not qc:
            warnings.warn(f"[{name}] header FWHM MEDIAN=0 不可用；"
                          f"seeing={seeing_fwhm_arcsec}\" 為假定常數，非量測值。")
        # bw 規則：要比物體大；場小就夾到全域，永不超過場尺寸
        if bkg_box_px is None:
            want = int(round(halo_diam_px or 256))
            bkg_box_px = min(want, min(ny, nx))       # 主場→256；NE 320×320→夾成 320≈全域
        det = DetectParams(bkg_box_px=bkg_box_px,
                           kernel_fwhm_px=round(seeing_fwhm_arcsec / PIXEL_SCALE_ARCSEC, 1),
                           use_stat_noise=use_stat_noise)
        return cls(name, Path(path), ny, nx, seeing_fwhm_arcsec, reference, det)

CUBES = {
  "main-nosky": CubeConfig.from_header(
      "main-nosky", "data/Haro11_nosky.fits", ("main-nosky", "main-wsky"),
      seeing_fwhm_arcsec=1.24, halo_diam_px=256),          # 559×499 → bw=256
  "NE-nosky": CubeConfig.from_header(
      "NE-nosky", "data/Haro11_NEpointing_esonosky.fits", ("NE-nosky", "NE-wsky"),
      seeing_fwhm_arcsec=1.24, halo_diam_px=200),          # 332×320 → bw 夾到 320（≈全域）
  # main-wsky / NE-wsky 同理...
}
```

要點：**bw 不再是全域 256**，而是「暈直徑 vs 場大小取小」的規則，主場算出 256、NE 自動夾成 ≈全域 —— 正好對上你發現的「NE 只有 320×332，256 幾乎全域」的需求，且**沒有 magic number**。`threshold_sigma`、`min_area_px`、`dilate_px` 的預設仍鎖在 CLAUDE.md 的物理規則上，不隨手動。

### 3.2 step1 / step2 怎麼消費它（含 STAT 噪音圖）

```python
# skysub/mask.py — step1
def build_mask(cfg: CubeConfig):
    with fits.open(cfg.path) as hd:
        cube = hd["DATA"].data
        wl   = wavelength_axis(hd["DATA"].header)
        ha   = halpha_narrowband_image(cube, wl)
        white = np.nansum(cube, 0); valid = white != 0
        if cfg.detect.use_stat_noise:                        # 新輸入：STAT → per-pixel err
            var = hd["STAT"].data
            err2d = np.sqrt(np.nanmean(var[line_window(wl)], 0))   # Hα 帶內的噪音
        else:
            err2d = None
    return detect_sep(ha, valid, cfg.detect, err2d)

def detect_sep(ha, valid, p: DetectParams, err2d=None):
    bkg = sep.Background(ha, mask=~valid, bw=p.bkg_box_px, bh=p.bkg_box_px, fw=3, fh=3)
    err = err2d if err2d is not None else bkg.rms()          # STAT 圖 or 無源估的 RMS
    _, seg = sep.extract(ha - bkg, p.threshold_sigma, err=err, mask=~valid,
                         minarea=p.min_area_px,
                         filter_kernel=gauss_kernel(15, p.kernel_fwhm_px),
                         segmentation_map=True)
    return ndi.binary_dilation((seg > 0) & valid, iterations=p.dilate_px) & valid
```

```python
# skysub/zap_run.py — step2
def run_zap(target: CubeConfig, mask_cube: str, mask_id: str, ncpu=16):
    mask = MASKS / mask_cube / mask_id / "mask.fits"
    run  = RUNS / f"zap-{target.name}__mask-{mask_cube}-{mask_id}"
    zap.process(str(target.path), mask=str(mask),
                outcubefits=str(run/"zap.fits"), skycubefits=str(run/"sky.fits"),
                varcurvefits=str(run/"var.fits"), ncpu=ncpu, overwrite=True)
    write_json(run/"config.json", {**asdict(target.detect),
               "target": target.name, "mask": f"{mask_cube}/{mask_id}", "git": git_sha()})
```

step1 只吃一個 `CubeConfig`；step2 吃「target cube 的 config + 一個 mask_id」。`use_stat_noise` 一個旗標就切換「無源估 RMS」與「STAT per-pixel err」，兩條路共用同一支 `detect_sep`，不分岔。

---

## 四、安全、可逆、增量的遷移計畫

**先立護欄（最重要）：** 巨大的 cube 都已 `.gitignore`，且據 `scripts/check_upload.sh`，`zap.fits` 已鏡像到 `gdrive:astro/cubes/<舊run名>/zap.fits`。**鐵律：凡是已上傳 Drive 的 cube 夾，不可直接 rename / `git mv`，否則 Drive 上那份會孤兒。** 對已上傳的 run，先用 symlink + `MANIFEST.md` 記錄「舊名↔新名」，Drive 端要嘛同步改名、要嘛就維持舊路徑並在 MANIFEST 記下對照。

**步驟（每步可獨立回退，無一步刪掉已算好的 cube）：**

0. **先把現行 pipeline commit 進 git。** `src/0706/`、`scripts/`、`docs/*.md` 現在都 untracked —— 先把這個「能跑的現況」存成基線 commit，再動任何東西。（回退＝丟掉一個 commit。）
1. **刪明確垃圾**（無科學影響、可逆）：`data/=1.4.0 … =8.0.0`（0-byte 導向殘檔）、各 `__pycache__/`、`src/astro.egg-info/`（都可再生）。
2. **新增 `src/skysub/`，以 `src/0706/` 為藍本重構**成套件 + `CubeConfig`，**先不刪 0706**。用一顆便宜、決定性的 mask 驗證：`skysub` 重建一張 mask，和既有 `mask.fits` 逐位元 diff 相同，才算對齊。（skysub 是新東西，0706 沒動，完全可逆。）
3. **刪三支壞 eval**（`eval_figures/scores/summarize`，本來就跑不動），或把還要的 fig1-4 移進 `eval/`。（git 可還原。）
4. **results/ 加法式重整**：建 `results/masks`、`runs`、`diagnostics`、`cache`。gitignore 的檔可直接 `mv`；**已上傳 Drive 的 cube 夾先留原地、用 symlink 指到新結構**，等 Drive 對照確認後再實體移動，並把「舊夾名 → 新 run_id」寫進 `results/MANIFEST.md`。順手刪掉空的殘夾 `cubes/wsky_maskfrom-box_claude-wsky/`。
5. **回填 provenance**：寫一支小工具，從既有 mask/run 的夾名反推 `meta.json` / `config.json`，讓舊產物也符合新規範、變得自我說明。
6. **圖重生到 `results/diagnostics/<cube>/`**，刪掉根目錄那堆散 PNG 與 `figs_archive/`（可再生；若論文有引用，改移到 `docs/paper` 資產夾）。
7. **更新 `.gitignore`**：續 ignore `*.fits` / `*.npz` / 大產物，但**明確解除 ignore 小 provenance**：`!results/**/meta.json`、`!results/**/config.json`、`!results/MANIFEST.md` —— 讓 provenance 進版控，cube 仍不進。
8. **把 `src/legacy` + `src/explore` 併入 `src/archive/`**（純改名，可逆）。
9. **skysub 驗證通過後，退役 `src/0706/`**（`git rm`；仍在歷史裡）。

每步都能單獨 revert；沒有任何一步刪掉已算好的 cube（只做 rename / symlink 加 MANIFEST）；已鏡像 Drive 的巨大 cube 因為有名稱對照表，不會孤兒。

---

## 五、對「複製一份舊資料夾再改」的裁決

**裁決：NE cube 不要複製資料夾。**

「複製再改」只有兩種情況合理：(a) 幾天內會刪掉的**一次性沙盒**；(b) 兩份未來要在**邏輯**上永久分岔（不只是參數不同）。NE 的差異**只在參數**（場大小、bw），這正是 per-cube `CubeConfig` 要表達的東西 —— fork 一份會複製約 10 個檔，之後每次修 bug 或改 eval 都得改兩遍（drift）。而日期夾 `src/0706` 本身就是上一次「複製再改」的化石，再做一次就得到 `src/0709`、`src/0714`…

**正解：用 cube 參數化。** 加一筆 `CUBES["NE-nosky"]`，然後 `python -m skysub.cli mask NE-nosky` / `... zap NE-nosky NE-nosky`。一條 code path，多顆 cube。

**若真的想要一塊「不動到主線」的實驗空間：** 開 **git branch**（例如 `exp/NE-tuning`），不是複製資料夾 —— 同一批檔、隔離歷史、能輕鬆併回或丟棄，且不會在硬碟上默默 drift。如果非要一個實體 scratch，**只複製你正在調的那一樣東西**（例如一個 `notebooks/NE_tuning.ipynb`，裡面 `import skysub`），絕不複製整條 pipeline —— 共用邏輯永遠單一來源。
