# sep / SExtractor 參數完整參考(API 對照與逐檔調參配方)

> 這份文件是本專案 source detection / segmentation 所用工具 **sep** 的**完整 API 與參數參考**,
> 並對照原始 **Source Extractor(SExtractor,Bertin & Arnouts 1996)** 說明每個參數的意義、
> SExtractor 設定檔關鍵字、預設值,以及**如何從任一 cube 的資料/header 推出每個參數**(逐檔配方)。
>
> **分工**:
> - 這份文件 = 完整 API 參考 + sep vs SExtractor 差異 + 通用逐檔調參配方。
> - Haro11 這批資料「為什麼這樣設」的物理推理,見
>   [`docs/segmentation-parameters-explained.md`](./segmentation-parameters-explained.md)。
> - 專案核心原則(**每個超參數都必須物理上站得住腳**;不可辯護就要主動警告)見
>   [`CLAUDE.md`](../CLAUDE.md) 的 Principle 2 與「Operational Checklist」表。本文與該表一致,不牴觸。
>
> 實際使用範例:`src/0706/step1_mask.py`(建遮罩正式流程)、`src/explore/sextract.py`(最小示範)。
>
> 本文所有函式簽章、預設值均**核對自本機安裝的 sep 版本**(見下節);凡是版本相依處都會標明版本。

---

## 0. 安裝版本與套件身分(先確認你用的是哪一個 sep)

本機安裝(以 `conda run -n astro` 環境為準,核對自套件本身):

| 項目 | 值 | 來源 |
|---|---|---|
| `sep.__version__` | **1.4.1** | `import sep; sep.__version__` |
| `sep.__file__` | `…/site-packages/sep.cpython-312-x86_64-linux-gnu.so` | C extension(Cython 包裝) |
| PyPI 套件名 | **`sep`**(非 `sep-pjw`) | `pip show sep`;`pip show sep-pjw` 無輸出 |
| Home-page | `https://github.com/sep-developers/sep` | 套件 metadata |

**套件家譜(sep vs sep-pjw)** — 需要辨明,因為網路上有兩個名字:

- `sep`(Kyle Barbary,`kbarbary/sep`)發行到 **`sep<=1.2.1`**,之後一度停止維護。
- **`sep-pjw`**(Peter Watson 維護)是為修 bug 而生的 fork,發行 **`1.3.0`–`1.3.8`**。
- 兩者現已合流:未來開發集中在 **`sep-developers/sep`**,即 **`sep>=1.4.0`**;`sep-pjw` 已於 2025 標記
  deprecated 並改以原本的 `sep` 名稱發行。**現在應安裝 `sep>=1.4.0`(本機即 1.4.1),不要再裝 `sep-pjw`。**
- 相容性提醒:`sep==1.2.1` 與 `sep>=1.4.0` 之間,**直接使用 C-API** 時可能有不相容(為修「大陣列索引 bug」),
  但 Python 層 API(本文所用)相容。

> 結論:本專案用的是官方合流後的 **`sep` 1.4.1**,以下所有簽章、預設值都以此版本為準。

---

## 1. sep 是什麼、與 SExtractor 的關係

### 1.1 一句話定位(依 SEP 論文 Barbary 2016, JOSS)

> 「SEP makes available the core algorithms of Source Extractor in a library of stand-alone functions
> and classes. These operate directly on in-memory arrays (no FITS files or configuration files). The
> code is derived from the Source Extractor code base (written in C) and aims to produce results
> **compatible with Source Extractor whenever possible**.」— Barbary (2016)

也就是說:**sep 把 SExtractor 的核心 C 演算法拆成可直接在記憶體 NumPy 陣列上呼叫的函式**,
不經過 FITS 檔、不經過設定檔。SExtractor 本身是一支命令列程式:讀 FITS、依 `.sex` 設定檔跑一連串工作、
最後輸出 FITS/ASCII catalog。sep 把中間那些演算法直接開放給你在 Python 裡呼叫。

### 1.2 sep **包含**哪些 SExtractor 功能(論文原文列舉)

> 「From Source Extractor, SEP includes background estimation, image segmentation (including on-the-fly
> filtering and source deblending), aperture photometry in circular and elliptical apertures, and source
> measurements such as Kron radius, "windowed" position fitting, and half-light radius.」— Barbary (2016)

對應到本專案用得到的部分:

| SExtractor 核心工作 | sep 對應 | 本專案是否用到 |
|---|---|---|
| 背景/雜訊估計(spatially variable) | `sep.Background`(`.back()` / `.rms()` / `.globalback` / `.globalrms`) | ✔ 用 `bkg` 減背景、`bkg.rms()` 當偵測門檻 |
| 門檻分割 segmentation | `sep.extract(..., segmentation_map=True)` | ✔ 產生 source mask |
| on-the-fly filtering(偵測前平滑) | `sep.extract(filter_kernel=..., filter_type=...)` | ✔ 高斯 matched filter(核=seeing) |
| deblending 拆分重疊源 | `sep.extract(deblend_nthresh=, deblend_cont=)` | ✔ 用預設 32 / 0.005 |
| CLEAN(清除偵測假影) | `sep.extract(clean=, clean_param=)` | 用預設 `clean=True` |
| 圓/橢圓孔徑測光 | `sum_circle` / `sum_ellipse` / `sum_circann` / `sum_ellipann` | 示範腳本用到;正式流程主要做偵測 |
| Kron 半徑、windowed 位置、半光半徑 | `kron_radius` / `winpos` / `flux_radius` | 備查(完整性) |

### 1.3 sep **額外新增、SExtractor 本體沒有**的功能(論文原文)

> 「Additionally, several features not in Source Extractor have been added:
> - Optimized matched filter for variable noise in source extraction.
> - Circular annulus and elliptical annulus aperture photometry functions.
> - Local background subtraction in shape consistent with aperture in aperture photometry functions.
> - Exact pixel overlap mode in all aperture photometry functions.
> - Masking of elliptical regions on images.」— Barbary (2016)

其中第一項(**針對變動雜訊優化的 matched filter**,即 `filter_type='matched'`)正是本專案偵測暗暈的核心;
詳見 §4.4。

### 1.4 sep **不包含**哪些 SExtractor 功能(用完整 SExtractor 才有的)

以下兩類:(a) 論文/文件直接說明 sep 不做的;(b) 由本機 `sep` API 沒有對應函式(核對 `dir(sep)`)、
而 SExtractor 參數手冊有列的功能。**要做這些事就得回去用完整 SExtractor(或 PSFEx),sep 不提供。**

| 完整 SExtractor 有、sep **沒有** | 說明 | 依據 |
|---|---|---|
| **FITS / WCS I/O** | sep 只吃/吐記憶體 NumPy 陣列,不讀 FITS、不解 WCS,天球座標(`ALPHA_J2000`/`DELTA_J2000`)得自己用 astropy WCS 算 | 論文:「operate directly on in-memory arrays (no FITS files …)」;`dir(sep)` 無 WCS |
| **設定檔(`.sex` / config file)** | 所有參數以函式引數傳入,沒有設定檔機制 | 論文:「… or configuration files」 |
| **星/星系分類 `CLASS_STAR`(類神經網路)** | SExtractor 內建 back-prop 神經網路輸出 stellarity index(0=延展,1=點源);sep 無此函式 | `dir(sep)` 無;SExtractor 手冊有 `CLASS_STAR` |
| **模型擬合 / PSF 測光(`MODEL_*`、`SPREAD_MODEL`)** | 需外部 PSFEx 產的 PSF 模型;SExtractor 才能做 PSF-corrected model fitting。sep 完全不做參數化模型擬合 | `dir(sep)` 無;需 PSFEx |
| **星等/零點 catalog 機制(`MAG_*`、`MAG_ZEROPOINT`)** | sep 只回傳 flux(ADU 求和),不換算星等、不管零點 | `extract` 回傳欄位只有 `flux`/`cflux`/`peak` 等,無 MAG |
| **Petrosian、ISOCOR 等成套 catalog 量** | sep 提供基本 moments/ellipse/Kron/flux_radius,但不輸出 SExtractor 那整套 catalog 參數 | `dir(sep)` 對照 |
| **`ASSOC`(與外部星表關聯)** | 無 | SExtractor 手冊有 `ASSOC_*` |
| **`WEIGHT_TYPE` 各種權重圖類型** | sep 只接受 `err`(σ 圖)或 `var`(變異圖)兩種每像素雜訊;沒有 SExtractor 的 `MAP_WEIGHT`/`MAP_RMS`/`MAP_VAR`/`BACKGROUND` 等多型別權重機制 | `extract` 只有 `err`/`var` 引數 |
| **多重 CHECKIMAGE 輸出** | sep 只給 segmentation map(`segmentation_map=True`)與 background/rms 陣列;不像 SExtractor 可吐 `-BACKGROUND`、`FILTERED`、`OBJECTS` 等一整組檢查圖 | API 對照 |

> **注意**:CLEAN 這一步 sep **有**做(`extract(clean=True)` 為預設),不屬於「不包含」清單。

### 1.5 數值等價性(誠實說明)

- 論文只承諾「**aims to produce results compatible with Source Extractor whenever possible**」,並明講
  「SEP is essentially a **fork** of Source Extractor that has already **diverged significantly** from the
  original code base」。
- 因此:sep 的背景/偵測是**同源 C 演算法**、預設值也對齊 SExtractor,結果在同設定下**高度接近**;但論文
  **未宣稱 bit-for-bit 逐位元相同**,且 sep 有原版沒有的路徑(如 variable-noise matched filter),那些路徑
  本就無從與原版逐位元比對。**「數值上與 SExtractor 高度相容但不保證逐位元相同」是本文可核實的最強結論;
  更強的「完全等價」宣稱本文未能查證,故不主張。**

### 1.6 sep 的實務陷阱(一定要處理,否則會報錯或吃記憶體)

1. **陣列必須是 native byte order 且 C-contiguous。**
   astropy 從 FITS 讀進來的是 **big-endian `>f4`**(FITS 標準,即使在 little-endian 機器上也是)。
   直接丟給 sep 會報:
   > `Input array with dtype '>f4' has non-native byte order. Only native byte order arrays are supported.`
   轉換方式(版本相依):
   - `numpy < 2.0`:`data = data.byteswap(inplace=True).newbyteorder()`
   - `numpy >= 2.0`:`data = data.astype(data.dtype.newbyteorder("="))`
   - 最省事:`data = np.ascontiguousarray(data, dtype=np.float32)`(同時解決 byte order + contiguity)。
     本專案 `sextract.py` 用 `.astype(np.float32)`、`step1_mask.py` 用 `np.ascontiguousarray(ha)`,都對。
2. **大/多團塊物件:調高兩個全域上限**(見 §4.6)。
   - `sep.set_extract_pixstack(size)`:內部 pixel buffer(**預設 300000**);影像大、源多時可能溢位,需調高。
   - `sep.set_sub_object_limit(limit)`:deblend 時單一物件的最大子物件數(**預設 1024**);
     延展亮源 deblend 出很多子塊時會撞上限,需調高(`sextract.py` 設 4096)。
3. **記憶體**:`Background` 會保存整張影像大小的背景與 rms 模型;`extract(segmentation_map=True)` 會多產一張
   與影像同大小的 int 標籤圖。對 MUSE 白光/窄帶 2D 影像(數百×數百)不成問題,但若對整個 cube 逐平面跑要留意。

---

## 2. 本專案的正式呼叫(對照 `src/0706/step1_mask.py`)

```python
import sep, numpy as np
from scipy import ndimage as ndi

invalid = ~valid                                   # valid = 有效視場(white != 0)
bkg = sep.Background(np.ascontiguousarray(ha), mask=invalid,
                     bw=256, bh=256, fw=3, fh=3)    # 背景框 256 > 暈;fw/fh=3 是預設中值濾波
_, seg = sep.extract(ha - bkg, 2.0, err=bkg.rms(), mask=invalid,
                     minarea=30,
                     filter_kernel=gauss(15, fwhm=6),   # matched filter,核 FWHM=seeing≈6px
                     deblend_nthresh=32, deblend_cont=0.005,
                     segmentation_map=True)
src = ndi.binary_dilation((seg > 0) & valid, iterations=6) & valid   # 事後膨脹 ≈ 1×seeing
```

> ⚠️ **數值待對齊**:上段忠實對照現行 `step1_mask.py`,其中 `fwhm=6` / `minarea=30` / `iterations=6`
> 源自 header `ESO QC EXPCOMB FWHM MEDIAN`,但**該關鍵字在本批資料為 0.0(未填)**,故這組是**未驗證的舊假設**。
> 依實測星點 PSF(≈4.06 px = 0.81″,見 §6),正確值應為 **核 FWHM≈4 px、minarea≈13 px、dilation≈4 px**;
> 程式碼的對齊由負責 `.py` 的人處理,本文的推導配方(§6)已採用實測值。

各參數為何是這些值(物理理由)見
[`docs/segmentation-parameters-explained.md`](./segmentation-parameters-explained.md);
本文以下給**通用**意義、SExtractor 對照、與**任一 cube 的推導公式**。

---

## 3. `sep.Background` 完整參數參考

**本機簽章(sep 1.4.1,核對自 `sep.Background.__doc__`)**:

```
Background(data, mask=None, maskthresh=0.0, bw=64, bh=64, fw=3, fh=3, fthresh=0.0)
```

背景估計的做法:把影像切成 `bw×bh` 的方格,每格用 sigma-clip 排掉源後估當地背景與 RMS,再以
`fw×fh` 格的中值濾波平滑格點,最後內插回每個像素。**它會自動排源估雜訊**,所以 `bkg.rms()` 是「乾淨天空」
的 σ(這正是門檻該用的 σ,見 §5)。

| sep 引數 | SExtractor 關鍵字 | 物理意義 | 型別 / 單位 | 預設 | 調大 / 調小的效果 | 如何選 |
|---|---|---|---|---|---|---|
| `data` | (輸入影像) | 要估背景的 2D 影像 | ndarray(native, C-contig) | — | — | 傳 native `float32`(見 §1.6) |
| `bw`, `bh` | `BACK_SIZE` | 背景方格邊長 | int / px | **64** | **大**→背景更平滑、不會把延展源當背景吃掉,但對真正的背景梯度反應變鈍;**小**→貼合小尺度背景變化,但**會把大於格子的延展源當成背景減掉** | **必須大於你要保留的最大天體**;延展暈就用全域或 `bw ≥ 物件直徑`(見 §6) |
| `fw`, `fh` | `BACK_FILTERSIZE` | 對背景格點做中值濾波的視窗(以「格」為單位) | int / 格 | **3** | 大→更能壓掉被亮源污染的格點,但背景更糊;小→背景更貼近局部 | 一般維持 3(SExtractor 預設亦 3) |
| `fthresh` | `BACK_FILTERTHRESH` | 中值濾波的門檻(格點差異超過才濾) | float | **0.0** | 提高→只在格點明顯異常時才濾 | 一般 0.0 |
| `mask` | (≈ `FLAG`/無效像素) | 標記無效/不參與估背景的像素 | ndarray(bool 或數值) | None | — | 把無效視場、已知源遮起來 |
| `maskthresh` | — | `mask` 為數值時的上限門檻(≤ 此值才算未遮罩) | float | **0.0** | — | bool mask 時用預設即可 |

**`BACK_TYPE`(AUTO / MANUAL)對照**:sep.Background **只做 AUTO**(永遠從資料估背景);若要「MANUAL 常數背景」,
自己減一個常數即可,sep 沒有對應開關。

**衍生屬性 / 方法**(核對自 `dir(sep.Background)` 與文件):

| 成員 | 型別 | 意義 |
|---|---|---|
| `bkg.globalback` | float | 全影像單一背景值(scalar) |
| `bkg.globalrms` | float | 全影像單一背景 RMS(scalar σ);雜訊近似均勻時可當 `err` 用 |
| `bkg.back()` | 2D ndarray | 每像素背景模型(與影像同大小) |
| `bkg.rms()` | 2D ndarray | 每像素背景 RMS 圖(σ 圖);雜訊不均勻時當 `err=` 傳入 `extract` |
| `bkg.subfrom(data)` | in-place | 從 `data` 就地減背景(等價 `data - bkg`) |
| `data - bkg` / `bkg.subfrom` | — | `Background` 物件支援直接被減,等於減掉 `back()` |

> 本專案傳 `err=bkg.rms()`(每像素 σ 圖),比傳單一 `globalrms` 更穩健,因為視場邊緣/拼接處雜訊不均。

---

## 4. `sep.extract` 完整參數參考

**本機完整簽章(sep 1.4.1,核對自 `inspect`;文件精簡版省略了 `var`/`gain`/`maskthresh`,但實際都接受)**:

```
extract(data, thresh, err=None, var=None, gain=None, mask=None, maskthresh=0.0,
        minarea=5, filter_kernel=default_kernel, filter_type='matched',
        deblend_nthresh=32, deblend_cont=0.005, clean=True, clean_param=1.0,
        segmentation_map=None)
```

其中 `default_kernel` = 3×3 的 `[[1,2,1],[2,4,2],[1,2,1]]`(float32),等同 SExtractor 預設 `default.conv`
(FWHM≈2px 的高斯)。

### 4.1 偵測與門檻

| sep 引數 | SExtractor 關鍵字 | 物理意義 | 型別 / 單位 | 預設 | 調大 / 調小 | 如何選 |
|---|---|---|---|---|---|---|
| `data` | (輸入影像) | **已減背景**的影像 | 2D ndarray | — | — | 傳 `image - bkg` |
| `thresh` | `DETECT_THRESH` | 偵測門檻;**給了 `err`/`var` 時是相對值**:像素 (j,i) 的絕對門檻 = `thresh * err[j,i]`(或 `thresh*sqrt(var)`) | float | — | 高→只抓亮源、漏暗源;低→抓到更暗但假陽性暴增 | **≥ 2σ**(見 §6 / §5) |
| `err` | (≈ `MAP_RMS` / `WEIGHT`) | 每像素雜訊 σ(**scalar 或 2D**)。把 `thresh` 變成相對門檻,並供 matched filter 用 | float / ndarray | None | — | 傳 `bkg.rms()`(σ 圖)或 `bkg.globalrms` |
| `var` | (≈ `MAP_VAR`) | 每像素**變異**(= σ²);與 `err` **擇一** | float / ndarray | None | — | 有 var 圖時用它 |
| `gain` | `GAIN` | data 單位→光子數的換算,**只影響 flux 誤差的 Poisson 項,不影響偵測** | float | None | — | 有增益才給;純偵測可省略 |
| `mask` | (無效像素) | 遮住的像素;遮罩**發生在濾波之前**,等於把 data 設 0、雜訊設 ∞ | ndarray | None | — | 遮無效視場 |
| `maskthresh` | — | `mask` 為數值時的門檻 | float | 0.0 | — | bool mask 用預設 |

### 4.2 面積與濾波(matched filter)

| sep 引數 | SExtractor 關鍵字 | 物理意義 | 型別 / 單位 | 預設 | 調大 / 調小 | 如何選 |
|---|---|---|---|---|---|---|
| `minarea` | `DETECT_MINAREA` | 一個偵測至少要有幾個相連(超門檻)像素才算源 | int / px | **5** | 高→濾掉更多小雜點,但可能漏掉小源;低→保留小源但撿到雜訊碎點 | **≈ 1 個 PSF 面積 = π(FWHM/2)²**(見 §6) |
| `filter_kernel` | `FILTER` + `FILTER_NAME` | 偵測前的 on-the-fly 平滑核;設 `None` 則不濾波 | 2D ndarray / None | 3×3 高斯 | 核寬大→更壓雜訊、利於暗延展源,但糊掉小結構、位置變差;核寬小→保細節但壓噪不足 | **高斯核,FWHM ≈ seeing(PSF FWHM)**(見 §4.4、§6) |
| `filter_type` | `DETECT_TYPE`(概念不同,見下) | `'matched'` 考慮 kernel 內的逐像素雜訊(需 `err`);`'conv'` 只做單純卷積、忽略像素間雜訊差異 | {'matched','conv'} | **'matched'** | 雜訊均勻時兩者等價;雜訊快速變動(如拼接重疊區)時 `'matched'` 偵測暗源更好 | 用預設 `'matched'`,並務必給 `err`(見 §4.4) |

> `filter_type` **不是** SExtractor 的 `DETECT_TYPE`(那是 CCD 線性 vs PHOTO 對數響應)。sep 假設線性(CCD)響應;
> `filter_type='matched'` 是 sep **新增**的 variable-noise 匹配濾波,是 SExtractor 本體沒有的路徑。

### 4.3 deblend 與 clean

| sep 引數 | SExtractor 關鍵字 | 物理意義 | 型別 | 預設 | 調大 / 調小 | 如何選 |
|---|---|---|---|---|---|---|
| `deblend_nthresh` | `DEBLEND_NTHRESH` | deblend 時在 [門檻, 峰值] 間切幾層等亮度來找子峰 | int | **32** | 多→更能分開靠近的源,計算量增;少→傾向不分 | 用預設 32 |
| `deblend_cont` | `DEBLEND_MINCONT` | 子峰要占母源總通量的最小比例才算獨立源;**設 1.0 完全關閉 deblend** | float [0,1] | **0.005** | 高→較不積極拆分(把黏在一起的當一個);低→更積極拆 | 用預設 0.005;要「不拆」設 1.0 |
| `clean` | `CLEAN` | 是否清除亮源周圍因翼/雜訊產生的假偵測 | bool | **True** | — | 用預設 True |
| `clean_param` | `CLEAN_PARAM` | CLEAN 的效率參數(見 SExtractor 手冊) | float | **1.0** | 大→清得更兇 | 用預設 1.0 |

### 4.4 matched filter 理論(為何核 FWHM ≈ seeing;對延展暈的限制)

- **原理**:偵測「已知形狀的訊號 + 白雜訊」時,理論上 S/N 最大化的濾波器,就是**與訊號本身同形狀的核**
  (matched filter 定理)。天文點源被大氣糊成 PSF,故點源偵測的最佳核 ≈ **PSF(高斯,FWHM = seeing)**。
- **為何是 seeing 這個尺度**:seeing FWHM 是影像裡「真實結構的最小尺度」——比它更細的變化不可能是真天體
  (都被大氣糊成 seeing 寬)。在這個尺度平滑,**最大程度壓掉雜訊、又不糊掉任何真結構**。核太小壓噪不足;
  核太大連真結構的位置形狀都糊掉。
- **為何 `filter_type='matched'` 需要 `err`**:variable-noise matched filter 會用每像素雜訊 σ 對 kernel 加權;
  沒有 `err` 就退化成單純卷積(`'conv'`)。所以本專案一定同時給 `err=bkg.rms()`。
- **對本專案 Haro11 暗暈的限制(重要)**:matched filter 對**點源**最佳,但 Haro11 的 Hα 暈是**延展、
  低表面亮度**的結構,其尺度遠大於 seeing。核 = seeing 能把暈的每像素 S/N 拉高到可用標準 2σ 抓到的程度
  (這正是我們的做法);但它**不是暈這種大尺度結構的嚴格最佳核**。**正解仍是「先用 matched filter 抬高 S/N、
  再配大背景框(§6)避免把暈當背景吃掉、再用標準 2σ」,而非把門檻壓到 2σ 以下**(見 §5 的反例)。

### 4.5 `extract` 回傳(重點欄位)

`extract` 回傳 structured array(每列一個源),`segmentation_map=True` 時另回傳一張 int 標籤圖
(0=無源,`i+1`=第 i 個源的像素)。常用欄位:

| 欄位 | 意義 |
|---|---|
| `x`, `y` | 質心(一階矩) |
| `npix` / `tnpix` | 屬於該源的像素數 / 超門檻(未卷積)像素數 |
| `flux` / `cflux` | 成員像素(未卷積 / 卷積後)通量和 |
| `peak` / `cpeak` | (未卷積 / 卷積後)峰值 |
| `a`, `b`, `theta` | 橢圓半長軸/半短軸/傾角(二階矩導出) |
| `xmin/xmax/ymin/ymax` | 邊界框 |
| `flag` | 抽取旗標(如 `OBJ_MERGED`、`OBJ_TRUNC`、`OBJ_DOVERFLOW`) |

> 本專案只需要 segmentation map(哪些像素是源),故用 `_, seg = sep.extract(..., segmentation_map=True)`
> 取第二個回傳即可。

### 4.6 全域限制(大/多團塊物件)

| 函式 | 預設 | 何時要調 | 對應 |
|---|---|---|---|
| `sep.set_extract_pixstack(size)` / `get_extract_pixstack()` | **300000** | 影像大、超門檻像素多時內部 pixel buffer 溢位 → 報 pixstack 相關錯誤,調高 | SExtractor 內部 `MEMORY_PIXSTACK` |
| `sep.set_sub_object_limit(limit)` / `get_sub_object_limit()` | **1024** | deblend 出很多子塊(延展亮源/星系)撞上限,調高(如 4096) | deblend 子物件上限 |

---

## 5. 孔徑測光與量測工具(完整性;本專案主要做偵測)

`dir(sep)` 提供的量測函式(簽章見 `sep.readthedocs.io/en/stable/reference.html`):

| 函式 | 用途 | SExtractor 對應概念 |
|---|---|---|
| `sum_circle(data, x, y, r, err=, var=, ...)` | 圓孔徑內通量和 | `FLUX_APER` |
| `sum_circann(data, x, y, rin, rout, ...)` | 圓環孔徑(sep 新增) | (SExtractor 無直接對應) |
| `sum_ellipse(data, x, y, a, b, theta, r, ...)` | 橢圓孔徑內通量和 | `FLUX_AUTO` 家族的孔徑 |
| `sum_ellipann(data, x, y, a, b, theta, rin, rout, ...)` | 橢圓環孔徑(sep 新增) | — |
| `kron_radius(data, x, y, a, b, theta, r)` | 算 Kron「半徑」 | `KRON_RADIUS`(→`MAG_AUTO`) |
| `flux_radius(data, x, y, rmax, frac, ...)` | 含指定通量比例的半徑(如半光半徑) | `FLUX_RADIUS` |
| `winpos(data, xinit, yinit, sig, ...)` | windowed 質心(更準的位置) | `XWIN_IMAGE`/`YWIN_IMAGE` |
| `mask_ellipse(arr, x, y, a, b, theta, r=)` | 在陣列上把橢圓區域標記為遮罩(sep 新增) | — |
| `ellipse_axes` / `ellipse_coeffs` | 橢圓參數 (a,b,θ) ↔ (cxx,cyy,cxy) 互換 | — |

孔徑測光都支援 `err`/`var`(誤差)、`gain`(Poisson 項)、`mask`、以及 exact/subpixel 重疊模式與
局部背景相減(sep 新增,見 §1.3)。**本專案正式流程不需要這些**,列此僅為完整參考。

---

## 6. 逐檔調參配方(關鍵交付:任一新 cube 如何從 header/資料推出每個參數)

**核心原則(CLAUDE.md Principle 2)**:每個參數都**從資料本身(header 關鍵字、量到的尺度)推出**,
不用猜、不為了讓數字好看或跑快而挑。以下公式對任一 MUSE(或類似)cube 通用。

### 6.1 推導公式

| 參數 | 由什麼推 | 公式 | Haro11 值 |
|---|---|---|---|
| **pixel scale** `pixscale` | header `CD1_1`(退而求其次 `CDELT1`) | `√(CD1_1² + CD2_1²) × 3600` (deg→arcsec);或 `|CD1_1|×3600` | 0.20 ″/px |
| **seeing FWHM(px)** | **實測 cube 內星點 PSF**(見下方說明);header `ESO QC EXPCOMB FWHM MEDIAN` 在本批資料 **= 0.0(未填),不可用** | `FWHM_px = median(2.3548·√(a·b))`;退路 `ESO OCS SGS AG FWHM{X,Y} MED` 或 `ESO TEL AMBI FWHM` ÷ pixscale | **≈4.06 px ≈ 0.81″**(實測) |
| **matched-filter 核 FWHM** | = seeing FWHM | `kernel_FWHM_px = seeing_FWHM_px` | ≈ **4 px** |
| **核尺寸(box)** | 由 σ=FWHM/2.355 取 ±3σ | `size = 2·⌈3σ⌉+1`(奇數) | 13 px |
| **偵測門檻** | 固定統計標準 | `thresh = 2.0`(≥2σ) | 2σ |
| **minarea** | 1 個 PSF 面積 | `π·(FWHM/2)²` | π·2² ≈ 13 → **13 px** |
| **dilation** | 1×seeing | `round(seeing_FWHM_px)` | ≈ **4 px** |
| **背景框 `bw`** | 必須 > 最大要保留的物件 | 見 §6.2 | 256 px(或全域) |

> **seeing 怎麼來(優先序;CLAUDE.md Principle 2 的物理可辯護做法)**:
> 1. **首選——直接從 cube 內的星點量 PSF FWHM**(物理上最站得住腳):先做**去發射線的連續譜白光影像**,
>    `sep.extract` 抽源,對每個源以二階矩算 `FWHM = 2.3548·√(a·b)`,只留**緊緻**(FWHM<8 px)、**圓**(b/a>0.6)、
>    **夠亮**的星,取中位數。本批 Haro11 實測 **≈4.06 px = 0.81″**(10 顆星,16–84% 範圍 3.58–4.77 px)。
> 2. **無星可用時的退路——header 代理值**:`ESO OCS SGS AG FWHMX/Y MED`(自動導星)≈0.886″≈4.4 px、
>    或 `ESO TEL AMBI FWHM`(DIMM)≈0.94–0.96″≈4.7 px;三者一致指向 ≈4 px。
> 3. ⚠️ **不可盲用**:`ESO QC EXPCOMB FWHM MEDIAN` 在 `Haro11_nosky.fits` 與 `Haro11_NEpointing_esonosky.fits`
>    **都是 0.0(未填)**,直接讀會得到 **0-px 核**;`ESO OCS SGS FWHM *` 同樣為 0.0。
>
> **舊值 6 px / 1.24″ 是未經 header 佐證的假設,已由上述實測(≈4 px / 0.81″)取代。** §6.4 的程式碼含
> `assert fwhm_px > 0` 的 code guard,確保空關鍵字不會再無聲地產生 0-px 核。

### 6.2 門檻為何 **≥ 2σ**(不可壓更低)

門檻是「純雜訊被誤判成源」的假陽性率(高斯單尾):

| 門檻 | 純雜訊超過的機率 | 判定 |
|---|---|---|
| **0.75σ** | **≈ 23%** | **物理站不住腳**(等於在撿雜訊) |
| 1.5σ | ≈ 6.7% | 偏低 |
| **2σ** | **≈ 2.3%** | **標準、可辯護**(天文常用 2–5σ) |

> ⚠️(CLAUDE.md Principle 2 的具體警告)**若有人為了抓暗暈把門檻壓到 0.75σ,這是物理上不可辯護的**
> (低於雜訊本身、23% 假陽性)。**正解不是降門檻,而是先用 matched filter(核=seeing)把訊號 S/N 抬高,
> 再用正常的 ≥2σ 門檻。** 這是本專案的既定做法,不是可選項。

### 6.3 背景框 `bw`:必須大於要保留的物件(失敗模式)

- **失敗模式**:`bw` 若**小於**延展物件,背景方格整個泡在物件裡,會把**物件本身當成背景**估出來並減掉 →
  物件消失、**無論門檻多低都偵測不到**。SExtractor 預設 `bw=64` 對 Haro11 的暈(Ø≈226 px)就是這個坑。
- **規則**:`bw` **要大於你想保留的最大天體直徑**。
  - 一般源:`bw` 取「大於最大物件」的 2 的次方(如 128/256)。
  - **巨大延展暈(本專案)**:用**全域背景**(`bw ≥ 影像邊長`)或 `bw ≥ 物件直徑`。Haro11 暈 Ø≈226 px → `bw=256`。
- 此點與 CLAUDE.md 檢查表一致(「halo Ø≈226 px → global / bw ≥ 256」)。

### 6.4 可直接複製的推導程式

```python
import numpy as np
from astropy.io import fits

def cube_detection_params(fits_path, hdu="DATA", object_diameter_px=None):
    """從一個 cube 的 header 推出所有 sep 偵測參數(通用,不靠猜)。
       object_diameter_px:你要保留的最大天體直徑(px)。延展暈就傳它或用全域背景。"""
    hdr = fits.getheader(fits_path, hdu)

    # 1) pixel scale(arcsec/px):優先用 CD 矩陣,退而求其次 CDELT
    if "CD1_1" in hdr:
        cd1_1 = hdr["CD1_1"]; cd2_1 = hdr.get("CD2_1", 0.0)
        pixscale = np.hypot(cd1_1, cd2_1) * 3600.0            # deg -> arcsec
    else:
        pixscale = abs(hdr["CDELT1"]) * 3600.0

    # 2) seeing FWHM(px):優先「量」不「讀」——見 measure_psf_fwhm_px()。
    #    ⚠️ ESO QC EXPCOMB FWHM MEDIAN 在本批資料 = 0.0(未填),直接讀會得到 0-px 核,絕不可盲用。
    fwhm_px = measure_psf_fwhm_px(fits_path)                  # 首選:實測 cube 內星點 PSF
    if not (fwhm_px and fwhm_px > 0):                         # 無星可用 → 退到 header seeing 代理
        for key in ("ESO OCS SGS AG FWHMX MED", "ESO OCS SGS AG FWHMY MED",
                    "ESO TEL AMBI FWHM"):                     # 皆為 arcsec;非 QC EXPCOMB
            v = hdr.get(key, 0.0)
            if v and v > 0:
                fwhm_px = v / pixscale
                break
    # code guard:空關鍵字/量測失敗都不得無聲地產生 0-px 核
    assert fwhm_px and fwhm_px > 0, (
        "seeing FWHM 量測與 header 代理皆失敗(QC EXPCOMB 關鍵字本批 = 0.0);"
        "拒絕產生 0-px 核——請提供可用星點或有效 seeing 代理。")

    # 3) 由尺度推偵測參數
    sigma = fwhm_px / 2.355
    ksize = int(2 * np.ceil(3 * sigma) + 1)                  # 奇數核 box(±3σ)
    p = dict(
        pixscale_arcsec = pixscale,
        seeing_fwhm_px  = fwhm_px,
        kernel_fwhm_px  = fwhm_px,                           # matched filter ≈ seeing
        kernel_size_px  = ksize,
        thresh_sigma    = 2.0,                              # ≥ 2σ(假陽性 2.3%),不可更低
        minarea_px      = int(round(np.pi * (fwhm_px / 2) ** 2)),   # ≈ 1 PSF 面積
        dilate_px       = int(round(fwhm_px)),              # ≈ 1×seeing
    )

    # 4) 背景框:必須 > 最大物件;延展暈就用全域背景
    if object_diameter_px is not None:
        p["bw_px"] = int(2 ** np.ceil(np.log2(object_diameter_px)))  # >= 物件的 2 次方
    else:
        p["bw_px"] = None                                    # None = 用全域背景(bw >= 影像邊長)
    return p


def gauss_kernel(size, fwhm):
    """高斯 matched filter 核(FWHM = seeing);正規化到 sum=1,回傳 native float32。"""
    x = np.arange(size) - size // 2
    g = np.exp(-(x ** 2) / (2 * (fwhm / 2.355) ** 2))
    k = np.outer(g, g)
    return (k / k.sum()).astype(np.float32)


def measure_psf_fwhm_px(fits_path):
    """首選 seeing 來源:直接從 cube 內的星點量 PSF FWHM(px)。
       物理上最站得住腳(CLAUDE.md Principle 2);當 header QC 關鍵字為 0.0 時尤其必要。
       步驟:去發射線的連續譜白光影像 → sep.extract → 每源以二階矩算
       FWHM = 2.3548·√(a·b),只留緊緻(<8 px)、圓(b/a>0.6)、夠亮的星 → 取中位數。
       無可用星回傳 None(交由呼叫端退到 header 代理)。"""
    import sep
    from astropy.io import fits
    cube  = fits.getdata(fits_path)                          # (nλ, ny, nx)
    white = np.nanmedian(cube, axis=0)                       # 連續譜白光(中值壓掉發射線)
    white = np.ascontiguousarray(white, np.float32)
    bkg   = sep.Background(white)
    obj   = sep.extract(white - bkg, 5.0, err=bkg.globalrms) # 星點夠亮,用一般 5σ
    if obj is None or len(obj) == 0:
        return None
    fwhm  = 2.3548 * np.sqrt(obj["a"] * obj["b"])            # 每源 FWHM(px)
    ba    = obj["b"] / obj["a"]                              # 圓度 b/a
    star  = (fwhm < 8) & (ba > 0.6) & (obj["flux"] > np.nanmedian(obj["flux"]))
    return float(np.median(fwhm[star])) if star.any() else None
```

用法示意(接 `sep`):

```python
p = cube_detection_params("data/Haro11_nosky.fits", object_diameter_px=226)
ker = gauss_kernel(p["kernel_size_px"], p["kernel_fwhm_px"])
bkg = sep.Background(np.ascontiguousarray(image, np.float32),
                     bw=p["bw_px"] or image.shape[1], bh=p["bw_px"] or image.shape[0], fw=3, fh=3)
_, seg = sep.extract(image - bkg, p["thresh_sigma"], err=bkg.rms(),
                     minarea=p["minarea_px"], filter_kernel=ker, filter_type="matched",
                     segmentation_map=True)
```

---

## 7. 一頁對照總表(sep ↔ SExtractor ↔ 本專案值 ↔ 推導)

| 概念 | sep API | SExtractor 關鍵字 | 本專案(Haro11)值 | 從哪推 |
|---|---|---|---|---|
| 背景框 | `Background(bw=,bh=)` | `BACK_SIZE` | 256 px | > 暈直徑(226 px)/ 全域 |
| 背景中值濾波 | `Background(fw=,fh=)` | `BACK_FILTERSIZE` | 3 | 預設 |
| 背景型別 | (只有 AUTO) | `BACK_TYPE` | AUTO | sep 恆估背景 |
| 每像素雜訊 σ | `err=bkg.rms()` | `MAP_RMS`/`WEIGHT` | `bkg.rms()` | 排源後估(乾淨天空) |
| 偵測門檻 | `extract(thresh=)` | `DETECT_THRESH` | 2σ | ≥2σ(假陽性 2.3%) |
| 最小面積 | `extract(minarea=)` | `DETECT_MINAREA` | 13 px | π(FWHM/2)² |
| 平滑核 | `filter_kernel=` | `FILTER`+`FILTER_NAME` | 高斯 FWHM≈4 | = seeing(實測 PSF) |
| 濾波型別 | `filter_type='matched'` | (sep 新增) | matched | 變動雜訊最佳,需 `err` |
| deblend 層數 | `deblend_nthresh=` | `DEBLEND_NTHRESH` | 32 | 預設 |
| deblend 對比 | `deblend_cont=` | `DEBLEND_MINCONT` | 0.005 | 預設 |
| CLEAN | `clean=`,`clean_param=` | `CLEAN`,`CLEAN_PARAM` | True,1.0 | 預設 |
| 膨脹(事後) | `scipy.ndimage.binary_dilation` | (SExtractor 無;事後處理) | 4 px | 1×seeing |
| 大物件上限 | `set_sub_object_limit` | `DEBLEND` 相關 | 視需要(4096) | 延展亮源才調 |
| pixel buffer | `set_extract_pixstack` | `MEMORY_PIXSTACK` | 視需要 | 大圖才調 |

---

## 8. 交叉引用與參考文獻

- **物理理由(Haro11 這批資料為什麼這樣設)**:[`docs/segmentation-parameters-explained.md`](./segmentation-parameters-explained.md)
- **核心原則與操作清單**:[`CLAUDE.md`](../CLAUDE.md)(Principle 2、Operational Checklist for Principle 2)
- **實作**:`src/0706/step1_mask.py`(正式流程)、`src/0706/settings.py`(參數定義)、`src/explore/sextract.py`(最小示範)

**權威來源**:

- Barbary, K. (2016). *SEP: Source Extractor as a library.* Journal of Open Source Software, 1(6), 58.
  doi:10.21105/joss.00058 —— sep 的定位、包含/新增功能、與 SExtractor 的關係。
- Bertin, E. & Arnouts, S. (1996). *SExtractor: Software for source extraction.* A&AS, 117, 393–404.
  doi:10.1051/aas:1996164 —— 原始演算法(背景、分割、deblend、CLEAN)。
- Bertin, E. (2016). *SExtractor.* http://www.astromatic.net/software/sextractor;
  參數手冊 https://sextractor.readthedocs.io/ —— 完整功能(`CLASS_STAR`、model fitting、WCS、`ASSOC`)。
- sep 官方文件:https://sep.readthedocs.io/en/stable/ —— API 參考、byte-order 要求、tutorial。
- 本機套件:`sep 1.4.1`(`github.com/sep-developers/sep`)—— 本文所有簽章、預設值的實測依據。
</content>
</invoke>
