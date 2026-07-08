"""
評估共用工具 —— 把「所有 blank(非源) spaxel」的逐波長殘餘統計出來(mean & median)。

為什麼用全部 blank、不取樣：
  評估度量用全部 blank spaxel,中位數/平均更精準、且完全可重現(不依賴隨機種子),
  天空線處也更平滑。評估直接讀 mask.fits,對所有 valid blank 統計,不做取樣。
  記憶體：逐波長分塊讀取,峰值 ~ chunk × N_blank × 4 bytes(數百 MB)。
  結果快取到 results/zap/blankstats/,三條曲線被 muse/zap 兩支圖共用,重跑瞬間完成。
  (mask 若重建,請清掉 results/zap/blankstats/ 讓快取失效。)

blank 的定義 = valid & ~source,兩個條件都要：
  ~source : 非源(遮罩外)。
  valid   : 在視場(FoV)內。⚠ FoV 外的 spaxel 在 raw cube 是 NaN,但 ZAP 會把它們
            填成有限值(≈0),若不排除會有 ~2.8 萬個邊緣 spaxel 把中位數往 0 拉偏。
            valid 由 raw(mask_from) 全波長 nansum != 0 決定(FoV 外全 NaN→0),與 step1 一致。
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings
import numpy as np
from astropy.io import fits

BLANKSTATS_DIR = settings.OUTPUT_ROOT / "blankstats"

def _valid2d(mask_from, chunk=200):
    """FoV 有效遮罩：raw(mask_from) 全波長 nansum != 0(FoV 外全 NaN→0)。快取成 .npy。"""
    cache_path = BLANKSTATS_DIR / f"valid_from-{mask_from}.npy"
    if cache_path.exists():
        return np.load(str(cache_path))
    with fits.open(str(settings.input_cube_path(mask_from)), memmap=True) as hdul:
        cube_data = hdul["DATA"].data; n_wavelengths, n_rows, n_cols = cube_data.shape
        flux_sum = np.zeros((n_rows, n_cols))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for start in range(0, n_wavelengths, chunk):
                flux_sum += np.nansum(np.asarray(cube_data[start:min(start+chunk, n_wavelengths)]), axis=0)
    valid = flux_sum != 0
    BLANKSTATS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), valid)
    return valid

def _mask_suffix(mask_from, mask_method):
    return mask_from if mask_method == "sep" else f"{mask_method}-{mask_from}"   # sep 隱含(相容既有快取)

def base_method(mask_method):
    """box_sep→sep, box_claude→claude；其餘不變。box run 的殘餘要在「整張 base blank」上量
       (含邊緣、可與標準 run 比),而非只在盒內(盒內是 ZAP 學習區,必然乾淨)。"""
    return mask_method[4:] if mask_method.startswith("box_") else mask_method

def _blank2d(mask_from, mask_method="sep"):
    """blank = valid(FoV 內) & ~source(該方法的遮罩) 的 2D 布林遮罩。"""
    source_mask = fits.getdata(str(settings.source_mask_path(mask_from, mask_method))).astype(bool)
    return _valid2d(mask_from) & ~source_mask

def blank_residual(cube_path, mask_from, tag, mask_method="sep", hdu="DATA", chunk=200, use_cache=True):
    """對 cube 的所有 blank spaxel,回傳逐波長 (mean_spectrum, median_spectrum)。tag+方法+來源 決定快取檔名。"""
    cache_path = BLANKSTATS_DIR / f"{tag}_maskfrom-{_mask_suffix(mask_from, mask_method)}.npz"
    if use_cache and cache_path.exists():
        cached = np.load(str(cache_path)); return cached["mean"], cached["median"]
    blank = _blank2d(mask_from, mask_method)
    with fits.open(str(cube_path), memmap=True) as hdul:
        cube_data = hdul[hdu].data
        n_wavelengths = cube_data.shape[0]
        mean_spectrum = np.empty(n_wavelengths); median_spectrum = np.empty(n_wavelengths)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)         # 略過 all-NaN 波長列的警告
            for start in range(0, n_wavelengths, chunk):
                stop = min(start + chunk, n_wavelengths)
                blank_slab = np.asarray(cube_data[start:stop])[:, blank]        # (n_chunk_wavelengths, n_blank_spaxels)
                mean_spectrum[start:stop]   = np.nanmean(blank_slab, axis=1)
                median_spectrum[start:stop] = np.nanmedian(blank_slab, axis=1)
    mean_spectrum = mean_spectrum.astype(float); median_spectrum = median_spectrum.astype(float)
    BLANKSTATS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(str(cache_path), mean=mean_spectrum, median=median_spectrum)
    return mean_spectrum, median_spectrum

def wavelength_axis(cube_name="nosky"):
    with fits.open(str(settings.input_cube_path(cube_name))) as hd:
        return settings.wavelength_axis(hd["DATA"].header)

# 三條 M1 共用曲線 --------------------------------------------------------------
#   standard = nosky raw（MUSE pipeline 扣天空後的殘餘 = 判讀基準/「標準處理」）
#   sky      = wsky raw （這片天空本身，畫在右軸灰線 / 當 % 包絡的基準）
#   zap      = 本 run 的 zap.fits（target 經 ZAP 扣天空後的殘餘）
def standard_residual(mask_from, mask_method="sep"):
    std_cube, _ = settings.reference_cubes(mask_from)   # 主指向→nosky；NE→NEnosky
    return blank_residual(settings.input_cube_path(std_cube), mask_from, f"{std_cube}-raw", mask_method)
def sky_residual(mask_from, mask_method="sep"):
    _, sky_cube = settings.reference_cubes(mask_from)   # 主指向→wsky；NE→NEwsky
    return blank_residual(settings.input_cube_path(sky_cube), mask_from, f"{sky_cube}-raw", mask_method)
def zap_residual(target, mask_from, mask_method="sep"): return blank_residual(settings.zap_path(target, mask_from, mask_method), mask_from, f"{target}+zap", mask_method)

def check_args(target, mask_from, mask_method="sep"):
    if target not in settings.CUBE_NAMES or mask_from not in settings.CUBE_NAMES:
        sys.exit(f"用法: <target> <mask_from> [mask_method]，target/mask_from 皆須 ∈ {settings.CUBE_NAMES}")
    if not settings.zap_path(target, mask_from, mask_method).exists():
        sys.exit(f"找不到 {settings.zap_path(target, mask_from, mask_method)}，請先跑 step2_zap.py {target} {mask_from} --mask-method {mask_method}")
