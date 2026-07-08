"""
src/0706 pipeline 的【唯一設定檔】：所有路徑 + 物理參數只在這裡定義一次。
process(step*) 與 evaluate(eval_*) 全部 import 這裡；改參數只改這一份，不會分岔。

※ 所有相對路徑都相對「專案根目錄」，請一律從根目錄執行，例如：
    conda run -n astro python src/0706/step1_mask.py
"""
from pathlib import Path
import numpy as np

# per-cube 偵測設定（CubeConfig / DetectParams + lazy registry）集中在 cube_config.py。
# 這裡 re-export，讓 step*/eval_* 只要 import settings 就拿得到（不製造第二處真相來源）。
from cube_config import CubeConfig, DetectParams, get_cube_config  # noqa: F401

# ======================= 輸出路徑（2×2 結構）=======================
OUTPUT_ROOT = Path("results/zap")          # 所有產物的根目錄
MASKS_DIR   = OUTPUT_ROOT / "masks"        # step1 產物：source mask + blank 座標（小）
CUBES_DIR   = OUTPUT_ROOT / "cubes"        # step2 產物：每個實驗一個子資料夾（大）
DIAGNOSTICS_DIR = OUTPUT_ROOT / "diagnostics"   # 尚未拔擢 / 比較用的產物（不覆蓋正式 masks/）
FIGURE_DIR  = OUTPUT_ROOT                  # .png 圖存這裡

MASK_METHOD = "sep"                        # 預設方法（SEP + matched filter + 2σ；另一個是 "claude"）

# ======================= 輸入 cube（唯讀，永不刪）=======================
RAW_CUBE_PATHS = {"nosky": Path("data/Haro11_nosky.fits"),
                  "wsky":  Path("data/Haro11_wsky.fits"),
                  "NEwsky":  Path("data/Haro11_NEpointing_wsky.fits"),      # NE pointing, 含天空
                  "NEnosky": Path("data/Haro11_NEpointing_esonosky.fits")}  # NE pointing, ESO 已扣天空
CUBE_NAMES = ("nosky", "wsky", "NEwsky", "NEnosky")
def input_cube_path(cube_name):
    """要處理的原始 cube（整張全視場，不空間裁切）。"""
    return RAW_CUBE_PATHS[cube_name]

def reference_cubes(cube_name):
    """該指向的參考 raw cube：(standard=已扣天空的真值, sky=含天空)。M1 藍線 / 天空包絡用。"""
    return ("NEnosky", "NEwsky") if str(cube_name).startswith("NE") else ("nosky", "wsky")

# --- step1 產物：每個 (方法 × cube) 一個資料夾，把該遮罩的產物包在一起、彼此不覆蓋 ---
def mask_dir(from_cube, method=MASK_METHOD):
    return MASKS_DIR / f"{method}_from-{from_cube}"            # 例：masks/sep_from-nosky/
def source_mask_path(from_cube, method=MASK_METHOD):
    return mask_dir(from_cube, method) / "mask.fits"           # 2D uint8：1=源, 0=可用天空
def blanks_path(from_cube, method=MASK_METHOD):
    return mask_dir(from_cube, method) / "blanks.npz"          # 亮源座標(sy,sx)+波長軸（供 M3；不存 blank 取樣）

# --- step2 產物：ZAP，依「ZAP 對象 × mask(方法 × 來源)」命名，每個實驗一個資料夾 ---
def _run_suffix(mask_from, mask_method):
    return mask_from if mask_method == "sep" else f"{mask_method}-{mask_from}"     # sep 隱含（相容既有 run）；claude 才加前綴
def run_dir(target, mask_from, mask_method="sep"):
    return CUBES_DIR / f"{target}_maskfrom-{_run_suffix(mask_from, mask_method)}"  # sep: wsky_maskfrom-nosky；claude: wsky_maskfrom-claude-nosky
def zap_path(target, mask_from, mask_method="sep"): return run_dir(target, mask_from, mask_method) / "zap.fits"   # 扣天空後 cube
def sky_path(target, mask_from, mask_method="sep"): return run_dir(target, mask_from, mask_method) / "sky.fits"   # ZAP 扣掉的天空
def var_path(target, mask_from, mask_method="sep"): return run_dir(target, mask_from, mask_method) / "var.fits"   # 變異曲線（診斷）

def ensure_output_dirs():
    for d in (MASKS_DIR, CUBES_DIR):
        d.mkdir(parents=True, exist_ok=True)
def ensure_mask_dir(from_cube, method=MASK_METHOD):
    d = mask_dir(from_cube, method); d.mkdir(parents=True, exist_ok=True); return d
def ensure_run_dir(target, mask_from, mask_method="sep"):
    d = run_dir(target, mask_from, mask_method); d.mkdir(parents=True, exist_ok=True); return d

# --- diagnostics：per-cube 的比較 / 尚未拔擢的產物落點（不覆蓋正式 masks/）---
def diagnostics_dir(cube):
    return DIAGNOSTICS_DIR / cube
def ensure_diagnostics_dir(cube):
    d = diagnostics_dir(cube); d.mkdir(parents=True, exist_ok=True); return d
def mask_output_dir(from_cube, method, promoted):
    """遮罩產物落點：已拔擢→正式 masks/<method>_from-<cube>/；未拔擢→diagnostics/<cube>/。
       用途：NE 新驗證參數的 mask 先落 diagnostics，不覆蓋已上傳 Drive 的正式 mask。"""
    return mask_dir(from_cube, method) if promoted else diagnostics_dir(from_cube)

# ============ 物理參數 ============
# ⚠️ Haro11 一律用整張 499×559 全視場，不做空間裁切：CGM 暈延伸到 ~75″(≈375px)，
#    任何 box 裁切都會把暈截斷（暈是本專案的科學目標）。
PIXEL_SCALE_ARCSEC = 0.20    # arcsec/px（header CD1_1）

# z=0.0206
# Hα 6562.8 → 6698.0 Å
# [NII]6548 → 6682.9 Å、[NII]6583 → 6719.1 Å
HALPHA_LINE_WINDOW     = (6692.0, 6708.0)                       # Hα(+核心) 發射線窗
HALPHA_IMAGE_CONTINUUM = ((6605.0, 6645.0), (6760.0, 6795.0))  # 做窄帶影像用（寬、乾淨，避開 [NII]）
HALPHA_FLUX_CONTINUUM  = ((6660.0, 6688.0), (6730.0, 6758.0))  # 量 Hα 通量用（窄、貼近線）← 刻意與上不同
SKY_LINE_WAVELENGTHS   = (5577, 6300, 8400)                    # 三條主要天空線（看殘餘）

# 源遮罩偵測【方法 "claude" / 對照】robust-MAD 手寫門檻（Hα + 白光雙通道；預設用 sep，此法保留供比較）
CLAUDE_HALPHA_THRESHOLD_SIGMA = 2.0   # Hα 門檻（σ_MAD 倍數）
CLAUDE_WHITE_THRESHOLD_SIGMA  = 3.0   # 白光門檻（σ_MAD 倍數）
CLAUDE_MIN_BLOB_PIXELS        = 20    # 最小連通塊像素（丟掉雜訊碎點）

# 源遮罩偵測【方法 B / 預設】SEP + matched filter + 2σ（參數由資料尺度推得，見 CLAUDE.md 操作清單）
MASK_BACKGROUND_BOX_PIXELS = 256    # 背景框 > 延展暈，才不會把暈當背景吸走
MASK_KERNEL_FWHM_PIXELS    = 6.0    # matched filter 核 FWHM ≈ seeing (1.24"/0.2")
MASK_THRESHOLD_SIGMA       = 2.0    # 偵測門檻 2σ（假陽性 2.3%）
MASK_MIN_AREA_PIXELS       = 30     # 最小連通面積 ≈ 1 PSF
MASK_DILATE_PIXELS         = 6      # 事後膨脹 ≈ 1×seeing（安全邊界）；兩種方法共用

# ======================= 共用純計算 helper =======================
def wavelength_axis(header):
    """由 FITS header (CRVAL3/CRPIX3/CD3_3) 算出每個波長平面的波長 [Å]。"""
    return header["CRVAL3"] + (np.arange(header["NAXIS3"]) + 1 - header["CRPIX3"]) * header["CD3_3"]

def halpha_narrowband_image(cube, wavelengths, continuum=HALPHA_IMAGE_CONTINUUM):
    """線內平均 − 連續譜平均 → 純 Hα 窄帶影像（step1 建遮罩與 fig4 共用同一份定義）。"""
    in_line = (wavelengths > HALPHA_LINE_WINDOW[0]) & (wavelengths < HALPHA_LINE_WINDOW[1])
    in_continuum = np.zeros_like(wavelengths, bool)
    for lo, hi in continuum:
        in_continuum |= (wavelengths > lo) & (wavelengths < hi)
    return (np.nanmean(cube[in_line], 0) - np.nanmean(cube[in_continuum], 0)).astype(np.float32)
