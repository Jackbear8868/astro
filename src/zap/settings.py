"""
src/zap pipeline 的【唯一設定檔】：所有路徑 + 物理參數只在這裡定義一次。
zap.py / mask.py / eval_spectrum.py 全部 import 這裡；改參數只改這一份，不會分岔。

※ 所有相對路徑都相對「專案根目錄」，請一律從根目錄執行，例如：
    conda run -n astro python src/zap/zap.py wsky nosky --ncpu 16
"""
from pathlib import Path
import numpy as np

# per-cube 偵測設定（CubeConfig / DetectParams + lazy registry）集中在 cube_config.py。
# 這裡 re-export，讓其他程式只要 import settings 就拿得到（不製造第二處真相來源）。
from cube_config import CubeConfig, DetectParams, get_cube_config  # noqa: F401

# ======================= 輸出路徑（2×2 結構）=======================
OUTPUT_ROOT = Path("results/zap")          # 所有產物的根目錄
MASKS_DIR   = OUTPUT_ROOT / "masks"        # mask 產物：source mask（小）
CUBES_DIR   = OUTPUT_ROOT / "cubes"        # zap 產物：每個實驗一個子資料夾（大）
DIAGNOSTICS_DIR = OUTPUT_ROOT / "diagnostics"   # 尚未拔擢 / 比較用的產物（不覆蓋正式 masks/）
FIGURE_DIR  = OUTPUT_ROOT                  # .png 圖存這裡

MASK_METHOD = "sep"                        # 預設方法（SEP + matched filter + 2σ）

# ======================= 輸入 cube（唯讀，永不刪）=======================
RAW_CUBE_PATHS = {"nosky": Path("data/Haro11_nosky.fits"),
                  "wsky":  Path("data/Haro11_wsky.fits"),
                  "NEwsky":  Path("data/Haro11_NEpointing_wsky.fits"),      # NE pointing, 含天空
                  "NEnosky": Path("data/Haro11_NEpointing_esonosky.fits")}  # NE pointing, ESO 已扣天空
CUBE_NAMES = ("nosky", "wsky", "NEwsky", "NEnosky")
def input_cube_path(cube_name):
    """要處理的原始 cube（整張全視場，不空間裁切）。"""
    return RAW_CUBE_PATHS[cube_name]

# --- mask 產物：每個 (方法 × cube) 一個資料夾，把該遮罩的產物包在一起、彼此不覆蓋 ---
def mask_dir(from_cube, method=MASK_METHOD):
    return MASKS_DIR / f"{method}_from-{from_cube}"            # 例：masks/sep_from-nosky/
def source_mask_path(from_cube, method=MASK_METHOD):
    return mask_dir(from_cube, method) / "mask.fits"           # 2D uint8：1=源, 0=可用天空

# --- zap 產物：依「ZAP 對象 × mask(方法 × 來源) × cfwidthSP」命名，每個實驗一個資料夾 ---
def _run_suffix(mask_from, mask_method):
    return mask_from if mask_method == "sep" else f"{mask_method}-{mask_from}"     # sep 隱含（相容既有 run）；其他方法才加前綴
def _cfwsp_suffix(cfwidthsp):
    return "" if cfwidthsp == 300 else f"_cfwsp{cfwidthsp}"                        # ZAP 預設 300 → 不加後綴（相容既有 run）；非預設才標出來
def run_dir(target, mask_from, mask_method="sep", cfwidthsp=300):
    return CUBES_DIR / f"{target}_maskfrom-{_run_suffix(mask_from, mask_method)}{_cfwsp_suffix(cfwidthsp)}"  # 例：NEwsky_maskfrom-seg2sigma_brq-NEnosky_cfwsp20
def zap_path(target, mask_from, mask_method="sep", cfwidthsp=300): return run_dir(target, mask_from, mask_method, cfwidthsp) / "zap.fits"   # 扣天空後 cube
def sky_path(target, mask_from, mask_method="sep", cfwidthsp=300): return run_dir(target, mask_from, mask_method, cfwidthsp) / "sky.fits"   # ZAP 扣掉的天空
def var_path(target, mask_from, mask_method="sep", cfwidthsp=300): return run_dir(target, mask_from, mask_method, cfwidthsp) / "var.fits"   # 變異曲線（診斷）

def ensure_output_dirs():
    for d in (MASKS_DIR, CUBES_DIR):
        d.mkdir(parents=True, exist_ok=True)
def ensure_mask_dir(from_cube, method=MASK_METHOD):
    d = mask_dir(from_cube, method); d.mkdir(parents=True, exist_ok=True); return d
def ensure_run_dir(target, mask_from, mask_method="sep", cfwidthsp=300):
    d = run_dir(target, mask_from, mask_method, cfwidthsp); d.mkdir(parents=True, exist_ok=True); return d

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

# ======================= 共用純計算 helper =======================
def wavelength_axis(header):
    """由 FITS header (CRVAL3/CRPIX3/CD3_3) 算出每個波長平面的波長 [Å]。"""
    return header["CRVAL3"] + (np.arange(header["NAXIS3"]) + 1 - header["CRPIX3"]) * header["CD3_3"]
