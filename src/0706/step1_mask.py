"""
step1 — 建 source / segmentation mask（可從 nosky 或 wsky 建；預設 sep）。
  輸入： settings.input_cube_path(from_cube)          （from_cube ∈ {nosky, wsky}）
  輸出： results/zap/masks/<method>_from-<from_cube>/mask.fits   （1=源, 0=可用天空；餵給 step2）
         results/zap/masks/<method>_from-<from_cube>/blanks.npz  （亮源座標 sy,sx + 波長軸；供 M3）
         每個 (方法 × cube) 各自一個資料夾，彼此不覆蓋。
  跑法： conda run -n astro python src/0706/step1_mask.py nosky          # 從 nosky 建（預設 sep）
         conda run -n astro python src/0706/step1_mask.py wsky           # 從 wsky 建
         conda run -n astro python src/0706/step1_mask.py nosky claude   # 改用方法 claude

  兩種偵測方法（同一張 Hα 窄帶影像，差在怎麼定門檻）：
    「sep」   （預設）: SEP + 高斯 matched filter(核=seeing) + 正常 2σ；
                        sep.Background 自動排源估 RMS（噪音從無源區估，門檻不被暈撐失真）。
    「claude」（對照）: robust-MAD 手寫門檻。median_filter 平滑 → 全圖 MAD σ →
                        「Hα 或 白光」超過門檻 → 連通塊過濾 → 膨脹。用 Hα + 白光雙通道。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings
import numpy as np
from astropy.io import fits

def _gauss_kernel(sz, fwhm):
    """高斯 matched filter 核（FWHM ≈ seeing）。"""
    x = np.arange(sz) - sz // 2
    g = np.exp(-(x**2) / (2 * (fwhm / 2.355)**2))
    k = np.outer(g, g)
    return (k / k.sum()).astype(np.float32)

def _detect_sep(ha, white, valid):
    """方法 sep：SEP + matched filter + 2σ。回傳源遮罩 bool。"""
    import sep
    from scipy import ndimage as ndi
    invalid = ~valid
    bkg = sep.Background(np.ascontiguousarray(ha), mask=invalid,
                         bw=settings.MASK_BACKGROUND_BOX_PIXELS, bh=settings.MASK_BACKGROUND_BOX_PIXELS, fw=3, fh=3)
    _, seg = sep.extract(ha - bkg, settings.MASK_THRESHOLD_SIGMA, err=bkg.rms(), mask=invalid,
                         minarea=settings.MASK_MIN_AREA_PIXELS, filter_kernel=_gauss_kernel(15, settings.MASK_KERNEL_FWHM_PIXELS),
                         deblend_nthresh=32, deblend_cont=0.005, segmentation_map=True)
    return ndi.binary_dilation((seg > 0) & valid, iterations=settings.MASK_DILATE_PIXELS) & valid

def _detect_claude(ha, white, valid):
    """方法 claude：robust-MAD 手寫門檻（Hα 或 白光）+ 連通塊過濾 + 膨脹。回傳源遮罩 bool。"""
    from scipy import ndimage as ndi
    def med_mad(z):                                 # 穩健 (median, σ_MAD)
        v = z[valid]; m = np.median(v); return m, 1.4826 * np.median(np.abs(v - m))
    ha_s = ndi.median_filter(ha, 3); wh_s = ndi.median_filter(white, 3)   # 先 3x3 中值壓單點雜訊
    mh, sh = med_mad(ha_s); mw, sw = med_mad(wh_s)
    src = (((ha_s - mh) > settings.CLAUDE_HALPHA_THRESHOLD_SIGMA * sh) |
           ((wh_s - mw) > settings.CLAUDE_WHITE_THRESHOLD_SIGMA * sw)) & valid
    lab, n = ndi.label(src)                         # 只留夠大的連通塊，丟掉雜訊碎點
    if n:
        big = np.zeros(n + 1, bool)
        big[1:] = ndi.sum(np.ones_like(lab), lab, range(1, n + 1)) >= settings.CLAUDE_MIN_BLOB_PIXELS
        src = big[lab]
    return ndi.binary_dilation(src, iterations=settings.MASK_DILATE_PIXELS) & valid

_DETECTORS = {"sep": _detect_sep, "claude": _detect_claude}

def build_mask(from_cube, method="sep"):
    """在指定 cube 上、用指定方法偵測源，回傳 (src, white, ha, valid, wl)。不存檔。
       from_cube ∈ {'nosky','wsky'}；method ∈ {'sep','claude'}。"""
    hd = fits.open(str(settings.input_cube_path(from_cube)))
    cube = hd["DATA"].data; hdr = hd["DATA"].header
    wl = settings.wavelength_axis(hdr)
    white = np.nansum(cube, axis=0).astype(np.float32)     # 白光：定位最亮點、判斷有效視場
    ha = settings.halpha_narrowband_image(cube, wl)        # 純 Hα 影像（共用 settings 定義）
    hd.close()
    valid = white != 0
    src = _DETECTORS[method](ha, white, valid)
    return src, white, ha, valid, wl

def mask(from_cube, method=settings.MASK_METHOD):
    """從 from_cube 偵測源 → 存 source mask + 亮源座標到 <method>_from-<from_cube>/。
       不再取樣 blank（M1/評估直接讀 mask.fits 用全部 valid blank；blanks.npz 只留亮源座標供 M3）。"""
    settings.ensure_output_dirs()
    src, white, ha, valid, wl = build_mask(from_cube, method)
    settings.ensure_mask_dir(from_cube, method)
    out_mask = settings.source_mask_path(from_cube, method)
    fits.writeto(str(out_mask), src.astype(np.uint8), overwrite=True)

    blank = valid & ~src
    sy, sx = np.unravel_index(np.nanargmax(np.where(valid, white, np.nan)), white.shape)
    np.savez(str(settings.blanks_path(from_cube, method)), wl=wl, sy=sy, sx=sx)   # 亮源座標(供 M3)+波長軸
    print(f"[step1 mask/{method}/from-{from_cube}] 源 {int(src.sum())} ({100*src.sum()/valid.sum():.0f}%), "
          f"blank {int(blank.sum())}（全部用於評估）, 亮源@({sy},{sx}) -> {out_mask.parent}")

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in settings.CUBE_NAMES:
        sys.exit("用法: step1_mask.py <nosky|wsky> [sep|claude]")
    from_cube = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else settings.MASK_METHOD
    if method not in _DETECTORS:
        sys.exit(f"未知方法 {method}；可用: {tuple(_DETECTORS)}")
    mask(from_cube, method)
