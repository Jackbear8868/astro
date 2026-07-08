"""
每個 cube 一組偵測參數的設定物件（回應 docs/progress/refactor-proposal.md 第三節）。

設計要點（見 CLAUDE.md 原則 2「參數要能物理辯護」）：
  * `DetectParams` 是「旋鈕」；`CubeConfig` 綁一顆 cube 的場大小 / seeing / 參數。
  * `from_header()`「能從資料推的就推」：
      - pixel scale ← header CD1_1
      - 場大小      ← header NAXIS1/NAXIS2
      - **PSF FWHM 由 cube 內恆星量測**（psf 方法），量不到才退回 header 代理
        （AG FWHM / AMBI FWHM）；`ESO QC EXPCOMB FWHM MEDIAN` 在本批檔案是 0.0（空），
        絕不使用，且會示警。量到的 FWHM 必 >0（assert）。
      - kernel_fwhm ≈ PSF、min_area ≈ π(FWHM/2)²、dilate ≈ round(PSF)。
      - bw 規則：小場用局部盒 bw≈場/5（去 IFU/天空殘餘的局部 offset），
        但夾在 256 以內（大場的暈 Ø≈226px 需要 256 才不會把暈當背景吸走）。
  * registry 用 **lazy**（get_cube_config + lru_cache）：只有真的要用某顆 cube 時
    才開檔量 PSF，import settings 不會去開 3–8GB 的 cube。

⚠️ 護欄：主場 nosky/wsky 已有既算好的遮罩與下游 ZAP 結果，這裡以「明確 override」
   把它們釘死在舊值（bw=256, kernel 6, minarea 30, dilate 6, 不用 STAT），保證可重現；
   只有 NE 指向拿新驗證過的參數（bw≈64, kernel≈4, STAT 逐像素噪音）。
"""
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache
import warnings
import numpy as np
from astropy.io import fits

# 本批 MUSE WFM cube 皆 0.2"/px；實際仍以各 cube 的 header CD1_1 為準（from_header 會讀）。
DEFAULT_PIXEL_SCALE_ARCSEC = 0.20


@dataclass(frozen=True)
class DetectParams:
    """SEP 偵測旋鈕。預設值鎖在 CLAUDE.md 的物理規則，實際值每 cube 由 from_header 推。"""
    bkg_box_px:      int              # 背景框；> 最大物體，或小場用局部盒去 offset
    kernel_fwhm_px:  float            # matched filter 核 FWHM ≈ seeing/PSF
    threshold_sigma: float = 2.0      # ≥2σ，永不下探（原則 2）
    min_area_px:     int   = 30       # ≈1 PSF 面積
    dilate_px:       int   = 6        # ≈1×seeing 安全邊界
    use_stat_noise:  bool  = False    # True → 用 cube STAT 延伸當 per-pixel err map
    stat_calib_region: tuple = None   # 量 k 用的 blank 區：(x0,x1) 或 (x0,x1,y0,y1)；None→全場 sigma-clip
    stat_calib_k:    float = None     # 明確給 k 就用它；None → 每 cube 量測（建議）


@dataclass(frozen=True)
class CubeConfig:
    """一顆 cube 的完整偵測設定（含出處註記）。"""
    name:      str
    path:      Path
    ny:        int                    # header NAXIS2
    nx:        int                    # header NAXIS1
    pixel_scale_arcsec: float         # header CD1_1（絕對值×3600）
    seeing_fwhm_px:     float         # 量到/推得的 PSF FWHM（px）
    seeing_source:      str           # 出處：'star' / 'header:AG' / 'header:AMBI' / 'constant'
    reference: tuple                  # (standard=已扣天空, sky=含天空) 供 M1
    detect:    DetectParams
    promoted:  bool = True            # True→寫進正式 masks/；False→寫 diagnostics/（尚未拔擢）

    @property
    def seeing_fwhm_arcsec(self) -> float:
        return self.seeing_fwhm_px * self.pixel_scale_arcsec

    @classmethod
    def from_header(cls, name, path, reference, *,
                    measure_psf=True,          # 開 cube 由恆星量 PSF
                    seeing_fwhm_arcsec=None,    # 量不到時的明確退路常數
                    bkg_box_px=None,            # 明確 override bw；否則走場/5 規則
                    kernel_fwhm_px=None, min_area_px=None, dilate_px=None,
                    threshold_sigma=2.0,
                    use_stat_noise=False, stat_calib_region=None, stat_calib_k=None,
                    promoted=True):
        path = Path(path)
        hdr = fits.getheader(str(path), "DATA")
        ny, nx = int(hdr["NAXIS2"]), int(hdr["NAXIS1"])
        pix = (round(abs(float(hdr["CD1_1"])) * 3600.0, 4)
               if "CD1_1" in hdr else DEFAULT_PIXEL_SCALE_ARCSEC)

        # ---- seeing / PSF FWHM：可靠度排序 恆星 > header 代理 > 明確常數 ----
        fwhm_px, src = _resolve_seeing_px(path, pix, measure_psf, seeing_fwhm_arcsec, name)
        assert fwhm_px is not None and fwhm_px > 0, \
            f"[{name}] 量到的 PSF FWHM 必須 >0（絕不接受 QC 空關鍵字給的 0px 核）"

        # ---- 由 PSF 推偵測旋鈕（可被明確 override 蓋掉，供護欄釘死主場）----
        if kernel_fwhm_px is None:
            kernel_fwhm_px = round(fwhm_px, 1)
        if min_area_px is None:
            min_area_px = max(5, int(round(np.pi * (fwhm_px / 2.0) ** 2)))   # ≈1 PSF 面積
        if dilate_px is None:
            dilate_px = int(round(fwhm_px))

        # ---- bw 規則：小場用局部盒 bw≈min(場)/5，夾在 256 以內 ----
        #   主場 499×559：暈 Ø≈226px，需要 256 才不會把暈當背景吸走 → 用明確 override 釘 256。
        #   NE 320×332：256≈全域，除不掉左側 IFU 殘條；場/5=64 才能去局部 offset
        #   （bw≥96 中央空區會爆假陽性，已由 sweep 驗證）。
        if bkg_box_px is None:
            bkg_box_px = min(256, int(round(min(ny, nx) / 5.0)))

        det = DetectParams(bkg_box_px=int(bkg_box_px), kernel_fwhm_px=float(kernel_fwhm_px),
                           threshold_sigma=float(threshold_sigma), min_area_px=int(min_area_px),
                           dilate_px=int(dilate_px), use_stat_noise=bool(use_stat_noise),
                           stat_calib_region=stat_calib_region, stat_calib_k=stat_calib_k)
        return cls(name, path, ny, nx, pix, round(float(fwhm_px), 2), src,
                   tuple(reference), det, bool(promoted))


# ============ PSF 量測（沿用 scratchpad/psf.py 驗證過的方法）============
def measure_psf_fwhm_px(path):
    """由 cube 內緻密、圓、亮的恆星量 Gaussian-equivalent FWHM（px）。量不到回傳 None。

    白光底圖刻意取一段乾淨、無發射線也無主要天空線的連續譜（6600–6640 & 6760–6795 Å，
    對 Haro11 z=0.0206 避開 Hα/[NII]/天空線）→ SEP 抽源 → 用二階矩換 FWHM →
    只留 fwhm<8px、圓度>0.6、亮度前 30% 的點源（排除延展暈）。"""
    import settings  # 延遲載入避免循環相依
    import sep
    with fits.open(str(path), memmap=True) as h:
        wl = settings.wavelength_axis(h["DATA"].header)
        band = ((wl > 6600) & (wl < 6640)) | ((wl > 6760) & (wl < 6795))
        img = np.nanmean(h["DATA"].data[band].astype(np.float32), 0).astype(np.float32)
    valid = np.isfinite(img) & (img != 0)
    img = np.where(valid, img, 0).astype(np.float32)
    bkg = sep.Background(np.ascontiguousarray(img), mask=~valid, bw=64, bh=64, fw=3, fh=3)
    obj = sep.extract(img - bkg, 5.0, err=bkg.rms(), mask=~valid, minarea=5, deblend_cont=0.005)
    if len(obj) == 0:
        return None
    a, b, flux = obj["a"], obj["b"], obj["flux"]
    fwhm = 2.3548 * np.sqrt(a * b)                       # Gaussian-equivalent FWHM (px)
    roundness = b / np.maximum(a, 1e-6)
    star = (fwhm < 8) & (roundness > 0.6) & (flux > np.nanpercentile(flux, 70))
    if star.sum() == 0:
        return None
    return float(np.median(fwhm[star]))


def _header_seeing_arcsec(path):
    """header seeing 代理（arcsec）：先 AG FWHM，再 AMBI FWHM；回傳 (value, source) 或 (None, None)。
    絕不用 ESO QC EXPCOMB FWHM MEDIAN（本批=0.0 空值）。"""
    h = fits.getheader(str(path), 0)
    qc = h.get("ESO QC EXPCOMB FWHM MEDIAN", 0.0)
    if not qc:
        warnings.warn(f"[{Path(path).name}] header 'ESO QC EXPCOMB FWHM MEDIAN'=0（空）不可用；"
                      f"改用恆星量測或 AG/AMBI 代理。")
    agx, agy = h.get("ESO OCS SGS AG FWHMX MED"), h.get("ESO OCS SGS AG FWHMY MED")
    if agx and agy:
        return 0.5 * (float(agx) + float(agy)), "header:AG"
    amb = h.get("ESO TEL AMBI FWHM")
    if not amb:
        s, e = h.get("ESO TEL AMBI FWHM START"), h.get("ESO TEL AMBI FWHM END")
        if s and e:
            amb = 0.5 * (float(s) + float(e))
    if amb:
        return float(amb), "header:AMBI"
    return None, None


def _resolve_seeing_px(path, pix, measure_psf, seeing_fwhm_arcsec, name):
    """依可靠度回傳 (fwhm_px, source_str)。"""
    if measure_psf:
        fwhm = measure_psf_fwhm_px(path)
        if fwhm and fwhm > 0:
            return fwhm, "star"
        warnings.warn(f"[{name}] cube 內找不到可用恆星量 PSF，退回 header/常數。")
    arcsec, src = _header_seeing_arcsec(path)
    if arcsec and arcsec > 0:
        return arcsec / pix, src
    if seeing_fwhm_arcsec and seeing_fwhm_arcsec > 0:
        return seeing_fwhm_arcsec / pix, "constant"
    return None, None


# ============ per-cube registry（lazy：用到才開檔量 PSF）============
# 主場 nosky/wsky：明確 override 釘死舊值（護欄，逐位元可重現，不量 PSF、不用 STAT）。
# NE 指向：measure_psf 由恆星量 ≈4px → kernel≈4/minarea≈13/dilate≈4；bw=64；STAT 逐像素噪音。
CUBE_SPECS = {
    "nosky": dict(
        name="nosky", path="data/Haro11_nosky.fits", reference=("nosky", "wsky"),
        measure_psf=False, seeing_fwhm_arcsec=1.24,           # 假定常數（header QC 空）
        bkg_box_px=256, kernel_fwhm_px=6.0, min_area_px=30, dilate_px=6,
        use_stat_noise=False, promoted=True),
    "wsky": dict(
        name="wsky", path="data/Haro11_wsky.fits", reference=("nosky", "wsky"),
        measure_psf=False, seeing_fwhm_arcsec=1.24,
        bkg_box_px=256, kernel_fwhm_px=6.0, min_area_px=30, dilate_px=6,
        use_stat_noise=False, promoted=True),
    "NEnosky": dict(
        name="NEnosky", path="data/Haro11_NEpointing_esonosky.fits",
        reference=("NEnosky", "NEwsky"),
        measure_psf=True, seeing_fwhm_arcsec=0.886,           # 退路：AG FWHM
        bkg_box_px=64, use_stat_noise=True, stat_calib_region=(110, 190),
        promoted=False),   # 新驗證參數，尚未拔擢 → 寫 diagnostics/，不覆蓋已上傳的舊 mask
    "NEwsky": dict(
        name="NEwsky", path="data/Haro11_NEpointing_wsky.fits",
        reference=("NEnosky", "NEwsky"),
        measure_psf=True, seeing_fwhm_arcsec=0.886,
        bkg_box_px=64, use_stat_noise=True, stat_calib_region=(110, 190),
        promoted=False),
}


@lru_cache(maxsize=None)
def get_cube_config(name):
    """取得某 cube 的 CubeConfig（lazy 建、快取）。第一次呼叫才可能開 cube 量 PSF。"""
    if name not in CUBE_SPECS:
        raise KeyError(f"未知 cube {name}；可用：{tuple(CUBE_SPECS)}")
    return CubeConfig.from_header(**CUBE_SPECS[name])
