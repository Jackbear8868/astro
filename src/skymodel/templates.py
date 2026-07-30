from pathlib import Path
import numpy as np
from astropy.io import fits
from scipy.interpolate import make_interp_spline

def load_sdss_template(path):
    """讀一條 SDSS spDR2 模板,回傳靜止波長上的三次 B-spline。

    模板為 log 波長等比取樣,在紅端比 MUSE 格點粗。線性內插的誤差隨 z
    週期性振盪,週期等於模板的取樣間距,會在 χ²(z) 上造成假的局部極大值。
    樣條只需在靜止波長上建一次,紅移只改變求值的位置。
    """
    with fits.open(path) as hdul:
        header   = hdul[0].header
        spectrum = hdul[0].data[0].astype(np.float64)

    spectrum[spectrum == 0] = np.nan
    
    lam_rest = 10.0 ** (header["COEFF0"] + header["COEFF1"] * np.arange(header["NAXIS1"]))
    good = np.isfinite(spectrum)
    return make_interp_spline(lam_rest[good], spectrum[good], k=3)

def redshift_to_grid(spline, z, lam_muse):
    """把模板紅移到 z、重採樣到 lam_muse。

    模板覆蓋不到的通道回傳 NaN。
    """
    return spline(lam_muse / (1.0 + z), extrapolate=False)
    
def template_on_grid(path, z, lam_muse):
    """讀檔 + 紅移 + 重採樣(單次使用的便利版)。"""
    return redshift_to_grid(load_sdss_template(path), z, lam_muse)
    
def air_to_vacuum(lam_air):
    """把空氣波長轉成真空波長(Morton 2000,IAU 標準)。

    MUSE cube 的 CTYPE3 = AWAV(空氣波長),SDSS 模板是真空波長。
    不轉換會造成約 83 km/s 的系統性紅移偏差。
    """
    s2 = (1e4 / lam_air) ** 2
    n = (1.0
         + 8.336624212083e-5
         + 2.408926869968e-2 / (130.1065924522 - s2)
         + 1.599740894897e-4 / (38.92568793293 - s2))
    return lam_air * n