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

def _eigen_spline(lam_rest, F):
    """把 (n_comp, n_wave) 的本徵譜做成「批次」spline。

    兩端的常數是填充,不是資料 —— 檔案把最後一個真值一直重複到邊界。
    建 spline 時包含進去,模板在那裡會變成一條假的水平線,擬合會很樂意
    用它去吸收殘差。所有成分同時停止變化的地方就是真實資料的邊界。

    y 傳入 (n_wave, n_comp),求值一次就回傳全部成分。spline 的成本主要
    在「二分搜尋找區間」,那一步和成分數無關,所以拿 4 條和拿 1 條一樣快,
    比逐條求值省下重複的搜尋。

    Returns
    -------
    BSpline
        求值後的形狀是 (n_out, n_comp);覆蓋不到的位置為 NaN。
    """
    d = np.abs(np.diff(F, axis=1)).max(axis=0)
    i = np.flatnonzero(d > 0)
    g = slice(i[0], i[-1] + 2)                  # +2:diff 少一個,且要含右端點
    return make_interp_spline(lam_rest[g], F[:, g].T, k=3)


def load_eigen_galaxy(path):
    """Bolton et al. 2012 的星系本徵譜(FITS bintable)→ 批次 spline。

    真實資料 1183 – 9840 A(靜止),4 條成分。chi2 那一欄裝的是未初始化的
    記憶體(1e-310 到 1e307),不要碰。同目錄的 .spec 是同一份資料的文字版,
    只保留 4 位小數 —— 高階成分會跨過零點,截斷後相對誤差發散,用 FITS。
    """
    d = fits.open(path)[1].data
    lam = np.asarray(d["wave"], np.float64)
    F   = np.vstack([np.asarray(d[f"flux{i}"], np.float64) for i in range(1, 5)])
    return _eigen_spline(lam, F)


def load_eigen_qso(path):
    """QSO 本徵譜(文字檔:第 1 欄波長,其餘為成分)→ 批次 spline。

    真實資料 605 – 8356 A(靜止),4 條成分。檔名寫 linear,但波長格點其實
    和星系檔、SDSS 模板一樣是等比的(dlog10 = 1e-4)。
    """
    q = np.loadtxt(path)
    return _eigen_spline(q[:, 0], q[:, 1:].T)


def redshift_to_grid(spline, z, lam_muse):
    """把模板紅移到 z、重採樣到 lam_muse。

    模板覆蓋不到的通道回傳 NaN。單一模板回傳 (nz,);本徵譜的批次 spline
    回傳 (nz, n_comp)。
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