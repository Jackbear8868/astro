"""SDSS spDR2 模板的載入器。

這些模板已經不在流程裡 —— step4 的恆星候選出自 `data/stellar_templates`
(見 `docs/stellar-templates.md`),星系側走 Bolton 2012 本徵譜。留在這裡是因為
本目錄下幾支診斷程式是在 SDSS 模板的年代寫的,而它們記錄的是當時的結果。

資料本身不在版控內,取得方式見 `docs/archive/sdss-templates.md`。
"""
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.interpolate import make_interp_spline

TPL_DIR = Path(__file__).resolve().parents[3] / "data/sdss_templates"
SDSS_DIR = TPL_DIR      # 同一個目錄,兩個名字都有程式在用


def load_sdss_template(path):
    """讀一條 SDSS spDR2 模板,回傳靜止波長上的三次 B-spline。

    模板在 log 波長上等距取樣,紅端比 MUSE 的格點還粗。線性內插的誤差會隨 z
    以模板的取樣間隔週期振盪,在 chi2(z) 上造出假的局部極小。改成先在靜止波長
    建一次 spline,紅移就只是換評估位置,不再引入這種週期性誤差。
    """
    with fits.open(path) as hdul:
        header   = hdul[0].header
        spectrum = hdul[0].data[0].astype(np.float64)

    spectrum[spectrum == 0] = np.nan

    lam_rest = 10.0 ** (header["COEFF0"] + header["COEFF1"] * np.arange(header["NAXIS1"]))
    good = np.isfinite(spectrum)
    return make_interp_spline(lam_rest[good], spectrum[good], k=3)
