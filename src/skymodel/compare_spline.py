from pathlib import Path
import numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline, LSQUnivariateSpline
from utils import detect_lines, running_median

def lsq_fit(spacing):
    t = xu[spacing // 2::spacing]
    return LSQUnivariateSpline(xu, yu, t[(t > xu[0]) & (t < xu[-1])], k=3, ext=3)

ROOT = Path(__file__).resolve().parents[2]
STEP01 = ROOT / "results/skymodel/step01"
WSKY = ROOT / "data/Haro11_NEpointing_wsky.fits"

mean_sky = np.load(STEP01 / "mean_wsky.npy")
line_mask = None

hdr = fits.getheader(WSKY, "DATA")
wl = hdr["CRVAL3"] + (np.arange(hdr["NAXIS3"]) + 1 - hdr["CRPIX3"]) * hdr["CD3_3"]

OUT = STEP01 / "spline_compare"
OUT.mkdir(exist_ok=True)

for _ in range(4):
    continuum, sigma, line_mask = detect_lines(mean_sky, exclude=line_mask, thresholds=(1, 2))

print(int(line_mask.sum()), "masked")

m = mean_sky.copy()
m[line_mask] = np.nan
raw_cont = running_median(m, 300)                      # 未修繕、有懸崖的版本（對照組）
x = np.arange(mean_sky.size)
unmasked_idx = np.isfinite(m) & np.isfinite(raw_cont)
xu, yu = x[unmasked_idx], raw_cont[unmasked_idx]

knots = xu[25::50]
spline_fits = {
    "Interpolated":      InterpolatedUnivariateSpline(xu, yu, k=3, ext=3),
    "Univariate with smoothing δ=0.05":  UnivariateSpline(xu, yu, k=3, s=len(xu) * 0.05**2, ext=3),
    "Univariate with smoothing δ=0.1":   UnivariateSpline(xu, yu, k=3, s=len(xu) * 0.1**2, ext=3),
    "LSQUnivariate with knots=30":           lsq_fit(30),
    "LSQUnivariate with knots=50":           lsq_fit(50),
}


for lo, hi, tag in [(wl[0], wl[-1], "full")]:
    plt.figure(figsize=(15, 5))
    plt.plot(wl, mean_sky, lw=0.5, color="gray", alpha=0.3, label="mean sky")
    plt.plot(wl, raw_cont, lw=0.8, color="black", label="raw running median")
    plt.plot(wl[unmasked_idx], yu, "o", ms=3, mfc="none", mec="black", alpha=0.5 ,mew=0.8, zorder=5, label="unmasked points")
    # plt.fill_between(wl, 0, 1, where=~unmasked_idx, transform=plt.gca().get_xaxis_transform(), color="0.93", zorder=0, label="masked region")
    for name, spl in spline_fits.items():
        plt.plot(wl, spl(x), lw=0.8, label=name)
    sel = (wl >= lo) & (wl <= hi)
    plt.xlim(lo, hi)
    plt.ylim(0, 40)
    plt.xlabel("wavelength [A]"); plt.ylabel("flux")
    plt.legend(fontsize=8)
    plt.savefig(OUT / f"spline_{tag}.png", dpi=300)
    plt.close()



for lo, hi, tag in [(5500, 6500, "6000ang"), (8500, 9300, "8900ang")]:
    plt.figure(figsize=(15, 5))
    plt.plot(wl, mean_sky, lw=0.5, color="gray", alpha=0.3, label="mean sky")
    plt.plot(wl, raw_cont, lw=0.8, color="black", label="raw running median")
    plt.plot(wl[unmasked_idx], yu, "o", ms=3, mfc="none", mec="black", alpha=0.5 ,mew=0.8, zorder=5, label="unmasked points")
    # plt.fill_between(wl, 0, 1, where=~unmasked_idx, transform=plt.gca().get_xaxis_transform(), color="0.93", zorder=0, label="masked region")
    for name, spl in spline_fits.items():
        plt.plot(wl, spl(x), lw=0.8, label=name)
    sel = (wl >= lo) & (wl <= hi)
    plt.xlim(lo, hi)
    plt.ylim(np.nanmin(raw_cont[sel]) - 5, np.nanmax(raw_cont[sel]) + 5)
    plt.xlabel("wavelength [A]"); plt.ylabel("flux")
    plt.legend(fontsize=8)
    plt.savefig(OUT / f"spline_{tag}.png", dpi=300)
    plt.close()



