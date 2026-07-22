from pathlib import Path
import numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.stats import skew, kurtosis
from scipy.optimize import nnls
from sklearn.cluster import KMeans
import time

from utils import estimate_continuum

def semi_NMF(X, basis, n_iter=15):
    """Semi-NMF: X ≈ W @ B, with B >= 0, W can be negative."""
    X_real = np.nan_to_num(X)
    B = basis.copy()
    for it in range(n_iter):
        W_real = X_real @ B.T @ np.linalg.pinv(B @ B.T)   # ① W-step:B 變了,重解最優的 W
        WtW = W_real.T @ W_real                # ② 準備 1:(K,K),大矩陣資訊壓縮進來
        WtX = W_real.T @ X_real                # ② 準備 2:(K,nz)
        Lc = np.linalg.cholesky(WtW)           # ② 開平方根:WtW = Lc @ Lc.T
        Y = np.linalg.solve(Lc, WtX)           # ② 一次解出全部 nz 個小 y
        for j in range(B.shape[1]):
            B[:, j], _ = nnls(Lc.T, Y[:, j])   # ② 迷你 nnls:等價於原本的大 nnls
    return W_real, B


def semi_NMF_mu(X, K, n_iter=300, eps=1e-9):
    """Ding-Li-Jordan (2010) 原版 semi-NMF:k-means 初始化 + 乘法更新。
        X: (n_samples, nz);回傳 W (n_samples,K), B (K,nz)。"""
    X = np.nan_to_num(X)

    # --- 初始化(論文 §2):對波長通道做 k-means ---
    km = KMeans(n_clusters=K, n_init=4, random_state=0).fit(X.T)
    G = np.zeros((X.shape[1], K), dtype=X.dtype)
    G[np.arange(X.shape[1]), km.labels_] = 1     # 指示矩陣:通道 i 屬於群 k → G[i,k]=1
    G += 0.2                                     # 論文原文:全體加 0.2,嚴格正出發

    for it in range(n_iter):
        # --- F-step:閉式解(跟你的 W-step 同一條公式) ---
        W = X @ G @ np.linalg.pinv(G.T @ G)

        # --- G-step:乘法更新(論文式 (8)) ---
        XtF = X.T @ W                            # (nz, K)
        FtF = W.T @ W                            # (K, K)
        XtF_p = (np.abs(XtF) + XtF) / 2          # A⁺:正的部分
        XtF_n = (np.abs(XtF) - XtF) / 2          # A⁻:負的部分(取成正值)
        FtF_p = (np.abs(FtF) + FtF) / 2
        FtF_n = (np.abs(FtF) - FtF) / 2
        G *= np.sqrt((XtF_p + G @ FtF_n) / (XtF_n + G @ FtF_p + eps))

        if (it + 1) % 100 == 0:
            print(f"  MU iter {it+1}/{n_iter}", flush=True)

    return W, G.T


def spectrum_stats(spec):
    """把一條光譜濃縮成摘要統計。"""
    spec = spec[np.isfinite(spec)]                     # 先丟掉 NaN
    return {
        "mean":          np.mean(spec),
        "sigma":         np.std(spec),
        "skewness":      skew(spec),
        "kurtosis":      kurtosis(spec),
        "rms_from_zero": np.sqrt(np.mean(spec**2)),    # sqrt(平方的平均) = 離 0 的均方根
    }


def plot_compare(wl, spec, spec_compare, out_path, label="ours", label_compare="nosky", ylim=(-20, 20), title=None):
      """對照圖：左光譜（藍=spec、橘虛線=spec_compare），右兩組 stats。"""
      fig, (ax, stat_ax) = plt.subplots(1, 2, figsize=(15.5, 4.5), gridspec_kw={"width_ratios": [5, 1]})
      ax.axhline(0, color="0.5", lw=0.5)
      ax.plot(wl, spec, lw=0.9, color="#1f77b4", label=label)
      ax.plot(wl, spec_compare, lw=0.9, ls="--", alpha=0.7, color="#e8710a", label=label_compare)
      ax.set_ylim(*ylim)
      ax.set_xlabel("wavelength [A]"); ax.set_ylabel("flux")
      if title:
          ax.set_title(title)
      ax.legend(fontsize=8)

      stat_ax.axis("off")
      def fmt(name, s):
          st = spectrum_stats(s)
          return f"[{name}]\n" + "\n".join(f"{k:<13} = {v:.4g}" for k, v in st.items())
      stat_ax.text(0, 0.95, fmt(label, spec), color="#1f77b4", va="top", family="monospace", fontsize=8, transform=stat_ax.transAxes)
      stat_ax.text(0, 0.45, fmt(label_compare, spec_compare), color="#e8710a", va="top", family="monospace", fontsize=8, transform=stat_ax.transAxes)

      fig.tight_layout()
      fig.savefig(out_path, dpi=145)
      plt.close()


ROOT = Path(__file__).resolve().parents[2]
STEP01 = ROOT / "results/skymodel/step01"
WSKY = ROOT / "data/Haro11_NEpointing_wsky.fits"
NOSKY = ROOT / "data/Haro11_NEpointing_esonosky.fits"
WHITELIGHT = STEP01 / "whitelight.fits"
SEG = STEP01 / "seg.fits"

white_mask = fits.getdata(WHITELIGHT)
seg_mask   = fits.getdata(SEG)
valid_mask = white_mask != 0
blank_mask = valid_mask & ~((seg_mask > 0) & valid_mask)

hdr = fits.getheader(WSKY, "DATA")
wsky = fits.getdata(WSKY, "DATA")
nosky = fits.getdata(NOSKY, "DATA")

mean_nosky = np.nanmean(nosky[:, blank_mask], axis=1)   # nosky 的 blank 平均光譜

nz = hdr["NAXIS3"]
wl = hdr["CRVAL3"] + (np.arange(hdr["NAXIS3"]) + 1 - hdr["CRPIX3"]) * hdr["CD3_3"]
blank_spectra = wsky[:, blank_mask]  # 2D (nz, n_blank)
mean_sky = np.nanmean(blank_spectra, axis=1)

start_time = time.time()

thresholds = (1, 2)
N_iterations = 20
LINE_MASK_DIR = STEP01 / "line_masks"
LINE_MASK_DIR.mkdir(parents=True, exist_ok=True)

continuum, sigma, line_mask, history = estimate_continuum(mean_sky, thresholds=thresholds, window=300, max_iter=N_iterations)

for i, (plt_continuum, plt_sigma, plt_line_mask) in enumerate(history):
    # --- Draw a chart for each iteration ---
    plt.figure(figsize=(12, 4))
    plt.plot(wl, mean_sky, lw=0.5, color="gray", label="mean sky")
    plt.plot(wl, plt_continuum, lw=0.5, color="blue", label="continuum")
    plt.plot(wl, plt_continuum + thresholds[0]*plt_sigma, lw=0.5, color="red", label=f"positive threshold ({thresholds[0]}σ)")
    plt.plot(wl, plt_continuum - thresholds[-1]*plt_sigma, lw=0.5, color="purple", label=f"negative threshold ({thresholds[-1]}σ)")
    plt.fill_between(wl, 0, mean_sky.max(), where=plt_line_mask, color="orange", alpha=0.2, label="detected lines")
    plt.ylim(0, 100)
    plt.xlabel("wavelength [A]"); plt.ylabel("flux")
    plt.title(f"iteration {i+1}: {int(plt_line_mask.sum())} lines")
    plt.legend()
    plt.savefig(LINE_MASK_DIR / f"line_mask_iter{i+1}.png", dpi=240)
    plt.close()

    plot_sky = mean_sky.copy()
    plot_sky[plt_line_mask] = np.nan       # 被遮的 bins → NaN → 線在那裡自動斷開留白
    # --- Draw a chart for non maked region ---
    plt.figure(figsize=(12, 4))
    plt.plot(wl, mean_sky, lw=0.5, color="gray", alpha=0.3 ,label="mean sky")
    plt.plot(wl, plot_sky, lw=0.5, color="red", label="mean sky (unmasked)")
    plt.ylim(0, 100)
    plt.xlabel("wavelength [A]"); plt.ylabel("flux")
    plt.title(f"iteration {i+1}: {int(nz - plt_line_mask.sum())} non masked lines")
    plt.legend()
    plt.savefig(LINE_MASK_DIR / f"non_line_mask_iter{i+1}.png", dpi=240)
    plt.close()

end_time = time.time()
print(f"Generate sky continuum took {end_time - start_time:.2f} seconds")
start_time = time.time()

# Use NMF to model the sky lines
L = blank_spectra - continuum[:, None]

X = np.nan_to_num(np.clip(L.T, 0, None))
K = 10
model = NMF(n_components=K, init="nndsvda", max_iter=300)
W = model.fit_transform(X)                       # (n_blank, K)：這些格子的振幅
basis = model.components_                            # (K, nz)：K 條基底模板
print("basis shape:", basis.shape)

end_time = time.time()
print(f"Generate sky line spectrum with NMF took {end_time - start_time:.2f} seconds")

sky_line = (W @ basis).T
sky = continuum[:, None] + sky_line
subtracted = blank_spectra - sky
mean_after  = np.nanmean(subtracted, axis=1)           
stats = spectrum_stats(mean_after)
print("Spectrum status with NMF: ",stats)
plot_compare(wl, mean_after, mean_nosky, STEP01 / "sky_subtracted_NMF.png")

# # Use semi-NMF to model the sky lines
# L = blank_spectra - continuum[:, None]
# start_time = time.time()
# X_real = np.nan_to_num(L.T)
# W_real, B = semi_NMF(X_real, basis, n_iter=15)

# end_time = time.time()
# print(f"Generate sky line spectrum with semi-NMF took {end_time - start_time:.2f} seconds")

# sky_line = (W_real @ B).T
# sky = continuum[:, None] + sky_line
# subtracted = blank_spectra - sky
# mean_after = np.nanmean(subtracted, axis=1)
# print(spectrum_stats(mean_after))
# plot_compare(wl, mean_after, mean_nosky, STEP01 / "sky_subtracted_3.png")

# print("line channels:    ", spectrum_stats(mean_after[line_mask]))
# print("line-free channels:", spectrum_stats(mean_after[~line_mask]))

# Use semi-NMF(official algorithm) to model the sky lines
L = blank_spectra - continuum[:, None]
start_time = time.time()
W_mu, B_mu = semi_NMF_mu(np.nan_to_num(L.T).astype(np.float32), K=10, n_iter=10)
end_time = time.time()
print(f"semi-NMF (paper: kmeans+MU) took {end_time - start_time:.2f} s")

sky_line = (W_mu @ B_mu).T
sky = continuum[:, None] + sky_line
subtracted = blank_spectra - sky
mean_after_mu = np.nanmean(subtracted, axis=1)
print("Spectrum status with semi-NMF: ",stats)
print("line:     ", spectrum_stats(mean_after_mu[line_mask]))
print("line-free:", spectrum_stats(mean_after_mu[~line_mask]))
plot_compare(wl, mean_after_mu, mean_nosky, STEP01 / "sky_subtracted_semi_NMF.png")