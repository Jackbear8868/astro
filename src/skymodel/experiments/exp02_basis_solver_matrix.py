"""ZAP 邏輯的參考版本:逐 spaxel 連續譜 + 兩種 basis x 四種解法的完整對照。

範圍:只在 blank region 上做,完全不碰 source 區。
所有 blank spaxel 都納入(不做訓練/評測切分),與 test.py 的做法一致。

完整的全因子設計:2 種連續譜 x 2 種 basis x 4 種解法 = 16 組。
沒有「特別待遇的基準線」—— 現行 test.py 的兩條路徑本身就是矩陣裡的兩格。

--- 2 種連續譜處理 ---
  shared   所有 spaxel 共用一條 mean sky 連續譜   <- 現行 test.py 的做法
  own      每條 spaxel 自己的連續譜                <- ZAP 的做法

--- 2 種 basis ---
  NMF      兩個矩陣都非負,所以「必須」把負值 clip 成 0  <- clip 會造成系統性正偏移
  PCA      都可正可負,不需要 clip                        <- ZAP 真正用的

--- 4 種解振幅的方式(都對每條 spaxel 各自解)---
  unw+nn     無權重 + 非負 (NNLS)
  chi2+nn    chi^2(1/STAT 加權) + 非負          <- 教授的加權 + 我們的非負
  unw+free   無權重 + 無約束
  chi2+free  chi^2(1/STAT 加權) + 無約束        <- 教授的方式(字面上)

--- 現行 test.py 的兩條路徑在矩陣裡的位置 ---
  (shared, NMF,     unw+nn)    = test.py 的 NMF 路徑

這樣可以一次只看一個變因:
  shared vs own       -> 只看「逐 spaxel 連續譜」的效果
  unw+* vs chi2+*     -> 只看「chi^2 加權」的效果
  *+nn  vs *+free     -> 只看「非負」的效果

記帳方式(所有方法一致,且是 source 區也能安全沿用的):
最後減掉的是「共用天光連續譜 + 重建的天光線」;逐 spaxel 連續譜只當擬合工具,
不永久扣除 —— 否則搬到 source 區會把 source 連續譜一起吃掉。
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF, PCA
from scipy.optimize import nnls
from scipy.stats import skew, kurtosis

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 本檔在 experiments/,utils.py 在上一層
from utils import estimate_continuum, running_median


# ---------------------------------------------------------------- 參數
K          = 10        # 天光線基底條數
WINDOW     = 300       # 連續譜 running median 視窗 (px)
THRESHOLDS = (1, 2)    # 線偵測門檻 (正, 負),教授指定
MAX_ITER   = 20        # estimate_continuum 迭代上限(只作用在 mean sky 上)
CHUNK      = 8000      # 分塊大小,控制記憶體


# ---------------------------------------------------------------- 工具
def spectrum_stats(spec):
    """把一條光譜濃縮成摘要統計。"""
    spec = spec[np.isfinite(spec)]
    return {
        "mean":          np.mean(spec),
        "sigma":         np.std(spec),
        "skewness":      skew(spec),
        "kurtosis":      kurtosis(spec),
        "rms_from_zero": np.sqrt(np.mean(spec**2)),
    }


def fmt_stats(name, spec):
    st = spectrum_stats(spec)
    return f"{name:<34}" + "  ".join(f"{k}={v:>10.4g}" for k, v in st.items())


def plot_compare(wl, spec, spec_compare, out_path, label="ours", label_compare="nosky",
                 ylim=(-20, 20), title=None):
    """對照圖:左光譜(藍=spec、橘虛線=spec_compare),右兩組 stats。"""
    fig, (ax, stat_ax) = plt.subplots(1, 2, figsize=(15.5, 4.5),
                                      gridspec_kw={"width_ratios": [5, 1]})
    ax.axhline(0, color="0.5", lw=0.5)
    ax.plot(wl, spec, lw=0.9, color="#1f77b4", label=label)
    ax.plot(wl, spec_compare, lw=0.9, ls="--", alpha=0.7, color="#e8710a", label=label_compare)
    ax.set_ylim(*ylim)
    ax.set_xlabel("wavelength [A]"); ax.set_ylabel("flux")
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8)

    stat_ax.axis("off")
    def block(name, s):
        st = spectrum_stats(s)
        return f"[{name}]\n" + "\n".join(f"{k:<13} = {v:.4g}" for k, v in st.items())
    stat_ax.text(0, 0.95, block(label, spec), color="#1f77b4",
                 va="top", family="monospace", fontsize=7, transform=stat_ax.transAxes)
    stat_ax.text(0, 0.45, block(label_compare, spec_compare), color="#e8710a",
                 va="top", family="monospace", fontsize=7, transform=stat_ax.transAxes)

    fig.tight_layout()
    fig.savefig(out_path, dpi=145)
    plt.close()


def per_spaxel_continuum(spectra, line_mask, window=WINDOW, chunk=CHUNK):
    """ZAP 的 continuum filter:每條 spaxel 沿波長取 running median(遮掉天光線通道)。

    數值定義就是 utils.running_median;這裡改用 pandas 的 rolling median 做量產,
    因為它算的是同一個東西但快約 87 倍(下方 verify_fast_median 會當場驗證)。
    spectra: (nz, n) -> 回傳 (nz, n) float32
    """
    nz, n = spectra.shape
    out = np.empty((nz, n), dtype=np.float32)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        M = spectra[:, lo:hi].astype(np.float64)
        M[line_mask, :] = np.nan                       # 天光線通道不參與中位數
        out[:, lo:hi] = (pd.DataFrame(M)
                         .rolling(window, center=True, min_periods=1)
                         .median().to_numpy().astype(np.float32))
        print(f"    continuum {hi}/{n}", flush=True)
    return out


def verify_fast_median(spectra, line_mask, n_check=50, window=WINDOW):
    """當場驗證 pandas rolling median 與 utils.running_median 等價。"""
    sel = np.random.default_rng(0).choice(spectra.shape[1], n_check, replace=False)
    M = spectra[:, sel].astype(np.float64)
    M[line_mask, :] = np.nan
    ref  = np.column_stack([running_median(M[:, j], window) for j in range(n_check)])
    fast = pd.DataFrame(M).rolling(window, center=True, min_periods=1).median().to_numpy()
    d = np.abs(fast - ref)
    print(f"verify vs running_median ({n_check} spaxels): "
          f"max diff {np.nanmax(d):.3e}, median diff {np.nanmedian(d):.3e}", flush=True)


def fit_amplitudes(r, sigma, A, nonneg):
    """解一條 spaxel 的天光線振幅。

    sigma=None  -> 無權重最小二乘   min ||r - A w||^2
    sigma given -> chi^2 最小化     min sum (r - A w)^2 / sigma^2
                   = || r/sigma - (A/sigma) w ||^2   (白化後就是普通最小二乘)
    nonneg=True -> 用 NNLS 加上 w >= 0 的限制
    """
    good = np.isfinite(r)
    if sigma is None:
        At, dt = A[good], r[good]
    else:
        good = good & np.isfinite(sigma) & (sigma > 0)
        At = A[good] / sigma[good][:, None]
        dt = r[good] / sigma[good]
    if At.shape[0] <= A.shape[1]:
        return np.zeros(A.shape[1])
    return nnls(At, dt)[0] if nonneg else np.linalg.lstsq(At, dt, rcond=None)[0]


def solve_all(r_fit, sigma_all, A, nonneg, weighted, tag):
    """對所有 spaxel 解振幅,回傳 W (K, n)。"""
    t0 = time.time()
    n = r_fit.shape[1]
    W = np.zeros((A.shape[1], n), dtype=np.float64)
    for j in range(n):
        W[:, j] = fit_amplitudes(r_fit[:, j].astype(np.float64),
                                 sigma_all[:, j].astype(np.float64) if weighted else None,
                                 A, nonneg)
        if (j + 1) % 20000 == 0:
            print(f"    {tag} {j+1}/{n}", flush=True)
    print(f"  {tag} solved in {time.time()-t0:.1f}s", flush=True)
    return W


def evaluate(W, A, spectra, var_all, continuum_shared, chunk=CHUNK):
    """統一的評測:blank region 平均殘差光譜 + 每條 spaxel 的 reduced chi^2。

    殘差定義(五個方法一致):resid = s - 共用天光連續譜 - A @ w
    """
    nz, n = spectra.shape
    ssum = np.zeros(nz); scnt = np.zeros(nz)
    chi2 = np.zeros(n);  dof  = np.zeros(n)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        resid = spectra[:, lo:hi] - continuum_shared[:, None] - (A @ W[:, lo:hi]).astype(np.float32)
        v = var_all[:, lo:hi]
        ok = np.isfinite(resid) & np.isfinite(v) & (v > 0)
        chi2[lo:hi] = np.where(ok, resid.astype(np.float64)**2 / np.where(ok, v, 1), 0).sum(axis=0)
        dof[lo:hi]  = ok.sum(axis=0) - A.shape[1]
        fin = np.isfinite(resid)
        ssum += np.where(fin, resid, 0).sum(axis=1)
        scnt += fin.sum(axis=1)
    return ssum / np.maximum(scnt, 1), chi2 / np.maximum(dof, 1)


# ---------------------------------------------------------------- 路徑與資料
ROOT   = Path(__file__).resolve().parents[3]  # experiments → skymodel → src → repo root
STEP01 = ROOT / "results/skymodel/step01"
WSKY   = ROOT / "data/Haro11_NEpointing_wsky.fits"
NOSKY  = ROOT / "data/Haro11_NEpointing_esonosky.fits"
OUT    = STEP01 / "zap_style"
OUT.mkdir(parents=True, exist_ok=True)

white_mask = fits.getdata(STEP01 / "whitelight.fits")
seg_mask   = fits.getdata(STEP01 / "seg.fits")
valid_mask = white_mask != 0
blank_mask = valid_mask & ~((seg_mask > 0) & valid_mask)
ys, xs = np.where(blank_mask)

hdr = fits.getheader(WSKY, "DATA")
nz  = hdr["NAXIS3"]
wl  = hdr["CRVAL3"] + (np.arange(nz) + 1 - hdr["CRPIX3"]) * hdr["CD3_3"]

t0 = time.time()
with fits.open(WSKY, memmap=True) as h:
    blank_spectra = np.array(h["DATA"].data[:, ys, xs], dtype=np.float32)   # (nz, n_blank)
    blank_var     = np.array(h["STAT"].data[:, ys, xs], dtype=np.float32)   # 變異數 sigma^2
with fits.open(NOSKY, memmap=True) as h:
    mean_nosky = np.nanmean(np.array(h["DATA"].data[:, ys, xs], dtype=np.float32), axis=1)
n_blank = blank_spectra.shape[1]
print(f"blank spaxels: {n_blank}   ({time.time()-t0:.1f}s to load)", flush=True)

blank_sigma = np.sqrt(blank_var)


# ---------------------------------------------------------------- 1. 全域線遮罩(用 mean sky)
# mean sky 只負責一件事:決定「哪些波長通道是天光線」。天光線在每條 spaxel 都落在
# 同樣的波長,所以一份高 S/N 的全域遮罩就夠;不在單條吵雜的 spaxel 上重跑迭代線偵測。
t0 = time.time()
mean_sky = np.nanmean(blank_spectra, axis=1)
continuum_shared, sigma_shared, line_mask, _ = estimate_continuum(
    mean_sky, thresholds=THRESHOLDS, window=WINDOW, max_iter=MAX_ITER)
print(f"line_mask: {100*line_mask.mean():.1f}% of channels   ({time.time()-t0:.1f}s)", flush=True)


# ---------------------------------------------------------------- 2. 逐 spaxel 連續譜
verify_fast_median(blank_spectra, line_mask)
t0 = time.time()
cont_own = per_spaxel_continuum(blank_spectra, line_mask)
r_own = blank_spectra - cont_own                      # 只剩窄的線 -> 給 B 系列擬合用
del cont_own
print(f"per-spaxel continuum took {time.time()-t0:.1f}s", flush=True)

r_shared = blank_spectra - continuum_shared[:, None]  # 共用連續譜 -> 給 A 用


# ---------------------------------------------------------------- 3. 全因子:2 連續譜 x 2 basis
# 沒有「特別待遇的基準線」—— 現行 test.py 的兩條路徑本來就是這個矩陣裡的兩格:
#   (shared, NMF,     unw+nn)    = test.py 的 NMF 路徑
CONTINUA = {"shared": r_shared,   # 所有 spaxel 共用一條 mean sky 連續譜(現行做法)
            "own":    r_own}      # 每條 spaxel 自己的連續譜(ZAP 做法)

BASIS_KINDS = ["NMF", "PCA"]


def learn_basis(kind, R):
    """從殘差 R (nz, n) 學 K 條天光線模板,回傳 (K, nz)。"""
    if kind == "NMF":       # 兩個矩陣都非負 -> 必須 clip 掉負值
        return NMF(n_components=K, init="nndsvda", max_iter=300).fit(
            np.nan_to_num(np.clip(R.T, 0, None))).components_
    if kind == "PCA":       # 都可正可負(ZAP 用的)-> 免 clip
        p = PCA(n_components=K - 1).fit(np.nan_to_num(R.T))
        return np.vstack([p.mean_[None, :], p.components_])   # 併入 mean,湊成 K 條才公平
    raise ValueError(kind)


bases = {}
for cname, R in CONTINUA.items():
    for kind in BASIS_KINDS:
        t0 = time.time()
        bases[(cname, kind)] = learn_basis(kind, R)
        print(f"basis [{cname:<6} {kind:<8}] took {time.time()-t0:.1f}s", flush=True)


# ---------------------------------------------------------------- 4. x 四種解法 = 16 組
SOLVERS = [("unw+nn",    False, True),
           ("chi2+nn",   True,  True),
           ("unw+free",  False, False),
           ("chi2+free", True,  False)]

CURRENT = {("shared", "NMF", "unw+nn"): "  <- test.py 的 NMF 路徑"}

results, notes = {}, {}
for (cname, kind), B in bases.items():
    A_mat = np.ascontiguousarray(B.T, dtype=np.float64)            # (nz, K)
    R = CONTINUA[cname]
    for sname, weighted, nonneg in SOLVERS:
        tag = f"{cname:<6} {kind:<8} {sname}"
        W = solve_all(R, blank_sigma, A_mat, nonneg=nonneg, weighted=weighted, tag=tag)
        results[tag] = evaluate(W, A_mat, blank_spectra, blank_var, continuum_shared)
        notes[tag] = CURRENT.get((cname, kind, sname), "")
        del W


# ---------------------------------------------------------------- 5. 統一評測輸出
HEAD = (f"{'cont   basis    solver':<32}{'mean(all)':>10}{'rms(all)':>10}{'rms(line)':>11}"
        f"{'mean(free)':>12}{'rms(free)':>11}{'red_chi2':>10}")

def row(name, mr, rc, note=""):
    return (f"{name:<32}{np.nanmean(mr):>10.3f}{np.sqrt(np.nanmean(mr**2)):>10.3f}"
            f"{np.sqrt(np.nanmean(mr[line_mask]**2)):>11.3f}"
            f"{np.nanmean(mr[~line_mask]):>12.3f}{np.sqrt(np.nanmean(mr[~line_mask]**2)):>11.3f}"
            f"{np.median(rc):>10.3f}{note}")

print()
print("=" * 96)
print("彙整表 —— blank region 平均殘差光譜 (rms 越接近 0 越好;red_chi2 理想 = 1)")
print("=" * 96)
print(HEAD)
for cname in CONTINUA:
    print("-" * 96)
    for kind in BASIS_KINDS:
        for sname, _, _ in SOLVERS:
            tag = f"{cname:<6} {kind:<8} {sname}"
            mr, rc = results[tag]
            print(row(tag, mr, rc, notes[tag]))
print("-" * 96)
print(f"{'-- ref: ESO nosky':<32}{np.nanmean(mean_nosky):>10.3f}"
      f"{np.sqrt(np.nanmean(mean_nosky**2)):>10.3f}"
      f"{np.sqrt(np.nanmean(mean_nosky[line_mask]**2)):>11.3f}"
      f"{np.nanmean(mean_nosky[~line_mask]):>12.3f}"
      f"{np.sqrt(np.nanmean(mean_nosky[~line_mask]**2)):>11.3f}{'':>10}")

for label, sl in [("全波長", slice(None)), ("只看天光線通道", line_mask), ("只看無線通道", ~line_mask)]:
    print()
    print("=" * 124)
    print(f"spectrum_stats —— {label}")
    print("=" * 124)
    for name, (mr, _) in results.items():
        print(fmt_stats(name, mr[sl]))
    print(fmt_stats("-- ref: ESO nosky", mean_nosky[sl]))


# ---------------------------------------------------------------- 6. 圖
# (a) 固定「連續譜 + 解法」,比三種 basis —— 直接看 clip / 分解方法的效果
for cname in CONTINUA:
    for sname, _, _ in SOLVERS:
        plt.figure(figsize=(13, 5))
        for kind in BASIS_KINDS:
            tag = f"{cname:<6} {kind:<8} {sname}"
            plt.plot(wl, results[tag][0], lw=0.6, label=tag)
        plt.plot(wl, mean_nosky, lw=0.6, ls="--", color="k", alpha=0.6, label="ESO nosky")
        plt.axhline(0, color="0.5", lw=0.5)
        plt.ylim(-20, 20); plt.xlabel("wavelength [A]"); plt.ylabel("flux")
        plt.title(f"basis comparison  (cont={cname}, solver={sname})")
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(OUT / f"cmp_basis_{cname}_{sname.replace('+','_')}.png", dpi=150); plt.close()

# (b) 固定「連續譜 + basis」,比四種解法 —— 直接看 chi^2 加權與非負的效果
for cname in CONTINUA:
    for kind in BASIS_KINDS:
        plt.figure(figsize=(13, 5))
        for sname, _, _ in SOLVERS:
            tag = f"{cname:<6} {kind:<8} {sname}"
            plt.plot(wl, results[tag][0], lw=0.6, label=tag)
        plt.plot(wl, mean_nosky, lw=0.6, ls="--", color="k", alpha=0.6, label="ESO nosky")
        plt.axhline(0, color="0.5", lw=0.5)
        plt.ylim(-20, 20); plt.xlabel("wavelength [A]"); plt.ylabel("flux")
        plt.title(f"solver comparison  (cont={cname}, basis={kind})")
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(OUT / f"cmp_solver_{cname}_{kind}.png", dpi=150); plt.close()

# (c) 兩條「現行 test.py 路徑」單獨拉出來跟 nosky 對照
for (cname, kind, sname), note in CURRENT.items():
    tag = f"{cname:<6} {kind:<8} {sname}"
    plot_compare(wl, results[tag][0], mean_nosky, OUT / f"current_{kind}.png",
                 label=tag, title=f"current test.py path: {kind}")

plt.figure(figsize=(9, 5))
for name, (_, rc) in results.items():
    plt.hist(rc, bins=np.linspace(0, np.percentile(rc, 99), 80), histtype="step", lw=1.0, label=name)
plt.axvline(1.0, color="k", ls="--", lw=1, label="ideal = 1")
plt.xlabel("reduced chi^2"); plt.ylabel("N spaxels")
plt.title("blank-region goodness of fit")
plt.legend(fontsize=5); plt.tight_layout()
plt.savefig(OUT / "reduced_chi2_hist.png", dpi=150); plt.close()

# (d) basis 模板長相對照(同一個連續譜處理下,三種分解方法)
for cname in CONTINUA:
    fig, axes = plt.subplots(K, 1, figsize=(13, 1.5 * K), sharex=True)
    for k in range(K):
        for kind, color in zip(BASIS_KINDS, ["#1f77b4", "#d62728"]):
            axes[k].plot(wl, bases[(cname, kind)][k], lw=0.6, color=color,
                         label=kind if k == 0 else None)
        axes[k].axhline(0, color="0.6", lw=0.4)
        axes[k].set_ylabel(f"k={k}", fontsize=7)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("wavelength [A]")
    fig.tight_layout(); fig.savefig(OUT / f"basis_compare_{cname}.png", dpi=130); plt.close()

print()
print(f"figures -> {OUT}")
