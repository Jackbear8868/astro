"""
ZAP 對照（記憶體友善版，一次只載入一個 cube，分步驟執行）。

一步一步跑（每步都是獨立 process，跑完再跑下一步）：
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py mask
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py zap   nosky
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py zap   wsky
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py collect nosky
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py collect wsky
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py figs
"""
import sys, os, time, warnings
import numpy as np
sys.path.insert(0, "libs/zap")
warnings.filterwarnings("ignore")
from astropy.io import fits

OUT = "results/zap"; os.makedirs(OUT, exist_ok=True)
RAW = {"nosky": "data/Haro11_nosky.fits", "wsky": "data/Haro11_wsky.fits"}
# ZAP/mask/collect 都用「裁切後」的 cube (省記憶體, 約標準 MUSE 大小)
FILES = {"nosky": f"{OUT}/crop_nosky.fits", "wsky": f"{OUT}/crop_wsky.fits"}
CROP = (slice(127, 347), slice(205, 425))   # (y, x) 220x220, 含星系@(237,315)+周圍天空
NBLANK = 8000   # 取樣空白 spaxel 數 (省記憶體)

# ---- step: crop (把大 cube 裁成小 cube, DATA+STAT, 修正 WCS) ----
def cmd_crop():
    sy, sx = CROP
    for tag in ("nosky", "wsky"):
        hd = fits.open(RAW[tag])
        prim = hd[0].copy()
        dh = hd["DATA"].copy(); sh = hd["STAT"].copy()
        dh.data = dh.data[:, sy, sx]; sh.data = sh.data[:, sy, sx]
        for h in (dh.header, sh.header):
            h["CRPIX1"] = h["CRPIX1"] - sx.start
            h["CRPIX2"] = h["CRPIX2"] - sy.start
        fits.HDUList([prim, dh, sh]).writeto(FILES[tag], overwrite=True)
        hd.close()
        print(f"[crop] {tag}: {dh.data.shape} -> {FILES[tag]}")

def wl_axis(hdr):
    return hdr["CRVAL3"] + (np.arange(hdr["NAXIS3"]) + 1 - hdr["CRPIX3"]) * hdr["CD3_3"]

# ---- step: mask (建源遮罩 + 快取 blank/源位置/wl) ----
def cmd_mask():
    hd = fits.open(FILES["nosky"]); cube = hd["DATA"].data; hdr = hd["DATA"].header
    wl = wl_axis(hdr)
    white = np.nansum(cube, axis=0).astype(np.float32); hd.close()
    valid = white != 0
    med = np.median(white[valid]); mad = np.median(np.abs(white[valid]-med))*1.4826
    src = ((white - med) > 2*mad) & valid          # 1=源
    fits.writeto(f"{OUT}/source_mask.fits", src.astype(np.uint8), overwrite=True)
    blank = valid & ~src
    sy, sx = np.unravel_index(np.nanargmax(np.where(valid, white, np.nan)), white.shape)
    # 取樣 blank 索引
    ys, xs = np.where(blank)
    rng = np.random.default_rng(0)
    if len(ys) > NBLANK:
        pick = rng.choice(len(ys), NBLANK, replace=False); ys, xs = ys[pick], xs[pick]
    np.savez(f"{OUT}/_cache.npz", wl=wl, by=ys, bx=xs, sy=sy, sx=sx)
    print(f"[mask] 源 {int(src.sum())}, blank {int(blank.sum())} (取樣 {len(ys)}), 亮源@({sy},{sx})")

# ---- step: zap (跑一個 cube) ----
def cmd_zap(tag):
    import zap
    t0 = time.time()
    print(f"[zap] {tag} 開始 ...", flush=True)
    zap.process(FILES[tag], outcubefits=f"{OUT}/{tag}_zap.fits",
                skycubefits=f"{OUT}/{tag}_skyremoved.fits",
                varcurvefits=f"{OUT}/{tag}_varcurve.fits",
                mask=f"{OUT}/source_mask.fits", ncpu=1, overwrite=True)
    print(f"[zap] {tag} 完成，{time.time()-t0:.0f}s", flush=True)

# ---- step: collect (從一個 raw cube + 它的 zap cube 抽小陣列, 一次只開一個) ----
def _summarize(path, by, bx, sy, sx, k6300):
    d = fits.open(path)["DATA"].data
    blankspec = d[:, by, bx]                       # (nl, Nblank)
    out = dict(med=np.nanmedian(blankspec, axis=1).astype(np.float32),
               std=np.nanstd(blankspec, axis=1).astype(np.float32),
               srcspec=d[:, sy, sx].astype(np.float32),
               img6300=d[k6300].astype(np.float32))
    del d, blankspec
    return out

def cmd_collect(tag):
    c = np.load(f"{OUT}/_cache.npz"); wl = c["wl"]
    k = int(np.argmin(abs(wl-6300)))
    raw = _summarize(FILES[tag], c["by"], c["bx"], int(c["sy"]), int(c["sx"]), k)
    print(f"[collect] {tag} raw done", flush=True)
    zp = _summarize(f"{OUT}/{tag}_zap.fits", c["by"], c["bx"], int(c["sy"]), int(c["sx"]), k)
    print(f"[collect] {tag} zap done", flush=True)
    np.savez(f"{OUT}/summ_{tag}.npz",
             **{f"raw_{k_}": v for k_, v in raw.items()},
             **{f"zap_{k_}": v for k_, v in zp.items()})
    print(f"[collect] 存 {OUT}/summ_{tag}.npz")

# ---- step: figs (只讀小 npz, 記憶體極省) ----
def cmd_figs():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    c = np.load(f"{OUT}/_cache.npz"); wl = c["wl"]
    N = np.load(f"{OUT}/summ_nosky.npz"); W = np.load(f"{OUT}/summ_wsky.npz")

    # 圖1: 空白 spaxel before/after (仿 Fig6) + 放大 6300
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    ax[0].plot(wl, N["raw_med"], "tab:blue", lw=0.6, label="nosky (MUSE model)")
    ax[0].plot(wl, N["zap_med"], "tab:red", lw=0.6, label="nosky + ZAP")
    ax[0].set_ylim(-20, 40); ax[0].legend(); ax[0].set_title("空白天空中位譜：ZAP 前後")
    s = (wl > 6250) & (wl < 6400)
    ax[1].plot(wl[s], N["raw_med"][s], "tab:blue", label="nosky")
    ax[1].plot(wl[s], N["zap_med"][s], "tab:red", label="nosky+ZAP")
    ax[1].legend(); ax[1].set_title("放大 [OI]6300"); ax[1].set_xlabel("Å")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig1_blank_spectrum.png", dpi=130); plt.close(fig)

    # 圖2: 殘餘 std
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(wl, N["raw_std"], "tab:blue", lw=0.6, label="nosky")
    ax.plot(wl, N["zap_std"], "tab:red", lw=0.6, label="nosky+ZAP")
    ax.set_ylim(0, 30); ax.legend(); ax.set_title("空白區逐波長殘餘 std (越低越好)"); ax.set_xlabel("Å")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_residual_std.png", dpi=130); plt.close(fig)

    # 圖3: 源保真 (Hα 區)
    fig, ax = plt.subplots(figsize=(12, 4))
    s = (wl > 6600) & (wl < 6800)
    ax.plot(wl[s], N["raw_srcspec"][s], "tab:blue", label="nosky")
    ax.plot(wl[s], N["zap_srcspec"][s], "tab:red", ls="--", label="nosky+ZAP")
    ax.legend(); ax.set_title("源保真檢查：Hα 應不變"); ax.set_xlabel("Å")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_source_preservation.png", dpi=130); plt.close(fig)

    # 圖4: Run B — ZAP 當扣天空器 vs MUSE model (非真值, 兩法互比)
    fig, ax = plt.subplots(figsize=(12, 4))
    muse_sky = W["raw_med"] - N["raw_med"]         # wsky - nosky = MUSE model 天空
    zap_sky  = W["raw_med"] - W["zap_med"]         # wsky - wsky_zap = ZAP 天空
    ax.plot(wl, muse_sky, "k", lw=0.6, label="wsky−nosky (MUSE model 天空)")
    ax.plot(wl, zap_sky, "tab:red", lw=0.6, label="wsky−wsky_zap (ZAP 天空)")
    ax.legend(); ax.set_title("Run B：兩種扣天空法互比 (非 ground truth)"); ax.set_xlabel("Å")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_runB_method_compare.png", dpi=130); plt.close(fig)

    # 量化
    def at(arr, lam): return float(arr[int(np.argmin(abs(wl-lam)))])
    print("\n===== 量化: 空白區殘餘 std (nosky -> +ZAP) =====")
    for lam in [5577, 6300, 8400]:
        print(f"  {lam}Å : {at(N['raw_std'],lam):6.2f} -> {at(N['zap_std'],lam):6.2f}")
    print(f"  整體中位 std: {np.median(N['raw_std']):.2f} -> {np.median(N['zap_std']):.2f}")
    print(f"已存 fig1..fig4 於 {OUT}/")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "crop": cmd_crop()
    elif cmd == "mask": cmd_mask()
    elif cmd == "zap": cmd_zap(sys.argv[2])
    elif cmd == "collect": cmd_collect(sys.argv[2])
    elif cmd == "figs": cmd_figs()
    else: print(__doc__)
