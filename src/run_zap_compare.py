"""
ZAP 對照（記憶體友善版，一次只載入一個 cube，分步驟執行）。

一步一步跑（每步都是獨立 process，跑完再跑下一步）：
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py crop          # 先裁切大 cube
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py mask
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py zap   nosky
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py zap   wsky
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py collect nosky
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py collect wsky
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py figs

重點：ZAP 是「扣天空 + 去殘差」工具，正確輸入是【還含天空的 cube = wsky】。
  - wsky + ZAP  → 應該把天空線扣到接近 MUSE 的 nosky 真值, 且保留源 (fig5, 主要驗證)。
  - nosky + ZAP → nosky 已被 MUSE 扣過天空(空白譜已平), ZAP 沒有天空可學, 只會灌入雜訊;
                  這是「對照組/null test」(fig1-3), 不是 ZAP 壞掉。
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
CROP = (slice(87, 387), slice(165, 465))     # (y, x) 300x300 標準 MUSE 大小, 星系@(237,315)居中
NBLANK = 8000   # 取樣空白 spaxel 數 (省記憶體)

# ---- step: crop (取 DATA+STAT, 修正 WCS) ----
#   crop        -> 裁成 CROP 指定的 300x300 (省記憶體)
#   crop full   -> 不做空間裁切, 用整張視場 (499x559, 不裁掉外圍 CGM, 但較吃記憶體/時間)
def cmd_crop(region="box"):
    sy, sx = (slice(None), slice(None)) if region == "full" else CROP
    for tag in ("nosky", "wsky"):
        hd = fits.open(RAW[tag])
        prim = hd[0].copy()
        dh = hd["DATA"].copy(); sh = hd["STAT"].copy()
        dh.data = dh.data[:, sy, sx]; sh.data = sh.data[:, sy, sx]
        for h in (dh.header, sh.header):
            h["CRPIX1"] = h["CRPIX1"] - (sx.start or 0)
            h["CRPIX2"] = h["CRPIX2"] - (sy.start or 0)
        fits.HDUList([prim, dh, sh]).writeto(FILES[tag], overwrite=True)
        hd.close()
        print(f"[crop] {tag} ({region}): {dh.data.shape} -> {FILES[tag]}")

def wl_axis(hdr):
    return hdr["CRVAL3"] + (np.arange(hdr["NAXIS3"]) + 1 - hdr["CRPIX3"]) * hdr["CD3_3"]

# ---- step: mask (建源遮罩 + 快取 blank/源位置/wl) ----
# ZAP 的天空 PCA 必須只用「真正空白」的 spaxel。Haro11 的電離氣 (Hα/CGM) 延展到
# 視場一大半，若只用白光 2σ 門檻 (~8%) 當源遮罩，剩下的「空白」區仍充滿星系發射線，
# ZAP 會把 Hα 當天空學起來→把源吃掉、空白區灌入線狀殘差。因此這裡用 Hα 窄帶 + 白光
# 連續譜雙重偵測, 去除單點雜訊, 再膨脹一圈安全邊界, 把整個星系與延展氣都遮乾淨。
HA_LINE = (6692.0, 6708.0)                 # Hα(+核心) 發射線窗 (z≈0.0206)
HA_CONT = ((6605.0, 6645.0), (6760.0, 6795.0))  # 兩側乾淨連續譜 (避開 [NII])
THR_HA, THR_WHITE = 2.0, 3.0               # Hα / 白光 偵測門檻 (σ)
MIN_BLOB, DILATE = 20, 6                   # 最小連通塊像素 / 膨脹圈數 (安全邊界)

def cmd_mask():
    from scipy import ndimage as ndi
    hd = fits.open(FILES["nosky"]); cube = hd["DATA"].data; hdr = hd["DATA"].header
    wl = wl_axis(hdr)
    white = np.nansum(cube, axis=0).astype(np.float32)
    li = (wl > HA_LINE[0]) & (wl < HA_LINE[1])
    ci = np.zeros_like(wl, bool)
    for a, b in HA_CONT: ci |= (wl > a) & (wl < b)
    ha = (np.nanmean(cube[li], 0) - np.nanmean(cube[ci], 0)).astype(np.float32)
    hd.close()
    valid = white != 0

    def robust(z):                         # 穩健 (median, σ_MAD)
        v = z[valid]; m = np.median(v); return m, 1.4826*np.median(np.abs(v-m))
    # 先 3x3 中值濾波壓掉單點雜訊, 再門檻 → 不會誤抓滿天椒鹽雜訊
    ha_s = ndi.median_filter(ha, 3); wh_s = ndi.median_filter(white, 3)
    mh, sh = robust(ha_s); mw, sw = robust(wh_s)
    src = (((ha_s - mh) > THR_HA*sh) | ((wh_s - mw) > THR_WHITE*sw)) & valid
    # 只留夠大的連通塊 (丟掉雜訊碎點)
    lab, n = ndi.label(src)
    if n:
        big = np.zeros(n+1, bool)
        big[1:] = ndi.sum(np.ones_like(lab), lab, range(1, n+1)) >= MIN_BLOB
        src = big[lab]
    src = ndi.binary_dilation(src, iterations=DILATE) & valid   # 安全邊界
    fits.writeto(f"{OUT}/source_mask.fits", src.astype(np.uint8), overwrite=True)

    blank = valid & ~src
    sy, sx = np.unravel_index(np.nanargmax(np.where(valid, white, np.nan)), white.shape)
    # 取樣 blank 索引
    ys, xs = np.where(blank)
    rng = np.random.default_rng(0)
    if len(ys) > NBLANK:
        pick = rng.choice(len(ys), NBLANK, replace=False); ys, xs = ys[pick], xs[pick]
    np.savez(f"{OUT}/_cache.npz", wl=wl, by=ys, bx=xs, sy=sy, sx=sx)
    res_ha = float(np.percentile(ha[ys, xs], 90))
    print(f"[mask] 源 {int(src.sum())} ({100*src.sum()/valid.sum():.0f}%), "
          f"blank {int(blank.sum())} (取樣 {len(ys)}), 亮源@({sy},{sx}), "
          f"blank 殘餘Hα 90pct={res_ha:.2f} (σ≈{robust(ha)[1]:.2f})")

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
    out = {f"raw_{k_}": v for k_, v in raw.items()}
    zpath = f"{OUT}/{tag}_zap.fits"
    if os.path.exists(zpath):                       # 若沒跑 ZAP (例如 nosky 只當參考真值) 就只存 raw
        zp = _summarize(zpath, c["by"], c["bx"], int(c["sy"]), int(c["sx"]), k)
        out.update({f"zap_{k_}": v for k_, v in zp.items()})
        print(f"[collect] {tag} zap done", flush=True)
    else:
        print(f"[collect] {tag}: 無 {zpath}, 只存 raw (當參考真值)", flush=True)
    np.savez(f"{OUT}/summ_{tag}.npz", **out)
    print(f"[collect] 存 {OUT}/summ_{tag}.npz")

# ---- step: figs (只讀小 npz, 記憶體極省) ----
# 圖用英文標題 (容器內 matplotlib 無 CJK 字型, 中文會變成空格子)。
def cmd_figs():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    c = np.load(f"{OUT}/_cache.npz"); wl = c["wl"]
    N = np.load(f"{OUT}/summ_nosky.npz"); W = np.load(f"{OUT}/summ_wsky.npz")
    at = lambda arr, lam: float(arr[int(np.argmin(abs(wl-lam)))])
    SKY = [5577, 6300, 8400]

    # ===== fig5: 主要驗證 — wsky+ZAP 是否還原 MUSE 的 nosky 真值 (且保留源) =====
    fig, ax = plt.subplots(3, 1, figsize=(13, 11))
    ax[0].plot(wl, W["raw_med"], color="0.6", lw=0.5, label="wsky raw (sky NOT removed)")
    ax[0].plot(wl, W["zap_med"], "tab:red", lw=0.6, label="wsky + ZAP")
    ax[0].plot(wl, N["raw_med"], "tab:blue", lw=0.6, label="nosky (MUSE reference truth)")
    ax[0].set_ylim(-30, 120); ax[0].legend()
    ax[0].set_title("Blank-sky median spectrum: ZAP removes sky lines down to ~MUSE truth")
    ax[1].plot(wl, W["zap_std"], "tab:red", lw=0.6, label="wsky + ZAP")
    ax[1].plot(wl, N["raw_std"], "tab:blue", lw=0.6, label="nosky (MUSE truth)")
    ax[1].set_ylim(0, 30); ax[1].legend(); ax[1].set_xlabel("wavelength [A]")
    ax[1].set_title("Blank-region residual std per wavelength: ZAP ~ MUSE (lower=better)")
    s = (wl > 6600) & (wl < 6800)
    ax[2].plot(wl[s], N["raw_srcspec"][s], "tab:blue", label="nosky (MUSE truth)")
    ax[2].plot(wl[s], W["zap_srcspec"][s], "tab:red", ls="--", label="wsky + ZAP")
    ax[2].legend(); ax[2].set_xlabel("wavelength [A]")
    ax[2].set_title("Source fidelity: wsky+ZAP Halpha matches MUSE truth (source preserved)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_zap_validation.png", dpi=130); plt.close(fig)

    # ===== fig1-3: 對照組 (nosky+ZAP, null test) — nosky 已無天空, ZAP 只會灌雜訊 =====
    # 只有在 nosky 也跑過 ZAP 時才畫 (整張版只把 nosky 當參考真值, 不跑 ZAP, 就跳過)
    if "zap_med" in N:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(wl, N["raw_med"], "tab:blue", lw=0.6, label="nosky (already sky-subtracted)")
        ax.plot(wl, N["zap_med"], "tab:red", lw=0.6, label="nosky + ZAP (injects noise)")
        ax.set_ylim(-20, 40); ax.legend(); ax.set_xlabel("wavelength [A]")
        ax.set_title("NULL TEST: nosky+ZAP (no sky to remove -> ZAP adds noise, expected)")
        fig.tight_layout(); fig.savefig(f"{OUT}/fig1_nulltest_blank.png", dpi=130); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(wl, N["raw_std"], "tab:blue", lw=0.6, label="nosky")
        ax.plot(wl, N["zap_std"], "tab:red", lw=0.6, label="nosky+ZAP")
        ax.set_ylim(0, 30); ax.legend(); ax.set_xlabel("wavelength [A]")
        ax.set_title("NULL TEST residual std: nosky+ZAP worse (no coherent sky -> noise)")
        fig.tight_layout(); fig.savefig(f"{OUT}/fig2_nulltest_std.png", dpi=130); plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(wl[s], N["raw_srcspec"][s], "tab:blue", label="nosky")
        ax.plot(wl[s], N["zap_srcspec"][s], "tab:red", ls="--", label="nosky+ZAP")
        ax.legend(); ax.set_xlabel("wavelength [A]")
        ax.set_title("NULL TEST source: nosky+ZAP over-subtracts Halpha (needs source mask)")
        fig.tight_layout(); fig.savefig(f"{OUT}/fig3_nulltest_source.png", dpi=130); plt.close(fig)
    else:
        print("(略過 fig1-3 null test: nosky 未跑 ZAP, 只當參考真值)")

    # ===== 量化: ZAP 正確用法 (wsky) 的天空線扣除 + 源保真 =====
    def haflux(spec):                                  # 6692-6708 減兩側連續譜
        li = (wl > 6692) & (wl < 6708); cb = ((wl > 6660)&(wl < 6688)) | ((wl > 6730)&(wl < 6758))
        return float(np.nansum(spec[li] - np.nanmean(spec[cb])))
    print("\n===== ZAP 驗證 (正確用法: wsky+ZAP, 與 MUSE nosky 真值對比) =====")
    print("  空白天空線中位殘餘 (wsky原始 -> wsky+ZAP | MUSE真值):")
    for lam in SKY:
        print(f"    {lam}Å: {at(W['raw_med'],lam):8.1f} -> {at(W['zap_med'],lam):7.2f}  | {at(N['raw_med'],lam):6.2f}")
    print(f"  空白區整體中位 std: wsky原始 {np.median(W['raw_std']):.2f} -> wsky+ZAP "
          f"{np.median(W['zap_std']):.2f}  (MUSE真值 {np.median(N['raw_std']):.2f})")
    hn, hz = haflux(N["raw_srcspec"]), haflux(W["zap_srcspec"])
    print(f"  源 Hα 積分通量: MUSE真值 {hn:.0f} | wsky+ZAP {hz:.0f}  (保留 {100*hz/hn:.1f}%)")
    print(f"  → 圖: {OUT}/fig5_zap_validation.png (主), fig1-3 為 nosky null test")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "crop": cmd_crop(sys.argv[2] if len(sys.argv) > 2 else "box")
    elif cmd == "mask": cmd_mask()
    elif cmd == "zap": cmd_zap(sys.argv[2])
    elif cmd == "collect": cmd_collect(sys.argv[2])
    elif cmd == "figs": cmd_figs()
    else: print(__doc__)
