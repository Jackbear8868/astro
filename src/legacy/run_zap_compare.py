"""
ZAP 對照（記憶體友善版，一次只載入一個 cube，分步驟執行）。

一步一步跑（每步都是獨立 process，跑完再跑下一步）：
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py crop          # 先裁切大 cube
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py mask
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py zap   nosky
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py zap   wsky
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py collect nosky
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py collect wsky
  PYTHONPATH=libs/zap python3 src/run_zap_compare.py report        # 只印量化驗證數字（不畫圖）

圖表改由獨立腳本畫（都只讀上面 collect 產生的 npz，記憶體極省）：
  python3 src/fig1_wsky_effect.py    # 天空線扣除（wsky raw / wsky+ZAP / nosky 真值）
  python3 src/fig2_nosky_effect.py   # nosky null test（雜訊放大）
  python3 src/fig3_source_halpha.py  # 源 Hα 保真
  python3 src/fig4_source_mask.py    # 源遮罩（讀 source_mask.fits）

重點：ZAP 是「扣天空 + 去殘差」工具，正確輸入是【還含天空的 cube = wsky】。
  - wsky + ZAP  → 應該把天空線扣到接近 MUSE 的 nosky 真值, 且保留源 (見 fig1 / fig3)。
  - nosky + ZAP → nosky 已被 MUSE 扣過天空(空白譜已平), ZAP 沒有天空可學, 只會灌入雜訊;
                  這是「對照組/null test」(見 fig2), 不是 ZAP 壞掉。
"""
import sys, os, time, warnings
import numpy as np
sys.path.insert(0, "libs/zap")
warnings.filterwarnings("ignore")
from astropy.io import fits

OUT = "results/zap"; os.makedirs(OUT, exist_ok=True)
FITSDIR = f"{OUT}/fits"; os.makedirs(FITSDIR, exist_ok=True)   # 大 .fits 全放這裡
NPZDIR = f"{OUT}/npz"; os.makedirs(NPZDIR, exist_ok=True)      # 中介 .npz (cache/summ) 放這裡; .png 留在 OUT
RAW = {"nosky": "data/Haro11_nosky.fits", "wsky": "data/Haro11_wsky.fits"}
# ZAP/mask/collect 都用「裁切後」的 cube (省記憶體, 約標準 MUSE 大小)
FILES = {"nosky": f"{FITSDIR}/crop_nosky.fits", "wsky": f"{FITSDIR}/crop_wsky.fits"}
CROP = (slice(87, 387), slice(165, 465))     # (y, x) 300x300 標準 MUSE 大小, 星系@(237,315)居中
NBLANK = 8000   # 取樣空白 spaxel 數 (省記憶體)

# ---- step 0: crop (取 DATA+STAT, 修正 WCS) ----
#   crop        -> 裁成 CROP 指定的 300x300 (省記憶體)
#   crop full   -> 不做空間裁切, 用整張視場 (499x559, 不裁掉外圍 CGM)
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

# ---- step 1: mask (資料驅動物理參數遮罩: matched filter + 正常門檻) ----
# 參數依資料實際尺度推得 (見 CLAUDE.md 原則2):
#   matched filter 核 FWHM = seeing (1.24"/0.2" ≈ 6px);  門檻 2σ (假陽性2.3%, 物理站得住腳);
#   minarea = 1 PSF 面積 ≈ 30px;  dilation = 1×seeing ≈ 6px;
#   背景框 bw=256 (> 延展暈, 才不會把暈當背景吸走), 用 sep 的 Background.rms() 自動排源估噪音
#   (噪音必須從無源區估, 否則暈會把 σ 撐大 → 門檻失真 → 漏暈)。
# 註: 只用 Hα 窄帶偵測 (核心在 Hα 也很亮)。若場中有亮的連續譜恆星, 需另加白光偵測 OR 進來。
HA_LINE = (6692.0, 6708.0)                 # Hα(+核心) 發射線窗 (z≈0.0206)
HA_CONT = ((6605.0, 6645.0), (6760.0, 6795.0))  # 兩側乾淨連續譜 (避開 [NII])
BW, FWHM, THR, MINAREA, DILATE = 256, 6.0, 2.0, 30, 6

def _gauss_kernel(sz, fwhm):
    x = np.arange(sz) - sz // 2
    g = np.exp(-(x**2) / (2 * (fwhm / 2.355)**2))
    k = np.outer(g, g)
    return (k / k.sum()).astype(np.float32)

def cmd_mask():
    """做出『源遮罩』+ 挑出『空白天空 spaxel』座標，產生兩個檔：
         - {FITSDIR}/source_mask.fits : 之後餵給 ZAP，讓它學天空時避開源
         - {NPZDIR}/_cache.npz        : 記「要看哪些像素」的座標清單 + 波長軸（很小，後續每步都讀它）

    為什麼在 nosky 上偵測源？nosky 已被 MUSE 扣過天空、空白區很平，源(星系 Hα)對比最乾淨、
    最好定位；做出的遮罩之後 wsky / nosky 共用。
    """
    import sep
    from scipy import ndimage as ndi

    # ---- (0) 讀「已扣天空」的 nosky cube；cube 形狀 = (波長平面, y, x) ----
    hd = fits.open(FILES["nosky"])
    cube = hd["DATA"].data
    hdr = hd["DATA"].header
    wl = wl_axis(hdr)                                   # 由 header(CRVAL3/CRPIX3/CD3_3) 算出每個平面的波長 [Å]

    # ---- (1) 合成一張「純 Hα」窄帶影像：線內平均 − 兩側連續譜平均 ----
    white = np.nansum(cube, axis=0).astype(np.float32)  # 全波段疊加＝白光影像（定位最亮點、判斷有效視場）
    li = (wl > HA_LINE[0]) & (wl < HA_LINE[1])          # 布林遮罩：落在 Hα 線窗內的波長平面
    ci = np.zeros_like(wl, bool)
    for a, b in HA_CONT:                                # 兩側「乾淨連續譜」窗（避開 [NII]），聯集起來
        ci |= (wl > a) & (wl < b)
    # 線內平均亮度 − 連續譜平均亮度 ⇒ 扣掉連續譜後，剩下純發射線訊號
    ha = (np.nanmean(cube[li], 0) - np.nanmean(cube[ci], 0)).astype(np.float32)
    hd.close()                                          # 立刻關檔釋放大 cube

    valid = white != 0                                  # 有效視場（白光非 0）
    invalid = ~valid                                    # 視場外：要 mask 掉，不參與偵測/估背景

    # ---- (2) 偵測源：大背景框 + 高斯 matched filter + 2σ 正常門檻 ----
    #   關鍵①：sep.Background 會自動把源排除再估背景/RMS ⇒ 噪音從「無源區」估，門檻才不會被暈撐失真。
    #   關鍵②：bw=bh=256 的大框 > 延展暈直徑，暈才不會被當成背景吸走（見 CLAUDE.md 原則2 的兩個陷阱）。
    bkg = sep.Background(np.ascontiguousarray(ha), mask=invalid, bw=BW, bh=BW, fw=3, fh=3)
    #   關鍵③：filter_kernel = 高斯 matched filter(FWHM≈seeing)，先把 S/N 拉高，再用正常 2σ 門檻；
    #          而不是把門檻壓到雜訊以下（那樣假陽性會爆高、物理站不住腳）。
    #   回傳 segmentation_map：seg>0 的像素＝被歸到某個源。
    _, seg = sep.extract(ha - bkg, THR, err=bkg.rms(), mask=invalid, minarea=MINAREA,
                         filter_kernel=_gauss_kernel(15, FWHM),
                         deblend_nthresh=32, deblend_cont=0.005, segmentation_map=True)
    #   把偵測到的源向外膨脹 DILATE(≈1×seeing) px 當安全邊界，避免源的翼漏進天空樣本。
    src = ndi.binary_dilation((seg > 0) & valid, iterations=DILATE) & valid

    # ---- (3) 源遮罩存成 FITS（uint8：1=源, 0=可用天空）----
    fits.writeto(f"{FITSDIR}/source_mask.fits", src.astype(np.uint8), overwrite=True)

    # ---- (4) 定義「空白天空 spaxel」＝有效視場、且不是源 ----
    blank = valid & ~src
    #   同時記住「最亮的那個 spaxel」座標（白光最大值處）＝星系核心；之後用它的光譜檢查源保真。
    sy, sx = np.unravel_index(np.nanargmax(np.where(valid, white, np.nan)), white.shape)

    # ---- (5) 從所有空白 spaxel 隨機抽最多 NBLANK(8000) 個（省記憶體；種子固定＝可重現）----
    ys, xs = np.where(blank)                            # 所有空白 spaxel 的 (y, x) 索引
    rng = np.random.default_rng(0)                      # 固定亂數種子 0 ⇒ 每次抽到同一批樣本
    if len(ys) > NBLANK:
        pick = rng.choice(len(ys), NBLANK, replace=False)
        ys, xs = ys[pick], xs[pick]

    # ---- (6) 存 _cache.npz：只存「要看哪些像素」的清單 + 波長軸 ----
    np.savez(f"{NPZDIR}/_cache.npz",
             wl=wl,          # 波長軸（畫圖 x 軸）
             by=ys, bx=xs,   # 空白天空 spaxel 的 y, x 索引（抽天空光譜用）
             sy=sy, sx=sx)   # 最亮 spaxel 的 y, x 索引（抽源光譜用）
    print(f"[mask] 源 {int(src.sum())} ({100*src.sum()/valid.sum():.0f}%), "
          f"blank {int(blank.sum())} (取樣 {len(ys)}), 亮源@({sy},{sx})")

# ---- step 1: zap (跑一個 cube) ----
def cmd_zap(tag, ncpu=16):
    import zap
    t0 = time.time()
    print(f"[zap] {tag} 開始 (ncpu={ncpu}) ...", flush=True)
    zap.process(FILES[tag], outcubefits=f"{FITSDIR}/{tag}_zap.fits",
                skycubefits=f"{FITSDIR}/{tag}_skyremoved.fits",
                varcurvefits=f"{FITSDIR}/{tag}_varcurve.fits",
                mask=f"{FITSDIR}/source_mask.fits", ncpu=ncpu, overwrite=True)
    print(f"[zap] {tag} 完成，{time.time()-t0:.0f}s", flush=True)

# ---- step 2: collect (把一個 raw cube + 它的 zap cube 壓成小摘要; 一次只開一個 cube 省記憶體) ----
def _summarize(path, by, bx, sy, sx, k6300):
    """打開『一個』cube，用 _cache.npz 記下的座標抽出 4 個小陣列後回傳（隨即釋放大 cube）。

    參數：
      by, bx : 空白天空 spaxel 的 y, x 索引（來自 _cache.npz）
      sy, sx : 最亮 spaxel 的 y, x 索引
      k6300  : 最接近 6300Å 的波長平面索引（存一張診斷影像用）
    回傳 dict：med / std / srcspec / img6300（呼叫端再加 raw_ 或 zap_ 前綴）。
    """
    d = fits.open(path)["DATA"].data                    # (波長平面, y, x)

    blankspec = d[:, by, bx]                            # 抽出所有空白天空光譜 → (波長, Nblank)
    out = dict(
        # 空白區「中位光譜」：沿 spaxel 取中位數 ⇒ 天空線扣得乾不乾淨（fig1 用）
        med=np.nanmedian(blankspec, axis=1).astype(np.float32),
        # 空白區「逐波長標準差」：spaxel 之間的散布 ⇒ 殘餘雜訊大不大（fig2 用）
        std=np.nanstd(blankspec, axis=1).astype(np.float32),
        # 最亮 spaxel 的整條光譜：源(Hα)有沒有被保住 / 被吃掉（fig3 用）
        srcspec=d[:, sy, sx].astype(np.float32),
        # 6300Å 那個波長平面的整張影像：純診斷用（檢查天空線平面長相）
        img6300=d[k6300].astype(np.float32))
    del d, blankspec                                    # 立刻釋放，避免同時常駐兩個大 cube
    return out

def cmd_collect(tag):                                   # tag = "nosky" 或 "wsky"
    # ---- (0) 讀座標清單；找出最接近 6300Å 的波長平面索引 ----
    c = np.load(f"{NPZDIR}/_cache.npz"); wl = c["wl"]
    k = int(np.argmin(abs(wl-6300)))

    # ---- (1) 先抽「原始 cube」的摘要 → key 前綴 raw_ ----
    raw = _summarize(FILES[tag], c["by"], c["bx"], int(c["sy"]), int(c["sx"]), k)
    print(f"[collect] {tag} raw done", flush=True)
    out = {f"raw_{k_}": v for k_, v in raw.items()}     # raw_med / raw_std / raw_srcspec / raw_img6300

    # ---- (2) 若這個 cube 有跑過 ZAP，再抽「ZAP 後 cube」的摘要 → key 前綴 zap_ ----
    zpath = f"{FITSDIR}/{tag}_zap.fits"
    if os.path.exists(zpath):
        zp = _summarize(zpath, c["by"], c["bx"], int(c["sy"]), int(c["sx"]), k)
        out.update({f"zap_{k_}": v for k_, v in zp.items()})   # zap_med / zap_std / zap_srcspec / zap_img6300
        print(f"[collect] {tag} zap done", flush=True)
    else:
        # 沒有 zap cube（例如整張版把 nosky 純當參考真值、不跑 ZAP）就只存 raw
        print(f"[collect] {tag}: 無 {zpath}, 只存 raw (當參考真值)", flush=True)

    # ---- (3) 全部存成 summ_{tag}.npz（raw_* 一定有；zap_* 視有無跑過 ZAP 而定）----
    np.savez(f"{NPZDIR}/summ_{tag}.npz", **out)
    print(f"[collect] 存 {OUT}/summ_{tag}.npz")

# ---- step 4: report (只讀小 npz, 印量化驗證數字; 圖改由 src/fig1-4_*.py 各自畫) ----
# 註: 原本的 cmd_figs() 會畫 fig5 三合一驗證圖 + fig1-3 null test, 已被獨立的
#     src/fig1_wsky_effect.py ~ fig4_source_mask.py 取代; 這裡只保留「量化驗證數字」輸出。
def cmd_report():
    """印出 wsky+ZAP 的量化驗證：天空線殘餘、空白區雜訊、源 Hα 通量保留率。
    只讀 collect 產生的小 npz，不畫任何圖（圖由 src/fig1-4_*.py 各自產生）。
    """
    c = np.load(f"{NPZDIR}/_cache.npz"); wl = c["wl"]
    N = np.load(f"{NPZDIR}/summ_nosky.npz"); W = np.load(f"{NPZDIR}/summ_wsky.npz")
    at = lambda arr, lam: float(arr[int(np.argmin(abs(wl-lam)))])   # 取最接近某波長 lam 的值
    SKY = [5577, 6300, 8400]                                        # 三條主要天空線

    def haflux(spec):                                  # 源 Hα 積分通量: 6692-6708 線內 減 兩側連續譜
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
    print("  → 圖表請跑 src/fig1_wsky_effect.py ~ fig4_source_mask.py")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "crop": cmd_crop(sys.argv[2] if len(sys.argv) > 2 else "box")
    elif cmd == "mask": cmd_mask()
    elif cmd == "zap": cmd_zap(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 16)
    elif cmd == "collect": cmd_collect(sys.argv[2])
    elif cmd == "report": cmd_report()
    else: print(__doc__)
