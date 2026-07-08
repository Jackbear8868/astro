"""
step1 — 建 source / segmentation mask（可從 nosky 或 wsky 建；預設 sep）。
  輸入： settings.input_cube_path(from_cube)          （from_cube ∈ {nosky, wsky}）
  輸出： results/zap/masks/<method>_from-<from_cube>/mask.fits   （1=源, 0=可用天空；餵給 step2）
         results/zap/masks/<method>_from-<from_cube>/blanks.npz  （亮源座標 sy,sx + 波長軸；供 M3）
         每個 (方法 × cube) 各自一個資料夾，彼此不覆蓋。
  跑法： conda run -n astro python src/0706/step1_mask.py nosky          # 從 nosky 建（預設 sep）
         conda run -n astro python src/0706/step1_mask.py wsky           # 從 wsky 建
         conda run -n astro python src/0706/step1_mask.py nosky claude   # 改用方法 claude

  兩種偵測方法（同一張 Hα 窄帶影像，差在怎麼定門檻）：
    「sep」   （預設）: SEP + 高斯 matched filter(核=seeing) + 正常 2σ；
                        sep.Background 自動排源估 RMS（噪音從無源區估，門檻不被暈撐失真）。
    「claude」（對照）: robust-MAD 手寫門檻。median_filter 平滑 → 全圖 MAD σ →
                        「Hα 或 白光」超過門檻 → 連通塊過濾 → 膨脹。用 Hα + 白光雙通道。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import replace
import settings
import numpy as np
from astropy.io import fits

def _gauss_kernel(sz, fwhm):
    """高斯 matched filter 核（FWHM ≈ seeing）。"""
    x = np.arange(sz) - sz // 2
    g = np.exp(-(x**2) / (2 * (fwhm / 2.355)**2))
    k = np.outer(g, g)
    return (k / k.sum()).astype(np.float32)

def _stat_variance_narrowband(hd, wl):
    """把 cube STAT 延伸依「和窄帶影像相同」的線內/連續譜窗傳遞成 2D 變異圖。
       ha = mean(line) − mean(cont)；Var(mean_n)=Σvar/n²，連續譜相減再加變異：
       var_nb = Σ STAT[line]/nL² + Σ STAT[cont]/nC²。"""
    in_line = (wl > settings.HALPHA_LINE_WINDOW[0]) & (wl < settings.HALPHA_LINE_WINDOW[1])
    in_cont = np.zeros_like(wl, bool)
    for lo, hi in settings.HALPHA_IMAGE_CONTINUUM:
        in_cont |= (wl > lo) & (wl < hi)
    nL, nC = int(in_line.sum()), int(in_cont.sum())
    SL = hd["STAT"].data[in_line].astype(np.float32)
    SC = hd["STAT"].data[in_cont].astype(np.float32)
    return (np.nansum(SL, 0) / nL**2 + np.nansum(SC, 0) / nC**2).astype(np.float32)

def _stat_detect_valid(ha, valid, serr):
    """STAT 路徑用的「乾淨偵測域」：白光有效 ∩ ha 有限 ∩ serr 有限且 >0。
       （white!=0 可能含 Hα 線平面全 NaN 或 STAT 塌成 0 的像素，會毒到 median/sep，須排除。）"""
    return valid & np.isfinite(ha) & np.isfinite(serr) & (serr > 0)

def _calibrate_stat_k(sub, serr, dvalid, region):
    """量 STAT 校正因子 k = MAD(ha−bkg, blank) / median(sqrt(var_nb), blank)。
       MUSE STAT 因 resampling covariance 低估真噪 ~40%，故 raw sqrt(STAT)(k=1) 的 2σ 其實 ~1.4σ。
       region=(x0,x1) 或 (x0,x1,y0,y1) 指定 blank；None → 全場 sigma-clip 去源後估。
       dvalid 需為乾淨偵測域（sub/serr 皆有限），確保 median 不被 NaN 毒到。"""
    ny, nx = sub.shape
    if region is not None:
        yy, xx = np.mgrid[0:ny, 0:nx]
        x0, x1 = region[0], region[1]
        y0, y1 = (region[2], region[3]) if len(region) == 4 else (0, ny)
        m = dvalid & (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
    else:                                           # 無指定 blank：3σ clip 去源，剩下當噪音區
        v = sub[dvalid]; c = v.copy()
        for _ in range(5):
            md = np.median(c); sd = 1.4826 * np.median(np.abs(c - md))
            c = c[np.abs(c - md) < 3 * sd]
        sd = 1.4826 * np.median(np.abs(c - np.median(c)))
        m = dvalid & (np.abs(sub - np.median(c)) < 3 * sd)
    s, e = sub[m], serr[m]
    mad = 1.4826 * np.median(np.abs(s - np.median(s)))
    return float(mad / np.median(e))

def _stat_err_map(ha, valid, det, var_nb):
    """建 STAT 逐像素噪音 err = k·sqrt(var_nb)。回傳 (乾淨偵測域 dvalid, 局部扣背景 sub, err, k)。"""
    import sep
    serr = np.sqrt(var_nb).astype(np.float32)
    dvalid = _stat_detect_valid(ha, valid, serr)
    ha0 = np.ascontiguousarray(np.where(dvalid, ha, 0).astype(np.float32))
    bkg = sep.Background(ha0, mask=~dvalid, bw=det.bkg_box_px, bh=det.bkg_box_px, fw=3, fh=3)
    sub = ha0 - bkg                                 # 仍扣局部背景當「訊號位準」
    k = det.stat_calib_k if det.stat_calib_k is not None \
        else _calibrate_stat_k(sub, serr, dvalid, det.stat_calib_region)
    err = (k * serr).astype(np.float32)
    med = float(np.median(err[dvalid]))             # dvalid 內皆有限 → med 有限；域外補 med（反正被 mask）
    err = np.where(dvalid, err, med).astype(np.float32)
    return dvalid, sub, err, k

def _detect_sep(ha, valid, det, var_nb=None):
    """方法 sep：SEP + matched filter + 2σ。噪音來源二選一：
         det.use_stat_noise=False → sep.Background 自動排源估的 RMS（主場，維持舊行為，逐位元相同）
         det.use_stat_noise=True  → cube STAT 逐像素噪音圖 × 每 cube 校正 k（NE）
       回傳源遮罩 bool（仍夾在 white!=0 的 valid 內）。"""
    import sep
    from scipy import ndimage as ndi
    use_stat = det.use_stat_noise and var_nb is not None
    if use_stat:
        dvalid, sub, err, _k = _stat_err_map(ha, valid, det, var_nb)
    else:                                           # 主場：與舊版逐位元相同
        dvalid = valid
        ha_c = np.ascontiguousarray(ha)
        bkg = sep.Background(ha_c, mask=~valid, bw=det.bkg_box_px, bh=det.bkg_box_px, fw=3, fh=3)
        sub = ha_c - bkg
        err = bkg.rms()
    _, seg = sep.extract(sub, det.threshold_sigma, err=err, mask=~dvalid,
                         minarea=det.min_area_px, filter_kernel=_gauss_kernel(15, det.kernel_fwhm_px),
                         deblend_nthresh=32, deblend_cont=0.005, segmentation_map=True)
    return ndi.binary_dilation((seg > 0) & dvalid, iterations=det.dilate_px) & valid

def _detect_claude(ha, white, valid, det):
    """方法 claude：robust-MAD 手寫門檻（Hα 或 白光）+ 連通塊過濾 + 膨脹。回傳源遮罩 bool。"""
    from scipy import ndimage as ndi
    def med_mad(z):                                 # 穩健 (median, σ_MAD)
        v = z[valid]; m = np.median(v); return m, 1.4826 * np.median(np.abs(v - m))
    ha_s = ndi.median_filter(ha, 3); wh_s = ndi.median_filter(white, 3)   # 先 3x3 中值壓單點雜訊
    mh, sh = med_mad(ha_s); mw, sw = med_mad(wh_s)
    src = (((ha_s - mh) > settings.CLAUDE_HALPHA_THRESHOLD_SIGMA * sh) |
           ((wh_s - mw) > settings.CLAUDE_WHITE_THRESHOLD_SIGMA * sw)) & valid
    lab, n = ndi.label(src)                         # 只留夠大的連通塊，丟掉雜訊碎點
    if n:
        big = np.zeros(n + 1, bool)
        big[1:] = ndi.sum(np.ones_like(lab), lab, range(1, n + 1)) >= settings.CLAUDE_MIN_BLOB_PIXELS
        src = big[lab]
    return ndi.binary_dilation(src, iterations=det.dilate_px) & valid

_DETECTORS = ("sep", "claude")

def build_mask(from_cube, method="sep"):
    """在指定 cube 上、用指定方法偵測源，回傳 (src, white, ha, valid, wl)。不存檔。
       偵測參數改由 settings.get_cube_config(from_cube) 提供（每 cube 一組）。
       from_cube ∈ CUBE_NAMES；method ∈ {'sep','claude'}。"""
    cfg = settings.get_cube_config(from_cube)
    hd = fits.open(str(settings.input_cube_path(from_cube)))
    cube = hd["DATA"].data; hdr = hd["DATA"].header
    wl = settings.wavelength_axis(hdr)
    white = np.nansum(cube, axis=0).astype(np.float32)     # 白光：定位最亮點、判斷有效視場
    ha = settings.halpha_narrowband_image(cube, wl)        # 純 Hα 影像（共用 settings 定義）
    var_nb = (_stat_variance_narrowband(hd, wl)
              if (method == "sep" and cfg.detect.use_stat_noise) else None)
    hd.close()
    valid = white != 0
    if method == "sep":
        src = _detect_sep(ha, valid, cfg.detect, var_nb)
    else:
        src = _detect_claude(ha, white, valid, cfg.detect)
    return src, white, ha, valid, wl

def mask(from_cube, method=settings.MASK_METHOD):
    """從 from_cube 偵測源 → 存 source mask + 亮源座標。
       落點依 cfg.promoted：已拔擢→正式 masks/<method>_from-<cube>/；
       未拔擢（NE 新驗證參數）→ diagnostics/<cube>/（不覆蓋已上傳 Drive 的舊 mask）。"""
    settings.ensure_output_dirs()
    cfg = settings.get_cube_config(from_cube)
    src, white, ha, valid, wl = build_mask(from_cube, method)
    out_dir = settings.mask_output_dir(from_cube, method, cfg.promoted)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mask = out_dir / "mask.fits"
    fits.writeto(str(out_mask), src.astype(np.uint8), overwrite=True)

    blank = valid & ~src
    sy, sx = np.unravel_index(np.nanargmax(np.where(valid, white, np.nan)), white.shape)
    np.savez(str(out_dir / "blanks.npz"), wl=wl, sy=sy, sx=sx)   # 亮源座標(供 M3)+波長軸
    tag = "" if cfg.promoted else "（未拔擢，寫 diagnostics）"
    print(f"[step1 mask/{method}/from-{from_cube}] 源 {int(src.sum())} ({100*src.sum()/valid.sum():.0f}%), "
          f"blank {int(blank.sum())}, 亮源@({sy},{sx}) -> {out_dir} {tag}")
    return out_dir


# ============ NE 新舊參數 A/B 驗證（科學閘：清潔度 + 源保存雙指標）============
_OLD_NE_DETECT = settings.DetectParams(bkg_box_px=256, kernel_fwhm_px=6.0, threshold_sigma=2.0,
                                       min_area_px=30, dilate_px=6, use_stat_noise=False)

def validate_ne(from_cube="NEnosky"):
    """開一次 NE cube，建 OLD(全域舊參數) 與 NEW(cfg 新參數) 兩張遮罩，比較：
       清潔度 = 左側假條% (x<100) / 中央空區% (x110-190)；
       源保存 = Hα 暈通量被遮住比例（是否吃掉低表面亮度外暈），並存 A/B 圖 + 徑向剖面。
       NEW 遮罩寫 diagnostics/<cube>/mask.fits（不覆蓋正式 masks/）。"""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cfg = settings.get_cube_config(from_cube)
    hd = fits.open(str(settings.input_cube_path(from_cube)))
    cube = hd["DATA"].data; wl = settings.wavelength_axis(hd["DATA"].header)
    white = np.nansum(cube, axis=0).astype(np.float32)
    ha = settings.halpha_narrowband_image(cube, wl)                                  # 偵測用（寬連續譜）
    haflux = settings.halpha_narrowband_image(cube, wl, settings.HALPHA_FLUX_CONTINUUM)  # 量通量用（窄連續譜）
    var_nb = _stat_variance_narrowband(hd, wl)
    hd.close()
    valid = white != 0

    # 先算一次 STAT 校正 k（供顯示），再用固定 k 建 NEW，確保顯示與實際遮罩用同一個 k
    _, _, _, k = _stat_err_map(ha, valid, cfg.detect, var_nb)
    det_new = replace(cfg.detect, stat_calib_k=k)

    old = _detect_sep(ha, valid, _OLD_NE_DETECT, None)          # 舊：bw256 + bkg.rms()
    new = _detect_sep(ha, valid, det_new, var_nb)              # 新：bw64 + STAT×k

    ny, nx = ha.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    left = valid & (xx < 100)                                   # 左側 IFU 假條
    mid  = valid & (xx >= 110) & (xx < 190)                     # 中央空區
    halo = valid & (xx >= 210) & (yy < 190)                     # 暈（含外暈）像素區
    frac = lambda m, s: 100.0 * (s & m).sum() / max(int(m.sum()), 1)

    # ---- 源保存：Hα 暈「通量」被遮住比例（比像素%更貼近科學：faint wings 不能掉）----
    src_side = valid & (xx >= 100) & (haflux > 0)               # 真源側（排除左假條）、只算正通量
    tot_flux = float(haflux[src_side].sum())
    fluxcap = lambda s: 100.0 * float(haflux[src_side & s].sum()) / tot_flux

    # ---- 徑向剖面：以最亮點為心，各環「被遮通量 / 總通量」，看外暈是否被 NEW 丟掉 ----
    cy, cx = np.unravel_index(np.nanargmax(np.where(valid, white, np.nan)), white.shape)
    rr = np.hypot(yy - cy, xx - cx) * cfg.pixel_scale_arcsec    # arcsec
    edges = np.arange(0, np.nanmax(rr[src_side]) + 3, 3.0)      # 3" 環
    rc, f_old, f_new = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        ring = src_side & (rr >= lo) & (rr < hi)
        tf = float(haflux[ring].sum())
        if tf <= 0:
            continue
        rc.append(0.5 * (lo + hi))
        f_old.append(100.0 * float(haflux[ring & old].sum()) / tf)
        f_new.append(100.0 * float(haflux[ring & new].sum()) / tf)

    m = dict(old_left=frac(left, old), new_left=frac(left, new),
             old_mid=frac(mid, old), new_mid=frac(mid, new),
             old_halo_pix=frac(halo, old), new_halo_pix=frac(halo, new),
             old_flux=fluxcap(old), new_flux=fluxcap(new),
             old_tot=100.0 * old.sum() / valid.sum(), new_tot=100.0 * new.sum() / valid.sum(),
             k=k)

    # ---- 存 NEW 遮罩 + 圖到 diagnostics（不覆蓋 masks/）----
    out_dir = settings.ensure_diagnostics_dir(from_cube)
    fits.writeto(str(out_dir / "mask.fits"), new.astype(np.uint8), overwrite=True)
    np.savez(str(out_dir / "blanks.npz"), wl=wl, sy=cy, sx=cx)

    fig, ax = plt.subplots(1, 3, figsize=(17, 6))
    d = np.where(valid, ha, np.nan); lo, hi = np.nanpercentile(d, [2, 99])
    ax[0].imshow(d, origin="lower", cmap="magma", vmin=lo, vmax=hi); ax[0].set_title("Halpha narrowband")
    ax[1].imshow(old, origin="lower", cmap="gray")
    ax[1].set_title(f"OLD bw256+bkg.rms  L={m['old_left']:.0f}% M={m['old_mid']:.0f}% flux={m['old_flux']:.0f}%")
    ax[2].imshow(new, origin="lower", cmap="gray")
    ax[2].set_title(f"NEW bw{cfg.detect.bkg_box_px}+STATx{m['k']:.2f}  "
                    f"L={m['new_left']:.0f}% M={m['new_mid']:.0f}% flux={m['new_flux']:.0f}%")
    for a in ax:
        a.axvline(100, color="cyan", lw=0.6); a.axvline(190, color="cyan", lw=0.4, ls=":")
    fig.tight_layout(); fig.savefig(str(out_dir / "compare_old_vs_new.png"), dpi=110); plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(rc, f_old, "o-", label="OLD (bw256+bkg.rms)")
    ax2.plot(rc, f_new, "s-", label="NEW (bw64+STAT)")
    ax2.set_xlabel("radius from core [arcsec]"); ax2.set_ylabel("Halpha flux masked in ring [%]")
    ax2.set_title("Source preservation vs radius (faint outer halo)"); ax2.legend(); ax2.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig(str(out_dir / "halo_flux_radial.png"), dpi=110); plt.close(fig2)

    print(f"\n[validate NE / {from_cube}]  (STAT calib k={m['k']:.2f}, "
          f"PSF={cfg.seeing_fwhm_px:.2f}px[{cfg.seeing_source}] -> kernel={cfg.detect.kernel_fwhm_px} "
          f"minarea={cfg.detect.min_area_px} dilate={cfg.detect.dilate_px} bw={cfg.detect.bkg_box_px})")
    print(f"  {'metric':<26}{'OLD':>8}{'NEW':>8}")
    print(f"  {'total masked %':<26}{m['old_tot']:>8.1f}{m['new_tot']:>8.1f}")
    print(f"  {'LEFT strip x<100 %':<26}{m['old_left']:>8.1f}{m['new_left']:>8.1f}   (want low)")
    print(f"  {'MID blank x110-190 %':<26}{m['old_mid']:>8.1f}{m['new_mid']:>8.1f}   (want low)")
    print(f"  {'HALO pixels masked %':<26}{m['old_halo_pix']:>8.1f}{m['new_halo_pix']:>8.1f}")
    print(f"  {'HALO FLUX preserved %':<26}{m['old_flux']:>8.1f}{m['new_flux']:>8.1f}   (want high/kept)")
    print(f"  -> figures + NEW mask.fits @ {out_dir}")
    return m

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in settings.CUBE_NAMES:
        sys.exit("用法: step1_mask.py <nosky|wsky|NEnosky|NEwsky> [sep|claude]")
    from_cube = sys.argv[1]
    method = sys.argv[2] if len(sys.argv) > 2 else settings.MASK_METHOD
    if method not in _DETECTORS:
        sys.exit(f"未知方法 {method}；可用: {tuple(_DETECTORS)}")
    # NE + sep：跑新舊 A/B 驗證（也會存 NEW mask 到 diagnostics）；其餘走一般 mask()
    if from_cube.startswith("NE") and method == "sep":
        validate_ne(from_cube)
    else:
        mask(from_cube, method)
