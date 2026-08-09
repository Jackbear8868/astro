"""把多顆曝光的 cube 加權合併到一個共同格點。

為什麼要自己合併:天空隨時間變。ESO 的合併品裡,每一個 spaxel 是不同曝光子集
的加權平均(dither + 90 度旋轉造成),於是天空的時間變化被寫成假的空間圖樣,
而我們的 sky basis 學的正是「跨 spaxel 的變化」—— 等於讓它去學那個假圖樣。
正確的順序是:每顆曝光各自學天空、各自扣掉,扣乾淨之後才合併。這支程式做的
就是最後那一步,而且不在乎輸入是扣過天空還是沒扣,所以同一支程式可以先拿
「沒扣天空的原始曝光」去和官方合併品比對,驗證幾何與加權對不對。

這批資料讓合併變得比一般情況簡單很多:

    所有 cube 的 CD 矩陣都是 diag(-5.5556e-05, +5.5556e-05) 度、非對角元為 0
    -> ESO pipeline 已經把每顆曝光重取樣到「北上東左、0.2 arcsec/px」的天球
       格點,儀器的 90 度旋轉已經被吸收掉。我們只需要處理「平移」。

    四顆曝光相對合併格點的波長偏移 < 0.001 個通道
    -> 波長軸完全不用內插。

平移怎麼做,由 --subpixel 決定,而**預設是 integer**。理由是實測出來的
(step5b_stat_check):

    雙線性內插是一次卷積,等效核寬 σ = √[f(1−f)],這批資料的小數位移
    f = 0.12/0.33/0.01 (x)、0.03/0.74/0.65 (y),對應 σ = 0.32-0.48 px;
    而位移直接四捨五入,對位誤差最多 0.35 px、rms 約 0.2 px。

    也就是說**取整的位置誤差比內插造成的模糊還小**。取整兩邊都贏:影像更銳利
    (PSF 4.03 px vs 4.14 px,seeing FWHM 約 4 px),而且完全不產生新的像素間
    相關 —— 純陣列切片。代價只有 <= 0.35 px = 0.07 arcsec 的絕對天測誤差,
    對「學天空、扣天空」無關緊要。

    實測佐證:雙線性版本的相鄰像素相關 rho(1) 從輸入的 0.19/0.25 被推高到
    0.31/0.39,大尺度的真實雜訊變成 STAT 說的 4.5 倍(輸入本來是 2.4 倍)。

變異數的傳遞。取整時是純搬移,變異數原樣搬過去。雙線性時是加權和 v = Σ wₖ xₖ,

    Var(v) = Σ wₖ² Var(xₖ)          <- 權重要平方

而不是 Σ wₖ Var(xₖ)。因為整幅影像用的是同一個平移量,四個權重是常數,所以這
件事可以直接算對,不需要近似。(注意:這只處理輸入各像素獨立的情形;內插本身
會讓輸出的相鄰像素相關,那個效應用 step5b_stat_check 去實測。)

宇宙線剔除(--reject)是為了對齊官方合併品的行為。官方 header 寫著
crtype=median、crsigma=10 —— 它有跨曝光剔除離群值,我們原本沒有。宇宙線永遠
是正的,少了這一步會系統性偏高;實測(step5b_bias_check)量到全場一致的
+0.1 sigma 正偏移,分布也有明顯的正尾巴。

    # 驗證:用官方合併品用過的那三顆,不扣天空,和官方品比
    conda run -n astro python src/skymodel/step5b_merge_cubes.py \\
        --inputs data/NEpointing_exps/Haro11_NEpointing_EXP{1,2,3}_wsky.fits \\
        --ref data/Haro11_NEpointing_wsky.fits --weight uniform \\
        --compare data/Haro11_NEpointing_wsky.fits
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT    = Path(__file__).resolve().parents[2]
STEP05B = ROOT / "results/skymodel/step05b"


def grid_offset(w_in, w_ref, nx, ny):
    """回傳 (ox, oy):目標格點的像素 (x, y) 對應到輸入的 (x + ox, y + oy)。

    同時檢查這個平移是不是真的與位置無關 —— CD 矩陣相同並不保證,兩個 cube 的
    CRVAL 不同會讓切平面投影帶進一個很小的位置相依項。在五個取樣點上量,回傳
    最大散布;散布大到不能忽略時,單純的平移就不夠,必須做完整的重投影。
    """
    xs = np.array([0, nx - 1, 0, nx - 1, nx // 2], float)
    ys = np.array([0, 0, ny - 1, ny - 1, ny // 2], float)
    xi, yi = w_in.world_to_pixel(w_ref.pixel_to_world(xs, ys))
    ox, oy = xi - xs, yi - ys
    return float(ox.mean()), float(oy.mean()), float(max(np.ptp(ox), np.ptp(oy)))


def white_image(path, w0, w1, step):
    """一張用來對齊的白光影像:在 [w0, w1) 的通道裡每 step 個取一個,取中位數。

    取中位數而不是平均 —— 天光線和宇宙線殘留會把平均拉走,而對齊只需要
    「哪裡有東西」。抽樣是為了不把整個 3 GB 的 cube 讀進記憶體;對齊用的是
    源的位置,抽樣不影響它。
    """
    with fits.open(path, memmap=True) as h:
        hd = h["DATA"].header
        j0 = int(round((w0 - hd["CRVAL3"]) / hd["CD3_3"]))
        j1 = int(round((w1 - hd["CRVAL3"]) / hd["CD3_3"]))
        j0, j1 = max(j0, 0), min(j1, hd["NAXIS3"])
        d = np.asarray(h["DATA"].data[j0:j1:step], np.float64)
    with np.errstate(invalid="ignore"):
        # NaN 原樣保留 —— 「哪裡沒有資料」是對齊時必須知道的資訊,
        # 填 0 會讓缺值區看起來像「一片很暗的天空」而參與比對。
        return np.nanmedian(d, axis=0)


def measure_offset(R, E):
    """從影像量出位移:回傳 (ox, oy),使得 R[y, x] 對應 E[y + oy, x + ox]。

    為什麼不能用 header 的 WCS:單顆曝光的 muse_scipost 產物沒有套過
    OFFSET_LIST,絕對天測差了 9-12 arcsec(用 Haro 11 的已知位置量到的);
    合併品套過,所以兩者的 WCS 不在同一個框架裡。曝光之間的相對位移兩邊
    一致,錯的是那個共同的絕對偏移 —— 但那個偏移足以讓合併整個失敗。

    做法是 FFT 互相關取峰值,再用拋物線內插細化到次像素。
    互相關定理:(R corr E)[u, v] = sum R[y,x] E[y+v, x+u] = IFFT(conj(F R) * F E)

    互相關必須**正規化**,這是一個實際踩過的坑。原始的白光影像是處處為正的
    (天空連續譜是一個巨大的常數底盤),未正規化的 sum R·E 會在「重疊像素最多」
    的位移處最大,而不是在「結構對齊」處最大。三顆曝光的視場形狀還不一樣
    (315x306 vs 306x315),於是每顆的偏差量都不同 —— 實測 EXP2 因此偏了 2.0 px,
    源的流量掉了 15%。

    正規化兩件事一起做:
        ① 兩張圖各自減掉自己的均值,讓底盤不再貢獻
        ② 除以該位移下的重疊像素數 N(u,v),而 N 用同樣的 FFT 對有效遮罩
           相關一次就得到
    只在 N 夠大的地方找峰值 —— 重疊很少時分母小,比值會亂跳。
    """
    mR, mE = np.isfinite(R), np.isfinite(E)
    A = np.where(mR, R - np.nanmean(R), 0.0)
    B = np.where(mE, E - np.nanmean(E), 0.0)

    ny = R.shape[0] + E.shape[0]
    nx = R.shape[1] + E.shape[1]          # 補零到兩者之和,避免循環卷積繞回來

    def pad(a):
        P = np.zeros((ny, nx)); P[:a.shape[0], :a.shape[1]] = a; return P

    F = lambda a: np.fft.rfft2(pad(a))
    C = np.fft.irfft2(np.conj(F(A)) * F(B), (ny, nx))
    N = np.fft.irfft2(np.conj(F(mR.astype(float))) * F(mE.astype(float)), (ny, nx))

    good = N > 0.3 * N.max()
    C = np.where(good, C / np.maximum(N, 1.0), -np.inf)

    ky, kx = np.unravel_index(int(np.argmax(C)), C.shape)

    def refine(c0, cm, cp):
        """拋物線頂點相對於中央樣本的位移。分母為 0 表示三點共線,退回整數。
        鄰點落在重疊不足的區域(值為 -inf)時同樣退回整數。"""
        if not (np.isfinite(cm) and np.isfinite(cp) and np.isfinite(c0)):
            return 0.0
        den = cm - 2 * c0 + cp
        return 0.0 if den == 0 else float(np.clip(0.5 * (cm - cp) / den, -0.5, 0.5))

    # 拋物線細化對整數位移是精確的,對次像素位移會系統性低估約 25%
    # (實測:真值 0.40 px 量到 0.34 px),也就是約 0.06 px = 0.012 arcsec 的誤差。
    dy = refine(C[ky, kx], C[(ky - 1) % ny, kx], C[(ky + 1) % ny, kx])
    dx = refine(C[ky, kx], C[ky, (kx - 1) % nx], C[ky, (kx + 1) % nx])
    oy = (ky if ky < ny // 2 else ky - ny) + dy
    ox = (kx if kx < nx // 2 else kx - nx) + dx
    return float(ox), float(oy)


def _take(a, dy, dx, NY, NX):
    """整數平移:out[:, y, x] = a[:, y + dy, x + dx],範圍外填 NaN。

    純切片,沒有任何內插 —— 平移量的整數部分不該付出插值的代價。
    """
    nz, ny, nx = a.shape
    out = np.full((nz, NY, NX), np.nan, np.float64)
    sy0, sy1 = max(0, dy), min(ny, NY + dy)
    sx0, sx1 = max(0, dx), min(nx, NX + dx)
    if sy0 < sy1 and sx0 < sx1:
        out[:, sy0 - dy:sy1 - dy, sx0 - dx:sx1 - dx] = a[:, sy0:sy1, sx0:sx1]
    return out


def shift_chunk(d, v, ox, oy, NY, NX, mode="integer"):
    """把一段波長的 (資料, 變異數) 平移到目標格點。

    mode="integer":位移四捨五入成整數,純陣列切片。沒有內插,所以不產生任何
    新的像素間相關,變異數原樣搬過去。
    注意四捨五入用的是量出來的位移,而拋物線細化對次像素位移會系統性低估
    約 25%(見 measure_offset)。這批資料的小數部分是 0.12/0.33/0.01(x)、
    0.03/0.74/0.65(y),就算補回那 25% 也不會跨過 0.5 的分界,取整是穩的。

    mode="bilinear":四個角的權重是常數(整幅同一個平移量),所以資料用
    Σ wₖ xₖ、變異數用 Σ wₖ² vₖ,兩者都是精確的。
    只有四個貢獻像素全部有效的目標像素才算有效 —— 邊界上做部分覆蓋的重normalise
    會在拼接處造成系統性的偏差,寧可少一圈像素。
    """
    if mode == "integer":
        dk = _take(d, int(round(oy)), int(round(ox)), NY, NX)
        vk = _take(v, int(round(oy)), int(round(ox)), NY, NX)
        good = np.isfinite(dk) & np.isfinite(vk) & (vk > 0)
        return np.where(good, dk, np.nan), np.where(good, vk, np.nan)

    iy, ix = int(np.floor(oy)), int(np.floor(ox))
    fy, fx = oy - iy, ox - ix

    w  = [((1 - fy) * (1 - fx), iy,     ix),
          ((1 - fy) * fx,       iy,     ix + 1),
          (fy * (1 - fx),       iy + 1, ix),
          (fy * fx,             iy + 1, ix + 1)]

    dsum = np.zeros((d.shape[0], NY, NX))
    vsum = np.zeros_like(dsum)
    good = np.ones_like(dsum, bool)
    for wk, dy, dx in w:
        dk = _take(d, dy, dx, NY, NX)
        vk = _take(v, dy, dx, NY, NX)
        ok = np.isfinite(dk) & np.isfinite(vk) & (vk > 0)
        good &= ok
        if wk == 0.0:                      # 權重為 0 的角不必累加,但仍要求它有效
            continue
        dsum += wk * np.where(ok, dk, 0.0)
        vsum += wk ** 2 * np.where(ok, vk, 0.0)

    return (np.where(good, dsum, np.nan),
            np.where(good, vsum, np.nan))


def outliers(D, V, ok, crsigma):
    """跨曝光的離群值(宇宙線)。D, V, ok 的形狀都是 (n_exp, ...)。

    只有三顆(以上)都在的地方才判得出誰是離群值 —— 兩顆的話中位數就是兩者的
    平均,偏離量必然相等,無從選擇。

    sigma 用 sqrt(STAT) 而不是各曝光之間的 MAD:三個值的 MAD 必然等於兩個
    偏離量裡較小的那一個(中位數自己的偏離是 0),雜訊極大而且會偶然接近 0,
    拿它當分母會誤殺。門檻是 10 sigma,宇宙線比雜訊高好幾個數量級,
    分母估得保守一點完全不影響它抓不抓得到。
    """
    n_ok = ok.sum(axis=0)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(np.where(ok, D, np.nan), axis=0)
        return ok & (n_ok >= 3) & (np.abs(D - med) >
                                   crsigma * np.sqrt(np.where(V > 0, V, np.nan)))


def white_light(d):
    """白光影像:沿波長取中位數。用中位數而不是平均 —— 天光線和宇宙線殘留
    會把平均拉走,而我們只想要一張「哪裡有東西」的圖。"""
    with np.errstate(invalid="ignore"):
        return np.nanmedian(d, axis=0)


def main():
    ap = argparse.ArgumentParser(description="把多顆曝光的 cube 合併到共同格點")
    ap.add_argument("--inputs", nargs="+", required=True, help="要合併的 cube")
    ap.add_argument("--ref", required=True,
                    help="提供目標格點的 cube。用官方合併品,我們的 seg 圖與遮罩"
                         "都在那個格點上,而且可以逐 voxel 比對")
    ap.add_argument("--out", default=None)
    ap.add_argument("--weight", choices=["ivar", "uniform"], default="ivar",
                    help="ivar = 1/variance(統計上最佳);uniform = 等權重"
                         "(ESO 用 weight=exptime,而這批曝光都是 700 s,等價)")
    ap.add_argument("--align", choices=["data", "wcs"], default="data",
                    help="位移從哪裡來。data = 白光影像互相關(預設);"
                         "wcs = 相信 header —— 對這批資料是錯的,單顆曝光沒套過 "
                         "OFFSET_LIST,絕對天測差 9-12 arcsec,只留著做對照")
    ap.add_argument("--align-range", type=float, nargs=2, default=[5000.0, 7000.0],
                    metavar=("LO", "HI"), help="做對齊用的波長範圍 (A)")
    ap.add_argument("--align-step", type=int, default=5,
                    help="對齊影像每幾個通道取一個")
    ap.add_argument("--subpixel", choices=["integer", "bilinear"], default="integer",
                    help="次像素位移怎麼處理。integer = 四捨五入成整數、純切片"
                         "(預設,實測較優,見檔頭);bilinear = 雙線性內插")
    ap.add_argument("--reject", choices=["median", "none"], default="median",
                    help="跨曝光的離群值(宇宙線)剔除。官方合併品用的是"
                         "crtype=median、crsigma=10,預設對齊它")
    ap.add_argument("--crsigma", type=float, default=10.0,
                    help="剔除門檻,單位是該 voxel 的 sqrt(STAT)")
    ap.add_argument("--compare", default=None,
                    help="合併完和這個 cube 比對,印出白光圖與逐通道的統計")
    ap.add_argument("--chunk", type=int, default=100, help="一次處理幾個波長平面")
    args = ap.parse_args()

    ref_h = fits.getheader(args.ref, "DATA")
    NX, NY, NZ = ref_h["NAXIS1"], ref_h["NAXIS2"], ref_h["NAXIS3"]
    w_ref = WCS(ref_h).celestial

    print(f"目標格點 {NX} x {NY} x {NZ}   (來自 {Path(args.ref).name})")
    R = (white_image(args.ref, *args.align_range, args.align_step)
         if args.align == "data" else None)

    print(f"{'exposure':<44}{'nx':>6}{'ny':>6}{'nz':>6}"
          f"{'ox':>10}{'oy':>10}{'WCS ox':>10}{'WCS oy':>10}{'dlam(ch)':>10}")

    offs = []
    for p in args.inputs:
        h = fits.getheader(p, "DATA")
        ox_w, oy_w, spread = grid_offset(WCS(h).celestial, w_ref,
                                         ref_h["NAXIS1"], ref_h["NAXIS2"])
        if args.align == "data":
            ox, oy = measure_offset(R, white_image(p, *args.align_range,
                                                   args.align_step))
        else:
            ox, oy = ox_w, oy_w
        dl = (h["CRVAL3"] - ref_h["CRVAL3"]) / ref_h["CD3_3"]
        offs.append((p, ox, oy, h["NAXIS3"], dl))
        print(f"{Path(p).name:<44}{h['NAXIS1']:>6}{h['NAXIS2']:>6}{h['NAXIS3']:>6}"
              f"{ox:>10.3f}{oy:>10.3f}{ox_w:>10.3f}{oy_w:>10.3f}{dl:>10.4f}")
        if spread > 0.05:
            raise SystemExit(f"{Path(p).name}: WCS 的位移隨位置變 {spread:.3f} px,"
                             "單純平移不夠,需要完整重投影")
        if abs(dl) > 0.05:
            raise SystemExit(f"{Path(p).name}: 波長偏移 {dl:.3f} 通道,需要波長內插")

    # 波長在外層、曝光在內層。剔除宇宙線需要同一個 voxel 的所有曝光同時在手上,
    # 所以不能像「一顆一顆累加」那樣串流。輸出直接存 float32 —— float64 的
    # 全 cube 累加器要 10 GB,而我們只需要 float32 的精度。
    data = np.full((NZ, NY, NX), np.nan, np.float32)
    var  = np.full((NZ, NY, NX), np.nan, np.float32)
    nexp = np.zeros((NZ, NY, NX), np.uint8)
    n_rej = 0

    print(f"\n次像素:{args.subpixel}   離群剔除:{args.reject}"
          + (f" ({args.crsigma:.0f} sigma)" if args.reject != "none" else ""))
    hs = [fits.open(p, memmap=True) for p, *_ in offs]
    try:
        for j in range(0, NZ, args.chunk):
            k = min(j + args.chunk, NZ)
            D, V = [], []
            for (p, ox, oy, nz, _), h in zip(offs, hs):
                e = min(k, nz)
                d = np.asarray(h["DATA"].data[j:e], np.float64)
                v = np.asarray(h["STAT"].data[j:e], np.float64)
                ds, vs = shift_chunk(d, v, ox, oy, NY, NX, args.subpixel)
                if e < k:                       # 這顆曝光的波長軸比較短,補缺值
                    pad = np.full((k - e, NY, NX), np.nan)
                    ds, vs = np.vstack([ds, pad]), np.vstack([vs, pad])
                D.append(ds); V.append(vs)
            D, V = np.stack(D), np.stack(V)     # (n_exp, nch, NY, NX)
            ok = np.isfinite(D) & np.isfinite(V) & (V > 0)

            if args.reject != "none":
                bad = outliers(D, V, ok, args.crsigma)
                n_rej += int(bad.sum())
                ok &= ~bad

            w = (np.where(ok, 1.0 / np.where(V > 0, V, 1.0), 0.0)
                 if args.weight == "ivar" else ok.astype(float))
            den  = w.sum(axis=0)
            num  = (w * np.where(ok, D, 0.0)).sum(axis=0)
            vnum = (w ** 2 * np.where(ok, V, 0.0)).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                data[j:k] = np.where(den > 0, num / den, np.nan)
                var[j:k]  = np.where(den > 0, vnum / den ** 2, np.nan)
            nexp[j:k] = ok.sum(axis=0)
            print(f"   {k:>5}/{NZ}", end="\r", flush=True)
    finally:
        for h in hs:
            h.close()
    print(f"   {NZ:>5}/{NZ}  done")
    if args.reject != "none":
        tot = float(nexp.sum()) + n_rej
        print(f"剔除的離群 voxel {n_rej:,} / {tot:,.0f} 個曝光樣本 "
              f"({100 * n_rej / max(tot, 1):.4f}%)")

    STEP05B.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else STEP05B / "merged.fits"

    # header 直接沿用 ref 的 —— WCS、BUNIT、單位全部要和目標格點一致。
    hdul = fits.HDUList([
        fits.PrimaryHDU(header=fits.getheader(args.ref, 0)),
        fits.ImageHDU(data, header=ref_h, name="DATA"),
        fits.ImageHDU(var, header=fits.getheader(args.ref, "STAT"), name="STAT"),
        # 有效曝光數:邊緣與壞像素讓每個 voxel 的深度不同,下游要用它才能
        # 正確判讀雜訊,不能假設整個 cube 一樣深。
        fits.ImageHDU(nexp, header=ref_h, name="NEXP"),
    ])
    hdul[0].header["HISTORY"] = (f"merged {len(offs)} cubes, weight={args.weight},"
                                 f" subpixel={args.subpixel}, reject={args.reject}"
                                 + (f"@{args.crsigma:g}sig"
                                    if args.reject != "none" else ""))
    for p, ox, oy, *_ in offs:
        hdul[0].header["HISTORY"] = (f"  input {Path(p).name} "
                                     f"shift ({ox:+.3f}, {oy:+.3f})")
    hdul.writeto(out, overwrite=True)

    cov = nexp.astype(np.float64).mean(axis=(1, 2))
    print(f"\n有效曝光數:全 cube 平均 {cov.mean():.2f}   "
          f"逐 voxel 分布 " + "  ".join(
              f"{n}:{100 * (nexp == n).mean():.1f}%" for n in range(len(offs) + 1)))
    print(f"saved -> {out}  ({out.stat().st_size / 1e9:.2f} GB)")

    if args.compare:
        print(f"\n{'=' * 72}\n和 {Path(args.compare).name} 比對")
        with fits.open(args.compare, memmap=True) as h:
            ref_d = np.asarray(h["DATA"].data, np.float64)
            ref_v = np.asarray(h["STAT"].data, np.float64)
        m = np.isfinite(data) & np.isfinite(ref_d) & (nexp == len(offs))
        print(f"兩邊都有效且滿曝光的 voxel:{100 * m.mean():.1f}%")

        r = (data[m] / ref_d[m])
        r = r[np.isfinite(r) & (np.abs(ref_d[m]) > 1e-3)]
        print(f"流量比 ours/ESO   中位 {np.median(r):.4f}   "
              f"16-84% [{np.percentile(r, 16):.4f}, {np.percentile(r, 84):.4f}]")

        d_ours, d_eso = white_light(np.where(nexp == len(offs), data, np.nan)), \
                        white_light(np.where(nexp == len(offs), ref_d, np.nan))
        g = np.isfinite(d_ours) & np.isfinite(d_eso)
        print(f"白光圖   ours 中位 {np.nanmedian(d_ours[g]):.4f}   "
              f"ESO 中位 {np.nanmedian(d_eso[g]):.4f}   "
              f"相關係數 {np.corrcoef(d_ours[g], d_eso[g])[0, 1]:.5f}")

        # STAT 的比值直接告訴我們「重取樣兩次 vs 一次」的代價
        rv = (var[m] / ref_v[m])
        rv = rv[np.isfinite(rv)]
        print(f"STAT 比 ours/ESO  中位 {np.median(rv):.4f}   "
              f"(< 1 代表我們的變異數比 ESO 小 —— 內插讓相鄰像素相關,"
              f"獨立假設下的傳遞會低估)")

        diff = (data[m] - ref_d[m]) / np.sqrt(ref_v[m])
        print(f"逐 voxel 差 (ours-ESO)/sigma_ESO   中位 {np.median(diff):+.4f}   "
              f"散布 {np.std(diff):.4f}")


if __name__ == "__main__":
    main()
