"""從 blank spaxels 學出天空模型的兩個成分:連續譜 C_sky 與 K 條天光線 basis。

輸出供 step4 的聯合擬合使用。天光線 basis 的分解方法可替換。
"""
from pathlib import Path
import numpy as np
from astropy.io import fits
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.utils.extmath import randomized_svd

from scipy import ndimage

from utils import (estimate_continuum, half_field_mask, main_source_mask,
                   main_source_group)
import argparse
import time

SEED       = 0           # 所有分解共用的亂數種子,確保 basis 可重現
K          = 25          # basis 條數 = 天空模型的自由度
WINDOW     = 300         # 連續譜 running median 視窗 (px)
THRESHOLDS = (1, 2)      # 線偵測門檻 (正, 負)
MAX_ITER   = 5           # estimate_continuum 迭代上限
CLIP_SIGMA = 30          # mean_sky 的 sigma-clip 門檻,單位是穩健散布 sg。
                         # 目標只是擋掉壞像素等級的離群值,不是修剪跨 spaxel 的
                         # 真實變化,所以門檻要遠高於後者的自然幅度
METHODS    = ["pca", "svd", "nmf", "rpca"]

def soft_threshold(X, tau):
    """逐元素軟閾值:sign(x) · max(|x| − tau, 0)。"""
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def robust_pca(X, K=10, n_iter=100, tol=1e-7, rho=1.5, rank_buffer=5, seed=0):
    """Principal Component Pursuit (Candes, Li, Ma & Wright 2011)。

    把矩陣分解成 X = L + S:
      L  低秩,所有 blank spaxel 共有的天光線結構     ← 我們要的
      S  稀疏,宇宙射線、壞像素、blank 區裡未偵測到的暗源

    以 inexact ALM 求解;每步的 SVD 用 randomized_svd 只取前 K+rank_buffer
    個奇異值,因為最後只需要前 K 條。

    Returns
    -------
    basis : ndarray, shape (K, n_features)
        L 的前 K 個右奇異向量。
    S : ndarray, shape (m, n)
        稀疏成分。用來診斷 RPCA 到底分離出了什麼:
        S 裡若出現源光譜(連續譜 + 紅移發射線),代表它成功抓出未偵測的暗源;
        若在天光線波長暴衝,代表 lambda 太小、天光線被誤判成 outlier。
    """
    m, n = X.shape
    lam = 1.0 / np.sqrt(max(m, n))                 # 論文的理論值,不是調參

    norm_F = np.linalg.norm(X)
    norm_2 = randomized_svd(X, n_components=1, random_state=seed)[1][0]

    Y  = X / float(max(norm_2, np.abs(X).max() / lam))    # 對偶變數初值(論文建議)
    mu = 1.25 / norm_2
    mu_max = mu * 1e7

    S = np.zeros_like(X)
    r = K + rank_buffer

    for it in range(n_iter):
        # --- L 步:對「奇異值」做軟閾值 ---
        U, sv, Vt = randomized_svd(X - S + Y / mu, n_components=r, random_state=seed)
        sv_t = np.maximum(sv - 1.0 / mu, 0.0)
        keep = sv_t > 0
        L = (U[:, keep] * sv_t[keep]) @ Vt[keep]

        # --- S 步:對「元素」做軟閾值 ---
        S = soft_threshold(X - L + Y / mu, lam / mu)

        # --- 對偶變數與懲罰係數 ---
        R   = X - L - S
        Y  += mu * R
        mu  = min(mu * rho, mu_max)

        err = np.linalg.norm(R) / norm_F
        if (it + 1) % 10 == 0:
            print(f"    PCP {it+1:3d}  rank(L)={int(keep.sum()):2d}  "
                  f"S非零={100*np.mean(S != 0):5.2f}%  err={err:.2e}", flush=True)
        if err < tol:
            print(f"    PCP converged at iter {it+1}", flush=True)
            break

    basis = randomized_svd(L, n_components=K, random_state=seed)[2]
    return basis, S

def zap_k(var, nsigma=5):
    """ZAP 的成分數判準。照抄 libs/zap/zap/zap.py:926 的 `_compute_deriv`。

    回傳 (K, deriv, mn1, std1),後三個是畫圖用的中間量。

    想法:把特徵值曲線(降冪)的一階差分看成「每多一條成分,還能再解釋掉多少
    變異數」。曲線一開始陡降 —— 那些成分描述的是天光線殘差,是我們要的;
    降幅逐漸趨於**線性**(二階導數歸零)之後,再往下就只是在移除雜訊與天體
    訊號。所以要找的是「降幅第一次回到平坦區水準」的那個位置。

        ① 只看前 25% 的成分 —— 後面早就進入平坦區,納進來只會稀釋統計
        ② deriv = diff(var[:npix])
        ③ 平坦區的基準取 deriv 的後 85%(跳過最前面 15% 的陡降段)
               mn1 = mean(deriv[ind:])   std1 = nsigma * std(deriv[ind:])
        ④ K = 第一個滿足 deriv >= mn1 - std1 的位置
           也就是降幅第一次不再顯著陡於平坦區

    nsigma=5 是 ZAP 的預設值,不是我們調的。門檻越鬆(nsigma 越大)K 越小。

    注意 ZAP 是**逐波長區段**各自做這件事(它把光譜切成數段,每段自己選 K),
    我們是全波段一組,所以兩邊的數字不能互相引用。
    """
    npix  = int(0.25 * var.shape[0])
    deriv = np.diff(var[:npix])
    ind   = int(0.15 * deriv.size)
    mn1   = deriv[ind:].mean()
    std1  = deriv[ind:].std() * nsigma
    # 第一個元素補 False:deriv[i] 描述的是「從第 i 條到第 i+1 條」的降幅,
    # 所以位置要往後挪一格才對得上成分編號。
    hit   = np.flatnonzero(np.append([False], deriv >= (mn1 - std1)))
    return (int(hit[0]) if hit.size else -1), deriv, mn1, std1


def learn_sky_basis(residual, K=10, method="pca"):
    """從 blank spaxels 的殘差學出 K 條天光線 basis。

    所有方法都回傳 (K, nz),也就是設計矩陣裡固定 K 個自由參數,
    這樣不同方法的 chi-square 才有可比性。

    Parameters
    ----------
    residual : ndarray, shape (nz, n_blank)
    K : int
    method : {"pca", "svd", "nmf", "rpca"}

    Returns
    -------
    basis : ndarray, shape (K, nz)
        K 條天光線 basis。列的順序即 step4 中係數的順序。
    sparse : ndarray or None
        只有 rpca 會回傳稀疏成分 S,shape (n_blank, nz);其餘方法為 None。
        呼叫端一律用 `basis, S = ...`,不需要 S 時寫 `basis, _ = ...`。
    """
    X = np.nan_to_num(residual.T).astype(np.float32)     # (n_blank, nz)

    # random_state=SEED 是必要的,不是保險:三者的預設演算法都是隨機化的
    # (TruncatedSVD/PCA 走 randomized SVD,NMF 的 cd solver 也有隨機成分),
    # 不指定的話每次跑出來的 basis 都不同,下游結果無法追溯。
    if method == "pca":
        p = PCA(n_components=K - 1, random_state=SEED).fit(X)
        return np.vstack([p.mean_[None, :], p.components_]), None

    if method == "svd":
        return TruncatedSVD(n_components=K, random_state=SEED).fit(X).components_, None

    if method == "nmf":
        return NMF(n_components=K, init="nndsvda", max_iter=300,
                   random_state=SEED).fit(np.clip(X, 0, None)).components_, None

    if method == "rpca":
        return robust_pca(X, K=K, seed=SEED)            # (basis, S)

    raise ValueError(f"unknown method: {method}")

ROOT = Path(__file__).resolve().parents[2]
# 預設的工作區與 cube。--work / --cube 可以換掉,讓同一支程式跑不同的 pointing:
# 每個 cube 一個工作區資料夾,底下的 step01/step02/... 結構完全相同。
WORK_DEFAULT = ROOT / "results/skymodel"
CUBE_DEFAULT = ROOT / "data/Haro11_NEpointing_wsky.fits"



def main():
    ap = argparse.ArgumentParser(description="從 blank spaxels 學天空連續譜與天光線 basis")
    ap.add_argument("--methods", nargs="+", default=["pca", "svd", "nmf"],
                    choices=METHODS, help="要跑哪些分解方法")
    ap.add_argument("-K", type=int, default=K, help="basis 條數")
    ap.add_argument("--xlim", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="只用這個 x 範圍(像素,含 LO 不含 HI)的 blank spaxel 學天空。"
                         "用途:主源的延展暈會滲進附近的 blank 樣本,限制在遠離它的"
                         "一條裡,樣本變少但更乾淨。代價是天空的空間變化只由視場的"
                         "一部分決定")
    ap.add_argument("--ylim", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="同 --xlim,但限制 y")
    ap.add_argument("--mask-half", action="store_true",
                    help="自動遮掉主源所在的那半邊視場(見 utils.half_field_mask)。"
                         "主源偏心時沿偏得較嚴重的那一軸切一半;主源在視場中央時"
                         "改挖掉一個置中矩形、留外圈。用途和 --xlim 相同,但界線由"
                         "資料算出,不必逐顆手給")
    ap.add_argument("--main-id", type=int, default=None,
                    help="主源的 segmentation ID。省略 = 自動取面積最大的源。"
                         "不要假設它是 1 —— SExtractor 的編號按偵測順序給")
    ap.add_argument("--r-far-src", type=float, default=None,
                    help="排除離**主源以外**的源這麼近(px)的 blank 格,避免小源的 "
                         "PSF 翼把源的光帶進天空模型的形狀裡。主源不歸它管 —— "
                         "主源用 --xlim / --ylim / --mask-half 排除。省略 = 不排除")
    ap.add_argument("--exclude-box", type=int, nargs=4, default=None,
                    metavar=("Y0", "Y1", "X0", "X1"),
                    help="這個框裡的 blank 不當天空的訓練樣本(含端點)。--xlim/--ylim "
                         "只能切邊,做不到「挖中間、留外圈」。用途:鑲嵌視場裡曝光數"
                         "不足的高雜訊區塊。框內的 spaxel 仍然會被扣天空,只是不參與"
                         "訓練")
    ap.add_argument("--seg", default=None,
                    help="用哪一份 segmentation 界定 blank。預設是 SExtractor 的 "
                         "step01/seg.fits;指到 step1b 產生的 seg_dil{r}.fits 就是"
                         "「把源外圍漏出來的光排除在天空樣本外」的版本")
    ap.add_argument("--work", default=str(WORK_DEFAULT),
                    help="這顆 cube 的工作區(底下有 step01/step02/...)。"
                         "一個 pointing 一個工作區,結構相同、彼此不干擾")
    ap.add_argument("--cube", default=str(CUBE_DEFAULT),
                    help="要學天空的 cube。**必須是含天空的 wsky** —— 已經被 ESO "
                         "扣過天空的 nosky 沒有天空可學,只會灌雜訊")
    ap.add_argument("--out", default=None,
                    help="輸出目錄。省略時寫到 {work}/step03(會覆蓋);做實驗時務必指定")
    args = ap.parse_args()

    work   = Path(args.work)
    STEP01 = work / "step01"
    out_dir = Path(args.out) if args.out else work / "step03"
    out_dir.mkdir(parents=True, exist_ok=True)
    WSKY = Path(args.cube)
    print(f"工作區 {work}   cube {WSKY.name}")

    white  = fits.getdata(STEP01 / "whitelight.fits")
    seg_f  = Path(args.seg) if args.seg else STEP01 / "seg.fits"
    seg    = fits.getdata(seg_f)
    print(f"segmentation: {seg_f.name}  源 spaxel {int((seg > 0).sum()):,}")

    valid_mask = white != 0
    blank_mask = valid_mask & ~((seg > 0) & valid_mask)
    n_all = int(blank_mask.sum())
    if args.xlim or args.ylim:
        yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
        if args.xlim:
            blank_mask &= (xx >= args.xlim[0]) & (xx < args.xlim[1])
        if args.ylim:
            blank_mask &= (yy >= args.ylim[0]) & (yy < args.ylim[1])
        print(f"空間限制 x={args.xlim} y={args.ylim}:"
              f"blank {n_all:,} -> {int(blank_mask.sum()):,}"
              f" ({100 * blank_mask.sum() / max(n_all, 1):.1f}%)")

    if args.mask_half:
        # 主源用「最亮像素所在的那一整團」,不是單一 ID —— deblender 會把
        # Haro 11 拆成數塊,選一塊會遮錯半邊(見 utils.main_source_group)。
        mg, mids, mk = main_source_group(seg, white)
        print(f"主源(最亮像素 y={mk[0]}, x={mk[1]} 所在的整團):"
              f"{len(mids)} 個 ID {mids[:8]}{'...' if len(mids) > 8 else ''}"
              f",共 {int(mg.sum()):,} px")
        half, txt = half_field_mask(seg, valid_mask, args.main_id, main=mg)
        n0 = int(blank_mask.sum()); blank_mask &= half
        print(f"--mask-half:{txt}")
        print(f"            blank {n0:,} -> {int(blank_mask.sum()):,}"
              f" ({100 * blank_mask.sum() / max(n0, 1):.1f}%)")

    if args.exclude_box:
        y0, y1, x0, x1 = args.exclude_box
        yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
        box = (yy >= y0) & (yy <= y1) & (xx >= x0) & (xx <= x1)
        n0 = int(blank_mask.sum()); blank_mask &= ~box
        print(f"--exclude-box y {y0}-{y1}, x {x0}-{x1}:"
              f"blank {n0:,} -> {int(blank_mask.sum()):,}"
              f" ({100 * blank_mask.sum() / max(n0, 1):.1f}%)")

    if args.r_far_src:
        # 只擋主源**以外**的源 —— 主源的暈比小源延伸得遠得多,適合它的半徑
        # 和適合小源的差一個量級,所以主源另外由 --mask-half / --xlim 處理。
        main, sid = main_source_mask(seg, args.main_id)
        others = (seg > 0) & ~main
        d = ndimage.distance_transform_edt(~others)
        n0 = int(blank_mask.sum()); blank_mask &= d > args.r_far_src
        print(f"--r-far-src {args.r_far_src:g} px(離主源 ID {sid} 以外的源):"
              f"blank {n0:,} -> {int(blank_mask.sum()):,}"
              f" ({100 * blank_mask.sum() / max(n0, 1):.1f}%)")

    print(f"blank spaxels: {int(blank_mask.sum())}")


    with fits.open(WSKY, memmap=True) as hdul:
        hdr = hdul["DATA"].header
        nz  = hdr["NAXIS3"]
        wl  = hdr["CRVAL3"] + (np.arange(nz) + 1 - hdr["CRPIX3"]) * hdr["CD3_3"]

        blank = np.empty((nz, int(blank_mask.sum())), np.float32)
        for j in range(0, nz, 200):
            d = np.asarray(hdul["DATA"].data[j:j+200], np.float32)
            blank[j:j+200] = d[:, blank_mask]

    # 只留光譜完整的 spaxel。大氣色散讓有效視野逐波長平移,視野邊緣因此有一圈
    # spaxel 只在部分波長被覆蓋。learn_sky_basis 會把缺失通道 nan_to_num 成 0
    # —— 那是捏造的資料,SVD 會認真去擬合它,所以直接要求 100% 覆蓋。
    complete = np.isfinite(blank).all(axis=0)
    print(f"完整光譜 {int(complete.sum()):,} / {blank.shape[1]:,} "
          f"({100*complete.mean():.1f}%),其餘為視野邊緣的部分覆蓋 spaxel,不採用")
    blank = blank[:, complete]

    # 逐通道 sigma-clip 之後再平均。平均數的崩潰點是 0% —— 單一通道裡只要有幾個
    # 極端負值,就足以把該通道的平均拉低,於是 estimate_continuum 把它判成一條
    # 「負的線」遮掉,是看不見的資料損失。
    #
    # 剪的方向是「同一通道、跨 spaxel」,不是沿波長:一條天光線在每個 spaxel 都亮,
    # 它的亮度就在該通道的中位數裡,不會被剪掉。
    #
    # 中心與散布用穩健的量,但平均那一步仍用平均數:跨 spaxel 的分布在亮線通道
    # 右偏,中位數會系統性偏低,那是偏差,樣本再多也不會消失。
    p16, med, p84 = np.percentile(blank, [16, 50, 84], axis=1)
    sg   = np.maximum((p84 - p16) / 2, 1e-6)
    keep = np.abs(blank - med[:, None]) <= CLIP_SIGMA * sg[:, None]
    # dtype=float64:blank 是 float32,累加上萬項會累積出可觀的捨入誤差
    mean_sky = (blank * keep).sum(axis=1, dtype=np.float64) / keep.sum(axis=1)
    print(f"mean_sky: sigma-clip {CLIP_SIGMA} sigma 剔除 {int((~keep).sum()):,} / "
          f"{keep.size:,} 個元素 ({100*(~keep).mean():.6f}%)")
    C_sky, sigma, line_mask, history = estimate_continuum(
        mean_sky, thresholds=THRESHOLDS, window=WINDOW, max_iter=MAX_ITER)
    print(f"line_mask: {100*line_mask.mean():.1f}% of channels  "
          f"({len(history)} iterations: "
          f"{' -> '.join(f'{100*h[2].mean():.1f}%' for h in history)})")

    np.save(out_dir / "wavelength.npy",    wl)
    np.save(out_dir / "mean_sky.npy",      mean_sky)
    np.save(out_dir / "sky_continuum.npy", C_sky)
    np.save(out_dir / "sky_sigma.npy",     sigma)
    np.save(out_dir / "line_mask.npy",     line_mask)

    # 每一輪的中間結果。遮罩不是逐輪累加的 —— 每輪都用原始 mean_sky 重算,
    # 上一輪只透過「把線挖成 NaN 再估連續譜」間接影響門檻,所以少數邊緣
    # 通道會被放回來。要判讀遮罩為何一路成長,必須連續譜與 sigma 一起看。
    np.save(out_dir / "iter_continuum.npy", np.array([h[0] for h in history]))
    np.save(out_dir / "iter_sigma.npy",     np.array([h[1] for h in history]))
    np.save(out_dir / "iter_line_mask.npy", np.array([h[2] for h in history]))

    # 同一個 keep 直接沿用。R = blank − C_sky 只差一個逐通道常數,而常數會同時
    # 平移 x 和它的中位數,所以 |x − med| / sg 完全不變 —— 兩處是同一個不等式。
    #
    # 被剔除的位置填該通道的典型殘差 med − C_sky,而不是 0:在天光線通道上
    # 填 0 等於宣稱那裡沒有線,med 是比較誠實的說法。
    residual = np.where(keep, blank - C_sky[:, None], (med - C_sky)[:, None])

    for method in args.methods:
        t0 = time.time()
        basis, S = learn_sky_basis(residual, K=args.K, method=method)
        np.save(out_dir / f"sky_basis_{method}_K{args.K}.npy", basis)   # 檔名帶 K,不同 K 可並存
        print(f"{method:13s} basis {basis.shape}  {time.time() - t0:6.1f}s", flush=True)

        if S is not None:
            energy = (S ** 2).sum(axis=1)                       # (n_blank,) 每個 spaxel
            top    = np.argsort(energy)[::-1][:50]
            np.save(out_dir / f"{method}_sparse_energy_K{args.K}.npy",  energy)
            np.save(out_dir / f"{method}_sparse_top_K{args.K}.npy",     S[top])
            np.save(out_dir / f"{method}_sparse_meanabs_K{args.K}.npy", np.abs(S).mean(axis=0))
            print(f"              S: 非零 {100*np.mean(S != 0):.2f}%,"
                  f" 已存前 50 個最強 spaxel 的稀疏成分")


if __name__ == "__main__":
    main()