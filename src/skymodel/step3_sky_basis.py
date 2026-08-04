"""從 blank spaxels 學出天空模型的兩個成分:連續譜 C_sky 與 K 條天光線 basis。

輸出供 step4 的聯合擬合使用。天光線 basis 的分解方法可替換。
"""
from pathlib import Path
import numpy as np
from astropy.io import fits
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.utils.extmath import randomized_svd

from utils import estimate_continuum
import argparse
import time

SEED       = 0           # 所有分解共用的亂數種子,確保 basis 可重現
K          = 25          # basis 條數 = 天空模型的自由度;K=10 時天光線的
                         # 模型誤差是 ESO 的 1.7 倍,K=25 降到 0.74 倍
WINDOW     = 300         # 連續譜 running median 視窗 (px)
THRESHOLDS = (1, 2)      # 線偵測門檻 (正, 負),教授指定
MAX_ITER   = 5           # estimate_continuum 迭代上限
CLIP_SIGMA = 30          # mean_sky 的 sigma-clip 門檻。不是調出來的:跨 spaxel 的
                         # 真實變化到 10 sigma 就結束(3->10 sigma 元素數掉 800 倍),
                         # 而最輕微的壞值在 197 sigma —— 10~100 之間結果完全相同,
                         # 30 離兩端各有 3 倍與 6.6 倍餘裕
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

ROOT   = Path(__file__).resolve().parents[2]
STEP01 = ROOT / "results/skymodel/step01"
STEP03 = ROOT / "results/skymodel/step03"
WSKY   = ROOT / "data/Haro11_NEpointing_wsky.fits"



def main():
    ap = argparse.ArgumentParser(description="從 blank spaxels 學天空連續譜與天光線 basis")
    ap.add_argument("--methods", nargs="+", default=["pca", "svd", "nmf"],
                    choices=METHODS, help="要跑哪些分解方法")
    ap.add_argument("-K", type=int, default=K, help="basis 條數")
    args = ap.parse_args()

    STEP03.mkdir(parents=True, exist_ok=True)

    white = fits.getdata(STEP01 / "whitelight.fits")
    seg   = fits.getdata(STEP01 / "seg.fits")

    valid_mask = white != 0
    blank_mask = valid_mask & ~((seg > 0) & valid_mask)
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
    # spaxel 只在部分波長被覆蓋(有的只剩 46/3801 個通道)。learn_sky_basis 會把
    # 缺失通道 nan_to_num 成 0 —— 那是捏造的資料,SVD 會認真去擬合它。實測在
    # 樣本數相同的條件下,混入 1.9% 這種 spaxel 會讓天光線的模型誤差惡化 28%。
    # 完整的 spaxel 有 8 萬多個,樣本綽綽有餘,所以直接要求 100% 覆蓋。
    complete = np.isfinite(blank).all(axis=0)
    print(f"完整光譜 {int(complete.sum()):,} / {blank.shape[1]:,} "
          f"({100*complete.mean():.1f}%),其餘為視野邊緣的部分覆蓋 spaxel,不採用")
    blank = blank[:, complete]

    # 逐通道 sigma-clip 之後再平均。平均數的崩潰點是 0% —— 通道 151 (4788 A) 有
    # 12 個 spaxel 的值是 −1,750 到 −548,029,把該通道的平均從 28.6 拉到 1.8,於是
    # estimate_continuum 把它判成一條「負的線」遮掉,是看不見的資料損失。
    #
    # 剪的方向是「同一通道、跨 spaxel」,不是沿波長:一條天光線在每個 spaxel 都亮,
    # 它的亮度就在該通道的中位數裡,不會被剪掉(實測門檻 >=10 sigma 時,線通道與
    # 線外通道的偏移都是 0.0000%)。
    #
    # 中心與散布用穩健的量,但平均那一步仍用平均數:中位數在亮線通道會系統性偏低
    # (跨 spaxel 的分布右偏,5577 A 低了 2.47),那是偏差,樣本再多也不會消失。
    p16, med, p84 = np.percentile(blank, [16, 50, 84], axis=1)
    sg   = np.maximum((p84 - p16) / 2, 1e-6)
    keep = np.abs(blank - med[:, None]) <= CLIP_SIGMA * sg[:, None]
    # dtype=float64:blank 是 float32,7 萬多項相加會累積出 ~0.2 的誤差
    mean_sky = (blank * keep).sum(axis=1, dtype=np.float64) / keep.sum(axis=1)
    print(f"mean_sky: sigma-clip {CLIP_SIGMA} sigma 剔除 {int((~keep).sum()):,} / "
          f"{keep.size:,} 個元素 ({100*(~keep).mean():.6f}%)")
    C_sky, sigma, line_mask, history = estimate_continuum(
        mean_sky, thresholds=THRESHOLDS, window=WINDOW, max_iter=MAX_ITER)
    print(f"line_mask: {100*line_mask.mean():.1f}% of channels  "
          f"({len(history)} iterations: "
          f"{' -> '.join(f'{100*h[2].mean():.1f}%' for h in history)})")

    np.save(STEP03 / "wavelength.npy",    wl)
    np.save(STEP03 / "mean_sky.npy",      mean_sky)
    np.save(STEP03 / "sky_continuum.npy", C_sky)
    np.save(STEP03 / "sky_sigma.npy",     sigma)
    np.save(STEP03 / "line_mask.npy",     line_mask)

    # 每一輪的中間結果。遮罩不是逐輪累加的 —— 每輪都用原始 mean_sky 重算,
    # 上一輪只透過「把線挖成 NaN 再估連續譜」間接影響門檻,所以少數邊緣
    # 通道會被放回來。要判讀遮罩為何一路成長,必須連續譜與 sigma 一起看。
    np.save(STEP03 / "iter_continuum.npy", np.array([h[0] for h in history]))
    np.save(STEP03 / "iter_sigma.npy",     np.array([h[1] for h in history]))
    np.save(STEP03 / "iter_line_mask.npy", np.array([h[2] for h in history]))

    # 同一個 keep 直接沿用。R = blank − C_sky 只差一個逐通道常數,而常數會同時
    # 平移 x 和它的中位數,所以 |x − med| / sg 完全不變 —— 兩處是同一個不等式。
    #
    # 被剔除的位置填該通道的典型殘差 med − C_sky。實測填 0 完全等價(rms 6.8757
    # vs 6.8759),因為 167 個元素在 2.95 億裡影響不到 SVD;但在天光線通道上
    # 「0」等於宣稱那裡沒有線,med 是比較誠實的說法。
    #
    # 效果(K=30, svd):那條「90% 能量壓在通道 151」的 basis 消失(n90 1 → 2),
    # blank 殘差 rms −0.56%,而真正的收穫在源這一側 —— 模板 027 對 ID 14 的
    # chi2_all(z) 動態範圍從 8,308 倍降到 13 倍。原本 z>0.368 以上整段被一條
    # 壞掉的 basis 假性禁止(chi2 跳到 10^8),那不是物理,是 12 個壞 spaxel。
    residual = np.where(keep, blank - C_sky[:, None], (med - C_sky)[:, None])

    for method in args.methods:
        t0 = time.time()
        basis, S = learn_sky_basis(residual, K=args.K, method=method)
        np.save(STEP03 / f"sky_basis_{method}_K{args.K}.npy", basis)   # 檔名帶 K,不同 K 可並存
        print(f"{method:13s} basis {basis.shape}  {time.time() - t0:6.1f}s", flush=True)

        if S is not None:
            energy = (S ** 2).sum(axis=1)                       # (n_blank,) 每個 spaxel
            top    = np.argsort(energy)[::-1][:50]
            np.save(STEP03 / f"{method}_sparse_energy_K{args.K}.npy",  energy)
            np.save(STEP03 / f"{method}_sparse_top_K{args.K}.npy",     S[top])
            np.save(STEP03 / f"{method}_sparse_meanabs_K{args.K}.npy", np.abs(S).mean(axis=0))
            print(f"              S: 非零 {100*np.mean(S != 0):.2f}%,"
                  f" 已存前 50 個最強 spaxel 的稀疏成分")


if __name__ == "__main__":
    main()