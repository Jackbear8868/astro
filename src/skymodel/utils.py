"""Shared utilities.

兩組東西:
  * 光譜方向 —— running median、逐輪天光線偵測(estimate_continuum 等)
  * 空間方向 —— 把逐 spaxel 的係數建成一個平滑的場(檔案後半的 s field 那一節)
"""

import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")              # 必須在 import pyplot 之前
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis


def running_median(spectrum, window=300):
    half = window // 2
    n = len(spectrum)
    result = np.empty(n)
    for j in range(n):
        result[j] = np.nanmedian(spectrum[max(0, j - half):j + half])
    return result


def detect_lines(mean_sky, exclude=None, thresholds = (1, 2), window=300):
    m = mean_sky.copy()
    if exclude is not None:
        m[exclude] = np.nan                        # 上一輪抓到的線 → 不參與 continuum
    continuum = running_median(m, window)
    
    x = np.arange(len(m))
    good = np.isfinite(m) & np.isfinite(continuum)
    xg, yg = x[good], continuum[good]
    spl = UnivariateSpline(xg, yg, k=3, s=len(xg) * 0.05**2, ext=3)
    continuum = spl(x)

    abs_diff = np.abs(m - continuum)
    sigma = running_median(abs_diff, window)       # 對波長方向取 running median

    line_mask = (mean_sky > continuum + thresholds[0] * sigma) | (mean_sky < continuum - thresholds[1] * sigma)
    return continuum, sigma, line_mask


def estimate_continuum(mean_sky, thresholds=(1, 2), window=300, max_iter=5, min_unmasked_frac=0.16):
    line_mask = None
    history = []

    for i in range(max_iter):
        continuum, sigma, new_mask = detect_lines(mean_sky, exclude=line_mask, thresholds=thresholds, window=window)
        
        unmasked_frac = 1.0 - new_mask.sum() / new_mask.size
        if unmasked_frac < min_unmasked_frac:
            print(f"Iteration {i+1}: unmasked fraction {unmasked_frac:.1%} < floor {min_unmasked_frac:.0%}. Stop iteration.")
            break

        if line_mask is not None and np.array_equal(new_mask, line_mask):
            break

        line_mask = new_mask
        history.append((continuum, sigma, line_mask))
    
    if not history:
        raise ValueError(f"First iteration already masked more than {1 - min_unmasked_frac:.0%} of the spectrum.\n"
        "Check the input spectrum and parameters.")

    return history[-1][0], history[-1][1], history[-1][2], history


def load_line_masks(path, cumulative=True):
    """讀 step3 存下的逐輪天光線遮罩,預設回傳累加版。

    存檔存的是「每一輪實際用的遮罩」—— 上面 estimate_continuum 的迴圈裡,
    每一輪的 new_mask 是 detect_lines 從頭算出來的,整個取代舊的,不是累加
    (`line_mask = new_mask`)。收斂條件 `np.array_equal(new_mask, line_mask)`
    也依賴這一點:累加的遮罩只會單調成長,永遠不可能相等。所以存檔必須維持
    非累加,那是過程的忠實紀錄。

    但拿來當「遮得越來越多」的序列比較時,非累加版有例外 —— 連續譜重擬之後
    個別通道會掉回門檻以下,於是第 i 輪並不嚴格包含第 i−1 輪。cumulative=True
    用 logical_or 前綴累積補上這個性質,讓通道數單調遞減。

    第 1 輪兩者相同,所以只讀 [0] 的呼叫端不受影響。
    """
    m = np.load(path)
    return np.logical_or.accumulate(m, axis=0) if cumulative else m


def fit_chi2_coefficients(residual, variance, basis):
    """固定 sky basis，以 inverse-variance weighted least squares 求一條光譜的係數。

    Parameters
    ----------
    residual : ndarray, shape (nz,)
        一個 spaxel 中準備拿來擬合 sky lines 的殘差光譜。
    variance : ndarray, shape (nz,)
        同一個 spaxel 的 MUSE STAT；其數值是每個波長的 variance。
    basis : ndarray, shape (K, nz)
        只從 blank spaxels 學到的 K 條固定 sky-line basis。

    Returns
    -------
    coefficients : ndarray, shape (K,)
        讓 chi-square 最小的 K 個 basis 係數。若有效資料不足或 basis
        不具完整 rank，回傳 K 個 NaN，表示無法可靠求解。
    """
    # 每一個條件的 shape 都是 (nz,)。只有 data、STAT 與全部 basis
    # 都是有限值，且 STAT > 0 的波長，才能參與 1/STAT 加權擬合。
    good = (
        np.isfinite(residual)
        & np.isfinite(variance)
        & (variance > 0)
        & np.all(np.isfinite(basis), axis=0)
    )

    n_coeff = basis.shape[0]  # K：需要求解的 sky-basis 係數數量

    # 有效觀測數必須多於未知係數數量，否則沒有多餘資料約束這個 fit。
    if good.sum() <= n_coeff:
        return np.full(n_coeff, np.nan)

    y = residual[good].astype(np.float64)          # (n_good,)
    A = basis[:, good].T.astype(np.float64)        # (n_good, K)

    # STAT = sigma^2；除以 sigma 等價於讓平方殘差帶有 1/STAT 權重。
    sigma = np.sqrt(variance[good].astype(np.float64))
    y_white = y / sigma                            # (n_good,)
    A_white = A / sigma[:, None]                   # (n_good, K)

    # 同時求解 K 個可正可負的係數，使 ||y_white - A_white @ w||^2 最小。
    coefficients, _, rank, _ = np.linalg.lstsq(
        A_white,
        y_white,
        rcond=None,
    )

    # rank < K 表示 basis 中沒有 K 個獨立方向，因此係數不是唯一解。
    if rank < n_coeff:
        return np.full(n_coeff, np.nan)

    return coefficients


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


def per_spaxel_continuum(
    spectra,
    line_mask,
    window=300,
    chunk=8000,
):
    """估計每個 spaxel 自己的平滑 continuum。

    Parameters
    ----------
    spectra : ndarray, shape (nz, n_spaxels)
        多個 spaxels 的光譜；每一欄是一個 spaxel。
    line_mask : ndarray of bool, shape (nz,)
        從 mean blank-sky spectrum 偵測出的全域 sky-line mask。
        True 表示該 wavelength 不參與 running median。
    window : int
        running median 的 wavelength window，單位是 spectral pixels。
    chunk : int
        每次同時處理的 spaxel 數量，只控制記憶體與速度，
        不改變 continuum 的科學定義。

    Returns
    -------
    continuum_own : ndarray, shape (nz, n_spaxels)
        每個 spaxel 各自估計的平滑 continuum。
    """
    nz, n_spaxels = spectra.shape

    continuum_own = np.empty(
        (nz, n_spaxels),
        dtype=np.float32,
    )

    for low in range(0, n_spaxels, chunk):
        high = min(low + chunk, n_spaxels)

        chunk_spectra = spectra[:, low:high].astype(
            np.float64,
            copy=True,
        )

        chunk_spectra[line_mask, :] = np.nan

        chunk_continuum = (
            pd.DataFrame(chunk_spectra)
            .rolling(
                window=window,
                center=True,
                min_periods=1,
            )
            .median()
            .to_numpy()
            .astype(np.float32)
        )

        continuum_own[:, low:high] = chunk_continuum

    return continuum_own


# ---------------------------------------------------------------------------
# s field —— 把逐 spaxel 的天空連續譜係數 s 建成一個空間上的場
# ---------------------------------------------------------------------------
# 為什麼要這樣做:逐 spaxel 自由解 s 的話,源旁邊那格看到多出來的源光,唯一能
# 解釋它的辦法就是把自己的 s 抬高 —— 那個逐 spaxel 的自由度就是源光被天空模型
# 吃掉的通道。改成「只用遠離源的 spaxel 定一個場,再外插到源附近」,一格的資料
# 撼動不了那個場,源的光在天空模型裡就無處可去,只能留在殘差裡(= 被保留)。
#
# 場的形式(見 rowcol_field / kernel_field):
#
#     s_hat(x, y) = mu + a(y) + b(x) + r(x, y)
#
# 前三項描述儀器造成的軸對齊條紋(沿整行/整列延伸,不是天空也不是源),
# 第四項是剩下的局部起伏。

TRUNCATE = 3.0      # 高斯核截在 3 sigma:含 99.7% 的質量。核的成本正比於截斷
                    # 半徑,所以截得比 scipy 預設的 4 sigma 短可以省下時間


def scale(a):
    """穩健散布 (p84 − p16)/2。

    不能用 rms/std:s 有少數擬合失敗的格,離群值極大,rms 與 std 會被那幾格
    單獨主宰,量到的不再是整體的散布。
    """
    a = a[np.isfinite(a)]
    return float((np.percentile(a, 84) - np.percentile(a, 16)) / 2) if a.size else np.nan


def noise_floor(img, m, axis=1):
    """相鄰格差分估求解雜訊。var(a−b) = 2·var(noise),所以要除 sqrt(2)。

    假設真值在相鄰兩格近似相同。若真有 1 px 尺度的結構,這個估計會偏高 ——
    所以它是雜訊的上界,也就是「可壓縮空間」的保守估計。
    """
    if axis == 0:
        d, mm = img[1:] - img[:-1], m[1:] & m[:-1]
    else:
        d, mm = img[:, 1:] - img[:, :-1], m[:, 1:] & m[:, :-1]
    return scale(d[mm]) / np.sqrt(2.0)


def nanmed(a, axis):
    """沿 axis 取中位數;整行/整列全是 NaN 時回 0 而不是 NaN。

    全 NaN 代表那一行沒有任何訓練點,偏移量無從估計。設 0 就是「不作修正」,
    是這種情況下唯一誠實的選擇 —— 但要知道它是一個假設,不是一個測量。
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nan_to_num(np.nanmedian(a, axis=axis))


def main_source_group(seg, white, bridge=3):
    """主星系的完整足跡 —— 最亮像素所在的那一**整團**源。回傳 (遮罩, ID 清單, 峰值座標)。

    為什麼不能用「面積最大」或「流量最大」的**單一** ID:SExtractor 的 deblender
    會把 Haro 11 拆開。它是合併星系,本來就有數個亮結,而拆不拆、拆成幾塊,取決於
    那一次的 seeing 與 dither,不同 exposure 並不一致。被拆開時,任何「選一個 ID」
    的規則都只會拿到星系的一部分,而下游用它來決定「遮掉哪半邊」與「排除周圍多少
    像素」—— 選錯一塊,兩件事都會遮錯位置。

    做法:把 seg > 0 膨脹 bridge 像素讓被拆開的結重新相連,取最亮像素所在的
    連通塊,再回傳該塊涵蓋的所有 seg ID。bridge 只是用來跨過 deblend 留下的
    一兩個像素縫隙,不是形狀參數。
    """
    k = np.unravel_index(np.nanargmax(np.where(np.isfinite(white), white, -np.inf)),
                         white.shape)
    src = seg > 0
    lab, _ = ndimage.label(ndimage.binary_dilation(src, iterations=bridge))
    grp = lab == lab[k]
    ids = sorted(int(i) for i in np.unique(seg[grp & src]) if i > 0)
    return np.isin(seg, ids), ids, k


def main_source_mask(seg, source_id=None, main_blob=True):
    """主源的遮罩。回傳 (布林遮罩, 用到的 seg ID)。

    source_id 省略時取**面積最大**的源 —— 不要寫死 seg == 1。SExtractor 的編號
    是按偵測順序給的,換一個 pointing 就不保證主星系還是 1 號,而寫死的話跑錯
    不會報錯,只會安靜地把某個小源當成主源去排除幾十像素。

    main_blob=True 只取最大的連通塊。一個 seg ID 可能由**數個不相連的塊**組成 ——
    那是 SExtractor 的 `CLEAN Y` 把它判為假偵測的天體「像素併進鄰近亮源」造成的。
    用整個 ID 算距離的話,那些散在遠處的小碎塊會各自撐出一圈排除區,把排除面積
    放大到遠超過主源本身。
    """
    if source_id is None:
        ids, cnt = np.unique(seg[seg > 0], return_counts=True)
        source_id = int(ids[np.argmax(cnt)])
    m = seg == source_id
    if main_blob:
        lab, n = ndimage.label(m)
        if n > 1:
            sz = ndimage.sum(m, lab, range(1, n + 1))
            m = lab == (int(np.argmax(sz)) + 1)
    return m, source_id


def half_field_mask(seg, valid, source_id=None, central=0.3, round_to=5,
                    main=None):
    """遮掉主源所在的那一半視場,回傳 (可用來學天空的布林遮罩, 說明字串)。

    規則:
      ① 找主源的主連通塊,算質心
      ② 量質心在 y、x 兩個方向偏離視場中心多遠,以半邊長為單位
      ③ 偏心量 >= central 的那一軸,遮掉主源所在的那一半
      ④ 兩軸都 < central(主源在視場中央)-> 改遮掉一個**以質心為中心的置中矩形**,
         矩形與視場同長寬比,大小取到「涵蓋一半的有效面積」,留下外圈
      ⑤ 所有界線取整到 round_to 的倍數(預設 5) —— 半個像素的精度在這裡沒有意義,
         整數的界線比較好記、好在文件裡引用、也比較好和別人對齊

    用途是讓「遮掉主源那半邊」的判斷能自動套到任何一個 pointing,不必逐顆手給
    界線;需要指定確切界線時仍可改用 step3 的 --xlim / --ylim。
    """
    def snap(v):
        return int(round(v / round_to) * round_to) if round_to else int(round(v))

    if main is None:
        main, sid = main_source_mask(seg, source_id)
    else:
        sid = "(group)"      # 呼叫端已用 main_source_group 算好整團
    ny, nx = seg.shape
    ys, xs = np.nonzero(main)
    cy, cx = ys.mean(), xs.mean()
    off_y, off_x = (cy - ny / 2) / (ny / 2), (cx - nx / 2) / (nx / 2)

    if max(abs(off_y), abs(off_x)) < central:
        # 主源在中央:挖掉一個置中矩形,大小二分搜尋到剛好一半的有效面積
        n_half = valid.sum() / 2.0
        yy, xx = np.mgrid[0:ny, 0:nx]
        lo, hi = 0.0, 1.0
        for _ in range(40):
            t = (lo + hi) / 2
            box = (np.abs(yy - cy) <= t * ny / 2) & (np.abs(xx - cx) <= t * nx / 2)
            if (box & valid).sum() < n_half:
                lo = t
            else:
                hi = t
        y0, y1 = snap(cy - hi * ny / 2), snap(cy + hi * ny / 2)
        x0, x1 = snap(cx - hi * nx / 2), snap(cx + hi * nx / 2)
        box = (yy >= y0) & (yy <= y1) & (xx >= x0) & (xx <= x1)
        return valid & ~box, (f"主源 ID {sid} 在視場中央(偏心 y {off_y:+.2f}, "
                              f"x {off_x:+.2f} 都 < {central:g}):挖掉 "
                              f"y {y0}-{y1}, x {x0}-{x1} 的矩形(涵蓋一半有效面積),"
                              f"留外圈")

    yy, xx = np.mgrid[0:ny, 0:nx]
    if abs(off_y) >= abs(off_x):
        cut = snap(ny / 2)
        keep = (yy < cut) if off_y > 0 else (yy >= cut)
        side = "上" if off_y > 0 else "下"
        txt = f"沿 y 切,主源在{side}半 -> 留 y {'<' if off_y > 0 else '>='} {cut}"
    else:
        cut = snap(nx / 2)
        keep = (xx < cut) if off_x > 0 else (xx >= cut)
        side = "右" if off_x > 0 else "左"
        txt = f"沿 x 切,主源在{side}半 -> 留 x {'<' if off_x > 0 else '>='} {cut}"
    return valid & keep, (f"主源 ID {sid} 質心 (y={cy:.0f}, x={cx:.0f}),"
                          f"偏心 y {off_y:+.2f} / x {off_x:+.2f};{txt}")


def ridge_from_data(resid, train, sigma, axis=1):
    """由資料推出 kernel_field 的 ridge —— 不是挑的,是算的。

        ridge = sigma_noise^2 / (S * n)

    其中 sigma_noise 是求解雜訊(相鄰格差分估),S 是殘差裡真實結構的變異數
    (總變異數扣掉雜訊),n = 4*pi*sigma^2 是完全覆蓋時的有效樣本數。

    這條式子的來歷:對「先驗 N(0, S)、觀測雜訊 sigma_noise^2、有效樣本數 m」
    的貝氏後驗均值,收縮係數是 m*S / (m*S + sigma_noise^2)。而 kernel_field 的
    m 正比於 den(完全覆蓋時 den=1、m=n),代入整理就得到 num / (den + ridge)。
    也就是說 **ridge 形式本身就是貝氏後驗均值**,不是事後補的平滑手段。
    """
    sn = noise_floor(resid, train, axis)
    S  = max(scale(resid[train]) ** 2 - sn ** 2, 0.0)
    n  = 4.0 * np.pi * sigma ** 2
    return float(sn ** 2 / (S * n)) if S > 0 else np.inf


def kernel_field(v, w, sigma, ridge):
    """歸一化卷積(核回歸)加上 ridge。回傳 (值, 收集到的權重 den)。

                      Sum_q  w(q) * K(p - q) * v(q)
        v_hat(p)  =  ------------------------------
                      Sum_q  w(q) * K(p - q)  +  ridge

    w 是訓練點指示函數,K 是高斯核。分子分母各自是一個卷積,所以兩次
    gaussian_filter 就算完,不必對幾萬個點跑迴圈。

    **分母是關鍵**,不是防呆。den = Sum w*K 的意義是「這個位置實際收集到了多少
    真實的權重」,值域 0-1(高斯核的權重和為 1)。在源區域內部 w 全是 0,分子分母
    都只拿得到外圈的貢獻 —— 洞裡的值完全由周圍的遠場決定。**外插不是另外寫的
    一段程式,它是這條式子的自然結果。** 少了分母的話,等於把源區當成「v = 0 的
    真實觀測」,估計值會被往 0 拉,在洞的邊緣與內部偏得最兇。

    ridge 是加在分母上的常數,做兩件事:

    ① **退場**。den 很小的地方,估計只由一兩個很遠的點決定,非常吵。加了 ridge
       之後 den -> 0 時 num 也 -> 0,v_hat 平滑地歸零。呼叫端把它當殘差項相加,
       所以歸零正好讓模型降級成沒有這一項的版本。

    ② **收縮**。即使覆蓋完整,估計仍帶雜訊,貝氏上就該往先驗 0 收一點。
       ridge_from_data 給的值讓收縮量正好等於後驗均值。

    為什麼不是用門檻(den <= min_weight 就設 0):門檻會在等高線上製造不連續,
    而那條等高線繞著每個源形成一圈。我們整個專案就在對付這種環,不該自己再造
    一個。單純把門檻拿掉也不行 —— 高斯核有**有限支撐**(TRUNCATE x sigma),
    離最近訓練點超過那個距離 den 恰為 0,不連續只是被推到支撐邊緣,而那裡的
    估計更吵。

    ridge 的形式**沒有任何分支、遮罩或門檻**,v_hat 處處連續。
    """
    num = ndimage.gaussian_filter(np.where(w, np.nan_to_num(v), 0.0), sigma,
                                  mode="constant", cval=0.0, truncate=TRUNCATE)
    den = ndimage.gaussian_filter(w.astype(np.float64), sigma,
                                  mode="constant", cval=0.0, truncate=TRUNCATE)
    return num / (den + ridge), den


def rowcol_field(s, w, n_iter=4):
    """s ≈ mu + a(y) + b(x),交替以中位數求解(Tukey 的 median polish)。

    回傳 (場, a, b)。

    為什麼是「相加」而不是一般的二維函數 f(x, y):一般的 f 等於每格一個數字,
    沒有壓縮、也就無法預測沒有資料的地方。相加的形式只有 1 + ny + nx 個參數,
    而且 **a(y) 被那一列的所有 spaxel 共用** —— 這正是資訊能橫向流動、伸得進
    大洞的原因:洞中央那一列的 a(y),是從同一列遠離源的那些格算出來的。

    能表示:橫條紋、直條紋、任何線性的斜向梯度(alpha*x + beta*y 本身可分離)。
    不能表示:只出現在某個位置、不沿整列或整行延伸的東西(統計上叫交互作用項)。
    那些會留在殘差裡,交給 kernel_field 處理。

    為什麼用中位數而不是平均:① 對壞格穩健;② **對「缺一大塊」自然** —— 一列
    即使被源佔掉大半,剩下那些格的中位數照樣是那一列偏移量的好估計。

    為什麼要交替:a 和 b 互相耦合。要量「這一列偏高多少」,得先把各行本身的偏移
    扣掉,否則量到的是「這一列剛好有哪些行還在」。n_iter 預設 4,是收斂所需輪數
    之上留的餘裕。

    注意簡併:每個 a(y) 加 c、每個 b(x) 減 c,場完全相同。所以 (mu, a, b) 各自
    不唯一,只有它們的和有意義;交替中位數會自然讓 median(a)、median(b) 落在 0
    附近,由 mu 吸收整體水準。要單獨解讀 a(y) 時必須記得這一點。
    """
    S  = np.where(w, s, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = float(np.nanmedian(S))
    a = np.zeros(s.shape[0])
    b = np.zeros(s.shape[1])
    for _ in range(n_iter):
        a = nanmed(S - mu - b[None, :], 1)
        b = nanmed(S - mu - a[:, None], 0)
    return mu + a[:, None] + b[None, :], a, b


def build_s_field(s, seg, blank, r_far, r_far_haro, clip, sigma, ridge=None,
                  main_id=None, exclude=None, main=None):
    """從逐 spaxel 的 s 圖建出空間場。回傳 (s_hat, 訓練遮罩, 實際用的 ridge)。

    Parameters
    ----------
    s : ndarray, shape (ny, nx)
        逐 spaxel 自由解出來的天空連續譜係數。源區的值不會被用到。
    seg : ndarray, shape (ny, nx)
        segmentation;0 = blank,>0 = 源,1 = Haro 11。
    blank : ndarray of bool, shape (ny, nx)
        可用的 blank spaxel(視場內、非源、光譜完整、s 有解)。
    r_far : float
        訓練點必須離**任何**源這麼遠(px)。要蓋過源的 PSF 翼,否則訓練點本身
        就帶著源光。代價只是樣本變少。
    r_far_haro : float or None
        主源專用的加碼半徑,只對主源生效。主源的延展暈比小源伸得遠得多,用同一個
        r_far 的話訓練點仍在暈裡,模型會把暈學成天空。None = 不加碼。
    main_id : int or None
        主源的 segmentation ID。None = 自動取面積最大的源(見 main_source_mask)。
        距離從它的**主連通塊**算,不是整個 ID —— 理由見 main_source_mask。
    exclude : ndarray of bool or None
        額外排除在訓練樣本外的格(仍然會被扣天空,只是不參與定場)。用途:
        鑲嵌視場裡曝光數不足、雜訊遠高於外圈的區塊 —— 拿它學天空等於把雜訊
        寫進場裡。
    clip : float
        |s − 中位| > clip x 穩健散布 的格不採用(剔掉擬合失敗的解)。
    sigma : float
        殘差項的核寬(px)。條紋已經被 rowcol_field 拿走,留給這一項的是細碎的
        局部起伏,核越大糊掉的細節越多。
    ridge : float or None
        見 kernel_field。None = 由資料推出(ridge_from_data),這是預設也是建議 ——
        它沒有任何人為挑的數字。

    Notes
    -----
    hybrid 的組合方式:先用 rowcol_field 拿走條紋,再對**殘差**做一次核回歸。
    核伸不到的地方殘差項平滑歸零,模型自動退化成 mu + a(y) + b(x) —— 所以不論
    洞多大都給得出值,而純核回歸在大洞中央會完全沒有值。
    """
    train = blank & (ndimage.distance_transform_edt(seg == 0) > r_far)
    if exclude is not None:
        train &= ~exclude
    if r_far_haro:
        m = main if main is not None else main_source_mask(seg, main_id)[0]
        train &= ndimage.distance_transform_edt(~m) > r_far_haro
    med = float(np.median(s[train]))
    train &= np.abs(s - med) <= clip * scale(s[train])

    M, _, _ = rowcol_field(s, train)
    R = s - M
    if ridge is None:
        ridge = ridge_from_data(R, train, sigma)
    r, _ = kernel_field(R, train, sigma, ridge)
    return M + r, train, ridge