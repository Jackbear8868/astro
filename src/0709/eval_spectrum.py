"""
eval_spectrum — 任選一塊空間區域，取平均(mean/median) sky-subtracted spectrum、畫圖、
                並沿波長方向收斂成一組摘要統計量(mean/sigma/skewness，每條 spectrum 各 3 個純量)。

  主要用法是 import 成函式庫：region 隨時可以自訂 x,y 範圍，不綁死幾種固定參數組合。
  region 只依賴 mask_from（決定 FoV 大小 + 對應哪張 source mask），跟「實際要讀哪個 FITS」
  是分開的兩件事，所以同一個 region 可以套在不同 cube（raw / zap.fits / sky.fits）上比較。

  但也可以直接當 CLI 跑（永遠包含 whole/blank 兩個 region，--box 可重複加自訂區域，
  --y-range 固定所有圖的 y 軸範圍，跑完存 png + spectra_report.json 到 --out-dir，並印到終端機）：
    conda run -n astro python src/0709/eval_spectrum.py \\
        --fits-path results/zap/cubes/NEwsky_maskfrom-seg1sigma_brq-NEnosky/zap.fits \\
        --mask-from NEnosky --mask-method seg1sigma_brq --y-range -20 20 \\
        --box spec2 40 120 20 100 --box spec3 180 260 220 280

  import 用法範例：
    import sys; sys.path.insert(0, "src/0709")
    import eval_spectrum as es
    region = es.region_blank("nosky")                                    # 整張 blank(valid & ~source)
    mean_spec, median_spec = es.extract_spectrum("results/zap/cubes/wsky_maskfrom-nosky/zap.fits", region)
    wl = es.wavelength_axis("nosky")
    es.plot_spectrum(wl, mean_spec, "results/zap/spec_blank_mean.png", label="wsky+ZAP blank mean")
    es.spectrum_stats(mean_spec)   # -> {"mean":..., "sigma":..., "skewness":...}

    # 或用三個便利包裝一次做完「建 region → 抽 spectrum → 畫圖」：
    es.plot_whole("results/zap/cubes/wsky_maskfrom-nosky/zap.fits", "nosky", "results/zap/spec_whole.png")
    es.plot_blank("results/zap/cubes/wsky_maskfrom-nosky/zap.fits", "nosky", "results/zap/spec_blank.png")
    es.plot_box("results/zap/cubes/wsky_maskfrom-nosky/zap.fits", "nosky", (0, 160), (166, 332),
                "results/zap/spec_box.png", exclude_source=True)

    # 只看特定波段（畫圖 + 統計都吃 wl_range，不用重讀 FITS）；WAVELENGTH_BANDS 內建三段常用波段：
    es.plot_blank(fits_path, "nosky", "results/zap/spec_blank_5000-7000.png", wl_range=es.WAVELENGTH_BANDS["5000-7000"])
    es.spectrum_stats_bands(mean_spec, wl)   # 一次算 WAVELENGTH_BANDS 三段的 stats

    # 多個 region × 多個波段一次做完（每個 region 只抽一次 spectrum，不必為每個 band 重讀 FITS）：
    regions = {"spec0_all": es.region_whole("nosky"),
               "spec1_nonmasked": es.region_blank("nosky"),
               "spec2_box": es.region_box("nosky", (40, 120), (20, 100))}   # (y_range, x_range)
    report = es.spectra_report(fits_path, "nosky", regions, out_dir="results/zap/cubes/wsky_maskfrom-nosky")
    # report = {"spec0_all": {"5000-9000": {...}, "5000-7000": {...}, "7000-9000": {...}}, ...}
    # spectra_report() 也自動存 out_dir/regions_overview.png：白光圖 + 每個 region 實際涵蓋範圍疊色。

    # 自訂 box 座標順序容易搞混（y 在前還是 x 在前？）——建 region 之前先比較候選座標，畫矩形示意圖：
    from astropy.io import fits as _fits
    background = _fits.getdata("results/zap/masks/sep_from-nosky/mask.fits")
    es.plot_region_overview(background, {"spec2": ((40, 120), (20, 100))},   # {name: (y_range, x_range)}
                             "results/zap/spec_region_overview.png")

    # 已經有 region array 的話（跟 spectra_report 的 regions 一樣），疊在白光圖上看實際涵蓋範圍：
    es.plot_regions_overlay(es.white_light_image(fits_path), regions, "results/zap/spec_regions_overlay.png")
"""
import os, sys, warnings
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings
import numpy as np
from scipy.stats import skew, kurtosis
from astropy.io import fits
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

BLANKSTATS_DIR = settings.OUTPUT_ROOT / "blankstats"

# 常用波長區間預設（Å）：目前三個 region spec 都要跑這三段。
WAVELENGTH_BANDS = {"5000-9000": (5000.0, 9000.0),
                    "5000-7000": (5000.0, 7000.0),
                    "7000-9000": (7000.0, 9000.0)}

# ======================= FoV valid =======================
def _valid2d(mask_from, chunk=200):
    """FoV 有效遮罩：raw(mask_from) 全波長 nansum != 0（FoV 外全 NaN → 0）。快取成 .npy。"""
    cache_path = BLANKSTATS_DIR / f"valid_from-{mask_from}.npy"
    if cache_path.exists():
        return np.load(str(cache_path))
    with fits.open(str(settings.input_cube_path(mask_from)), memmap=True) as hdul:
        cube_data = hdul["DATA"].data; n_wavelengths, n_rows, n_cols = cube_data.shape
        flux_sum = np.zeros((n_rows, n_cols))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for start in range(0, n_wavelengths, chunk):
                flux_sum += np.nansum(np.asarray(cube_data[start:min(start + chunk, n_wavelengths)]), axis=0)
    valid = flux_sum != 0
    BLANKSTATS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), valid)
    return valid

# ======================= Region spec：都回傳 2D bool array（shape 同 _valid2d(mask_from)）=======================
def region_whole(mask_from):
    """整張有效 FoV（排除邊緣 NaN / ZAP 填零的假訊號，不是陣列全部像素）。"""
    return _valid2d(mask_from)

def region_blank(mask_from, mask_method="sep"):
    """valid & ~source：FoV 內、且非源的 spaxel。"""
    source = fits.getdata(str(settings.source_mask_path(mask_from, mask_method))).astype(bool)
    return _valid2d(mask_from) & ~source

def region_box(mask_from, y_range, x_range, exclude_source=False, mask_method="sep"):
    """valid & box[y0:y1, x0:x1]（origin='lower' 陣列座標，跟 mask.py 一致）。
       exclude_source=True 時再排除該 mask_method 的 source mask。"""
    valid = _valid2d(mask_from)
    box = np.zeros_like(valid, dtype=bool)
    y0, y1 = y_range; x0, x1 = x_range
    box[y0:y1, x0:x1] = True
    region = valid & box
    if exclude_source:
        source = fits.getdata(str(settings.source_mask_path(mask_from, mask_method))).astype(bool)
        region &= ~source
    return region

# ======================= Spectrum 抽取 =======================
def extract_spectrum(fits_path, region, hdu="DATA", chunk=200):
    """對任意 fits_path，在 region 內的 spaxel 逐波長算 nanmean/nanmedian（分塊讀，省記憶體）。"""
    with fits.open(str(fits_path), memmap=True) as hdul:
        cube_data = hdul[hdu].data
        n_wavelengths = cube_data.shape[0]
        mean_spectrum = np.empty(n_wavelengths); median_spectrum = np.empty(n_wavelengths)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # 略過 all-NaN 波長列的警告
            for start in range(0, n_wavelengths, chunk):
                stop = min(start + chunk, n_wavelengths)
                slab = np.asarray(cube_data[start:stop])[:, region]   # (n_chunk_wavelengths, n_spaxel)
                mean_spectrum[start:stop]   = np.nanmean(slab, axis=1)
                median_spectrum[start:stop] = np.nanmedian(slab, axis=1)
    return mean_spectrum.astype(float), median_spectrum.astype(float)

def wavelength_axis(mask_from):
    with fits.open(str(settings.input_cube_path(mask_from))) as hd:
        return settings.wavelength_axis(hd["DATA"].header)

def white_light_image(fits_path, hdu="DATA", chunk=200):
    """整個 cube 沿波長方向 nanmean，得到白光（broadband continuum）影像——比 mask.fits
       更能看出實際場景長什麼樣（源、暈的形狀），用來當 region 示意圖的背景。分塊讀，省記憶體。"""
    with fits.open(str(fits_path), memmap=True) as hdul:
        cube_data = hdul[hdu].data
        n_wavelengths, ny, nx = cube_data.shape
        total = np.zeros((ny, nx)); count = np.zeros((ny, nx))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for start in range(0, n_wavelengths, chunk):
                stop = min(start + chunk, n_wavelengths)
                slab = np.asarray(cube_data[start:stop])
                finite = np.isfinite(slab)
                total += np.where(finite, slab, 0).sum(axis=0)
                count += finite.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan).astype(np.float32)

# ======================= 畫圖 =======================
def plot_spectrum(wavelengths, spectrum, out_path, label=None, title=None, wl_range=None, y_range=None):
    """單條 flux vs wavelength 曲線，0 基準線，右側顯示 spectrum_stats 摘要，存檔。
       wl_range=(lo,hi) 只放大該波段（不影響傳入的資料本身；右側的 stats 也只算這段，跟圖對得上）。
       y_range=(ymin,ymax) 固定 y 軸顯示範圍；不給則沿用 matplotlib 自動縮放。"""
    stats = spectrum_stats(spectrum, wavelengths, wl_range)
    fig, (ax, stat_ax) = plt.subplots(1, 2, figsize=(15.5, 4.5), gridspec_kw={"width_ratios": [5, 1]})
    ax.axhline(0, color="0.5", lw=0.5)
    ax.plot(wavelengths, spectrum, lw=0.8, label=label)
    ax.set_xlim(*wl_range) if wl_range is not None else ax.set_xlim(wavelengths.min(), wavelengths.max())
    if y_range is not None: ax.set_ylim(*y_range)
    ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
    ax.set_ylabel(r"Flux [$10^{-20}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]")
    if title: ax.set_title(title)
    if label: ax.legend(loc="upper left", fontsize=8)

    stat_ax.axis("off")
    key_width = max(len(k) for k in stats)
    stat_text = "\n".join(f"{k:<{key_width}} = {v:.4g}" for k, v in stats.items())
    stat_ax.text(0.0, 0.5, stat_text, fontsize=10, va="center", ha="left", family="monospace",
                 transform=stat_ax.transAxes)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=145); plt.close(fig)
    print(f"[eval_spectrum] saved {out_path}")

def _plot_region(fits_path, mask_from, region, out_path, statistic="mean", label=None, title=None, wl_range=None, y_range=None):
    mean_spec, median_spec = extract_spectrum(fits_path, region)
    spectrum = mean_spec if statistic == "mean" else median_spec
    wl = wavelength_axis(mask_from)
    band_note = f"  [{wl_range[0]:.0f}-{wl_range[1]:.0f}Å]" if wl_range is not None else ""
    plot_spectrum(wl, spectrum, out_path, label=label, title=(title or "") + band_note, wl_range=wl_range, y_range=y_range)
    return mean_spec, median_spec

def plot_whole(fits_path, mask_from, out_path, statistic="mean", wl_range=None, y_range=None):
    return _plot_region(fits_path, mask_from, region_whole(mask_from), out_path, statistic,
                         label=f"whole FoV ({statistic})", title="Whole valid FoV", wl_range=wl_range, y_range=y_range)

def plot_blank(fits_path, mask_from, out_path, mask_method="sep", statistic="mean", wl_range=None, y_range=None):
    return _plot_region(fits_path, mask_from, region_blank(mask_from, mask_method), out_path, statistic,
                         label=f"blank ({statistic})", title=f"Blank (valid & ~source, mask={mask_method} from {mask_from})",
                         wl_range=wl_range, y_range=y_range)

def plot_box(fits_path, mask_from, y_range, x_range, out_path, exclude_source=False, mask_method="sep", statistic="mean", wl_range=None, y_axis_range=None):
    region = region_box(mask_from, y_range, x_range, exclude_source, mask_method)
    return _plot_region(fits_path, mask_from, region, out_path, statistic,
                         label=f"box y={y_range} x={x_range} ({statistic})",
                         title=f"Box y={y_range} x={x_range}" + ("  (source excluded)" if exclude_source else ""),
                         wl_range=wl_range, y_range=y_axis_range)

def plot_region_overview(background, regions, out_path, title=None):
    """畫 background（2D array，例如某張 mask 或 region_whole 的 valid FoV）+ 疊上多個
       矩形 region 外框與標籤，用來確認「這個 box 在整張圖的哪裡」（例如判斷 y,x 座標順序）。
       regions = {name: (y_range, x_range)}，每個給不同顏色。適合在還沒建出實際 region
       array 之前，先比較幾個候選座標——已經有 region array 的話用下面的 plot_regions_overlay。"""
    fig, ax = plt.subplots(figsize=(7, 7 * background.shape[0] / background.shape[1]))
    finite = background[np.isfinite(background)]
    vmin, vmax = np.nanpercentile(finite, [1, 99]) if finite.size else (0, 1)
    ax.imshow(background, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, (y_range, x_range)) in enumerate(regions.items()):
        y0, y1 = y_range; x0, x1 = x_range
        color = colors[i % len(colors)]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, lw=2))
        ax.text(x0, y1, name, color=color, fontsize=9, va="bottom", ha="left",
                bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))
    if title: ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=140); plt.close(fig)
    print(f"[eval_spectrum] saved {out_path}")

def plot_regions_overlay(background, regions, out_path, title=None):
    """畫 background（2D array，例如 white_light_image() 或某張 mask）+ 疊上多個「實際」
       region 的半透明色塊（不是理想矩形，blank/box+exclude_source 這種有缺口的 region 也
       畫得出真實形狀）。regions = {name: 2D bool array}，跟 spectra_report() 吃的格式一致；
       spectra_report() 每次都會自動呼叫這支，存 regions_overview.png。"""
    fig, ax = plt.subplots(figsize=(7, 7 * background.shape[0] / background.shape[1]))
    finite = background[np.isfinite(background)]
    vmin, vmax = np.nanpercentile(finite, [1, 99]) if finite.size else (0, 1)
    ax.imshow(background, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (name, region) in enumerate(regions.items()):
        color = colors[i % len(colors)]
        overlay = np.zeros((*region.shape, 4))
        overlay[region] = (*mcolors.to_rgb(color), 0.35)
        ax.imshow(overlay, origin="lower")
        ys, xs = np.where(region)
        if len(ys):
            ax.text(xs.min(), ys.max(), name, color=color, fontsize=9, va="bottom", ha="left",
                    bbox=dict(facecolor="black", alpha=0.5, pad=1, edgecolor="none"))
    if title: ax.set_title(title)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=140); plt.close(fig)
    print(f"[eval_spectrum] saved {out_path}")

# ======================= 摘要統計：一條 spectrum → 一組純量 =======================
def spectrum_stats(spectrum, wavelengths=None, wl_range=None):
    """把一條 spectrum 陣列沿波長方向（或其中一段 wl_range=(lo,hi)）收斂成一組摘要純量
       （NaN 波長平面自動排除）。wl_range 需搭配 wavelengths 一起傳，才知道要篩哪一段。

    回傳：
      mean, sigma, skewness   — 你原本要的三個（sigma = 對自己 mean 的標準差）。
      kurtosis                — 第 4 階動差（Fisher，常態=0）。推薦原因：mean/sigma/skewness
                                是動差族的第 1-3 階，kurtosis 是自然的第 4 階；數值大代表這條
                                spectrum 有比常態雜訊更重的尾巴——對殘餘天空線來說，這通常表示
                                還留著幾根沒扣乾淨的尖刺，而不是均勻雜訊，sigma 一個數字看不出來。
      rms_from_zero           — sqrt(mean(x²))，相對「flux=0」量、不是相對自己的 mean。
                                ⚠ 純數學恆等式 rms_from_zero² = sigma² + mean²（E[X²]=Var(X)+E[X]²），
                                不含 mean/sigma 以外的新資訊，可以直接算出來，不是獨立量出來的東西。
                                留著純粹是方便：sky subtraction 的物理目標是殘餘趨近 0，不是趨近
                                「這條 spectrum 自己的平均值」，所以想一眼看「扣天空扣得好不好」時，
                                不用自己心算 sqrt(sigma²+mean²) 把系統性偏移(mean)和雜訊(sigma)
                                合在一起比較——只是個算好的捷徑，不是第四個獨立統計量。
    """
    values = np.asarray(spectrum, dtype=float)
    if wl_range is not None:
        if wavelengths is None:
            raise ValueError("spectrum_stats: wl_range 需要搭配 wavelengths 才能篩選波段")
        lo, hi = wl_range
        values = values[(wavelengths >= lo) & (wavelengths <= hi)]
    finite = values[np.isfinite(values)]
    return {"mean": float(np.mean(finite)),
            "sigma": float(np.std(finite)),
            "skewness": float(skew(finite)),
            "kurtosis": float(kurtosis(finite)),
            "rms_from_zero": float(np.sqrt(np.mean(finite ** 2)))}

def spectrum_stats_bands(spectrum, wavelengths, bands=WAVELENGTH_BANDS):
    """對同一條(已抽好的) spectrum，一次算多個波長區間的 stats，不必重讀 FITS。
       回傳 {band_name: {mean, sigma, skewness}}，預設用 WAVELENGTH_BANDS 那三段。"""
    return {name: spectrum_stats(spectrum, wavelengths, wl_range) for name, wl_range in bands.items()}

# ======================= 批次：多個 region × 多個波段，各畫一張圖 + 各算一組 stats =======================
def spectra_report(fits_path, mask_from, regions, out_dir, bands=WAVELENGTH_BANDS, statistic="mean", y_range=None):
    """regions = {region_name: 2D bool region array}。每個 region 只抽一次 spectrum（不必為
       每個 band 重讀 FITS），對每個 band 各畫一張圖、各算一組 spectrum_stats。另外自動存一張
       regions_overview.png（白光圖 + 所有 region 疊色），不用另外呼叫 plot_regions_overlay。
       y_range=(ymin,ymax) 固定所有圖的 y 軸範圍（不給則各圖各自自動縮放）。
       回傳 {region_name: {band_name: stats}}，圖存到 out_dir/spec_<region_name>_<band_name>.png。"""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    wl = wavelength_axis(mask_from)

    wl_img = white_light_image(fits_path)
    plot_regions_overlay(wl_img, regions, out_dir / "regions_overview.png",
                         title=f"White-light + regions  ({Path(fits_path).name})")

    report = {}
    for region_name, region in regions.items():
        mean_spec, median_spec = extract_spectrum(fits_path, region)
        spectrum = mean_spec if statistic == "mean" else median_spec
        report[region_name] = {}
        for band_name, wl_range in bands.items():
            out_path = out_dir / f"spec_{region_name}_{band_name}.png"
            plot_spectrum(wl, spectrum, out_path, label=f"{region_name} ({statistic})",
                          title=f"{region_name}  [{band_name}Å]", wl_range=wl_range, y_range=y_range)
            report[region_name][band_name] = spectrum_stats(spectrum, wl, wl_range)
    return report

if __name__ == "__main__":
    import json
    argv = sys.argv[1:]
    usage = ("用法: eval_spectrum.py --fits-path PATH --mask-from CUBE [--mask-method M] [--out-dir DIR] "
              "[--statistic mean|median] [--y-range YMIN YMAX] [--box NAME Y0 Y1 X0 X1 ...]\n"
              "  永遠包含 whole/blank 兩個 region；--box 可重複給多個自訂 x,y 範圍。\n"
              "  --y-range 固定所有圖的 y 軸顯示範圍（不給則各圖各自自動縮放）。")
    fits_path = mask_from = None
    mask_method, out_dir, statistic, y_range = "sep", None, "mean", None
    boxes = []
    i = 0
    while i < len(argv):
        if argv[i] == "--fits-path":
            fits_path = argv[i + 1]; i += 2
        elif argv[i] == "--mask-from":
            mask_from = argv[i + 1]; i += 2
        elif argv[i] == "--mask-method":
            mask_method = argv[i + 1]; i += 2
        elif argv[i] == "--out-dir":
            out_dir = argv[i + 1]; i += 2
        elif argv[i] == "--statistic":
            statistic = argv[i + 1]; i += 2
        elif argv[i] == "--y-range":
            y_range = (float(argv[i + 1]), float(argv[i + 2])); i += 3
        elif argv[i] == "--box":
            name, y0, y1, x0, x1 = argv[i + 1:i + 6]
            boxes.append((name, (int(y0), int(y1)), (int(x0), int(x1))))
            i += 6
        else:
            sys.exit(f"未知參數: {argv[i]}")
    if fits_path is None or mask_from is None:
        sys.exit(usage)
    if out_dir is None:
        out_dir = str(Path(fits_path).parent)

    regions = {"whole": region_whole(mask_from), "blank": region_blank(mask_from, mask_method)}
    for name, y_box_range, x_range in boxes:
        regions[name] = region_box(mask_from, y_box_range, x_range)

    report = spectra_report(fits_path, mask_from, regions, out_dir, statistic=statistic, y_range=y_range)
    report_path = Path(out_dir) / "spectra_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[eval_spectrum] saved {report_path}")
    print(json.dumps(report, indent=2))