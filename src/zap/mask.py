"""
mask — 讀取既有 source mask，做後處理，存成新方法名（不覆蓋原始版本）。
  目前處理：右下角四分之一整塊強制設為 source（=1），修掉延展源在該角落的鋸齒/缺口邊界。
  輸入： settings.source_mask_path(from_cube, method)
  輸出： settings.source_mask_path(from_cube, f"{method}_{out_suffix}")   （預設 out_suffix="brq"）
         同資料夾下 mask_compare.png（before/after 對照圖，process() 自動存，不用另外呼叫畫圖函式）
  跑法： conda run -n astro python src/zap/mask.py --from-cube NEnosky --method seg1sigma
         conda run -n astro python src/zap/mask.py --from-cube NEnosky --method seg2sigma

  想另外畫任意 mask（例如比較不同 cube 的同一個 method），可直接呼叫畫圖函式：
    import sys; sys.path.insert(0, "src/zap")
    import mask
    mask.plot_mask("results/zap/masks/seg1sigma_brq_from-NEnosky/mask.fits", "out.png")
    mask.plot_mask_grid([("path1.fits", "title1"), ("path2.fits", "title2")], "out_grid.png")
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings
import numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def mask_bottom_right_quarter(mask):
    """右下四分之一（origin='lower'：列前半=下, 欄後半=右）強制設 1；其餘保留原值。"""
    ny, nx = mask.shape
    out = mask.copy()
    out[:ny // 2, nx // 2:] = 1
    return out

def process(from_cube, method, out_suffix="brq"):
    src_path = settings.source_mask_path(from_cube, method)
    mask = fits.getdata(str(src_path)).astype(np.uint8)
    out = mask_bottom_right_quarter(mask)
    out_dir = settings.ensure_mask_dir(from_cube, f"{method}_{out_suffix}")
    out_path = out_dir / "mask.fits"
    fits.PrimaryHDU(out).writeto(str(out_path), overwrite=True)
    print(f"[mask] {src_path} -> {out_path}  (+{int(out.sum()) - int(mask.sum())} px 被強制設為 source)")
    fig_path = out_dir / "mask_compare.png"
    plot_mask_grid([(str(src_path), f"{method} (before)"), (str(out_path), f"{method}_{out_suffix} (after)")], fig_path)
    return out_path

def plot_mask(mask_fits_path, out_path, title=None):
    """讀一張 mask.fits，畫黑白示意圖存檔（白=source/排除，黑=可用天空）。"""
    data = fits.getdata(str(mask_fits_path))
    fig, ax = plt.subplots(figsize=(6, 6 * data.shape[0] / data.shape[1]))
    ax.imshow(data, origin="lower", cmap="gray", vmin=0, vmax=1)
    ax.set_title(title or str(mask_fits_path))
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=130); plt.close(fig)
    print(f"[mask] saved {out_path}")

def plot_mask_grid(mask_specs, out_path, ncols=2):
    """mask_specs = [(mask_fits_path, title), ...]，排成 grid 存成一張比較圖。"""
    nrows = -(-len(mask_specs) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (mask_fits_path, title) in zip(axes, mask_specs):
        data = fits.getdata(str(mask_fits_path))
        ax.imshow(data, origin="lower", cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
    for ax in axes[len(mask_specs):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=130); plt.close(fig)
    print(f"[mask] saved {out_path}")

if __name__ == "__main__":
    argv = sys.argv[1:]
    from_cube = method = None
    out_suffix = "brq"
    i = 0
    while i < len(argv):
        if argv[i] == "--from-cube":
            from_cube = argv[i + 1]; i += 2
        elif argv[i] == "--method":
            method = argv[i + 1]; i += 2
        elif argv[i] == "--out-suffix":
            out_suffix = argv[i + 1]; i += 2
        else:
            sys.exit(f"未知參數: {argv[i]}")
    usage = f"用法: mask.py --from-cube <{'|'.join(settings.CUBE_NAMES)}> --method <method> [--out-suffix brq]"
    if from_cube not in settings.CUBE_NAMES or method is None:
        sys.exit(usage)
    process(from_cube, method, out_suffix)
