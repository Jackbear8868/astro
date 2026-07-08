"""
eval_mask_compare — 2×2 遮罩比較：兩種方法(sep/claude) × 兩個 cube(nosky/wsky)。
  輸入： settings.input_cube_path("nosky"), input_cube_path("wsky")
  輸出： results/zap/masks/<method>_from-<cube>/mask.fits（共 4 個，沿用 step1 結構）
         results/zap/fig_mask_compare_2x2.png
         終端機：覆蓋率 2×2 表 + 延伸半徑 + 兩軸各自的 IoU
  跑法： conda run -n astro python src/0706/eval_mask_compare.py

  同時回答兩個問題：
    (橫看) cube 差別 ── 用 wsky 自己建遮罩，跟借乾淨的 nosky 差多少？（暗暈有沒有漏）
    (直看) 方法 差別 ── claude(robust-MAD) 跟 sep(SEP+matched filter) 圈出的源差多少？
  註：只寫 mask.fits，不動 blanks.npz（那由 step1 產生）。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings
import numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import step1_mask

METHODS = ("sep", "claude")

def compare():
    settings.ensure_output_dirs()

    # ---- 建 4 個遮罩（2 方法 × 2 cube），存檔並記錄 ----
    masks = {}
    for method in METHODS:
        for cube_name in settings.CUBE_NAMES:
            src, white, ha, valid, wl = step1_mask.build_mask(cube_name, method)
            settings.ensure_mask_dir(cube_name, method)
            fits.writeto(str(settings.source_mask_path(cube_name, method)), src.astype(np.uint8), overwrite=True)
            masks[(method, cube_name)] = dict(src=src, white=white, ha=ha, valid=valid)
            print(f"[{method:6s} / {cube_name:5s}] 覆蓋 {100*src.sum()/valid.sum():5.1f}%  ({int(src.sum())} px)")

    # 共同底圖/中心：用 nosky 乾淨 Hα、nosky 白光最亮點
    ref = masks[("sep", "nosky")]
    ha_ref = ref["ha"]
    cy, cx = np.unravel_index(np.nanargmax(np.where(ref["valid"], ref["white"], np.nan)), ha_ref.shape)
    yy, xx = np.mgrid[0:ha_ref.shape[0], 0:ha_ref.shape[1]]
    r = np.hypot(yy - cy, xx - cx) * settings.PIXEL_SCALE_ARCSEC

    def cov(method, cube_name):  return 100 * masks[(method, cube_name)]["src"].sum() / masks[(method, cube_name)]["valid"].sum()
    def rext(method, cube_name):
        s = masks[(method, cube_name)]["src"] & masks[(method, cube_name)]["valid"]
        return float(r[s].max()) if s.any() else 0.0
    def iou(k1, k2):
        v = masks[k1]["valid"] & masks[k2]["valid"]
        a, b = masks[k1]["src"], masks[k2]["src"]
        return int((a & b & v).sum()) / max(int(((a | b) & v).sum()), 1)

    # ---- 數字 ----
    print("\n===== 覆蓋率 2×2（%）=====")
    print(f"{'方法(列)':<12s}{'nosky':>10s}{'wsky':>10s}")
    for method in METHODS:
        print(f"{method:<12s}{cov(method,'nosky'):>10.1f}{cov(method,'wsky'):>10.1f}")
    print("\n===== 遮罩最遠半徑 2×2（arcsec）=====")
    print(f"{'方法(列)':<12s}{'nosky':>10s}{'wsky':>10s}")
    for method in METHODS:
        print(f"{method:<12s}{rext(method,'nosky'):>10.1f}{rext(method,'wsky'):>10.1f}")
    print("\n===== 兩軸差異（IoU＝交集/聯集，越接近 1 越像）=====")
    print(f"  方法差異  sep vs claude (固定 nosky) = {iou(('sep','nosky'), ('claude','nosky')):.2f}")
    print(f"  方法差異  sep vs claude (固定 wsky ) = {iou(('sep','wsky'),  ('claude','wsky')):.2f}")
    print(f"  cube差異  nosky vs wsky (固定 sep )  = {iou(('sep','nosky'), ('sep','wsky')):.2f}")
    print(f"  cube差異  nosky vs wsky (固定 claude)= {iou(('claude','nosky'), ('claude','wsky')):.2f}")

    # ---- 圖：2×2 輪廓（列=方法, 欄=cube），全部疊在同一張 nosky Hα 上 ----
    sig  = 1.4826 * np.median(np.abs(ha_ref[ref["valid"]] - np.median(ha_ref[ref["valid"]])))
    vmax = np.percentile(ha_ref[ref["valid"]], 99)
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    for i, method in enumerate(METHODS):
        for j, cube_name in enumerate(settings.CUBE_NAMES):
            ax = axes[i][j]
            ax.imshow(ha_ref, origin="lower", vmin=-sig, vmax=vmax, cmap="gray")
            ax.contour(masks[(method, cube_name)]["src"].astype(float), levels=[0.5], colors="red", linewidths=0.9)
            ax.plot(cx, cy, "y+", ms=8)
            ax.set_title(f"{method} / {cube_name}   cover={cov(method,cube_name):.0f}%  r_max={rext(method,cube_name):.0f}\"")
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Source mask 2×2  —  rows: method(sep/claude)   cols: cube(nosky/wsky)   "
                 "(contour over nosky Hα)", fontsize=12)
    fig.tight_layout()
    out = settings.FIGURE_DIR / "fig_mask_compare_2x2.png"
    fig.savefig(str(out), dpi=120); plt.close(fig)
    print(f"\nsaved {out}")

if __name__ == "__main__":
    compare()
