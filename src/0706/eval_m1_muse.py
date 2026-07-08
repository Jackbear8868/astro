"""
M1 · MUSE convention（Weilbacher+2020 Fig 15）——「天空扣得乾不乾淨」的 W20 判準。
  中位數殘餘 vs ±1%(黑)/5%/10% 平均天空的包絡；linear、0 中心、共用 y 軸；
  全視圖 + [O I]5577 + OH 7-3 兩個 zoom。統計量 = median(W20 用 median，對離群穩健)。

  用法: conda run -n astro python src/0706/eval_m1_muse.py <target> <mask_from> [mask_method=sep|claude|box_sep|box_claude]
  讀: results/zap/cubes/<target>_maskfrom-.../zap.fits、nosky/wsky raw、mask
  寫: 同資料夾 fig_M1_muse.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings, eval_common
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

X_LABEL = r"Wavelength [$\mathrm{\AA}$]"
Y_LABEL = r"Residual flux [$10^{-20}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]"
Y_LIMIT = 5.0  # 與 W20 Fig 15 同：共用 ±5 的零中心 y 軸

def main(target, mask_from, mask_method="sep"):
    eval_common.check_args(target, mask_from, mask_method)
    blank_method = eval_common.base_method(mask_method)   # box run：殘餘在整張 base blank 上量(含邊緣)
    std_cube, _  = settings.reference_cubes(mask_from)     # 該指向的標準(已扣天空)cube：主→nosky, NE→NEnosky
    wavelengths  = eval_common.wavelength_axis(mask_from)  # 用該指向 cube 的波長軸(NE 為 3801 plane)
    _, standard_median = eval_common.standard_residual(mask_from, blank_method)   # standard raw（MUSE 標準）
    sky_mean, _        = eval_common.sky_residual(mask_from, blank_method)        # wsky raw（原始天空，% 包絡基準：W20 用 mean 天空）
    zap_cache_tag = f"{target}+zap" if mask_method == blank_method else f"{target}+zap-{mask_method}"
    _, zap_median = eval_common.blank_residual(settings.zap_path(target, mask_from, mask_method), mask_from, zap_cache_tag, blank_method)

    fig = plt.figure(figsize=(15, 4.8))
    grid = fig.add_gridspec(1, 3, width_ratios=[3, 1, 1])
    panels = [(None, "full range"), ((5560, 5595), "[O I] 5577"), ((8755, 8810), "OH 7-3 band")]
    for panel_index, (wavelength_range, panel_title) in enumerate(panels):
        ax = fig.add_subplot(grid[0, panel_index])
        for percent, shade in [(0.10, "0.85"), (0.05, "0.62"), (0.01, "0.0")]:   # 10 淺 / 5 深 / 1 黑
            ax.fill_between(wavelengths, -percent*np.abs(sky_mean), percent*np.abs(sky_mean), color=shade, lw=0, zorder=0)
        ax.axhline(0, color="0.5", lw=0.4, zorder=1)
        ax.plot(wavelengths, standard_median, color="tab:blue", lw=0.7, alpha=0.8, zorder=3, label=f"{std_cube} (MUSE standard)")
        ax.plot(wavelengths, zap_median,       color="tab:red",  lw=0.9, zorder=5, label=f"{target}+ZAP (median)")
        ax.set_xlim(wavelengths.min(), wavelengths.max()) if wavelength_range is None else ax.set_xlim(*wavelength_range)
        ax.set_ylim(-Y_LIMIT, Y_LIMIT); ax.set_xlabel(X_LABEL, fontsize=8.5); ax.tick_params(labelsize=7.5); ax.set_title(panel_title, fontsize=9)
    left_panel = fig.axes[0]; left_panel.set_ylabel(Y_LABEL, fontsize=8.5); left_panel.legend(loc="upper left", fontsize=6.8, framealpha=0.9)
    fig.suptitle("Sky-subtraction residuals  ·  MUSE convention (Weilbacher+2020 Fig 15)", fontsize=13, y=0.99)
    fig.text(0.5, 0.905, f"target = {target}  ·  mask = {mask_method} from {mask_from}  ·  residual over full-field {blank_method} blank",
             ha="center", fontsize=8.5, color="0.35")
    fig.text(0.995, 0.01, "bands: ±1 % (black) / 5 % / 10 % of mean sky   ·   criterion: continuum <1 %, lines <5 % (goal 2 %)",
             ha="right", fontsize=7, color="0.4")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path = settings.run_dir(target, mask_from, mask_method) / "fig_M1_muse.png"
    fig.savefig(str(out_path), dpi=145); plt.close(fig)
    print(f"[M1 muse] saved {out_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3: sys.exit("用法: eval_m1_muse.py <target> <mask_from> [mask_method=sep|claude|box_sep|box_claude]")
    main(*sys.argv[1:4])
