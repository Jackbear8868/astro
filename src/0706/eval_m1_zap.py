"""
M1 · ZAP convention（Soto+2016 Fig 1）——「標準處理 vs ZAP」的雙 panel 對照。
  左 panel = Standard processing (MUSE pipeline) = nosky raw；右 panel = ZAP = target+ZAP。
  雙軸：左=殘餘 flux，右=原始天空(灰線，wsky raw)；linear；無誤差帶（原文沒有畫任何雜訊帶）。

  mean 與 median 各畫一張（拆開，分別看）:
    fig_M1_zap_mean.png    統計量 = mean  （Soto 原文；在大遮罩上被離群 spaxel 主導 → 天空線尖刺）
    fig_M1_zap_median.png  統計量 = median（對離群穩健，乾淨）
  註：nosky（左）本來就已被 MUSE 扣過天空，mean≈median（僅差 ~0.1），差異主要在 ZAP（右）側。

  y 軸固定（不自動縮放），mean/median 各用適合自己的範圍，各 run 之間可直接比較：
    mean   圖左軸 ±MEAN_Y_LIMIT  （大，容納天空線尖刺）
    median 圖左軸 ±MEDIAN_Y_LIMIT（小，放大才看得出 nosky/ZAP 結構）
    右軸天空一律 0..SKY_Y_LIMIT。

  用法: conda run -n astro python src/0706/eval_m1_zap.py <target> <mask_from> [mask_method=sep|claude|box_sep|box_claude]
  讀: results/zap/cubes/<target>_maskfrom-.../zap.fits、nosky/wsky raw、mask
  寫: 同資料夾 fig_M1_zap_mean.png, fig_M1_zap_median.png
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings, eval_common
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

X_LABEL   = r"Wavelength [$\mathrm{\AA}$]"
Y_LABEL   = r"Residual flux [$10^{-20}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]"
SKY_LABEL = r"Original sky flux (grey) [$10^{-20}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}}$]"

MEAN_Y_RANGE   = (-10, 25)    # mean 圖左軸範圍
MEDIAN_Y_RANGE = (-10, 25)    # median 圖左軸範圍
SKY_Y_LIMIT    = 1500   # 右軸(原始天空)固定 0..（天空峰值 ~1400）
SHOW_SKY       = False  # 暫時關閉右軸灰色「原始天空」；改回 True 即恢復

def _draw(out_path, statistic, target, mask_from, mask_method, wavelengths,
          standard_residual, zap_residual, sky_spectrum, residual_y_range, std_cube="nosky"):
    """單張 ZAP-convention 圖：statistic ∈ {'mean','median'}；左軸固定 residual_y_range=(low,high)。"""
    fig, (standard_panel, zap_panel) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    panels = [(standard_panel, f"Standard processing (MUSE pipeline)  =  {std_cube}", standard_residual, "tab:blue"),
              (zap_panel,      f"ZAP  =  {target} + ZAP",                             zap_residual,      "tab:red")]
    for ax, panel_title, residual_spectrum, color in panels:
        if SHOW_SKY:
            sky_axis = ax.twinx()
            sky_axis.plot(wavelengths, sky_spectrum, color="0.62", lw=0.6, zorder=1)
            sky_axis.set_ylim(-SKY_Y_LIMIT*0.03, SKY_Y_LIMIT)                  # 右軸固定
            if ax is zap_panel:
                sky_axis.set_ylabel(SKY_LABEL, color="0.45", fontsize=8.5); sky_axis.tick_params(axis="y", labelcolor="0.5")
            else:
                sky_axis.set_yticklabels([])
            ax.set_zorder(sky_axis.get_zorder()+1); ax.patch.set_visible(False)
        ax.axhline(0, color="k", lw=0.4, zorder=2)
        ax.plot(wavelengths, residual_spectrum, color=color, lw=0.9, zorder=4)
        ax.set_xlim(wavelengths.min(), wavelengths.max()); ax.set_ylim(*residual_y_range)  # 左軸固定
        ax.set_xlabel(X_LABEL, fontsize=9); ax.set_title(panel_title, fontsize=10)
    standard_panel.set_ylabel(Y_LABEL, fontsize=9)
    fig.suptitle(f"Sky-subtraction residuals  ·  ZAP convention (Soto+2016 Fig 1)  ·  statistic = {statistic}  (y {residual_y_range[0]}..{residual_y_range[1]})",
                 fontsize=13, y=0.99)
    sky_note = "grey = original sky (right axis)  ·  " if SHOW_SKY else ""
    fig.text(0.5, 0.925, f"target = {target}  ·  mask = {mask_method} from {mask_from}  ·  {sky_note}linear, no error band",
             ha="center", fontsize=8.5, color="0.35")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(str(out_path), dpi=145); plt.close(fig)
    print(f"[M1 zap/{statistic}] saved {out_path}")

def main(target, mask_from, mask_method="sep"):
    eval_common.check_args(target, mask_from, mask_method)
    blank_method = eval_common.base_method(mask_method)   # box run：殘餘在整張 base blank 上量(含邊緣)
    std_cube, _  = settings.reference_cubes(mask_from)     # 該指向的標準(已扣天空)cube：主→nosky, NE→NEnosky
    wavelengths  = eval_common.wavelength_axis(mask_from)  # 用該指向 cube 的波長軸(NE 為 3801 plane)
    standard_mean, standard_median = eval_common.standard_residual(mask_from, blank_method)
    sky_mean,      sky_median      = eval_common.sky_residual(mask_from, blank_method)
    zap_cache_tag = f"{target}+zap" if mask_method == blank_method else f"{target}+zap-{mask_method}"
    zap_mean, zap_median = eval_common.blank_residual(settings.zap_path(target, mask_from, mask_method), mask_from, zap_cache_tag, blank_method)
    run_folder = settings.run_dir(target, mask_from, mask_method)
    _draw(run_folder/"fig_M1_zap_mean.png",   "mean",   target, mask_from, mask_method, wavelengths, standard_mean,   zap_mean,   sky_mean, MEAN_Y_RANGE,   std_cube)
    _draw(run_folder/"fig_M1_zap_median.png", "median", target, mask_from, mask_method, wavelengths, standard_median, zap_median, sky_mean, MEDIAN_Y_RANGE, std_cube)

if __name__ == "__main__":
    if len(sys.argv) < 3: sys.exit("用法: eval_m1_zap.py <target> <mask_from> [mask_method=sep|claude|box_sep|box_claude]")
    main(*sys.argv[1:4])
