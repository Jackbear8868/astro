"""
zap — 用 ZAP 扣天空。指定「ZAP 對象 target」「用哪個 cube 建的 mask（mask_from）」「mask 方法」。
  輸入： settings.input_cube_path(target)                        （DATA + STAT）
         settings.source_mask_path(mask_from, mask_method)       （已建好的 source mask）
  輸出： results/zap/cubes/<target>_maskfrom-[<method>-]<mask_from>/   （sep 方法省略前綴，claude 才加）
             zap.fits   （扣天空後的 cube；DATA + STAT，STAT 為原始照抄）
             sky.fits   （ZAP 學到並扣掉的天空）
             var.fits   （ZAP 逐波長變異曲線，診斷用）
  跑法： conda run -n astro python src/zap/zap.py --target wsky --mask-from nosky --ncpu 16
         conda run -n astro python src/zap/zap.py --target wsky --mask-from nosky --mask-method claude --ncpu 16

  2×2 = {nosky, wsky}(對象) × {nosky, wsky}(mask 來源)，共 4 次（預設 sep mask）。
  註：ZAP 正確用法是餵「還含天空的 wsky」；nosky 也可跑，兩顆同等對待、一起比較。
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "libs/zap")          # 讓 import zap 找得到（相對專案根目錄）
import settings

def zap(target, mask_from, mask_method="sep", ncpu=16, cfwidthsp=300):
    import zap
    mask_fits = settings.source_mask_path(mask_from, mask_method)
    if not mask_fits.exists():
        sys.exit(f"找不到 mask {mask_fits}；請先建立這個 mask")
    settings.ensure_run_dir(target, mask_from, mask_method, cfwidthsp)
    t0 = time.time()
    label = f"{target} (mask {mask_method} from {mask_from}, cfwidthSP={cfwidthsp})"
    print(f"[zap] {label} 開始 (ncpu={ncpu}) ...", flush=True)
    # cfwidthSVD 維持 ZAP 預設 300（SVD 基底用寬連續譜濾波）；只調 cfwidthSP（per-spectrum 特徵值用的視窗）。
    zap.process(str(settings.input_cube_path(target)),
                outcubefits=str(settings.zap_path(target, mask_from, mask_method, cfwidthsp)),
                skycubefits=str(settings.sky_path(target, mask_from, mask_method, cfwidthsp)),
                varcurvefits=str(settings.var_path(target, mask_from, mask_method, cfwidthsp)),
                mask=str(mask_fits), ncpu=ncpu, cfwidthSP=cfwidthsp, overwrite=True)
    print(f"[zap] {label} 完成，{time.time()-t0:.0f}s -> {settings.zap_path(target, mask_from, mask_method, cfwidthsp)}", flush=True)

if __name__ == "__main__":
    argv = sys.argv[1:]
    target = mask_from = None
    mask_method, ncpu, cfwidthsp = "sep", 16, 300
    i = 0
    while i < len(argv):
        if argv[i] == "--target":
            target = argv[i + 1]; i += 2
        elif argv[i] == "--mask-from":
            mask_from = argv[i + 1]; i += 2
        elif argv[i] == "--mask-method":
            mask_method = argv[i + 1]; i += 2
        elif argv[i] == "--ncpu":
            ncpu = int(argv[i + 1]); i += 2
        elif argv[i] == "--cfwidthsp":
            cfwidthsp = int(argv[i + 1]); i += 2
        else:
            sys.exit(f"未知參數: {argv[i]}")
    usage = (f"用法: zap.py --target <{'|'.join(settings.CUBE_NAMES)}> --mask-from <{'|'.join(settings.CUBE_NAMES)}> "
             "[--mask-method sep|claude] [--ncpu N] [--cfwidthsp N]\n"
             "  --cfwidthsp：ZAP per-spectrum 連續譜濾波視窗（預設 300；ZAP 建議 20–50 更貼合源）。")
    if target not in settings.CUBE_NAMES or mask_from not in settings.CUBE_NAMES:
        sys.exit(usage)
    zap(target, mask_from, mask_method, ncpu, cfwidthsp)
