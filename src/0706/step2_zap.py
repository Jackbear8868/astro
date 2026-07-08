"""
step2 — 用 ZAP 扣天空。指定「ZAP 對象 target」「用哪個 cube 建的 mask（mask_from）」「mask 方法」。
  輸入： settings.input_cube_path(target)                        （DATA + STAT）
         settings.source_mask_path(mask_from, mask_method)       （step1 建的 source mask）
  輸出： results/zap/cubes/<target>_maskfrom-[<method>-]<mask_from>/   （sep 方法省略前綴，claude 才加）
             zap.fits   （扣天空後的 cube；DATA + STAT，STAT 為原始照抄）
             sky.fits   （ZAP 學到並扣掉的天空）
             var.fits   （ZAP 逐波長變異曲線，診斷用）
  跑法： conda run -n astro python src/0706/step2_zap.py wsky nosky --ncpu 16
         conda run -n astro python src/0706/step2_zap.py wsky nosky --mask-method claude --ncpu 16

  2×2 = {nosky, wsky}(對象) × {nosky, wsky}(mask 來源)，共 4 次（預設 sep mask）。
  註：ZAP 正確用法是餵「還含天空的 wsky」；nosky 也可跑，兩顆同等對待、一起比較。
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "libs/zap")          # 讓 import zap 找得到（相對專案根目錄）
import settings

def zap(target, mask_from, mask_method="sep", ncpu=16):
    import zap
    mask_fits = settings.source_mask_path(mask_from, mask_method)
    if not mask_fits.exists():
        sys.exit(f"找不到 mask {mask_fits}；請先跑 step1: step1_mask.py {mask_from} {mask_method}")
    settings.ensure_run_dir(target, mask_from, mask_method)
    t0 = time.time()
    label = f"{target} (mask {mask_method} from {mask_from})"
    print(f"[step2 zap] {label} 開始 (ncpu={ncpu}) ...", flush=True)
    zap.process(str(settings.input_cube_path(target)),
                outcubefits=str(settings.zap_path(target, mask_from, mask_method)),
                skycubefits=str(settings.sky_path(target, mask_from, mask_method)),
                varcurvefits=str(settings.var_path(target, mask_from, mask_method)),
                mask=str(mask_fits), ncpu=ncpu, overwrite=True)
    print(f"[step2 zap] {label} 完成，{time.time()-t0:.0f}s -> {settings.zap_path(target, mask_from, mask_method)}", flush=True)

if __name__ == "__main__":
    argv = sys.argv[1:]
    ncpu = 16
    mask_method = "sep"
    if "--ncpu" in argv:
        i = argv.index("--ncpu"); ncpu = int(argv[i + 1]); del argv[i:i + 2]
    if "--mask-method" in argv:
        i = argv.index("--mask-method"); mask_method = argv[i + 1]; del argv[i:i + 2]
    if len(argv) < 2 or argv[0] not in settings.CUBE_NAMES or argv[1] not in settings.CUBE_NAMES:
        sys.exit("用法: step2_zap.py <target: nosky|wsky> <mask_from: nosky|wsky> [--mask-method sep|claude] [--ncpu N]")
    zap(argv[0], argv[1], mask_method, ncpu)
