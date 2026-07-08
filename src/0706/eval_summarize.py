"""
eval_summarize — 把「一個 cube 的 raw + 它的 zap」壓成小摘要（fits → npz）。
  輸入： settings.input_cube_path(cube_name)  +  fits/{cube_name}_zap.fits（若存在）  +  npz/_cache.npz
  輸出： results/zap/npz/summ_{cube_name}.npz
           raw_med / raw_std / raw_srcspec / raw_img6300   （原始 cube）
           zap_med / zap_std / zap_srcspec / zap_img6300   （ZAP 後 cube，若有跑過）
  跑法： conda run -n astro python src/0706/eval_summarize.py wsky

  這是 evaluate 的「reduce」階段：把幾 GB 的大 cube 縮成幾 MB 的摘要，
  之後 scores / figures 只讀這些小 npz，完全不用再開大 cube。
  兩個 cube 都 summarize 後，raw/zap × nosky/wsky 就湊齊「四方數值」。
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings
import numpy as np
from astropy.io import fits

def _summarize(path, by, bx, sy, sx, k6300):
    """打開一個 cube，用座標抽出 4 個小陣列後回傳（隨即釋放大 cube）。"""
    d = fits.open(str(path))["DATA"].data                  # (波長平面, y, x)
    blankspec = d[:, by, bx]                               # 空白天空光譜 (波長, Nblank)
    out = dict(
        med=np.nanmedian(blankspec, axis=1).astype(np.float32),   # 中位光譜：天空線扣乾淨沒
        std=np.nanstd(blankspec, axis=1).astype(np.float32),      # 逐波長 std：殘餘雜訊
        srcspec=d[:, sy, sx].astype(np.float32),                  # 最亮 spaxel 光譜：源保真
        img6300=d[k6300].astype(np.float32))                      # 6300Å 影像：診斷用
    del d, blankspec
    return out

def summarize(cube_name):
    c = np.load(str(settings.CACHE_NPZ_PATH)); wl = c["wl"]
    k = int(np.argmin(abs(wl - 6300)))

    # raw（原始 cube）一定抽
    raw = _summarize(settings.input_cube_path(cube_name), c["by"], c["bx"], int(c["sy"]), int(c["sx"]), k)
    out = {f"raw_{kk}": v for kk, v in raw.items()}

    # zap（ZAP 後 cube）有跑過才抽
    if settings.zap_cube_path(cube_name).exists():
        zp = _summarize(settings.zap_cube_path(cube_name), c["by"], c["bx"], int(c["sy"]), int(c["sx"]), k)
        out.update({f"zap_{kk}": v for kk, v in zp.items()})
        print(f"[summarize] {cube_name}: raw + zap 都抽好")
    else:
        print(f"[summarize] {cube_name}: 只有 raw（找不到 {settings.zap_cube_path(cube_name)}，還沒跑 step2）")

    np.savez(str(settings.summary_npz_path(cube_name)), **out)
    print(f"[summarize] 存 {settings.summary_npz_path(cube_name)}")

if __name__ == "__main__":
    cube_names = [sys.argv[1]] if len(sys.argv) > 1 else list(settings.CUBE_NAMES)
    for cube_name in cube_names:
        summarize(cube_name)
