"""一次算好 SVD (interactive)，再用 reprocess() 快速掃不同特徵譜數量 (nevals)，
找出「源保真 + 雜訊不灌入 + 天空線殘餘下降」的最佳成分數，並寫出最終 nosky_zap.fits。"""
import sys, warnings, numpy as np
sys.path.insert(0, "libs/zap"); warnings.filterwarnings("ignore")
from astropy.io import fits
import zap

OUT = "results/zap"
c = np.load(f"{OUT}/_cache.npz"); wl = c["wl"]; by, bx = c["by"], c["bx"]
sy, sx = int(c["sy"]), int(c["sx"])
W = lambda a, b: (wl > a) & (wl < b)
cont = W(7000, 7120)                       # line-free → 純雜訊
li, cb = W(6692, 6708), (W(6660, 6688) | W(6730, 6758))
ks = {lam: int(np.argmin(abs(wl - lam))) for lam in (5577, 6300)}

raw = fits.open(f"{OUT}/crop_nosky.fits")["DATA"].data
def metrics(cube):
    s = cube[cont][:, by, bx]; s = s - np.nanmedian(s, axis=0)
    noise = float(np.nanmedian(np.nanstd(s, axis=0)))           # 每spaxel line-free RMS
    skystd = {lam: float(np.nanstd(cube[k, by, bx])) for lam, k in ks.items()}  # 天空線spatial std
    sp = cube[:, sy, sx]; base = np.nanmean(sp[cb])
    ha = float(np.nansum(sp[li] - base))                        # 源 Hα 積分
    return noise, skystd, ha
n0, sky0, ha0 = metrics(raw)
print(f"[raw ] noise={n0:.2f}  sky5577_std={sky0[5577]:.1f}  sky6300_std={sky0[6300]:.1f}  Ha={ha0:.0f}")

print("計算 SVD (約 30 分鐘) ...", flush=True)
zobj = zap.process(f"{OUT}/crop_nosky.fits", mask=f"{OUT}/source_mask.fits",
                   ncpu=1, interactive=True, overwrite=True)
print(f"自動選擇 nevals = {zobj.nevals}", flush=True)

rows = []
for N in [3, 5, 8, 10, 12, 15, 20, 25, 30, 40, int(zobj.nevals[0])]:
    zobj.reprocess(nevals=[N])
    n, sky, ha = metrics(zobj.cleancube)
    rows.append((N, n, sky[5577], sky[6300], 100*ha/ha0))
    print(f"  N={N:3d}: noise={n:5.2f} (raw {n0:.2f}) | sky5577_std={sky[5577]:6.1f} "
          f"sky6300_std={sky[6300]:6.1f} (raw {sky0[5577]:.1f}/{sky0[6300]:.1f}) | Ha保留={100*ha/ha0:5.1f}%", flush=True)

# 選 N：源保留≥98% 且 line-free 雜訊增幅最小（雜訊<1.5×raw 優先），同時天空線spatial std 不高於 raw
good = [r for r in rows if r[4] >= 98 and r[1] <= 1.5*n0 and r[2] <= sky0[5577] and r[3] <= sky0[6300]]
pick = max(good, key=lambda r: r[0]) if good else min(rows, key=lambda r: r[1]/n0 + abs(100-r[4])/2)
N = pick[0]
print(f"\n==> 選定 nevals = {N}  (noise={pick[1]:.2f}, Ha保留={pick[4]:.1f}%)", flush=True)
zobj.reprocess(nevals=[N])
zobj.writeskycube(skycubefits=f"{OUT}/nosky_skyremoved.fits", overwrite=True)
zobj.mergefits(f"{OUT}/nosky_zap.fits", overwrite=True)
print(f"已寫出 {OUT}/nosky_zap.fits (nevals={N})")
