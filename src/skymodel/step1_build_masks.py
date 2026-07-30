from pathlib import Path
import numpy as np
from astropy.io import fits
import argparse

import matplotlib
matplotlib.use("Agg")              # 關鍵：先設定「畫到檔案」模式
import matplotlib.pyplot as plt    # 畫圖的主介面，慣例縮寫成 plt

ROOT = Path(__file__).resolve().parents[2]

def main():
    ap = argparse.ArgumentParser(description="whitelight+seg → 遮罩；從 wsky cube 抽 blank 平均天空譜")
    ap.add_argument("dir", type=Path, help="per-cube 工作資料夾（內含 whitelight.fits、seg.fits；輸出也放這）")
    ap.add_argument("--wsky", type=Path, required=True, help="含天空的 cube（抽 blank 光譜用）")
    args = ap.parse_args()

    WSKY_CUBE = args.wsky

    white = fits.getdata(args.dir / "whitelight.fits")   # 每格的平均亮度（視場外=0）
    seg   = fits.getdata(args.dir / "seg.fits")           # 每格屬於第幾號源（0=背景

    valid_mask  = white != 0        # 白光 ≠ 0 → 視場內（因為視場外被我們填成 0 了）
    source_mask = (seg > 0) & valid_mask # seg > 0 → 有源
    blank_mask  = valid_mask & ~source_mask   # 視場內、而且不是源 → 純天空
    
    ids, counts = np.unique(seg[source_mask], return_counts=True)
    print(f"sources: {len(ids)}, source spaxels: {counts.sum()}, blank spaxels: {blank_mask.sum()}")

    order = np.argsort(counts)[::-1]
    for i in order[:15]:
        print(f"  ID {ids[i]:>4d}  {counts[i]:>6d} spaxels   SNR x{np.sqrt(counts[i]):.1f}")
        
    for cut in (13, 50, 100, 300):
        print(f"  >= {cut:>4d} spaxels: {(counts >= cut).sum():>4d} sources, {counts[counts >= cut].sum():>6d} spaxels")

    with fits.open(WSKY_CUBE) as hdul:
        hdr = hdul["DATA"].header
        nz = hdr["NAXIS3"]
        wavelength_range = hdr["CRVAL3"] + (np.arange(nz) + 1 - hdr["CRPIX3"]) * hdr["CD3_3"]
        mean_wsky = np.empty(nz, np.float32)

        for j in range(0, nz, 200):
            data = np.asarray(hdul["DATA"].data[j:j+200], np.float32)   # 一次讀 200 個波長平面避免記憶體爆掉
            mean_wsky[j:j+200] = np.nanmean(data[:, blank_mask], axis=1)      # 抽 blank + 平均
        
        plt.figure(figsize=(13, 4))
        plt.plot(wavelength_range, mean_wsky, lw=0.5)
        plt.xlabel("wavelength [A]")
        plt.ylabel("flux")
        plt.title("mean sky spectrum from blank spaxels (WSKY cube)")
        plt.tight_layout()
        plt.savefig(args.dir / "mean_wsky.png", dpi=130)
        print("saved", args.dir / "mean_wsky.png")
        np.save(args.dir / "mean_wsky.npy", mean_wsky)
        print("saved", args.dir / "mean_wsky.npy")


if __name__ == "__main__":
    main()