from pathlib import Path
import numpy as np
from astropy.io import fits
import argparse

import matplotlib
matplotlib.use("Agg")              # 關鍵：先設定「畫到檔案」模式
import matplotlib.pyplot as plt    # 畫圖的主介面，慣例縮寫成 plt

ROOT = Path(__file__).resolve().parents[2]

def main():
    ap = argparse.ArgumentParser(description="cube → 白光影像（whitelight.fits + 預覽 png）")
    ap.add_argument("cube", type=Path, help="輸入 cube（.fits）")
    ap.add_argument("--out", type=Path, default=None, help="輸出資料夾（預設 results/skymodel/<cube檔名>/）")
    args = ap.parse_args()
    out = args.out or ROOT / "results/skymodel" / args.cube.stem
    out.mkdir(parents=True, exist_ok=True)

    NOSKY_CUBE = args.cube
    WHITE_LIGHT_CUBE = out / "whitelight.fits"
    WHITE_LIGHT_IMAGE = out / "whitelight.png"

    with fits.open(NOSKY_CUBE) as hdul: # hdu list
        data = hdul["DATA"].data
        white = np.nanmean(data, axis=0)
        white = np.nan_to_num(white, nan=0.0)
        fits.writeto(WHITE_LIGHT_CUBE, white, overwrite=True)

        plt.figure(figsize=(6, 6))                              # 開一張畫布，6x6 吋
        plt.imshow(white, origin="lower", cmap="gray",
                vmin=np.nanpercentile(white, 5),
                vmax=np.nanpercentile(white, 99))            # 把 2D 陣列畫成灰階影像
        plt.colorbar()                                          # 旁邊加一條顏色對照尺
        plt.savefig(WHITE_LIGHT_IMAGE, dpi=130)                 # 把畫布存成 PNG


if __name__ == "__main__":
    main()