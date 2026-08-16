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
    # --out 沒有預設值。原本會用 cube 的檔名開一個新目錄 —— 那讓「餵錯一個 cube」
    # 的代價變成「results/ 底下多一個沒人知道是什麼的資料夾」,而且不會有任何提示。
    ap.add_argument("--out", type=Path, required=True, help="輸出資料夾")
    args = ap.parse_args()
    out = args.out
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