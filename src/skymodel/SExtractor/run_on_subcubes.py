"""
run_on_subcubes — 對 mosaic 的 14 個 sub cube（nosky）各跑一次教授的 SExtractor 工作流。

  流程（= README.md §3/§5 的配方，逐 cube 重複；default.sex 原封不動）：
    1. 讀本地 data/nosky/DATACUBE_FINAL_ESOSKY_<n>.fits（14 顆已全數在本地）
    2. 壓 2D 偵測影像：det_white = 全譜 nanmean，無效像素填 0（SExtractor 不吃 NaN）
       （教授指示：SExtractor 偵測影像用 whitelight）
    3. 在本資料夾跑 `sex det_white.fits -c default.sex`
    4. 收 seg.fits / test.cat / nosky.fits ＋ det 影像 → results/skymodel/sextractor_sub/<nn>/

  已有 seg.fits 的 sub cube 自動跳過（可中斷重跑）。
  用法：conda run -n astro python src/skymodel/SExtractor/run_on_subcubes.py
"""
import shutil, subprocess, time
from pathlib import Path
import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[3]

SEXDIR = Path(__file__).resolve().parent
DATADIR = ROOT / "data" / "nosky"                        # 14 顆 DATACUBE_FINAL_ESOSKY_<n>.fits
OUTBASE = ROOT / "results/skymodel/sextractor_sub"
N_SUB = 14

T0 = time.time()
def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def detection_image(cube_path):
    """cube → det_white（whitelight＝全譜 nanmean；README §5 配方）。"""
    with fits.open(cube_path, memmap=True) as h:
        hd = h["DATA"] if "DATA" in h else h[1]
        nz = hd.shape[0]
        s = c = None
        for j in range(0, nz, 400):
            b = np.asarray(hd.data[j:j + 400], np.float32)
            if s is None:
                s = np.zeros(b.shape[1:], np.float64); c = np.zeros(b.shape[1:], np.int64)
            s += np.nansum(b, 0); c += np.isfinite(b).sum(0)
    return np.where(c > 0, s / np.maximum(c, 1), 0.0).astype(np.float32)


def main():
    for n in range(1, N_SUB + 1):
        out = OUTBASE / f"{n:02d}"
        if (out / "seg.fits").exists():
            log(f"sub {n:02d}：已有 seg.fits，跳過")
            continue
        out.mkdir(parents=True, exist_ok=True)
        cube = DATADIR / f"DATACUBE_FINAL_ESOSKY_{n}.fits"
        if not cube.exists():
            log(f"sub {n:02d}：找不到 {cube.name}，跳過")
            continue

        log(f"sub {n:02d}：壓偵測影像 …")
        fits.writeto(out / "det_white.fits", detection_image(cube), overwrite=True)

        log(f"sub {n:02d}：SExtractor …")
        r = subprocess.run(["sex", str((out / "det_white.fits").resolve()), "-c", "default.sex"],
                           cwd=SEXDIR, capture_output=True, text=True)
        if r.returncode != 0:
            log(f"sub {n:02d}：sex 失敗：{r.stderr.strip()[:300]}")
        else:
            for f in ("seg.fits", "test.cat", "nosky.fits"):
                shutil.move(SEXDIR / f, out / f)
            seg = fits.getdata(out / "seg.fits")
            log(f"sub {n:02d}：完成  objects={int(seg.max())}  覆蓋 {100*(seg>0).mean():.1f}%")
    log("全部完成")


if __name__ == "__main__":
    main()
