"""
run_sextractor — 對任意 cube 各跑一次 SExtractor 偵測。

  流程（逐 cube）：
    1. 壓 2D 偵測影像：det_white = 全譜 nanmean，無效像素填 0
    2. 跑 `sex det_white.fits -c default.sex`
    3. 收 seg.fits / test.cat / nosky.fits ＋ det 影像 → <out>/<cube檔名>/

  已有 seg.fits 的 cube 自動跳過（可中斷重跑）。
  用法：conda run -n astro python src/skymodel/SExtractor/run_sextractor.py data/nosky
  （可給資料夾或單一 .fits 檔、也可多個路徑；--out 改輸出根資料夾）
"""
import argparse, shutil, subprocess, time
from pathlib import Path
import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[3]

SEXDIR = Path(__file__).resolve().parent

T0 = time.time()
def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def collect_cubes(inputs):
    """把命令列給的路徑展開成 cube 清單：資料夾 → 其中所有 *.fits；檔案 → 直接收。"""
    cubes = []
    for p in map(Path, inputs):
        if p.is_dir():
            cubes += sorted(p.glob("*.fits"))   # sorted：glob 不保證順序（見上面的課）
        elif p.exists():
            cubes.append(p)
        else:
            log(f"找不到 {p}，略過")
    return cubes


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
    ap = argparse.ArgumentParser(description="對多顆 cube 跑教授的 SExtractor 工作流")
    ap.add_argument("inputs", nargs="+", help="cube 檔案或資料夾（資料夾＝其中所有 *.fits）")
    ap.add_argument("--out", type=Path, default=ROOT / "results/skymodel/sextractor",
                    help="輸出根資料夾（每顆 cube 一個子資料夾，以檔名命名）")

    args = ap.parse_args()


    for cube in collect_cubes(args.inputs):
        name = cube.stem                      # 檔名去掉 .fits，如 DATACUBE_FINAL_ESOSKY_3
        out = args.out / name

        if (out / "seg.fits").exists():
            log(f"sub {name}：已有 seg.fits，跳過")
            continue
        out.mkdir(parents=True, exist_ok=True)

        log(f"sub {name}：壓偵測影像 …")
        fits.writeto(out / "det_white.fits", detection_image(cube), overwrite=True)

        log(f"sub {name}：SExtractor …")
        r = subprocess.run(["sex", str((out / "det_white.fits").resolve()), "-c", "default.sex"],
                           cwd=SEXDIR, capture_output=True, text=True)
        if r.returncode != 0:
            log(f"sub {name}：sex 失敗：{r.stderr.strip()[:300]}")
        else:
            for f in ("seg.fits", "test.cat", "nosky.fits"):
                shutil.move(SEXDIR / f, out / f)
            seg = fits.getdata(out / "seg.fits")
            log(f"sub {name}：完成  objects={int(seg.max())}  覆蓋 {100*(seg>0).mean():.1f}%")
    log("全部完成")


if __name__ == "__main__":
    main()
