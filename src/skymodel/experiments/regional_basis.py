"""分區學 basis 能不能降低 blank 的 sigma?

動機:blank 殘差裡可能有「非全場共有」的空間結構(逐行/逐列的偏移、方向不
對稱的空間相關)。MUSE 的 slice 沿特定方向排列,切片級的系統性(例如 LSF 隨
視場變化)正好會產生這種形態。若真是如此,「全場共用一組 basis」本身就是問題。

驗證必須是樣本外的:分區一定會讓「用來學 basis 的那些 spaxel」殘差變小,
那只是自由度變多。所以每個 spaxel 隨機分成 train / test 兩半,basis 只用
train 學,分數只在 test 上算。基準(N=1)用完全相同的 train/test 切分。

分區方式三種都試:沿 x 的帶、沿 y 的帶、二維方塊。

    conda run -n astro python src/skymodel/experiments/regional_basis.py
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from sklearn.decomposition import TruncatedSVD

ROOT   = Path(__file__).resolve().parents[3]
STEP01 = ROOT / "results/skymodel/step01"
STEP03 = ROOT / "results/skymodel/step03"
WSKY   = ROOT / "data/Haro11_NEpointing_wsky.fits"

SEED = 0


def evaluate(Dtr, Dte, C_sky, K, lm):
    """用 train 學 basis,在 test 上算殘差。回傳 (線內 sigma, 線外 sigma, rms)。"""
    if Dtr.shape[1] < 4 * K or Dte.shape[1] == 0:
        return None
    B = TruncatedSVD(n_components=K, random_state=SEED).fit((Dtr - C_sky[:, None]).T)
    sky = np.vstack([C_sky, B.components_])
    c = np.linalg.pinv(sky.T) @ Dte              # 未加權,和 fit_blank 的快路徑相同
    Rte = Dte - sky.T @ c
    return (Rte[lm].std(axis=0), Rte[~lm].std(axis=0),
            np.sqrt((Rte ** 2).mean(axis=0)))


def main():
    ap = argparse.ArgumentParser(description="分區 basis vs 全場單一 basis")
    ap.add_argument("-K", type=int, default=30)
    ap.add_argument("--splits", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    lm    = np.load(STEP03 / "iter_line_mask.npy")[0]
    C_sky = np.load(STEP03 / "sky_continuum.npy")

    with fits.open(WSKY, memmap=True) as h:
        D = np.asarray(h["DATA"].data, np.float32)
    nz, ny, nx = D.shape
    D = D.reshape(nz, -1)
    ok = ((white != 0).ravel() & (seg.ravel() == 0) & np.isfinite(D).all(axis=0))
    idx = np.flatnonzero(ok)
    De = D[:, idx].astype(np.float32)
    del D
    yy, xx = np.divmod(idx, nx)

    # train / test 隨機分半。兩半的空間分布相同,所以分區與否都用同一組切分。
    rng = np.random.default_rng(SEED)
    tr = rng.random(idx.size) < 0.5
    print(f"blank {idx.size:,} 個   train {int(tr.sum()):,}   test {int((~tr).sum()):,}"
          f"   K={args.K}   線內 {int(lm.sum())}/{nz} 通道\n")

    def run(label, region_id, nreg):
        """region_id 給每個 spaxel 一個區號;nreg 是區數。"""
        si, so, rm = [], [], []
        for r in range(nreg):
            m = region_id == r
            out = evaluate(De[:, m & tr], De[:, m & ~tr], C_sky, args.K, lm)
            if out is None:
                print(f"{label:>16}   區 {r} 樣本不足,跳過")
                continue
            si.append(out[0]); so.append(out[1]); rm.append(out[2])
        si = np.concatenate(si); so = np.concatenate(so); rm = np.concatenate(rm)
        rel = "" if base is None else f"{100*(np.median(rm)/base-1):>9.2f}%"
        print(f"{label:>16}{nreg:>6}{np.median(si):>13.4f}{np.median(so):>13.4f}"
              f"{np.median(rm):>10.4f}{rel}", flush=True)
        return np.median(rm), np.median(si)

    print(f"{'分區方式':>16}{'區數':>6}{'線內 sigma':>13}{'線外 sigma':>13}"
          f"{'rms':>10}{'vs 基準':>10}")
    print("-" * 70)
    base = None
    base, base_si = run("baseline (1 basis)", np.zeros(idx.size, int), 1)

    for n in args.splits:
        if n == 1:
            continue
        rid = np.minimum((xx * n) // nx, n - 1)
        run(f"x-bands ({n})", rid, n)
    for n in args.splits:
        if n == 1:
            continue
        rid = np.minimum((yy * n) // ny, n - 1)
        run(f"y-bands ({n})", rid, n)
    for n in args.splits:
        if n < 4:
            continue
        k = int(np.sqrt(n))
        rid = (np.minimum((yy * k) // ny, k - 1) * k
               + np.minimum((xx * k) // nx, k - 1))
        run(f"tiles {k}x{k}", rid, k * k)

    print(f"\n基準的線內 sigma = {base_si:.4f},rms = {base:.4f}")
    print("負的 vs 基準 = 分區比較好。改善若小於約 0.5%,落在 train/test 切分的"
          "隨機起伏內,不能當成實質改善。")


if __name__ == "__main__":
    main()
