"""把 step5 的實際輸出和 ESO nosky 在 blank 區逐 spaxel 比較。

先前的 K 掃描是模擬(一半 spaxel 學 basis、另一半評),這支比的是真正跑完
整條 pipeline 的 cube —— 用的是全部 blank spaxel 學出來的 basis,和實際
會拿去做科學的那份產物完全相同。

四個指標,互相獨立地回答不同問題:
    rms      總結:這個 cube 到底好不好(rms² = 偏移² + sigma²)
    |偏移|   零點:s 固定的處理帶來的優勢
    線內 sigma  天光線扣得乾不乾淨 —— K 的效果集中在這裡
    線外 sigma  連續譜區的抖動 —— 兩邊都貼著光子雜訊,預期平手

勝率是「該 spaxel 我們比 ESO 小」的比例,不是中位數的比較 —— 中位數贏
可能只是少數 spaxel 拉開的,勝率才說得清楚是不是普遍現象。

    conda run -n astro python src/skymodel/experiments/compare_eso_tags.py \
        --tags svd_K25_s_1.0 svd_K30_s_1.0
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT   = Path(__file__).resolve().parents[3]
STEP01 = ROOT / "results/skymodel/step01"
STEP03 = ROOT / "results/skymodel/step03"
STEP05 = ROOT / "results/skymodel/step05"
WSKY   = ROOT / "data/Haro11_NEpointing_wsky.fits"
ESO    = ROOT / "data/Haro11_NEpointing_esonosky.fits"

MIN_COVERAGE = 0.9


def metrics(R, IN, OUT):
    """R 形狀 (nz, n_spaxel);回傳四個逐 spaxel 的指標。"""
    return {"rms":  np.sqrt(np.mean(R ** 2, axis=0)),
            "off":  np.abs(R.mean(axis=0)),
            "sin":  R[IN].std(axis=0),
            "sout": R[OUT].std(axis=0)}


def main():
    ap = argparse.ArgumentParser(description="step5 產物 vs ESO nosky")
    ap.add_argument("--tags", nargs="+", default=["svd_K25_s_1.0", "svd_K30_s_1.0"])
    args = ap.parse_args()

    seg   = fits.getdata(STEP01 / "seg.fits")
    white = fits.getdata(STEP01 / "whitelight.fits")
    lm1   = np.load(STEP03 / "iter_line_mask.npy")[0]

    with fits.open(WSKY, memmap=True) as h:
        V = np.asarray(h["STAT"].data, np.float32)
    nz = V.shape[0]
    V  = V.reshape(nz, -1)

    cov   = np.isfinite(V).sum(axis=0) / nz
    blank = (white != 0).ravel() & (seg.ravel() == 0) & (cov >= MIN_COVERAGE)

    E = fits.getdata(ESO).reshape(nz, -1)
    ok = blank & np.isfinite(E).all(0) & np.isfinite(V).all(0)
    subs = {}
    for t in args.tags:
        S = fits.getdata(STEP05 / f"sky_subtracted_{t}.fits").reshape(nz, -1)
        ok &= np.isfinite(S).all(0)
        subs[t] = S
    idx = np.flatnonzero(ok)

    IN, OUT = lm1, ~lm1
    ref = metrics(E[:, idx].astype(np.float64), IN, OUT)
    Ve  = V[:, idx].astype(np.float64)
    # STAT 給的是每通道變異數 → 逐 spaxel 的光子雜訊下限
    floor_in  = np.sqrt(np.mean(Ve[IN],  axis=0))
    floor_out = np.sqrt(np.mean(Ve[OUT], axis=0))
    del V, Ve

    print(f"blank 可用 {idx.size:,} 個 spaxel   線內 {IN.sum()}/{nz} 通道\n")
    print(f"{'':>16}{'rms':>10}{'|偏移|':>10}{'線內 sigma':>12}{'線外 sigma':>12}")
    print("-" * 60)
    print(f"{'ESO nosky':>16}{np.median(ref['rms']):>10.4f}{np.median(ref['off']):>10.4f}"
          f"{np.median(ref['sin']):>12.4f}{np.median(ref['sout']):>12.4f}")
    print(f"{'光子雜訊下限':>16}{'—':>10}{'—':>10}"
          f"{np.median(floor_in):>12.4f}{np.median(floor_out):>12.4f}")

    for t in args.tags:
        m = metrics(subs[t][:, idx].astype(np.float64), IN, OUT)
        print(f"{t:>16}{np.median(m['rms']):>10.4f}{np.median(m['off']):>10.4f}"
              f"{np.median(m['sin']):>12.4f}{np.median(m['sout']):>12.4f}")
        subs[t] = m

    print(f"\n{'勝率(比 ESO 小的比例)':>16}")
    print("-" * 60)
    for t in args.tags:
        m = subs[t]
        print(f"{t:>16}{100*np.mean(m['rms'] < ref['rms']):>9.1f}%"
              f"{100*np.mean(m['off'] < ref['off']):>9.1f}%"
              f"{100*np.mean(m['sin'] < ref['sin']):>11.1f}%"
              f"{100*np.mean(m['sout'] < ref['sout']):>11.1f}%")

    # 扣掉光子雜訊後剩下的才是模型誤差:sigma² = 雜訊² + 模型²
    # (不做自由度修正 —— K 大時 basis 集中在高雜訊的強線通道,修正會失效)
    print(f"\n{'模型誤差 sqrt(sigma²-雜訊²)':>16}{'線內':>22}{'線外':>12}")
    print("-" * 60)
    def model_err(s, f):
        return np.sqrt(np.maximum(np.median(s) ** 2 - np.median(f) ** 2, 0))
    e_in, e_out = model_err(ref['sin'], floor_in), model_err(ref['sout'], floor_out)
    print(f"{'ESO nosky':>16}{e_in:>22.4f}{e_out:>12.4f}")
    for t in args.tags:
        m = subs[t]
        a, b = model_err(m['sin'], floor_in), model_err(m['sout'], floor_out)
        print(f"{t:>16}{a:>22.4f}{b:>12.4f}"
              f"   ({a/e_in:.2f}x, {b/e_out:.2f}x ESO)")


if __name__ == "__main__":
    main()
