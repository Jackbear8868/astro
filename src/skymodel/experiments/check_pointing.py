"""一個 pointing 的驗收 —— 天空扣乾淨了嗎、源保住了嗎。

    conda run -n astro python src/skymodel/experiments/check_pointing.py p04 p08 p12

兩個條件必須同時成立,缺一不可(CLAUDE.md 原則 1):

    ① 遠場   殘差 ≈ 0        天空真的被扣掉了
    ② 源旁   殘差 ≈ P(λ)     源的光真的被留下了

P(λ) = <環的原始資料> − <遠場的原始資料>,**完全沒有經過擬合**,是「那裡到底多了
什麼光」的模型無關答案。只看殘差是不夠的 —— 把源扣光也會讓殘差變小。

保留率要扣掉遠場殘差
--------------------
環上的殘差是兩樣東西相加:

    環的殘差  =  源沒被扣掉的光  +  天空沒扣乾淨的部分
    遠場殘差  =                    天空沒扣乾淨的部分    (那裡沒有源)

所以 `殘差 / P` 會把「天空少扣的量」算成「源保留得好」——**一個直接獎勵少扣天空
的指標,方向是反的**。扣掉遠場殘差之後,天空扣得更乾淨才不會被記成保留率退步。

**校正的前提與限制**:它假設「天空的殘留在環上和在遠場一樣多」,所以只擋得掉
**全場一致的偏移**。源旁邊特有的**局部**誤差擋不掉,而那個局部誤差和環上的損失
可以是同一個量級。校正後的數字比未校正可信,但它不是真值。

兩欄都印出來,不要只留校正後的 —— 兩者差多少本身就是「天空扣得乾不乾淨」的訊息。

主源用 utils.main_source_group(最亮像素所在的那一整團),不是單一 seg ID ——
SExtractor 的 deblender 會把 Haro 11 拆成數塊,而拆法逐次觀測不同。
"""
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import main_source_group  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
# 乾淨窗口:沒有強天光線、也沒有 Haro 11 的強發射線,量連續譜用
WINDOWS = ((5250, 5450), (5600, 5750), (6050, 6200))


def check(work, cube, run=None):
    W = ROOT / "results/skymodel" / work
    seg = fits.getdata(W / "step01/seg.fits").astype(int)
    white = fits.getdata(W / "step01/whitelight.fits")
    wl = np.load(W / "step03/wavelength.npy")
    ch = np.flatnonzero(np.logical_or.reduce([(wl >= a) & (wl < b) for a, b in WINDOWS]))

    main, mids, _ = main_source_group(seg, white, W / "step04b")
    edge = ndimage.distance_transform_edt(white != 0)
    d_all = ndimage.distance_transform_edt(seg == 0)
    d_main = ndimage.distance_transform_edt(~main)
    others = (seg > 0) & ~main
    d_oth = ndimage.distance_transform_edt(~others) if others.any() else d_all * 0 + 1e9
    base = (seg == 0) & (white != 0) & (edge > 15)

    zones = {
        "far":        base & (d_all > 30) & (d_main > 110),
        "small 1-3":  base & (d_oth > 1) & (d_oth <= 3) & (d_main > 30),
        "main 1-3":   base & (d_main > 1) & (d_main <= 3) & (d_oth > 6),
        "main 3-10":  base & (d_main > 3) & (d_main <= 10) & (d_oth > 6),
    }

    def band(p, hdu=0):
        with fits.open(p, memmap=True) as h:
            d = h[hdu].data if h[hdu].data is not None else h["DATA"].data
            return np.asarray(d[ch], np.float32)

    raw = band(ROOT / cube)
    subs = sorted((W / "step05").glob("*/sky_subtracted.fits"))
    if run:
        subs = [p for p in subs if run in p.parent.name]
    if not subs:
        print(f"{work}: 找不到 step05 的輸出" + (f"(--run {run})" if run else "")); return
    # 多個 run 並存時不能默默挑一個 —— 挑錯了表格看起來完全正常
    if len(subs) > 1:
        raise SystemExit(f"★ {work} 有 {len(subs)} 個 step05 的 run,用 --run 指定:\n  "
                         + "\n  ".join(p.parent.name for p in subs))
    sub = band(subs[0])
    print(f"  run = {subs[0].parent.name}")
    far = zones["far"]
    if far.sum() < 30:
        print(f"{work}: 遠場格數不足({int(far.sum())}),無法定基線"); return
    rf = np.nanmean(raw[:, far], axis=1)

    print(f"\n{work}  主源 {len(mids)} 個 seg ID,共 {int(main.sum()):,} px   "
          f"({subs[-1].parent.name})")
    print(f"{'區域':>12}{'spaxel':>8}{'P (真值)':>11}{'殘差':>10}"
          f"{'保留率':>9}{'校正後':>9}")
    print("-" * 61)
    far_res = float(np.nanmedian(np.nanmean(sub[:, far], axis=1)))
    for nm, m in zones.items():
        if m.sum() < 30:
            print(f"{nm:>12}{int(m.sum()):>8}   格數不足")
            continue
        P = float(np.nanmedian(np.nanmean(raw[:, m], axis=1) - rf))
        v = float(np.nanmedian(np.nanmean(sub[:, m], axis=1)))
        if nm == "far" or abs(P) <= 1e-6:
            print(f"{nm:>12}{int(m.sum()):>8}{P:>11.4f}{v:>10.4f}"
                  f"{'—':>9}{'—':>9}")
        else:
            print(f"{nm:>12}{int(m.sum()):>8}{P:>11.4f}{v:>10.4f}"
                  f"{v / P:>9.2f}{(v - far_res) / P:>9.2f}")


CUBES = {"p04": "data/wshy/DATACUBE_FINAL_4.fits",
         "p08": "data/wshy/DATACUBE_FINAL_8.fits",
         "p12": "data/wshy/DATACUBE_FINAL_12.fits",
         "wfm": "data/Haro11_wsky.fits"}

if __name__ == "__main__":
    argv = sys.argv[1:]
    run = None
    if "--run" in argv:
        i = argv.index("--run")
        run = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    for w in argv:
        check(w, CUBES.get(w, f"data/wshy/DATACUBE_FINAL_{w.lstrip('p0') or w}.fits"), run)
