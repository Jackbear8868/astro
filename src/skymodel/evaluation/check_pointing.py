"""一個 pointing 的驗收 —— 天空扣乾淨了嗎、源保住了嗎。

    conda run -n astro python src/skymodel/evaluation/check_pointing.py p04 p08 p12

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, arcsinh_stretch, load_field, pointing_dir, zones  # noqa: E402
from utils import main_source_group  # noqa: E402

# 乾淨窗口:沒有強天光線、也沒有 Haro 11 的強發射線,量連續譜用
WINDOWS = ((5250, 5450), (5600, 5750), (6050, 6200))

# 環的顏色,圖上兩邊共用 —— 左邊的地圖和右邊的長條要靠顏色對起來
ZCOL = {"far": "#7f7f7f", "small 1-3": "#2ca02c",
        "main 1-3": "#1f77b4", "main 3-10": "#ff7f0e"}


def draw(work, white, Z, out):
    """四個環在視場的哪裡 —— 數字在表格裡,這張只回答「那是天上的哪一塊」。

    和 box/map.png 同一個角色:光譜圖只有名稱和座標,看不出那塊區域長什麼樣、
    離星系多遠。
    """
    fig, ax = plt.subplots(figsize=(9, 9))
    img, vmax = arcsinh_stretch(white, white != 0)
    ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=vmax)
    for nm, m in Z.items():
        rgba = np.zeros(m.shape + (4,))
        rgba[m] = list(matplotlib.colors.to_rgb(ZCOL[nm])) + [0.75]
        ax.imshow(rgba, origin="lower")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=[plt.Line2D([], [], color=ZCOL[nm], lw=6,
                                  label=f"{nm}   {int(Z[nm].sum()):,} spaxels")
                       for nm in Z], fontsize=10, loc="lower left",
              framealpha=0.85)
    ax.set_title(work, fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def check(work, cube, run=None):
    W = ROOT / "results/skymodel" / work
    seg, white, _ = load_field(W)
    wl = np.load(W / "step03/wavelength.npy")
    ch = np.flatnonzero(np.logical_or.reduce([(wl >= a) & (wl < b) for a, b in WINDOWS]))

    main, mids, _ = main_source_group(seg, white, W / "step04")
    Z = zones(seg, white, main)

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
    far = Z["far"]
    if far.sum() < 30:
        print(f"{work}: 遠場格數不足({int(far.sum())}),無法定基線"); return
    rf = np.nanmean(raw[:, far], axis=1)

    print(f"\n{work}  主源 {len(mids)} 個 seg ID,共 {int(main.sum()):,} px   "
          f"({subs[-1].parent.name})")
    print(f"{'區域':>12}{'spaxel':>8}{'P (真值)':>11}{'殘差':>10}"
          f"{'保留率':>9}{'校正後':>9}")
    print("-" * 61)
    far_res = float(np.nanmedian(np.nanmean(sub[:, far], axis=1)))
    drawn = {}
    for nm, m in Z.items():
        if m.sum() < 30:
            print(f"{nm:>12}{int(m.sum()):>8}   格數不足")
            continue
        P = float(np.nanmedian(np.nanmean(raw[:, m], axis=1) - rf))
        v = float(np.nanmedian(np.nanmean(sub[:, m], axis=1)))
        drawn[nm] = m
        if nm == "far" or abs(P) <= 1e-6:
            print(f"{nm:>12}{int(m.sum()):>8}{P:>11.4f}{v:>10.4f}"
                  f"{'—':>9}{'—':>9}")
        else:
            print(f"{nm:>12}{int(m.sum()):>8}{P:>11.4f}{v:>10.4f}"
                  f"{v / P:>9.2f}{(v - far_res) / P:>9.2f}")

    # 環的地圖和環的光譜放同一個目錄 —— 看 zone/main_3_10.png 的時候,要知道
    # 那一圈在哪,不必再回去找另一支腳本的輸出。
    draw(work, white, drawn, pointing_dir(work, "zone") / "map.png")


CUBES = {"p04": "data/wshy/DATACUBE_FINAL_4.fits",
         "p08": "data/wshy/DATACUBE_FINAL_8.fits",
         "p12": "data/wshy/DATACUBE_FINAL_12.fits"}

if __name__ == "__main__":
    argv = sys.argv[1:]
    run = None
    if "--run" in argv:
        i = argv.index("--run")
        run = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    for w in argv:
        check(w, CUBES.get(w, f"data/wshy/DATACUBE_FINAL_{w.lstrip('p0') or w}.fits"), run)
