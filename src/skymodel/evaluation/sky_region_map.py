"""學天空的範圍 —— 兩個階段各自用了哪些 spaxel。

pipeline 裡有**兩個**不同的「學天空範圍」,遮罩不一樣,不能混為一談:

    step3  天空 basis     blank = 視場內 & seg == 0,再套逐顆目視選定的
                          --xlim / --ylim / --exclude-box。**不做膨脹。**
    step5  s 空間場       在上面的基礎上再要求離**任何**源 > r_far、離主源
                          > r_far_haro,並剔除 |s − 中位| > clip x 散布 的格。
                          這一層才是「避開不穩定的區域」。

為什麼兩層的判準不同
--------------------
basis 學的是天光線的**形狀**,一條天光線在每個 blank spaxel 都長一樣,源的
PSF 翼混進來影響有限,所以只需要避開主源的暈(那是 --xlim/--ylim 在做的)。

s 學的是天空連續譜的**振幅**,而星系的連續譜和天空連續譜形狀太像 —— 訓練點
只要沾到一點源光,s 就被墊高,天空模型跟著長高,扣掉時把源吃掉。所以它要退得
更遠,而且要把擬合失敗的格剔掉。

範圍參數的唯一來源是 run_pointing.sh 的 case 區塊,這支直接去讀它,不另外抄一份。

    conda run -n astro python src/skymodel/evaluation/sky_region_map.py --work results/skymodel/p01
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, arcsinh_stretch, load_field, pointing_dir  # noqa: E402
from utils import build_s_field, main_source_group  # noqa: E402


def region_args(n):
    """從 run_pointing.sh 的 case 區塊讀出第 n 顆的空間限制。

    不在這裡抄一份 —— 兩處各存一份的話,改了一邊沒改另一邊,圖就會和實際跑的
    設定不符,而圖上看起來完全正常。
    """
    txt = (ROOT / "src/skymodel/run_pointing.sh").read_text()
    m = re.search(rf"^\s*{n}\)\s*REGION=\((.*?)\)\s*;;", txt, re.M)
    if not m:
        raise SystemExit(f"★ run_pointing.sh 的 case 區塊裡沒有 #{n}")
    tok = m.group(1).split()
    out = {}
    for k in ("--xlim", "--ylim", "--exclude-box"):
        if k in tok:
            i = tok.index(k)
            n_val = 4 if k == "--exclude-box" else 2
            out[k.lstrip("-").replace("-", "_")] = [int(v) for v in tok[i + 1:i + 1 + n_val]]
    return out


def main():
    ap = argparse.ArgumentParser(description="學天空的範圍:basis 與 s 場各自用了哪些格")
    ap.add_argument("--work", required=True)
    ap.add_argument("--run", default=None, help="預設取唯一的 *_sfield")
    args = ap.parse_args()

    W = ROOT / args.work
    n = int(W.name[1:])
    run = (W / "step05" / args.run if args.run
           else sorted((W / "step05").glob("*_sfield"))[0])

    seg, white, valid = load_field(W)
    reg = region_args(n)

    # --- step3:blank,再套空間限制 ---
    blank = valid & (seg == 0)
    keep = np.ones_like(blank)
    yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
    if "xlim" in reg:
        keep &= (xx >= reg["xlim"][0]) & (xx < reg["xlim"][1])
    if "ylim" in reg:
        keep &= (yy >= reg["ylim"][0]) & (yy < reg["ylim"][1])
    if "exclude_box" in reg:
        y0, y1, x0, x1 = reg["exclude_box"]
        keep &= ~((yy >= y0) & (yy <= y1) & (xx >= x0) & (xx <= x1))
    basis_train = blank & keep

    # --- step5:再退開源,並剔除擬合失敗的格 ---
    p = json.loads((run / "meta.json").read_text())["s_field_params"]
    s = np.load(run / "s_free.npy").astype(float)
    main, ids, _ = main_source_group(seg, np.where(valid, white, np.nan), W / "step04")
    ok = valid & (seg == 0) & np.isfinite(s)
    _, s_train = build_s_field(s, seg, ok, p["r_far"], p["r_far_haro"], p["clip"],
                               main=main)

    img, vmax = arcsinh_stretch(white, valid)
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 7.4))
    for a, m, ttl in (
        (ax[0], basis_train,
         f"step3  sky basis     {int(basis_train.sum()):,} spaxels"),
        (ax[1], s_train,
         f"step5  s field       {int(s_train.sum()):,} spaxels")):
        a.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=vmax)
        rgba = np.zeros(m.shape + (4,))
        rgba[m] = [0.13, 0.83, 0.08, 0.45]
        a.imshow(rgba, origin="lower")
        a.contour(seg > 0, levels=[0.5], colors="#39ff14", linewidths=0.4, alpha=.6)
        a.set_title(ttl, fontsize=12)
        a.set_xticks([]); a.set_yticks([])

    # 空間限制畫成框,才看得出「這條線是人訂的」而不是資料的邊界
    ny, nx = seg.shape
    if "xlim" in reg or "ylim" in reg:
        x0, x1 = reg.get("xlim", [0, nx])
        y0, y1 = reg.get("ylim", [0, ny])
        ax[0].add_patch(mpatches.Rectangle((x0 - .5, y0 - .5), min(x1, nx) - x0,
                                           min(y1, ny) - y0, fill=False,
                                           ec="#ff7f0e", lw=2.0, ls="--"))
    if "exclude_box" in reg:
        y0, y1, x0, x1 = reg["exclude_box"]
        ax[0].add_patch(mpatches.Rectangle((x0 - .5, y0 - .5), x1 - x0 + 1,
                                           y1 - y0 + 1, fill=False,
                                           ec="#e8272c", lw=2.0, ls="--"))

    lim = "  ".join(f"{k} {v}" for k, v in reg.items()) or "(無空間限制)"
    fig.suptitle(f"{W.name}    step3: {lim}    "
                 f"step5: > {p['r_far']:.0f} px from any source, "
                 f"> {p['r_far_haro']:.0f} px from Haro 11, "
                 f"clip {p['clip']:.0f} sigma", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    o = pointing_dir(W.name) / "sky_region.png"
    fig.savefig(o, dpi=140, bbox_inches="tight")
    print(f"{W.name}  blank {int(blank.sum()):,} -> basis {int(basis_train.sum()):,}"
          f" ({100*basis_train.sum()/blank.sum():.1f}%)"
          f" -> s field {int(s_train.sum()):,}"
          f" ({100*s_train.sum()/blank.sum():.1f}%)   -> {o}")


if __name__ == "__main__":
    main()
