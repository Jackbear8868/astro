"""畫出 estimate_continuum 每一輪的連續譜、門檻與線遮罩。

回答的問題:
  - 遮罩為什麼一路成長?(遮掉發射線 → 連續譜下降、sigma 縮小 → 門檻下移)
  - 遮罩是逐輪累加的嗎?(不是 —— 每輪都用原始 mean_sky 重新判定)
  - 迭代停在哪裡、為什麼停(收斂 or 撞到 min_unmasked_frac 地板)

需要 step3 存下的 continuum_iterations.npz。若還沒有,重跑一次該顆的 pipeline 即可:
    conda run -n astro python src/skymodel/pipeline.py configs/pNN.yaml

輸出 results/skymodel/evaluation/sky_basis/linemask_iters_{pNN}/ 底下,每一輪兩張:
    iter{N}_masked.png    被遮掉的通道換顏色,連同連續譜與上下門檻
    iter{N}_unmasked.png  沒被遮掉的通道 —— 連續譜就是在這些通道上擬的

--with-rejected 會多畫一輪。estimate_continuum 是先判斷停止條件才存檔,所以
觸發停止的那一輪不在 continuum_iterations.npz 裡,沒有它就看不到迭代是停在什麼
狀態上。這個
旗標從最後一輪的遮罩把它重算出來,存成 iter{N+1}_*_rejected.png,檔名和採用的
那幾輪分開,不會蓋掉任何東西。

兩張的檔名講的是這些通道被怎麼處理,不是它們「是不是線」:判準只知道某個通道
超出了門檻,叫它「線」是多一層解讀。各輪的通道數看上面印出來的表。

用法:
    python src/skymodel/experiments/plot_linemask_iters.py --work results/skymodel/p01
    python src/skymodel/experiments/plot_linemask_iters.py --work results/skymodel/p01 --ylim 0 120
    python src/skymodel/experiments/plot_linemask_iters.py --work results/skymodel/p01 --with-rejected
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import detect_lines  # noqa: E402

ROOT    = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "results/skymodel/evaluation/sky_basis"

THRESHOLDS = (1, 2)      # (正, 負);與 pipeline.py 的 sky_basis 一致
WINDOW     = 300         # running median 視窗;同上
MIN_UNMASKED_FRAC = 0.16 # utils.estimate_continuum 的停止地板;同上

# 模型(連續譜與它的門檻包絡)用暖色,資料與判定結果用冷色。這樣密集區裡唯一的
# 暖色粗線就是連續譜 —— 而連續譜正是整個迭代在估的東西,它不能被埋在被標記的
# 通道裡。兩條門檻用同色系深淺,因為它們是同一個包絡的上下緣,不是兩種東西。
C_SKY, C_CONT, C_UP, C_DN = "0.72", "#b30000", "#e6550d", "#fdae61"
C_HI,  C_LO = "#2ca02c", "#6a51a3"
C_KEEP = "#d62728"      # 沒被遮掉的通道 —— 連續譜就是在這些點上擬的

# masked 與 unmasked 是同一輪的兩種看法,要能在投影片上疊著切換,所以兩張的幾何
# 必須逐像素相同。做法是把座標框釘死在畫布的這個矩形上(左, 下, 寬, 高,單位是
# 畫布比例),並且存檔不裁切 —— bbox_inches="tight" 是依內容裁的,而兩張的圖例
# 內容本來就不一樣,裁出來的寬度就會差一點,疊起來就是整張圖跳動。
# 右邊留下的 0.116 是圖例的位置:圖例畫在座標框外,不佔資料的空間。它要放得下
# 兩張圖裡最寬的那一列("not masked"),放不下就會被畫布右緣切掉半個字。
AXES_RECT     = (0.042, 0.115, 0.842, 0.785)
LEGEND_ANCHOR = (1.005, 1.0)     # 錨在座標框右上角外側,兩張圖同一個點

# 這些圖是投影片上的整頁圖,離螢幕遠,字和線都要比論文裡的粗一號。
FS_TITLE, FS_LABEL, FS_TICK, FS_LEGEND = 26, 22, 18, 20
LW_SKY, LW_CONT, LW_THRESH = 1.2, 3.2, 1.8


def main():
    ap = argparse.ArgumentParser(description="畫 line_mask 每一輪的演變")
    ap.add_argument("--work", required=True,
                    help="pointing 的工作區,例如 results/skymodel/p01")
    ap.add_argument("--ylim", type=float, nargs=2, metavar=("LO", "HI"), default=(0, 100),
                    help="光譜圖的 y 軸範圍。天光線最高衝到數百,畫得下的話連續譜和"
                         "門檻會被壓成貼著 0 的一條 —— 上限是為了看連續譜而砍的,"
                         "被切掉的線頂端不是這張圖要看的東西")
    ap.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"),
                    default=(24, 7),
                    help="畫布尺寸(吋)。3801 個通道橫向排開,窄的畫布會把天光線擠成一團。"
                         "圖例的欄寬是畫布寬度的一個比例(見 AXES_RECT),所以窄到"
                         "二十吋以下時圖例的字會被右緣切掉")
    ap.add_argument("--dpi", type=int, default=220)
    ap.add_argument("--with-rejected", action="store_true",
                    help="多畫觸發停止的那一輪。它沒有被 estimate_continuum 存下來,"
                         "會從最後一輪的遮罩重算,檔名帶 _rejected")
    args = ap.parse_args()

    W = ROOT / args.work
    STEP03 = W / "step03"

    need = ["continuum_iterations.npz"]
    missing = [f for f in need if not (STEP03 / f).exists()]
    if missing:
        raise SystemExit(
            f"{STEP03} 缺少 {', '.join(missing)}。step3 必須用有存 history 的版本重跑一次:\n"
            "  conda run -n astro python src/skymodel/pipeline.py configs/pNN.yaml\n"
            "  (方法、K 與學天空的空間範圍都在 config 裡,不必另外帶)")

    wl = np.load(STEP03 / "wavelength.npy")
    ms = np.load(STEP03 / "blank_mean_spectrum.npy")
    it = np.load(STEP03 / "continuum_iterations.npz")
    C, S, M = it["continuum"], it["threshold"], it["line_mask"]   # 各 (n_iter, nz)
    n_saved = M.shape[0]
    n_iter  = n_saved

    if args.with_rejected:
        # 先確認本檔的 WINDOW / THRESHOLDS 能重現最後一輪。重現得出來,才有資格
        # 拿同一組參數去算一輪 step03 沒存的東西;對不上的話畫出來的是另一組
        # 參數下的結果,而圖上沒有任何地方看得出這件事。
        prev = M[-2] if n_saved >= 2 else None
        _, _, chk = detect_lines(ms, exclude=prev, thresholds=THRESHOLDS, window=WINDOW)
        if not np.array_equal(chk, M[-1]):
            raise SystemExit(
                f"★ 用 WINDOW={WINDOW} THRESHOLDS={THRESHOLDS} 重算第 {n_saved} 輪,"
                f"和 step03 存的差 {int((chk != M[-1]).sum())} 個通道 —— "
                f"這兩個值和 step3 當時用的對不上,第 {n_saved + 1} 輪畫出來不會是真的")
        c_r, s_r, m_r = detect_lines(ms, exclude=M[-1],
                                     thresholds=THRESHOLDS, window=WINDOW)
        C = np.vstack([C, c_r[None]])
        S = np.vstack([S, s_r[None]])
        M = np.vstack([M, m_r[None]])
        n_iter += 1

    # ---------------- 診斷表 ----------------
    print(f"{n_saved} 輪迭代"
          + (f" (+1 輪未採用)" if args.with_rejected else "")
          + f"   {wl.size} 通道 ({wl.min():.1f}-{wl.max():.1f} A air)\n")
    print(f"{'輪':>3}{'遮罩通道':>10}{'比例':>9}{'高於':>8}{'低於':>8}"
          f"{'新增':>8}{'移除':>8}{'continuum 中位數':>18}{'sigma 中位數':>14}")
    print("-" * 88)
    for i in range(n_iter):
        if i == 0:
            add = rem = "—"
        else:
            add = int((M[i] & ~M[i-1]).sum())
            rem = int((~M[i] & M[i-1]).sum())
        n_hi = int((ms > C[i] + THRESHOLDS[0] * S[i]).sum())
        n_lo = int((ms < C[i] - THRESHOLDS[1] * S[i]).sum())
        lab = f"{i+1}*" if i >= n_saved else f"{i+1}"
        print(f"{lab:>3}{M[i].sum():>10}{100*M[i].mean():>8.1f}%{n_hi:>8}{n_lo:>8}"
              f"{add:>8}{rem:>8}{np.median(C[i]):>18.3f}{np.median(S[i]):>14.4f}")
    if args.with_rejected:
        unmasked = 1.0 - M[-1].mean()
        print(f"\n* 第 {n_iter} 輪未採用:未遮罩 {100*unmasked:.1f}% "
              f"< 地板 {100*MIN_UNMASKED_FRAC:.0f}%,迭代在這裡停下,"
              f"最後採用的是第 {n_saved} 輪")

    # 聯集只看採用的那幾輪 —— 未採用的那輪不屬於迭代的結果。
    union = np.logical_or.reduce(M[:n_saved])
    last = M[n_saved - 1]
    print(f"\n各輪聯集 = 最後一輪嗎: {np.array_equal(union, last)}"
          f"   (聯集 {union.sum()} vs 最後一輪 {last.sum()})")
    print("→ 不相等代表遮罩不是逐輪累加,每輪都用原始 mean_sky 重新判定。")

    # ---------------- 每一輪一張圖 ----------------
    # 目錄名帶上工作區名稱:每顆 pointing 的 mean_sky 不同,遮罩也不同,
    # 寫進同一個目錄的話後跑的那顆會無聲蓋掉前一顆。
    out = FIGURES / f"linemask_iters_{W.name}"
    out.mkdir(parents=True, exist_ok=True)

    def save(fig, name):
        # 不裁切:每張都正好是 figsize x dpi 個像素,理由見 AXES_RECT。
        fig.savefig(out / name, dpi=args.dpi)
        plt.close(fig)

    def blank_panel(title):
        """開一張空圖:畫布、座標框、字級、軸標籤,兩張圖共用這一段。

        座標框用 add_axes 釘在 AXES_RECT,不交給自動版面:框的位置只由那一個常數
        決定,不會被 rcParams 的預設邊界、或日後某人加的 tight_layout 挪走,右邊
        也才有一塊寬度確定的地方放圖例。
        """
        fig = plt.figure(figsize=args.figsize)
        a = fig.add_axes(AXES_RECT)
        # x 範圍明寫:主圖用 add_collection 加資料,它不會帶動自動縮放。兩張圖
        # 共用這一行,波長也就一定對得上。
        a.set_xlim(wl.min(), wl.max())
        a.set_ylim(*args.ylim)
        a.set_xlabel("wavelength [$\\AA$]", fontsize=FS_LABEL)
        a.set_ylabel("flux", fontsize=FS_LABEL)
        a.tick_params(labelsize=FS_TICK)
        a.set_title(title, fontsize=FS_TITLE)
        return fig, a

    def legend(a, handles):
        """圖例:兩張圖同一個錨點、同一組間距,只有列數不同。"""
        a.legend(handles=handles, fontsize=FS_LEGEND, loc="upper left",
                 bbox_to_anchor=LEGEND_ANCHOR, borderaxespad=0, frameon=False,
                 handlelength=1.8, handletextpad=0.8, labelspacing=0.7)

    def panel(i, title, name):
        """一輪:mean_sky、連續譜、上下兩條門檻,被判成線的通道換顏色。

        detect_lines 的判準是雙向的
        (ms > C + t0*sigma) | (ms < C - t1*sigma),所以「被判成線」可以精確
        拆成上下兩側,低於 -t1*sigma 的那一半才不會被忽略。

        上色的是 mean_sky 曲線**本身**,不是從連續譜畫到資料值的垂直線。
        資料在一個通道只有一個值,「從連續譜長上去」是畫圖時加的幾何:每個
        勉強過關的通道都會畫出一根短棒,幾百根疊起來就在連續譜上方糊成一條
        實心色帶,看起來像資料真的長那樣,而且把連續譜自己蓋掉。
        """
        hi = ms > C[i] + THRESHOLDS[0] * S[i]
        lo = ms < C[i] - THRESHOLDS[1] * S[i]
        # 重建出來的遮罩必須和 step3 存下的那一份相同。不同就代表這裡的
        # THRESHOLDS 和 step3 實際用的值對不上,圖上畫的門檻是假的。
        if not np.array_equal(hi | lo, M[i]):
            n = int(((hi | lo) != M[i]).sum())
            print(f"  ! iter{i+1}: 用 THRESHOLDS={THRESHOLDS} 重建的遮罩和 step3 存的差 {n} 個通道"
                  f" —— step3 當時的 --line-thresholds 應該不是這組")

        # 每一段接相鄰兩個通道,顏色由兩端點決定:任一端被標記,整段就算標記。
        # 只認單一端點的話,一條跨好幾個通道的天光線會在峰腰留下灰色斷口。
        pts  = np.column_stack([wl, ms]).reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        cols = np.array([C_SKY] * len(segs), dtype=object)
        cols[hi[:-1] | hi[1:]] = C_HI
        cols[lo[:-1] | lo[1:]] = C_LO

        fig, a = blank_panel(title)
        a.add_collection(LineCollection(segs, colors=list(cols),
                                        linewidths=LW_SKY, zorder=2))
        a.plot(wl, C[i], lw=LW_CONT, color=C_CONT, zorder=4)
        a.plot(wl, C[i] + THRESHOLDS[0] * S[i], lw=LW_THRESH, color=C_UP, zorder=5)
        a.plot(wl, C[i] - THRESHOLDS[1] * S[i], lw=LW_THRESH, color=C_DN, zorder=5)
        # LineCollection 沒有可用的圖例代表,所以圖例的樣本自己造。
        # 上下兩色不列進圖例:綠色在 +sigma 線之上、紫色在 -sigma 線之下,
        # 圖上看得出來,再寫一次只是把圖例拉長。
        legend(a, [
            Line2D([], [], color=C_SKY,  lw=2.6, label="mean sky"),
            Line2D([], [], color=C_CONT, lw=3.4, label="continuum"),
            Line2D([], [], color=C_UP,   lw=2.2, label=f"+{THRESHOLDS[0]}$\\sigma$"),
            Line2D([], [], color=C_DN,   lw=2.2, label=f"-{THRESHOLDS[1]}$\\sigma$")])
        save(fig, name)

    def panel_unmasked(i, name, title):
        """只有 mean_sky 和「沒被遮掉的通道」—— 不畫連續譜與門檻。

        線遮罩那張要看的是「門檻怎麼把線切出來」,所以連續譜與 ±sigma 是主角。
        這張要看的是「剩下哪些通道」,多畫三條線只會蓋住那些點。直接把未遮罩的
        通道畫成紅點疊在 mean_sky 上,點的疏密就是可用資料的分布。
        """
        keep = ~M[i]
        fig, a = blank_panel(title)
        a.plot(wl, ms, lw=LW_SKY, color=C_SKY)
        a.plot(wl[keep], ms[keep], ".", ms=4.0, color=C_KEEP, lw=0)
        # 圖例的紅點自己造,不用 markerscale 去放大圖上的那顆。markerscale 是
        # 乘法,圖上的點一改,圖例的點就跟著改;這裡要的是一顆和字級相稱的點,
        # 和圖上那顆多大無關 —— 圖上的點小到單獨一顆在圖例裡幾乎看不見。
        legend(a, [
            Line2D([], [], color=C_SKY, lw=2.6, label="mean sky"),
            Line2D([], [], color=C_KEEP, marker=".", ms=20, lw=0,
                   label="not masked")])
        save(fig, name)

    for i in range(n_iter):
        # 未採用的那輪走同一段程式,標題也用同一個格式 —— 只有檔名帶 _rejected,
        # 讓它不會蓋掉採用的那幾輪。兩套畫法或兩套標題會讓兩張圖之間多出一個
        # 「是不是畫法不一樣」的解釋,而它們本來就該長得一樣。
        sfx = "_rejected" if i >= n_saved else ""
        title = f"iteration {i+1}"
        panel(i, title, f"iter{i+1}_masked{sfx}.png")
        panel_unmasked(i, f"iter{i+1}_unmasked{sfx}.png", title)

    print(f"\nsaved {2 * n_iter} figures -> {out}")


if __name__ == "__main__":
    main()
