# 海報／投影片用的排版版本

這裡的程式**不問任何新的問題**。每一支都對應隔壁 `evaluation/` 的一支,讀同樣的資料、
畫同樣的曲線,只是換一套排版:線加粗、字放大、刻度精簡、長寬比指定、legend 另存成一
張透明圖。印在海報上要看得清楚,和在螢幕上診斷要看得仔細,是兩種需求。

科學內容有任何改變都不屬於這裡 —— 那要改的是 `evaluation/` 的那一支。

| 這裡 | 對應 | 畫的是 |
|---|---|---|
| `halo_poster.py` | (無,海報專用) | 源外環的 raw(wsky)對 signal(我們扣完天光的結果) |
| `eso_poster.py` | `zone_spectra.py --zones outside --cubes ours eso` | 源外環的 ours 對 ESO pipeline |
| `continuum_poster.py` | `pointing_curves.py --curve continuum` | 14 顆 pointing 的天光連續譜 |
| `eigen_poster.py` | `plot_eigen.py --mode panels` | galaxy eigenspectra 與恆星模板 |
| `basis_poster.py` | `sky_basis.py` 的 `top{N}.png` | step3 學到的前幾條天光線基底 |

---

## 共通約定

- **`--figsize W H` 是整張圖的尺寸**,不是每個面板的。`plot_eigen.py` 的同名參數是每個
  面板的高度基準(會再乘面板數),在這裡不是 —— 你給的比例就是印出來的比例。
- **`--suffix` 另存**,試不同尺寸不會蓋掉已經放進海報的那張。
- **dpi 一律 300**,PNG 檔頭也寫 300,匯進排版軟體會直接按實體尺寸放置。
- **legend 另存成一張透明背景的圖**,可以擺在海報上任何位置,兩張共用一組圖示時不必
  各放一份。
- **字級是絕對點數**,不隨畫布縮放。同一組數值在 20 吋寬的圖上和 8 吋寬的圖上看起來
  差很多,所以小圖用的 `--fs` 本來就該比大圖小。

## 快取

`halo_poster.py` 和 `eso_poster.py` 要讀兩顆 ~3 GB 的 cube 才能算出環平均。結果快取在

```
results/skymodel/evaluation/poster_cache/{halo,eso}_pNN.npz
```

一顆 pointing 一個檔 —— 環平均綁在那個 field 上,不能互用。只是要調字級或比例時,第二次
之後是幾秒鐘的事。想重新讀 cube 就把對應的 `.npz` 刪掉。

## 檔名會撞

除了 `halo_poster.py` 之外,每一支寫出的檔名都和它對應的 `evaluation/` 程式**完全相同**:

```
eso_poster.py        p NN/halo/outside_vs_eso_outside_0_10_px.png   ← zone_spectrae.py --separate
continuum_poster.py  sky_basis/continuum_compare.png                ← pointing_curvesinuum_compare.py
eigen_poster.py      templates/eigen_{kind}_panels_muse*.png        ← plot_eigen.py --mode panels
basis_poster.py      pNN/basis/top5.png                             ← sky_basis.py
```

所以**重跑 `evaluation/` 那一支會把海報版蓋掉**,反之亦然。海報定稿之後要保險,就用
`--suffix` 另存一份,或把要用的圖複製到海報專案裡。

## 指令

```bash
P=src/skymodel/evaluation/poster
conda run -n astro python $P/halo_poster.py --pointing p01 --figsize 20 10
conda run -n astro python $P/halo_poster.py --pointing p01 --figsize 20 10 --zoom
conda run -n astro python $P/eso_poster.py --pointing p01
conda run -n astro python $P/continuum_poster.py
conda run -n astro python $P/eigen_poster.py --kind galaxy --figsize 8 3 --only 1 2
conda run -n astro python $P/eigen_poster.py --kind star --figsize 9 5 \
    --only-full --class-labels --only G K M
conda run -n astro python $P/basis_poster.py --work results/skymodel/p01 --figsize 10 5
```

`--only` 可以用整個標籤(`component 1`)、恆星檔名的開頭字母(`G`)、或標籤的最後一個字
(`1`)來指定要畫哪幾條;顏色索引在篩選**之前**就固定,所以 G 在只畫三條時仍然是紫色,
和畫七條時一致。
