#!/usr/bin/env bash
# 把一個 pointing 的整條 pipeline 跑完。
#
#   ./run_pointing.sh 4          # 跑 DATACUBE_FINAL_4
#   ./run_pointing.sh 4 5 6      # 連續跑好幾個
#
# 每個 pointing 一個工作區 results/skymodel/pNN/,底下 step01/step02/... 結構相同。
#
# seg 用教授交付的那一份,不自己跑 SExtractor。遮罩決定哪些 spaxel 拿去學天空,
# 是科學結果的一部分,不是實作細節 —— 由教授的產品定義。
#
# 白光仍然從 nosky 算:它不再用於偵測,但下游要靠它找主源(最亮像素所在的那一團),
# wsky 的天空連續譜會把整張圖墊高,最亮像素的位置就不可靠了。
# 每一步的完整輸出寫進 $W/stepN.log。step3 的空間限制統計、step4 的逐源分類表
# (含 margin 欄 —— 那是分類穩不穩的唯一指標)只印在終端機的話,跑完就永久消失。
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
RUN="conda run --no-capture-output -n astro python"

for N in "$@"; do
  P=$(printf "p%02d" "$N")
  W=$ROOT/results/skymodel/$P
  WSKY=$ROOT/data/wshy/DATACUBE_FINAL_${N}.fits
  NOSKY=$ROOT/data/nosky/DATACUBE_FINAL_ESOSKY_${N}.fits
  [ -f "$WSKY" ]  || { echo "★ 找不到 $WSKY";  exit 1; }
  [ -f "$NOSKY" ] || { echo "★ 找不到 $NOSKY"; exit 1; }

  # 學天光的空間範圍 —— 使用者目視判讀 pseudo-r 的等光度線之後親自定的
  # (圖:results/skymodel/evaluation/masking/prof_seg/visual_pNN.png)。這是使用者的決定,
  # 不是推導出來的值。
  #
  # 上界一律寫 9999 而不是各顆真正的 NAXIS —— --xlim/--ylim 是和像素座標比大小,
  # 給超過視場的值不會有事,但**寫小了會安靜地少學一塊**。14 個手抄的尺寸只會多
  # 14 個打錯的機會,而它換不到任何東西。
  #
  # #14 的 Haro 11 在視場正中央,要的是「挖中間、留外圈」。--xlim/--ylim 是 AND
  # 起來的保留範圍,做不到;改用 --exclude-box 排除中間那塊。
  case $N in
     1) REGION=(--xlim 0 165) ;;
     2) REGION=(--xlim 0 160) ;;
     3) REGION=(--ylim 170 9999) ;;
     4) REGION=(--ylim 170 9999) ;;
     5) REGION=(--ylim 170 9999) ;;
     6) REGION=(--ylim 170 9999) ;;
     7) REGION=(--ylim 175 9999) ;;
     8) REGION=(--ylim 0 125) ;;
     9) REGION=(--xlim 0 100) ;;
    10) REGION=(--ylim 0 125) ;;
    11) REGION=(--ylim 0 125) ;;
    12) REGION=(--ylim 0 125) ;;
    13) REGION=(--ylim 0 125) ;;
    14) REGION=(--exclude-box 75 250 60 270) ;;
     *) echo "★ #$N 沒有指定學天光的範圍"; exit 1 ;;
  esac

  echo "================================================================"
  echo "  pointing #$N  ->  $W   學天光範圍 ${REGION[*]}"
  echo "================================================================"
  T0=$(date +%s)

  echo "--- [1/6] step1 白光（從 nosky）"
  $RUN src/skymodel/step1_whitelight.py "$NOSKY" --out "$W/step01" \
       > "$W/step1.log" 2>&1

  echo "--- [2/6] 教授的 segmentation"
  PROF_SEG=$ROOT/data/wsky_seg/DATACUBE_FINAL_${N}_seg.fits
  [ -f "$PROF_SEG" ] || { echo "★ 找不到 $PROF_SEG"; exit 1; }
  cp "$PROF_SEG" "$W/step01/seg.fits"
  # 只比尺寸。**形狀相同不等於同一個像素格點**,但目前做不到更好 —— step1 寫
  # whitelight.fits 時沒有帶 header,那張圖沒有 WCS 可比(教授的 seg 和 cube 都有)。
  # 要真正擋住錯位,得先讓 step1 保留 WCS。
  $RUN -c "
from astropy.io import fits; import numpy as np, sys
s = fits.getdata('$W/step01/seg.fits'); w = fits.getdata('$W/step01/whitelight.fits')
if s.shape != w.shape:
    sys.exit(f'★ seg {s.shape} 與白光 {w.shape} 尺寸不同')
print(f'    {len(np.unique(s))-1} 個源,遮罩 {100*(s>0).mean():.1f}%')"

  echo "--- [3/6] step2 源光譜（nosky,分類用）"
  $RUN src/skymodel/step2_object_spectra.py --work "$W" --cube "$NOSKY" \
       --out "$W/step02" > "$W/step2.log" 2>&1

  echo "--- [4/6] step3 sky basis（學天光的範圍：${REGION[*]}）"
  $RUN src/skymodel/step3_sky_basis.py --methods svd -K 30 \
       --work "$W" --cube "$WSKY" "${REGION[@]}" 2>&1 \
       | tee "$W/step3.log" | grep -E "空間限制|exclude-box|blank spaxels|svd "

  echo "--- [5/6] step4 模板擬合與分類"
  # step4 的可變參數集中在這裡,底下的 BEST 用同一組變數組出檔名 —— 改了參數
  # 卻忘了改 BEST 的話,step5 會安安靜靜地讀上一次的分類檔跑完。
  SFIX=0.0
  WIN=(4600 8000)
  LITER=1
  $RUN src/skymodel/step4_fit_source.py --id all --basis svd -K 30 --s-fix "$SFIX" \
       --star-window "${WIN[@]}" --gal-window "${WIN[@]}" --line-mask-iter "$LITER" \
       --spec-dir "$W/step02" --work "$W" 2>&1 | tee "$W/step4.log" | tail -3

  echo "--- [6/6] step5 逐 spaxel 擬合（--s-field）"
  # 檔名的組法是 step4_fit_source.py 的 make_tag();這裡是它的第二份實作,
  # 所以組成的每一段都必須來自上面那組變數,不能手抄。
  BEST=$W/step04/classification_nobasis_s${SFIX}_${WIN[0]}-${WIN[1]}_${WIN[0]}-${WIN[1]}_L${LITER}cum.npz
  $RUN src/skymodel/step5_fit_spaxels.py --basis svd -K 30 --s-field \
       --work "$W" --cube "$WSKY" --best "$BEST" 2>&1 \
       | tee "$W/step5.log" | grep -E "s 空間場|ridge|blank 以場|源區域|saved"

  echo "*** pointing #$N 完成，耗時 $(( $(date +%s) - T0 )) 秒"
  df -h "$ROOT" | tail -1
done
