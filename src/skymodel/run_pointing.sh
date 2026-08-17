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
  $RUN src/skymodel/step1_whitelight.py "$NOSKY" --out "$W/step01" >/dev/null

  echo "--- [2/6] 教授的 segmentation"
  PROF_SEG=$ROOT/data/wsky_seg/DATACUBE_FINAL_${N}_seg.fits
  [ -f "$PROF_SEG" ] || { echo "★ 找不到 $PROF_SEG"; exit 1; }
  cp "$PROF_SEG" "$W/step01/seg.fits"
  # seg 和白光必須是同一個像素格點。對不上的話下游不會報錯,只會安靜地把遮罩
  # 套錯位置,所以在這裡擋掉。
  $RUN -c "
from astropy.io import fits; import numpy as np, sys
s = fits.getdata('$W/step01/seg.fits'); w = fits.getdata('$W/step01/whitelight.fits')
if s.shape != w.shape:
    sys.exit(f'★ seg {s.shape} 與白光 {w.shape} 尺寸不同')
print(f'    {len(np.unique(s))-1} 個源,遮罩 {100*(s>0).mean():.1f}%')"

  echo "--- [3/6] step2 源光譜（nosky,分類用）"
  $RUN src/skymodel/step2_object_spectra.py --work "$W" --cube "$NOSKY" \
       --out "$W/step02" >/dev/null

  echo "--- [4/6] step3 sky basis（學天光的範圍：${REGION[*]}）"
  $RUN src/skymodel/step3_sky_basis.py --methods svd -K 30 \
       --work "$W" --cube "$WSKY" "${REGION[@]}" 2>&1 \
       | grep -E "空間限制|exclude-box|blank spaxels|svd "

  echo "--- [5/6] step4 模板擬合與分類"
  $RUN src/skymodel/step4_fit_source.py --id all --basis svd -K 30 --s-fix 0.0 \
       --star-window 4600 8000 --gal-window 4600 8000 --line-mask-iter 1 \
       --spec-dir "$W/step02" --work "$W" --num-workers 16 2>&1 | tail -3

  echo "--- [6/6] step5 逐 spaxel 擬合（--s-field）"
  BEST=$W/step04/classification_nobasis_s0.0_4600-8000_4600-8000_L1cum.npz
  $RUN src/skymodel/step5_fit_spaxels.py --basis svd -K 30 --s-field \
       --work "$W" --cube "$WSKY" --sky-dir "$W/step03" --best "$BEST" 2>&1 \
       | grep -E "s 空間場|ridge|blank 以場|源區域|saved"

  echo "*** pointing #$N 完成，耗時 $(( $(date +%s) - T0 )) 秒"
  df -h "$ROOT" | tail -1
done
