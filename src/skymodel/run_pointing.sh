#!/usr/bin/env bash
# 把一個 pointing 的整條 pipeline 跑完。
#
#   ./run_pointing.sh 4          # 跑 DATACUBE_FINAL_4
#   ./run_pointing.sh 4 5 6      # 連續跑好幾個
#
# 每個 pointing 一個工作區 results/skymodel/pNN/,底下 step01/step02/... 結構相同。
#
# 為什麼偵測影像用 nosky 而不是 wsky:白光是沿波長平均,wsky 的天空連續譜會把
# 整張圖整片墊高,遠高於源本身的淨流量,SExtractor 的背景與雜訊估計因此完全不同,
# 延展的源會被切得比實際小。偵測要用扣過天空的。
#
# DETECT_THRESH 顯式給在指令列,蓋掉 default.sex 裡的值 —— 門檻直接決定 seg 的
# 範圍,不該藏在設定檔裡。
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
SEX=/local/feather/conda_envs/astro/bin/sex
RUN="conda run --no-capture-output -n astro python"

for N in "$@"; do
  P=$(printf "p%02d" "$N")
  W=$ROOT/results/skymodel/$P
  WSKY=$ROOT/data/wshy/DATACUBE_FINAL_${N}.fits
  NOSKY=$ROOT/data/nosky/DATACUBE_FINAL_ESOSKY_${N}.fits
  [ -f "$WSKY" ]  || { echo "★ 找不到 $WSKY";  exit 1; }
  [ -f "$NOSKY" ] || { echo "★ 找不到 $NOSKY"; exit 1; }

  # 學天光的空間範圍 —— 使用者目視判讀 pseudo-r 的等光度線之後親自定的
  # (圖:results/skymodel/figures/prof_seg/visual_pNN.png)。這是使用者的決定,
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

  echo "--- [2/6] SExtractor (DETECT_THRESH 2.0)"
  (cd src/skymodel/SExtractor && $SEX "$W/step01/whitelight.fits" -c default.sex \
      -DETECT_THRESH 2.0 -CATALOG_NAME "$W/step01/test.cat" \
      -CHECKIMAGE_TYPE SEGMENTATION -CHECKIMAGE_NAME "$W/step01/seg.fits") 2>&1 \
      | grep -E "sextracted" | tail -1

  echo "--- [3/6] step2 源光譜（nosky,分類用）"
  $RUN src/skymodel/step2_object_spectra.py --work "$W" --cube "$NOSKY" \
       --out "$W/step02_eso" >/dev/null

  echo "--- [4/6] step3 sky basis（學天光的範圍：${REGION[*]}）"
  $RUN src/skymodel/step3_sky_basis.py --methods svd -K 30 \
       --work "$W" --cube "$WSKY" "${REGION[@]}" 2>&1 \
       | grep -E "空間限制|exclude-box|blank spaxels|svd "

  echo "--- [5/6] step4 模板擬合與分類"
  $RUN src/skymodel/step4_fit_source.py --id all --basis svd -K 30 --s-fix 0.0 \
       --star-window 4700 8000 --gal-window 4700 8000 --line-mask-iter 1 \
       --spec-dir "$W/step02_eso" --work "$W" --num-workers 16 2>&1 | tail -3

  echo "--- [6/6] step5 逐 spaxel 擬合（--s-field）"
  BEST=$W/step04b/classification_nobasis_s0.0_4700-8000_4700-8000_L1cum__eso.npz
  $RUN src/skymodel/step5_fit_spaxels.py --basis svd -K 30 --s-field \
       --work "$W" --cube "$WSKY" --sky-dir "$W/step03" --best "$BEST" 2>&1 \
       | grep -E "s 空間場|ridge|blank 以場|源區域|saved"

  echo "*** pointing #$N 完成，耗時 $(( $(date +%s) - T0 )) 秒"
  df -h "$ROOT" | tail -1
done
