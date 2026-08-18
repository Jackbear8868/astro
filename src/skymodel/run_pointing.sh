#!/usr/bin/env bash
# Run the full pipeline for one pointing.
#
#   ./run_pointing.sh 4          # run DATACUBE_FINAL_4
#   ./run_pointing.sh 4 5 6      # run several in sequence
#
# Each pointing gets its own working directory results/skymodel/pNN/, with an
# identical step01/step02/... structure underneath.
#
# The segmentation is the one delivered by the professor, not self-run
# SExtractor. The mask determines which spaxels are used to learn the sky --
# it is part of the science result, not an implementation detail, and is
# defined by the professor's product.
#
# White light is still computed from the nosky cube: it is no longer used for
# detection, but downstream needs it to locate the main source (the blob
# containing the brightest pixel); the sky continuum in the wsky cube raises
# the entire image, making the brightest-pixel location unreliable.
# Full output from each step is written to $W/stepN.log. The spatial-range
# statistics from step3, the per-source classification table from step4
# (including the margin column -- the only indicator of classification
# stability) would be permanently lost if only printed to the terminal.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)
RUN="conda run --no-capture-output -n astro python"

for N in "$@"; do
  P=$(printf "p%02d" "$N")
  W=$ROOT/results/skymodel/$P
  WSKY=$ROOT/data/wshy/DATACUBE_FINAL_${N}.fits
  NOSKY=$ROOT/data/nosky/DATACUBE_FINAL_ESOSKY_${N}.fits
  [ -f "$WSKY" ]  || { echo "★ File not found: $WSKY";  exit 1; }
  [ -f "$NOSKY" ] || { echo "★ File not found: $NOSKY"; exit 1; }

  # Spatial range for sky learning -- set by the user after visually inspecting
  # isophotes in pseudo-r images
  # (figure: results/skymodel/evaluation/masking/prof_seg/visual_pNN.png).
  # This is the user's decision, not a derived value.
  #
  # The upper bound is always 9999 rather than the actual NAXIS -- --xlim/--ylim
  # compare against pixel coordinates, so exceeding the field of view is harmless,
  # but **writing it too small silently omits a region**. Typing 14 actual
  # dimensions only adds 14 chances for a typo with zero benefit.
  #
  # For #14 Haro 11 sits in the centre of the field; the goal is "cut out the
  # middle, keep the outer ring". --xlim/--ylim are ANDed as a retention range
  # and cannot express this; --exclude-box is used instead to exclude the centre.
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
     *) echo "★ #$N has no sky-learning region defined"; exit 1 ;;
  esac

  echo "================================================================"
  echo "  pointing #$N  ->  $W   sky-learning region: ${REGION[*]}"
  echo "================================================================"
  T0=$(date +%s)

  echo "--- [1/7] step1 white light (from nosky)"
  $RUN src/skymodel/step1_whitelight.py "$NOSKY" --out "$W/step01" \
       > "$W/step1.log" 2>&1

  echo "--- [2/7] professor's segmentation"
  PROF_SEG=$ROOT/data/wsky_seg/DATACUBE_FINAL_${N}_seg.fits
  [ -f "$PROF_SEG" ] || { echo "★ File not found: $PROF_SEG"; exit 1; }
  cp "$PROF_SEG" "$W/step01/seg.fits"
  # Matching shapes does not guarantee the same pixel grid, so the WCS must
  # also be compared. The check is "where does the same pixel point on the sky",
  # not a keyword-by-keyword comparison -- the professor's seg uses the CD
  # matrix while the cube uses PC+CDELT, different encodings that can describe
  # the same grid; and CRPIX differs by 0.01 px (275.9 vs 275.91), which a
  # literal comparison would flag as misalignment.
  $RUN -c "
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import numpy as np, sys
s, hs = fits.getdata('$W/step01/seg.fits', header=True)
w, hw = fits.getdata('$W/step01/whitelight.fits', header=True)
if s.shape != w.shape:
    sys.exit(f'★ seg {s.shape} and white light {w.shape} have different shapes')
if 'CTYPE1' not in hw:
    sys.exit('★ white light has no WCS -- written by an old step1; please re-run step1')
ny, nx = s.shape
y = np.array([0, 0, ny - 1, ny - 1, ny // 2])
x = np.array([0, nx - 1, 0, nx - 1, nx // 2])
ws, ww = WCS(hs).celestial, WCS(hw).celestial  # seg header still carries the cube's 3rd axis
sep = ws.pixel_to_world(x, y).separation(ww.pixel_to_world(x, y)).arcsec
off = sep.max() / (proj_plane_pixel_scales(ww)[0] * 3600)
if off > 0.1:
    sys.exit(f'★ seg and white light pixel grids offset by {off:.2f} px (max over corners and centre)')
print(f'    {len(np.unique(s))-1} sources, mask {100*(s>0).mean():.1f}%, grid offset {off:.3f} px')"

  echo "--- [3/7] step2 source spectra (nosky, for classification)"
  $RUN src/skymodel/step2_object_spectra.py --work "$W" --cube "$NOSKY" \
       --out "$W/step02" > "$W/step2.log" 2>&1

  echo "--- [4/7] step3 sky basis (sky-learning region: ${REGION[*]})"
  $RUN src/skymodel/step3_sky_basis.py --methods svd -K 30 \
       --work "$W" --cube "$WSKY" "${REGION[@]}" 2>&1 \
       | tee "$W/step3.log" | grep -E "spatial restriction|exclude-box|blank spaxels|svd "

  echo "--- [5/7] step4 template fitting and classification"
  # Variable parameters for step4 are gathered here; BEST below assembles its
  # filename from the same set of variables -- changing a parameter without
  # updating BEST would let step5 silently use the previous classification file.
  SFIX=0.0
  WIN=(4600 8000)
  LITER=1
  $RUN src/skymodel/step4_fit_source.py --id all --basis svd -K 30 --s-fix "$SFIX" \
       --star-window "${WIN[@]}" --gal-window "${WIN[@]}" --line-mask-iter "$LITER" \
       --spec-dir "$W/step02" --work "$W" 2>&1 | tee "$W/step4.log" | tail -3

  echo "--- [6/7] step5 build s-field"
  # The filename is assembled following step4_fit_source.py's make_tag(); this
  # is a second implementation of the same logic, so every segment must come
  # from the variable set above, not be hand-typed.
  BEST=$W/step04/classification_nobasis_s${SFIX}_${WIN[0]}-${WIN[1]}_${WIN[0]}-${WIN[1]}_L${LITER}cum.npz
  $RUN src/skymodel/step5_s_field.py --basis svd -K 30 \
       --work "$W" --cube "$WSKY" --best "$BEST" 2>&1 \
       | tee "$W/step5.log" | grep -E "main source|s spatial field|saved"

  echo "--- [7/7] step6 final sky subtraction"
  $RUN src/skymodel/step6_fit_sky.py --basis svd -K 30 \
       --work "$W" --cube "$WSKY" --best "$BEST" \
       --s-field "$W/step05/s_hat.npy" 2>&1 \
       | tee "$W/step6.log" | grep -E "blank|source |saved"

  echo "*** pointing #$N done, elapsed $(( $(date +%s) - T0 )) s"
  df -h "$ROOT" | tail -1
done
