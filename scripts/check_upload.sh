#!/bin/bash
# 一鍵檢查 rclone 上傳狀態。隨時可在 Claude Code 輸入框用  ! bash scripts/check_upload.sh  跑。
# 可選第 1 參數：log 檔路徑（預設 = maskfrom-nosky zap.fits 那次上傳的 log）。
LOG="${1:-/tmp/claude-1035/-local-feather-workspace-astro/b667f393-b741-4796-86ad-74480f3f1e93/scratchpad/rclone_zap_masknosky.log}"

echo "===== rclone 上傳狀態 ====="
if pgrep -f "rclone copy results/zap" >/dev/null 2>&1; then
  echo "狀態 : ● 執行中"
else
  echo "狀態 : ○ 未執行（已完成或已停止）"
fi

echo "進度 : $(grep 'GBytes' "$LOG" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//' || echo '尚無')"
echo "檔案 : $(grep -E 'Transferred:[^G]*[0-9]+ / [0-9]+,' "$LOG" 2>/dev/null | tail -1 | sed 's/^[[:space:]]*//')"
ERRS=$(grep -ciE 'ratelimit|error|403|fatal' "$LOG" 2>/dev/null)
echo "錯誤 : ${ERRS:-0} 次 rate-limit/error"
[ "${ERRS:-0}" != "0" ] && grep -iE 'ratelimit|error|403|fatal' "$LOG" 2>/dev/null | tail -2

echo "----- 遠端已到的 zap.fits（可能受 API 限額，稍等再試）-----"
timeout 45 rclone lsf gdrive:astro/cubes -R --include '*/zap.fits' 2>&1 | sed 's/^/  /'
