#!/usr/bin/env bash
# 用 step4 的 Δχ² 比較 step3 產出的六種 sky basis。
#
# 為什麼要用 Δχ² 而不是 blank 區的殘差:
#   blank 殘差是循環論證 —— 未加權擬合用未加權 rms 評分、加權擬合用 reduced chi2 評分,
#   每一方都在自己的考卷上贏。Δχ² 量的是「加入源之後解釋力改善多少」,
#   天空模型和源模型都沒有在最佳化它,所以能跳出這個循環。
#
# 裁判:Haro 11 的文獻紅移 z = 0.0206。哪一組 basis 讓 z 最接近、dchi2 最大,哪一組就好。
#
# 用法:  bash src/skymodel/tools/compare_basis_dchi2.sh [ID]

set -uo pipefail

ID="${1:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$ROOT/src/skymodel/step4_find_template.py"

echo "ID $ID   z 精掃 0.018 - 0.023  step 2.5e-5   (文獻值 z = 0.0206)"
echo
printf '%-14s %6s %10s %14s %9s %11s\n' basis tpl z dchi2 s red_chi2
printf -- '------------------------------------------------------------------\n'

for b in svd pca nmf rpca; do
    line=$(conda run -n astro python "$SCRIPT" \
               --id "$ID" --zmin 0.018 --zmax 0.023 --zstep 2.5e-5 --basis "$b" 2>/dev/null \
           | awk -v id="$ID" '$1 == id && NF >= 8 {print $3, $4, $7, $6, $8}')

    if [ -z "$line" ]; then
        printf '%-14s %s\n' "$b" "(擬合失敗或無結果)"
    else
        # shellcheck disable=SC2086
        printf '%-14s %6s %10s %14s %9s %11s\n' "$b" $line
    fi
done
