#!/usr/bin/env bash
# configs/ 안의 데이터셋 절대경로(/home/user/joon)를 내 컴퓨터의 repo 경로로 일괄 치환.
# 새 컴퓨터에서 clone 후 한 번 실행: bash scripts/localize_paths.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLD_ROOT="/home/user/joon"

if [ "$REPO_ROOT" = "$OLD_ROOT" ]; then
    echo "repo 경로가 이미 $OLD_ROOT 이므로 바꿀 것이 없습니다."
    exit 0
fi

FILES=$(grep -rl "$OLD_ROOT" "$REPO_ROOT/configs" 2>/dev/null || true)
if [ -z "$FILES" ]; then
    echo "치환할 경로가 없습니다 (이미 적용됐거나 configs가 비어 있음)."
    exit 0
fi

echo "$FILES" | while read -r f; do
    sed -i "s|$OLD_ROOT|$REPO_ROOT|g" "$f"
    echo "updated: $f"
done
echo "완료. configs의 데이터셋 경로가 $REPO_ROOT 기준으로 바뀌었습니다."
