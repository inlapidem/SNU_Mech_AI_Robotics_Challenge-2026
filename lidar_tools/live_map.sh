#!/usr/bin/env bash
# 10초마다 현재 맵을 PNG로 저장. 사용법: ./live_map.sh [출력경로] [주기초]
# 결과 파일(기본): ~/lidar_ws/maps/live_map.png  (계속 덮어쓰기)
export PATH="$HOME/.local/bin:$PATH"
source /opt/ros/humble/setup.bash
source "$HOME/lidar_ws/install/setup.bash"
exec python3 "$HOME/lidar_ws/live_map.py" "$@"
