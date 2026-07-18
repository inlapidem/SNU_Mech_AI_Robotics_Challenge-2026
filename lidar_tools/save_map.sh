#!/usr/bin/env bash
# [터미널 3] 완성된 맵 저장. 사용법: ./save_map.sh [맵이름]
# 결과물: ~/lidar_ws/maps/<이름>.pgm + <이름>.yaml
set -e
export PATH="$HOME/.local/bin:$PATH"
source /opt/ros/humble/setup.bash
source "$HOME/lidar_ws/install/setup.bash"

NAME="${1:-my_map}"
mkdir -p "$HOME/lidar_ws/maps"
cd "$HOME/lidar_ws/maps"

echo "맵 저장 중 -> $HOME/lidar_ws/maps/${NAME}.pgm / .yaml"
# nav2_map_server 가 있으면 그걸로, 없으면 slam_toolbox 서비스로 직렬화 저장
if ros2 pkg list 2>/dev/null | grep -q nav2_map_server; then
  ros2 run nav2_map_server map_saver_cli -f "${NAME}"
else
  echo "nav2_map_server 없음 -> slam_toolbox serialize 서비스 사용"
  ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph "{filename: '$HOME/lidar_ws/maps/${NAME}'}"
fi
echo "완료."
