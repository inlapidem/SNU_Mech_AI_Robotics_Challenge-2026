#!/usr/bin/env bash
# [터미널 2] SLAM 시작: TF + slam_toolbox (맵 생성)
set -e
export PATH="$HOME/.local/bin:$PATH"
source /opt/ros/humble/setup.bash
source "$HOME/lidar_ws/install/setup.bash"

# /scan 이 올라와 있는지 확인
if ! ros2 topic list 2>/dev/null | grep -q "^/scan$"; then
  echo "경고: /scan 토픽이 없습니다. 먼저 [터미널 1]에서 start_lidar.sh 를 실행하세요."
  exit 1
fi

echo "SLAM(slam_toolbox) 시작... rviz2 에서 Fixed Frame=map 으로 맵을 확인하세요."
ros2 launch "$HOME/lidar_ws/map_launch.py"
