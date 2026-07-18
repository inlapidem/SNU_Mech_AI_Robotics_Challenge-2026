#!/usr/bin/env bash
# [터미널 1] RPLIDAR C1 드라이버 실행 -> /scan 발행
set -e
export PATH="$HOME/.local/bin:$PATH"
source /opt/ros/humble/setup.bash
source "$HOME/lidar_ws/install/setup.bash"
echo "RPLIDAR C1 드라이버 시작 (/dev/ttyUSB0, 460800bps)..."
ros2 launch sllidar_ros2 sllidar_c1_launch.py
