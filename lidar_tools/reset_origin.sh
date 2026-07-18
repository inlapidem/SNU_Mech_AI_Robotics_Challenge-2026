#!/usr/bin/env bash
# 위치 원점 리셋 — wall_localizer 의 현재 위치를 (0,0)으로,
# 그리고 지금 보이는 '가장 큰 벽'을 X축으로 다시 고정한다. (지도/slam 은 건드리지 않음)
export PATH="$HOME/.local/bin:$PATH"
source /opt/ros/humble/setup.bash 2>/dev/null
source "$HOME/lidar_ws/install/setup.bash" 2>/dev/null
ros2 service call /wall_localizer/reset std_srvs/srv/Trigger >/dev/null 2>&1
echo "origin reset"
