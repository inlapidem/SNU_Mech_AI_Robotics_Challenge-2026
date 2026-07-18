#!/usr/bin/env bash
# SLAM 맵 초기화 — 누적된 지도를 지우고 slam_toolbox 를 처음부터 다시 시작한다.
# (라이다 드라이버 / wall_localizer / 뷰어(live_map,wall_map,웹서버)는 건드리지 않음)
# 웹의 '지도 초기화' 버튼이 map_server.py 를 통해 이 스크립트를 실행한다.
export PATH="$HOME/.local/bin:$PATH"
source /opt/ros/humble/setup.bash 2>/dev/null
source "$HOME/lidar_ws/install/setup.bash" 2>/dev/null

# 기존 SLAM 스택만 종료 (스크립트 파일 내용은 프로세스 cmdline 에 안 들어가므로 pkill 안전)
pkill -f async_slam_toolbox_node 2>/dev/null
pkill -f tf_base_to_laser 2>/dev/null
pkill -f tf_odom_to_base 2>/dev/null
pkill -f 'ros2 launch .*map_launch' 2>/dev/null
sleep 2

# 빈 맵부터 새로 시작
setsid ros2 launch "$HOME/lidar_ws/map_launch.py" >/dev/null 2>&1 </dev/null &
echo "map reset: slam restarted"
