#!/usr/bin/env bash
# 라이다 스택 전체 종료 (드라이버·SLAM·위치추정·PNG·웹서버).
echo "[stop_all] 종료 중…"
for pat in sllidar_node async_slam_toolbox_node tf_base_to_laser tf_odom_to_base \
           'ros2 launch .*map_launch' wall_localizer.py live_map.py wall_map.py \
           map_server.py http.server; do
  pkill -f "$pat" 2>/dev/null
done
sleep 1
echo "[stop_all] 완료."
