#!/usr/bin/env bash
# 라이다 스택 상태 한눈에 보기.
export PATH="$HOME/.local/bin:$PATH"
source /opt/ros/humble/setup.bash 2>/dev/null
echo "=== 라이다 스택 상태 ==="
chk() { pgrep -f "$1" >/dev/null && echo "  ✓ $2" || echo "  ✗ $2 (꺼짐)"; }
chk sllidar_node            "라이다 드라이버"
chk async_slam_toolbox_node "SLAM (slam_toolbox)"
chk wall_localizer.py       "위치추정 (wall_localizer)"
chk live_map.py             "점지도 PNG (live_map)"
chk wall_map.py             "벽선 PNG (wall_map)"
chk map_server.py           "웹서버 (map_server, 8000)"
echo "--- ROS 토픽 ---"
scan=$(timeout 4 ros2 topic list 2>/dev/null)
echo "$scan" | grep -q '^/scan$' && echo "  ✓ /scan" || echo "  ✗ /scan"
echo "$scan" | grep -q '^/map$'  && echo "  ✓ /map"  || echo "  ✗ /map"
echo "$scan" | grep -q '^/wall_position$' && echo "  ✓ /wall_position" || echo "  ✗ /wall_position"
ss -tlnp 2>/dev/null | grep -q ':8000' && echo "  ✓ 웹 8000 리스닝" || echo "  ✗ 웹 8000"
