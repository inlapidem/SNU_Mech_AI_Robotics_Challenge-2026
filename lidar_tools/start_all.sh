#!/usr/bin/env bash
# RPLIDAR C1 라이다 스택 일괄 실행.
#   1) 드라이버  2) SLAM  3) 위치추정  4) 점지도PNG  5) 벽선PNG  6) 웹서버(버튼)
# 기존 인스턴스를 먼저 정리한 뒤 순서대로 백그라운드 실행. 각 로그는 ~/lidar_ws/logs/ 에.
# 상태: ~/lidar_ws/status.sh   종료: ~/lidar_ws/stop_all.sh
export PATH="$HOME/.local/bin:$PATH"
source /opt/ros/humble/setup.bash 2>/dev/null
source "$HOME/lidar_ws/install/setup.bash" 2>/dev/null

WS="$HOME/lidar_ws"; LOG="$WS/logs"; mkdir -p "$LOG"

echo "[start_all] 기존 인스턴스 정리…"
# 스크립트 파일 내용은 프로세스 cmdline 에 안 들어가므로 pkill 안전(자기 자신 안 죽임)
for pat in sllidar_node async_slam_toolbox_node tf_base_to_laser tf_odom_to_base \
           'ros2 launch .*map_launch' wall_localizer.py live_map.py wall_map.py \
           map_server.py http.server; do
  pkill -f "$pat" 2>/dev/null
done
sleep 2

run() {  # run <표시이름> <로그파일> <명령...>
  local name="$1" logf="$2"; shift 2
  setsid "$@" > "$LOG/$logf" 2>&1 < /dev/null &
  echo "[start_all]   ▶ $name   (로그: $LOG/$logf)"
}

echo "[start_all] 1/6 라이다 드라이버…"
run "라이다 드라이버" driver.log ros2 launch sllidar_ros2 sllidar_c1_launch.py
for i in $(seq 1 15); do ros2 topic list 2>/dev/null | grep -q '^/scan$' && break; sleep 1; done
if ros2 topic list 2>/dev/null | grep -q '^/scan$'; then
  echo "[start_all]     /scan OK"
else
  echo "[start_all]     ✗ /scan 안 나옴 — 라이다 연결/USB-A 포트 확인 ($LOG/driver.log)"
fi

echo "[start_all] 2/6 SLAM (slam_toolbox + TF)…"
run "SLAM" slam.log ros2 launch "$WS/map_launch.py"
sleep 3

echo "[start_all] 3/6 위치추정 (wall_localizer)…"
run "위치추정" wall_localizer.log python3 "$WS/wall_localizer.py"

echo "[start_all] 4/6 점지도 PNG (live_map)…"
run "점지도 PNG" live_map.log python3 "$WS/live_map.py" "$WS/maps/live_map.png" 1.0

echo "[start_all] 5/6 벽선 PNG (wall_map)…"
run "벽선 PNG" wall_map.log python3 "$WS/wall_map.py" "$WS/maps/wall_map.png" 1.0

echo "[start_all] 6/6 웹서버 (map_server, 버튼 지원)…"
run "웹서버" web.log python3 "$WS/map_server.py" 8000
sleep 2

LAN=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^(10|192\.168)\.' | head -1)
TS=$(tailscale ip -4 2>/dev/null | head -1)
echo
echo "[start_all] 완료 — 브라우저 접속 (Ctrl+F5로 새로고침):"
echo "   젯슨 화면 : http://localhost:8000/view.html"
[ -n "$LAN" ] && echo "   LAN      : http://$LAN:8000/view.html"
[ -n "$TS" ]  && echo "   Tailscale: http://$TS:8000/view.html"
echo "   상태: ~/lidar_ws/status.sh   |   종료: ~/lidar_ws/stop_all.sh"
