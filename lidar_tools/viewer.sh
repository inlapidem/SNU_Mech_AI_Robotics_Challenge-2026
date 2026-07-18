#!/usr/bin/env bash
# 맵 실시간 웹뷰어 서버. 노트북 브라우저에서 http://<젯슨IP>:8000 접속.
# 사용법: ./viewer.sh [포트]   (기본 8000)
PORT="${1:-8000}"
cd "$HOME/lidar_ws/maps"
echo "웹뷰어 시작: http://<젯슨IP>:${PORT}  (종료: Ctrl+C)"
echo "  - Tailscale:  http://100.75.201.61:${PORT}"
echo "  - LAN:        http://10.141.160.11:${PORT}"
echo "  ('지도 초기화' 버튼 지원 서버)"
# 정적 http.server 대신 map_server.py: 파일 서빙 + POST /api/reset(지도 초기화) 처리
exec python3 "$HOME/lidar_ws/map_server.py" "$PORT"
