# lidar_tools — RPLidar C1 운용 스크립트 (원본: ~/lidar_ws)

Jetson 의 colcon 작업공간 `~/lidar_ws` 에서 쓰는 커스텀 스크립트 보존본
(2026-07-18 레포로 복사). 실제 실행은 여전히 `~/lidar_ws` 에서:
build/install (colcon 산출물)과 src/sllidar_ros2 (외부 드라이버)는
레포에 포함하지 않음 — 재구축: sllidar_ros2 클론 후 colcon build.

- start_lidar.sh / start_mapping.sh / start_all.sh / stop_all.sh : 기동/정지
- wall_localizer.py / wall_map.py : 벽 정합 위치추정 (레포 localization/ 의 변형)
- live_map.py / map_server.py / render_map.py / viewer.sh : 맵 시각화
- slam_params.yaml : SLAM 파라미터
- save_map.sh / reset_map.sh / reset_origin.sh : 맵 저장/초기화
