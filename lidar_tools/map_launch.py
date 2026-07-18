#!/usr/bin/env python3
# 맵 생성용 런치 파일: TF(고정) + slam_toolbox
# 라이다 드라이버(sllidar_c1_launch.py)는 별도 터미널에서 먼저 실행되어 있어야 함.
#
# TF 트리:
#   map -> odom            (slam_toolbox 가 발행)
#   odom -> base_footprint (고정, 바퀴 오도메트리 없음)
#   base_footprint -> laser(고정, 라이다 장착 위치. 로봇에 실제 장착 시 오프셋 수정)

from launch import LaunchDescription
from launch_ros.actions import Node

SLAM_PARAMS = '/home/teamtwo/lidar_ws/slam_params.yaml'


def generate_launch_description():
    return LaunchDescription([
        # base_footprint -> laser : 라이다 장착 위치 (여기선 원점 = 로봇 중심에 장착 가정)
        # 실제 로봇에 올릴 때는 --x/--y/--z(미터), --yaw/--pitch/--roll(라디안) 로 오프셋 지정
        Node(
            package='tf2_ros', executable='static_transform_publisher',
            name='tf_base_to_laser',
            arguments=['--frame-id', 'base_footprint', '--child-frame-id', 'laser'],
        ),
        # odom -> base_footprint : 오도메트리 소스가 없으므로 고정. slam_toolbox 의 스캔정합이
        # map -> odom 보정으로 실제 이동을 흡수한다.
        Node(
            package='tf2_ros', executable='static_transform_publisher',
            name='tf_odom_to_base',
            arguments=['--frame-id', 'odom', '--child-frame-id', 'base_footprint'],
        ),
        # SLAM
        Node(
            package='slam_toolbox', executable='async_slam_toolbox_node',
            name='slam_toolbox', output='screen',
            parameters=[SLAM_PARAMS],
        ),
    ])
