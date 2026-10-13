#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
mkdir -p ~/maps
ros2 run nav2_map_server map_saver_cli -f ~/maps/${1:-lab_$(date +%Y%m%d_%H%M)} --ros-args -p save_map_timeout:=10.0 2>&1 | grep -v "^\[INFO\]" | tail -3
ls -la ~/maps | tail -4
