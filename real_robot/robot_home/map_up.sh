#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
source ~/limo_ros2_ws/install/setup.bash
echo "=== 0. stop leftovers ==="
for p in "[l]imo_base" "[l]imo_start" "[y]dlidar" "[s]lam_toolbox" "[a]sync_slam" "[s]tatic_transform_publisher.*laser_link"; do
  pkill -f "$p" 2>/dev/null && echo "    stopped $p"; done
sleep 3
echo "=== 1. base + lidar (limo_start) ==="
setsid nohup ros2 launch limo_bringup limo_start.launch.py </dev/null >/tmp/limo_start.log 2>&1 &
sleep 12
echo "=== 2. measured lidar transform ==="
bash ~/lidar_tf_up.sh || { echo "STOP: lidar transform"; exit 1; }
echo "=== 3. slam_toolbox (base_frame base_link) ==="
sed -e 's/base_frame: base_footprint/base_frame: base_link/' \
    -e 's/minimum_travel_distance: 0.5/minimum_travel_distance: 0.2/' \
    -e 's/minimum_travel_heading: 0.5/minimum_travel_heading: 0.2/' \
    -e 's/max_laser_range: 20.0/max_laser_range: 12.0/' \
    -e 's/map_update_interval: 5.0/map_update_interval: 2.0/' \
    /opt/ros/humble/share/slam_toolbox/config/mapper_params_online_async.yaml > ~/slam_params_lab.yaml
grep -E "base_frame|minimum_travel|max_laser_range|map_update" ~/slam_params_lab.yaml
setsid nohup ros2 launch slam_toolbox online_async_launch.py slam_params_file:=$HOME/slam_params_lab.yaml \
    use_sim_time:=false </dev/null >/tmp/slam.log 2>&1 &
sleep 10
echo "=== 4. checks ==="
timeout 6 ros2 topic echo /odom --once >/dev/null 2>&1 && echo "  OK  /odom" || echo "  MISSING /odom"
timeout 6 ros2 topic echo /scan --once >/dev/null 2>&1 && echo "  OK  /scan" || echo "  MISSING /scan"
timeout 8 ros2 run tf2_ros tf2_echo map base_link 2>/dev/null | grep -m1 Translation && echo "  OK  map -> base_link" || echo "  MISSING map -> base_link (see /tmp/slam.log)"
timeout 5 ros2 topic echo --once /limo_status 2>/dev/null | grep battery
echo "Drive (second terminal, robot):"
echo "  source /opt/ros/humble/setup.bash; ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p speed:=0.10 -p turn:=0.25"
