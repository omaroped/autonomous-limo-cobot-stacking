#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
source ~/limo_ros2_ws/install/setup.bash
MAP=${1:-$HOME/maps/lab_20261001.yaml}
PARAMS=$HOME/limo_ros2_ws/src/limo_ros2/limo_bringup/param/nav2.yaml
echo "=== 0. stop slam / old nav ==="
for p in "[a]sync_slam" "[s]lam_toolbox" "[b]ringup_launch" "[n]av2_" "[a]mcl" "[m]ap_server" "[b]t_navigator" "[c]ontroller_server" "[p]lanner_server" "[b]ehavior_server" "[w]aypoint_follower" "[s]moother_server" "[v]elocity_smoother" "[l]ifecycle_manager" "[c]omponent_container_isolated"; do
  pkill -f "$p" 2>/dev/null && echo "    stopped $p"; done
sleep 3
echo "=== 1. base, lidar, measured lidar TF present? ==="
timeout 6 ros2 topic echo /odom --once >/dev/null 2>&1 && echo "  OK  /odom" || { echo "  MISSING /odom -- run: bash ~/map_up.sh  (then Ctrl+C nothing; it detaches)"; exit 1; }
timeout 6 ros2 topic echo /scan --once >/dev/null 2>&1 && echo "  OK  /scan" || { echo "  MISSING /scan"; exit 1; }
timeout 6 ros2 run tf2_ros tf2_echo base_link laser_link 2>/dev/null | grep -m1 Translation
echo "=== 2. Nav2 with $MAP ==="
setsid nohup ros2 launch nav2_bringup bringup_launch.py map:=$MAP params_file:=$PARAMS use_sim_time:=false \
    </dev/null >/tmp/nav2.log 2>&1 &
sleep 20
for n in map_server amcl controller_server planner_server bt_navigator; do
  s=$(timeout 6 ros2 lifecycle get /$n 2>/dev/null | head -1); echo "  $n: ${s:-NO ANSWER}"; done
timeout 5 ros2 topic echo --once /limo_status 2>/dev/null | grep battery
echo "Next: set the start pose in RViz (2D Pose Estimate)"
