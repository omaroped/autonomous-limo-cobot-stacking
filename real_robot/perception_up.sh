#!/bin/bash

source /opt/ros/humble/setup.bash
source ~/limo_ros2_ws/install/setup.bash

echo "=== 1. removing any base_link -> camera_link publisher (vendor's is wrong) ==="
pkill -f "[s]tatic_transform_publisher.*base_link camera_link" 2>/dev/null \
  && echo "    killed the old one" || echo "    none was running"
pkill -f "[s]tatic_transform_publisher.*camera_link camera_depth_optical_frame" 2>/dev/null
sleep 1

echo "=== 2. publishing the MEASURED camera transforms ==="
setsid nohup ros2 run tf2_ros static_transform_publisher \
    0.112 0.020 0.040 0 0 0 base_link camera_link \
    </dev/null >/tmp/cam_tf_mount.log 2>&1 &
setsid nohup ros2 run tf2_ros static_transform_publisher \
    0 0 0 -1.570796 0 -1.570796 camera_link camera_depth_optical_frame \
    </dev/null >/tmp/cam_tf_optical.log 2>&1 &
sleep 2

echo "=== 3. camera ==="
if pgrep -f "[d]abai.launch.py" >/dev/null; then
    echo "    already running, left alone"
else
    setsid nohup ros2 launch orbbec_camera dabai.launch.py depth_registration:=true \
        enable_soft_filter:=false \
        </dev/null >/tmp/cam.log 2>&1 &
    echo "    started, waiting for it to enumerate"
    sleep 16
fi

echo "=== 4. box_pose_estimator (hardware parameters, target_policy rightmost, depth_mode mask) ==="
pkill -f "[b]ox_pose_est" 2>/dev/null && sleep 2
setsid nohup python3 ~/box_pose_estimator.py --ros-args \
  -p box_edge_m:=0.025 \
  -p search_x_min:=0.25 -p search_x_max:=1.10 \
  -p search_y_abs:=0.45 \
  -p search_z_min:=-0.09 -p search_z_max:=0.05 \
  -p min_area_fraction:=0.15 \
  -p target_policy:=rightmost -p depth_mode:=mask \
  -r /depth_camera/depth/camera_info:=/camera/depth/camera_info \
  -r /depth_camera/depth/image_raw:=/camera/depth/image_raw \
  -r /rgb/image_raw:=/camera/color/image_raw \
  </dev/null >/tmp/box.log 2>&1 &
sleep 10

echo "=== 5. live camera viewer on port 8090 ==="
pkill -f "[l]ive_view.py" 2>/dev/null && sleep 1
setsid nohup python3 ~/live_view.py </dev/null >/tmp/view.log 2>&1 &
echo "    open http://$(hostname -I | awk '{print $1}'):8090 on the laptop"
sleep 3

echo
echo "=================== VERIFY ==================="
echo "--- navigation still alive? (all five must be listed) ---"
NAV=$(timeout 25 ros2 node list 2>/dev/null \
  | grep -E "^/(amcl|controller_server|planner_server|bt_navigator|behavior_server)$" | sort)
echo "$NAV"
[ "$(echo "$NAV" | grep -c .)" -eq 5 ] || echo "  !! NOT ALL FIVE NAV NODES SEEN -- check before running a pick"
echo "--- camera transform (MUST read 0.112, 0.020, 0.040) ---"
timeout 8 ros2 run tf2_ros tf2_echo base_link camera_link 2>/dev/null \
  | grep -m1 "Translation" || echo "  no transform"
echo "--- /box_pose (put a blue cube in front of the camera first) ---"
timeout 8 ros2 topic echo /box_pose --field pose.position --once 2>/dev/null \
  | tr '\n' ' ' || echo "  nothing seen yet"
echo
echo "=============================================="
