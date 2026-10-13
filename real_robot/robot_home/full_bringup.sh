#!/usr/bin/env bash

set -e
source /opt/ros/humble/setup.bash
source ~/limo_ros2_ws/install/setup.bash

echo "=== 0. clearing anything already running ==="
pkill -f "[l]imo_base"   2>/dev/null && echo "    stopped a standalone limo_base" || true
pkill -f "[l]imo_start"  2>/dev/null || true
pkill -f "[y]dlidar"     2>/dev/null || true
sleep 3

echo "=== 1. base + lidar (limo_start) ==="
setsid nohup ros2 launch limo_bringup limo_start.launch.py </dev/null >/tmp/limo_start.log 2>&1 &
sleep 12

echo "=== 2. camera, correct transform, perception ==="
bash ~/perception_up.sh

echo
echo "=================== READY? ==================="
ok=1
chk () {  # topic, human name
    if timeout 6 ros2 topic echo "$1" --once >/dev/null 2>&1; then
        echo "  OK    $2  ($1)"
    else
        echo "  MISSING  $2  ($1)"; ok=0
    fi
}
chk /odom     "base driver"
chk /scan     "lidar (bumper guard)"
chk /box_pose "cube detection"

echo "--- camera transform (MUST be 0.112, 0.020, 0.040) ---"
timeout 5 ros2 run tf2_ros tf2_echo base_link camera_link 2>/dev/null | grep -m1 Translation || echo "  MISSING"

echo "--- where the cube is right now ---"
timeout 6 ros2 topic echo /box_pose --field pose.position --once 2>/dev/null || echo "  no cube seen"

echo
if [ "$ok" = 1 ]; then
    echo "ALL PRESENT. Before starting a series, check that y above is between -0.040 and"
    echo "+0.040 -- the tracks cannot rotate finer than ~5 deg (~70 mm at 0.8 m), so a"
    echo "smaller correction than that oscillates and times out. Nudge the robot if needed."
    echo
    echo "  python3 ~/face_and_pick.py --setback 0.020 --repeat 20 --series S6 --lights on"
else
    echo "SOMETHING IS MISSING above -- do not start a series, it will abort on that step."
    echo "  base  : tail -20 /tmp/limo_start.log"
    echo "  camera: tail -20 /tmp/cam.log"
fi
echo "=============================================="
