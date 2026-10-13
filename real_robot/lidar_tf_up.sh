#!/usr/bin/env bash

source /opt/ros/humble/setup.bash
source ~/limo_ros2_ws/install/setup.bash
X=0.108
YAW=-0.13614

echo "=== 1. current base_link -> laser_link (vendor) ==="
cur=$(timeout 6 ros2 run tf2_ros tf2_echo base_link laser_link 2>/dev/null | grep -m2 -E "Translation|RPY \(degree\)")
echo "${cur:-  none found}"
Z=$(echo "$cur" | grep -m1 Translation | sed -E 's/.*\[([^,]+), *([^,]+), *([^]]+)\].*/\3/' | tr -d ' ')
if [ -z "$Z" ]; then
    echo "  could not read the vendor height; is limo_start running?  Set it by hand: Z_OVERRIDE=0.10 bash ~/lidar_tf_up.sh"
    Z=${Z_OVERRIDE:-}
    [ -z "$Z" ] && exit 1
fi
echo "  keeping the vendor height z = $Z m"

echo "=== 2. removing the vendor's laser transform publisher ==="
pkill -f "[s]tatic_transform_publisher.*laser_link" && echo "    killed" || echo "    none was running"
sleep 2
if timeout 4 ros2 run tf2_ros tf2_echo base_link laser_link 2>/dev/null | grep -q Translation; then
    echo "  STOP: base_link -> laser_link is STILL published by something else"
    echo "        (robot_state_publisher from a URDF? another launch?). Not starting a second one."
    ps aux | grep -E "[r]obot_state_publisher|[s]tatic_transform" | cut -c1-160
    exit 1
fi
echo "    gone -- nothing else publishes it"

echo "=== 3. starting the measured transform: x $X, yaw -7.8 deg, z $Z ==="
setsid nohup ros2 run tf2_ros static_transform_publisher \
    $X 0 $Z $YAW 0 0 base_link laser_link </dev/null >/tmp/lidar_tf.log 2>&1 &
sleep 3

echo "=== 4. check ==="
n=$(pgrep -fc "[s]tatic_transform_publisher.*laser_link")
echo "  publishers of laser_link: $n (must be 1)"
timeout 6 ros2 run tf2_ros tf2_echo base_link laser_link 2>/dev/null | grep -m2 -E "Translation|RPY \(degree\)"
echo "  expected: Translation [0.108, 0.000, $Z], RPY (degree) [0.000, -0.000, -7.800]"
