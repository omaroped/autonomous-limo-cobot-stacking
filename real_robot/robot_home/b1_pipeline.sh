#!/usr/bin/env bash
# Starts the arm feedback node and MoveIt, runs face_and_pick.py --arm moveit with the given
# options, then stops both. Base, lidar, camera and (for --pipeline) Nav2 must already run.
source /opt/ros/humble/setup.bash
source ~/moveit_ws/install/setup.bash
fuser /dev/ttyACM0 && { echo "ARM PORT BUSY - abort"; exit 1; }
setsid python3 ~/real_arm_feedback_node.py > /tmp/b1_pipe_fb.log 2>&1 &
FB=$!
sleep 9
setsid ros2 launch limo_car real_arm_moveit.launch.py use_rviz:=false arm_control:=feedback initial_positions_file:=/tmp/real_initial_positions.yaml > /tmp/b1_pipe_launch.log 2>&1 &
LA=$!
sleep 35
source ~/limo_ros2_ws/install/setup.bash
python3 ~/face_and_pick.py --arm moveit "$@"
grep -E "start guard|STOPPED|resend" /tmp/b1_pipe_fb.log | tail -5
kill -INT -- -$LA; sleep 5; kill -INT -- -$FB 2>/dev/null; sleep 3; kill -9 -- -$LA -$FB 2>/dev/null
sleep 2; echo "leftovers: $(pgrep -fc '[r]os2_control_node|[m]ove_group|[r]eal_arm_feedback')"
