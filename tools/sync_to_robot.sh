#!/usr/bin/env bash

set -euo pipefail

ROBOT="${1:?usage: tools/sync_to_robot.sh user@robot-address}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../src" && pwd)"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

echo "=== staging the trimmed package set ==="

rsync -a --exclude="__pycache__" --exclude="*.pyc" \
  "$REPO_ROOT/limo_car/" "$STAGING/limo_car/"
rsync -a "$REPO_ROOT/limo_cobot_moveit_config/" "$STAGING/limo_cobot_moveit_config/"

MCD_SRC="$REPO_ROOT/mycobot_description"
MCD_DST="$STAGING/mycobot_description"
mkdir -p "$MCD_DST/urdf"
cp "$MCD_SRC/package.xml" "$MCD_SRC/setup.py" "$MCD_DST/"
[ -f "$MCD_SRC/setup.cfg" ] && cp "$MCD_SRC/setup.cfg" "$MCD_DST/"
cp -r "$MCD_SRC/resource" "$MCD_DST/"
cp -r "$MCD_SRC/mycobot_description" "$MCD_DST/mycobot_description"
rsync -a --exclude="*.STEP" "$MCD_SRC/urdf/mycobot_280_m5" "$MCD_DST/urdf/"
rsync -a "$MCD_SRC/urdf/adaptive_gripper" "$MCD_DST/urdf/"

echo "staged: $(du -sh "$STAGING" | cut -f1)"
echo

echo "=== deploying to $ROBOT:~/moveit_ws/src ==="
ssh "$ROBOT" 'mkdir -p ~/moveit_ws/src'
rsync -az "$STAGING/" "$ROBOT:~/moveit_ws/src/"

echo
echo "=== building on the robot (underlay: ~/limo_ros2_ws) ==="
ssh "$ROBOT" '
  source /opt/ros/humble/setup.bash
  source ~/limo_ros2_ws/install/setup.bash
  cd ~/moveit_ws && colcon build
'

echo
echo "DONE. On the robot: source ~/limo_ros2_ws/install/setup.bash && source ~/moveit_ws/install/setup.bash"
