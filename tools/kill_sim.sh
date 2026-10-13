#!/usr/bin/env bash

set -uo pipefail

PATTERN='[g]zserver|[g]zclient|[r]viz2/rviz2|[m]oveit_ros_move_group/move_group|[r]obot_state_publisher/robot_state_publisher|[l]imo_car/nav_pick_orchestrator|[l]imo_car/grasp_attacher|[l]imo_car/smart_grasp|[l]imo_car/box_pose_estimator|[l]imo_car/tag_dock_estimator|[l]imo_car/base_pin|[l]imo_car/base_joint_state_pub|[n]av2_[a-z_]*/|[c]ontroller_manager/spawner'

pids=$(pgrep -f "$PATTERN" 2>/dev/null | tr '\n' ' ')

if [ -z "${pids// }" ]; then
  echo "Nothing running — already clean."
  exit 0
fi

echo "Found simulation processes:"
ps -o pid,cmd -p ${pids} 2>/dev/null | tail -n +2 | cut -c1-110
echo
echo "Killing: ${pids}"
kill -9 ${pids} 2>/dev/null
sleep 2

remaining=$(pgrep -f "$PATTERN" 2>/dev/null | tr '\n' ' ')
if [ -n "${remaining// }" ]; then
  echo "Second pass on survivors: ${remaining}"
  kill -9 ${remaining} 2>/dev/null
  sleep 2
fi

final=$(pgrep -f "$PATTERN" 2>/dev/null | tr '\n' ' ')
if [ -z "${final// }" ]; then
  echo "CLEAN — verified nothing left."
  exit 0
else
  echo "STILL ALIVE (investigate before launching):"
  ps -o pid,cmd -p ${final} 2>/dev/null | tail -n +2 | cut -c1-110
  exit 1
fi
