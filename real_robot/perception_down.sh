#!/bin/bash

pkill -f "[f]ace_and_pick.py"                                  && echo "stopped face_and_pick"
pkill -f "[b]ox_pose_estimator.py"                             && echo "stopped box_pose_estimator"
pkill -f "[l]ive_view.py"                                      && echo "stopped live_view"
pkill -f "[d]abai.launch.py"                                   && echo "stopped camera launch"
pkill -f "[_]_node:=camera_container"                          && echo "stopped camera container"
pkill -f "[s]tatic_transform_publisher.*base_link camera_link" && echo "stopped camera mount TF"
pkill -f "[s]tatic_transform_publisher.*camera_depth_optical"  && echo "stopped camera optical TF"
sleep 2
echo "--- perception processes left (should be empty):"
pgrep -af "[b]ox_pose_estimator|[l]ive_view|[d]abai.launch|[_]_node:=camera_container|[f]ace_and_pick" || echo "  none"
