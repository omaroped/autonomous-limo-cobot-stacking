# Setting up the robot

The robot is a LIMO PRO with a Jetson Orin Nano 8 GB, JetPack 6, ROS 2 Humble. It has two
workspaces:

- `~/limo_ros2_ws`: the vendor drivers (`limo_ros2`, `ydlidar_ros2_driver`, `YDLidar-SDK`,
  `OrbbecSDK_ROS2`). Base, lidar and camera run from here.
- `~/moveit_ws`: MoveIt for the arm. It holds `limo_car`, `limo_cobot_moveit_config` and
  `mycobot_description` from this repository, and `topic_based_ros2_control`
  ([github.com/PickNikRobotics/topic_based_ros2_control](https://github.com/PickNikRobotics/topic_based_ros2_control)).

## What goes where

| file | on the robot |
|---|---|
| `robot_home/*` | `~/` |
| `perception_up.sh`, `perception_down.sh`, `lidar_tf_up.sh`, `arm_control.py` | `~/` |
| `vendor_nav2_humble_fixed.yaml` | `~/limo_ros2_ws/src/limo_ros2/limo_bringup/param/nav2.yaml` |
| `ydlidar_config/limo_tminipro.yaml` | `~/limo_ros2_ws/src/ydlidar_ros2_driver/params/` |

The scripts in `~/` call each other there, for example `full_bringup.sh` runs
`~/perception_up.sh`.

## Install from scratch

1. Flash JetPack 6 and install ROS 2 Humble.
2. `sudo apt install -y ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-slam-toolbox ros-humble-teleop-twist-keyboard ros-humble-nav2-map-server ros-humble-moveit`
3. `pip install pymycobot==3.3.4` (version 4 changed the API).
4. Put the vendor drivers into `~/limo_ros2_ws/src`, fix `ydlidar_launch.py` (below), build.
5. udev: YDLidar rule `10c4:ea60 -> /dev/ydlidar`; for the camera run
   `OrbbecSDK_ROS2/orbbec_camera/scripts/install_udev_rules.sh`.
6. Build `~/moveit_ws` on top of `~/limo_ros2_ws` (`tools/sync_to_robot.sh` copies and builds
   the three packages from this repository).
7. Copy the files from the table above.

### ydlidar_launch.py

`ydlidar_ros2_driver/launch/ydlidar_launch.py` uses the old launch API (`node_executable`,
`node_name`, `node_namespace`). On Humble the launch stops with
`TypeError: LifecycleNode.__init__() missing 2 required keyword-only arguments`. Fix:

```bash
cd ~/limo_ros2_ws/src/ydlidar_ros2_driver/launch
cp -n ydlidar_launch.py ydlidar_launch.py.bak
sed -i 's/node_executable=/executable=/; s/node_name=/name=/; s/node_namespace=/namespace=/' ydlidar_launch.py
cd ~/limo_ros2_ws && colcon build --packages-select ydlidar_ros2_driver
```

Use `limo_tminipro.yaml` from `ydlidar_config/` as the parameter file.

## Running the task

1. `bash ~/full_bringup.sh` starts the base, lidar and camera.
2. `bash ~/lidar_tf_up.sh` sets the measured lidar position.
3. `bash ~/nav_up.sh ~/maps/MAP.yaml` starts Nav2. Make the map once with `map_up.sh`,
   drive around, then `map_save.sh`.
4. On the laptop, open RViz with `nav.rviz` and set the start pose with 2D Pose Estimate.
5. `bash ~/b1_pipeline.sh ...` as in the main README.

`bash ~/perception_down.sh` stops the camera and perception nodes.

## Ports

| device | port | USB id |
|---|---|---|
| myCobot 280 arm | `/dev/ttyACM0` | CH340 `1a86:55d4` |
| YDLidar T-mini Pro | `/dev/ttyUSB0` -> `/dev/ydlidar` | CP2102 `10c4:ea60`, 230400 baud |
| Orbbec DaBai DC1 camera | USB | `2bc5:0657` and `2bc5:0557` |
| LIMO base | `/dev/ttyTHS1` | publishes `odom` -> `base_link` |

- The arm needs about 2 s after the port opens before it answers `get_angles()`.
- Only one program may open the arm port. In the task this is `real_arm_feedback_node.py`.
- The lidar publishes with Best Effort QoS; in RViz set the topic to Best Effort.
- Always shut the Jetson down with `sudo shutdown -h now` and wait before cutting power.
- If the Jetson does not boot (blinking cursor after UEFI), reseat the NVMe SSD: the M.2
  card can move out of its slot when the robot drives.
