# Autonomous LIMO Cobot stacking

This is the code from my bachelor thesis at Hochschule Rhein-Waal, "Autonomous Mobile
Manipulation: Design and Evaluation of a Pick-and-Place and Object Stacking Pipeline on the
LIMO Cobot".

A LIMO PRO robot with a myCobot 280 arm picks up small cubes from one table, drives to a second
table with an AprilTag and stacks them three high. It runs in Gazebo and on the real robot.

The code is shared as it is. It is the last version I used, and I don't maintain it.

## Folders

- `src/limo_car`: the simulation: robot model, world, map, Nav2 settings and the task nodes
- `src/limo_cobot_moveit_config`: MoveIt 2 settings for the arm
- `src/gazebo_grasp_plugin`: a Gazebo plugin that holds the cube while it is gripped
- `src/mycobot_description`: the arm and gripper model
- `real_robot`: the programs that run on the robot
- `tools`: scripts to set up and run the simulation
- `patches`: a patch for `gazebo_ros2_control`

## Simulation

You need Ubuntu 22.04, ROS 2 Humble, Gazebo 11, Nav2 and MoveIt 2:

```bash
sudo apt install ros-humble-gazebo-ros-pkgs ros-humble-navigation2 ros-humble-nav2-bringup \
  ros-humble-moveit ros-humble-ros2-control ros-humble-ros2-controllers
```

`gazebo_ros2_control` has to be built from source with the patch:

```bash
cd src
git clone -b 0.4.12 https://github.com/ros-controls/gazebo_ros2_control.git
cd gazebo_ros2_control && git apply ../../patches/gazebo_ros2_control_local.patch
```

Build and start from the repository root:

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build

source tools/sim_env.sh
ros2 launch limo_car nav_pick.launch.py place_stack_levels:=3
```

`sim_env.sh` keeps the simulation on its own ROS network, so it can't move a real robot on the
same network. Run `tools/kill_sim.sh` before you start again.

To run the task several times without the Gazebo window:

```bash
TIMEOUT=1500 LAUNCH_ARGS="place_stack_levels:=3 exit_when_done:=true" tools/batch_runs.sh 10
```

The results go to `data/`.

## Real robot

The robot runs the AgileX LIMO workspace for the base, lidar and camera, and MoveIt for the arm.
`real_robot/SETUP_ON_ROBOT.md` explains the setup and the start order. The task is
`real_robot/robot_home/face_and_pick.py`, and I started it like this:

```bash
bash ~/b1_pipeline.sh --pipeline --stack 3 --tag-dock --lidar-square --place-b-height 0.130 \
     --target-x 0.18 --place-b-in 0.044 --stack-x camera --series test
```

## Third-party code

- `src/limo_car` is based on AgileX's [limo_ros2](https://github.com/agilexrobotics/limo_ros2), Apache-2.0.
- `src/mycobot_description` comes from Elephant Robotics' [mycobot_ros2](https://github.com/elephantrobotics/mycobot_ros2), BSD-2-Clause.
- `src/gazebo_grasp_plugin` is based on GazeboGraspFix by Jennifer Buehler ([gazebo-pkgs](https://github.com/JenniferBuehler/gazebo-pkgs)), GPLv3.

Each of these folders has its own license file. Everything else is my own work under the MIT
license in `LICENSE`.
