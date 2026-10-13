import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    real_arm_xacro = os.path.join(
        get_package_share_directory('limo_car'), 'gazebo', 'real_arm.xacro')

    moveit_config = (
        MoveItConfigsBuilder("limo_cobot", package_name="limo_cobot_moveit_config")
        .robot_description(file_path=real_arm_xacro)
        .planning_pipelines(pipelines=["ompl"], default_planning_pipeline="ompl")
        .to_moveit_configs()
    )

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {
                "publish_robot_description_semantic": True,
                "trajectory_execution.allowed_goal_duration_margin": 5.0,
                "publish_planning_scene": True,
                "publish_geometry_updates": True,
                "publish_state_updates": True,
                "publish_transforms_updates": True,
                "monitor_dynamics": False,
            },
        ],
    )
    return LaunchDescription([move_group_node])
