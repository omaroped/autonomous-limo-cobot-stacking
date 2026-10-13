import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch.conditions import IfCondition

def generate_launch_description():
    pkg = get_package_share_directory("limo_cobot_moveit_config")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    base_joints = Node(
        package="limo_car",
        executable="base_joint_state_pub",
        name="base_joint_state_pub",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )
    move_group = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, "launch", "move_group.launch.py")),
        launch_arguments={"use_sim_time": use_sim_time}.items(),
    )
    rviz = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, "launch", "moveit_rviz.launch.py")),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "rviz_config": rviz_config
        }.items(),
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=os.path.join(pkg, "config", "moveit.rviz")),
        base_joints,
        move_group,
        rviz,
    ])
