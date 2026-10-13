import os

from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_moveit_rviz_launch

def generate_launch_description():
    real_arm_xacro = os.path.join(
        get_package_share_directory('limo_car'), 'gazebo', 'real_arm.xacro')

    moveit_config = (
        MoveItConfigsBuilder("limo_cobot", package_name="limo_cobot_moveit_config")
        .robot_description(file_path=real_arm_xacro)
        .to_moveit_configs()
    )
    return generate_moveit_rviz_launch(moveit_config)
