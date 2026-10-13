import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    use_rviz = LaunchConfiguration('use_rviz')

    limo_car_pkg = get_package_share_directory('limo_car')
    xacro_file = os.path.join(limo_car_pkg, 'gazebo', 'real_arm.xacro')
    controllers_yaml = os.path.join(limo_car_pkg, 'config', 'mycobot_controllers.yaml')

    arm_control = LaunchConfiguration('arm_control')
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file, ' arm_control:=', arm_control,
                 ' initial_positions_file:=', LaunchConfiguration('initial_positions_file')]),
        value_type=str)
    feedback_yaml = os.path.join(limo_car_pkg, 'config', 'mycobot_controllers_real_feedback.yaml')

    def _robot_state_publisher(context):
        remap = []
        if arm_control.perform(context) == 'feedback':
            remap = [('/tf', '/real_arm/tf'), ('/tf_static', '/real_arm/tf_static')]
        return [Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{'robot_description': robot_description, 'use_sim_time': False}],
            remappings=remap,
        )]

    robot_state_publisher = OpaqueFunction(function=_robot_state_publisher)

    def _controller_manager(context):
        params = [{'robot_description': robot_description}, controllers_yaml]
        if arm_control.perform(context) == 'feedback':
            params.append(feedback_yaml)
        return [Node(package='controller_manager', executable='ros2_control_node',
                     output='screen', parameters=params)]
    controller_manager = OpaqueFunction(function=_controller_manager)

    joint_state_broadcaster_spawner = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager-timeout', '30'])

    arm_controller_spawner = Node(
        package='controller_manager', executable='spawner',
        arguments=['mycobot_arm_controller', '--controller-manager-timeout', '30'])

    gripper_controller_spawner = Node(
        package='controller_manager', executable='spawner',
        arguments=['mycobot_gripper_controller', '--controller-manager-timeout', '30'])

    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('limo_cobot_moveit_config'),
                         'launch', 'moveit_real.launch.py')),
        launch_arguments={'use_rviz': use_rviz}.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('initial_positions_file', default_value=os.path.join(
            get_package_share_directory('limo_cobot_moveit_config'), 'config', 'initial_positions.yaml'),
            description='start pose; feedback mode: /tmp/real_initial_positions.yaml (written by the feedback node)'),
        DeclareLaunchArgument('arm_control', default_value='mock',
                              description='mock (August, echo) | feedback (real joint states, B1 trial)'),
        robot_state_publisher,
        controller_manager,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        gripper_controller_spawner,
        moveit,
    ])
