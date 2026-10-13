import os
from os import environ, pathsep

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from launch.conditions import IfCondition

def generate_launch_description():
    if os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        raise RuntimeError(
            '\n\nSIMULATION BLOCKED: this terminal is not isolated from the real robot '
            '(shared ROS network -- the simulation could drive the real LIMO).\n'
            'Run first:   source tools/sim_env.sh\n')

    package_name = 'limo_car'

    world_config = LaunchConfiguration('world')
    rviz_config  = LaunchConfiguration('rvizconfig')

    pkg_path = get_package_share_directory(package_name)

    default_world_path      = os.path.join(pkg_path, 'worlds', 'final_map.world')
    default_rviz_config_path = os.path.join(pkg_path, 'rviz',   'gazebo.rviz')

    world_arg = DeclareLaunchArgument(
        'world', default_value=default_world_path,
        description='Absolute path to Gazebo world file')

    rviz_arg = DeclareLaunchArgument(
        'rvizconfig', default_value=default_rviz_config_path,
        description='Absolute path to rviz config file')

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Whether to launch RViz')

    use_gzclient_arg = DeclareLaunchArgument(
        'use_gzclient', default_value='true',
        description='Whether to launch Gazebo client (GUI)')

    odometry_source_arg = DeclareLaunchArgument(
        'odometry_source', default_value='1',
        description='1 = WORLD (exact odom), 0 = ENCODER (realistic drift; '
                    'requires localization:=amcl)')

    spawn_x_arg = DeclareLaunchArgument(
        'spawn_x', default_value='-2.0',
        description='Robot spawn X in the map/world frame')
    spawn_yaw_arg = DeclareLaunchArgument(
        'spawn_yaw', default_value='-1.5708',
        description='Robot spawn yaw (rad) in the map/world frame')

    spawn_y_arg = DeclareLaunchArgument(
        'spawn_y', default_value='7.0',
        description='Robot spawn Y (m). 7.0 = at the table (isolated grasp test); '
                    'larger (e.g. 7.6) = farther back so the robot drives up for an approach.')

    use_camera_arg = DeclareLaunchArgument(
        'use_camera', default_value='true',
        description='Spawn the depth camera. Set false for GUI driving/nav demos to avoid '
                    'the intermittent Gazebo GPU depth-camera segfault.')

    drive_mode_arg = DeclareLaunchArgument(
        'drive_mode', default_value='diff',
        description='Drive mode (ackermann | diff)')

    gazebo_ros_share  = get_package_share_directory('gazebo_ros')
    mycobot_models = os.path.dirname(get_package_share_directory('mycobot_description'))
    limo_models = os.path.dirname(get_package_share_directory('limo_car'))
    model_path  = os.path.join(gazebo_ros_share, 'models') + pathsep + mycobot_models + pathsep + limo_models
    from ament_index_python.packages import get_package_prefix
    patched_gzros2_lib = os.path.join(get_package_prefix('gazebo_ros2_control'), 'lib')
    grasp_plugin_lib = os.path.join(
        os.path.dirname(get_package_prefix('limo_car')), 'gazebo_grasp_plugin', 'lib')
    plugin_path = (patched_gzros2_lib + pathsep + grasp_plugin_lib
                   + pathsep + '/opt/ros/humble/lib')
    if 'GAZEBO_MODEL_PATH' in environ:
        model_path  += pathsep + environ['GAZEBO_MODEL_PATH']
    if 'GAZEBO_PLUGIN_PATH' in environ:
        plugin_path += pathsep + environ['GAZEBO_PLUGIN_PATH']

    gazebo_env = {
        'GAZEBO_MODEL_PATH':  model_path,
        'GAZEBO_PLUGIN_PATH': plugin_path,
        '__NV_PRIME_RENDER_OFFLOAD': '1',
        '__GLX_VENDOR_LIBRARY_NAME': 'nvidia',
        '__VK_LAYER_NV_optimus': 'NVIDIA_only',
    }

    gzserver_cmd = ExecuteProcess(
        cmd=['gzserver', '--verbose', world_config,
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so'],
        additional_env=gazebo_env,
        output='screen'
    )

    gzclient_cmd = ExecuteProcess(
        cmd=['gzclient'],
        additional_env=gazebo_env,
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_gzclient'))
    )

    mbot = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(pkg_path, 'launch', 'ackermann.launch.py')
        ]),
        launch_arguments={
            'use_sim_time': 'true',
            'world': world_config,
            'drive_mode': LaunchConfiguration('drive_mode'),
            'use_camera': LaunchConfiguration('use_camera'),
            'odometry_source': LaunchConfiguration('odometry_source')
        }.items()
    )

    spawn_entity = ExecuteProcess(
        cmd=['/usr/bin/python3',
             '/opt/ros/humble/lib/gazebo_ros/spawn_entity.py',
             '-topic', 'robot_description',
             '-entity', 'mbot',
             '-x', LaunchConfiguration('spawn_x'),
             '-y', LaunchConfiguration('spawn_y'),
             '-z', '0.0',
             '-Y', LaunchConfiguration('spawn_yaw')],
        output='screen'
    )

    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        output='screen', arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('use_rviz')))

    joint_state_broadcaster_spawner = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster'])

    arm_controller_spawner = Node(
        package='controller_manager', executable='spawner',
        arguments=['mycobot_arm_controller'])

    gripper_controller_spawner = Node(
        package='controller_manager', executable='spawner',
        arguments=['mycobot_gripper_controller'])

    return LaunchDescription([
        world_arg,
        rviz_arg,
        odometry_source_arg,
        spawn_x_arg,
        spawn_y_arg,
        spawn_yaw_arg,
        use_rviz_arg,
        use_gzclient_arg,
        drive_mode_arg,
        use_camera_arg,
        mbot,
        gzserver_cmd,
        gzclient_cmd,
        spawn_entity,
        joint_state_broadcaster_spawner,
        arm_controller_spawner,
        gripper_controller_spawner,
        rviz_node,
    ])
