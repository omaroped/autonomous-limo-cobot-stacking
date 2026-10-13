import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, LifecycleNode
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    pkg = get_package_share_directory('limo_car')

    drive_mode   = LaunchConfiguration('drive_mode')
    localization = LaunchConfiguration('localization')

    params_file = PythonExpression(["'", os.path.join(pkg, 'config', 'nav2_limo_'), drive_mode, ".yaml'"])
    map_yaml    = os.path.join(pkg, 'maps', 'thesis_map.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    is_amcl = IfCondition(PythonExpression(["'", localization, "' == 'amcl'"]))
    is_gt   = IfCondition(PythonExpression(["'", localization, "' == 'ground_truth'"]))

    declare_use_sim = DeclareLaunchArgument(
        'use_sim_time', default_value='true', description='Use simulation clock')
    declare_drive_mode = DeclareLaunchArgument(
        'drive_mode', default_value='diff', description='Drive mode (ackermann | diff)')
    declare_init_x = DeclareLaunchArgument(
        'initial_pose_x', default_value='-2.0',
        description="AMCL initial pose X (ignored under localization:=ground_truth)")
    declare_init_y = DeclareLaunchArgument(
        'initial_pose_y', default_value='7.0',
        description="AMCL initial pose Y (ignored under localization:=ground_truth)")
    declare_init_yaw = DeclareLaunchArgument(
        'initial_pose_yaw', default_value='-1.5708',
        description="AMCL initial pose yaw (ignored under localization:=ground_truth)")

    declare_localization = DeclareLaunchArgument(
        'localization', default_value='ground_truth',
        description='ground_truth (static map->odom, sim) | amcl (LiDAR localization)')

    map_server = LifecycleNode(
        package='nav2_map_server', executable='map_server', name='map_server',
        namespace='', output='screen',
        parameters=[params_file, {'use_sim_time': use_sim_time, 'yaml_filename': map_yaml}])

    amcl = LifecycleNode(
        package='nav2_amcl', executable='amcl', name='amcl',
        namespace='', output='screen', condition=is_amcl,
        parameters=[params_file, {
            'use_sim_time':      use_sim_time,
            'set_initial_pose':  True,
            'initial_pose.x':    ParameterValue(LaunchConfiguration('initial_pose_x'),
                                                value_type=float),
            'initial_pose.y':    ParameterValue(LaunchConfiguration('initial_pose_y'),
                                                value_type=float),
            'initial_pose.z':    0.0,
            'initial_pose.yaw':  ParameterValue(LaunchConfiguration('initial_pose_yaw'),
                                                value_type=float),
        }])

    gt_map_to_odom = Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='gt_map_to_odom', output='screen', condition=is_gt,
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        parameters=[{'use_sim_time': use_sim_time}])

    def lnode(p, e, n):
        return LifecycleNode(package=p, executable=e, name=n, namespace='',
                             output='screen',
                             parameters=[params_file, {'use_sim_time': use_sim_time}])

    planner      = lnode('nav2_planner',           'planner_server',    'planner_server')
    controller   = lnode('nav2_controller',        'controller_server', 'controller_server')
    smoother     = lnode('nav2_smoother',          'smoother_server',   'smoother_server')
    behaviors    = lnode('nav2_behaviors',         'behavior_server',   'behavior_server')
    bt_nav       = lnode('nav2_bt_navigator',      'bt_navigator',      'bt_navigator')
    waypoint     = lnode('nav2_waypoint_follower', 'waypoint_follower', 'waypoint_follower')

    nav_nodes = ['planner_server', 'controller_server', 'smoother_server',
                 'behavior_server', 'bt_navigator', 'waypoint_follower']

    wait_for_map_server = Node(
        package='limo_car', executable='wait_for_lifecycle_node',
        name='wait_for_map_server', output='screen',
        arguments=['map_server'] + nav_nodes)

    lifecycle_amcl_node = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_navigation', output='screen', condition=is_amcl,
        parameters=[{'use_sim_time': use_sim_time, 'autostart': True, 'bond_timeout': 0.0,
                     'node_names': ['map_server', 'amcl'] + nav_nodes}])

    lifecycle_gt_node = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_navigation', output='screen', condition=is_gt,
        parameters=[{'use_sim_time': use_sim_time, 'autostart': True, 'bond_timeout': 0.0,
                     'node_names': ['map_server'] + nav_nodes}])

    start_lifecycle_managers = RegisterEventHandler(OnProcessExit(
        target_action=wait_for_map_server,
        on_exit=[lifecycle_amcl_node, lifecycle_gt_node]))

    return LaunchDescription([
        declare_use_sim, declare_drive_mode, declare_localization,
        declare_init_x, declare_init_y, declare_init_yaw,
        map_server,
        amcl, gt_map_to_odom,
        planner, controller, smoother, behaviors, bt_nav, waypoint,
        wait_for_map_server, start_lifecycle_managers,
    ])
