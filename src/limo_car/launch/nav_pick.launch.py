"""
nav_pick.launch.py — ONE COMMAND to run the full autonomous pick-and-place.

Starts (in dependency order):
  1. Gazebo simulation  (ackermann_gazebo.launch.py — sim + robot + controllers)
  2. MoveIt move_group  (moveit.launch.py)
  3. Nav2 stack         (nav2_limo.launch.py — map, AMCL, planner, controller)
  4. box_pose_estimator (perception → /box_pose)
  5. tag_dock_estimator (AprilTag → /place_tag_pose)
  6. nav_pick_orchestrator (the mission node — waits for everything above)

Usage:
    ros2 launch limo_car nav_pick.launch.py
"""

import os
import re

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    EmitEvent,
    DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, SetLaunchConfiguration,
    OpaqueFunction
)
from launch.conditions import IfCondition
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


class SceneGeometryError(RuntimeError):
    """The scene is not physically achievable, or the world file disagrees."""


class LocalizationConfigError(RuntimeError):
    """The localization and odometry settings cannot work together."""


def validate_localization(localization, odometry_source):
    """Refuse configurations that degrade silently instead of failing.

    ENCODER odometry (odometry_source=0) integrates real wheel rotation, so /odom
    drifts exactly as it does on hardware. Ground-truth localization publishes a
    FIXED zero map->odom transform, which is only correct while odom cannot drift.
    Put the two together and the drift accumulates behind a frozen transform with
    nothing to correct it: the robot's believed map pose slowly separates from
    reality, with no error message.

    AMCL matches the laser against the map and corrects map->odom as the drift
    accumulates, so ENCODER is allowed only with AMCL.
    """
    if str(odometry_source) == '0' and localization != 'amcl':
        raise LocalizationConfigError(
            f"odometry_source:=0 (ENCODER) requires localization:=amcl, "
            f"but localization is '{localization}'.\n"
            f"  ENCODER odometry drifts like the real robot. Ground-truth "
            f"localization publishes a FIXED map->odom transform and cannot correct "
            f"that drift, so the robot's believed position would separate from "
            f"reality with nothing to catch it.\n"
            f"  To run with drifting odometry, as on the robot:\n"
            f"    ros2 launch limo_car nav_pick.launch.py "
            f"localization:=amcl odometry_source:=0\n"
            f"  For the default perfect-localization sim, leave both unset.")


def load_scene(pkg):
    """Read config/scene.yaml, derive the docking geometry, and validate it.

    This is the only place docking distances are computed. Everything the
    orchestrator needs is derived here and passed down as ROS parameters.

    Raises SceneGeometryError if the scene cannot physically work.
    """
    with open(os.path.join(pkg, 'config', 'scene.yaml')) as f:
        s = yaml.safe_load(f)

    rb, pt, kt = s['robot'], s['place_table'], s['pickup_table']
    bumper, reach, clear = rb['bumper_x'], rb['arm_reach'], rb['bumper_clearance']
    ground_z = rb['base_link_ground_z']
    side = pt['side']

    # ── Derive ───────────────────────────────────────────────────────────────
    # The dock is anchored to the bumper: stop with dock_gap between the bumper and
    # the table's near face. The arm-reach limits are checked below; they do not set
    # the target.
    box_offset   = kt.get('box_offset', 0.0)
    dock_gap     = rb.get('dock_gap', clear)

    # Pickup: the box sits box_offset in front of the table centre, so it is
    # (depth/2 - box_offset) beyond the near face.
    kt_face      = kt['depth'] / 2.0 - box_offset       # box -> table near face

    dock_range   = bumper + dock_gap                    # place:  base_link -> tag (on the face)
    stop_dist    = bumper + dock_gap + kt_face          # pickup: base_link -> box

    # Place table top in base_link frame (world top_z - base_link ground z), for the arm's z target.
    place_surface_base_z  = pt['top_z'] - ground_z

    # ── Validate ─────────────────────────────────────────────────────────────
    # Both docks are bumper-anchored, so "too close to the table" means "dock_gap
    # below the hard minimum".
    if dock_gap < clear:
        raise SceneGeometryError(
            f"robot.dock_gap = {dock_gap:.3f} m is below the hard minimum "
            f"robot.bumper_clearance = {clear:.3f} m.\n"
            f"  The bumper would be commanded into the table face.")

    # A big enough table (or a small enough dock_gap on a deep table) could put the
    # target past what the arm can reach.
    place_centre = dock_range + side / 2.0        # base_link -> place table centre
    if place_centre > reach:
        raise SceneGeometryError(
            f"the arm cannot reach the place table centre.\n"
            f"  dock_range = bumper_x + dock_gap = {bumper:.3f} + {dock_gap:.3f} = "
            f"{dock_range:.3f} m, table centre a further side/2 = {side / 2.0:.3f} m "
            f"=> {place_centre:.3f} m.\n"
            f"  arm_reach is {reach:.3f} m.\n"
            f"  Reduce robot.dock_gap or place_table.side.")

    if stop_dist > reach:
        raise SceneGeometryError(
            f"the arm cannot reach the box at the pickup dock.\n"
            f"  stop_dist = bumper_x + dock_gap + (depth/2 - box_offset) = "
            f"{bumper:.3f} + {dock_gap:.3f} + {kt_face:.3f} = {stop_dist:.3f} m.\n"
            f"  arm_reach is {reach:.3f} m.\n"
            f"  Reduce robot.dock_gap, pickup_table.depth, or raise "
            f"pickup_table.box_offset.")

    # ── Cross-check against the world file, which Gazebo actually loads ──────
    # scene.yaml cannot drive Gazebo, so the two must be kept in step by hand.
    world = open(os.path.join(pkg, 'worlds', 'final_map.world')).read()
    m = re.search(r"<model name='place_table'>.*?<box><size>([\d.]+) ([\d.]+) ([\d.]+)</size>",
                  world, re.S)
    if m:
        wx, wy, wz = (float(v) for v in m.groups())
        if abs(wx - side) > 1e-6 or abs(wy - side) > 1e-6 or abs(wz - pt['top_z']) > 1e-6:
            raise SceneGeometryError(
                f"scene.yaml and final_map.world disagree about place_table:\n"
                f"  scene.yaml : {side} x {side} x {pt['top_z']}\n"
                f"  world file : {wx} x {wy} x {wz}\n"
                f"  Change BOTH together.")

    sp = rb.get('spawn_pose', {'x': -2.0, 'y': 7.0, 'yaw': -1.5708})

    s['derived'] = {
        'spawn_x':                float(sp['x']),
        'spawn_y':                float(sp['y']),
        'spawn_yaw':              float(sp['yaw']),
        'dock_range':             dock_range,
        'stop_distance':          stop_dist,
        'arm_reach':              reach,
        'bumper_gap':             dock_range - bumper,
        'bumper_x':               float(bumper),
        'place_surface_base_z':   place_surface_base_z,
        'box_width':              float(s['box']['width']),
    }
    print(f"[scene] place table {side*100:.0f} cm square, top {pt['top_z']*100:.0f} cm | "
          f"dock_range {dock_range:.3f} m | bumper gap {dock_range - bumper:.3f} m | "
          f"arm reaches table centre ✓")
    print(f"[scene] place surface at {place_surface_base_z:+.4f} m in base_link")
    return s


def _check_localization_config(context, *args, **kwargs):
    """Resolve the two settings and validate them before anything starts.

    Has to be an OpaqueFunction: LaunchConfiguration values do not exist while the
    launch description is being built, only once the context is populated. Raising
    here aborts the launch cleanly, before Gazebo or Nav2 come up.
    """
    validate_localization(
        LaunchConfiguration('localization').perform(context),
        LaunchConfiguration('odometry_source').perform(context))
    return []


def generate_launch_description():
    # Safety: the laptop and the real robot share the ROS network (domain 0), so a
    # simulation's /cmd_vel could reach the real base driver. Simulation terminals
    # must be isolated: source sim_env.sh first.
    if os.environ.get('ROS_LOCALHOST_ONLY') != '1':
        raise RuntimeError(
            '\n\nSIMULATION BLOCKED: this terminal is not isolated from the real robot '
            '(shared ROS network -- the simulation could drive the real LIMO).\n'
            'Run first:   source tools/sim_env.sh\n')
    pkg      = get_package_share_directory('limo_car')
    moveit   = get_package_share_directory('limo_cobot_moveit_config')
    scene    = load_scene(pkg)
    place    = scene['place_table']
    pickup   = scene['pickup_table']
    derived  = scene['derived']

    # Defaults come from scene.yaml's robot.spawn_pose, so there is one place to
    # change the start pose and the Gazebo spawn, AMCL's seed and everything derived
    # from it move together. Override all three on the command line to test off-spawn
    # behaviour:  ros2 launch limo_car nav_pick.launch.py spawn_y:=5.5
    spawn_x   = LaunchConfiguration('spawn_x',   default=str(derived['spawn_x']))
    spawn_y   = LaunchConfiguration('spawn_y',   default=str(derived['spawn_y']))
    spawn_yaw = LaunchConfiguration('spawn_yaw', default=str(derived['spawn_yaw']))
    # Which localization the whole pipeline runs (passed on to nav2_limo.launch.py).
    localization = LaunchConfiguration('localization', default='ground_truth')
    use_rviz = LaunchConfiguration('use_rviz', default='true')
    use_gzclient = LaunchConfiguration('use_gzclient', default='true')
    drive_mode = LaunchConfiguration('drive_mode', default='diff')
    place_stack_levels = LaunchConfiguration('place_stack_levels', default='1')
    # Derived in load_scene(). Overridable on the command line for experiments.
    dock_range = LaunchConfiguration('dock_range', default=str(derived['dock_range']))
    metrics_csv = LaunchConfiguration('metrics_csv', default='')
    # Which box to grasp when several are visible: 'rightmost' (deterministic,
    # for multi-box stacking) or 'nearest'.
    target_policy = LaunchConfiguration('target_policy', default='rightmost')

    # ── Argument declarations ─────────────────────────────────────────────────
    spawn_x_arg = DeclareLaunchArgument(
        'spawn_x', default_value=str(derived['spawn_x']),
        description='Robot spawn X (map frame). Default from scene.yaml robot.spawn_pose.')
    spawn_yaw_arg = DeclareLaunchArgument(
        'spawn_yaw', default_value=str(derived['spawn_yaw']),
        description='Robot spawn yaw, rad (map frame). Default from scene.yaml.')
    odometry_source_arg = DeclareLaunchArgument(
        'odometry_source', default_value='1',
        description='Wheel odometry: 1 = WORLD (exact, the sim default), '
                    '0 = ENCODER (drifts like the real robot). ENCODER requires '
                    'localization:=amcl and the launch refuses the other combination.')
    localization_arg = DeclareLaunchArgument(
        'localization', default_value='ground_truth',
        description='ground_truth = static identity map->odom (perfect, sim only). '
                    'amcl = real LiDAR localization against the map. Use amcl with '
                    'odometry_source:=0 to run with drifting odometry, as on the robot.')

    spawn_y_arg = DeclareLaunchArgument(
        'spawn_y', default_value='7.0',
        description='Robot spawn Y. Default 7.0 → 3 m from the table (nav problem).')

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='true',
        description='Whether to launch RViz')

    use_gzclient_arg = DeclareLaunchArgument(
        'use_gzclient', default_value='true',
        description='Whether to launch Gazebo client (GUI)')

    drive_mode_arg = DeclareLaunchArgument(
        'drive_mode', default_value='diff',
        description='Drive mode (ackermann | diff)')

    target_policy_arg = DeclareLaunchArgument(
        'target_policy', default_value='rightmost',
        description="Which box to grasp when several are visible: 'rightmost' "
                    "(deterministic, for multi-box stacking) or 'nearest'.")

    place_stack_levels_arg = DeclareLaunchArgument(
        'place_stack_levels', default_value='1',
        description='Number of boxes to pick+stack at the place table in one run '
                    '(the world has 3 distinct stack_box_N boxes on the pickup table).')

    dock_range_arg = DeclareLaunchArgument(
        'dock_range', default_value=str(derived['dock_range']),
        description=f"AprilTag dock stop distance (m). Derived from config/scene.yaml "
                    f"as bumper_x + dock_gap = {scene['robot']['bumper_x']:.3f} + "
                    f"{scene['robot']['dock_gap']:.3f} = {derived['dock_range']:.3f}. "
                    f"Bumper gap "
                    f"{derived['bumper_gap']*1000:.0f} mm. To change how close the robot "
                    f"parks, edit robot.dock_gap in scene.yaml -- it is validated there "
                    f"against both bumper_clearance and arm_reach. Do not hardcode: a "
                    f"value below {scene['robot']['bumper_x']:.3f} m is physically "
                    f"unreachable and the robot will grind its wheels against the table.")

    metrics_csv_arg = DeclareLaunchArgument(
        'metrics_csv', default_value='',
        description='Path for the per-cycle metrics CSV. Empty = auto-name under data/.')

    # Keep the top-level use_rviz before the gazebo include overwrites it.
    # ackermann_gazebo.launch.py has its own 'use_rviz' argument, and the 'false'
    # passed to it below overwrites the shared launch-context value; MoveIt reads
    # it 8 s later, so it must read this saved copy.
    preserve_use_rviz = SetLaunchConfiguration('moveit_use_rviz', use_rviz)

    # ── 1. Gazebo + robot + controllers ──────────────────────────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, 'launch', 'ackermann_gazebo.launch.py')),
        launch_arguments={
            'spawn_x': spawn_x,
            'spawn_y': spawn_y,
            'spawn_yaw': spawn_yaw,
            'odometry_source': LaunchConfiguration('odometry_source'),
            'use_rviz': 'false',
            'use_gzclient': use_gzclient,
            'drive_mode': drive_mode,
        }.items())


    # ── 2. MoveIt (move_group) — delay 8 s so controllers come up first ──────
    moveit_launch = TimerAction(
        period=8.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(moveit, 'launch', 'moveit.launch.py')),
            launch_arguments={
                'use_sim_time': 'true',
                'use_rviz': LaunchConfiguration('moveit_use_rviz'),
                'rviz_config': os.path.join(pkg, 'rviz', 'nav_monitor.rviz'),
            }.items())])

    # ── 3. Nav2 — delay 12 s so the robot is spawned + controllers ready ─────
    nav2_launch = TimerAction(
        period=12.0,
        actions=[IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg, 'launch', 'nav2_limo.launch.py')),
            launch_arguments={
                'use_sim_time': 'true',
                'drive_mode': drive_mode,
                'localization': localization,
                # AMCL's seed is the same pose Gazebo spawns the robot at.
                'initial_pose_x': spawn_x,
                'initial_pose_y': spawn_y,
                'initial_pose_yaw': spawn_yaw,
            }.items())])

    # ── 4. box_pose_estimator — delay 15 s ───────────────────────────────────
    box_estimator = TimerAction(
        period=15.0,
        actions=[Node(
            package='limo_car',
            executable='box_pose_estimator',
            name='box_pose_estimator',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                # 'rightmost': three identical boxes stand at the same distance, so
                # 'nearest' would flip between them on perception noise. 'rightmost'
                # orders them by base_link y; each pick removes a box, so the next
                # cycle has a new rightmost.
                'target_policy': target_policy,
                # Explicit sim search window, independent of the node's defaults
                # (which are set for the real lab). A wider window lets the
                # estimator pick up blobs that are not boxes.
                'search_x_max':   1.60,
                'search_y_abs':   0.60,
                'search_z_min':  -0.10,
                'search_z_max':   0.20,
                'min_area_fraction': 0.15,
                # The real Orbbec DaBai returns no depth closer than ~0.25-0.30 m
                # (measured; spec 0.30). The simulated camera renders down to 0.15 m
                # (sensor.xacro <near>), so the estimator ignores depth below 0.30 m and
                # the simulation has the same blind zone as the robot.
                'depth_min': ParameterValue(LaunchConfiguration('camera_depth_min', default='0.30'),
                                            value_type=float),
            }])])

    # ── 4b. tag_dock_estimator — delay 15 s (parallel with box_estimator) ────
    # Detects the AprilTag 36h11 on the place table and publishes /place_tag_pose.
    # Uses the same camera topics as box_pose_estimator.
    tag_estimator = TimerAction(
        period=15.0,
        actions=[Node(
            package='limo_car',
            executable='tag_dock_estimator',
            name='tag_dock_estimator',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'tag_size':    place['tag_size'],    # from config/scene.yaml
            }])])

    # ── 6. nav_pick_orchestrator — delay 25 s (Nav2 fully active) ────────────
    orchestrator = TimerAction(
        period=25.0,
        actions=[Node(
            package='limo_car',
            executable='nav_pick_orchestrator',
            # exit_when_done:=true (batches): end the whole launch when the task is finished,
            # instead of idling until the batch's wall-clock timeout.
            on_exit=[EmitEvent(event=Shutdown(reason='orchestrator finished'),
                               condition=IfCondition(LaunchConfiguration('exit_when_done',
                                                                         default='false')))],
            name='nav_pick_orchestrator',
            output='screen',
            parameters=[{'use_sim_time': True,
                          'place_stack_levels': ParameterValue(place_stack_levels, value_type=int),
                          'dock_range': ParameterValue(dock_range, value_type=float),
                          'metrics_csv': metrics_csv,
                          # ── scene geometry, all from config/scene.yaml ──────
                          'table_map_x':      float(place['centre_map'][0]),
                          'table_map_y':      float(place['centre_map'][1]),
                          'nav_place_x':      float(place['nav_approach'][0]),
                          'nav_place_y':      float(place['nav_approach'][1]),
                          'nav_place_yaw':    float(place['nav_approach'][2]),
                          'nav_goal_x':       float(pickup['nav_approach'][0]),
                          'nav_goal_y':       float(pickup['nav_approach'][1]),
                          'nav_goal_yaw':     float(pickup['nav_approach'][2]),
                          'stop_distance':    float(derived['stop_distance']),
                          # place_box() clamps the drop point against this.
                          'arm_reach':        float(derived['arm_reach']),
                          # Place table surface in base_link (see load_scene). Nothing
                          # measures the place table's height before release, so this
                          # value is the place height.
                          'place_surface_base_z':  float(derived['place_surface_base_z']),
                          # World-frame top_z, for comparing against Gazebo ground
                          # truth in the metrics harness (world frame, not base_link).
                          'place_table_top_z_world': float(place['top_z']),
                          'place_table_side': float(place['side']),
                          'box_width': float(derived['box_width']),
                          'bumper_x': float(derived['bumper_x'])}])])

    return LaunchDescription([
        spawn_x_arg,
        spawn_y_arg,
        spawn_yaw_arg,
        localization_arg,
        odometry_source_arg,
        # Validate the pair BEFORE any node starts.
        OpaqueFunction(function=_check_localization_config),
        use_rviz_arg,
        use_gzclient_arg,
        drive_mode_arg,
        target_policy_arg,
        place_stack_levels_arg,
        dock_range_arg,
        metrics_csv_arg,
        DeclareLaunchArgument('exit_when_done', default_value='false',
                              description='Shut the whole launch down when the orchestrator exits.'),
        DeclareLaunchArgument('camera_depth_min', default_value='0.30',
                              description='Estimator depth floor (m); 0.30 = real Orbbec DaBai blind zone.'),
        preserve_use_rviz,
        gazebo,
        moveit_launch,
        nav2_launch,
        box_estimator,
        tag_estimator,
        orchestrator,
    ])
