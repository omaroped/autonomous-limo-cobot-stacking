#!/usr/bin/python3
"""
nav_pick_orchestrator.py — the simulated stacking mission.

For each of the place_stack_levels cubes:
  1. Drive to table A (Nav2 to a station, then camera-guided approach).
  2. Pick the cube top-down with MoveIt (straight vertical approach, close, lift,
     check with Gazebo ground truth that the cube left the table).
  3. Drive to table B and dock on its AprilTag (lidar for the final distance).
  4. Place the cube on the stack (straight descent, release, retract).
  5. Back away; the next cycle starts.

Started by nav_pick.launch.py, which also starts Gazebo, Nav2 and MoveIt.
Each run writes an event log, a grasp-attempt CSV and a metrics CSV.
"""

import csv
import faulthandler
import json
import math
import os
import signal
import statistics
import threading
import time

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import SingleThreadedExecutor
from rclpy.action import ActionClient
from rclpy.duration import Duration

from geometry_msgs.msg import (PoseStamped, Pose, Point, PointStamped,
                                Twist)
from std_msgs.msg import Float64MultiArray, Float64
from action_msgs.msg import GoalStatus
from nav_msgs.msg import Odometry

from nav2_msgs.action import NavigateToPose

from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.srv import ApplyPlanningScene, GetPositionIK, GetPositionFK
from moveit_msgs.msg import (
    MotionPlanRequest, Constraints, JointConstraint,
    PositionConstraint, OrientationConstraint, BoundingVolume,
    CollisionObject, PlanningScene, RobotState, RobotTrajectory,
)
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectoryPoint

from gazebo_msgs.srv import GetEntityState

from lifecycle_msgs.srv import ChangeState, GetState
from lifecycle_msgs.msg import Transition

from sensor_msgs.msg import JointState, LaserScan
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point


# ── Arm interface ─────────────────────────────────────────────────────────────
# run() talks to the arm only through this interface. Every call returns one
# ArmResult. SimArmBackend wraps the MoveIt methods of the node.
from arm_interface import ArmResult


class SimArmBackend:
    """Simulation backend: the node's MoveIt methods behind the arm interface."""
    name = 'sim'

    def __init__(self, node):
        self.n = node

    def pick(self, x, y, z):
        """Top-down pick of the cube centre (x, y, z) in base_link."""
        self.n.grasp_and_retract(x, y, z)
        held = self.n._weld_active      # set by the verified lift (Gazebo ground truth)
        return ArmResult(ArmResult.SUCCEEDED if held else ArmResult.FAILED,
                         reason='' if held else 'grasp not verified',
                         held=ArmResult.HOLDING if held else ArmResult.EMPTY,
                         evidence='gazebo lift check')

    def place(self, level):
        """Place the held cube on stack level `level` at the latched drop point.
        SUCCEEDED means the motion completed; where the cube landed is a separate check."""
        ok = self.n.place_box(level=level)
        return ArmResult(ArmResult.SUCCEEDED if ok else ArmResult.FAILED,
                         reason='' if ok else 'place motion failed')

    def prepare_travel(self):
        """Fold the arm to the travel posture before the base moves."""
        ok = self.n._go_named_safe('travel')
        return ArmResult(ArmResult.SUCCEEDED if ok else ArmResult.FAILED,
                         reason='' if ok else 'travel posture not reached after retry',
                         arm_clear=bool(ok))


# ── Navigation goal ───────────────────────────────────────────────────────────
# Fallback only. The live values come from config/scene.yaml through
# nav_pick.launch.py (self._nav_goal_x etc.); tune scene.yaml, not these.
NAV_GOAL_X   = -2.0
NAV_GOAL_Y   =  4.75    # outside the pickup table's inflated edge (4.10 + 0.45)
NAV_GOAL_YAW = -1.5708  # −π/2 → robot faces −Y (toward the table)

# The pickup-to-place leg (Nav2 drive + tag dock) took about 100-120 s, so 120 s
# was too tight; 240 s leaves room.
NAV_TIMEOUT_SEC = 240.0

# ── Backup ────────────────────────────────────────────────────────────────────
BACKUP_VEL_X = -0.15   # m/s reverse
BACKUP_SEC   =  2.5    # back up ~0.38 m

# ── Gripper ───────────────────────────────────────────────────────────────────
# 6-element commands with the mirror signs [1,1,-1,-1,-1,1]; the order must match
# the joints list in config/mycobot_controllers.yaml. (Single-element mimic commands
# made gazebo_ros2_control rename the followers "<joint>_mimic", which broke their TF.)
GRIPPER_OPEN  = [ 0.15,  0.15, -0.15, -0.15, -0.15,  0.15]
# Closed: -0.45 gives a finger gap a little smaller than the 24.7 mm cube (see
# grasp_close_floor). Joint limit -0.7.
GRIPPER_GRASP = [-0.45, -0.45,  0.45,  0.45,  0.45, -0.45]
GRIPPER_TOPIC = '/mycobot_gripper_controller/commands'

# ── How the box is held ──────────────────────────────────────────────────────
# libgazebo_grasp_plugin (gazebo/mycobot_ros2_control.xacro) creates an ODE fixed
# joint between gripper_base and the box once enough finger contacts accumulate, and
# removes it when the gripper opens past release_position (0.0). This node does not
# command the grasp. `_weld_active` is an observation: ground truth says the box came
# off the table.

# ── Gripper close ────────────────────────────────────────────────────────────
# The close steps the command down to grasp_close_floor. The simulated fingers are
# position controlled and never stall on the cube, so the close always runs to the
# floor; whether the cube is held is decided after the lift (_verify_grasp_truth).
CLOSE_STEP_RAD    = 0.005    # rad per step
CLOSE_STEP_SEC    = 0.08     # s per step; at 0.04 the finger sank into the cube instead of pushing it
MAX_GRASP_RETRIES = 2        # retries after a failed lift before the pick is aborted

# ── MoveIt frames / group ─────────────────────────────────────────────────────
PLANNING_FRAME = 'base_link'
TCP_LINK       = 'gripper_tcp'
ARM_GROUP      = 'arm'
ARM_JOINTS = ['joint2_to_joint1', 'joint3_to_joint2', 'joint4_to_joint3',
              'joint5_to_joint4', 'joint6_to_joint5', 'joint6output_to_joint6']
NAMED_STATES = {
    'home':   [-1.5708,  0.0,  0.0,  0.0,  0.0,  0.0],
    'ready':  [-1.5708, -0.5, -0.6,  1.1,  0.0, -1.5708],
    'travel': [-1.5708,  0.4, -0.5, -0.5,  0.0,  0.0],
}

# ── Grasp geometry ───────────────────────────────────────────────────────────
TOPDOWN_QUAT = (-0.7071, 0.0, 0.0, 0.7071)
# Correction added to the perceived cube centre (grasp_off_x/y/z parameters). Zero:
# calibrate it from the [perception check] log, not from the drawing.
GRASP_OFFSET = (0.0, 0.0, 0.0)  # perception → grasp target offset
FIXED_BOX    = (0.28, 0.015, 0.08)     # fallback if perception unavailable

# ── Straight approach from our own small-step solver ─────────────────────────────
# Near the reach limit KDL ignores its seed, so solving the points of a straight line
# one by one can return a different arm posture for the next point, which the arm can
# only reach by swinging (a reconfiguration). plan_straight_approach() therefore builds
# the line backwards: take a grasp posture, raise the gripper APPROACH_STEP at a time,
# and solve each step as the smallest joint change from the previous posture (damped
# least squares on a numerical Jacobian from /compute_fk). No posture switch can happen
# inside the line. The top of the chain is the pre-grasp pose; the line is replayed down.
ARM_LIMITS = [(-2.9321, 2.9321), (-2.4434, 2.4434), (-2.6179, 2.6179),
              (-2.6179, 2.6179), (-2.7052, 2.7925), (-3.14159, 3.14159)]  # mycobot_280_m5.urdf
PREGRASP_RISE  = 0.06    # pre-grasp height above the grasp point -- same as the real robot
APPROACH_STEP  = 0.005   # metres between two points of the straight line
STEP_MAX_JOINT = 0.15    # rad; a bigger single-joint change between two points = a posture switch
PATH_JOINT_SPEED = 0.3   # rad/s; _run_joint_path's cap = STEP_MAX_JOINT per 0.5 s point
# Fingers close left-right, as on the real robot (front-back closing made the back finger
# pass over the cube's back edge). +90 and -90 deg are the 180-degree twins of one grasp;
# the wrist sits at a different angle in each, so one may have a straight line where the
# other runs into a joint limit.
GRASP_YAWS = (math.pi / 2, -math.pi / 2)
GRASP_IK_CANDIDATES = 4  # grasp postures sampled per yaw (KDL restarts randomly)

# ── Is the grip real? ──────────────────────────────────────────────────────────────
# The grasp plugin can attach a box that one fingertip only touched from the side. A real
# two-finger grip centres the box between the jaws, so a large jaw offset means the
# fingers closed beside the box, and a real gripper would not hold it.
FINGER_FACE_HALF = 0.018  # finger face is 36 mm wide (gripper.xacro collision box)
GRASP_JAW_TOL    = 0.010  # box centre within 10 mm of the jaw centre


def _qmul(a, b):
    """Hamilton product of two (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def _qmat(q):
    """Rotation matrix of a geometry_msgs quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _spin_about_vertical(q, yaw):
    """Rotate orientation q about the PLANNING frame's vertical axis by yaw.

    Pre-multiplying by a world-z rotation spins the gripper about the vertical whatever
    its own axes are. (_topdown_yaw_quat post-multiplies, i.e. rotates about the
    GRIPPER's z axis, which for TOPDOWN_QUAT points sideways.)"""
    return _qmul((0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)), q)


def _json_default(v):
    """Event-log encoder: numpy values as numbers, anything else by its repr."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return repr(v)


def _rotvec(q):
    """Rotation vector (axis * angle) of an (x, y, z, w) quaternion, shortest way round."""
    x, y, z, w = q
    if w < 0.0:
        x, y, z, w = -x, -y, -z, -w
    s = math.sqrt(x * x + y * y + z * z)
    if s < 1e-12:
        return np.zeros(3)
    return np.array([x, y, z]) * (2.0 * math.atan2(s, w) / s)
# TCP height above the cube centre at the grasp. The finger tips hang below the TCP, so
# grasping at the centre made the closing fingers push the table (the base slid).
GRASP_ABOVE  = 0.0105
# The finger tips hang ~20 mm below the grasp point, below the cube's bottom, so the
# descent stops this far above the surface and the cube drops the last few mm. Same as
# the real robot (PLACE_GAP in moveit_arm.py), on the table and on the stack.
PLACE_RELEASE_GAP = 0.003

# Table collision in planning scene (add it so MoveIt routes around it)
TABLE_SIZE     = (0.10, 0.32, 0.10)   # metres (world: 0.30×0.08×0.10; pad Y/X slightly)
BOX_HALF_H     = 0.01235               # half of the 24.7 mm cube (same size as the real cube)

# ── Stacking geometry (vertical box-stacking — thesis target task) ────────────
BOX_HEIGHT       = 2 * BOX_HALF_H      # full box height = one stack level
REALIGN_OVERSHOOT = 0.02     # pick-side sideways slide goes this much past the box
BOX_FRESH_AGE = 0.3          # s: a /box_pose older than this (per /box_detection_age) is a replay
STACK_CHECK_XY = 0.020       # final stack check: box within this of the table centre (xy)
STACK_CHECK_Z = 0.010        # ... and of its level height (z)
STACK_RELEASE_MARGIN = 0.0   # levels >= 1: extra release height (none, as the real robot)
STACK_HOVER_TRIES = (0.06, 0.045, 0.03)   # levels >= 1: straight-descent start heights to try
STACK_HOVER_UPPER = 0.06   # levels >= 1: straight descent from 6 cm above (see place_box)
STACK_HOVER      =  0.10               # TCP hover above the current stack top before placing
# Fallback only; the live value is self._stop_distance, from config/scene.yaml.
STOP_DISTANCE    =  0.22               # box distance from base_link at the pickup dock

# ── Place table — fiducial-guided docking ─────────────────────────────────────
# Nav2 pre-dock goal: on the table's north-face normal line, close enough to see the
# AprilTag and clear of the table's inflation zone (closer goals made Nav2 report
# "Failed to make progress"). The tag dock covers the rest.
NAV_PLACE_X    = -4.0        # on the table-centre's north normal line
NAV_PLACE_Y    = -0.50
NAV_PLACE_YAW  = -1.5708    # -π/2: robot faces -Y (toward the table face)

# Closed-loop dock parameters (dock_to_tag)
#
# All of these are fallbacks. The live values are derived in
# nav_pick.launch.py:load_scene() from config/scene.yaml and passed as ROS
# parameters (dock_range, table_map_x/y), so the world, the launch file and the
# estimators all use one table definition. Tune config/scene.yaml, not these.
PLACE_DOCK_RANGE   = 0.20   # fallback: stop when tag is this far from base_link (m)
TABLE_MAP_X        = -4.0   # fallback: place table centre, map X
TABLE_MAP_Y        = -2.0   # fallback: place table centre, map Y

# ── Sideways step (_crab_sideways): pivot 90 deg, drive, pivot back ─────────
PLACE_DOCK_CRAB_ROT = 0.8    # rad/s for the 90 deg pivots
PLACE_DOCK_CRAB_FWD = 0.10   # m/s for the sideways leg — slow, it is open-loop on odom
# The dock stops this far short of the target, measures standing still, and finishes on
# odometry. The camera loses the tag about 0.4 m from the table, so near the table
# /place_tag_pose is the last sighting replayed through localisation (up to ~32 mm long).
# The final hop and the drop x therefore come from the lidar gap read at this stop; the tag
# replay is only the fallback. The lidar cannot see closer than 0.10 m, so this keeps the
# table face at least ~0.12 m from the lidar at the read.
PLACE_DOCK_FINAL_HOP  = 0.07

# ── Match the real robot ─────────────────────────────────────────────────────────
# The real robot runs face_and_pick.py (src/ros2_ws/real_robot/robot_home/). The
# simulation follows the same task logic, so the remaining sim-vs-real differences
# come from the hardware. Each constant names its face_and_pick.py counterpart.
MATCH_A_FORWARD     = 0.25   # --a-forward: drive toward table A after the station
MATCH_FINAL_READ    = 0.45   # STAGE2_RANGE: last cube reading here, then blind to the dock
MATCH_SECOND_LOOK   = 0.08   # second look only if the far reading is > 0.45 + this
# centre_then_square(): centre first, then square once, then only extra shifts.
MATCH_PICK_Y_TOL    = 0.045  # Y_TOL: the pick accepts |y| below this after centring
MATCH_LATERAL_MAX   = 0.075  # trial(): a far or near look beyond this stops the pick
MATCH_SHIFT_MIN     = 0.030  # SHIFT_MIN: smaller offsets are left to the arm
MATCH_SQUARE_TOL    = math.radians(2.0)   # SQUARE_TOL: no squaring turn below this
MATCH_EXTRA_SHIFTS  = 5      # MAX_EXTRA_SHIFTS
MATCH_STATION_TOL   = math.radians(2.0)   # nav_to(): turn to the station heading if > 2 deg off
MATCH_TURN_RADIUS   = 0.245  # TURN_RADIUS: nothing inside this circle before turning on the spot
MATCH_TURN_FOOT_X, MATCH_TURN_FOOT_Y = 0.150, 0.12   # TURN_FOOT_X/Y (chassis footprint)
MATCH_BACK_STEP, MATCH_BACK_MAX = 0.05, 0.35          # turn_clear(): reverse 5 cm steps, max 0.35 m
# tag_dock(): centre on the tag and square, then place on the tag centre y if it lies in
# this band (TAG_Y_MIN/MAX).
MATCH_TAG_Y_MIN, MATCH_TAG_Y_MAX = -0.060, 0.040
MATCH_DOCK_X = 0.65          # DOCK_X: first drive in until the tag is this far ahead

# How far above the computed touch height the descent switches from one fast
# trajectory to single steps. Contact can only happen inside this band; stepping
# the whole 10 cm cost ~25 s per place.
DESCEND_SLOW_BAND = 0.015

# ── Planning ──────────────────────────────────────────────────────────────────
PLAN_ATTEMPTS = 10
PLAN_TIME_SEC =  5.0
VEL_SCALE     =  0.2
ACC_SCALE     =  0.2
# A MoveIt goal that has not finished after this long is cancelled, so no old
# trajectory keeps running while the next command starts.
EXEC_TIMEOUT_SEC = 90.0
MOVEIT_SUCCESS = 1


class NavPickOrchestrator(Node):

    def __init__(self):
        super().__init__('nav_pick_orchestrator')

        # Nav2
        self._nav = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # MoveIt
        self._move  = ActionClient(self, MoveGroup, '/move_action')
        self._fk_cli = self.create_client(GetPositionFK, '/compute_fk')
        self._ik    = self.create_client(GetPositionIK, '/compute_ik')
        # Runs our own precomputed joint paths (_run_joint_path).
        self._exec  = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self._scene = self.create_client(ApplyPlanningScene, '/apply_planning_scene')

        # Gripper
        self._gripper      = self.create_publisher(Float64MultiArray, GRIPPER_TOPIC, 10)
        self._last_gripper   = [0.0] * len(GRIPPER_OPEN)   # last command sent
        self._weld_active    = False
        self._gripper_actual = GRIPPER_OPEN[0]   # actual position from /joint_states
        self._arm_actual = None                  # live ARM joint positions (IK seed)
        # Every subscription that only caches a latest value lives on a side node spun in
        # its own thread. On the main node they were served one callback per spin_once,
        # slower than /tf, /joint_states and /odom arrive, so the cached values fell
        # seconds behind inside the drive loops (a carried box was reported as lost).
        self._io = rclpy.create_node(
            'nav_pick_orchestrator_io',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', value=self.get_parameter('use_sim_time').value)])
        self._io_exec = SingleThreadedExecutor()
        self._io_exec.add_node(self._io)
        threading.Thread(target=self._io_exec.spin, daemon=True).start()
        self._io.create_subscription(JointState, '/joint_states',
                                 self._gripper_joint_cb, 10)
        self._scan = None
        self._io.create_subscription(LaserScan, '/scan', self._scan_cb, qos_profile_sensor_data)

        # Perception
        self._latest_box = None
        self._io.create_subscription(PoseStamped, '/box_pose', self._box_cb, 10)
        # Age (s) since box_pose_estimator's last real detection. /box_pose keeps
        # replaying the last known position when nothing is visible, so this is the
        # only way to tell a live sighting from a stale one.
        self._box_age = float('inf')
        self._io.create_subscription(Float64, '/box_detection_age', self._box_age_cb, 10)

        # Odometry (for the blind final approach: the camera cannot see the box
        # closer than 0.30 m, so the last part is a measured drive)
        self._odom = None
        self._odom_yaw = None
        self._io.create_subscription(Odometry, '/odom', self._odom_cb, 10)
        # Box (x, y, z) in base_link recorded at the dock: x from the odometry-measured
        # drive, y/z from the far perception. The grasp aims at this.
        self._dock_box = None

        # Name of the box currently in the gripper — set by _claim_held_box(), which
        # also removes it from _remaining_boxes. Keeps ground-truth queries pointed at
        # the carried box. See _candidate_box_names().
        self._held_box_name = None
        self._target_box_name = None   # set by _verify_grasp_truth once a box is seen to rise

        # ── Calibration knobs (live, `ros2 param set` works without a relaunch) ──
        self.declare_parameter('grasp_off_x', GRASP_OFFSET[0])
        self.declare_parameter('grasp_off_y', GRASP_OFFSET[1])
        self.declare_parameter('grasp_off_z', GRASP_OFFSET[2])
        self.declare_parameter('box_name', 'stack_box_0')

        # Gripper command at the end of the close (rad). -0.45 leaves a finger gap a
        # little smaller than the 24.7 mm cube; at -0.42 the far finger did not touch,
        # at -0.46 the fingers sank into the cube.
        self.declare_parameter('grasp_close_floor', -0.45)

        self._match_tag_y = None
        self.declare_parameter('box_width', 0.0247)   # set from scene.yaml by the launch
        self.declare_parameter('bumper_x', 0.158)    # chassis front ahead of base_link (scene.yaml)
        self.declare_parameter('place_table_side', 0.08)   # overridden from scene.yaml by the launch

        # ── Scene geometry, supplied by nav_pick.launch.py from config/scene.yaml ──
        # The module constants are fallbacks for running this node bare.
        self.declare_parameter('dock_range',      PLACE_DOCK_RANGE)    # tag distance at stop
        self.declare_parameter('table_map_x',        TABLE_MAP_X)
        self.declare_parameter('table_map_y',        TABLE_MAP_Y)
        self.declare_parameter('nav_place_x',        NAV_PLACE_X)
        self.declare_parameter('nav_place_y',        NAV_PLACE_Y)
        self.declare_parameter('nav_place_yaw',      NAV_PLACE_YAW)
        self.declare_parameter('nav_goal_x',         NAV_GOAL_X)
        self.declare_parameter('nav_goal_y',         NAV_GOAL_Y)
        self.declare_parameter('nav_goal_yaw',       NAV_GOAL_YAW)
        self.declare_parameter('stop_distance',      STOP_DISTANCE)
        self.declare_parameter('arm_reach',          0.24)   # config/scene.yaml robot.arm_reach
        self._arm = SimArmBackend(self)
        # Place-table surface height in base_link, derived in nav_pick.launch.py from
        # config/scene.yaml (world top_z - robot.base_link_ground_z). The place table is
        # not perceived at place time, so place_box() computes the release height from
        # this value.
        self.declare_parameter('place_surface_base_z',  -0.045)   # fallback: 0.10-0.145
        self.declare_parameter('place_table_top_z_world', 0.10)   # world frame, for metrics ground truth

        g = lambda n: float(self.get_parameter(n).value)
        self._table_map_x       = g('table_map_x')
        self._table_map_y       = g('table_map_y')
        self._nav_place_x       = g('nav_place_x')
        self._nav_place_y       = g('nav_place_y')
        self._nav_place_yaw     = g('nav_place_yaw')
        self._nav_goal_x        = g('nav_goal_x')
        self._nav_goal_y        = g('nav_goal_y')
        self._nav_goal_yaw      = g('nav_goal_yaw')
        self._stop_distance     = g('stop_distance')
        self._arm_reach         = g('arm_reach')
        self._place_surface_base_z = g('place_surface_base_z')
        self._place_table_top_z_world = g('place_table_top_z_world')
        self.get_logger().info(
            f'[scene] place table centre map=({self._table_map_x:.2f},'
            f'{self._table_map_y:.2f}) dock_range={g("dock_range"):.3f} '
            f'stop_distance={self._stop_distance:.3f} '
            f'place_surface_base_z={self._place_surface_base_z:+.4f}')
        self.declare_parameter('place_stack_levels', 1)                 # boxes to stack at place table
        # Cube 1 drop point: this far toward the robot from the table centre, at least 0.22 m.
        self.declare_parameter('place_near_edge_bias', 0.03)

        # Tag-based place dock (tag_dock_estimator topics)
        self._latest_tag_pose  = None   # /place_tag_pose  PoseStamped in base_link
        self._latched_drop     = None   # (x, y) arm target, set when the place dock ends
        self._docked_via_tag   = False  # which dock branch actually ran (see navigate_to_place_table)

        # TF buffer (map ↔ base_link lookups)
        self._tf_buf      = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self._io)
        self._io.create_subscription(PoseStamped,  '/place_tag_pose',   self._tag_pose_cb,   10)

        # cmd_vel (for backup)
        self._cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Gazebo services (ground truth)
        self._get_state = self.create_client(GetEntityState, '/get_entity_state')

        # ── Metrics ──────────────────────────────────────────────────────────────
        self._metrics: list[dict] = []   # one dict per cycle, flushed to CSV at end
        self._cycle_start: float  = 0.0
        self.declare_parameter('metrics_csv', '')   # empty = auto-name under data/

        # Event log, only when THESIS_EVENT_LOG is set (batch runs). One JSON line per
        # event; the evaluator joins it with the ground-truth recording by sim time.
        self._events = None
        self._run_end_done = False
        ev_path = os.environ.get('THESIS_EVENT_LOG', '')
        if ev_path:
            self._events = open(ev_path, 'a', buffering=1)
            # Thread stacks every 60 s and on SIGUSR1, to locate a freeze.
            self._stacks = open(ev_path + '.stacks.txt', 'w', buffering=1)
            faulthandler.register(signal.SIGUSR1, file=self._stacks, all_threads=True)
            faulthandler.dump_traceback_later(60, repeat=True, file=self._stacks)
            for name, stage in [('navigate_to_table', 'nav_pick'), ('visual_docking', 'pick_dock'),
                                ('_square_to_table', 'square'), ('_crab_sideways', 'sideways'),
                                ('get_box_xyz', 'perception'), ('grasp_and_retract', 'grasp'),
                                ('close_gripper', 'close'), ('_verify_grasp_truth', 'lift_verify'),
                                ('_check_transport_retained', 'carry'),
                                ('navigate_to_place_table', 'place_dock'), ('place_box', 'place')]:
                self._wrap_stage(name, stage)

    def _event(self, name, **fields):
        if self._events is None:
            return
        cyc = getattr(self, '_current_cycle', None)
        rec = {'event': name, 'sim_t': round(self.get_clock().now().nanoseconds * 1e-9, 4),
               'wall_t': round(time.time(), 4), 'cycle': cyc.get('cycle') if cyc else None}
        rec.update(fields)
        self._events.write(json.dumps(rec, default=_json_default) + '\n')

    def _wrap_stage(self, name, stage):
        fn = getattr(self, name)

        def wrapped(*args, **kwargs):
            self._event('stage_start', stage=stage, args=list(args), kwargs=kwargs)
            result = fn(*args, **kwargs)
            self._event('stage_done', stage=stage, result=result)
            return result
        setattr(self, name, wrapped)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _box_cb(self, msg):
        self._latest_box = msg

    def _box_age_cb(self, msg):
        self._box_age = msg.data

    def _tag_pose_cb(self, msg: PoseStamped):
        self._latest_tag_pose = msg

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        self._odom = (p.x, p.y)
        # Heading too: the sideways manoeuvre turns by a commanded angle, closed on
        # odometry.
        q = msg.pose.pose.orientation
        self._odom_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                    1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _odom_dist_since(self, x0, y0):
        """Straight-line distance the base has driven since (x0, y0) in odom."""
        if self._odom is None:
            return None
        return math.hypot(self._odom[0] - x0, self._odom[1] - y0)

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _cycle_begin(self, cycle_idx: int):
        """Mark the start of cycle `cycle_idx` (0-indexed)."""
        self._cycle_start = time.time()
        self._current_cycle = {
            'cycle':              cycle_idx,
            'box_name':           None,
            'pick_success':       False,
            'transport_retained': False,
            'dock_via_tag':       False,
            'place_success':      False,
            'placement_err_x':    float('nan'),
            'placement_err_y':    float('nan'),
            'placement_err_z':    float('nan'),
            'cycle_time_s':       float('nan'),
            'stack_levels_final': float('nan'),   # set on the last row by _final_stack_check
        }

    def _final_stack_check(self, settle=2.0):
        """After the final retreat: how many levels of the stack are really standing?

        The per-cycle place metric only looks at the box just placed, so a later place
        that knocks a lower box off would go unseen. This checks every box from ground
        truth: level k is in place if some box sits at the table centre (xy within
        STACK_CHECK_XY) at the level-k height (z within STACK_CHECK_Z). Counts
        consecutive levels from the table up."""
        if not self._get_state.service_is_ready():
            return None
        time.sleep(settle)
        cx = float(self.get_parameter('table_map_x').value)
        cy = float(self.get_parameter('table_map_y').value)
        z0 = float(self._place_table_top_z_world) + BOX_HALF_H
        poses = {}
        for name in getattr(self, '_all_box_names', []):
            req = GetEntityState.Request(); req.name = name; req.reference_frame = 'world'
            fut = self._get_state.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
            if fut.done() and fut.result() is not None and fut.result().success:
                poses[name] = fut.result().state.pose.position
        levels = 0
        while any(math.hypot(p.x - cx, p.y - cy) <= STACK_CHECK_XY
                  and abs(p.z - (z0 + levels * BOX_HEIGHT)) <= STACK_CHECK_Z
                  for p in poses.values()):
            levels += 1
        for name, p in sorted(poses.items(), key=lambda kv: kv[1].z):
            self.get_logger().info(
                f'[final check] {name}: ({p.x:+.3f}, {p.y:+.3f}, {p.z:.3f}), '
                f'{math.hypot(p.x - cx, p.y - cy) * 1000:.0f} mm from the table centre')
        self.get_logger().info(f'[final check] STACK STANDING: {levels} level(s)')
        return levels

    def _claim_held_box(self):
        """Identify which box in self._remaining_boxes is held (well above the
        pickup-table rest height, e.g. after go_named('home')) and remove it from the
        list, so the per-box metrics use the right box. Falls back to the first
        remaining name if the Gazebo query is unavailable or inconclusive."""
        HELD_Z_THRESH = 0.16   # table-rest z=0.12; a held/retracted box reads much higher
        best_name, best_z = None, -1.0
        for name in self._remaining_boxes:
            if not self._get_state.service_is_ready():
                continue
            req = GetEntityState.Request()
            req.name = name
            req.reference_frame = 'world'
            fut = self._get_state.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
            if fut.done() and fut.result() is not None and fut.result().success:
                z = fut.result().state.pose.position.z
                if z > best_z:
                    best_name, best_z = name, z
        if best_name is None or best_z < HELD_Z_THRESH:
            held = self._remaining_boxes.pop(0)
            self.get_logger().warn(f'_claim_held_box: inconclusive query — assuming {held}')
            self._held_box_name = held
            return held
        self._remaining_boxes.remove(best_name)
        self.get_logger().info(f'_claim_held_box: identified held box = {best_name} (z={best_z:.3f})')
        # Remember it: the box is no longer in _remaining_boxes, which the ground-truth
        # helpers search, so they need this name to find the carried box.
        self._held_box_name = best_name
        return best_name

    def _cycle_measure_placement(self, expected_x: float, expected_y: float,
                                 expected_z: float, box_name: str = None):
        """Query Gazebo for the actual box position and record XYZ error."""
        if not self._get_state.service_is_ready():
            return
        name = box_name if box_name is not None else self.get_parameter('box_name').value
        req = GetEntityState.Request()
        req.name = name
        req.reference_frame = 'world'
        fut = self._get_state.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        if not fut.done() or not fut.result().success:
            self.get_logger().warn('[metrics] get_entity_state failed — placement error not recorded')
            return
        p = fut.result().state.pose.position
        self._current_cycle['placement_err_x'] = round(p.x - expected_x, 4)
        self._current_cycle['placement_err_y'] = round(p.y - expected_y, 4)
        self._current_cycle['placement_err_z'] = round(p.z - expected_z, 4)
        err_xy = math.hypot(p.x - expected_x, p.y - expected_y)
        self.get_logger().info(
            f'[metrics] box world=({p.x:.3f},{p.y:.3f},{p.z:.3f}) '
            f'target=({expected_x:.3f},{expected_y:.3f},{expected_z:.3f}) '
            f'err_xy={err_xy:.4f} m')

    def _cycle_end(self):
        """Finalise the current cycle dict and append to the metrics list."""
        if not hasattr(self, '_current_cycle'):
            return
        self._current_cycle['cycle_time_s'] = round(time.time() - self._cycle_start, 2)
        self._metrics.append(self._current_cycle)
        self.get_logger().info(
            f'[metrics] cycle {self._current_cycle["cycle"]} done: '
            f'pick={self._current_cycle["pick_success"]} '
            f'retain={self._current_cycle["transport_retained"]} '
            f'place={self._current_cycle["place_success"]} '
            f't={self._current_cycle["cycle_time_s"]}s')

    def flush_metrics(self):
        """Write all collected metrics to a CSV file under data/."""
        if not self._metrics:
            return
        csv_path = self.get_parameter('metrics_csv').value
        if not csv_path:
            try:
                data_dir = os.path.expanduser('~/thesis_ws/src/data')
                os.makedirs(data_dir, exist_ok=True)
            except Exception:
                data_dir = '/tmp'
            ts = time.strftime('%Y%m%d_%H%M%S')
            csv_path = os.path.join(data_dir, f'metrics_{ts}.csv')
        fieldnames = list(self._metrics[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._metrics)
        self.get_logger().info(f'[metrics] wrote {len(self._metrics)} rows → {csv_path}')

    def _grasp_attempts_csv_path(self):
        if not hasattr(self, '_grasp_csv_path'):
            try:
                data_dir = os.path.expanduser('~/thesis_ws/src/data')
                os.makedirs(data_dir, exist_ok=True)
            except Exception:
                data_dir = '/tmp'
            ts = time.strftime('%Y%m%d_%H%M%S')
            self._grasp_csv_path = os.path.join(data_dir, f'grasp_attempts_{ts}.csv')
        return self._grasp_csv_path

    def _log_grasp_attempt(self, attempt, verified):
        """Append one row per grasp attempt at once, so the outcome survives a crash.
        Centring, measured while the box is held up:
          jaw_mm   — between the jaws (0 = centred)
          slide_mm — along the flat finger faces; |slide| > 18 mm means the box is
                     past the edge of the 36 mm face."""
        path = self._grasp_attempts_csv_path()
        cycle = self._current_cycle['cycle'] if hasattr(self, '_current_cycle') else -1
        cen = getattr(self, '_last_centering', None)
        row = {
            'timestamp':         time.strftime('%Y-%m-%d %H:%M:%S'),
            'cycle':             cycle,
            'attempt':           attempt,
            'verified_success':  verified,
            'centering_jaw_mm':   round(cen[0] * 1000, 2) if cen else None,
            'centering_slide_mm': round(cen[1] * 1000, 2) if cen else None,
        }
        self._last_centering = None   # don't carry a stale reading into the next row
        write_header = not os.path.exists(path)
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self.get_logger().info(f'[verify] logged grasp attempt → {path}')

    def _offset(self):
        """Current grasp offset (live ROS params — tunable without relaunch)."""
        return (self.get_parameter('grasp_off_x').value,
                self.get_parameter('grasp_off_y').value,
                self.get_parameter('grasp_off_z').value)

    # ── Step 1: Navigate ──────────────────────────────────────────────────────

    def _lifecycle_state(self, node):
        """Current state label of a lifecycle node (diagnostic: change_state requests
        can fail silently, e.g. "Unable to start transition 3 from current state
        active")."""
        cli = self.create_client(GetState, f'/{node}/get_state')
        if not cli.wait_for_service(timeout_sec=2.0):
            return '<service unavailable>'
        fut = cli.call_async(GetState.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=3.0)
        res = fut.result()
        return res.current_state.label if res is not None else '<no response>'

    def _activate_nav2_cmdvel(self):
        """Re-activate the Nav2 cmd_vel nodes (in case a previous mission
        deactivated them for docking) so navigation can drive again."""
        # Only these two are launched (no velocity_smoother, no collision_monitor).
        for node in ('controller_server', 'behavior_server'):
            before = self._lifecycle_state(node)
            cli = self.create_client(ChangeState, f'/{node}/change_state')
            if not cli.wait_for_service(timeout_sec=3.0):
                self.get_logger().warn(f'[lifecycle] {node}/change_state unavailable — skipping (was {before})')
                continue
            req = ChangeState.Request()
            req.transition.id = Transition.TRANSITION_ACTIVATE
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            res = fut.result()
            after = self._lifecycle_state(node)
            self.get_logger().info(
                f'[lifecycle] activate {node}: {before} -> {after} '
                f'(change_state accepted={getattr(res, "success", None)})')

    def navigate_to_table(self):
        self.get_logger().info('=== Step 1: Navigate to table ===')
        self._activate_nav2_cmdvel()
        # AMCL is seeded in nav2_limo.launch.py from scene.yaml's robot.spawn_pose,
        # not here.

        if not self._nav.wait_for_server(timeout_sec=30.0):
            self.get_logger().error(
                'NavigateToPose action server not available — '
                'is nav2_limo.launch.py running?')
            return False

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = self._nav_goal_x
        goal.pose.pose.position.y = self._nav_goal_y
        # Yaw NAV_GOAL_YAW → quaternion
        half = self._nav_goal_yaw / 2.0
        goal.pose.pose.orientation.z = math.sin(half)
        goal.pose.pose.orientation.w = math.cos(half)

        self.get_logger().info(
            f'sending Nav2 goal → ({self._nav_goal_x:.2f}, {self._nav_goal_y:.2f}, '
            f'yaw={self._nav_goal_yaw:.3f})')

        gh = None
        for attempt in range(15):
            send_fut = self._nav.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_fut, timeout_sec=10.0)
            if send_fut.done():
                gh = send_fut.result()
                if gh is not None and gh.accepted:
                    self.get_logger().info('Nav2 goal accepted — driving to table…')
                    break
            self.get_logger().warn(f'Nav2 goal rejected or timed out (attempt {attempt+1}/15) — waiting for bt_navigator to become active...')
            end_time = self.get_clock().now() + Duration(seconds=2.0)
            while self.get_clock().now() < end_time:
                rclpy.spin_once(self, timeout_sec=0.1)
        else:
            self.get_logger().error('Nav2 goal rejected after 15 attempts — aborting')
            return False
        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=NAV_TIMEOUT_SEC)
        if not res_fut.done():
            self.get_logger().error(f'Nav2 did not complete within {NAV_TIMEOUT_SEC} s')
            return False

        res = res_fut.result()
        if res.status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(f'Nav2 navigation failed with status: {res.status}')
            return False

        self.get_logger().info('Nav2 navigation complete ✓')
        return True

    # ── Nav2 cmd_vel hand-off ─────────────────────────────────────────────────

    def _target_map_x(self, base_x, base_y, label='target'):
        """Map-frame x of a point given in base_link. Returns None if TF is unavailable.

        The target's base_link y is the lateral offset only when the robot is already
        square to the target, which it is not at a Nav2 staging pose: a residual heading
        error of ~15 deg over a 1.5 m standoff reads as ~0.39 m of offset. The
        base_link -> map transform contains the real heading, so the map x is correct
        however the robot is pointed.
        """
        try:
            pt = PointStamped()
            pt.header.frame_id = PLANNING_FRAME
            pt.point.x, pt.point.y = float(base_x), float(base_y)
            tf = self._tf_buf.lookup_transform('map', PLANNING_FRAME, rclpy.time.Time())
            return do_transform_point(pt, tf).point.x
        except Exception as e:
            self.get_logger().warn(f'[realign] no map transform for {label} ({e})')
            return None

    def _align_to_target_box(self):
        """Read the target box at staging range and slide sideways to face it square-on.

        Runs before visual_docking(), whose rotate-to-centre would turn a lateral
        offset into a diagonal approach. The robot stands still at the staging pose and
        the box is ~0.73 m away, well inside the camera's range.

        Not fatal: if the box is not visible or Nav2 refuses the goal, the cycle goes on
        with a diagonal approach.
        """
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.1)
        b = self._read_box_live()
        if b is None:
            self.get_logger().info(
                '[realign] no box reading at the staging pose — skipping alignment')
            return
        # Map x, not base_link y (see _target_map_x).
        map_x = self._target_map_x(b[0], b[1], label='target box')
        if map_x is None:
            return
        self._nav_goal_x = self.realign_perpendicular_to(
            map_x, self._nav_goal_x, self._nav_goal_y, self._nav_goal_yaw,
            label='target box', overshoot=REALIGN_OVERSHOOT)

    def realign_perpendicular_to(self, target_map_x, goal_x, goal_y, goal_yaw,
                                 label='target', tolerance=0.02, timeout=60.0,
                                 overshoot=0.0):
        """Slide sideways via Nav2 so the approach stays perpendicular to the table.

        If the robot turned to face an offset target and drove at it, it would reach
        the box diagonally (an outer box 0.10 m off the centre line at 0.73 m is a
        7.8 deg approach), and the gripper would close on a rotated box. A differential
        base cannot strafe, so this re-sends the staging goal shifted sideways with
        the same yaw; the docking then has almost nothing left to rotate.

        target_map_x is the target's x in the map frame (an absolute position, not an
        offset; see _target_map_x). Both tables are approached along -Y, so lining up
        the staging x with the target x makes the approach perpendicular.
        """
        delta = target_map_x - goal_x
        if abs(delta) <= tolerance:
            self.get_logger().info(
                f'[realign] {label} is {delta*1000:+.0f} mm off centre — '
                f'within {tolerance*1000:.0f} mm, approach is already perpendicular')
            return goal_x

        # Overshoot: Nav2 stops within its goal tolerance and the slide tended to fall short.
        new_x = target_map_x + math.copysign(overshoot, delta)
        self.get_logger().info(
            f'[realign] {label} is at map x {target_map_x:.3f}, {delta*1000:+.0f} mm off '
            f'the approach centre line. Sliding the staging pose {delta:+.3f} m '
            f'({goal_x:.3f} -> {new_x:.3f}) so the approach stays perpendicular '
            f'instead of diagonal.')

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = new_x
        goal.pose.pose.position.y = goal_y
        half = goal_yaw / 2.0
        goal.pose.pose.orientation.z = math.sin(half)
        goal.pose.pose.orientation.w = math.cos(half)

        send_fut = self._nav.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut, timeout_sec=10.0)
        gh = send_fut.result() if send_fut.done() else None
        if gh is None or not gh.accepted:
            self.get_logger().warn(
                '[realign] Nav2 rejected the alignment goal — continuing with the '
                'diagonal approach rather than aborting the cycle')
            return goal_x

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=timeout)
        if not res_fut.done():
            self.get_logger().warn('[realign] alignment goal timed out — continuing')
            return goal_x

        self.get_logger().info('[realign] aligned — approach is now perpendicular')
        # Returned so the caller can keep its goal in sync: a later realign must
        # measure from where the robot now is, not the original staging pose.
        return new_x

    def _silence_nav2_cmdvel(self):
        """Deactivate the Nav2 nodes that publish /cmd_vel (controller_server and
        behavior_server) so the orchestrator owns the topic during docking."""
        for node in ('controller_server', 'behavior_server'):
            before = self._lifecycle_state(node)
            cli = self.create_client(ChangeState, f'/{node}/change_state')
            if not cli.wait_for_service(timeout_sec=3.0):
                self.get_logger().warn(f'[lifecycle] {node}/change_state unavailable — skipping (was {before})')
                continue
            req = ChangeState.Request()
            req.transition.id = Transition.TRANSITION_DEACTIVATE
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
            res = fut.result()
            after = self._lifecycle_state(node)
            self.get_logger().info(
                f'[lifecycle] deactivate {node}: {before} -> {after} '
                f'(change_state accepted={getattr(res, "success", None)}) (released /cmd_vel)')

    # ── Step 1.5: Visual docking ──────────────────────────────────────────────

    def _read_box_live(self):
        """One fresh /box_pose reading (x, y) in base_link, or None if missing or
        out of the plausible range. Single readings, for the docking loops."""
        rclpy.spin_once(self, timeout_sec=0.05)
        p = self._latest_box
        if p is None or p.header.frame_id != PLANNING_FRAME:
            return None
        # box_pose_estimator re-publishes its last detection with fresh stamps even when
        # nothing is seen; only accept a real sighting.
        if self._box_age > BOX_FRESH_AGE:
            return None
        x, y = p.pose.position.x, p.pose.position.y
        if x < 0.10 or x > 1.5:
            return None
        return x, y

    def _all_box_world_z(self):
        """World z of every candidate box, as {name: z}.

        Ground truth for the grasp check. All boxes are measured because "which box
        are we holding" is what the check has to find out: whichever box rose is the
        one in the gripper. (Guessing the target first measured an untouched box and
        reported a good lift as a failure.)
        """
        if not self._get_state.service_is_ready():
            return {}
        out = {}
        for name in self._candidate_box_names():
            req = GetEntityState.Request()
            req.name = name
            req.reference_frame = 'world'      # Gazebo's own frame, not TF
            fut = self._get_state.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
            if fut.done() and fut.result() is not None and fut.result().success:
                out[name] = fut.result().state.pose.position.z
        return out

    def _measure_grasp_centering(self):
        """How far off-centre did the box end up between the fingers?

        Measured while the box is held up: then the box is fixed relative to the
        gripper, so the box position minus the tool centre is the grasp error. Uses
        Gazebo ground truth for the box and TF for gripper_tcp.

        A consistent sign across attempts is a systematic offset (grasp_off_y), not noise.

        Returns (jaw, slide, along) in the gripper frame, or None:
          jaw   — between the jaws (the "is it centred" number)
          slide — along the flat finger faces
          along — down the finger axis (depth into the gap)
        """
        box_p = self._box_world_pose()
        if box_p is None:
            return None
        tw = self._true_world_tf(TCP_LINK)
        if tw is None:
            return None
        t, R = tw
        d = np.array([box_p.x, box_p.y, box_p.z]) - t

        # Rotate the world-frame offset into the gripper frame (R^T · d) so the
        # components do not depend on the robot's yaw.
        local = R.T @ d

        # Gripper-frame axes (gripper.xacro):
        #   x — the jaw-opening direction. Finger pivots sit at x = ∓0.012 and
        #       gripper_tcp at x = 0, so the tool frame is centred between the jaws.
        #   y — along the fingers (the approach axis; points down in a top-down grasp).
        #   z — the hinge axis: the box can slide along the flat finger faces in this
        #       direction, so an offset here means the box is near a face edge.
        jaw, along, slide = float(local[0]), float(local[1]), float(local[2])
        self.get_logger().info(
            f'[centering] box vs tool centre (gripper frame): '
            f'between-jaws={jaw * 1000:+.1f} mm  '
            f'along-faces={slide * 1000:+.1f} mm  '
            f'depth-into-gap={along * 1000:+.1f} mm  → '
            f'{"CENTRED" if abs(jaw) < 0.005 else "OFF-CENTRE"} between jaws; '
            f'{"within" if abs(slide) < FINGER_FACE_HALF else "PAST THE EDGE OF"} the finger faces')
        self._last_centering = (jaw, slide, along)
        return jaw, slide, along

    def _true_world_tf(self, child):
        """True pose of TF frame `child` in the Gazebo world: (position, rotation matrix),
        or None. The robot model's pose ('mbot', origin base_footprint) composed with TF
        base_footprint -> child. With encoder odometry 'odom' is not the world frame, so
        the ground-truth checks must not read the robot from 'odom'."""
        if not self._get_state.service_is_ready():
            return None
        req = GetEntityState.Request()
        req.name = 'mbot'
        req.reference_frame = 'world'
        fut = self._get_state.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        if not (fut.done() and fut.result() is not None and fut.result().success):
            return None
        p = fut.result().state.pose
        try:
            tf = self._tf_buf.lookup_transform('base_footprint', child, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f'[truth] TF base_footprint→{child} unavailable: {e}')
            return None
        Rw = _qmat(p.orientation)
        tc = tf.transform.translation
        t = np.array([p.position.x, p.position.y, p.position.z]) + Rw @ np.array([tc.x, tc.y, tc.z])
        return t, Rw @ _qmat(tf.transform.rotation)

    def _box_world_pose(self):
        """Full box position in the Gazebo world frame (sim ground truth), or None."""
        if not self._get_state.service_is_ready():
            return None
        names = self._candidate_box_names()
        for name in names:
            req = GetEntityState.Request()
            req.name = name
            req.reference_frame = 'world'
            fut = self._get_state.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
            if fut.done() and fut.result() is not None and fut.result().success:
                return fut.result().state.pose.position
        return None

    def _candidate_box_names(self):
        """Which Gazebo models the ground-truth helpers should query, best guess first.

        Order matters — the helpers return the first name that resolves:
          1. the box we are carrying, if known (_claim_held_box() removes it from
             _remaining_boxes).
          2. the boxes still waiting on the pickup table.
          3. a fixed fallback list, if neither is known.
        """
        names = []
        held = getattr(self, '_held_box_name', None)
        if held:
            names.append(held)
        # The box _verify_grasp_truth() saw rise, i.e. the one in the gripper. Known
        # only after a lift; it keeps the centring measurement and the place step
        # pointed at the right object.
        target = getattr(self, '_target_box_name', None)
        if target and target not in names:
            names.append(target)
        names += [n for n in (getattr(self, '_remaining_boxes', None) or [])
                  if n not in names]
        if not names:
            names = ['stack_box_0', 'stack_box_1', 'stack_box_2',
                     'target_box', 'grasp_test_box']
        return names

    def _held_diag(self, label):
        """Log whether the box is still held after one named motion (sim ground truth)."""
        held = self._box_is_held()
        self.get_logger().info(f'[diag] held after {label}: {held}')
        return held

    def _box_is_held(self, max_dist=0.06):
        """Is the box still in the hand right now? Ground-truth distance from the box
        to gripper_tcp.

        _weld_active is set once at the lift; the plugin joint can still break later
        (contacts lost on a hard turn, the box knocked on the table edge), so transport
        retention is measured again here.

        A held box reads ~0.00 m from the TCP; a dropped box is several times further
        than max_dist. The distance is logged on every call.

        Returns True/False, or None if ground truth is unavailable (caller decides).
        """
        box_p = self._box_world_pose()
        if box_p is None:
            return None
        tw = self._true_world_tf(TCP_LINK)
        if tw is None:
            return None
        dist = float(np.linalg.norm(np.array([box_p.x, box_p.y, box_p.z]) - tw[0]))
        held = dist <= max_dist
        self.get_logger().info(
            f'[held] box is {dist:.3f} m from {TCP_LINK} (limit {max_dist:.3f}) → '
            f'{"STILL HELD" if held else "NOT HELD — box was lost"}')
        return held

    def _check_transport_retained(self):
        """Did the box survive the drive to the place table? Sampled at arrival.

        Also re-syncs _weld_active, so a box lost in transit skips the place step.
        Falls back to the latched flag if ground truth is unavailable.
        """
        held = self._box_is_held()
        if held is not None:
            if self._weld_active and not held:
                self.get_logger().error(
                    '[transport] box was LOST in transit — the physics joint broke '
                    'somewhere between the pickup table and here')
            self._weld_active = bool(held)
            return bool(held)
        self.get_logger().warn(
            '[transport] ground truth unavailable — reporting the latched grasp flag')
        return self._weld_active

    def _verify_grasp_truth(self, z_before, min_rise=0.02):
        """Did the box leave the table? Compares the true world z of every box before
        the close with after the lift.

        The lift raises the TCP by 0.06 m; min_rise is 0.02 m so a held box that sags
        in the fingers still counts, while a box left on the table moves 0.000 m.

        Returns True / False, or None if Gazebo state is unavailable.
        """
        if not z_before:
            return None
        z_after = self._all_box_world_z()
        if not z_after:
            return None

        # Which box rose the most? That is the one in the gripper.
        risen = {n: z_after[n] - z_before[n] for n in z_after if n in z_before}
        if not risen:
            return None
        name = max(risen, key=risen.get)
        rise = risen[name]
        ok = rise >= min_rise

        others = ', '.join(f'{n}{risen[n]:+.3f}' for n in sorted(risen) if n != name)
        self.get_logger().info(
            f'[verify] GROUND TRUTH: {name} z {z_before[name]:.3f} → {z_after[name]:.3f} '
            f'(rise {rise:+.3f} m, need ≥{min_rise:.3f}) → '
            f'{"LIFTED (grasped)" if ok else "DID NOT MOVE (failed)"}'
            + (f'  [others: {others}]' if others else ''))

        # Latch the winner so the centering measurement and the place step follow the
        # box we are actually carrying.
        if ok:
            self._target_box_name = name
            # Also the held box: _candidate_box_names() asks for it first, and it must not
            # still name the box of the previous cycle (already on the place table).
            self._held_box_name = name
        return ok

    def _drive_forward(self, drive, d_start, timeout=25.0):
        """Drive straight forward `drive` metres, measured by odometry, then hard stop.

        Odometry rather than the camera: a camera-feedback stop needs the forward and
        lateral errors to settle at the same moment, which often did not happen, and
        the robot crept into the table.

        STOP_MARGIN: the commanded speed never drops below 0.06 m/s and one more control
        cycle passes before the stop takes effect, so the robot overshoots a little.
        Stopping 5 mm short absorbs that; the arm reaches to 0.24 m.
        """
        STOP_MARGIN = 0.005
        drive = max(0.0, drive - STOP_MARGIN)
        if drive <= 0.001:
            return 0.0
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._odom is not None:
                break
        ox, oy = self._odom if self._odom is not None else (0.0, 0.0)

        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            disp = self._odom_dist_since(ox, oy)
            disp = 0.0 if disp is None else disp
            remaining = drive - disp
            self.get_logger().info(
                f'approach: driven {disp:.3f}/{drive:.3f} m  (box ≈ {d_start - disp:.3f} m)',
                throttle_duration_sec=0.5)
            if remaining <= 0.0:
                self.get_logger().info(f'stop — {d_start - drive:.3f} m from the box')
                break
            t = Twist()
            t.linear.x = max(0.06, min(0.14, 0.6 * remaining))   # ease off near the stop
            self._cmd_vel_pub.publish(t)
            time.sleep(0.05)
        else:
            self.get_logger().warn('approach timed out — stopping')

        self._cmd_vel_pub.publish(Twist())
        time.sleep(0.5)
        # Return what was actually driven, not what was asked for (they differ by
        # STOP_MARGIN, a timeout, odometry error). The caller aims the arm with it.
        final = self._odom_dist_since(ox, oy)
        return 0.0 if final is None else float(final)

    # ── Square pick approach ──────────────────────────────────────────────────────────
    # Turning the base to face the box would turn a lateral offset into an approach angle,
    # and the fingers would meet the box diagonally. So: square to the table first, measure
    # the box's lateral offset (true once square), let the arm reach it if it is small,
    # move the base sideways only if it is not, and drive in holding the square heading.

    def _table_face_error(self, ref_yaw=None):
        """Heading error to the table's front face (rad, + = turn left), and its source.

        Primary: a straight line fitted to the lidar points on the table face ahead. Backup:
        the table's known approach direction in the map (ref_yaw, default nav_goal_yaw = the
        pick table) against the robot's map heading. Returns (err, source) or (None, None)."""
        err_l = None
        sc = self._scan
        if sc is not None:
            age = (self.get_clock().now() - rclpy.time.Time.from_msg(sc.header.stamp)).nanoseconds * 1e-9
            if age > 0.5:
                sc = None                          # stale scan: do not fit it
        if sc is not None:
            try:
                tf = self._tf_buf.lookup_transform(PLANNING_FRAME, sc.header.frame_id,
                                                   rclpy.time.Time())
                q = tf.transform.rotation
                lyaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                a = sc.angle_min + np.arange(len(sc.ranges)) * sc.angle_increment + lyaw
                r = np.asarray(sc.ranges, float)
                ok = np.isfinite(r) & (r > 0.12) & (r < 1.6) & (np.abs(a) < math.radians(25))
                P = np.column_stack((r[ok] * np.cos(a[ok]) + tf.transform.translation.x,
                                     r[ok] * np.sin(a[ok]) + tf.transform.translation.y))
                if len(P) >= 20:
                    rng, best = np.random.default_rng(0), None
                    for _ in range(200):
                        i, j = rng.choice(len(P), 2, replace=False)
                        d = P[j] - P[i]
                        if math.hypot(*d) < 0.08:
                            continue
                        nrm = np.array([-d[1], d[0]]) / math.hypot(*d)
                        inl = np.abs((P - P[i]) @ nrm) < 0.01
                        if best is None or inl.sum() > best.sum():
                            best = inl
                    if best is not None and best.sum() >= 20:
                        Q = P[best]
                        span = np.ptp(Q @ np.array([0.0, 1.0]))
                        nrm = np.linalg.svd(Q - Q.mean(axis=0))[2][1]
                        if nrm @ Q.mean(axis=0) < 0:
                            nrm = -nrm
                        phi = math.atan2(nrm[1], nrm[0])
                        if span > 0.12 and abs(phi) < math.radians(30):
                            err_l = phi
            except Exception as e:
                self.get_logger().warn(f'[square] lidar fit failed: {e}',
                                       throttle_duration_sec=2.0)
        err_m = None
        try:
            tf = self._tf_buf.lookup_transform('map', PLANNING_FRAME, rclpy.time.Time())
            q = tf.transform.rotation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            err_m = self._wrap((self._nav_goal_yaw if ref_yaw is None else ref_yaw) - yaw)
        except Exception:
            pass
        # A lidar line that disagrees with the map by more than 5 deg is probably another surface.
        if err_l is not None and err_m is not None and abs(err_l - err_m) > math.radians(5.0):
            self.get_logger().warn(
                f'[square] lidar line {math.degrees(err_l):+.1f} deg disagrees with the map '
                f'{math.degrees(err_m):+.1f} deg -- ignoring the lidar fit', throttle_duration_sec=2.0)
            err_l = None
        self._last_square_errs = (err_l, err_m)
        if err_l is not None:
            return err_l, 'lidar'
        if err_m is not None:
            return err_m, 'map'
        return None, None

    def _lidar_front_gap(self, half_width=0.03, newer_than=None):
        """Chassis front -> first surface straight ahead, from the lidar (m), or None.

        Median x (base_link) of the scan points in a strip |y| < half_width in front of the
        robot, minus bumper_x. newer_than: only accept a scan stamped after this time."""
        sc = self._scan
        if sc is None:
            return None
        stamp = rclpy.time.Time.from_msg(sc.header.stamp)
        if (self.get_clock().now() - stamp).nanoseconds * 1e-9 > 0.5:
            return None
        if newer_than is not None and stamp <= newer_than:
            return None
        try:
            tf = self._tf_buf.lookup_transform(PLANNING_FRAME, sc.header.frame_id, rclpy.time.Time())
        except Exception:
            return None
        q = tf.transform.rotation
        lyaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        a = sc.angle_min + np.arange(len(sc.ranges)) * sc.angle_increment + lyaw
        r = np.asarray(sc.ranges, float)
        ok = np.isfinite(r) & (r > 0.05) & (r < 1.0)
        x = r[ok] * np.cos(a[ok]) + tf.transform.translation.x
        y = r[ok] * np.sin(a[ok]) + tf.transform.translation.y
        sel = (np.abs(y) < half_width) & (x > 0.05)
        if sel.sum() < 3:
            return None
        # The lidar has a minimum range (0.10 m, as the real T-mini Pro). Closer than that
        # the simulated rays start inside the 80 mm table block and return its back face,
        # which looks like a valid range, so callers read from further back (see
        # PLACE_DOCK_FINAL_HOP); this only refuses faces right at the edge of the zone.
        if float(np.median(r[ok][sel])) < sc.range_min + 0.01:
            return None
        return float(np.median(x[sel])) - float(self.get_parameter('bumper_x').value)

    def _lidar_gap_standing(self, timeout=2.0):
        """_lidar_front_gap() from a scan stamped after this call. Call it with the base
        stopped: an older scan was taken while moving (at 0.05 m/s, 0.5 s is 25 mm)."""
        t0 = self.get_clock().now()
        end = time.time() + timeout
        while time.time() < end:
            gap = self._lidar_front_gap(newer_than=t0)
            if gap is not None:
                return gap
            time.sleep(0.05)
        return None

    def _square_to_table(self, tol=math.radians(1.0), timeout=15.0, ref_yaw=None):
        """Turn on the spot until the table face is square to the axle (both errors logged).
        Returns True only if it converged."""
        ok, deadline = 0, time.time() + timeout
        while time.time() < deadline:
            time.sleep(0.13)                       # a fresh scan (lidar 8 Hz)
            err, src = self._table_face_error(ref_yaw)
            if err is None:
                self.get_logger().warn('[square] no heading reference')
                break
            if abs(err) < tol:
                ok += 1
                if ok >= 3:
                    break
                self._cmd_vel_pub.publish(Twist())
                continue
            ok = 0
            w = max(-0.4, min(0.4, 1.0 * err))
            if abs(w) < 0.12:
                w = math.copysign(0.12, w)
            t = Twist(); t.angular.z = w
            self._cmd_vel_pub.publish(t)
        self._cmd_vel_pub.publish(Twist()); time.sleep(0.3)
        self._table_face_error(ref_yaw)
        err_l, err_m = self._last_square_errs
        self.get_logger().info(
            ('[square] squared to the table: lidar ' if ok >= 3 else '[square] NOT squared: lidar ')
            + ('n/a' if err_l is None else f'{math.degrees(err_l):+.2f} deg')
            + ', map ' + ('n/a' if err_m is None else f'{math.degrees(err_m):+.2f} deg'))
        return ok >= 3

    def _crab_sideways(self, dy):
        """Shift the base dy metres to its left (+) or right (-): turn 90, drive, turn back."""
        pivot = math.copysign(math.pi / 2.0, dy)
        if not self._rotate_by(pivot):
            return False
        drove = self._drive_straight(abs(dy))
        back = self._rotate_by(-pivot)            # turn back even if the drive failed
        return drove and back

    # ── The real robot's steps (face_and_pick.py) ─────────────────────────────

    def _map_yaw(self):
        """Robot heading in the map frame (rad), or None."""
        try:
            tf = self._tf_buf.lookup_transform('map', PLANNING_FRAME, rclpy.time.Time())
        except Exception:
            return None
        q = tf.transform.rotation
        return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def _align_to_station(self, goal_yaw, label):
        """nav_to(): after Nav2, turn to the station's saved heading if more than 2 deg off.
        The real robot reads its heading from AMCL; here from the same map frame."""
        for _ in range(10):
            rclpy.spin_once(self, timeout_sec=0.1)
        y = self._map_yaw()
        if y is None:
            self.get_logger().warn(f'[match] {label}: no map heading — not aligning')
            return
        d = self._wrap(goal_yaw - y)
        self.get_logger().info(f'[match] ALIGN at {label}: heading off by {math.degrees(d):+.1f} deg')
        self._event('station_align', station=label, err_deg=math.degrees(d))
        if abs(d) > MATCH_STATION_TOL:
            self._rotate_by(d)

    def _turn_clear(self):
        """turn_clear(): never turn on the spot with something inside the turning circle.
        Reverse in 5 cm steps (max 0.35 m) until the lidar sees nothing within it."""
        backed = 0.0
        while True:
            time.sleep(0.3)                                   # a fresh scan
            sc = self._scan
            if sc is None:
                self.get_logger().error('[match] no /scan — refusing to turn without the clearance check')
                return False
            try:
                tf = self._tf_buf.lookup_transform(PLANNING_FRAME, sc.header.frame_id,
                                                   rclpy.time.Time())
            except Exception:
                self.get_logger().error('[match] no lidar transform — refusing to turn')
                return False
            q = tf.transform.rotation
            lyaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            a = sc.angle_min + np.arange(len(sc.ranges)) * sc.angle_increment + lyaw
            r = np.asarray(sc.ranges, float)
            ok = np.isfinite(r) & (r > sc.range_min) & (r < sc.range_max)
            x = r[ok] * np.cos(a[ok]) + tf.transform.translation.x
            y = r[ok] * np.sin(a[ok]) + tf.transform.translation.y
            out = ~((np.abs(x) <= MATCH_TURN_FOOT_X) & (np.abs(y) <= MATCH_TURN_FOOT_Y))
            d = float(np.min(np.hypot(x[out], y[out]))) if out.any() else None
            if d is None or d >= MATCH_TURN_RADIUS:
                if backed:
                    self.get_logger().info(f'[match] turning circle clear after reversing {backed:.2f} m')
                return True
            if backed >= MATCH_BACK_MAX:
                self.get_logger().error(
                    f'[match] obstacle {d * 1000:.0f} mm inside the turning circle even after '
                    f'reversing {backed:.2f} m — not turning')
                return False
            self.get_logger().info(
                f'[match] obstacle {d * 1000:.0f} mm inside the turning circle '
                f'({MATCH_TURN_RADIUS * 1000:.0f} mm) — reverse 5 cm first')
            self._drive_back(MATCH_BACK_STEP)
            backed += MATCH_BACK_STEP

    def _drive_back(self, dist, speed=0.08, timeout=10.0):
        """Reverse `dist` metres, closed on odometry."""
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._odom is not None:
                break
        if self._odom is None:
            return 0.0
        x0, y0 = self._odom
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            gone = self._odom_dist_since(x0, y0) or 0.0
            if gone >= dist:
                break
            t = Twist(); t.linear.x = -speed
            self._cmd_vel_pub.publish(t)
            time.sleep(0.05)
        self._cmd_vel_pub.publish(Twist())
        time.sleep(0.3)
        return self._odom_dist_since(x0, y0) or 0.0

    def _match_centre_then_square(self, label, look, ok, ref_yaw):
        """centre_then_square(): heading error once (lidar, map heading as backup), target
        once, shift so the target lies on the centre line AFTER the squaring turn
        (s = y - x*tan(th)), turn once, then only extra shifts (no squaring again) until
        ok(y). look() -> (x, y) of the target in base_link, or None. Returns y or None."""
        if not self._turn_clear():
            return None
        time.sleep(0.13)                           # a fresh scan (lidar 8 Hz)
        th, src = self._table_face_error(ref_yaw)
        first = look()
        if th is None or first is None:
            self.get_logger().error(f'[match] {label}: no heading or no target — stopping')
            return None
        x, y = first
        s = y - x * math.tan(th)
        self.get_logger().info(f'[match] {label}: heading error {math.degrees(th):+.1f} deg ({src}), '
                               f'target x {x:.3f} y {y * 1000:+.0f} mm -> shift {s * 1000:+.0f} mm, '
                               f'then turn {math.degrees(th):+.1f} deg')
        if abs(s) >= MATCH_SHIFT_MIN and not self._crab_sideways(s):
            self.get_logger().error(f'[match] {label}: sideways shift failed')
            return None
        if abs(th) > MATCH_SQUARE_TOL and not self._rotate_by(th):
            self.get_logger().error(f'[match] {label}: squaring turn failed')
            return None
        time.sleep(0.3)
        self._table_face_error(ref_yaw)
        err_l, err_m = self._last_square_errs
        self._event('table_square', label=label, round=0, ok=True, shift=s,
                    lidar_deg=None if err_l is None else math.degrees(err_l),
                    map_deg=None if err_m is None else math.degrees(err_m))
        for rnd in range(1, MATCH_EXTRA_SHIFTS + 2):
            t = look()
            if t is None:
                self.get_logger().error(f'[match] {label}: target lost after centring')
                return None
            if ok(t[1]):
                self.get_logger().info(f'[match] {label}: centred and square, y={t[1] * 1000:+.0f} mm '
                                       f'after {rnd - 1} extra shift(s)')
                return t[1]
            if rnd == MATCH_EXTRA_SHIFTS + 1:
                break
            self.get_logger().info(f'[match] {label}: residual {t[1] * 1000:+.0f} mm -> extra '
                                   f'sideways shift {rnd}/{MATCH_EXTRA_SHIFTS}')
            if not self._crab_sideways(t[1]):
                self.get_logger().error(f'[match] {label}: sideways shift failed')
                return None
        self.get_logger().error(f'[match] {label}: not centred — stopping')
        return None

    def _look_box(self):
        b = self.get_box_xyz(samples=10, timeout=10.0)
        return None if b is None else (b[0], b[1])

    def _match_approach(self, label, ref_yaw=None, centre=True):
        """trial() of face_and_pick.py up to the pick: face the cube (stationary reading,
        turn on odometry), far look, drive to MATCH_FINAL_READ, second look, then drive
        blind to the dock. The last reading is taken at ~0.60 m because the real camera has
        no depth closer than ~0.30 m. Sets self._dock_box.
        Also used for the approach onto the stack at the place table (cubes 2..N), with
        centre=False: the tag dock has just centred and squared the robot there."""
        # 1. centre on the cube, then square to the table (turning toward the cube would
        #    leave the robot at an angle to the table, and the grasp would be diagonal).
        if centre and self._match_centre_then_square(
                label, self._look_box, lambda v: abs(v) < MATCH_PICK_Y_TOL, ref_yaw) is None:
            return False
        # 2. far look (decides only the distance to drive)
        far = self.get_box_xyz(samples=20, timeout=20.0)
        if far is None:
            self.get_logger().error(f'[match] {label}: lost after the turn')
            return False
        near, cons = far, None
        if abs(far[1]) > MATCH_LATERAL_MAX:
            self.get_logger().error(f'[match] {label}: {far[1] * 1000:+.0f} mm lateral, beyond the arm')
            return False
        # 3. drive to the accurate-reading range, second look (decides the target)
        if far[0] - MATCH_FINAL_READ > MATCH_SECOND_LOOK:
            moved = self._drive_forward(far[0] - MATCH_FINAL_READ, far[0])
            near = self.get_box_xyz(samples=20, timeout=20.0)
            if near is None:
                self.get_logger().error(f'[match] {label}: lost on the second look')
                return False
            cons = near[0] - (far[0] - moved)
            self.get_logger().info(f'[match] {label}: far/near consistency {cons * 1000:+.0f} mm')
            if abs(near[1]) > MATCH_LATERAL_MAX:
                self.get_logger().error(f'[match] {label}: {near[1] * 1000:+.0f} mm lateral, beyond the arm')
                return False
        # 4. front face -> centre, then blind to the dock distance
        x_c = near[0] + float(self.get_parameter('box_width').value) / 2.0
        driven = self._drive_forward(x_c - self._stop_distance, x_c)
        measured_x = x_c - driven
        self._dock_box = (measured_x, near[1], near[2])
        self._event('match_approach', label=label, far=list(far), near=list(near),
                    consistency=cons, blind=x_c - self._stop_distance, driven=driven,
                    target=list(self._dock_box))
        self.get_logger().info(
            f'[match] {label}: last reading at {near[0]:.3f} m, drove {driven:.3f} m blind, '
            f'target ({measured_x:.3f}, {near[1]:+.3f}, {near[2]:+.3f})')
        return True

    def _look_tag(self):
        """Tag centre (x, y) in base_link: median of up to 15 readings, or None."""
        xs, ys = [], []
        end = time.time() + 3.0
        while time.time() < end and len(ys) < 15:
            t = self._read_tag_live()
            if t is not None:
                xs.append(t[0]); ys.append(t[1])
            time.sleep(0.05)
        if len(ys) < 3:
            self.get_logger().error('[match] tag dock: no AprilTag seen')
            return None
        ty = statistics.median(ys)
        self.get_logger().info(f'[match] tag dock: tag centre y {ty * 1000:+.0f} mm '
                               f'(spread {(max(ys) - min(ys)) * 1000:.1f} mm, n={len(ys)})')
        self._event('tag_lateral', attempt=self._tag_looks, tag_y=ty, spread=max(ys) - min(ys))
        self._tag_looks += 1
        return statistics.median(xs), ty

    def _match_tag_lateral(self):
        """tag_dock(): centre on the tag, square once, then place on the tag centre y if it
        is inside the arm's band (else stop, the box stays in the gripper).
        Sets self._match_tag_y."""
        self._tag_looks = 0
        first = self._look_tag()
        if first is None:
            self.get_logger().error('[match] tag dock failed — stopping (not placing blind)')
            return False
        if first[0] > MATCH_DOCK_X + 0.05:
            self.get_logger().info(f'[match] table B: tag at x {first[0]:.3f} -> drive '
                                   f'{first[0] - MATCH_DOCK_X:.3f} m in before docking')
            self._drive_forward(first[0] - MATCH_DOCK_X, first[0])
        ty = self._match_centre_then_square('table B', self._look_tag,
                                            lambda v: MATCH_TAG_Y_MIN <= v <= MATCH_TAG_Y_MAX,
                                            self._nav_place_yaw)
        if ty is None:
            self.get_logger().error('[match] tag dock failed — stopping (not placing blind)')
            return False
        self._match_tag_y = ty
        return True

    def _match_place_dock(self, level):
        """face_and_pick pipeline at table B: station heading, tag centre y, then
        level 0: close the gap on the lidar and place on the table (place_ahead);
        levels >= 1: approach the stack the camera sees, exactly like a pick (trial())."""
        self._align_to_station(self._nav_place_yaw, 'place station')
        if not self._match_tag_lateral():
            return False
        if level >= 1:
            # Carry pose of the real robot: arm straight up, so the camera sees the stack.
            self._go_named_safe('home')
            if not self._match_approach(f'stack L{level}',
                                        ref_yaw=self._nav_place_yaw, centre=False):
                return False
            self._latched_drop = (self._dock_box[0], self._dock_box[1])
            return True
        # level 0 — place_ahead(): lidar gap, drive in. The simulated lidar cannot read a
        # face closer than 0.10 m (it then returns the table's far side), so the last
        # PLACE_DOCK_FINAL_HOP is driven on odometry from a gap read before it.
        bumper = float(self.get_parameter('bumper_x').value)
        target_gap = float(self.get_parameter('dock_range').value) - bumper
        g0 = self._lidar_gap_standing()
        if g0 is None or g0 > 0.75:
            self.get_logger().error(f'[match] lidar gap to the place table {g0} — not driving blind')
            return False
        first = g0 - target_gap - PLACE_DOCK_FINAL_HOP
        if first > 0.003:
            self._drive_forward(first, g0)
        gap_l = self._lidar_gap_standing()
        if gap_l is None:
            self.get_logger().error('[match] no lidar gap before the final hop — not placing')
            return False
        hop = gap_l - target_gap
        self._event('dock_gap_read', gap_lidar_pre=gap_l, tag_range=None, dock_range=target_gap + bumper,
                    dock_gap_target=target_gap, tag_hop=None, hop_cmd=hop)
        driven = self._drive_forward(hop, gap_l) if hop > 0.003 else 0.0
        gap_final = gap_l - driven
        self._event('dock_hop_done', hop_odom=driven, gap_final_est=gap_final)
        raw_x = bumper + gap_final + float(self.get_parameter('place_table_side').value) / 2.0
        # Not the real robot's rule (table face + 44 mm): the simulated arm finds no straight
        # descent at x 0.229 m; its place column is x 0.22 (reach_map.csv).
        final_x = max(0.22, raw_x - float(self.get_parameter('place_near_edge_bias').value))
        self._latched_drop = (final_x, self._match_tag_y)
        self._place_table_centre = (raw_x, self._match_tag_y)
        self.get_logger().info(f'[match] place dock: gap {g0 * 1000:.0f} -> {gap_l * 1000:.0f} mm, '
                               f'hop {driven * 1000:.0f} mm, drop ({final_x:.3f}, {self._match_tag_y:+.3f})')
        return True

    def visual_docking(self):
        """Pick-table dock, as face_and_pick.py --pipeline: turn to the station heading,
        drive MATCH_A_FORWARD toward the table, then _match_approach() (square, centre,
        look, drive blind to the grasp distance)."""
        self.get_logger().info('=== Step 1.5: Land → align → approach (diff) ===')
        # Hand /cmd_vel over from Nav2 to us, or the dock crawls/stalls.
        self._silence_nav2_cmdvel()

        self._align_to_station(self._nav_goal_yaw, 'pick station')
        self.get_logger().info(f'[match] FORWARD {MATCH_A_FORWARD:.2f} m toward the pick table')
        self._drive_forward(MATCH_A_FORWARD, MATCH_A_FORWARD)
        return self._match_approach('pick')

    # ── Step 3: Perceive box ──────────────────────────────────────────────────

    def get_box_xyz(self, samples=8, timeout=20.0):
        self.get_logger().info('=== Step 3: Perceiving box ===')
        readings = []
        deadline = time.time() + timeout
        while time.time() < deadline and len(readings) < samples:
            self._latest_box = None
            t0 = time.time()
            while self._latest_box is None and time.time() < t0 + 1.0:
                rclpy.spin_once(self, timeout_sec=0.1)
            p = self._latest_box
            # only real sightings, not the estimator's replays of its last detection
            if (p is not None and p.header.frame_id == PLANNING_FRAME
                    and self._box_age <= BOX_FRESH_AGE):
                readings.append((p.pose.position.x,
                                 p.pose.position.y,
                                 p.pose.position.z))
        if len(readings) < 3:
            self.get_logger().warn(
                'Not enough /box_pose readings — no box estimate')
            return None
        box = (statistics.median(r[0] for r in readings),
               statistics.median(r[1] for r in readings),
               statistics.median(r[2] for r in readings))
        self.get_logger().info(
            f'box @ base_link {tuple(round(v, 3) for v in box)} '
            f'(median of {len(readings)} readings)')
        return box

    # ── Gripper helpers ───────────────────────────────────────────────────────

    def _gripper_joint_cb(self, msg: JointState):
        try:
            idx = msg.name.index('gripper_controller')
            self._gripper_actual = msg.position[idx]
        except ValueError:
            pass
        # Also latch the arm joints: solve_ik_seeded() seeds IK from the arm's actual
        # configuration, which keeps the solution in the posture the arm is already in.
        try:
            self._arm_actual = [msg.position[msg.name.index(j)] for j in ARM_JOINTS]
        except ValueError:
            pass

    def _scan_cb(self, msg):
        self._scan = msg

    def set_gripper(self, values, label, duration=0.8, steps=20):
        start = self._last_gripper
        for i in range(1, steps + 1):
            a = i / steps
            interp = [s + a * (v - s) for s, v in zip(start, values)]
            msg = Float64MultiArray()
            msg.data = [float(x) for x in interp]
            self._gripper.publish(msg)
            rclpy.spin_once(self, timeout_sec=duration / steps)
        self._last_gripper = list(values)
        self.get_logger().info(f'gripper → {label}')

        # Opening the fingers past the plugin's release_position is the release; there
        # is no detach call. Clearing the flag here covers every release path (place,
        # retry, abort). RELEASE_POSITION mirrors <release_position> in
        # gazebo/mycobot_ros2_control.xacro; keep the two in sync.
        RELEASE_POSITION = 0.0
        if self._weld_active and values[0] > RELEASE_POSITION:
            self._weld_active = False
            self.get_logger().info(
                f'[grasp] fingers opened to {values[0]:+.3f} > release_position '
                f'{RELEASE_POSITION:+.3f} — physics joint released')

        time.sleep(0.3)

    def _spin_for(self, seconds: float):
        """Spin callbacks for a real wall-clock duration.

        rclpy.spin_once(timeout_sec=X) is not a sleep: it returns as soon as it runs
        one callback, and with /joint_states at 50 Hz there is always one pending. Used
        as a sleep, the whole gripper close ran in ~65 ms instead of seconds.
        """
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=max(0.001, end - time.time()))

    def close_gripper(self):
        """Step the gripper command down to grasp_close_floor, CLOSE_STEP_RAD every
        CLOSE_STEP_SEC. The grasp plugin attaches the cube once the fingers touch it;
        whether the cube is really held is checked after the lift."""
        self._weld_active = False
        cmd   = self._last_gripper[0]
        floor = float(self.get_parameter('grasp_close_floor').value)
        rclpy.spin_once(self, timeout_sec=0.05)
        while cmd > floor:
            cmd = max(floor, cmd - CLOSE_STEP_RAD)
            t   = (GRIPPER_OPEN[0] - cmd) / (GRIPPER_OPEN[0] - GRIPPER_GRASP[0])
            msg = Float64MultiArray()
            msg.data = [float(o + t * (c - o))
                        for o, c in zip(GRIPPER_OPEN, GRIPPER_GRASP)]
            self._gripper.publish(msg)
            self._last_gripper = msg.data[:]
            self._spin_for(CLOSE_STEP_SEC)
        self.get_logger().info(
            f'[close] gripper at {cmd:+.3f} rad (actual {self._gripper_actual:+.3f})')
        time.sleep(0.3)

    # ── MoveIt helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _joint_constraints(values, tol=0.01):
        # 0.01 rad: with 0.15 MoveIt accepted a pose still 0.15 rad off, which at the
        # wrist is enough to drive the gripper into the table.
        c = Constraints()
        for name, val in zip(ARM_JOINTS, values):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(val)
            jc.tolerance_above = tol
            jc.tolerance_below = tol
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        return c

    @staticmethod
    def _pose_constraints(x, y, z, quat, pos_tol=0.01, ori_tol=0.1):
        """pos_tol is the radius of the sphere MoveIt must land the TCP inside.

        1 cm: a larger sphere let MoveIt stop centimetres off the commanded point and
        still report success. A tighter goal is harder to solve.
        """
        c = Constraints()
        target = Pose()
        target.position = Point(x=float(x), y=float(y), z=float(z))
        target.orientation.x, target.orientation.y, \
            target.orientation.z, target.orientation.w = (float(q) for q in quat)

        pc = PositionConstraint()
        pc.header.frame_id = PLANNING_FRAME
        pc.link_name = TCP_LINK
        region = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [pos_tol]
        region.primitives.append(sphere)
        region.primitive_poses.append(target)
        pc.constraint_region = region
        pc.weight = 1.0
        c.position_constraints.append(pc)

        oc = OrientationConstraint()
        oc.header.frame_id = PLANNING_FRAME
        oc.link_name = TCP_LINK
        oc.orientation = target.orientation
        oc.absolute_x_axis_tolerance = ori_tol
        oc.absolute_y_axis_tolerance = ori_tol
        oc.absolute_z_axis_tolerance = ori_tol
        oc.weight = 1.0
        c.orientation_constraints.append(oc)
        return c

    def _send(self, constraints, label, plan_only=False, plan_time=PLAN_TIME_SEC):
        goal = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = ARM_GROUP
        req.goal_constraints = [constraints]
        req.num_planning_attempts = PLAN_ATTEMPTS
        req.allowed_planning_time = plan_time
        req.max_velocity_scaling_factor = VEL_SCALE
        req.max_acceleration_scaling_factor = ACC_SCALE
        goal.request = req
        goal.planning_options.plan_only = plan_only

        send_fut = self._move.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut, timeout_sec=15.0)
        if not send_fut.done():
            self.get_logger().error(f'[{label}] goal not accepted within 15 s')
            return False
        gh = send_fut.result()
        if gh is None or not gh.accepted:
            self.get_logger().error(f'[{label}] goal rejected')
            return False
        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=EXEC_TIMEOUT_SEC)
        if not res_fut.done():
            # Cancel, or the goal keeps running on the server while the next command
            # starts.
            self.get_logger().error(
                f'[{label}] no result within {EXEC_TIMEOUT_SEC:.0f} s — cancelling')
            cancel_fut = gh.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_fut, timeout_sec=5.0)
            return False
        ok = res_fut.result().result.error_code.val == MOVEIT_SUCCESS
        self.get_logger().info(f'[{label}] {"OK" if ok else "FAILED"}')
        return ok

    def _wait_for_move_group(self, timeout=60.0):
        self.get_logger().info('waiting for move_group…')
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._move.wait_for_server(timeout_sec=2.0):
                self.get_logger().info('move_group connected')
                return True
        return False

    def go_named(self, name):
        if name not in NAMED_STATES:
            return False
        self.get_logger().info(f'[{name}] moving to named pose...')
        return self._send(self._joint_constraints(NAMED_STATES[name]), name)

    def _go_named_safe(self, name):
        """go_named() with one retry and a logged result. Used before the base moves:
        if the arm is still down at the box, driving away drags the gripper through it."""
        if self.go_named(name):
            return True
        self.get_logger().warn(f'[{name}] first attempt failed — retrying once before driving')
        if self.go_named(name):
            return True
        self.get_logger().error(
            f'[{name}] FAILED after retry — proceeding anyway, but the arm may not '
            f'be clear. This is exactly the condition that drags a placed/held box.')
        return False

    def go_pose(self, x, y, z, q, label):
        self.get_logger().info(f'[{label}] planning…')
        return self._send(self._pose_constraints(x, y, z, q), label)

    def solve_ik_seeded(self, x, y, z, q, label, timeout=1.0, quiet=False):
        """One IK solve at (x, y, z, q), seeded from the arm's actual configuration.

        Returns a joint list, or None. Seeding from where the arm already is keeps the
        answer in the same posture when the solver converges from the seed.
        """
        if self._arm_actual is None:
            self.get_logger().warn(f'[{label}] no /joint_states yet — cannot seed IK')
            return None
        if not self._ik.service_is_ready():
            self._ik.wait_for_service(timeout_sec=2.0)
        if not self._ik.service_is_ready():
            self.get_logger().warn(f'[{label}] /compute_ik unavailable')
            return None

        req = GetPositionIK.Request()
        req.ik_request.group_name    = ARM_GROUP
        req.ik_request.ik_link_name  = TCP_LINK
        req.ik_request.avoid_collisions = True
        # ik_request.timeout is a builtin_interfaces/Duration message; `Duration` here
        # is rclpy's class, so convert it.
        req.ik_request.timeout = Duration(seconds=timeout).to_msg()
        req.ik_request.robot_state.joint_state.name     = list(ARM_JOINTS)
        req.ik_request.robot_state.joint_state.position = list(self._arm_actual)
        # is_diff: this joint_state names only the 6 arm joints. Without is_diff MoveIt
        # takes it as a complete state and sets every other joint to 0.
        req.ik_request.robot_state.is_diff = True
        ps = req.ik_request.pose_stamped
        ps.header.frame_id = PLANNING_FRAME
        ps.pose.position = Point(x=float(x), y=float(y), z=float(z))
        (ps.pose.orientation.x, ps.pose.orientation.y,
         ps.pose.orientation.z, ps.pose.orientation.w) = (float(v) for v in q)

        fut = self._ik.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        if not fut.done() or fut.result() is None:
            self.get_logger().warn(f'[{label}] IK service gave no answer')
            return None
        res = fut.result()
        if res.error_code.val != MOVEIT_SUCCESS:
            self.get_logger().warn(
                f'[{label}] IK failed (error_code {res.error_code.val})')
            return None
        try:
            names = list(res.solution.joint_state.name)
            sol = [res.solution.joint_state.position[names.index(j)]
                   for j in ARM_JOINTS]
        except (ValueError, IndexError):
            self.get_logger().warn(f'[{label}] IK answer missing arm joints')
            return None

        if not quiet:
            delta = max(abs(a - b) for a, b in zip(sol, self._arm_actual))
            self.get_logger().info(
                f'[{label}] IK solved, largest joint change from current: {delta:.4f} rad')
        return sol

    def descend_straight_until_resistance(self, x, y, q_unused, z_start, z_floor, label,
                                          yaws=None):
        """Place descent on a pre-planned straight line, from z_start down to z_floor.

        The whole descent is planned once by plan_straight_approach(), which holds one
        arm posture from top to bottom. (Planning every step separately let the arm
        wander between postures.) The open air is flown as one trajectory; only the last
        DESCEND_SLOW_BAND is stepped.

        Returns (reached_z, chain, i_stop) -- chain[0] is the bottom, chain[-1] the top,
        and chain[i_stop] is where the arm stopped, so the caller can retract back up the
        same line. Returns (None, None, 0) if no straight line exists or the fast part
        fails, and (None, chain, i_stop) if a step fails; then the box is still held."""
        rise = max(APPROACH_STEP, z_start - z_floor)
        # yaws: the place must reuse the wrist angle the box is held in. The two
        # GRASP_YAWS differ by 180 deg, so a free choice could start with a wrist flip.
        plan = self.plan_straight_approach(x, y, z_floor, label=f'{label} line', rise=rise,
                                           yaws=yaws)
        if plan is None:
            return None, None, 0
        _, chain = plan                       # chain[0] = at z_floor, chain[-1] = at z_start
        dz = rise / (len(chain) - 1) if len(chain) > 1 else rise

        # Most of the descent is open air, and stepping all of it costs a second per
        # 5 mm. Fly the air as one trajectory and step only the last DESCEND_SLOW_BAND.
        i_stop = len(chain) - 1
        i_slow = min(i_stop, int(math.ceil(DESCEND_SLOW_BAND / dz)) if dz > 0 else 0)
        fast = list(reversed(chain[i_slow:i_stop]))
        if fast:
            if not self._run_joint_path(fast, f'{label} fast'):
                # Not contact: this segment ends DESCEND_SLOW_BAND above the touch height,
                # so a failure is the trajectory not executing. Abort holding the box.
                self.get_logger().error(
                    f'[{label}] fast descent to z={z_floor + i_slow * dz:.4f} did not '
                    f'complete — this segment is still open air, so it is NOT contact. '
                    f'Aborting the descent with the box still held.')
                return None, None, 0
            i_stop = i_slow
            self._spin_for(0.15)

        # Walk down the rest of the planned line one step at a time.
        for i in range(i_stop - 1, -1, -1):
            z_next = z_floor + i * dz
            if not self._run_joint_path([chain[i]], f'{label} feel'):
                # The simulated arm is position-controlled and never stalls on contact, so
                # a failed step is an execution failure: stop, box still held; the caller
                # retracts.
                self.get_logger().error(
                    f'[{label}] step to z={z_next:.4f} did not complete — an execution '
                    f'failure, not contact. Aborting the descent with the box still held.')
                return None, chain, i_stop
            i_stop = i
            self._spin_for(0.15)
        else:
            # The normal case: z_floor is the computed touch height.
            self.get_logger().info(
                f'[{label}] reached the computed touch height z={z_floor:.4f} '
                f'without an earlier stall')
        return z_floor + i_stop * dz, chain, i_stop

    def add_table(self, bx, by, bz):
        """Register the pickup table as a MoveIt collision object."""
        table_top = bz - BOX_HALF_H
        co = CollisionObject()
        co.header.frame_id = PLANNING_FRAME
        co.id = 'pickup_table'
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = list(TABLE_SIZE)
        pose = Pose()
        pose.position = Point(x=float(bx), y=float(by),
                              z=float(table_top - TABLE_SIZE[2] / 2.0))
        pose.orientation.w = 1.0
        co.primitives.append(prim)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(co)
        if self._scene.wait_for_service(timeout_sec=5.0):
            fut = self._scene.call_async(ApplyPlanningScene.Request(scene=scene))
            rclpy.spin_until_future_complete(self, fut)
            self.get_logger().info(f'added table collision @ z={pose.position.z:.3f}')

    def _place_table_collision(self, add):
        """Place table as a MoveIt collision box, so the path to the hover pose does not
        press the arm into the table (that once levered the base up onto its nose)."""
        c = getattr(self, '_place_table_centre', None)
        if add and c is None:
            return
        co = CollisionObject()
        co.header.frame_id = PLANNING_FRAME
        co.id = 'place_table'
        if add:
            side = float(self.get_parameter('place_table_side').value)
            top = float(self._place_surface_base_z)
            height = float(self.get_parameter('place_table_top_z_world').value)
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.BOX
            prim.dimensions = [side, side, height]
            pose = Pose()
            pose.position = Point(x=float(c[0]), y=float(c[1]), z=top - height / 2.0)
            pose.orientation.w = 1.0
            co.primitives.append(prim)
            co.primitive_poses.append(pose)
            co.operation = CollisionObject.ADD
        else:
            co.operation = CollisionObject.REMOVE
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(co)
        if not add:
            st = CollisionObject()
            st.header.frame_id = PLANNING_FRAME
            st.id = 'place_stack'
            st.operation = CollisionObject.REMOVE
            scene.world.collision_objects.append(st)
        if self._scene.wait_for_service(timeout_sec=5.0):
            fut = self._scene.call_async(ApplyPlanningScene.Request(scene=scene))
            rclpy.spin_until_future_complete(self, fut)
            self.get_logger().info(
                f'{"added" if add else "removed"} place table collision'
                + (f' @ ({c[0]:.3f}, {c[1]:+.3f})' if add else ''))

    def _place_stack_collision(self, x, y, bottom, top):
        """The cubes already stacked, as a MoveIt box (cube footprint, table top -> stack top)."""
        if top - bottom < 0.005:
            return
        co = CollisionObject()
        co.header.frame_id = PLANNING_FRAME
        co.id = 'place_stack'
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [BOX_HEIGHT, BOX_HEIGHT, float(top - bottom)]
        pose = Pose()
        pose.position = Point(x=float(x), y=float(y), z=float((top + bottom) / 2.0))
        pose.orientation.w = 1.0
        co.primitives.append(prim)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(co)
        if self._scene.wait_for_service(timeout_sec=5.0):
            fut = self._scene.call_async(ApplyPlanningScene.Request(scene=scene))
            rclpy.spin_until_future_complete(self, fut)
            self.get_logger().info(f'added stack collision up to z={top:.3f}')

    def remove_table(self):
        co = CollisionObject()
        co.header.frame_id = PLANNING_FRAME
        co.id = 'pickup_table'
        co.operation = CollisionObject.REMOVE
        scene = PlanningScene(is_diff=True)
        scene.world.collision_objects.append(co)
        if self._scene.wait_for_service(timeout_sec=5.0):
            fut = self._scene.call_async(ApplyPlanningScene.Request(scene=scene))
            rclpy.spin_until_future_complete(self, fut)
            self.get_logger().info('removed table collision → retreat can plan freely')

    def _true_boxes_in_base(self):
        """Gazebo's true centre of every box, expressed in base_link: {name: ndarray(3)}."""
        out = {}
        tw = self._true_world_tf('base_link')
        if tw is None:
            return out
        t, R = tw
        names = list(dict.fromkeys(self._candidate_box_names() +
                                   ['stack_box_0', 'stack_box_1', 'stack_box_2']))
        for name in names:
            req = GetEntityState.Request()
            req.name = name
            req.reference_frame = 'world'
            fut = self._get_state.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
            if fut.done() and fut.result() is not None and fut.result().success:
                p = fut.result().state.pose.position
                out[name] = R.T @ (np.array([p.x, p.y, p.z]) - t)
        return out

    def _log_perception_vs_truth(self, tx, ty, tz):
        """Camera estimate vs the true centre of the box it is aiming at, in base_link.

        The box it is aiming at = the one whose true centre is nearest the estimate
        (the boxes stand ~10 cm apart). error = camera - truth, mm;
        + = camera places the box further forward / further left / higher."""
        boxes = self._true_boxes_in_base()
        if not boxes:
            self.get_logger().warn('[perception check] Gazebo box poses unavailable')
            return
        cam = np.array([tx, ty, tz])
        name = min(boxes, key=lambda k: np.linalg.norm(boxes[k] - cam))
        tb = boxes[name]
        err = cam - tb
        self.get_logger().info(
            f'[perception check] base_link, box centre: camera ({tx:.3f}, {ty:+.3f}, {tz:.3f}) '
            f'vs {name} truth ({tb[0]:.3f}, {tb[1]:+.3f}, {tb[2]:.3f}) -> error '
            f'x {err[0] * 1000:+.1f} mm, y {err[1] * 1000:+.1f} mm, z {err[2] * 1000:+.1f} mm')
        others = ', '.join(f'{k} ({v[0]:.3f}, {v[1]:+.3f})' for k, v in boxes.items() if k != name)
        if others:
            self.get_logger().info(f'[perception check] other boxes: {others}')

    def _log_tcp_at_grasp(self, tx, ty, tz):
        """At the bottom of the approach, before the fingers close: where the gripper
        really is vs where it was sent (arm error) vs the nearest box's true centre
        (total miss). base_link, mm, + = further forward / left / higher."""
        try:
            tf = self._tf_buf.lookup_transform('base_link', TCP_LINK, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f'[at grasp] TF base_link→{TCP_LINK} unavailable: {e}')
            return
        tcp = np.array([tf.transform.translation.x, tf.transform.translation.y,
                        tf.transform.translation.z])
        cmd = np.array([tx, ty, tz])
        arm = tcp - cmd
        msg = (f'[at grasp] base_link: tool at ({tcp[0]:.3f}, {tcp[1]:+.3f}, {tcp[2]:.3f}), '
               f'sent to ({tx:.3f}, {ty:+.3f}, {tz:.3f}) -> arm error x {arm[0] * 1000:+.1f}, '
               f'y {arm[1] * 1000:+.1f}, z {arm[2] * 1000:+.1f} mm')
        boxes = self._true_boxes_in_base()
        if boxes:
            name = min(boxes, key=lambda k: np.linalg.norm(boxes[k] - tcp))
            miss = tcp - boxes[name]
            msg += (f'; tool vs {name} true centre: x {miss[0] * 1000:+.1f}, '
                    f'y {miss[1] * 1000:+.1f}, z {miss[2] * 1000:+.1f} mm')
        self.get_logger().info(msg)

    # ── Straight approach: our own small-step solver ─────────────────────────
    def _fk(self, joints):
        """Gripper TCP pose in PLANNING_FRAME for the given arm joints, via /compute_fk.
        Returns (position ndarray(3), quaternion (x, y, z, w)) or None."""
        if not self._fk_cli.wait_for_service(timeout_sec=2.0):
            return None
        req = GetPositionFK.Request()
        req.header.frame_id = PLANNING_FRAME
        req.fk_link_names = [TCP_LINK]
        rs = RobotState()
        rs.joint_state.name = list(ARM_JOINTS)
        rs.joint_state.position = [float(v) for v in joints]
        req.robot_state = rs
        fut = self._fk_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        res = fut.result() if fut.done() else None
        if res is None or res.error_code.val != MOVEIT_SUCCESS or not res.pose_stamped:
            return None
        pose = res.pose_stamped[0].pose
        return (np.array([pose.position.x, pose.position.y, pose.position.z]),
                (pose.orientation.x, pose.orientation.y, pose.orientation.z,
                 pose.orientation.w))

    @staticmethod
    def _pose_error(p, o, p_goal, q_goal):
        """6-vector: position error (m) and orientation error (rotation vector, rad)."""
        o_inv = (-o[0], -o[1], -o[2], o[3])
        return np.concatenate([p_goal - p, _rotvec(_qmul(q_goal, o_inv))])

    def _ik_step(self, start, p_goal, q_goal, iters=10, tol_p=0.0005,
                 tol_r=math.radians(0.5), eps=1e-4, damping=0.01):
        """Smallest joint change from `start` that puts the TCP at (p_goal, q_goal).

        Damped least squares on a numerical Jacobian. Starting from the given posture
        and moving only as far as needed keeps the arm in one posture, which KDL's
        random restarts do not do near the reach limit."""
        q = np.array(start, dtype=float)
        for _ in range(iters):
            f = self._fk(q)
            if f is None:
                return None
            p, o = f
            e = self._pose_error(p, o, p_goal, q_goal)
            if np.linalg.norm(e[:3]) < tol_p and np.linalg.norm(e[3:]) < tol_r:
                return q
            J = np.zeros((6, 6))
            for i in range(6):
                qi = q.copy()
                qi[i] += eps
                fi = self._fk(qi)
                if fi is None:
                    return None
                pi, oi = fi
                J[:3, i] = (pi - p) / eps
                J[3:, i] = _rotvec(_qmul(oi, (-o[0], -o[1], -o[2], o[3]))) / eps
            dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
            dq = np.clip(dq, -0.1, 0.1)
            q = np.array([min(max(v, lo), hi) for v, (lo, hi) in zip(q + dq, ARM_LIMITS)])
        f = self._fk(q)
        if f is None:
            return None
        e = self._pose_error(f[0], f[1], p_goal, q_goal)
        return q if (np.linalg.norm(e[:3]) < tol_p and np.linalg.norm(e[3:]) < tol_r) else None

    def plan_straight_approach(self, tx, ty, tz, label='straight approach', rise=None,
                               yaws=None):
        """Plan a truly straight vertical approach BACKWARDS from the grasp.

        For each wrist option (the two 180-degree twins) and a few sampled grasp
        postures: start AT the grasp and walk up PREGRASP_RISE in APPROACH_STEP steps,
        each the smallest joint change from the previous posture. A step that needs
        more than STEP_MAX_JOINT on any joint is a posture switch -> that candidate is
        dropped. Returns (q_target, chain) with chain[0] = grasp ... chain[-1] =
        pre-grasp, choosing the candidate with the least joint travel; or None, in
        which case the caller must NOT fall back to a swinging plan."""
        # rise defaults to the grasp's PREGRASP_RISE; the place path passes its own
        # (hover is further above the table than a pre-grasp is above a box).
        rise = PREGRASP_RISE if rise is None else float(rise)
        n = max(1, int(round(rise / APPROACH_STEP)))
        best = None
        for yaw in (GRASP_YAWS if yaws is None else yaws):
            q_target = _spin_about_vertical(TOPDOWN_QUAT, yaw)
            tag = f'{label} yaw {math.degrees(yaw):+.0f}°'
            tried = []
            for _ in range(GRASP_IK_CANDIDATES):
                g = self.solve_ik_seeded(tx, ty, tz, q_target, tag, quiet=True)
                if g is None:
                    continue
                g = np.array(g, dtype=float)
                if any(np.max(np.abs(g - t)) < 0.05 for t in tried):
                    continue                       # same posture as one already tried
                tried.append(g)
                chain, reason = [g], ''
                for k in range(1, n + 1):
                    nxt = self._ik_step(chain[-1], np.array([tx, ty, tz + k * APPROACH_STEP]),
                                        q_target)
                    if nxt is None:
                        reason = f'no solution {k * APPROACH_STEP * 1000:.0f} mm up'
                        break
                    jump = float(np.max(np.abs(nxt - chain[-1])))
                    if jump > STEP_MAX_JOINT:
                        reason = (f'posture switch {k * APPROACH_STEP * 1000:.0f} mm up '
                                  f'({jump:.2f} rad on one joint)')
                        break
                    chain.append(nxt)
                if reason:
                    self.get_logger().info(f'[{tag}] candidate {len(tried)}: {reason}')
                    continue
                travel = sum(float(np.sum(np.abs(b - a))) for a, b in zip(chain, chain[1:]))
                self.get_logger().info(
                    f'[{tag}] candidate {len(tried)}: straight line OK, {n} steps, '
                    f'joint travel {travel:.3f} rad ({travel / rise:.1f} rad/m)')
                if best is None or travel < best[2]:
                    best = (q_target, chain, travel, yaw)
        if best is None:
            self.get_logger().error(
                f'[{label}] no straight vertical line found at ({tx:.3f}, {ty:+.3f}, {tz:.3f}) '
                f'for either wrist option')
            return None
        self.get_logger().info(
            f'[{label}] chosen: wrist yaw {math.degrees(best[3]):+.0f}°, '
            f'{best[2] / rise:.1f} rad/m')
        self._last_chosen_yaw = best[3]
        return best[0], best[1]

    def _run_joint_path(self, configs, label, sec_per_point=0.5):
        """Execute a chain of nearby joint configurations as ONE trajectory.

        Starts from the arm's measured joints so MoveIt's start-state check passes;
        consecutive points are a few mm apart, so the controller's interpolation
        between them keeps the TCP on the planned straight line."""
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.05)
        traj = RobotTrajectory()
        jt = traj.joint_trajectory
        jt.joint_names = list(ARM_JOINTS)
        pts = ([list(self._arm_actual)] if self._arm_actual is not None else []) + \
              [list(c) for c in configs]
        # Each segment takes at least as long as its largest joint move needs at
        # PATH_JOINT_SPEED, so a large first move (measured posture -> first point of
        # the line) does not snap.
        t = 0.0
        first_move = peak = 0.0
        for i, c in enumerate(pts):
            if i > 0:
                move = max(abs(a - b) for a, b in zip(c, pts[i - 1]))
                dt = max(1.0 if i == 1 else sec_per_point, move / PATH_JOINT_SPEED)
                first_move = move if i == 1 else first_move
                peak = max(peak, move / dt)
                t += dt
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in c]
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int(round((t - int(t)) * 1e9))
            jt.points.append(pt)
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj
        if not self._exec.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f'[{label}] /execute_trajectory unavailable')
            return False
        send_fut = self._exec.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut, timeout_sec=15.0)
        gh = send_fut.result() if send_fut.done() else None
        if gh is None or not gh.accepted:
            self.get_logger().error(f'[{label}] trajectory rejected')
            return False
        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=t + 20.0)
        if not res_fut.done():
            # Stop the arm before anyone acts on the failure.
            self.get_logger().error(f'[{label}] no result after {t + 20.0:.0f} s -- cancelling')
            rclpy.spin_until_future_complete(self, gh.cancel_goal_async(), timeout_sec=3.0)
        ok = (res_fut.done() and
              res_fut.result().result.error_code.val == MOVEIT_SUCCESS)
        self.get_logger().info(f'[{label}] {"OK" if ok else "FAILED"} '
                               f'({len(configs)} points, {t:.1f} s; first move {first_move:.2f} rad, '
                               f'peak {peak:.2f} rad/s)')
        return ok

    def grasp_and_retract(self, bx, by, bz):
        self.get_logger().info('=== Steps 4–5: Iterative top-down grasp ===')
        q = TOPDOWN_QUAT
        HOVER      = PREGRASP_RISE   # 6 cm, as on the real robot
        GRASP_Z    = GRASP_ABOVE

        if not self._wait_for_move_group():
            return False
        if not self.go_named('ready'):
            return False
        self.set_gripper(GRIPPER_OPEN, 'open')

        # add_table() registers the pickup table as a collision object so planning
        # respects it during the descent; remove_table() (below) undoes this before
        # the lift so the retreat is not blocked by a phantom collision.
        tx, ty, tz = bx, by, bz
        self._log_perception_vs_truth(tx, ty, tz)
        self.add_table(bx, by, bz)

        # ── straight vertical approach, planned backwards ───────────────────────
        # Near the reach limit KDL answers nearby points with different arm postures;
        # the arm then swings between them and a finger pushes the cube.
        # plan_straight_approach() keeps one posture along the whole line, or reports
        # that none exists. There is no swinging fallback on purpose.
        plan = self.plan_straight_approach(tx, ty, tz + GRASP_Z)
        if plan is not None:
            self._grasp_yaw = getattr(self, '_last_chosen_yaw', None)
        if plan is None:
            self.get_logger().error('grasp: no straight approach at this point — aborting')
            return False
        q, approach = plan           # q = chosen fingers-left-right orientation
        if not self._send(self._joint_constraints(list(approach[-1])),
                          f'pre-grasp ({PREGRASP_RISE * 100:.0f} cm above)'):
            self.get_logger().error('pre-grasp pose failed — aborting')
            return False
        if not self._run_joint_path(approach[::-1], 'straight approach down'):
            self.get_logger().error('straight approach failed — aborting')
            return False
        self._log_tcp_at_grasp(tx, ty, tz)

        # Remove table collision so the lift is not blocked by a phantom
        # table-gripper collision. Done once, before the attempt loop.
        self.remove_table()

        # ── close → lift → verify loop ──────────────────────────────────────
        # A retry means "we lifted and the box was still on the table".
        verified = False
        attempt = 0
        for attempt in range(1, MAX_GRASP_RETRIES + 2):
            if attempt > 1:
                self.get_logger().warn(
                    f'[grasp] not verified — retry {attempt - 1}/{MAX_GRASP_RETRIES}')
                self.set_gripper(GRIPPER_OPEN, 'open')
                # Same straight line: from wherever the lift left the arm to the
                # pre-grasp, then down the planned line again.
                if not self._run_joint_path(approach[::-1], f'straight retry {attempt - 1}'):
                    break

            # Heights of all candidate boxes before the close. The check after the
            # lift takes whichever one rose -- that is the box in the gripper -- so it
            # does not depend on guessing the target right.
            z_before = self._all_box_world_z()

            self.close_gripper()

            # Let the arm settle and /joint_states catch up before the lift is
            # planned; a stale start state makes MoveIt reject the trajectory.
            for _ in range(5):
                rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.3)

            # Lift straight up the planned line, in the same posture as the approach.
            lifted = self._run_joint_path(approach[1:], f'straight lift {HOVER * 100:.0f} cm')
            if not lifted:
                self.get_logger().warn('lift attempts failed — continuing anyway')

            # Did the box leave the table? Gazebo ground truth, checked before the
            # retract to home.
            if not lifted:
                verified = False
            else:
                verified = self._verify_grasp_truth(z_before)
                if verified is None:
                    self.get_logger().warn('[verify] Gazebo state unavailable — counted as failed')
                    verified = False

            # _weld_active = "the box is held", an observation: it measurably left the
            # table. It gates the place step and feeds pick_success in the metrics CSV.
            self._weld_active = bool(verified)
            # Measure centring while the box is held up, the only moment it is rigidly
            # located relative to the gripper. A grip far off centre would not hold on
            # a real gripper, so it counts as failed.
            unrealistic = False
            if verified:
                c = self._measure_grasp_centering()
                if c is not None and (abs(c[0]) >= GRASP_JAW_TOL
                                      or abs(c[1]) >= FINGER_FACE_HALF):
                    self.get_logger().error(
                        f'[grasp] NOT A REAL GRIP — box {c[0] * 1000:+.1f} mm between the jaws '
                        f'(limit {GRASP_JAW_TOL * 1000:.0f}), {c[1] * 1000:+.1f} mm along the faces '
                        f'(limit {FINGER_FACE_HALF * 1000:.0f}). The sim plugin attached it, a real '
                        f'gripper would not hold it: counted as FAILED, no retry.')
                    verified = False
                    unrealistic = True
                    self._weld_active = False

            self._log_grasp_attempt(attempt, verified)
            if verified:
                self.get_logger().info(f'[grasp] VERIFIED on attempt {attempt}')
                break
            if unrealistic:
                break

        if hasattr(self, '_current_cycle'):
            # pick_success only trusts an explicit True — None (inconclusive) or
            # False both leave it False, since neither is confirmed success.
            self._current_cycle['pick_success'] = bool(verified)

        if not verified:
            self.get_logger().error(
                f'[grasp] failed after {attempt} attempts (the box did not leave '
                f'the table) — aborting')
            self.set_gripper(GRIPPER_OPEN, 'open')
            self.go_named('ready')
            return False

        # Retract to home
        self.go_named('home')
        self.get_logger().info('=== Grasp + Retract complete ===')
        return True

    # ── Step 5b: Place onto the stack ───────────────────────────────────────────

    def back_up(self):
        self.get_logger().info('=== Step 6: Backing up ===')
        twist = Twist()
        twist.linear.x = BACKUP_VEL_X
        end = time.time() + BACKUP_SEC
        while time.time() < end:
            self._cmd_vel_pub.publish(twist)
            rclpy.spin_once(self, timeout_sec=0.1)
        # Stop
        self._cmd_vel_pub.publish(Twist())
        self.get_logger().info(f'backed up ~{abs(BACKUP_VEL_X) * BACKUP_SEC:.2f} m ✓')

    # ── Tag-dock helpers ──────────────────────────────────────────────────────

    def _read_tag_live(self):
        """One fresh /place_tag_pose reading: (x, y, range, bearing, yaw) or None."""
        rclpy.spin_once(self, timeout_sec=0.05)
        p = self._latest_tag_pose
        if p is None or p.header.frame_id != PLANNING_FRAME:
            return None
        x, y = p.pose.position.x, p.pose.position.y
        tag_range = math.hypot(x, y)
        if tag_range < 0.05 or tag_range > 3.0:
            return None
        bearing = math.atan2(y, x)
        q = p.pose.orientation
        # Heading of the tag's face normal (+Z, toward the robot) in base_link: the third
        # column of the rotation matrix. A squarely facing robot reads pi.
        yaw = math.atan2(2.0 * (q.y * q.z - q.w * q.x),
                         2.0 * (q.x * q.z + q.w * q.y))
        return x, y, tag_range, bearing, yaw

    # ── Tag-based dock controller ─────────────────────────────────────────────

    @staticmethod
    def _wrap(a):
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _rotate_by(self, dyaw, timeout=8.0):
        """Turn in place by `dyaw` radians (signed), closed-loop on odometry yaw.

        The two pivots of a sideways step (_crab_sideways) must cancel: whatever the
        first one gets wrong is left as heading error after the second. Closing the
        loop on odometry, instead of timing the turn, keeps that error small.
        """
        if self._odom_yaw is None:
            rclpy.spin_once(self, timeout_sec=0.5)
        if self._odom_yaw is None:
            self.get_logger().warn('[dock B] no odometry yaw — cannot pivot precisely')
            return False
        if not self._turn_clear():
            return False

        target   = self._wrap(self._odom_yaw + dyaw)
        deadline = time.time() + timeout
        TOL      = 0.02   # rad ≈ 1.1°
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._odom_yaw is None:
                continue
            err = self._wrap(target - self._odom_yaw)
            if abs(err) < TOL:
                self._cmd_vel_pub.publish(Twist())
                time.sleep(0.25)
                return True
            cmd = Twist()
            w = max(-PLACE_DOCK_CRAB_ROT, min(PLACE_DOCK_CRAB_ROT, 2.0 * err))
            if abs(w) < 0.25:          # minimum turn rate, so the base does not stall
                w = math.copysign(0.25, w)
            cmd.angular.z = w
            self._cmd_vel_pub.publish(cmd)
        self._cmd_vel_pub.publish(Twist())
        time.sleep(0.25)
        self.get_logger().warn(f'[dock B] pivot of {math.degrees(dyaw):+.0f}° timed out')
        return False

    def _drive_straight(self, dist, speed=PLACE_DOCK_CRAB_FWD, timeout=15.0):
        """Drive forward `dist` metres (positive only), closed-loop on odom distance."""
        if self._odom is None:
            rclpy.spin_once(self, timeout_sec=0.5)
        if self._odom is None:
            self.get_logger().warn('[dock B] no odometry — cannot drive a measured leg')
            return False

        x0, y0   = self._odom
        deadline = time.time() + timeout
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            gone = self._odom_dist_since(x0, y0)
            if gone is None:
                continue
            if gone >= dist:
                self._cmd_vel_pub.publish(Twist())
                time.sleep(0.25)
                return True
            cmd = Twist()
            cmd.linear.x = speed
            self._cmd_vel_pub.publish(cmd)
        self._cmd_vel_pub.publish(Twist())
        time.sleep(0.25)
        self.get_logger().warn(f'[dock B] straight leg of {dist:.3f} m timed out')
        return False

    # ── Place: navigate to second table ──────────────────────────────────────

    def navigate_to_place_table(self, level=0):
        """Nav2 to the pre-dock position, then creep forward to place distance."""
        self.get_logger().info('=== Place Step 1: Navigate to place table ===')
        self._activate_nav2_cmdvel()

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x = self._nav_place_x
        goal.pose.pose.position.y = self._nav_place_y
        half = self._nav_place_yaw / 2.0
        goal.pose.pose.orientation.z = math.sin(half)
        goal.pose.pose.orientation.w = math.cos(half)

        gh = None
        for attempt in range(15):
            send_fut = self._nav.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_fut, timeout_sec=10.0)
            if send_fut.done():
                gh = send_fut.result()
                if gh is not None and gh.accepted:
                    self.get_logger().info('Nav2 goal accepted — driving to place table…')
                    break
            self.get_logger().warn(f'Place nav goal rejected (attempt {attempt + 1}/15) — retrying…')
            end = self.get_clock().now() + Duration(seconds=2.0)
            while self.get_clock().now() < end:
                rclpy.spin_once(self, timeout_sec=0.1)
        else:
            self.get_logger().error('Could not reach place table — aborting place')
            return False

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut, timeout_sec=NAV_TIMEOUT_SEC)
        if not res_fut.done() or res_fut.result().status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error('Navigation to place table failed')
            return False

        self.get_logger().info('Nav2 arrived — docking to place table…')
        self._event('nav_done', leg='place')
        self._held_diag('Nav2 transit to the place table')

        # face_and_pick: no Nav2 sideways realign, no tag-angle squaring, no map-position
        # fallback. Station heading, tag centre y, then the lidar (or the stack look).
        self._silence_nav2_cmdvel()
        time.sleep(0.5)
        self._docked_via_tag = self._match_place_dock(level)
        if not self._docked_via_tag:
            self.get_logger().error('[match] place dock failed — the box stays in the gripper')
            return False
        self.get_logger().info(f'Place dock ready — arm target locked {self._latched_drop}')
        self._event('dock_done', drop=self._latched_drop, via_tag=True)
        return True

    # ── Place: lower box onto table and release ───────────────────────────────

    def place_box(self, level: int = 0):
        """Move arm to place the held box on the place table at the given stack level.

        Sequence: ready → hover above table → descend to surface → release → lift → home.
        Arm target (px, py) = self._latched_drop set by navigate_to_place_table().
        Heights use the same calibrated stack_surface_z as the grasp path.
        """
        self.get_logger().info(f'=== Place Step 2: Place box at level {level} ===')

        # Ensure MoveIt is alive after the navigation pause
        self._arm_clear = True
        if not self._wait_for_move_group():
            self.get_logger().error('MoveIt not ready — aborting the place, the box stays held')
            return False

        px, py = self._latched_drop if self._latched_drop is not None else (0.22, 0.0)
        # Clamp to the arm's reach (not to the pickup stop distance, which is shorter
        # and would put the box on the table's edge).
        if px > self._arm_reach:
            self.get_logger().warn(
                f'[place] drop x {px:.3f} exceeds arm_reach {self._arm_reach:.3f} — clamping')
            px = self._arm_reach
        # Hover in the same wrist orientation the box is held in, so the descent does
        # not start with a large wrist turn while carrying the box.
        held_yaw = getattr(self, '_grasp_yaw', None)
        q  = TOPDOWN_QUAT if held_yaw is None else _spin_about_vertical(TOPDOWN_QUAT, held_yaw)

        # The place table's surface height, from config/scene.yaml (via
        # nav_pick.launch.py). Not self._table_top_base_z, which is the pickup
        # table's; the two tables differ by 4 cm in height.
        surface_z = self._place_surface_base_z
        self.get_logger().info(f'[place] surface_z={surface_z:.4f} (scene-derived)')

        if level == 0:
            # Nothing stacked yet -- the calibrated table height is the ground truth,
            # there is nothing to measure.
            rest_surface = surface_z
        else:
            # face_and_pick: STACK_SURFACE = table height + k x cube edge (arithmetic). The
            # camera gives the stack's x/y (_match_approach), not its height.
            rest_surface = surface_z + level * BOX_HEIGHT
            self.get_logger().info(f'[match] stack top L{level}: table + {level} x box = {rest_surface:.4f}')
            self._event('height_fused', level=level, sources={'geometry': rest_surface},
                        kept=['geometry'], fused=rest_surface)

        hover_z      = rest_surface + BOX_HALF_H + STACK_HOVER
        place_z      = rest_surface + BOX_HALF_H + GRASP_ABOVE + PLACE_RELEASE_GAP
        if level >= 1:
            # From the full stack hover there is often no straight line down (a posture
            # switch on the way). Start the straight descent closer, as the pick's
            # pre-grasp does.
            hover_z = place_z + STACK_HOVER_UPPER
            # Release a few mm above the estimated contact: without contact sensing (the
            # position-controlled arm never stalls) an estimate a few mm low means pressing
            # the stack; a few mm high only means a short drop.
            place_z += STACK_RELEASE_MARGIN
            # At level 2 even 6 cm can be too high. Plan-only check (no motion) from each
            # height in STACK_HOVER_TRIES; take the first that works.
            held_yaw_ = getattr(self, '_grasp_yaw', None)
            for h in STACK_HOVER_TRIES:
                if self.plan_straight_approach(
                        px, py, place_z, label=f'place L{level} line check {h * 100:.1f} cm',
                        rise=h, yaws=None if held_yaw_ is None else (held_yaw_,)) is not None:
                    hover_z = place_z + h
                    self.get_logger().info(
                        f'[place] L{level}: straight descent from {h * 100:.1f} cm above')
                    break
            else:
                self.get_logger().warn(
                    f'[place] L{level}: no straight descent from any of {STACK_HOVER_TRIES}')
            # The stack itself as an obstacle, so the move to the (now lower) hover keeps clear.
            self._place_stack_collision(px, py, surface_z, rest_surface)

        self.get_logger().info(
            f'Place arm: x={px} y={py} hover_z={hover_z:.3f} place_z={place_z:.3f}')

        self._place_table_collision(True)

        # ── go to ready pose first so the arm starts from a known configuration ──
        if not self.go_named('ready'):
            self.get_logger().warn('[place] ready failed — continuing anyway')

        # ── hover above the table surface ────────────────────────────────────────
        if not self.go_pose(px, py, hover_z, q, f'place hover L{level}'):
            # Never open the gripper here: the arm is not over the table, so a release
            # would drop the box wherever the hand is.
            self.get_logger().error(
                f'[place] hover unreachable at x={px:.3f} y={py:+.3f} — aborting the '
                f'place WITHOUT releasing. The box stays in the gripper.')
            self.go_named('ready')
            self._place_table_collision(False)
            return False

        # ── descend to the computed release height ───────────────────────────────
        # The descent stops at place_z. Going lower would push the held box into the
        # surface (contact in Gazebo is soft, the arm does not stall), and the box can
        # then shake loose.
        z_floor = place_z
        held_yaw = getattr(self, '_grasp_yaw', None)
        reached_z, place_chain, i_stop = self.descend_straight_until_resistance(
            px, py, q, z_start=hover_z, z_floor=z_floor, label=f'place set L{level}',
            yaws=None if held_yaw is None else (held_yaw,))
        if reached_z is None:
            # No straight line down, or a step of it failed. No swinging fallback on
            # purpose: it can knock the box off the table.
            self.get_logger().error(
                f'[place L{level}] descent failed at x={px:.3f} y={py:+.3f} '
                f'— aborting WITHOUT releasing. The box stays in the gripper.')
            if place_chain is not None and place_chain[i_stop + 1:]:
                # Part-way down: back up the same line before swinging to 'ready'.
                self._run_joint_path(place_chain[i_stop + 1:], f'place abort retract L{level}')
            self._go_named_safe('ready')
            return False
        if abs(reached_z - z_floor) < 1e-4:
            self.get_logger().info(
                f'[place] released at the computed height z={reached_z:.4f} '
                f'(no contact detected on the way down)')
        else:
            self.get_logger().info(
                f'[place] stopped early on resistance at z={reached_z:.4f} '
                f'(computed {place_z:.4f}, {(reached_z - place_z) * 1000:+.0f} mm)')

        # Release: open the fingers slowly (not a snap-open) so the box is set down
        # rather than dropped.
        self._event('release_command', level=level, target=[px, py, place_z], reached_z=reached_z,
                    rest_surface=rest_surface)
        time.sleep(0.2)
        self.set_gripper(GRIPPER_OPEN, 'release', duration=1.6, steps=32)
        self._place_table_collision(False)
        time.sleep(0.5)   # let box settle

        # Retract back up the line we came down: it is a checked single-posture
        # straight line, so the open gripper cannot swing into the placed box.
        up = place_chain[i_stop + 1:]         # empty only if contact came at the very top
        retracted = self._run_joint_path(up, f'place retract L{level}') if up else True
        if not retracted:
            self.get_logger().error(
                f'[place retract L{level}] FAILED — arm may still be down at the box. '
                f'Forcing home before anything else moves the base.')

        home_ok = self._go_named_safe('home')
        self._event('arm_clear', level=level, retracted=retracted, home_ok=home_ok)
        if not retracted and not home_ok:
            # The box is down, but the arm may still straddle it: the caller must not move
            # the base.
            self._arm_clear = False
            self.get_logger().error(
                f'[place L{level}] arm never confirmed clear of the box after release '
                f'-- the base will not move.')
        self.get_logger().info(f'=== Box placed at level {level} ✓ ===')
        return True

    # ── Main sequence ─────────────────────────────────────────────────────────

    def _backup_from_table(self, seconds: float):
        """Reverse at BACKUP_VEL_X for `seconds` to clear whichever table we're at."""
        _tw = Twist(); _tw.linear.x = BACKUP_VEL_X
        end = time.time() + seconds
        while time.time() < end:
            self._cmd_vel_pub.publish(_tw)
            rclpy.spin_once(self, timeout_sec=0.1)
        self._cmd_vel_pub.publish(Twist())
        time.sleep(0.3)

    def run(self):
        self.get_logger().info('══════════════════════════════════════════════')
        self.get_logger().info('  Nav-Pick Orchestrator — STARTING SEQUENCE   ')
        self.get_logger().info('══════════════════════════════════════════════')

        # 0. Travel pose — wait for MoveIt first, then fold arm DOWN before driving so
        #    joint2 (horizontal hinge) doesn't act as a pendulum during navigation.
        self._wait_for_move_group()
        self._arm.prepare_travel()

        # 1. Navigate to pickup table
        if not self.navigate_to_table():
            self.get_logger().error('Navigation failed — aborting')
            return

        time.sleep(0.5)

        # 1.25. Slide sideways to face the target box square-on. With three boxes on
        #       the table the outer ones sit 0.10 m off the staging centre line, and
        #       visual_docking() would turn that into a 7.8 degree diagonal approach.
        self._align_to_target_box()

        # 1.5. Visual docking
        if not self.visual_docking():
            self.get_logger().error('Approach failed — no grasp attempted')
            if not hasattr(self, '_current_cycle'):
                self._cycle_begin(0)              # the cycle record starts after docking
            self._current_cycle['pick_success'] = False
            self._cycle_end()
            self.flush_metrics()
            self.back_up()
            return

        # 2. (the base is not pinned: it stands on its own wheels, as on the robot)
        time.sleep(1.0)

        # 3. Grasp target (docked geometry in front of front bumper)
        base = self._dock_box if self._dock_box is not None else self.get_box_xyz()
        bx = self._stop_distance
        by = 0.0
        bz = -0.015
        if base is not None:
            # x comes from the dock's last measurement of the box (the camera reading
            # and the measured blind-hop correction), not from a fixed distance.
            bx = float(base[0])
            if bx > self._arm_reach or bx < 0.15:
                self.get_logger().warn(
                    f'[orchestrator] measured box x {bx:.3f} outside [0.15, '
                    f'{self._arm_reach:.3f}] — clamping. The dock ended somewhere it '
                    f'should not have; the grasp will be aimed short.')
                bx = max(0.15, min(self._arm_reach, bx))
            # Keep measured lateral Y and vertical Z within reachable physical envelope
            by = max(-0.05, min(0.05, float(base[1])))
            bz = max(-0.15, min(0.10, float(base[2])))
        base = (bx, by, bz)
        self.get_logger().info(f'[orchestrator] target calibrated for front reach → {base}')

        # ── FULL PIPELINE: pick → transport → dock → place (multi-level) ────────
        place_levels = max(1, int(self.get_parameter('place_stack_levels').value))

        # Order here is not the pick order. _claim_held_box() identifies which box is
        # actually held by GROUND TRUTH -- whichever one is elevated and nearest the
        # gripper -- so this list only has to name every box that exists. The pick
        # order is decided by box_pose_estimator's target_policy ('rightmost' for
        # multi-box, set in nav_pick.launch.py), which is deterministic: as boxes are
        # consumed the remaining ones present a new rightmost each cycle.
        self._remaining_boxes = ['stack_box_0', 'stack_box_1', 'stack_box_2']
        self._all_box_names = list(self._remaining_boxes)   # for the final stack check

        # Level 0: first pick (robot is already docked at pickup table)
        self._cycle_begin(0)
        ox, oy, oz = self._offset()
        bx, by, bz = base[0] + ox, base[1] + oy, base[2] + oz
        self.get_logger().info(
            f'grasp target = {tuple(round(v, 3) for v in base)} + offset '
            f'({ox:.3f}, {oy:.3f}, {oz:.3f}) → ({bx:.3f}, {by:.3f}, {bz:.3f})')
        self._arm.pick(bx, by, bz)
        self._current_cycle['pick_success'] = self._weld_active

        if not self._weld_active:
            self.get_logger().warn('Weld not active after grasp — skipping place step')
            self._cycle_end()
            self.flush_metrics()
            self.back_up()
            return

        self._current_cycle['box_name'] = self._claim_held_box()

        # Diagnostic checkpoints (log only, no effect on the pass/fail metric below):
        # if the box is lost before arrival, they show which motion lost it.
        held = self._box_is_held()
        self.get_logger().info(
            f'[diag] held after grasp_and_retract (incl. its internal home '
            f'retraction): {held}')

        # Transport: unpin, travel pose, clear pickup table
        self._arm.prepare_travel()
        held = self._box_is_held()
        self.get_logger().info(f"[diag] held after go_named('travel'): {held}")
        self.get_logger().info('Clearing pickup table — reversing 0.6 m before Nav2…')
        self._backup_from_table(4.0)
        held = self._box_is_held()
        self.get_logger().info(f'[diag] held after _backup_from_table: {held}')

        # Navigate to place table and dock (first time — sets _latched_drop)
        nav_ok = self.navigate_to_place_table()
        result_str = f'OK  drop={self._latched_drop}' if nav_ok else 'FAILED'
        self.get_logger().info(f'navigate_to_place_table → {result_str}')
        self._current_cycle['dock_via_tag'] = bool(nav_ok and self._docked_via_tag)
        # Measured at arrival (not departure), from Gazebo ground truth: did the box
        # survive the transport?
        self._current_cycle['transport_retained'] = self._check_transport_retained()

        if not nav_ok:
            self.get_logger().error('Place dock failed — aborting')
            self._cycle_end()
            self.flush_metrics()
            self.back_up()
            return

        # The box has to still be in the gripper for a place to mean anything.
        if not self._current_cycle['transport_retained']:
            self.get_logger().error(
                'Box was lost in transit — skipping the place sequence '
                '(nothing to put down)')
            self._cycle_end()
            self.flush_metrics()
            self.back_up()
            return


        # Place + multi-level stacking loop
        for lvl in range(place_levels):
            self.get_logger().info(f'Placing level {lvl}…')
            place_ok = self._arm.place(lvl).ok

            # Measure where the box actually landed (Gazebo ground truth, sim only)
            # before deciding success: place_box() returning True only means the arm's
            # motion executed, not where the box ended up.
            stack_top_z = self._place_table_top_z_world + lvl * BOX_HEIGHT + BOX_HALF_H
            self._cycle_measure_placement(self._table_map_x, self._table_map_y, stack_top_z,
                                          box_name=self._current_cycle['box_name'])

            if place_ok:
                ex = self._current_cycle.get('placement_err_x')
                ey = self._current_cycle.get('placement_err_y')
                ez = self._current_cycle.get('placement_err_z')
                # The errors start as NaN, not None, and every comparison with NaN is
                # False, so check for finite values explicitly.
                def _finite(v):
                    return v is not None and math.isfinite(v)
                if not (_finite(ex) and _finite(ey) and _finite(ez)):
                    self.get_logger().error(
                        '[place] ground truth unavailable — placement UNVERIFIED, '
                        'marking place_success=False')
                    place_ok = False
                elif _finite(ex) and _finite(ey):
                    err_xy = math.hypot(ex, ey)
                    # Table half-side is 0.04 m (scene.yaml place_table.side/2);
                    # 0.03 m leaves a small margin for a box still counted "placed"
                    # near the edge, while catching anything that actually missed.
                    PLACE_XY_TOL, PLACE_Z_TOL = 0.03, 0.03
                    if err_xy > PLACE_XY_TOL or (ez is not None and abs(ez) > PLACE_Z_TOL):
                        self.get_logger().error(
                            f'[place] arm motion completed but the box MISSED the '
                            f'table: err_xy={err_xy:.3f} m (tol {PLACE_XY_TOL}), '
                            f'err_z={ez} — marking place_success=False')
                        place_ok = False

            self._current_cycle['place_success'] = place_ok
            self._cycle_end()

            if not getattr(self, '_arm_clear', True):
                self.get_logger().error(f'arm not clear of the stack after level {lvl} — '
                                        f'stopping; the base does not move')
                break
            if not place_ok:
                self.get_logger().error(f'place_box failed at level {lvl} — stopping stack')
                break

            if lvl + 1 >= place_levels:
                break   # all levels done

            if not self._remaining_boxes:
                self.get_logger().warn('No boxes left to fetch — stopping stack')
                break

            # ── Fetch next box ───────────────────────────────────────────────
            self._cycle_begin(lvl + 1)
            self.get_logger().info(f'Fetching box for level {lvl + 1}…')
            self._arm.prepare_travel()
            # Back up ~1.1 m at the measured rate: after a shorter backup the robot is
            # still inside the table's inflated costmap zone and the global planner
            # finds no path from the start pose (inflation_radius 0.45 m,
            # nav2_limo_diff.yaml).
            self._backup_from_table(12.0)


            self._activate_nav2_cmdvel()
            nav_ok_fetch = self.navigate_to_table()
            if not nav_ok_fetch:
                # One retry: back up further, past whatever the local costmap reacts
                # to, then let Nav2 plan the whole route again.
                self.get_logger().warn(
                    'Re-navigation to pickup failed once — backing up further and '
                    'retrying a single time before giving up')
                self._backup_from_table(3.0)
                nav_ok_fetch = self.navigate_to_table()
            if not nav_ok_fetch:
                self.get_logger().error('Re-navigation to pickup failed — stopping stack')
                self._current_cycle['pick_success'] = False
                self._cycle_end()
                break

            # The outer boxes sit 0.10 m off the centre line; without this sideways
            # step the robot approaches them at ~8 deg and grasps rotated faces.
            self._align_to_target_box()
            if not self.visual_docking():
                self.get_logger().error('Approach failed for next level — stopping stack')
                self._current_cycle['pick_success'] = False
                self._cycle_end()
                break
            time.sleep(1.0)

            base2 = self._dock_box if self._dock_box else FIXED_BOX
            ox2, oy2, oz2 = self._offset()
            bx2, by2, bz2 = base2[0] + ox2, base2[1] + oy2, base2[2] + oz2
            self._arm.pick(bx2, by2, bz2)
            self._current_cycle['pick_success'] = self._weld_active

            if not self._weld_active:
                self.get_logger().error('Pick failed for next level — stopping stack')
                self._cycle_end()
                break

            self._current_cycle['box_name'] = self._claim_held_box()

            self._arm.prepare_travel()
            self._backup_from_table(4.0)

            nav_ok2 = self.navigate_to_place_table(level=lvl + 1)
            self._current_cycle['transport_retained'] = self._check_transport_retained()
            self._current_cycle['dock_via_tag'] = bool(nav_ok2 and self._docked_via_tag)
            if not nav_ok2:
                self.get_logger().error('Re-nav to place table failed — stopping stack')
                self._cycle_end()
                break
            if not self._current_cycle['transport_retained']:
                self.get_logger().error(
                    'Box was lost in transit — stopping stack (nothing to put down)')
                self._cycle_end()
                break

        # 6. Back up and write metrics
        # place_box() leaves the arm at 'home'; fold it to 'travel' before the base
        # moves, as everywhere else.
        if getattr(self, '_arm_clear', True):
            self._arm.prepare_travel()
            self.back_up()
        levels = self._final_stack_check()
        if levels is not None and self._metrics:
            self._metrics[-1]['stack_levels_final'] = levels
        self.flush_metrics()

        verdict = ('COMPLETE ✓' if levels == place_levels else
                   f'ENDED — {levels if levels is not None else "?"} of {place_levels} level(s) standing')
        self._event('run_end', levels=levels, requested=place_levels, arm_clear=getattr(self, '_arm_clear', True))
        self._run_end_done = True
        self.get_logger().info('══════════════════════════════════════════════')
        self.get_logger().info(f'  SEQUENCE {verdict}')
        self.get_logger().info('══════════════════════════════════════════════')


def main(args=None):
    rclpy.init(args=args)
    node = NavPickOrchestrator()
    try:
        node._event('run_start')
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if not node._run_end_done:
            node._event('run_end', levels=None, note='run() returned early or was interrupted')
        node.flush_metrics()   # write CSV even on Ctrl-C
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
