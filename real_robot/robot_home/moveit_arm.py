#!/usr/bin/env python3
# Moves the real arm through MoveIt. face_and_pick.py uses it with --arm moveit.
# With ARM_SEG=joints the hover and park moves are sent as one joint command.
import json
import math
import os
import time

import rclpy
from geometry_msgs.msg import Point, Pose
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32

JOINTS = ['joint2_to_joint1', 'joint3_to_joint2', 'joint4_to_joint3',
          'joint5_to_joint4', 'joint6_to_joint5', 'joint6output_to_joint6']
FRAME, GROUP, TCP = 'base_footprint', 'arm', 'gripper_tcp'
DX, DY, DZ = -0.0044, -0.0012, 0.0270
GRASP_RAISE = 0.0055
PLACE_GAP = 0.003
VEL_PLAN = 0.20  # joint-space moves (25 % measured lag 16 deg vs 20 deg guard)
VEL_CART = 0.20
MAX_BRANCH_DEG = 30.0
PATH_MARGIN_DEG = 10.0
CART_MIN_FRACTION = 0.95
MAX_RETRIES = 2
SETTLE_S = 2.5  # feedback node's final correction after each move
GRIP_OPEN, GRIP_CLOSE, GRIP_THRESHOLD = 100, 0, 15
ARM_SEG = os.environ.get('ARM_SEG', 'off')
SEG_SPEED = int(os.environ.get('ARM_SEG_SPEED', '6'))  # firmware speed, joint moves
SEG_SPEED_NEAR = 4
SEG_ARRIVE_DEG = 2.0

import sys
sys.path.insert(0, os.path.expanduser('~'))
from arm_control import ArmError

def _deg(r):
    return [math.degrees(v) for v in r]

def grip_reached(cmd, v):
    if v is None:
        return False
    if cmd == GRIP_OPEN:
        return v >= 85
    if cmd == GRIP_CLOSE:
        return v < 90
    return True

class MoveItArm:
    def __init__(self, node, verbose=True):
        self.n = node
        self.verbose = verbose
        fk = json.load(open(os.path.expanduser('~/moveit_fk_check.json')))
        self.q = fk['grasp']['quat_xyzw']
        taught = json.load(open(os.path.expanduser('~/pick_poses.json')))['poses']
        self.hover_ref = taught['hover']['angles']
        self.state = {}
        self.t_state = 0.0
        self.retries = 0
        self.grip_hist = []
        node.create_subscription(JointState, '/real_arm/joint_states', self._on_state, 10)
        node.create_subscription(Int32, '/real_arm/gripper_value',
                                 lambda m: self.grip_hist.append(m.data), 10)
        self.grip_pub = node.create_publisher(Int32, '/real_arm/gripper_cmd', 10)
        if ARM_SEG not in ('off', 'joints', 'all'):
            raise ArmError(f'ARM_SEG={ARM_SEG}: use off, joints or all')
        self.seg_pub = node.create_publisher(JointState, '/real_arm/segment', 10)
        self.plan_cli = ActionClient(node, MoveGroup, '/move_action')
        self.exec_cli = ActionClient(node, ExecuteTrajectory, '/execute_trajectory')
        self.cart_cli = node.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.ik_cli = node.create_client(GetPositionIK, '/compute_ik')
        if not (self.plan_cli.wait_for_server(timeout_sec=20)
                and self.exec_cli.wait_for_server(timeout_sec=20)
                and self.cart_cli.wait_for_service(timeout_sec=20)
                and self.ik_cli.wait_for_service(timeout_sec=20)):
            raise ArmError('MoveIt not available -- is the b1 stack (feedback node + '
                           'real_arm_moveit.launch.py arm_control:=feedback) running?')
        self._spin(3.0)
        if len(self.state) < 6:
            raise ArmError('no /real_arm/joint_states -- is real_arm_feedback_node.py running?')
        self._log(f'MoveIt arm ready; real joints (deg) {[round(a, 1) for a in self.real()]}; '
                  f'one command per move: {ARM_SEG}' + (f' (speed {SEG_SPEED})' if ARM_SEG != 'off' else ''))

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def _spin(self, s):
        end = time.time() + s
        while time.time() < end:
            rclpy.spin_once(self.n, timeout_sec=0.05)

    def _wait(self, fut, timeout=60.0):
        end = time.time() + timeout
        while not fut.done() and time.time() < end:
            rclpy.spin_once(self.n, timeout_sec=0.05)
        if not fut.done():
            raise ArmError('MoveIt did not answer in time')
        return fut.result()

    def _on_state(self, m):
        self.state.update(zip(m.name, m.position))
        self.t_state = time.time()

    def _recover(self, label, err):
        if 'execution failed' not in str(err) or self.retries >= MAX_RETRIES:
            raise err
        self._log(f'  [{label}] {err}; waiting for the arm readings (up to 15 s)')
        end, ok_since = time.time() + 15.0, None
        while time.time() < end:
            self._spin(0.1)
            if time.time() - self.t_state < 0.3:
                ok_since = ok_since or time.time()
                if time.time() - ok_since > 1.5:
                    break
            else:
                ok_since = None
        else:
            raise ArmError(f'{label}: arm readings did not come back in 15 s -- arm holds')
        self.retries += 1
        self._log(f'  [{label}] readings back -- re-planning the move from the real pose '
                  f'(retry {self.retries}/{MAX_RETRIES})')

    def real(self):
        return _deg([self.state[j] for j in JOINTS])

    def _execute(self, traj, label):
        g = ExecuteTrajectory.Goal()
        g.trajectory = traj
        gh = self._wait(self.exec_cli.send_goal_async(g))
        if not gh.accepted:
            raise ArmError(f'{label}: execution rejected')
        code = self._wait(gh.get_result_async(), timeout=120).result.error_code.val
        self._spin(SETTLE_S)
        if code != 1:
            raise ArmError(f'{label}: execution failed (code {code}) -- arm holds')
        self._log(f'  [{label}] done, real (deg) {[round(a, 1) for a in self.real()]}')

    def _segment(self, goal_deg, label, speed):
        m = JointState()
        m.name = list(JOINTS)
        m.position = [math.radians(v) for v in goal_deg]
        m.velocity = [float(speed)]
        size = max(abs(a - b) for a, b in zip(goal_deg, self.real()))
        limit = 10.0 + size / 4.0
        self.seg_pub.publish(m)
        t0, last, still, err = time.time(), None, None, size
        while time.time() - t0 < limit:
            self._spin(0.1)
            if time.time() - self.t_state > 0.5:
                last = still = None
                continue
            r = self.real()
            err = max(abs(a - b) for a, b in zip(goal_deg, r))
            moving = last is None or max(abs(a - b) for a, b in zip(r, last)) > 0.05
            last = r
            if err <= SEG_ARRIVE_DEG and not moving:
                still = still or time.time()
                if time.time() - still > 0.3:
                    break
            else:
                still = None
        else:
            raise ArmError(f'{label}: execution failed (one command: still {err:.1f} deg off after '
                           f'{limit:.0f} s) -- arm holds')
        self._spin(SETTLE_S)
        self._log(f'  [{label}] done (one command, {size:.0f} deg), real (deg) '
                  f'{[round(a, 1) for a in self.real()]}')

    @staticmethod
    def _model_target(base_x, base_y, floor_h):
        return base_x + DX, base_y + DY, floor_h + DZ

    def _check_path(self, jt, goal_deg, label, ref_deg=None):
        idx = [jt.joint_names.index(j) for j in JOINTS]
        start = [math.degrees(jt.points[0].positions[i]) for i in idx]
        end = [math.degrees(jt.points[-1].positions[i]) for i in idx]
        goal = goal_deg if goal_deg is not None else end
        worst = 0.0
        for pt in jt.points:
            for k, i in enumerate(idx):
                v = math.degrees(pt.positions[i])
                lo, hi = sorted((start[k], goal[k]))
                worst = max(worst, lo - v, v - hi)
        if worst > PATH_MARGIN_DEG:
            raise ArmError(f'{label}: planned path leaves the start-goal range by {worst:.1f} deg '
                           f'-- NOT executed')
        if ref_deg is not None:
            off = max(abs(a - b) for a, b in zip(end, ref_deg))
            if off > MAX_BRANCH_DEG:
                raise ArmError(f'{label}: planned end pose is {off:.0f} deg from the taught pose '
                               f'(other IK branch?) -- NOT executed. end={[round(a) for a in end]}')
        return end

    def _plan(self, constraints, vel, label):
        from moveit_msgs.msg import MotionPlanRequest
        req = MotionPlanRequest()
        req.group_name = GROUP
        req.num_planning_attempts = 5
        req.allowed_planning_time = 5.0
        req.max_velocity_scaling_factor = vel
        req.max_acceleration_scaling_factor = vel
        req.goal_constraints = [constraints]
        g = MoveGroup.Goal()
        g.request = req
        g.planning_options.plan_only = True
        gh = self._wait(self.plan_cli.send_goal_async(g))
        if not gh.accepted:
            raise ArmError(f'{label}: plan goal rejected')
        r = self._wait(gh.get_result_async()).result
        if r.error_code.val != 1:
            raise ArmError(f'{label}: planning failed (code {r.error_code.val}) -- nothing moved')
        return r.planned_trajectory

    def move_joints(self, target_deg, label, vel=VEL_PLAN):
        c = Constraints()
        for j, v in zip(JOINTS, target_deg):
            jc = JointConstraint()
            jc.joint_name, jc.position = j, math.radians(v)
            jc.tolerance_above = jc.tolerance_below = math.radians(0.5)
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        for attempt in (0, 1):
            traj = self._plan(c, vel, label)
            end = self._check_path(traj.joint_trajectory, target_deg, label)
            try:
                if ARM_SEG != 'off':
                    return self._segment(end, label, SEG_SPEED)
                return self._execute(traj, label)
            except ArmError as e:
                if attempt:
                    raise
                self._recover(label, e)

    def move_pose(self, x, y, z, label, vel=VEL_PLAN):
        req = GetPositionIK.Request()
        req.ik_request.group_name, req.ik_request.ik_link_name = GROUP, TCP
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout = Duration(sec=1)
        req.ik_request.robot_state.joint_state.name = list(JOINTS)
        req.ik_request.robot_state.joint_state.position = [math.radians(a) for a in self.hover_ref]
        req.ik_request.robot_state.is_diff = True
        ps = PoseStamped()
        ps.header.frame_id = FRAME
        ps.pose.position = Point(x=float(x), y=float(y), z=float(z))
        (ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z,
         ps.pose.orientation.w) = self.q
        req.ik_request.pose_stamped = ps
        r = self._wait(self.ik_cli.call_async(req), timeout=10)
        if r is None or r.error_code.val != 1:
            raise ArmError(f'{label}: no IK solution (code {None if r is None else r.error_code.val})'
                           f' -- NOT executed')
        js = r.solution.joint_state
        sol = [math.degrees(js.position[js.name.index(j)]) for j in JOINTS]
        off = max(abs(a - b) for a, b in zip(sol, self.hover_ref))
        if off > MAX_BRANCH_DEG + 20:
            raise ArmError(f'{label}: IK solution {off:.0f} deg from the taught pose -- NOT executed. '
                           f'sol={[round(a) for a in sol]}')
        if sol[1] < -85.0:
            raise ArmError(f'{label}: IK solution J2 {sol[1]:.0f} < -85 (camera) -- NOT executed')
        self._log(f'  [{label}] IK on the taught branch (max {off:.0f} deg from hover pose): '
                  f'{[round(a, 1) for a in sol]}')
        self.move_joints(sol, label, vel)

    def move_straight(self, x, y, z, label, vel=VEL_CART):
        try:
            return self._move_straight(x, y, z, label, vel)
        except ArmError as e:
            self._recover(label, e)
        return self._move_straight(x, y, z, label, vel)

    def _move_straight(self, x, y, z, label, vel):
        req = GetCartesianPath.Request()
        req.header.frame_id = FRAME
        req.group_name, req.link_name = GROUP, TCP
        p = Pose()
        p.position = Point(x=float(x), y=float(y), z=float(z))
        (p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w) = self.q
        req.waypoints = [p]
        req.max_step = 0.002
        req.jump_threshold = 0.0
        req.revolute_jump_threshold = 0.15
        req.prismatic_jump_threshold = 0.0
        req.avoid_collisions = True
        r = self._wait(self.cart_cli.call_async(req), timeout=15)
        if r is None or r.fraction < CART_MIN_FRACTION:
            raise ArmError(f'{label}: straight line only {0 if r is None else r.fraction:.0%} '
                           f'planned -- NOT executed')
        traj = r.solution
        k = 1.0 / vel
        for pt in traj.joint_trajectory.points:
            t = (pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9) * k
            pt.time_from_start.sec, pt.time_from_start.nanosec = int(t), int((t - int(t)) * 1e9)
            pt.velocities = [v / k for v in pt.velocities]
            pt.accelerations = [a / (k * k) for a in pt.accelerations]
        end = self._check_path(traj.joint_trajectory, None, label)
        if ARM_SEG == 'all':
            return self._segment(end, label, SEG_SPEED_NEAR)
        self._execute(traj, label)

    def _grip(self, value, timeout=15.0):
        for attempt in range(3):
            self.grip_hist = []
            self.grip_pub.publish(Int32(data=value))
            t0 = time.time()
            while time.time() - t0 < timeout:
                rclpy.spin_once(self.n, timeout_sec=0.1)
                h = self.grip_hist
                if h and not grip_reached(value, h[-1]) and time.time() - t0 > 4.0:
                    break
                if len(h) >= 3 and h[-1] == h[-2] == h[-3] and grip_reached(value, h[-1]):
                    return h[-1]
            else:
                return self.grip_hist[-1] if self.grip_hist else None
            self._log(f'  gripper did not move -- resending ({attempt + 1}/2)')
        if value == GRIP_OPEN:
            raise ArmError(f'gripper OPEN not confirmed (last reading '
                           f'{self.grip_hist[-1] if self.grip_hist else None}) -- arm holds, no retreat')
        return self.grip_hist[-1] if self.grip_hist else None

    def open_gripper(self):
        return self._grip(GRIP_OPEN)

    def grip_value(self):
        return self._grip(-1, timeout=6.0)

    def park(self):
        self.move_joints([0.0] * 6, 'park', vel=0.10)
        a = self.real()
        self._log(f'parked: {[round(v, 2) for v in a]}')
        return a, []

    def pick_at(self, base_x, base_y, cube_centre_h, hover=0.06, lift=0.06, trace=None):
        grasp_h = cube_centre_h + GRASP_RAISE
        x, y, zg = self._model_target(base_x, base_y, grasp_h)
        if trace is not None:
            trace.update(cmd_x=base_x, cmd_y=base_y, cmd_hover_h=grasp_h + hover,
                         cmd_grasp_h=grasp_h, cmd_rz=-38.0)
        self._log(f'MoveIt pick at base_link x={base_x:.3f} y={base_y:.3f}, grasp {grasp_h*100:.1f} cm '
                  f'(model tcp {x:.3f} {y:.3f} {zg:.3f})')
        self.open_gripper()
        self.move_pose(x, y, zg + hover, 'hover')
        self.move_straight(x, y, zg, 'descend')
        v_close = self._grip(GRIP_CLOSE)
        self.move_straight(x, y, zg + lift, 'lift')
        v_lift = self.grip_value()
        held = v_lift is not None and GRIP_THRESHOLD < v_lift < 80
        if trace is not None:
            trace.update(grip_close=v_close, grip_lift=v_lift)
        self._log(f'  gripper close={v_close} lift={v_lift} -> {"HELD" if held else "NOT HELD"}')
        return held, v_lift

    def place_at(self, base_x, base_y, surface_h, cube_edge=0.0247, hover=0.06,
                 approach_gap=0.002, trace=None):
        centre = surface_h + cube_edge / 2.0 + PLACE_GAP
        x, y, zc = self._model_target(base_x, base_y, centre + GRASP_RAISE)
        if trace is not None:
            trace.update(cmd_x=base_x, cmd_y=base_y, cmd_drop_h=centre)
        self._log(f'MoveIt place at base_link x={base_x:.3f} y={base_y:.3f}, cube centre '
                  f'{centre*100:.1f} cm')
        self.move_pose(x, y, zc + hover, 'place hover')
        self.move_straight(x, y, zc, 'place descend')
        self.open_gripper()
        self.move_straight(x, y, zc + hover, 'place retreat')
        return None
