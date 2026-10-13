#!/usr/bin/env python3
# Runs the task on the real robot: find a cube on table A, pick it, drive to table B and place
# or stack it. I start it through b1_pipeline.sh, which also starts the arm feedback node and
# MoveIt. VERSIONS.md gives the command used with each version.
# The robot turns until the camera sees the cube straight ahead, so the turn does not rely on
# odometry. Each trial writes one row to ~/pick_series_<series>.csv.
import argparse, csv, hashlib, math, os, random, statistics as st, sys, time
import cv2
import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Float64
sys.path.insert(0, '/home/agilex')
from arm_control import ArmController, ArmError

TARGET_X   = 0.21  # m, cube centre ahead of base_link when the drive stops
CUBE_EDGE  = 0.0247  # m, real cube
FACE_TO_CENTRE = CUBE_EDGE / 2.0  # the camera sees the front face; the centre is half an edge further
GRASP_BACK = 0.011  # m, I grasp this far behind the measured centre (set with hover checks)
ODOM_SCALE = 1.0526
BASE_H     = 0.15
Y_TOL      = 0.045  # m, sideways offset the arm takes without moving the base
SPREAD_MAX = 0.030
NEAR_Y_JUMP = 0.015  # m, near and far look disagree more than this: another blob, stop
def target_jumped(far_y, near_y):
    return abs(near_y - far_y) >= NEAR_Y_JUMP

def one_target(samples, expect=None):
    s = sorted(samples, key=lambda b: b[1])
    if s[-1][1] - s[0][1] <= SPREAD_MAX:
        return list(samples)
    k = max(range(1, len(s)), key=lambda i: s[i][1] - s[i - 1][1])
    lo, hi = s[:k], s[k:]
    if expect is None:
        return lo if len(lo) > len(hi) else hi
    med = lambda g: st.median(b[1] for b in g)
    return lo if abs(med(lo) - expect) <= abs(med(hi) - expect) else hi

K_ROT, W_MIN, W_MAX = 1.2, 0.10, 0.35
K_LIN, V_MIN, V_MAX = 0.8, 0.06, 0.16
K_YAW, WMAX_HOLD = 1.6, 0.35

BUMPER_X            = 0.161  # m, base_link to the front bumper
SCAN_AT_CONTACT     = 0.0725  # m, lidar reading with the bumper on the table
SCAN_MIN_AT_CONTACT = 0.0637  # safety-cone minimum at contact
CONE_CENTRE = math.radians(3)
CONE_SAFETY = math.radians(20)
SAFETY_MIN  = 0.072  # m, stop if the nearest beam ahead reads less
SCAN_STALE  = 0.5  # s; an older scan does not count as a guard
TURN_RADIUS = 0.245  # m, nothing may be inside this circle before a turn
TURN_FOOT_X, TURN_FOOT_Y = 0.150, 0.12  # x < bumper (0.161) so a touching table counts
TURN_LIDAR_X, TURN_LIDAR_YAW = 0.108, math.radians(-7.8)  # as lidar_tf_up.sh

SQUARE_FOV    = math.radians(25)  # lidar beams within +-25 deg ahead (station_check.py)
SQUARE_TOL    = 2.0  # deg: square enough, no turn
SQUARE_ACCEPT = 4.0  # deg: still accepted after SQUARE_TRIES turns
SQUARE_TRIES  = 3
SQUARE_AGREE  = 5.0  # deg
PICK_GAP_MIN  = 0.010  # m, the final approach stops this far before table A
TURN_LEAD_DEG = 2.0  # in-place turns stop this early (the base coasts ~2 deg)
SQUARE_REF    = [None]

def turn_clearance(xs, ys):
    best = None
    for x, y in zip(xs, ys):
        if abs(x) <= TURN_FOOT_X and abs(y) <= TURN_FOOT_Y:
            continue
        d = math.hypot(x, y)
        if best is None or d < best:
            best = d
    return best
MAX_DETECTION_AGE = 0.5

def yaw_of(q):
    return math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

def face_fit_nearest(x, y, band=0.06):
    ok = (x > TURN_FOOT_X) & (np.abs(np.arctan2(y, x - TURN_LIDAR_X)) < SQUARE_FOV)
    x, y = x[ok], y[ok]
    if len(x) < 20:
        return None
    keep = x < np.percentile(x, 10) + band
    P = np.column_stack((x[keep], y[keep]))
    if len(P) < 20:
        return None
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
    if best is None or best.sum() < 20:
        return None
    Q = P[best]
    nrm = np.linalg.svd(Q - Q.mean(axis=0))[2][1]
    if nrm @ Q.mean(axis=0) < 0:
        nrm = -nrm
    phi = math.degrees(math.atan2(nrm[1], nrm[0]))
    return phi if np.ptp(Q[:, 1]) > 0.12 and abs(phi) < 30 else None

class N(Node):
    def __init__(self):
        super().__init__('face_and_pick')
        self.box, self.p, self.yaw = [], None, None
        self.last = None
        self.t_look = 0.0
        self.det_age = float('inf')
        self.create_subscription(Float64, '/box_detection_age', self._a, 20)
        self.create_subscription(PoseStamped, '/box_pose', self._b, 20)
        self.create_subscription(Odometry, '/odom', self._o, 20)
        self.scan = None
        self.safety_stopped = False
        self.create_subscription(LaserScan, '/scan', self._s, qos_profile_sensor_data)
        self.cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.last_drive, self.last_detect = {}, {}

    def snapshot(self, prefix, timeout=2.0):
        got = {}
        subs = [self.create_subscription(
                    Image, topic, lambda m, k=k: got.setdefault(k, m), qos_profile_sensor_data)
                for k, topic in (('rgb', '/camera/color/image_raw'),
                                 ('depth', '/camera/depth/image_raw'))]
        end = time.time() + timeout
        while len(got) < 2 and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        for sub in subs:
            self.destroy_subscription(sub)
        saved = []
        try:
            m = got.get('rgb')
            if m is not None:
                a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
                cv2.imwrite(prefix + '_rgb.png', a[:, :, ::-1] if m.encoding == 'rgb8' else a)
                saved.append('rgb')
            m = got.get('depth')
            if m is not None:
                dt = np.uint16 if m.encoding in ('16UC1', 'mono16') else np.float32
                np.save(prefix + '_depth.npy',
                        np.frombuffer(m.data, dt).reshape(m.height, m.width))
                saved.append('depth')
        except Exception as e:
            print(f'  (snapshot failed: {e})', flush=True)
        return os.path.basename(prefix) + ':' + '+'.join(saved) if saved else 'none'

    def _s(self, m):
        a = m.angle_min + np.arange(len(m.ranges)) * m.angle_increment + TURN_LIDAR_YAW
        r = np.array(m.ranges, float)
        ok = np.isfinite(r) & (r > m.range_min)
        self.scan_pts = (time.time(), TURN_LIDAR_X + r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok]))
        centre, cone = [], []
        for i, r in enumerate(m.ranges):
            if not (math.isfinite(r) and r > m.range_min):
                continue
            a = abs(m.angle_min + i * m.angle_increment)
            if a <= CONE_CENTRE:
                centre.append(r)
            if a <= CONE_SAFETY:
                cone.append(r)
        self.scan = (time.time(),
                     st.median(centre) if centre else None,
                     min(cone) if cone else None)

    def gap(self, n=10, timeout=5.0):
        vals, seen = [], None
        end = time.time() + timeout
        while len(vals) < n and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            s = self.scan
            if s and s[0] != seen and s[1] is not None:
                seen = s[0]
                vals.append(s[1])
        if len(vals) < 3:
            return None
        return st.median(vals) - SCAN_AT_CONTACT

    def _a(self, m):
        self.det_age = m.data

    def _b(self, m):
        if self.det_age > MAX_DETECTION_AGE:
            return
        p = m.pose.position
        self.last = (p.x, p.y, p.z)
        if m.header.stamp.sec + m.header.stamp.nanosec * 1e-9 >= self.t_look:
            self.box.append((p.x, p.y, p.z))

    def _o(self, m):
        self.p = (m.pose.pose.position.x, m.pose.pose.position.y)
        self.yaw = yaw_of(m.pose.pose.orientation)

    def detect(self, n=20, timeout=20, label='look', expect=None):
        self.box, self.t_look = [], time.time()
        t = time.time() + timeout
        while len(self.box) < n and time.time() < t:
            rclpy.spin_once(self, timeout_sec=0.05)
        if len(self.box) < 4:
            raise RuntimeError(f'{label}: only {len(self.box)} detections')
        raw = max(b[1] for b in self.box) - min(b[1] for b in self.box)
        keep = one_target(self.box, expect)
        if len(keep) < len(self.box):
            print(f'  {label}: TWO targets in the samples (spread y {raw:.3f}) -> kept {len(keep)} '
                  f'at y {st.median(b[1] for b in keep):+.3f}, dropped {len(self.box) - len(keep)}', flush=True)
        if len(keep) < 4:
            raise RuntimeError(f'{label}: only {len(keep)} detections of one target')
        xs, ys, zs = zip(*keep)
        m = (st.median(xs), st.median(ys), st.median(zs))
        self.last_detect = {'n': len(keep), 'spread_y': max(ys) - min(ys)}
        print(f'  {label}: {len(keep)} samples  x={m[0]:+.3f} y={m[1]:+.3f} '
              f'z={m[2]:+.3f}  spread y={max(ys)-min(ys):.3f}', flush=True)
        return m

    def _turn_once(self, timeout):
        t_end = time.time() + timeout
        ok = 0
        while time.time() < t_end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.last is None:
                continue
            y = self.last[1]
            t = Twist()
            if abs(y) < Y_TOL:
                ok += 1
                self.cmd.publish(t)
                if ok >= 8:
                    return True
            else:
                ok = 0
                w = K_ROT * y
                t.angular.z = math.copysign(
                    min(W_MAX, max(W_MIN, abs(w))), w)
                self.cmd.publish(t)
            time.sleep(0.05)
        self.cmd.publish(Twist())
        return False

    def face_target(self, timeout=45, passes=3):
        if args.pipeline:
            for i in range(1, passes + 1):
                try:
                    mx, my, _ = self.detect(10, 10, f'face {i}')
                except RuntimeError as e:
                    print(f'  {e}', flush=True)
                    return False
                if abs(my) < Y_TOL:
                    print(f'  centred after {i - 1} turn(s): y={my:+.3f}', flush=True)
                    return True
                self.turn_by(math.degrees(math.atan2(my, mx)))
                time.sleep(1.0)
            print('  could not centre the target within the allowed passes', flush=True)
            return False
        for i in range(1, passes + 1):
            if not self._turn_once(timeout / passes):
                print(f'  pass {i}: timed out while turning', flush=True)
                return False
            time.sleep(1.2)
            try:
                _, my, _ = self.detect(12, 10, f'pass {i} check')
            except RuntimeError as e:
                print(f'  {e}', flush=True)
                return False
            if abs(my) < Y_TOL:
                print(f'  centred after {i} pass(es): y={my:+.3f}', flush=True)
                return True
            print(f'  pass {i}: residual y={my:+.3f}, correcting', flush=True)
        print('  could not centre the target within the allowed passes', flush=True)
        return False

    def face_angle(self, n=10, timeout=3.0):
        vals, last = [], None
        end = time.time() + timeout
        while len(vals) < n and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            sp = getattr(self, 'scan_pts', None)
            if sp is None or sp[0] == last:
                continue
            last = sp[0]
            a = face_fit_nearest(np.asarray(sp[1]), np.asarray(sp[2]))
            if a is not None:
                vals.append(a)
        return st.median(vals) if len(vals) >= 3 else None

    def square_to_table(self, label):
        for i in range(1, SQUARE_TRIES + 1):
            a = self.heading_error(label, i)
            if a is None:
                raise RuntimeError(f'{label}: no lidar face and no AMCL heading -- not squaring')
            if abs(a) <= SQUARE_TOL:
                return a
            self.turn_by(a)
            time.sleep(1.0)
        a = self.heading_error(label, SQUARE_TRIES + 1)
        if a is not None and abs(a) <= SQUARE_ACCEPT:
            print(f'  {label}: table face {a:+.1f} deg -- accepted', flush=True)
            return a
        raise RuntimeError(f'{label}: not square to the table after {SQUARE_TRIES} turns ({a})')

    def heading_error(self, label, i, tag=None):
        a_l = self.face_angle()
        a_m = None
        if SQUARE_REF[0] is not None:
            y = amcl_yaw()
            if y is not None:
                a_m = (SQUARE_REF[0] - math.degrees(y) + 180) % 360 - 180
        if tag is not None:
            from tag_look import angle_near
            tag = tag['heading_err_deg'] if a_l is None else angle_near(tag, a_l)
        ref = a_m if tag is None else tag
        if a_l is not None and (ref is None or abs(a_l - ref) <= SQUARE_AGREE):
            src, a = 'lidar', a_l
        else:
            src, a = 'AMCL', a_m
        print(f'  {label}: heading error {"n/a" if a is None else f"{a:+.1f}"} deg from {src} '
              f'(lidar {"n/a" if a_l is None else f"{a_l:+.1f}"}, AMCL {"n/a" if a_m is None else f"{a_m:+.1f}"}'
              f'{"" if tag is None else f", tag {tag:+.1f}"}; '
              f'try {i})', flush=True)
        return a

    def turn_clear(self):
        backed = 0.0
        while True:
            end = time.time() + 1.0
            while time.time() < end:
                rclpy.spin_once(self, timeout_sec=0.05)
            sp = getattr(self, 'scan_pts', None)
            if sp is None or time.time() - sp[0] > SCAN_STALE:
                raise RuntimeError('no fresh /scan -- refusing to turn without the clearance check')
            d = turn_clearance(sp[1], sp[2])
            if d is None or d >= TURN_RADIUS:
                if backed:
                    print(f'  turning circle clear after reversing {backed:.2f} m', flush=True)
                return
            if backed >= 0.35:
                raise RuntimeError(f'obstacle {d*1000:.0f} mm inside the turning circle even after '
                                   f'reversing {backed:.2f} m -- not turning')
            print(f'  obstacle {d*1000:.0f} mm from the robot centre is inside the turning circle '
                  f'({TURN_RADIUS*1000:.0f} mm) -> reverse 5 cm first', flush=True)
            self.drive_back(0.05)
            backed += 0.05

    def turn_by(self, deg, rate=0.5, timeout=20.0):
        t = time.time() + 8
        while self.yaw is None and time.time() < t:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.yaw is None:
            raise RuntimeError('no /odom — the base driver is not running.')
        self.turn_clear()
        y0, target, sign = self.yaw, math.radians(abs(deg)), (1.0 if deg > 0 else -1.0)
        stop_at = max(target - math.radians(TURN_LEAD_DEG), target / 2.0)
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)
            done = abs(math.atan2(math.sin(self.yaw - y0), math.cos(self.yaw - y0)))
            if done >= stop_at:
                break
            v = Twist(); v.angular.z = sign * rate
            self.cmd.publish(v)
        self.cmd.publish(Twist())
        for _ in range(20): rclpy.spin_once(self, timeout_sec=0.05)
        turned = math.degrees(math.atan2(math.sin(self.yaw - y0), math.cos(self.yaw - y0)))
        print(f'  turned {turned:+.1f} deg by odometry (asked {deg:+.1f})', flush=True)
        return turned

    def drive_back(self, real):
        t = time.time() + 8
        while self.p is None and time.time() < t:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.p is None:
            raise RuntimeError('no /odom — the base driver is not running. '
                               'Start it with: ros2 launch limo_base limo_base.launch.py')
        if real < 0:
            raise RuntimeError(f'drive({real:.3f}): negative forward distance refused')
        tgt = abs(real) / ODOM_SCALE
        ox, oy, oyaw = self.p[0], self.p[1], self.yaw
        end = time.time() + 60
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)
            if math.hypot(self.p[0]-ox, self.p[1]-oy) >= tgt:
                break
            v = Twist()
            v.linear.x = -max(V_MIN, min(V_MAX, K_LIN*tgt))
            e = math.atan2(math.sin(self.yaw-oyaw), math.cos(self.yaw-oyaw))
            v.angular.z = max(-WMAX_HOLD, min(WMAX_HOLD, -K_YAW*e))
            self.cmd.publish(v)
        self.cmd.publish(Twist())
        for _ in range(20): rclpy.spin_once(self, timeout_sec=0.05)
        print(f'  reversed {math.hypot(self.p[0]-ox, self.p[1]-oy)*ODOM_SCALE:.3f} m',
              flush=True)

    def acquire(self, tries=3, back=0.40):
        for i in range(1, tries + 1):
            try:
                return self.detect(10, 12, f'look {i}')
            except RuntimeError as e:
                print(f'  {e}', flush=True)
                if i == tries:
                    raise
                print(f'  target not visible — reversing {back} m to recover',
                      flush=True)
                self.drive_back(back)
                time.sleep(1.5)

    def drive(self, real):
        t = time.time() + 8
        while self.p is None and time.time() < t:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.p is None:
            raise RuntimeError('no /odom — the base driver is not running.')
        t = time.time() + 3
        while self.scan is None and time.time() < t:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.scan is None:
            raise RuntimeError('no /scan — refusing to drive forward without the '
                               'bumper guard. Is limo_start (lidar) running?')
        tgt = abs(real) / ODOM_SCALE
        ox, oy, oyaw = self.p[0], self.p[1], self.yaw
        end = time.time() + 90
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.02)
            d = math.hypot(self.p[0]-ox, self.p[1]-oy)
            if d >= tgt: break
            s = self.scan
            if time.time() - s[0] > SCAN_STALE:
                self.cmd.publish(Twist())
                raise RuntimeError(f'/scan went stale ({time.time()-s[0]:.1f} s) '
                                   f'while driving — stopped.')
            if s[2] is not None and s[2] < SAFETY_MIN:
                self.safety_stopped = True
                print(f'  SAFETY STOP: obstacle {(s[2]-SCAN_MIN_AT_CONTACT)*1000:.0f} mm '
                      f'from the bumper (nearest beam {s[2]*1000:.0f} mm), '
                      f'{(tgt-d)*ODOM_SCALE*1000:.0f} mm short of the planned stop',
                      flush=True)
                break
            v = Twist()
            v.linear.x = max(V_MIN, min(V_MAX, K_LIN*(tgt-d)))
            e = math.atan2(math.sin(self.yaw-oyaw), math.cos(self.yaw-oyaw))
            v.angular.z = max(-WMAX_HOLD, min(WMAX_HOLD, -K_YAW*e))
            self.cmd.publish(v)
        self.cmd.publish(Twist())
        for _ in range(20): rclpy.spin_once(self, timeout_sec=0.05)
        moved = math.hypot(self.p[0]-ox, self.p[1]-oy) * ODOM_SCALE
        dyaw = math.degrees(math.atan2(math.sin(self.yaw-oyaw), math.cos(self.yaw-oyaw)))
        self.last_drive = {'cmd': abs(real), 'odom': moved, 'yaw_deg': dyaw}
        print(f'  drove {moved:.3f} m real (heading changed {dyaw:+.1f} deg)', flush=True)
        return moved

ap = argparse.ArgumentParser()
ap.add_argument('--setback', type=float, default=None,
                help='tape: table FRONT EDGE to the cube FRONT FACE, metres (e.g. 0.020). '
                     'Only valid for trial 1 -- the robot re-places the cube after that.')
ap.add_argument('--series', default=time.strftime('%Y%m%d_%H%M'),
                help='name of this series; snapshots go to ~/pick_snapshots/<series>/')
ap.add_argument('--lights', choices=['on', 'off'], default='on',
                help='room lights, stored with each trial')
ap.add_argument('--note', default='', help='free text stored with every trial')
ap.add_argument('--pause', action='store_true',
                help='stop after the approach and wait for Enter before the pick')
ap.add_argument('--drive-test', type=float, nargs='+', metavar='M',
                help='drive these distances one by one with the mission drive (Enter before each), '
                     'print odometry and lidar change, then exit; no arm, no camera')
ap.add_argument('--hover-check', action='store_true',
                help='MoveIt only: after the approach the open fingers go 15 mm above the grasp '
                     'point and hold for a tape check (Enter = park, no pick)')
ap.add_argument('--repeat', type=int, default=1,
                help='autonomous trials: after each successful pick the cube is put back, '
                     'the arm parks, the robot reverses a random distance, and it goes again')
ap.add_argument('--back-min', type=float, default=0.50, help='random reverse, metres (min)')
ap.add_argument('--back-max', type=float, default=0.70, help='random reverse, metres (max)')
ap.add_argument('--place-b-height', type=float, default=0.125, help='table B top above the floor, m')
ap.add_argument('--place-b-gap', type=float, default=0.02, help='planned bumper -> table B gap, m')
ap.add_argument('--place-b-in', type=float, default=0.03,
                help='place the cube centre this far in from table B\'s front edge, m')
ap.add_argument('--target-x', type=float, default=TARGET_X,
                help='pick: drive until the cube centre is this far ahead of base_link, m. The lidar still '
                     f'stops the base {PICK_GAP_MIN*1000:.0f} mm before table A')
ap.add_argument('--place-ahead', type=float, default=None, metavar='GAP',
                help='pipeline without a pick: Table B is straight ahead, bumper about GAP m from '
                     'its front face (Nav2 station). Close the gap on the lidar to --place-b-gap, '
                     'place the HELD cube --place-b-in from the edge on --place-b-height, park, reverse 0.30 m')
ap.add_argument('--pipeline', action='store_true',
                help='whole task in ONE process so the gripper is never re-initialised while '
                     'holding: Nav2 -> station A, align, forward --a-forward, pick (hold), Nav2 -> '
                     'station B, align, place ahead (--place-ahead GAP, default 0.50). Stations: ~/stations.yaml')
ap.add_argument('--stack', type=int, default=0, metavar='N',
                help='with --pipeline: stack N cubes on table B. Cube 1 on the table (place ahead), '
                     'cubes 2..N on top of the cube the CAMERA finds at B (same look/approach as a pick). '
                     'Carry with the arm straight up (park pose) so the camera sees the stack. '
                     'All cubes stand on table A from the start.')
ap.add_argument('--localize', action='store_true',
                help='with --pipeline: before driving, make sure AMCL knows where the robot is '
                     '(global localization + slow turns if unsure; stop if it cannot)')
ap.add_argument('--tag-dock', action='store_true',
                help='with --pipeline: at table B, square to the AprilTag and place on its centre line')
ap.add_argument('--lidar-square', action='store_true',
                help='with --pipeline: centre first, then square. One heading read (lidar if it agrees with '
                     'AMCL, else AMCL), one sideways shift planned for the coming turn, one squaring turn, '
                     'then extra shifts if still off centre (MAX_EXTRA_SHIFTS). At table A on the cube, '
                     'at table B on the AprilTag.')
ap.add_argument('--stack-from', type=int, default=0, metavar='K',
                help='with --stack: start at level K (K cubes already stacked on table B)')
ap.add_argument('--stack-x', choices=('table', 'camera'), default='table',
                help='with --stack: forward place x for cubes 2..N. table = lidar table face + '
                     '--place-b-in (the cube 1 command repeated); camera = the pick rule on the cube '
                     'below (camera - drive - GRASP_BACK). y always comes from the camera.')
ap.add_argument('--a-forward', type=float, default=0.25,
                help='with --pipeline: drive this far toward table A after the station (detector range)')
ap.add_argument('--arm', choices=['firmware', 'moveit'], default='firmware',
                help='arm path: firmware (ArmController, default) or moveit (MoveItArm: MoveIt on the '
                     'real arm via real_arm_feedback_node.py; start the b1 stack first)')
args, ros_args = ap.parse_known_args()

STAGE2_RANGE = 0.45
PLACE_GAP = 0.008  # m, release height above the surface (the arm settles a few mm low)
LOG = os.path.expanduser(f'~/pick_series_{args.series}.csv')
SNAP_DIR = os.path.expanduser(f'~/pick_snapshots/{args.series}')
FIELDS = [
    'time', 'series', 'trial', 'note', 'lights', 'script_md5', 'arm_md5',
    'camera_x', 'odom_scale', 'target_x', 'setback', 'edge_centre',
    'far_x', 'far_y', 'far_z', 'far_n', 'far_spread_y',
    'near_pred_x', 'near_x', 'near_y', 'near_z', 'near_n', 'near_spread_y',
    'near_consistency', 'snap_far', 'snap_near',
    'drive1_cmd', 'drive1_odom', 'drive1_yaw', 'drive2_cmd', 'drive2_odom', 'drive2_yaw',
    'safety_stop', 'gap_look', 'gap_final', 'stack_closer',
    'true_centre_final', 'pick_err',
    'cmd_x', 'cmd_y', 'cmd_hover_h', 'cmd_grasp_h', 'cmd_rz',
    'hover_x', 'hover_y', 'hover_h', 'hover_rz', 'hover_j1', 'hover_j2', 'hover_j6',
    'grasp_x', 'grasp_y', 'grasp_h', 'grasp_rz', 'grasp_j1', 'grasp_j2', 'grasp_j6',
    'lifted_rz', 'lifted_j1', 'lifted_j6', 'grip_close', 'grip_lift',
    'result',
    'place_cmd_x', 'place_cmd_y', 'place_cmd_h', 'drop_x', 'drop_y', 'drop_h', 'drop_rz',
    'place', 'stack_x_cam',
    'back_cmd',
    't_find', 't_approach', 't_final', 't_pick', 't_place', 't_back', 't_total',
]

def md5_of(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:10]

def camera_x_live(timeout=5.0):
    tnode = rclpy.create_node('face_and_pick_tf')
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, tnode)
    x = None
    end = time.time() + timeout
    while x is None and time.time() < end:
        rclpy.spin_once(tnode, timeout_sec=0.1)
        try:
            x = buf.lookup_transform('base_link', 'camera_link',
                                     rclpy.time.Time()).transform.translation.x
        except tf2_ros.TransformException:
            pass
    tnode.destroy_node()
    return x

class Abort(Exception):
    def __init__(self, result):
        super().__init__(result)
        self.result = result

def mm(v):
    return f'{v*1000:+.0f} mm' if v is not None else 'n/a'

def save(rec):
    new = not os.path.exists(LOG)
    with open(LOG, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
        if new:
            w.writeheader()
        w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in rec.items()})
    print(f'  logged to {LOG}', flush=True)

def flatten(rec, prefix, snap):
    for k, v in (snap or {}).items():
        rec[f'{prefix}_{k}'] = v

STACK_SURFACE = [None]
STACK_HOVER = 0.04  # m, hover height above a stack place
def stack_place_x(g, place_in):
    if g is None or not -0.01 <= g <= 0.15:
        return None
    return BUMPER_X + g + place_in

def stack_x_camera(bx, moved):
    return bx - moved - GRASP_BACK

STACK_GAP_MIN = 0.005
STACK_STEP = 0.010  # m, drive this much closer when the arm has no IK at the stack
def closer_step(g, err):
    if g is None or not err.startswith('place hover:'):
        return None
    if 'NOT executed' not in err and 'nothing moved' not in err:
        return None
    d = min(STACK_STEP, g - STACK_GAP_MIN)
    return d if d >= 0.002 else None

def pick_drive(bx, g0, target_x):
    d_cmd = bx - target_x
    if g0 is not None and 0.0 < g0 < bx - BUMPER_X and g0 - PICK_GAP_MIN < d_cmd:
        return g0 - PICK_GAP_MIN, PICK_GAP_MIN
    return d_cmd, None

def return_turn(side, yaw0, yaw):
    t = math.degrees(math.atan2(math.sin(yaw0 - yaw), math.cos(yaw0 - yaw)))
    return t if abs(t + side) <= 10.0 else -side

def shift_sideways(lat):
    side = 90.0 if lat > 0 else -90.0
    n.drive_back(0.15)
    yaw0 = n.yaw
    n.turn_by(side)
    n.drive(abs(lat))
    n.turn_by(return_turn(side, yaw0, n.yaw))
    n.drive(0.15)

SHIFT_MIN = 0.030
MAX_EXTRA_SHIFTS = 5  # sideways shifts allowed after the first one
def centre_plan(th_deg, x, y):
    return y - x * math.tan(math.radians(th_deg))

def centre_then_square(rec, label, look, ok=None, tag=None):
    ok = ok or (lambda v: abs(v) < Y_TOL)
    try:
        x, y = look(f'{label} look')
        th = n.heading_error(label, 1, tag=tag() if tag else None)
        if th is None:
            raise RuntimeError(f'{label}: no lidar face and no AMCL heading')
    except RuntimeError as e:
        print(f'  {e}', flush=True)
        raise Abort('NOT_SQUARE')
    s = centre_plan(th, x, y)
    print(f'  {label}: heading error {th:+.1f} deg, target x {x:.3f} y {y*1000:+.0f} mm -> '
          f'shift {s*1000:+.0f} mm{" (arm absorbs it, no shift)" if abs(s) < SHIFT_MIN else ""}, '
          f'then turn {th:+.1f} deg', flush=True)
    rec['note'] = (rec.get('note') or '') + f' | {label}: hdg {th:+.1f} y {y*1000:+.0f} shift {s*1000:+.0f}'
    if abs(s) >= SHIFT_MIN:
        shift_sideways(s)
    if abs(th) > SQUARE_TOL:
        n.turn_by(th)
        time.sleep(1.0)
    for rnd in range(1, MAX_EXTRA_SHIFTS + 2):
        try:
            _, y2 = look(f'{label} check {rnd}')
        except RuntimeError as e:
            print(f'  {e}', flush=True)
            raise Abort('LOST_AFTER_CENTRING')
        if ok(y2):
            print(f'  centred and square: y={y2*1000:+.0f} mm after {rnd - 1} extra shift(s)', flush=True)
            rec['note'] += f' res {y2*1000:+.0f}'
            return y2
        if rnd == MAX_EXTRA_SHIFTS + 1:
            break
        print(f'  residual {y2*1000:+.0f} mm -> extra sideways shift {rnd}/{MAX_EXTRA_SHIFTS} '
              f'(no re-squaring)', flush=True)
        shift_sideways(y2)
    raise Abort('NOT_CENTRED')

def look_cube(label):
    x, y, _ = n.detect(10, 10, label, expect=0.0 if 'check' in label else None)
    return x, y

def trial(rec, edge, park_first, tag):
    n.safety_stopped = False
    t0 = time.time()
    if park_first:
        print('=== 0. park the arm before driving ===', flush=True)
        arm.open_gripper(); arm.park()

    t = time.time()
    if args.lidar_square:
        print('=== 1. CENTRE ON THE TARGET (sideways), THEN SQUARE TO THE TABLE ===', flush=True)
    else:
        print('=== 1. TURN TO FACE THE TARGET (camera closed-loop) ===', flush=True)
    try:
        n.acquire()
    except RuntimeError:
        raise Abort('NOT_VISIBLE')
    if args.lidar_square and STACK_SURFACE[0] is not None and args.tag_dock:
        print('  stack: already centred and squared by the tag dock', flush=True)
    elif args.lidar_square:
        centre_then_square(rec, 'target', look_cube)
    elif not n.face_target():
        raise Abort('NOT_CENTRED')
    time.sleep(1.5)

    print('=== 2. LOOK (this reading decides the grasp) ===', flush=True)
    try:
        bx, by, bz = n.detect(20, 20, 'after turn', expect=0.0)
    except RuntimeError as e:
        print(f'  {e}', flush=True)
        raise Abort('LOST_AFTER_TURN')
    rec.update(far_x=bx, far_y=by, far_z=bz, far_n=n.last_detect.get('n'),
               far_spread_y=n.last_detect.get('spread_y'))
    rec['snap_far'] = n.snapshot(os.path.join(SNAP_DIR, f'{tag}_far'))
    rec['t_find'] = time.time() - t
    cube_h = bz + BASE_H
    print(f'  cube {bx:.3f} m ahead, {by*100:+.1f} cm lateral, '
          f'{cube_h*100:.1f} cm high', flush=True)
    if abs(by) > 0.075:
        print(f'  STOPPING: {by*100:+.1f} cm lateral is beyond the arm\'s range even '
              f'after turning.', flush=True)
        raise Abort('LATERAL_OUT_OF_RANGE')

    t = time.time()
    moved = 0.0
    if bx - STAGE2_RANGE > 0.08:
        print('=== 3a. APPROACH to the accurate-reading range ===', flush=True)
        try:
            moved += n.drive(bx - STAGE2_RANGE)
        except RuntimeError as e:
            print(f'  {e}', flush=True)
            raise Abort('DRIVE_ABORTED')
        rec.update(drive1_cmd=n.last_drive['cmd'], drive1_odom=n.last_drive['odom'],
                   drive1_yaw=n.last_drive['yaw_deg'], near_pred_x=bx - moved)
        if n.safety_stopped:
            print('  the bumper guard fired on the FIRST approach — the table is much '
                  'nearer than the camera says. Stopping.', flush=True)
            raise Abort('SAFETY_STOP_EARLY')
        print('=== 3b. SECOND LOOK (this one decides the grasp) ===', flush=True)
        try:
            bx2, by2, bz2 = n.detect(20, 20, 'near', expect=by)
        except RuntimeError as e:
            print(f'  {e}\n  lost the target on the second look.', flush=True)
            raise Abort('LOST_ON_SECOND_LOOK')
        if target_jumped(by, by2):
            print(f'  STOPPING: the near look is {(by2 - by)*1000:+.0f} mm sideways from the far look '
                  f'(far y {by*1000:+.0f}, near y {by2*1000:+.0f}, limit {NEAR_Y_JUMP*1000:.0f} mm): '
                  f'another blob, not the cube.', flush=True)
            rec.update(near_x=bx2, near_y=by2, near_z=bz2)
            raise Abort('TARGET_JUMPED')
        dh = (bz2 - bz) * 100
        print(f'  cube {bx2:.3f} m ahead, {by2*100:+.1f} cm lateral, '
              f'{(bz2+BASE_H)*100:.1f} cm high   '
              f'(height revised {dh:+.1f} cm from the far reading)', flush=True)
        cons = bx2 - (bx - moved)
        print(f'  far/near consistency {cons*1000:+.0f} mm', flush=True)
        bx, by, bz = bx2, by2, bz2
        moved = 0.0
        cube_h = bz + BASE_H
        rec.update(near_x=bx, near_y=by, near_z=bz, near_n=n.last_detect.get('n'),
                   near_spread_y=n.last_detect.get('spread_y'), near_consistency=cons)
        if abs(by) > 0.075:
            print(f'  STOPPING: {by*100:+.1f} cm lateral is beyond the arm\'s range.',
                  flush=True)
            raise Abort('LATERAL_OUT_OF_RANGE')
    else:
        print('=== 3. already within accurate range, no second look needed ===',
              flush=True)
        rec.update(near_x=bx, near_y=by, near_z=bz)
    rec['snap_near'] = n.snapshot(os.path.join(SNAP_DIR, f'{tag}_near'))
    rec['t_approach'] = time.time() - t

    g_look = n.gap()
    rec['gap_look'] = g_look

    bx += FACE_TO_CENTRE
    print(f'=== 3c. FINAL APPROACH  (cube centre {bx:.3f}, '
          f'+{FACE_TO_CENTRE*1000:.1f} mm from the reported face) ===', flush=True)
    t = time.time()
    g0 = g_look
    if STACK_SURFACE[0] is not None:
        try:
            d_cmd = approach_step(g0, args.place_b_gap, STAGE2_RANGE)
        except RuntimeError as e:
            print(f'  {e}', flush=True)
            raise Abort('STACK_NO_GAP')
        print(f'  stack: lidar gap {mm(g0)} -> drive {d_cmd:.3f} m to {args.place_b_gap*1000:.0f} mm '
              f'from table B', flush=True)
        gap_goal = args.place_b_gap
    else:
        d_cmd, gap_goal = pick_drive(bx, g0, args.target_x)
        if gap_goal is not None:
            print(f'  lidar gap {g0*1000:.0f} mm: stop {PICK_GAP_MIN*1000:.0f} mm before the table '
                  f'(drive {d_cmd:.3f} instead of {bx - args.target_x:.3f}; arm reaches x={bx - d_cmd:.3f})',
                  flush=True)
    try:
        d_final = n.drive(d_cmd)
    except RuntimeError as e:
        print(f'  {e}', flush=True)
        raise Abort('DRIVE_ABORTED')
    moved += d_final
    rec.update(drive2_cmd=n.last_drive['cmd'], drive2_odom=n.last_drive['odom'],
               drive2_yaw=n.last_drive['yaw_deg'], safety_stop=n.safety_stopped)

    g_final = n.gap()
    if gap_goal is not None and g_final is not None and g_final > gap_goal + 0.010:
        print(f'  second lidar step: gap {g_final*1000:.0f} mm -> {gap_goal*1000:.0f} mm', flush=True)
        try:
            moved += n.drive(approach_step(g_final, gap_goal, 0.05))
        except RuntimeError as e:
            print(f'  {e}', flush=True)
            raise Abort('DRIVE_ABORTED')
        g_final = n.gap()
    rec['gap_final'] = g_final
    rec['t_final'] = time.time() - t
    print(f'  bumper -> table gap now {mm(g_final)}'
          f'{"   (touching)" if g_final is not None and g_final < 0.003 else ""}',
          flush=True)
    if edge is not None and g_final is not None:
        true_final = BUMPER_X + g_final + edge
        rec.update(true_centre_final=true_final, pick_err=(bx - moved) - true_final)
        print(f'  pick target x={bx - moved:.3f}, cube centre really at {true_final:.3f}'
              f'  -> pick error {mm(rec["pick_err"])}   (+ = aims too far forward)',
              flush=True)

    if args.pause:
        try:
            input('  >>> PAUSED before the pick. Check it, then Enter to pick '
                  '(Ctrl-C to stop the series) ')
        except KeyboardInterrupt:
            print('\n  stopped at the pause', flush=True)
            raise Abort('SKIPPED')

    if STACK_SURFACE[0] is not None:
        rec['stack_x_cam'] = bx - moved
        rec['note'] = (rec.get('note') or '') + f' | stack x {args.stack_x}'
        closer = 0
        while True:
            sx_table = stack_place_x(g_final, args.place_b_in)
            if sx_table is None:
                print(f'  REFUSING: no lidar gap to table B near contact ({g_final}) -- not placing blind',
                      flush=True)
                raise Abort('STACK_NO_GAP')
            if args.stack_x == 'camera':
                sx = stack_x_camera(bx, moved)
                src = (f'camera {bx:.3f} - drive {moved:.3f} - {GRASP_BACK*1000:.0f} mm; '
                       f'table rule said {sx_table:.3f}')
            else:
                sx = sx_table
                src = (f'table face {sx - args.place_b_in:.3f} + {args.place_b_in*1000:.0f} mm; '
                       f'camera - odometry said {bx - moved:.3f}')
            print(f'=== 4. STACK: place the HELD cube on the detected cube at x={sx:.3f} y={by:+.3f}, '
                  f'surface {STACK_SURFACE[0]*100:.1f} cm ({src}) ===', flush=True)
            ptr = {}
            try:
                if not (0.18 <= sx <= 0.238):
                    raise ArmError(f'place hover: x={sx:.3f} outside the reach band 0.180-0.238 '
                                   f'-- NOT executed')
                arm.place_at(sx, by, STACK_SURFACE[0], cube_edge=CUBE_EDGE, hover=STACK_HOVER,
                             approach_gap=PLACE_GAP, trace=ptr)
                break
            except ArmError as e:
                d = closer_step(g_final, str(e)) if sx >= 0.18 else None
                if d is None:
                    if not (0.18 <= sx <= 0.238):
                        print(f'  REFUSING: {e}', flush=True)
                        raise Abort('STACK_OUT_OF_REACH')
                    raise
                closer += 1
                print(f'  {e}\n  too far for the arm: drive {d*1000:.0f} mm closer to table B and try '
                      f'again (retry {closer})', flush=True)
                try:
                    moved += n.drive(d)
                except RuntimeError as e2:
                    print(f'  {e2}', flush=True)
                    raise Abort('DRIVE_ABORTED')
                g_final = n.gap()
                rec.update(gap_final=g_final, stack_closer=closer)
                print(f'  bumper -> table gap now {mm(g_final)}', flush=True)
        rec.update(place='STACK_OK', place_cmd_x=ptr.get('cmd_x', ''), place_cmd_y=ptr.get('cmd_y', ''),
                   place_cmd_h=ptr.get('cmd_drop_h', ''))
        flatten(rec, 'drop', {k: v for k, v in (ptr.get('drop') or {}).items()
                              if k in ('x', 'y', 'h', 'rz')})
        arm.park()
        rec['result'] = 'STACKED'
        return True, sx, by, cube_h
    if args.hover_check:
        tx = bx - moved - GRASP_BACK
        x, y, zg = arm._model_target(tx, by, cube_h + 0.0055)
        print(f'=== 4. HOVER CHECK: target x={tx:.3f} y={by:+.3f} (camera {bx:.3f} - drive {moved:.3f} '
              f'- {GRASP_BACK*1000:.0f} mm); lidar gap now {g_final} ===', flush=True)
        rec.update(cmd_x=tx, cmd_y=by)
        try:
            arm.open_gripper()
            arm.move_pose(x, y, zg + 0.06, 'hover')
            arm.move_straight(x, y, zg + 0.015, 'check height')
            input('  >>> HOLDING. Tape: (a) bumper -> cube centre, (b) finger centre - cube centre '
                  '(+ = beyond the cube). Enter = park ')
        except ArmError as e:
            print(f'\nARM ABORTED: {e}', flush=True)
            raise Abort('ARM_ABORTED')
        except KeyboardInterrupt:
            print('\n  stopped at the hold', flush=True)
        arm.park()
        raise Abort('HOVER_CHECK')
    print('=== 4. PICK ===', flush=True)
    t = time.time()
    tr = {}
    try:
        held, val = arm.pick_at(bx - moved - GRASP_BACK, by, cube_h, trace=tr)
    except ArmError as e:
        print(f'\nARM ABORTED: {e}', flush=True)
        raise Abort('ARM_ABORTED')
    finally:
        rec.update({k: tr[k] for k in ('cmd_x', 'cmd_y', 'cmd_hover_h', 'cmd_grasp_h',
                                       'cmd_rz', 'grip_close', 'grip_lift') if k in tr})
        flatten(rec, 'hover', tr.get('hover'))
        flatten(rec, 'grasp', tr.get('grasp'))
        rec['lifted_rz'] = (tr.get('lifted') or {}).get('rz', '')
        rec['lifted_j1'] = (tr.get('lifted') or {}).get('j1', '')
        rec['lifted_j6'] = (tr.get('lifted') or {}).get('j6', '')
        rec['t_pick'] = time.time() - t
        rec['t_total'] = time.time() - t0
    rec['result'] = 'SUCCESS' if held else 'FAILED'
    g = tr.get('grasp') or {}
    if g:
        print(f'  wrist angle at the grasp: {g["rz"]:.1f} deg (commanded {tr["cmd_rz"]:.1f}, '
              f'error {g["rz"] - tr["cmd_rz"]:+.1f}); after lift {rec["lifted_rz"]}',
              flush=True)
    print(f'\nRESULT: {rec["result"]}  gripper={val}', flush=True)
    return held, bx - moved, by, cube_h

rclpy.init(args=[sys.argv[0]] + ros_args,
           signal_handler_options=SignalHandlerOptions.NO)
os.makedirs(SNAP_DIR, exist_ok=True)
n = N()
if args.drive_test:
    try:
        for i, dist in enumerate(args.drive_test, 1):
            input(f'  >>> drive {i}: mark the floor now. Enter = drive {dist*1000:.0f} mm ')
            g0 = n.gap()
            moved = n.drive(dist)
            time.sleep(1.0)
            g1 = n.gap()
            lid = f'{(g0 - g1)*1000:.1f} mm' if g0 is not None and g1 is not None else '-'
            print(f'  DRIVE {i}: asked {dist*1000:.0f}  odometry {moved*1000:.1f} '
                  f'(raw {moved/ODOM_SCALE*1000:.1f}, scale {ODOM_SCALE})  lidar change {lid}  '
                  f'lidar gap now {mm(g1) if g1 else "-"}', flush=True)
        input('  >>> mark the floor now. Enter = end ')
    except KeyboardInterrupt:
        print('\n  stopped', flush=True)
    finally:
        n.cmd.publish(Twist())
        rclpy.shutdown()
    sys.exit(0)
if args.arm == 'moveit':
    from moveit_arm import ARM_SEG, MoveItArm
    arm = MoveItArm(n)
    if ARM_SEG != 'off':
        args.note = f'{args.note} arm_seg={ARM_SEG}'.strip()
else:
    arm = ArmController()
frozen = dict(
    series=args.series, note=args.note, lights=args.lights,
    script_md5=md5_of(os.path.abspath(__file__)),
    arm_md5=md5_of(sys.modules['moveit_arm'].__file__ if args.arm == 'moveit'
                   else sys.modules['arm_control'].__file__),
    camera_x=camera_x_live(), odom_scale=ODOM_SCALE, target_x=args.target_x)
print(f'series {args.series}: script {frozen["script_md5"]}, arm {frozen["arm_md5"]}, '
      f'camera x {frozen["camera_x"]}, lights {args.lights}', flush=True)
if frozen['camera_x'] is None:
    print('  WARNING: base_link -> camera_link not found. Run ./perception_up.sh first.',
          flush=True)
def place_on_b(face_x, rec, py=0.0):
    bx = face_x + args.place_b_in
    print(f'=== B6. PLACE on table B at x={bx:.3f} y={py:+.3f}, surface {args.place_b_height*100:.1f} cm ===',
          flush=True)
    if not (0.18 <= bx <= 0.238):
        print(f'  REFUSING: x={bx:.3f} is outside the safe reach band 0.180-0.238 m. '
              f'The cube stays in the gripper.', flush=True)
        rec['place'] = 'OUT_OF_REACH'
        return False
    ptr = {}
    arm.place_at(bx, py, args.place_b_height, cube_edge=CUBE_EDGE,
                 approach_gap=PLACE_GAP, trace=ptr)
    rec.update(place='B_OK', place_cmd_x=ptr.get('cmd_x', ''), place_cmd_y=ptr.get('cmd_y', ''),
               place_cmd_h=ptr.get('cmd_drop_h', ''))
    flatten(rec, 'drop', {k: v for k, v in (ptr.get('drop') or {}).items()
                          if k in ('x', 'y', 'h', 'rz')})
    arm.park()
    return True

summary = []
def approach_step(g, target, gap0):
    if g is None:
        raise RuntimeError('no lidar gap to table B -- not driving blind')
    if g <= target + 0.010:
        return 0.0
    if g <= gap0 + 0.25:
        return g - target
    raise RuntimeError(f'lidar gap {g:.3f} m is beyond the expected {gap0 + 0.25:.2f} m -- '
                       f'table B not where expected, not driving')

def place_ahead(gap0, rec, py=0.0):
    print('\n################ PLACE AHEAD on table B (pipeline) ################', flush=True)
    try:
        g = n.gap()
        print(f'  lidar gap to table B: {g}', flush=True)
        d = approach_step(g, args.place_b_gap, gap0)
        if d > 0:
            n.drive(d)
        g = n.gap()
        d = approach_step(g, args.place_b_gap, 0.05)
        if d > 0:
            n.drive(d)
            g = n.gap()
        face_x = BUMPER_X + (g if g is not None and -0.01 <= g <= 0.15 else args.place_b_gap)
        print(f'  final gap {g} -> table B face at x={face_x:.3f}', flush=True)
        if place_on_b(face_x, rec, py):
            print('=== B7. REVERSE 0.30 m ===', flush=True)
            n.drive_back(0.30)
        print(f'PLACE AHEAD DONE: {rec.get("place")}', flush=True)
    except (RuntimeError, ArmError, KeyboardInterrupt) as e:
        n.cmd.publish(Twist())
        print(f'  PLACE AHEAD stopped: {e}', flush=True)
        rec['place'] = 'B_FAILED'
        raise
    n.cmd.publish(Twist())

def amcl_state(timeout=5.0):
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from rclpy.qos import DurabilityPolicy, QoSProfile
    got = []
    sub = n.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', got.append,
                                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    end = time.time() + timeout
    while not got and time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.1)
    n.destroy_subscription(sub)
    if not got:
        return None
    p = got[-1].pose
    c = p.covariance
    return (p.pose.position.x, p.pose.position.y, yaw_of(p.pose.orientation),
            math.sqrt(max(c[0], c[7])), math.degrees(math.sqrt(max(c[35], 0.0))))

LOC_SD_XY, LOC_SD_YAW = 0.25, 15.0

def localize():
    from std_srvs.srv import Empty
    st0 = amcl_state()
    if st0 and st0[3] < LOC_SD_XY and st0[4] < LOC_SD_YAW:
        print(f'=== LOCALIZE: AMCL already sure (sd {st0[3]*100:.0f} cm, {st0[4]:.0f} deg) ===', flush=True)
        return
    print('=== LOCALIZE: AMCL unsure -> global localization + slow turns ===', flush=True)
    cli = n.create_client(Empty, '/reinitialize_global_localization')
    if not cli.wait_for_service(timeout_sec=5):
        raise RuntimeError('AMCL global localization service not available')
    fut = cli.call_async(Empty.Request())
    rclpy.spin_until_future_complete(n, fut, timeout_sec=5)
    for k in range(4):
        n.turn_by(120.0)
        time.sleep(1.0)
        st = amcl_state()
        if st:
            print(f'  after {120*(k+1)} deg: sd {st[3]*100:.0f} cm, {st[4]:.0f} deg at '
                  f'({st[0]:.2f}, {st[1]:.2f})', flush=True)
        if st and st[3] < LOC_SD_XY and st[4] < LOC_SD_YAW:
            print('  localized', flush=True)
            return
    raise RuntimeError('localization did not converge -- set 2D Pose Estimate in RViz and rerun')

def amcl_yaw(timeout=5.0):
    from geometry_msgs.msg import PoseWithCovarianceStamped
    from rclpy.qos import DurabilityPolicy, QoSProfile
    got = []
    sub = n.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', got.append,
                                QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    end = time.time() + timeout
    while not got and time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.1)
    n.destroy_subscription(sub)
    return yaw_of(got[-1].pose.pose.orientation) if got else None

def nav_to(name, st):
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient
    s_ = st[name]
    cli = ActionClient(n, NavigateToPose, 'navigate_to_pose')
    if not cli.wait_for_server(timeout_sec=10):
        raise RuntimeError('Nav2 navigate_to_pose not available')
    g = NavigateToPose.Goal()
    g.pose.header.frame_id = 'map'
    g.pose.header.stamp = n.get_clock().now().to_msg()
    g.pose.pose.position.x, g.pose.pose.position.y = s_['x'], s_['y']
    yw = math.radians(s_['yaw_deg'])
    g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(yw / 2), math.cos(yw / 2)
    print(f'=== NAV2 -> station {name} ({s_["x"]:.2f}, {s_["y"]:.2f}, {s_["yaw_deg"]:.0f} deg) ===',
          flush=True)
    t0 = time.time()
    fut = cli.send_goal_async(g)
    rclpy.spin_until_future_complete(n, fut)
    gh = fut.result()
    if not gh.accepted:
        raise RuntimeError('Nav2 goal rejected')
    rf = gh.get_result_async()
    try:
        while not rf.done():
            rclpy.spin_once(n, timeout_sec=0.1)
    except KeyboardInterrupt:
        gh.cancel_goal_async()
        raise
    status = rf.result().status
    print(f'  Nav2 status {status} (4 = SUCCEEDED) in {time.time() - t0:.0f} s', flush=True)
    if status != 4:
        raise RuntimeError(f'Nav2 to {name} did not succeed (status {status})')
    for _ in range(10):
        rclpy.spin_once(n, timeout_sec=0.1)
    y = amcl_yaw()
    if y is not None:
        d = (s_['yaw_deg'] - math.degrees(y) + 180) % 360 - 180
        print(f'=== ALIGN: heading off by {d:+.1f} deg -> turn ===', flush=True)
        if abs(d) > 2.0:
            n.turn_by(d)

TAG_Y_MIN, TAG_Y_MAX = -0.060, 0.040
DOCK_X = 0.65  # m, tag distance at which table B counts as docked
_TAG = [None]

def tag_dock(rec):
    from tag_look import TagLook
    if _TAG[0] is None:
        _TAG[0] = TagLook(n)
    print('=== TAG DOCK: read the AprilTag ===', flush=True)
    if args.lidar_square:
        last = [None]

        def look_tag(label):
            r = _TAG[0].read(n=15)
            if r is None:
                raise RuntimeError(f'{label}: no AprilTag seen -- not placing blind')
            print(f"  {label}: tag {r['id']} x {r['x']:.3f} y {r['y']*1000:+.0f} mm "
                  f"(spread {r['spread_y_m']*1000:.1f} mm, tag angle {r['heading_err_deg']:+.1f} deg)",
                  flush=True)
            last[0] = r
            return r['x'], r['y']
        r0 = _TAG[0].read(n=15)
        if r0 is None:
            raise RuntimeError('tag dock: no AprilTag seen -- stopping (not placing blind)')
        if r0['x'] > DOCK_X + 0.05:
            print(f"  table B: tag at x {r0['x']:.3f} -> drive {r0['x'] - DOCK_X:.3f} m in before docking",
                  flush=True)
            n.drive(r0['x'] - DOCK_X)
            time.sleep(1.0)
        try:
            lat = centre_then_square(rec, 'table B', look_tag, ok=lambda v: TAG_Y_MIN <= v <= TAG_Y_MAX,
                                     tag=lambda: last[0])
        except Abort as e:
            raise RuntimeError(f'tag dock: {e.result} -- stopping (not placing blind)')
        if not (TAG_Y_MIN <= lat <= TAG_Y_MAX):
            raise RuntimeError(f'tag y {lat*1000:+.0f} mm outside the arm band after centring -- stopping')
        print(f'  docked: placing {lat*1000:+.0f} mm sideways', flush=True)
        return lat
    shifted = False
    for attempt in range(4):
        r = _TAG[0].read(n=15)
        if r is None:
            raise RuntimeError('tag dock: no AprilTag seen -- stopping (not placing blind)')
        print(f"  tag {r['id']}: range {r['range_m']:.3f} m, square-turn {r['heading_err_deg']:+.1f} deg, "
              f"tag y {r['y']*1000:+.0f} mm (spread {r['spread_y_m']*1000:.1f} mm)", flush=True)
        lat = r['y']
        if TAG_Y_MIN <= lat <= TAG_Y_MAX:
            print(f'  docked: placing {lat*1000:+.0f} mm sideways (tag angle {r["heading_err_deg"]:+.1f} deg, logged only)', flush=True)
            rec['note'] = (rec.get('note') or '') + f" | tag y {lat*1000:+.0f} mm hdg {r['heading_err_deg']:+.1f}"
            return lat
        if shifted:
            raise RuntimeError(f'tag y {lat*1000:+.0f} mm still outside the arm band after a shift -- stopping')
        print(f'  tag y {lat*1000:+.0f} mm outside the arm band -> shift sideways', flush=True)
        shift_sideways(lat)
        if args.lidar_square:
            n.square_to_table('table B')
        shifted = True
    raise RuntimeError('tag dock: lateral not inside the arm band after the shift -- stopping')

if args.place_ahead is not None and not args.pipeline:
    place_ahead(args.place_ahead, {})
    raise SystemExit(0)
if args.pipeline:
    import yaml
    st_ = yaml.safe_load(open(os.path.expanduser('~/stations.yaml')))
    gap0 = args.place_ahead if args.place_ahead is not None else 0.50
    t_pipe = time.time()
    rec = {k: '' for k in FIELDS}
    rec.update(frozen)
    rec.update(time=time.strftime('%Y-%m-%d %H:%M:%S'), trial=1, note='PIPELINE ' + args.note)
    try:
        for k in range(args.stack_from, max(1, args.stack)):
            if k > args.stack_from:
                save(rec); summary.append(dict(rec))
                rec = {kk: '' for kk in FIELDS}
                rec.update(frozen)
                rec.update(time=time.strftime('%Y-%m-%d %H:%M:%S'), trial=k + 1,
                           note=f'PIPELINE stack level {k} ' + args.note)
                print(f'\n>>> cube {k + 1}/{args.stack}: next cube from table A', flush=True)
            print(f'\n################ CUBE {k + 1} ################', flush=True)
            STACK_SURFACE[0] = None
            arm.park()
            if args.localize:
                localize()
            nav_to('A', st_)
            SQUARE_REF[0] = st_['A']['yaw_deg']
            if args.a_forward > 0.01:
                print(f'=== FORWARD {args.a_forward:.2f} m toward table A ===', flush=True)
                n.drive(args.a_forward)
            held, px, py, ch = trial(rec, None, park_first=False, tag=f'pipe{k + 1}')
            print(f'=== PICK: {"HELD" if held else "NOT HELD"} ===', flush=True)
            if not held:
                arm.open_gripper(); arm.park()
                raise RuntimeError('pick failed -- pipeline stopped')
            print('=== CARRY POSE: arm straight up (camera clear) ===', flush=True)
            arm.park()
            print('=== REVERSE 0.30 m (clear table A) ===', flush=True)
            n.drive_back(0.30)
            nav_to('B', st_)
            SQUARE_REF[0] = st_['B']['yaw_deg']
            if args.lidar_square and not args.tag_dock:
                n.square_to_table('table B')
            tag_y = tag_dock(rec) if args.tag_dock else 0.0
            if k == 0:
                place_ahead(gap0, rec, tag_y)
            else:
                STACK_SURFACE[0] = args.place_b_height + k * CUBE_EDGE
                trial(rec, None, park_first=False, tag=f'stack{k + 1}')
                STACK_SURFACE[0] = None
                print('=== REVERSE 0.30 m (clear the stack) ===', flush=True)
                n.drive_back(0.30)
            print(f'=== CUBE {k + 1} DONE: place {rec.get("place")} ===', flush=True)
        print(f'################ PIPELINE DONE in {time.time() - t_pipe:.0f} s: '
              f'place {rec.get("place")} ################', flush=True)
    except (RuntimeError, ArmError, Abort, KeyboardInterrupt) as e:
        n.cmd.publish(Twist())
        print(f'################ PIPELINE STOPPED: {e} ################', flush=True)
    finally:
        rec['t_total'] = time.time() - t_pipe
        save(rec)
        n.cmd.publish(Twist())
    raise SystemExit(0)
try:
    for i in range(1, args.repeat + 1):
        print(f'\n################ TRIAL {i}/{args.repeat} ################', flush=True)
        rec = {k: '' for k in FIELDS}
        rec.update(frozen)
        setback = args.setback if i == 1 else None
        edge = None if setback is None else setback + FACE_TO_CENTRE
        rec.update(time=time.strftime('%Y-%m-%d %H:%M:%S'), trial=i,
                   setback='' if setback is None else setback,
                   edge_centre='' if edge is None else edge)
        carry_on = False
        t_trial = time.time()
        try:
            held, px, py, ch = trial(rec, edge, park_first=(i == 1), tag=f't{i:02d}')
            if not held:
                print('  gripper is empty — the cube may have been pushed. '
                      'Stopping the series.', flush=True)
                arm.open_gripper(); arm.park()
            else:
                print('=== 5. PUT THE CUBE BACK ===', flush=True)
                t = time.time()
                ptr = {}
                try:
                    arm.place_at(px, py, ch - CUBE_EDGE / 2.0, cube_edge=CUBE_EDGE,
                                 approach_gap=PLACE_GAP, trace=ptr)
                    rec['place'] = 'OK'
                except ArmError as e:
                    rec['place'] = 'FAILED'
                    print(f'  PLACE FAILED: {e}\n  stopping the series.', flush=True)
                finally:
                    rec.update(place_cmd_x=ptr.get('cmd_x', ''),
                               place_cmd_y=ptr.get('cmd_y', ''),
                               place_cmd_h=ptr.get('cmd_drop_h', ''))
                    flatten(rec, 'drop', {k: v for k, v in (ptr.get('drop') or {}).items()
                                          if k in ('x', 'y', 'h', 'rz')})
                    rec['t_place'] = time.time() - t
                if rec['place'] == 'OK':
                    arm.park()
                    carry_on = True
                    if i < args.repeat:
                        back = random.uniform(args.back_min, args.back_max)
                        rec['back_cmd'] = back
                        print(f'=== 6. REVERSE {back:.2f} m for the next trial ===',
                              flush=True)
                        t = time.time()
                        try:
                            n.drive_back(back)
                        except RuntimeError as e:
                            print(f'  REVERSE FAILED: {e}', flush=True)
                            carry_on = False
                        rec['t_back'] = time.time() - t
                        time.sleep(1.5)
        except Abort as a:
            rec['result'] = a.result
            print(f'  series stopped: {a.result}', flush=True)
        finally:
            rec['t_total'] = time.time() - t_trial
            save(rec)
            summary.append(dict(rec))
        if not carry_on:
            break
except KeyboardInterrupt:
    print('\n  Ctrl-C — stopping', flush=True)
finally:
    try:
        n.cmd.publish(Twist())
        n.cmd.publish(Twist())
    except Exception:
        pass
    print(f'\n================ SERIES {args.series} SUMMARY ================', flush=True)
    for r in summary:
        pe = r.get('pick_err')
        rz = r.get('grasp_rz')
        crz = r.get('cmd_rz')
        print(f'  trial {r["trial"]:>2}: {r.get("result") or "?":20s} '
              f'grip={r.get("grip_lift")!s:4s} '
              f'wrist_err={(f"{rz - crz:+.1f}deg") if isinstance(rz, float) and isinstance(crz, float) else "-":9s} '
              f'consist={mm(r["near_consistency"]) if isinstance(r.get("near_consistency"), float) else "-":8s} '
              f'pick_err={mm(pe) if isinstance(pe, float) else "-":8s} '
              f'place={r.get("place") or "-"}', flush=True)
    ok = sum(1 for r in summary if r.get('result') == 'SUCCESS')
    print(f'  {ok}/{len(summary)} picks held.  Snapshots: {SNAP_DIR}', flush=True)
    print('=' * 60, flush=True)
    if rclpy.ok():
        rclpy.shutdown()
