#!/usr/bin/env python3
# Cartesian control of the real myCobot 280 through pymycobot, with checks before each move.
import time

from pymycobot.mycobot import MyCobot

BASE_TO_ARM_X   = 0.030
FINGER_Z_OFFSET = 0.0525  # fingertips sit this far below the reported flange
Z_UNDERSHOOT    = 0.008  # arm lands low by ~this at ~0.22 m extension
GRASP_RX, GRASP_RY, GRASP_RZ = -180.0, 0.0, -38.0
J2_MIN_DEG      = -85.0  # below this the upper link approaches the camera
POSE_TOL        = 0.003  # m; accept a pose within this of the target on every axis
HOVER_TOL       = 0.015  # m; the approach pose ABOVE the target only has to be roughly

GAIN_BAND       = 0.015  # m; above this a height correction is commanded 1:1, so a
GRASP_RAISE     = 0.012

LATERAL_COMP    = 0.000  # m; shift targets this far to the robot's LEFT (+y)

BACKLASH_X_COMP = 0.000  # m; feedforward reach correction -- MEASURED AS ZERO

MAX_REACH_X     = 0.243

GRIP_OPEN, GRIP_CLOSE = 100, 0
GRIP_THRESHOLD  = 15  # readback above this = holding something
GRIP_TYPE       = 4

class ArmError(RuntimeError):
    pass

class ArmController:

    def __init__(self, port='/dev/ttyACM0', baud=115200, speed=20, verbose=True):
        self.mc = MyCobot(port, baud)
        self.speed = speed
        self.verbose = verbose
        time.sleep(2)
        if not self.mc.get_angles():
            raise ArmError('arm did not reply to get_angles() — powered on? cable?')
        self.mc.set_gripper_mode(0)
        self.mc.init_eletric_gripper()
        time.sleep(1)

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def to_arm(self, base_x, base_y, floor_h):
        return ((base_y + LATERAL_COMP) * 1000.0,
                -(base_x + BACKLASH_X_COMP + BASE_TO_ARM_X) * 1000.0,
                (floor_h - FINGER_Z_OFFSET) * 1000.0)

    def state(self):
        c, a = self.mc.get_coords(), self.mc.get_angles()
        if not c or not a:
            raise ArmError('no reply from arm')
        return {
            'base_x': -c[1] / 1000.0 - BASE_TO_ARM_X,
            'base_y': c[0] / 1000.0 - LATERAL_COMP,
            'floor_h': c[2] / 1000.0 + FINGER_Z_OFFSET,
            'j2': a[1], 'coords': c, 'angles': a,
        }

    def grip_value(self):
        for t in (GRIP_TYPE, 3, 1):
            try:
                v = self.mc.get_gripper_value(t)
                if v is not None:
                    return v
            except Exception:
                pass
        return None

    def _grip_settled(self, timeout=14):
        last, stable = None, 0
        for _ in range(timeout):
            time.sleep(1)
            v = self.grip_value()
            if v is not None and v == last:
                stable += 1
                if stable >= 3:
                    return v
            else:
                stable = 0
            last = v
        return self.grip_value()

    def open_gripper(self):
        self.mc.set_gripper_value(GRIP_OPEN, 30)
        time.sleep(3)
        return self.grip_value()

    def close_gripper(self):
        self.mc.set_gripper_value(GRIP_CLOSE, 20)
        return self._grip_settled()

    def _wait_wrist(self, rz, timeout=8.0, tol_deg=2.0):
        t0 = time.time()
        while True:
            s = self.state()
            err = (s['coords'][5] - rz + 180.0) % 360.0 - 180.0
            if abs(err) <= tol_deg:
                waited = time.time() - t0
                if waited > 0.3:
                    self._log(f'  wrist settled after {waited:.1f} s (rz {s["coords"][5]:.1f}, '
                              f'commanded {rz:.1f})')
                return s
            if time.time() - t0 > timeout:
                self._log(f'  WARNING: wrist still {err:+.1f} deg off after {timeout:.0f} s '
                          f'(rz {s["coords"][5]:.1f}, commanded {rz:.1f}) — continuing')
                return s
            time.sleep(0.2)

    def move_to(self, base_x, base_y, floor_h, steps=4, settle=3.0,
                compensate=True, rz=GRASP_RZ, tol=None):
        if base_x > MAX_REACH_X:
            raise ArmError(f'{base_x:.3f} m exceeds measured reach {MAX_REACH_X:.3f} m')

        tx, ty, tz = self.to_arm(base_x, base_y, floor_h)
        if compensate:
            tz += Z_UNDERSHOOT * 1000.0

        c = self.mc.get_coords()
        for i in range(1, steps + 1):
            f = i / float(steps)
            self.mc.send_coords([c[0] + (tx - c[0]) * f,
                                 c[1] + (ty - c[1]) * f,
                                 c[2] + (tz - c[2]) * f,
                                 GRASP_RX, GRASP_RY, rz], self.speed, 0)
            time.sleep(settle)
            s = self.state()
            if s['j2'] < J2_MIN_DEG:
                raise ArmError(
                    f'J2 {s["j2"]:.1f} deg past the {J2_MIN_DEG} limit at step '
                    f'{i}/{steps} — stopped before reaching the target (camera)')

        tol = POSE_TOL if tol is None else tol

        for _ in range(3):
            s = self.state()
            ex = base_x - s['base_x']
            ey = base_y - s['base_y']
            ez = floor_h - s['floor_h']
            if max(abs(ex), abs(ey), abs(ez)) < tol:
                return self._wait_wrist(rz)
            c = self.mc.get_coords()
            self.mc.send_coords([c[0] + ey * 1000.0,
                                 c[1] - ex * 1000.0,
                                 c[2] + ez * 1000.0,
                                 GRASP_RX, GRASP_RY, rz], 15, 0)
            time.sleep(5)
            if self.state()['j2'] < J2_MIN_DEG:
                raise ArmError('J2 guard tripped while correcting position')
        s = self.state()
        err = max(abs(base_x - s['base_x']), abs(base_y - s['base_y']),
                  abs(floor_h - s['floor_h']))
        if err > tol * 2:
            raise ArmError(
                f'could not reach target: asked x={base_x:.3f} y={base_y:.3f} '
                f'h={floor_h:.3f}, reached x={s["base_x"]:.3f} y={s["base_y"]:.3f} '
                f'h={s["floor_h"]:.3f} — residual {err*1000:.0f} mm. Likely outside '
                f'the arm\'s workspace at this height.')
        return self._wait_wrist(rz)

    def settle_height(self, floor_h, base_x=None, base_y=None,
                      tries=6, tol=0.004, gain=1.4):
        hold_x = None if base_y is None else (base_y + LATERAL_COMP) * 1000.0
        hold_y = (None if base_x is None else
                  -(base_x + BACKLASH_X_COMP + BASE_TO_ARM_X) * 1000.0)

        for _ in range(tries):
            s = self.state()
            err = floor_h - s['floor_h']
            if abs(err) < tol:
                return s
            step = err * (gain if abs(err) < GAIN_BAND else 1.0)
            c = self.mc.get_coords()
            self.mc.send_coords([c[0] if hold_x is None else hold_x,
                                 c[1] if hold_y is None else hold_y,
                                 c[2] + step * 1000.0,
                                 GRASP_RX, GRASP_RY, GRASP_RZ], 15, 0)
            time.sleep(4)
            if self.state()['j2'] < J2_MIN_DEG:
                raise ArmError('J2 guard tripped while settling height')
        s = self.state()
        resid = (floor_h - s['floor_h']) * 1000.0
        if abs(resid) > tol * 1000.0:
            self._log(f'  WARNING: height did not converge — asked '
                      f'{floor_h*100:.1f} cm, reached {s["floor_h"]*100:.1f} cm '
                      f'({resid:+.0f} mm). Fingertip clearance to the table is '
                      f'{abs(resid):.0f} mm less than intended.')
        return s

    @staticmethod
    def _snap(s):
        c, a = s['coords'], s['angles']
        return {'x': s['base_x'], 'y': s['base_y'], 'h': s['floor_h'],
                'rz': c[5], 'j1': a[0], 'j2': a[1], 'j6': a[5]}

    def pick_at(self, base_x, base_y, cube_centre_h, hover=0.06, lift=0.06,
                trace=None):
        grasp_h = cube_centre_h + GRASP_RAISE
        if trace is not None:
            trace.update(cmd_x=base_x, cmd_y=base_y, cmd_hover_h=grasp_h + hover,
                         cmd_grasp_h=grasp_h, cmd_rz=GRASP_RZ)
        self._log(f'pick at base_link x={base_x:.3f} y={base_y:.3f}, '
                  f'cube centre {cube_centre_h*100:.1f} cm, '
                  f'grasping at {grasp_h*100:.1f} cm '
                  f'(+{GRASP_RAISE*1000:.0f} mm for fingertip clearance)')
        self.open_gripper()
        s = self.move_to(base_x, base_y, grasp_h + hover, tol=HOVER_TOL)
        self._log(f'  hover   x={s["base_x"]:.3f} y={s["base_y"]:.3f} '
                  f'h={s["floor_h"]*100:.1f}cm J2={s["j2"]:.1f} rz={s["coords"][5]:.1f}')
        if trace is not None:
            trace['hover'] = self._snap(s)

        s = self.settle_height(grasp_h, base_x=base_x, base_y=base_y)
        self._log(f'  descend x={s["base_x"]:.3f} y={s["base_y"]:.3f} '
                  f'h={s["floor_h"]*100:.1f}cm J2={s["j2"]:.1f} rz={s["coords"][5]:.1f}')
        if trace is not None:
            trace['grasp'] = self._snap(s)

        v_close = self.close_gripper()
        c = self.mc.get_coords()
        self.mc.send_coords([c[0], c[1], c[2] + lift * 1000.0,
                             GRASP_RX, GRASP_RY, GRASP_RZ], 15, 0)
        time.sleep(6)
        v_lift = self.grip_value()

        held = v_lift is not None and v_lift > GRIP_THRESHOLD
        if trace is not None:
            trace.update(grip_close=v_close, grip_lift=v_lift)
            try:
                trace['lifted'] = self._snap(self.state())
            except ArmError:
                pass
        self._log(f'  gripper close={v_close} lift={v_lift} -> '
                  f'{"HELD" if held else "NOT HELD"}')
        return held, v_lift

    def place_at(self, base_x, base_y, surface_h, cube_edge=0.0247,
                 hover=0.06, approach_gap=0.002, trace=None):
        centre = surface_h + cube_edge / 2.0 + approach_gap
        if trace is not None:
            trace.update(cmd_x=base_x, cmd_y=base_y, cmd_drop_h=centre)
        s = self.move_to(base_x, base_y, centre + hover, tol=HOVER_TOL)
        self._log(f'  hover   x={s["base_x"]:.3f} h={s["floor_h"]*100:.1f}cm')
        s = self.settle_height(centre, base_x=base_x, base_y=base_y)
        self._log(f'  at drop x={s["base_x"]:.3f} h={s["floor_h"]*100:.1f}cm')
        if trace is not None:
            trace['drop'] = self._snap(s)
        self.open_gripper()
        c = self.mc.get_coords()
        self.mc.send_coords([c[0], c[1], c[2] + hover * 1000.0,
                             GRASP_RX, GRASP_RY, GRASP_RZ], 15, 0)
        time.sleep(5)
        return self.state()

    def park(self):
        self.mc.send_angles([0, 0, 0, 0, 0, 0], 25)
        time.sleep(10)
        a = self.mc.get_angles()
        lim = [(-168, 168), (-135, 135), (-150, 150),
               (-145, 145), (-165, 165), (-180, 180)]
        bad = [i + 1 for i, (v, (lo, hi)) in enumerate(zip(a, lim))
               if not (lo < v < hi)]
        self._log(f'parked: {a}' + ('  ALL INSIDE LIMITS' if not bad
                                    else f'  OUT OF LIMITS: J{bad}'))
        return a, bad

def recover_joint_limits(port='/dev/ttyACM0', baud=115200):
    mc = MyCobot(port, baud)
    time.sleep(2)
    a = mc.get_angles()
    if not a:
        raise ArmError('no reply — arm powered on? cable seated?')
    lim = [(-160, 160), (-130, 130), (-145, 145),
           (-135, 135), (-160, 160), (-175, 175)]
    fixed = [max(lo, min(hi, v)) for v, (lo, hi) in zip(a, lim)]
    if fixed == a:
        print(f'all joints already legal: {a}')
        return a
    print(f'out of range {a} -> commanding {fixed}')
    mc.send_angles(fixed, 20)
    time.sleep(8)
    print(f'now: {mc.get_angles()}')
    return mc.get_angles()

if __name__ == '__main__':
    arm = ArmController()
    s = arm.state()
    print(f'base_link x={s["base_x"]:.3f} y={s["base_y"]:.3f} '
          f'height={s["floor_h"]*100:.1f}cm J2={s["j2"]:.1f} '
          f'gripper={arm.grip_value()}')
