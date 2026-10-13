#!/usr/bin/env python3
import sys

class ArmResult:
    SUCCEEDED, FAILED, TIMED_OUT, CANCELLED = 'SUCCEEDED', 'FAILED', 'TIMED_OUT', 'CANCELLED'
    HOLDING, EMPTY, UNKNOWN = 'HOLDING', 'EMPTY', 'UNKNOWN'

    def __init__(self, status, reason='', held=None, evidence='', arm_clear=None, detail=None):
        self.status = status
        self.reason = reason
        self.held = held if held is not None else ArmResult.UNKNOWN
        self.evidence = evidence
        self.arm_clear = arm_clear
        self.detail = detail or {}

    @property
    def ok(self):
        return self.status == ArmResult.SUCCEEDED

    def __repr__(self):
        return (f'ArmResult({self.status}, held={self.held}, arm_clear={self.arm_clear}, '
                f'reason={self.reason!r}, evidence={self.evidence!r})')

class RealArmBackend:
    name = 'real'

    def __init__(self, drop_point, table_top_h, base_link_height=0.15, cube_edge=0.0247,
                 grasp_back=0.002, release_gap=0.008, arm_module_dir='/home/agilex',
                 port='/dev/ttyACM0', arm=None, log=print):
        if arm_module_dir not in sys.path:
            sys.path.insert(0, arm_module_dir)
        import arm_control
        self._ac = arm_control
        self.arm = arm if arm is not None else arm_control.ArmController(port=port)
        self.drop_point = drop_point
        self.table_top_h = float(table_top_h)
        self.base_h = float(base_link_height)
        self.cube_edge = float(cube_edge)
        self.grasp_back = float(grasp_back)
        self.release_gap = float(release_gap)
        self.log = log
        self._holding = False

    def _reach_error(self, x):
        if x > self._ac.MAX_REACH_X:
            return f'x {x:.3f} m beyond the measured reach {self._ac.MAX_REACH_X:.3f} m'
        return None

    def pick(self, x, y, z):
        gx = x - self.grasp_back
        err = self._reach_error(gx)
        if err:
            return ArmResult(ArmResult.FAILED, reason='target rejected: ' + err,
                             held=ArmResult.EMPTY if not self._holding else ArmResult.HOLDING,
                             evidence='no motion', arm_clear=True)
        tr = {}
        try:
            held, v_lift = self.arm.pick_at(gx, y, z + self.base_h, trace=tr)
        except self._ac.ArmError as e:
            return ArmResult(ArmResult.FAILED, reason=f'arm aborted: {e}',
                             held=ArmResult.UNKNOWN, evidence='arm aborted mid-motion',
                             arm_clear=False, detail=tr)
        self._holding = bool(held)
        return ArmResult(ArmResult.SUCCEEDED if held else ArmResult.FAILED,
                         reason='' if held else 'gripper empty after lift',
                         held=ArmResult.HOLDING if held else ArmResult.EMPTY,
                         evidence=f'gripper readback after lift {v_lift} '
                                  f'(> {self._ac.GRIP_THRESHOLD} = holding)',
                         arm_clear=None, detail=tr)

    def place(self, level):
        dp = self.drop_point()
        if dp is None:
            return ArmResult(ArmResult.FAILED, reason='no drop point (dock not latched)',
                             held=ArmResult.HOLDING if self._holding else ArmResult.UNKNOWN,
                             evidence='no motion', arm_clear=True)
        x, y = float(dp[0]), float(dp[1])
        err = self._reach_error(x)
        if err:
            return ArmResult(ArmResult.FAILED, reason='target rejected: ' + err,
                             held=ArmResult.HOLDING if self._holding else ArmResult.UNKNOWN,
                             evidence='no motion', arm_clear=True)
        surface_h = self.table_top_h + level * self.cube_edge
        tr = {}
        try:
            self.arm.place_at(x, y, surface_h, cube_edge=self.cube_edge,
                              approach_gap=self.release_gap, trace=tr)
        except self._ac.ArmError as e:
            return ArmResult(ArmResult.FAILED, reason=f'arm aborted: {e}',
                             held=ArmResult.UNKNOWN, evidence='arm aborted mid-motion',
                             arm_clear=False, detail=tr)
        self._holding = False
        tr['surface_h'] = surface_h
        return ArmResult(ArmResult.SUCCEEDED,
                         held=ArmResult.EMPTY,
                         evidence='gripper open command completed (placement not verified)',
                         arm_clear=None, detail=tr)

    def prepare_travel(self):
        try:
            if self._holding:
                s = self.arm.state()
                ok = s['j2'] >= self._ac.J2_MIN_DEG
                return ArmResult(ArmResult.SUCCEEDED if ok else ArmResult.FAILED,
                                 reason='' if ok else f"J2 {s['j2']:.1f} deg past the guard",
                                 held=ArmResult.HOLDING, evidence='carry in the lifted pick pose',
                                 arm_clear=ok, detail={'angles': s['angles']})
            angles, bad = self.arm.park()
            off = [round(a, 1) for a in angles if abs(a) > 2.0]
            ok = not bad and not off
            return ArmResult(ArmResult.SUCCEEDED if ok else ArmResult.FAILED,
                             reason='' if ok else f'park not confirmed: out of limits {bad}, '
                                                  f'>2 deg from zero {off}',
                             held=ArmResult.EMPTY, evidence='parked',
                             arm_clear=ok, detail={'angles': angles})
        except self._ac.ArmError as e:
            return ArmResult(ArmResult.FAILED, reason=f'arm error: {e}', arm_clear=False)

    def stop(self):
        try:
            self.arm.mc.stop()
            return ArmResult(ArmResult.SUCCEEDED, evidence='stop command sent (untested)',
                             arm_clear=False)
        except Exception as e:
            return ArmResult(ArmResult.FAILED, reason=f'stop failed: {e}', arm_clear=False)
