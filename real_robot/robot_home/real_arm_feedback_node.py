#!/usr/bin/env python3
# The only program that talks to the arm over serial. It publishes the real joint angles
# (25 Hz) so MoveIt plans from where the arm really is, and sends MoveIt's joint commands
# to the arm (at most 10 Hz).
import csv
import math
import os
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32
from pymycobot.mycobot import MyCobot

JOINTS = ['joint2_to_joint1', 'joint3_to_joint2', 'joint4_to_joint3',
          'joint5_to_joint4', 'joint6_to_joint5', 'joint6output_to_joint6']
READ_HZ = 25.0
WRITE_HZ = 10.0
SPEED = 20
START_TOL_DEG = 3.0
MAX_STEP_DEG = 20.0
MIN_STEP_DEG = 0.2
J2_MIN_DEG = -85.0
STALE_S = 2.0
READ_FRESH_S = 0.2
MAX_PAUSES = 3
SETTLE_S = 0.6
CORR_TOL = 0.3
CORR_MAX = 3.0
CORR_TRIES = 3
SEG_MAX_DEG = 150.0
SEG_RESEND_S = 1.5
GRIP_TYPE = 4  # as arm_control.py
GRIP_READ_S = 15.0
INIT_FILE = '/tmp/real_initial_positions.yaml'  # real start pose, for MoveIt's initial state

class RealArmFeedback(Node):
    def __init__(self):
        super().__init__('real_arm_feedback')
        self.mc = MyCobot('/dev/ttyACM0', 115200)
        real = None
        for _ in range(4):
            time.sleep(2.0)
            real = self.mc.get_angles()
            if real and real != -1 and len(real) == 6:
                break
        if not real or real == -1 or len(real) != 6:
            raise RuntimeError('arm did not answer get_angles() on /dev/ttyACM0')
        self.real = list(real)
        self.t_real = time.time()
        self.pauses = 0
        with open(INIT_FILE, 'w') as f:
            f.write('initial_positions:\n' + ''.join(
                f'  {j}: {math.radians(a):.5f}\n' for j, a in zip(JOINTS, real)))
        self.t_cmd_change = time.time()
        self.real_hist = []
        self.tries = 0
        self.resends = 0
        self.real_at_sent = None
        self.corr_step = 0.0
        self.state = 'WAIT_FIRST'
        self.cmd = None
        self.cmd_prev = None
        self.sent = None
        self.t_sent = 0.0
        self.seg_new = None
        self.t_seg = 0.0
        self.seg_resends = 0
        self.seg_speed = SPEED
        self.stream_cmd = None
        self.t_stream_change = 0.0
        self.pub = self.create_publisher(JointState, '/real_arm/joint_states', 10)
        self.create_subscription(JointState, '/real_arm/joint_commands', self.on_cmd, 10)
        self.create_subscription(JointState, '/real_arm/segment', self.on_segment, 10)
        self.grip_cmd = None
        self.grip_until = 0.0
        self.t_grip_read = 0.0
        self.grip_pub = self.create_publisher(Int32, '/real_arm/gripper_value', 10)
        self.create_subscription(Int32, '/real_arm/gripper_cmd', self.on_grip, 10)
        self.create_timer(1.0 / READ_HZ, self.cycle)
        path = os.path.expanduser(time.strftime('~/real_arm_feedback_%Y%m%d_%H%M%S.csv'))
        self.f = open(path, 'w', newline='')
        self.w = csv.writer(self.f)
        self.w.writerow(['t', 'state'] + [f'real_{i}' for i in range(1, 7)]
                        + [f'cmd_{i}' for i in range(1, 7)] + [f'sent_{i}' for i in range(1, 7)])
        self.get_logger().info(f'real arm at start (deg): {[round(a, 2) for a in real]}; '
                               f'publishing /real_arm/joint_states at {READ_HZ:.0f} Hz; log {path}')

    def stop(self, reason):
        if not self.state.startswith('STOPPED'):
            self.state = f'STOPPED({reason})'
            try:
                self.mc.stop()
            except Exception as e:
                self.get_logger().error(f'firmware stop failed: {e}')
            self.get_logger().error(f'FORWARDING STOPPED: {reason}. Firmware STOP sent.')

    def pause(self, reason):
        self.pauses += 1
        if self.pauses > MAX_PAUSES:
            self.stop(f'{reason} ({self.pauses}th time)')
            return
        self.state = f'PAUSED({reason})'
        try:
            self.mc.stop()
        except Exception as e:
            self.get_logger().error(f'firmware stop failed: {e}')
        self.get_logger().error(f'FORWARDING PAUSED ({self.pauses}/{MAX_PAUSES}): {reason}. '
                                f'Firmware STOP sent; waiting for the readings.')

    def on_cmd(self, msg):
        try:
            c = [math.degrees(msg.position[msg.name.index(j)]) for j in JOINTS]
        except (ValueError, IndexError):
            return
        if self.state == 'SEGMENT':
            if self.stream_cmd is None or max(abs(a - b) for a, b in zip(c, self.stream_cmd)) > 1e-3:
                self.t_stream_change = time.time()
            self.stream_cmd = c
            return
        if self.cmd is None or max(abs(a - b) for a, b in zip(c, self.cmd)) > 1e-3:
            self.t_cmd_change, self.tries = time.time(), 0
        self.cmd = c

    def on_segment(self, msg):
        try:
            g = [math.degrees(msg.position[msg.name.index(j)]) for j in JOINTS]
        except (ValueError, IndexError):
            return
        speed = int(msg.velocity[0]) if msg.velocity else SPEED
        self.seg_new = (g, max(1, min(30, speed)))

    def start_segment(self, now):
        goal, speed = self.seg_new
        self.seg_new = None
        if self.state.startswith(('STOPPED', 'PAUSED')) or now - self.t_real > READ_FRESH_S:
            self.get_logger().error(f'segment refused: state {self.state}, last reading '
                                    f'{now - self.t_real:.2f} s old')
            return
        if goal[1] < J2_MIN_DEG:
            self.stop(f'camera guard: segment J2 {goal[1]:.1f} < {J2_MIN_DEG}')
            return
        size = max(abs(g - r) for g, r in zip(goal, self.real))
        if size > SEG_MAX_DEG:
            self.stop(f'segment guard: {size:.0f} deg from the real arm')
            return
        self.mc.send_angles(list(goal), speed)
        self.seg_speed = speed
        if self.state != 'SEGMENT':
            self.stream_cmd = self.cmd
        self.t_stream_change = 0.0
        self.state = 'SEGMENT'
        self.cmd, self.sent, self.cmd_prev = list(goal), list(goal), list(goal)
        self.t_sent = self.t_cmd_change = self.t_seg = now
        self.tries, self.resends, self.seg_resends = 0, 0, 0
        self.real_at_sent = list(self.real)
        self.get_logger().info(f'segment: {size:.1f} deg in one command, speed {speed}')

    def leave_segment(self, now):
        sc = self.stream_cmd
        if (sc is None or self.t_stream_change <= self.t_seg or now - self.t_real > READ_FRESH_S
                or max(abs(c - r) for c, r in zip(sc, self.real)) > START_TOL_DEG):
            return False
        self.state, self.cmd, self.sent, self.cmd_prev, self.tries = 'FORWARDING', list(sc), None, None, 0
        self.t_cmd_change = now
        self.get_logger().info('segment done -- forwarding the MoveIt stream again')
        return True

    def seg_resend(self, now):
        if (self.seg_resends < 2 and now - self.t_sent > SEG_RESEND_S and self.arm_stopped(now)
                and max(abs(c - r) for c, r in zip(self.cmd, self.real)) > CORR_MAX):
            self.seg_resends += 1
            self.get_logger().warn(f'segment not reached -- resend {self.seg_resends}/2')
            self.mc.send_angles(list(self.cmd), self.seg_speed)
            self.t_sent = now
            return True
        return False

    def on_grip(self, msg):
        self.grip_cmd = max(-1, min(100, int(msg.data)))

    def gripper(self, now):
        if self.grip_cmd is not None:
            if self.t_sent == now:
                return
            if self.grip_cmd < 0:
                self.grip_until, self.t_grip_read = now + 5.0, 0.0
            elif self.state.startswith('STOPPED'):
                self.get_logger().error('gripper command refused: forwarding has stopped')
            else:
                v = self.grip_cmd
                self.mc.set_gripper_value(v, 15 if v > 50 else 20)
                self.get_logger().info(f'gripper -> {v}')
                self.grip_until, self.t_grip_read = now + GRIP_READ_S, now
            self.grip_cmd = None
        elif now < self.grip_until and now - self.t_grip_read >= 1.0:
            self.t_grip_read = now
            v = self.mc.get_gripper_value(GRIP_TYPE)
            if v is not None and v != -1:
                self.grip_pub.publish(Int32(data=int(v)))

    def arm_stopped(self, now):
        old = [a for t, a in self.real_hist if now - t >= 0.3]
        return bool(old) and max(abs(a - b) for a, b in zip(old[-1], self.real)) < 0.1

    def cycle(self):
        if not rclpy.ok():
            return
        now = time.time()
        a = self.mc.get_angles()
        if a and a != -1 and len(a) == 6:
            self.real, self.t_real = list(a), now
            if self.state.startswith('PAUSED'):
                self.state = 'REARM'
                self.get_logger().warn('arm readings are back -- waiting for a command at the real pose')
            self.real_hist = [(t, v) for t, v in self.real_hist if now - t < 1.0] + [(now, list(a))]
            js = JointState()
            js.header.stamp = self.get_clock().now().to_msg()
            js.name = list(JOINTS)
            js.position = [math.radians(v) for v in self.real]
            self.pub.publish(js)
        elif now - self.t_real > STALE_S and self.state in ('WAIT_FIRST', 'FORWARDING', 'REARM', 'SEGMENT'):
            self.pause(f'no arm reading for {now - self.t_real:.2f} s')
        self.forward(now)
        self.gripper(now)
        self.w.writerow([f'{now:.3f}', self.state] + [f'{v:.2f}' for v in self.real]
                        + ([f'{v:.2f}' for v in self.cmd] if self.cmd else [''] * 6)
                        + ([f'{v:.2f}' for v in self.sent] if self.sent else [''] * 6))

    def forward(self, now):
        if self.seg_new is not None:
            self.start_segment(now)
        if self.cmd is None or self.state.startswith(('STOPPED', 'PAUSED')):
            return
        seg = self.state == 'SEGMENT' and not self.leave_segment(now)
        diff_real = max(abs(c - r) for c, r in zip(self.cmd, self.real))
        fresh = now - self.t_real <= READ_FRESH_S
        if self.state == 'REARM':
            if not fresh or diff_real > START_TOL_DEG:
                return
            self.state, self.sent, self.cmd_prev, self.tries = 'FORWARDING', None, None, 0
            self.get_logger().warn(f're-armed ({diff_real:.2f} deg) -- forwarding')
        if self.state == 'WAIT_FIRST':
            if diff_real > START_TOL_DEG:
                self.stop(f'start guard: first command {diff_real:.1f} deg from the real arm')
                return
            self.state = 'FORWARDING'
            self.get_logger().info(f'start guard OK ({diff_real:.2f} deg) -- forwarding')
        if seg:
            pass
        elif not fresh and self.sent is not None:
            diff_sent = max(abs(c - v) for c, v in zip(self.cmd, self.sent))
            if diff_sent > MAX_STEP_DEG:
                self.stop(f'step guard (no fresh reading): command {diff_sent:.1f} deg from the last sent')
                return
        elif diff_real > MAX_STEP_DEG:
            self.stop(f'step guard: command {diff_real:.1f} deg from the real arm')
            return
        if self.cmd[1] < J2_MIN_DEG:
            self.stop(f'camera guard: commanded J2 {self.cmd[1]:.1f} < {J2_MIN_DEG}')
            return
        if now - self.t_sent < 1.0 / WRITE_HZ:
            return
        if os.environ.get('FB_DEBUG') and now - getattr(self, '_dbg_t', 0) > 1.0:
            self._dbg_t = now
            self.get_logger().info(
                f'dbg settle {now - self.t_cmd_change:.2f} tries {self.tries} '
                f'stopped {self.arm_stopped(now)} since_sent {now - self.t_sent:.2f} '
                f'maxerr {max(abs(c - r) for c, r in zip(self.cmd, self.real)):.2f}')
        if seg and fresh and self.seg_resend(now):
            return
        if (fresh and now - self.t_cmd_change > SETTLE_S and self.tries < CORR_TRIES
                and self.arm_stopped(now) and now - self.t_sent > SETTLE_S):
            err = [c - r for c, r in zip(self.cmd, self.real)]
            if (self.tries > 0 and self.resends < 2 and self.real_at_sent is not None
                    and max(abs(e) for e in err) > CORR_TOL
                    and self.corr_step >= 1.0
                    and max(abs(a - b) for a, b in zip(self.real, self.real_at_sent)) < 0.1):
                self.resends += 1
                self.get_logger().warn(f'arm did not respond -- resend {self.resends}/2')
                self.mc.send_angles(list(self.sent), SPEED)
                self.t_sent = now
                return
            base = self.sent if self.sent is not None else self.cmd
            corr = [b + (e if abs(e) > CORR_TOL else 0.0) for b, e in zip(base, err)]
            if (CORR_TOL < max(abs(e) for e in err)
                    and max(abs(k - c) for k, c in zip(corr, self.cmd)) <= CORR_MAX):
                if corr[1] < J2_MIN_DEG:
                    self.stop(f'camera guard: correction J2 {corr[1]:.1f} < {J2_MIN_DEG}')
                    return
                self.tries += 1
                self.get_logger().info(f'final correction {self.tries}/{CORR_TRIES}: max error '
                                       f'{max(abs(e) for e in err):.2f} deg')
                self.mc.send_angles(corr, SPEED)
                self.corr_step = max(abs(k - b) for k, b in zip(corr, base))
                self.sent, self.t_sent = corr, now
                self.real_at_sent, self.resends = list(self.real), 0
                return
        if self.tries > 0 or seg:
            return
        settled = self.cmd_prev is not None and \
            max(abs(c - p) for c, p in zip(self.cmd, self.cmd_prev)) < 1e-3
        self.cmd_prev = list(self.cmd)
        ref = self.sent if self.sent is not None else self.real
        change = max(abs(c - s) for c, s in zip(self.cmd, ref))
        if change < MIN_STEP_DEG and not (settled and change > 1e-3):
            return
        self.mc.send_angles(list(self.cmd), SPEED)
        self.sent, self.t_sent = list(self.cmd), now

def main():
    rclpy.init()
    n = RealArmFeedback()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        n.f.close()
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
