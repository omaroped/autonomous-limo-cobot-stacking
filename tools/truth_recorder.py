#!/usr/bin/env python3
# Records the simulator ground truth during a run: robot base, all three cubes and the gripper.
import argparse
import json
import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from gazebo_msgs.msg import LinkStates, ModelStates

TCP_IN_GRIPPER_BASE = np.array([0.0, 0.06, -0.0075])  # gripper_base_to_tcp, ackermann_with_sensor.xacro
BOXES = ['stack_box_0', 'stack_box_1', 'stack_box_2']
GRIP_LINKS = ['mbot::gripper_base', 'mbot::joint6_flange']  # first one found is used

def rot(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])

FLANGE_TO_GB = (np.array([0.0, 0.0, 0.034]), rpy(1.579, 0.0, 1.5708))

def quat(Rm):
    w = math.sqrt(max(0.0, 1 + Rm[0, 0] + Rm[1, 1] + Rm[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2])) / 2, Rm[2, 1] - Rm[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - Rm[0, 0] + Rm[1, 1] - Rm[2, 2])) / 2, Rm[0, 2] - Rm[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - Rm[0, 0] - Rm[1, 1] + Rm[2, 2])) / 2, Rm[1, 0] - Rm[0, 1])
    return [x, y, z, w]

def gripper_base_pose(link_name, p, q):
    if link_name.endswith('joint6_flange'):
        Rf = rot(q)
        return p + Rf @ FLANGE_TO_GB[0], quat(Rf @ FLANGE_TO_GB[1])
    return p, list(q)

def pose7(p):
    return [p.position.x, p.position.y, p.position.z,
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]

class Recorder(Node):
    def __init__(self, out, hz):
        super().__init__('truth_recorder', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.period = 1.0 / hz
        self.out = open(out, 'w', buffering=1)
        self.meta_path = out + '.meta.json'
        cols = ['sim_t', 'wall_t']
        for k in ['base', 'box0', 'box1', 'box2', 'grip']:
            cols += [f'{k}_{c}' for c in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')]
        cols += ['tcp_x', 'tcp_y', 'tcp_z']
        self.out.write(','.join(cols) + '\n')
        self.models = None
        self.grip = None
        self.grip_name = None
        self.last_t = -1.0
        self.rows = 0
        self.create_subscription(ModelStates, '/model_states', self._models, 10)
        self.create_subscription(LinkStates, '/link_states', self._links, 10)

    def _links(self, m):
        if self.grip_name is None:
            for g in GRIP_LINKS:
                if g in m.name:
                    self.grip_name = g
                    json.dump({'grip_link': g, 'grip_frame': 'gripper_base',
                               'tcp_offset_in_grip': TCP_IN_GRIPPER_BASE.tolist(),
                               'robot_links': [n for n in m.name if n.startswith('mbot::')]},
                              open(self.meta_path, 'w'), indent=1)
                    break
        if self.grip_name in m.name:
            self.grip = m.pose[m.name.index(self.grip_name)]

    def _models(self, m):
        t = self.get_clock().now().nanoseconds * 1e-9
        if t - self.last_t < self.period - 1e-6 or 'mbot' not in m.name:
            return
        self.last_t = t
        idx = {n: i for i, n in enumerate(m.name)}
        vals = [f'{t:.4f}', f'{time.time():.4f}']
        vals += [f'{v:.7f}' for v in pose7(m.pose[idx['mbot']])]
        for b in BOXES:
            vals += [f'{v:.7f}' for v in pose7(m.pose[idx[b]])] if b in idx else [''] * 7
        if self.grip is not None:
            g = pose7(self.grip)
            gp, gq = gripper_base_pose(self.grip_name, np.array(g[:3]), g[3:])
            tcp = gp + rot(gq) @ TCP_IN_GRIPPER_BASE
            vals += [f'{v:.7f}' for v in list(gp) + list(gq)] + [f'{v:.7f}' for v in tcp]
        else:
            vals += [''] * 10
        self.out.write(','.join(vals) + '\n')
        self.rows += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--hz', type=float, default=20.0)
    a = ap.parse_args()
    rclpy.init()
    n = Recorder(a.out, a.hz)
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, Exception):
        pass
    finally:
        n.out.close()
        print(f'truth_recorder: {n.rows} rows -> {a.out}', file=sys.stderr)

if __name__ == '__main__':
    main()
