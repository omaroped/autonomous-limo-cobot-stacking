#!/usr/bin/env python3
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

LIDAR_X, LIDAR_YAW, BUMPER_X = 0.108, math.radians(-7.8), 0.161
FOV = math.radians(25)  # use beams within +-25 deg of straight ahead

class Check(Node):
    def __init__(self):
        super().__init__('station_check')
        self.scan = None
        self.cube = None
        self.create_subscription(LaserScan, '/scan', lambda m: setattr(self, 'scan', m),
                                 qos_profile_sensor_data)
        self.create_subscription(PoseStamped, '/box_pose',
                                 lambda m: setattr(self, 'cube', (time.time(), m)), 10)

    def face(self):
        s = self.scan
        if s is None:
            return None
        a = s.angle_min + np.arange(len(s.ranges)) * s.angle_increment + LIDAR_YAW
        r = np.array(s.ranges)
        ok = np.isfinite(r) & (r > s.range_min) & (np.abs(a) < FOV)
        if ok.sum() < 10:
            return None
        x = LIDAR_X + r[ok] * np.cos(a[ok])
        y = r[ok] * np.sin(a[ok])
        near = x < np.median(x) + 0.05
        if near.sum() < 8:
            return None
        k, c = np.polyfit(y[near], x[near], 1)
        return -math.degrees(math.atan(k)), (c - BUMPER_X) * 1000.0

def main():
    rclpy.init()
    n = Check()
    print('Ctrl+C to stop.  Target: face 0 +-1 deg, cube y 0 +-10 mm, gap as wanted.')
    try:
        while True:
            end = time.time() + 0.5
            while time.time() < end:
                rclpy.spin_once(n, timeout_sec=0.05)
            f = n.face()
            fs = '   no face seen' if f is None else f'turn {f[0]:+5.1f} deg to square  bumper gap {f[1]:6.0f} mm'
            cs = '  cube: none'
            if n.cube and time.time() - n.cube[0] < 1.0:
                p = n.cube[1].pose.position
                cs = f'  cube: x {p.x:.3f} m  y {p.y*1000:+5.0f} mm'
            print(fs + cs, flush=True)
    except (KeyboardInterrupt, Exception):
        pass
    if rclpy.ok():
        rclpy.shutdown()

if __name__ == '__main__':
    main()
