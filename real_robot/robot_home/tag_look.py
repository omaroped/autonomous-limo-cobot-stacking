#!/usr/bin/env python3
import math
import statistics as st
import time

import cv2
import numpy as np
import rclpy
import tf2_ros
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

TAG_SIZE = 0.065
BASE = 'base_link'

def _quat_to_R(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

class TagLook:
    def __init__(self, node):
        self.n = node
        self.K = self.D = None
        self.frame = None
        self.img = None
        self.det = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11),
            cv2.aruco.DetectorParameters())
        self.tfb = tf2_ros.Buffer()
        self.tfl = tf2_ros.TransformListener(self.tfb, node)
        node.create_subscription(CameraInfo, '/camera/color/camera_info', self._info, 5)
        node.create_subscription(Image, '/camera/color/image_raw', self._image, 2)

    def _info(self, m):
        self.K = np.array(m.k, float).reshape(3, 3)
        self.D = np.array(m.d, float) if len(m.d) else np.zeros(5)
        self.frame = m.header.frame_id

    def _image(self, m):
        self.img = m

    def _gray(self, m):
        a = np.frombuffer(m.data, np.uint8)
        if m.encoding in ('rgb8', 'bgr8'):
            a = a.reshape(m.height, m.width, 3)
            return cv2.cvtColor(a, cv2.COLOR_RGB2GRAY if m.encoding == 'rgb8' else cv2.COLOR_BGR2GRAY)
        if m.encoding in ('mono8', '8UC1'):
            return a.reshape(m.height, m.width)
        raise RuntimeError(f'unsupported image encoding {m.encoding}')

    def _one(self):
        m, self.img = self.img, None
        if m is None or self.K is None:
            return None
        corners, ids, _ = self.det.detectMarkers(self._gray(m))
        if ids is None or len(ids) == 0:
            return None
        sizes = [cv2.contourArea(c.reshape(4, 2)) for c in corners]
        i = int(np.argmax(sizes))
        h = TAG_SIZE / 2.0
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
        n_sol, rvecs, tvecs, errs = cv2.solvePnPGeneric(
            obj, corners[i].reshape(4, 2).astype(np.float32), self.K, self.D,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not n_sol:
            return None
        tf = self.tfb.lookup_transform(BASE, self.frame, rclpy.time.Time())
        Rb = _quat_to_R(tf.transform.rotation)
        tb = np.array([tf.transform.translation.x, tf.transform.translation.y,
                       tf.transform.translation.z])
        best, angles = None, []
        for rvec, tvec, err in zip(rvecs, tvecs, np.asarray(errs).reshape(-1)):
            Rc, _ = cv2.Rodrigues(rvec)
            p = Rb @ np.asarray(tvec).reshape(3) + tb
            nrm = Rb @ Rc[:, 2]
            u = -nrm[:2] / max(np.linalg.norm(nrm[:2]), 1e-9)
            angles.append(math.degrees(math.atan2(u[1], u[0])))
            if best is None or err < best[0]:
                best = (err, p, nrm)
        return int(ids.flatten()[i]), best[1], best[2], angles

    def read(self, n=15, timeout=10.0):
        got = []
        end = time.time() + timeout
        while len(got) < n and time.time() < end:
            rclpy.spin_once(self.n, timeout_sec=0.05)
            try:
                r = self._one()
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException,
                    tf2_ros.ConnectivityException):
                r = None
            if r is not None:
                got.append(r)
        if len(got) < 3:
            return None
        ids = [g[0] for g in got]
        p = np.array([[st.median(g[1][k] for g in got) for k in range(3)]]).reshape(3)
        nv = np.array([st.median(g[2][k] for g in got) for k in range(3)])
        u = -nv[:2] / np.linalg.norm(nv[:2])
        heading = math.degrees(math.atan2(u[1], u[0]))
        lateral = -p[0] * u[1] + p[1] * u[0]
        rng = p[0] * u[0] + p[1] * u[1]
        spread = max(g[1][1] for g in got) - min(g[1][1] for g in got)
        return {'id': max(set(ids), key=ids.count), 'n': len(got),
                'x': float(p[0]), 'y': float(p[1]), 'z': float(p[2]),
                'heading_err_deg': heading, 'lateral_m': float(lateral), 'range_m': float(rng),
                'spread_y_m': float(spread), 'angles': [g[3] for g in got]}

def angle_near(r, ref):
    return st.median(min(a, key=lambda v: abs(v - ref)) for a in r['angles'])

def main():
    rclpy.init()
    node = Node('tag_look')
    t = TagLook(node)
    r = t.read()
    if r is None:
        print('NO TAG seen (need >= 3 detections in 10 s)')
    else:
        print(f"tag id {r['id']} (n={r['n']}): x {r['x']:.3f} y {r['y']:+.3f} z {r['z']:+.3f} m | "
              f"turn {r['heading_err_deg']:+.1f} deg to square | lateral {r['lateral_m']*1000:+.0f} mm "
              f"| range {r['range_m']:.3f} m | spread y {r['spread_y_m']*1000:.1f} mm")
    rclpy.shutdown()

if __name__ == '__main__':
    main()
