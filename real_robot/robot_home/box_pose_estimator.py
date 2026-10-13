#!/usr/bin/python3
# Finds the blue cube in the robot camera (colour filter + depth) and publishes its position in
# base_link on /box_pose. The depth point lies on the cube's front face; face_and_pick.py adds
# half a cube edge to get the centre.

import time

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64

import tf2_ros
from tf2_geometry_msgs import do_transform_point

from cv_bridge import CvBridge
import cv2
import numpy as np

BLUE_HSV_LO = np.array([85, 100, 30])
BLUE_HSV_HI = np.array([135, 255, 255])

MIN_CONTOUR_AREA = 30  # px², ignore noise
MASK_MIN_PX      = 5
TARGET_FRAME     = 'base_link'  # MoveIt arm-planning frame

DEPTH_MIN_DEFAULT = 0.15  # = sensor.xacro <near>. Real Orbbec DaBai is 0.30 --
DEPTH_MAX_DEFAULT = 3.0  # override via the depth_min parameter on hardware.

SEARCH_X_MIN_DEFAULT = 0.15  # = near clip; nothing closer is measurable
SEARCH_X_MAX_DEFAULT = 2.50  # wide acquisition range
SEARCH_Y_ABS_DEFAULT = 1.20  # full FOV coverage across entire room
SEARCH_Z_MIN_DEFAULT = -0.60  # accommodates lower stand heights in lab
SEARCH_Z_MAX_DEFAULT = 0.35  # above table

BOX_EDGE_M          = 0.035  # DEFAULT = SIM cube edge (worlds/final_map.world).
MIN_AREA_FRACTION   = 0.02  # robust for partial occlusion / oblique views

MAX_AREA_FACTOR     = 4.0

FRONT_TO_CENTRE = 0.161 - 0.01235

def draw_overlay(bgr, best, rejected, cx, window):
    vis = bgr.copy()
    h, w = vis.shape[:2]
    font, aa = cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA
    white, light, green, orange, red = (255, 255, 255), (200, 200, 200), (80, 220, 80), (0, 170, 255), (70, 70, 240)
    top, bottom = 30, 24
    c = int(round(cx))
    for y0 in range(top + 6, h - bottom, 14):
        cv2.line(vis, (c, y0), (c, min(y0 + 7, h - bottom)), white, 1, aa)
    for (u, v) in rejected:
        cv2.circle(vis, (u, v), 10, red, 1, aa)
        cv2.putText(vis, 'outside', (u + 13 if u < w - 70 else u - 66, v + 4), font, 0.4, red, 1, aa)
    reading = ''
    if best is not None:
        x, y, u, v, area = best
        s = int(np.sqrt(area) / 2) + 8
        cv2.rectangle(vis, (u - s, v - s), (u + s, v + s), green, 2, aa)
        cv2.drawMarker(vis, (u, v), green, cv2.MARKER_CROSS, 10, 1, aa)
        cv2.line(vis, (c, v), (u, v), green, 1, aa)
        status, colour = 'CUBE', green
        reading = (f'front->centre {(x - FRONT_TO_CENTRE) * 100:.1f} cm   '
                   f'{abs(y) * 100:.1f} cm {"LEFT" if y > 0 else "RIGHT"}')
    elif rejected:
        status, colour = 'OUTSIDE SEARCH AREA', orange
    else:
        status, colour = 'SEARCHING', orange
    vis[:top] = (vis[:top] * 0.3).astype(np.uint8)
    vis[h - bottom:] = (vis[h - bottom:] * 0.3).astype(np.uint8)
    (tw, _), _ = cv2.getTextSize(status, font, 0.5, 1)
    cv2.rectangle(vis, (6, 6), (18 + tw, top - 6), colour, -1)
    cv2.putText(vis, status, (12, top - 10), font, 0.5, (0, 0, 0), 1, aa)
    if reading:
        (rw, _), _ = cv2.getTextSize(reading, font, 0.55, 1)
        cv2.putText(vis, reading, (w - rw - 10, top - 10), font, 0.55, white, 1, aa)
    area_txt = (f'search area: {window[0] * 100:.0f}-{window[1] * 100:.0f} cm from robot centre, '
                f'{window[2] * 100:.0f} cm each side' if window else 'search area OFF')
    cv2.putText(vis, area_txt, (8, h - 8), font, 0.4, light, 1, aa)
    (lw, _), _ = cv2.getTextSize('dashed line = robot centre', font, 0.4, 1)
    cv2.putText(vis, 'dashed line = robot centre', (w - lw - 8, h - 8), font, 0.4, light, 1, aa)
    return vis

class BoxPoseEstimator(Node):

    def __init__(self):
        super().__init__('box_pose_estimator')

        self.bridge = CvBridge()
        self._depth = None
        self._fx = self._fy = None
        self._cx = self._cy = None
        self._img_w = self._img_h = None
        self._depth_frame = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.declare_parameter('target_policy', 'nearest')

        self.declare_parameter('search_enabled', True)
        self.declare_parameter('search_x_min', SEARCH_X_MIN_DEFAULT)
        self.declare_parameter('search_x_max', SEARCH_X_MAX_DEFAULT)
        self.declare_parameter('search_y_abs', SEARCH_Y_ABS_DEFAULT)
        self.declare_parameter('search_z_min', SEARCH_Z_MIN_DEFAULT)
        self.declare_parameter('search_z_max', SEARCH_Z_MAX_DEFAULT)
        self.declare_parameter('depth_min', DEPTH_MIN_DEFAULT)
        self.declare_parameter('depth_mode', 'window')
        self.declare_parameter('depth_max', DEPTH_MAX_DEFAULT)
        self.declare_parameter('min_area_fraction', MIN_AREA_FRACTION)
        self.declare_parameter('max_area_factor', MAX_AREA_FACTOR)
        self.declare_parameter('box_edge_m', BOX_EDGE_M)

        self.create_subscription(CameraInfo, '/depth_camera/depth/camera_info',
                                 self._caminfo_cb, 10)
        self.create_subscription(Image, '/depth_camera/depth/image_raw',
                                 self._depth_cb, 10)
        self.create_subscription(Image, '/rgb/image_raw',
                                 self._rgb_cb, 10)

        self.pose_pub   = self.create_publisher(PoseStamped, '/box_pose', 10)
        self.marker_pub = self.create_publisher(Marker, '/box_pose_marker', 10)
        self.debug_pub  = self.create_publisher(Image, '/box_detection_debug', 10)
        self.age_pub    = self.create_publisher(Float64, '/box_detection_age', 10)

        self._last_pose_map = None
        self._last_real_detect_t = None
        self.create_timer(0.05, self._publish_latched)

        self.get_logger().info('box_pose_estimator started — waiting for camera…')

    def _caminfo_cb(self, msg: CameraInfo):
        self._fx, self._fy = msg.k[0], msg.k[4]
        self._cx, self._cy = msg.k[2], msg.k[5]
        self._img_w, self._img_h = msg.width, msg.height

    def _depth_cb(self, msg: Image):
        try:
            if msg.encoding == '16UC1':
                depth_mm = self.bridge.imgmsg_to_cv2(msg, '16UC1')
                d = depth_mm.astype(np.float32) / 1000.0
            else:
                d = self.bridge.imgmsg_to_cv2(msg, '32FC1').astype(np.float32)
                finite_vals = d[np.isfinite(d) & (d > 0)]
                if len(finite_vals) > 0 and np.median(finite_vals) > 10.0:
                    d = d / 1000.0
            self._depth = d
            self._depth_frame = msg.header.frame_id
        except Exception as e:
            self.get_logger().warn(f'depth convert failed: {e}', throttle_duration_sec=5.0)

    def _rgb_cb(self, msg: Image):
        if self._depth is None or self._fx is None or self._depth_frame is None:
            self.get_logger().info('Waiting for depth + camera_info…',
                                   throttle_duration_sec=3.0)
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'rgb convert failed: {e}', throttle_duration_sec=5.0)
            return

        h, w = bgr.shape[:2]
        depth = self._depth
        if depth.shape[0] != h or depth.shape[1] != w:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        candidates = self._detect_box_pixels(bgr, depth, h, w)
        if not candidates:
            self.get_logger().info('No blue box visible.', throttle_duration_sec=3.0)
            self._publish_debug_image(bgr, msg.header, None, [])
            return

        try:
            tf_base = self.tf_buffer.lookup_transform(
                TARGET_FRAME, self._depth_frame,
                rclpy.time.Time(), timeout=Duration(seconds=0.2))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'TF {self._depth_frame}→{TARGET_FRAME} unavailable: {e}',
                throttle_duration_sec=3.0)
            return

        kept, rejected, too_small, too_large = [], [], [], []
        for (cu, cv_, cz, carea) in candidates:
            cbase = do_transform_point(self._deproject(cu, cv_, cz), tf_base)
            entry = (cbase, cu, cv_, cz, carea)
            if not self._in_search_window(cbase.point):
                rejected.append(entry)
            else:
                verdict = self._size_verdict(carea, cz)
                if verdict == 'small':
                    too_small.append(entry)
                elif verdict == 'large':
                    too_large.append(entry)
                else:
                    kept.append(entry)

        if too_large:
            hi = float(self.get_parameter('max_area_factor').value)
            self.get_logger().warn(
                f'{len(too_large)} blob(s) TOO LARGE to be the box, rejected: ' +
                ', '.join(f'{e[4]:.0f}px2 at {e[3]:.2f} m '
                          f'(max {self._expected_area(e[3]) * hi:.0f}px2)'
                          for e in too_large[:3]),
                throttle_duration_sec=2.0)

        if too_small:
            frac = float(self.get_parameter('min_area_fraction').value)
            self.get_logger().warn(
                f'{len(too_small)} blob(s) too small to be the box, rejected: ' +
                ', '.join(f'{e[4]:.0f}px² at {e[3]:.2f} m '
                          f'(need {self._expected_area(e[3]) * frac:.0f}px²)'
                          for e in too_small[:3]),
                throttle_duration_sec=2.0)

        if rejected:
            self.get_logger().info(
                f'{len(rejected)} blob(s) outside the search window, rejected: ' +
                ', '.join(f'({r[0].point.x:.3f}, {r[0].point.y:+.3f}, {r[0].point.z:+.3f})'
                          for r in rejected[:3]),
                throttle_duration_sec=2.0)

        if not kept:
            self.get_logger().warn(
                f'All {len(candidates)} blue blob(s) fell outside the search window — '
                f'nothing published. If this is wrong, widen it live, e.g. '
                f'`ros2 param set /box_pose_estimator search_y_abs 0.9`',
                throttle_duration_sec=3.0)
            self._publish_debug_image(bgr, msg.header, None, rejected)
            return

        policy = self.get_parameter('target_policy').value
        if policy == 'rightmost':
            best = min(kept, key=lambda s: s[0].point.y)
        elif policy == 'largest':
            best = max(kept, key=lambda s: s[4])
        else:
            best = min(kept, key=lambda s: s[3])
        if len(kept) > 1:
            self.get_logger().info(
                f'{len(kept)} box(es) in window — targeting {policy} '
                f'(x={best[0].point.x:.3f} y={best[0].point.y:+.3f} '
                f'z={best[0].point.z:+.3f}, area={best[4]:.0f}px²)',
                throttle_duration_sec=1.0)

        pt_base, u, v, z, _ = best
        pt = self._deproject(u, v, z)

        try:
            tf_map = self.tf_buffer.lookup_transform(
                'map', self._depth_frame,
                rclpy.time.Time(), timeout=Duration(seconds=0.2))
            self._last_pose_map = do_transform_point(pt, tf_map)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            self.get_logger().info(
                'map frame unavailable — publishing live detections only, no latch',
                throttle_duration_sec=5.0)

        self._last_real_detect_t = time.time()

        pose = PoseStamped()
        pose.header.frame_id = TARGET_FRAME
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position = pt_base.point
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)
        self._publish_marker(pose)
        self._publish_debug_image(bgr, msg.header, best, rejected)

        self.get_logger().info(
            f'box @ base_link: x={pt_base.point.x:.3f} y={pt_base.point.y:.3f} '
            f'z={pt_base.point.z:.3f} (cam Z={z:.3f})', throttle_duration_sec=1.0)

    def _deproject(self, u, v, z):
        pt = PointStamped()
        pt.header.frame_id = self._depth_frame
        pt.header.stamp = rclpy.time.Time().to_msg()
        pt.point.x = float((u - self._cx) * z / self._fx)
        pt.point.y = float((v - self._cy) * z / self._fy)
        pt.point.z = float(z)
        return pt

    def _in_camera_view(self, p):
        if self._fx is None or self._img_w is None:
            return True
        if p.z <= 0.0:
            return False
        u = p.x * self._fx / p.z + self._cx
        v = p.y * self._fy / p.z + self._cy
        return 0 <= u < self._img_w and 0 <= v < self._img_h

    def _expected_area(self, depth):
        if self._fx is None or depth <= 0:
            return 0.0
        edge = float(self.get_parameter('box_edge_m').value)
        w = edge * self._fx / depth
        return w * w

    def _is_plausible_size(self, area, depth):
        return self._size_verdict(area, depth) == 'ok'

    def _size_verdict(self, area, depth):
        expected = self._expected_area(depth)
        if expected <= 0.0:
            return 'ok'
        frac = float(self.get_parameter('min_area_fraction').value)
        if frac > 0.0 and area < frac * expected:
            return 'small'
        hi = float(self.get_parameter('max_area_factor').value)
        if hi > 0.0 and area > hi * expected:
            return 'large'
        return 'ok'

    def _in_search_window(self, p):
        if not self.get_parameter('search_enabled').value:
            return True
        return (self.get_parameter('search_x_min').value <= p.x
                <= self.get_parameter('search_x_max').value
                and abs(p.y) <= self.get_parameter('search_y_abs').value
                and self.get_parameter('search_z_min').value <= p.z
                <= self.get_parameter('search_z_max').value)

    def _detect_box_pixels(self, bgr, depth, h, w):
        d_min = float(self.get_parameter('depth_min').value)
        d_max = float(self.get_parameter('depth_max').value)

        b = bgr[:, :, 0].astype(int)
        g = bgr[:, :, 1].astype(int)
        r = bgr[:, :, 2].astype(int)
        rgb_mask = ((b > g + 15) & (b > r + 25) & (b > 50)).astype(np.uint8) * 255

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, BLUE_HSV_LO, BLUE_HSV_HI)
        mask = hsv_mask

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        c_res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = c_res[0] if len(c_res) == 2 else c_res[1]

        candidates = []
        n_bad_depth = 0
        n_no_cube_depth = 0
        use_mask = str(self.get_parameter('depth_mode').value) == 'mask'
        erode_k = np.ones((5, 5), np.uint8)
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_CONTOUR_AREA:
                continue
            m = cv2.moments(c)
            if m['m00'] == 0:
                continue
            u = int(m['m10'] / m['m00'])
            v = int(m['m01'] / m['m00'])

            if use_mask:
                x0, y0, bw, bh = cv2.boundingRect(c)
                sub = np.zeros((bh, bw), np.uint8)
                cv2.drawContours(sub, [c - np.array([x0, y0])], -1, 255, -1)
                sub = cv2.erode(sub, erode_k)
                d = depth[y0:y0 + bh, x0:x0 + bw][sub > 0]
                d = d[np.isfinite(d) & (d >= d_min) & (d <= d_max)]
                if d.size >= MASK_MIN_PX:
                    candidates.append((u, v, float(np.median(d)), float(area)))
                else:
                    n_no_cube_depth += 1
                continue

            depths = []
            for dy in range(-10, 20):
                for dx in range(-18, 19):
                    ny, nx = v + dy, u + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        d = depth[ny, nx]
                        if np.isfinite(d) and d_min <= d <= d_max:
                            depths.append(d)
            if depths:
                z = float(np.median(depths))
                candidates.append((u, v, z, float(area)))
            else:
                n_bad_depth += 1
        if n_no_cube_depth:
            self.get_logger().info(
                f'{n_no_cube_depth} blue blob(s) had fewer than {MASK_MIN_PX} valid depth '
                f'pixels on the blob itself — discarded (depth_mode mask)',
                throttle_duration_sec=3.0)
        if n_bad_depth:
            self.get_logger().info(
                f'{n_bad_depth} blue blob(s) had no depth sample inside '
                f'[{d_min:.2f}, {d_max:.2f}] m — discarded',
                throttle_duration_sec=3.0)
        return candidates

    def _publish_debug_image(self, bgr, header, best, rejected):
        tgt = None
        if best is not None:
            pt_base, u, v, _z, area = best
            tgt = (pt_base.point.x, pt_base.point.y, u, v, area)
        window = None
        if self.get_parameter('search_enabled').value:
            window = (self.get_parameter('search_x_min').value, self.get_parameter('search_x_max').value,
                      self.get_parameter('search_y_abs').value)
        vis = draw_overlay(bgr, tgt, [(r[1], r[2]) for r in rejected], self._cx, window)

        try:
            m = Image()
            m.header = header
            m.height, m.width = vis.shape[:2]
            m.encoding = 'bgr8'
            m.is_bigendian = 0
            m.step = m.width * 3
            m.data = vis.tobytes()
            self.debug_pub.publish(m)
        except Exception as e:
            self.get_logger().warn(f'debug image publish failed: {e!r}',
                                   throttle_duration_sec=5.0)

    def _publish_latched(self):
        age = float('inf') if self._last_real_detect_t is None \
            else time.time() - self._last_real_detect_t
        self.age_pub.publish(Float64(data=age))

        if self._last_pose_map is None:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                TARGET_FRAME, 'map',
                rclpy.time.Time(), timeout=Duration(seconds=0.05))
            pt_base = do_transform_point(self._last_pose_map, tf)

            if not self._in_search_window(pt_base.point):
                self.get_logger().info(
                    f'latched box now outside the search window '
                    f'({pt_base.point.x:.3f}, {pt_base.point.y:+.3f}, '
                    f'{pt_base.point.z:+.3f}) — not republishing',
                    throttle_duration_sec=5.0)
                return

            if self._depth_frame is not None:
                try:
                    tf_opt = self.tf_buffer.lookup_transform(
                        self._depth_frame, 'map',
                        rclpy.time.Time(), timeout=Duration(seconds=0.05))
                    pt_opt = do_transform_point(self._last_pose_map, tf_opt)
                    if not self._in_camera_view(pt_opt.point):
                        self.get_logger().info(
                            f'latched box no longer in the camera view '
                            f'(optical z={pt_opt.point.z:.3f} m) — not republishing',
                            throttle_duration_sec=5.0)
                        return
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException):
                    pass

            pose = PoseStamped()
            pose.header.frame_id = TARGET_FRAME
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position = pt_base.point
            pose.pose.orientation.w = 1.0

            self.pose_pub.publish(pose)
            self._publish_marker(pose)
        except Exception as e:
            self.get_logger().debug(f'TF map→{TARGET_FRAME} transform failed: {e}')

    def _publish_marker(self, pose: PoseStamped):
        m = Marker()
        m.header = pose.header
        m.ns = 'box_pose'
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose = pose.pose
        m.scale.x = m.scale.y = m.scale.z = 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = 0.1, 0.6, 1.0, 0.9
        self.marker_pub.publish(m)

def main(args=None):
    rclpy.init(args=args)
    node = BoxPoseEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
