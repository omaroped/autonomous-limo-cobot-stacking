#!/usr/bin/python3
"""
box_pose_estimator — turn the blue box into a 3D pose for the arm.

Finds blue blobs by colour, deprojects each blob's pixel + depth into a 3D point
with the camera intrinsics, transforms it into the arm's planning frame
(base_link) via TF2, and publishes the chosen box as a geometry_msgs/PoseStamped
on /box_pose. nav_pick_orchestrator.py subscribes to it.

Pipeline:
  /rgb/image_raw (bgr) ── HSV mask ── closest blue contour centroid (u, v)
  /depth_camera/depth/image_raw (32FC1) ── depth Z at (u, v)
  /depth_camera/depth/camera_info ── fx, fy, cx, cy (intrinsics)
        X = (u-cx)·Z/fx,  Y = (v-cy)·Z/fy,  Z = Z      (optical frame, REP 103)
  TF2: camera optical frame → base_link
        publish /box_pose (PoseStamped, orientation = identity; the grasp
        orientation is chosen by the orchestrator, not here)
"""

import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Float64

import tf2_ros
from tf2_geometry_msgs import do_transform_point

# cv_bridge is not used: its prebuilt extension is compiled against NumPy 1.x and
# fails to import with NumPy 2. sensor_msgs/Image carries everything needed
# (height, width, encoding, data), so images are converted directly.


def _imgmsg_to_array(msg):
    """sensor_msgs/Image -> numpy array, for the three encodings this node sees."""
    if msg.encoding == '16UC1':
        return np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    if msg.encoding == '32FC1':
        return np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
    if msg.encoding in ('bgr8', 'rgb8'):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    raise ValueError(f'_imgmsg_to_array: unsupported encoding {msg.encoding!r}')


# HSV range for target box (widened to detect light blue physical boxes & ambient room lighting)
BLUE_HSV_LO = np.array([85,  30,  40])
BLUE_HSV_HI = np.array([135, 255, 255])

MIN_CONTOUR_AREA = 30          # px², ignore noise
TARGET_FRAME     = 'base_link'  # MoveIt arm-planning frame

# ── Depth validity band ──────────────────────────────────────────────────────
# The camera cannot measure outside its own clip planes (gazebo/sensor.xacro:
# <near>0.15</near>, <far>3.0</far>), but the Gazebo plugin is configured
# <min_depth>0.001</min_depth> / <max_depth>300.0</max_depth>, so it publishes
# values well outside that range anyway. Depth outside the band is discarded:
# a too-small depth scales the pixel offset down and throws the point off-axis.
DEPTH_MIN_DEFAULT = 0.15   # = sensor.xacro <near>. nav_pick.launch.py sets 0.30,
DEPTH_MAX_DEFAULT = 3.0    #   the real Orbbec DaBai's blind zone.

# ── Search window (base_link, metres) ────────────────────────────────────────
# The plausible volume for a box the robot is about to pick. Everything outside
# is rejected before it can ever become a grasp target.
#
# Height is derived from config/scene.yaml, not guessed:
#     pickup_table.top_z (0.14) + box.size/2 (0.02) - robot.base_link_ground_z (0.145)
#   = +0.015 m for the box centre in base_link.
# The band below is much wider than that on purpose: it is a sanity gate against
# floor/ceiling artifacts, not a tight fit.
#
# X spans the whole approach: the box is first sighted around 0.9 m and the dock
# ends at about 0.24 m, so the window stays valid through every phase.
SEARCH_X_MIN_DEFAULT = 0.15    # = near clip; nothing closer is measurable
SEARCH_X_MAX_DEFAULT = 1.60    # beyond the ~0.92 m first sighting, with margin
SEARCH_Y_ABS_DEFAULT = 0.60    # ~the FOV half-width at max range (0.92*tan34deg)
SEARCH_Z_MIN_DEFAULT = -0.10   # 0.115 m below the expected box centre
SEARCH_Z_MAX_DEFAULT = 0.20    # 0.185 m above it
# nav_pick.launch.py also passes these values explicitly.

# ── Size plausibility ────────────────────────────────────────────────────────
BOX_EDGE_M          = 0.0247   # sim cube edge (worlds/final_map.world)
MIN_AREA_FRACTION   = 0.15    # robust for partial occlusion / oblique views; lower
                              # values accept small noise specks as boxes

# Upper bound on the same geometry. A blob far larger than the cube can subtend at its
# own measured depth is not the cube, whatever colour it is (in the lab, a blue chair
# was ~9x the expected area).
#
# 4.0 on area = 2x on linear width. Generous on purpose: the cube is normally seen as
# two faces at once (top plus a side), and blur inflates a small blob's mask.
MAX_AREA_FACTOR     = 4.0


class BoxPoseEstimator(Node):

    def __init__(self):
        super().__init__('box_pose_estimator')

        self._depth = None          # latest depth image (np.float32)
        self._fx = self._fy = None  # intrinsics
        self._cx = self._cy = None
        self._img_w = self._img_h = None
        self._depth_frame = None    # optical frame id from the depth image header

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Target-selection policy among multiple visible boxes:
        #   'nearest'   — closest box by camera depth (default)
        #   'rightmost' — deterministic rightmost-first ordering in base_link
        #                 (nav_pick.launch.py sets this)
        self.declare_parameter('target_policy', 'nearest')

        # ── Search window ────────────────────────────────────────────────────
        # All live-tunable, so the window can be widened/narrowed against a running
        # sim without a rebuild:
        #   ros2 param set /box_pose_estimator search_y_abs 0.30
        self.declare_parameter('search_enabled', True)
        self.declare_parameter('search_x_min', SEARCH_X_MIN_DEFAULT)
        self.declare_parameter('search_x_max', SEARCH_X_MAX_DEFAULT)
        self.declare_parameter('search_y_abs', SEARCH_Y_ABS_DEFAULT)
        self.declare_parameter('search_z_min', SEARCH_Z_MIN_DEFAULT)
        self.declare_parameter('search_z_max', SEARCH_Z_MAX_DEFAULT)
        self.declare_parameter('depth_min', DEPTH_MIN_DEFAULT)
        self.declare_parameter('depth_max', DEPTH_MAX_DEFAULT)
        # Live-tunable: raise to reject more aggressively, set 0.0 to disable.
        #   ros2 param set /box_pose_estimator min_area_fraction 0.25
        self.declare_parameter('min_area_fraction', MIN_AREA_FRACTION)
        # Upper size bound -- see MAX_AREA_FACTOR. Set <= 0 to disable:
        #   ros2 param set /box_pose_estimator max_area_factor 0.0
        self.declare_parameter('max_area_factor', MAX_AREA_FACTOR)
        # Cube edge for the size gate: 24.7 mm in the sim, 25 mm in the lab.
        self.declare_parameter('box_edge_m', BOX_EDGE_M)

        # Subscribers
        self.create_subscription(CameraInfo, '/depth_camera/depth/camera_info',
                                 self._caminfo_cb, 10)
        self.create_subscription(Image, '/depth_camera/depth/image_raw',
                                 self._depth_cb, 10)
        self.create_subscription(Image, '/rgb/image_raw',
                                 self._rgb_cb, 10)

        # Publishers
        self.pose_pub   = self.create_publisher(PoseStamped, '/box_pose', 10)
        self.marker_pub = self.create_publisher(Marker, '/box_pose_marker', 10)
        self.debug_pub  = self.create_publisher(Image, '/box_detection_debug', 10)
        # Freshness signal: /box_pose alone can't tell a consumer whether the pose
        # was just seen or is a replayed latch from seconds ago (both use the
        # current timestamp) — /box_detection_age exposes that gap explicitly.
        self.age_pub    = self.create_publisher(Float64, '/box_detection_age', 10)

        # Latch the last good detection in the global 'map' frame and republish it at 20 Hz,
        # transformed dynamically to base_link so the coordinate remains accurate as the robot moves.
        self._last_pose_map = None
        self._last_real_detect_t = None   # wall-clock time of the last ACTUAL detection
        self.create_timer(0.05, self._publish_latched)

        self.get_logger().info('box_pose_estimator started — waiting for camera…')

    # ── Sensor callbacks ──────────────────────────────────────────────────────

    def _caminfo_cb(self, msg: CameraInfo):
        # K = [fx 0 cx; 0 fy cy; 0 0 1]
        self._fx, self._fy = msg.k[0], msg.k[4]
        self._cx, self._cy = msg.k[2], msg.k[5]
        self._img_w, self._img_h = msg.width, msg.height

    def _depth_cb(self, msg: Image):
        try:
            if msg.encoding == '16UC1':
                depth_mm = _imgmsg_to_array(msg)
                d = depth_mm.astype(np.float32) / 1000.0
            else:
                d = _imgmsg_to_array(msg).astype(np.float32)
                # If values are in mm (median > 10), convert to meters
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
            bgr = _imgmsg_to_array(msg)
            if msg.encoding == 'rgb8':
                # The sim camera plugin (sensor.xacro, format R8G8B8) publishes
                # rgb8, not bgr8. Without the swap, blue and red stay swapped and
                # the blue detector (b>g+15, b>r+25) finds nothing.
                bgr = bgr[:, :, ::-1]
        except Exception as e:
            self.get_logger().warn(f'rgb convert failed: {e}', throttle_duration_sec=5.0)
            return

        h, w = bgr.shape[:2]
        depth = self._depth
        # Handle rgb/depth resolution mismatch (e.g. Orbbec DaBai 640x480 RGB vs 640x400 depth).
        if depth.shape[0] != h or depth.shape[1] != w:
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

        candidates = self._detect_box_pixels(bgr, depth, h, w)
        if not candidates:
            self.get_logger().info('No blue box visible.', throttle_duration_sec=3.0)
            self._publish_debug_image(bgr, msg.header, None, [])
            return

        # ── Search window ────────────────────────────────────────────────────
        # Deproject every candidate into base_link first, discard the ones outside
        # the plausible pick volume, and only then apply target_policy to whatever
        # survives, so an implausible blob can never win the selection.
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

        # ── Selection among the survivors ────────────────────────────────────
        # Multiple boxes may legitimately be visible at once (the stacking pickup
        # table). target_policy decides between them — see the param declaration.
        policy = self.get_parameter('target_policy').value
        if policy == 'rightmost':
            # Deterministic rightmost-first ordering in the robot's own frame (most
            # negative base_link y). As boxes are consumed one at a time, whichever
            # remain naturally present a new "rightmost" each cycle, so no cycle-count
            # tracking is needed here.
            best = min(kept, key=lambda s: s[0].point.y)
        else:
            best = min(kept, key=lambda s: s[3])     # 'nearest' — smallest depth
        if len(kept) > 1:
            self.get_logger().info(
                f'{len(kept)} box(es) in window — targeting {policy} '
                f'(x={best[0].point.x:.3f} y={best[0].point.y:+.3f} '
                f'z={best[0].point.z:+.3f}, area={best[4]:.0f}px²)',
                throttle_duration_sec=1.0)

        pt_base, u, v, z, _ = best
        pt = self._deproject(u, v, z)   # optical-frame point, for the map latch below

        # Latch the point in 'map' so _publish_latched() can republish it as the
        # robot moves. Only reached with an in-window detection.
        # When 'map' does not exist (e.g. a standalone test without Nav2) there is
        # simply nothing to latch, and only the live detections below are published.
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

        # Publish the direct depth→base_link point computed for the window test
        # above, not a round trip through map, which would add the localisation
        # error to a measurement the arm uses.
        self._last_real_detect_t = time.time()   # a genuine detection happened just now

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

    # ── Search-window helpers ─────────────────────────────────────────────────

    def _deproject(self, u, v, z):
        """Pixel + depth → 3D point in the camera optical frame (REP 103)."""
        pt = PointStamped()
        pt.header.frame_id = self._depth_frame
        pt.header.stamp = rclpy.time.Time().to_msg()   # latest available transform
        pt.point.x = float((u - self._cx) * z / self._fx)
        pt.point.y = float((v - self._cy) * z / self._fy)
        pt.point.z = float(z)
        return pt

    def _in_camera_view(self, p):
        """Is this optical-frame point on the camera's line of sight at all?

        In front of the lens, and projecting onto the sensor. Not a depth-clip
        check: a latched point is a remembered world coordinate, not a reading. At
        the dock the box is inside the near clip, so a clip check here would stop
        the latch for the whole grasp phase.

        Used on the latched replay path, where a fixed world point is re-expressed
        in base_link at 20 Hz as the robot moves and can drift out of view.
        """
        if self._fx is None or self._img_w is None:
            return True                      # no intrinsics yet — cannot judge
        if p.z <= 0.0:
            return False                     # behind the lens
        u = p.x * self._fx / p.z + self._cx
        v = p.y * self._fy / p.z + self._cy
        return 0 <= u < self._img_w and 0 <= v < self._img_h

    def _expected_area(self, depth):
        """Pixel area a BOX_EDGE-wide object subtends at this depth."""
        if self._fx is None or depth <= 0:
            return 0.0
        edge = float(self.get_parameter('box_edge_m').value)
        w = edge * self._fx / depth
        return w * w

    def _size_verdict(self, area, depth):
        """'ok' | 'small' | 'large' -- which side of the geometry a blob falls on.

        Both bounds are geometry against the blob's own measured depth, so they need
        no per-range constant: they scale themselves.
        """
        expected = self._expected_area(depth)
        if expected <= 0.0:
            return 'ok'                      # no intrinsics yet -- cannot judge
        frac = float(self.get_parameter('min_area_fraction').value)
        if frac > 0.0 and area < frac * expected:
            return 'small'
        hi = float(self.get_parameter('max_area_factor').value)
        if hi > 0.0 and area > hi * expected:
            return 'large'
        return 'ok'

    def _in_search_window(self, p):
        """Is this base_link point inside the plausible pick volume?

        An axis-aligned box rather than a projected image ROI on purpose: it is
        stated in the frame the arm actually plans in, so it stays meaningful if the
        camera is re-mounted or its intrinsics change, and it is directly comparable
        against the numbers in config/scene.yaml and reach_map.csv.
        """
        if not self.get_parameter('search_enabled').value:
            return True
        return (self.get_parameter('search_x_min').value <= p.x
                <= self.get_parameter('search_x_max').value
                and abs(p.y) <= self.get_parameter('search_y_abs').value
                and self.get_parameter('search_z_min').value <= p.z
                <= self.get_parameter('search_z_max').value)

    # ── Detection helper ──────────────────────────────────────────────────────

    def _detect_box_pixels(self, bgr, depth, h, w):
        """Return a list of (u, v, median_depth, area) for every valid blue blob — not
        just the nearest. Selection among candidates (e.g. rightmost-first ordering) is
        done by the caller in base_link, not here in pixel/depth space.

        Depth is the median over the blob's own pixels (eroded), with samples outside
        [depth_min, depth_max] discarded. Using only the blob's pixels keeps the table
        behind the box out of the reading."""
        d_min = float(self.get_parameter('depth_min').value)
        d_max = float(self.get_parameter('depth_max').value)

        # Robust dual segmentation: Combined HSV + RGB Color Dominance
        b = bgr[:, :, 0].astype(int)
        g = bgr[:, :, 1].astype(int)
        r = bgr[:, :, 2].astype(int)
        rgb_mask = ((b > g + 15) & (b > r + 25) & (b > 50)).astype(np.uint8) * 255

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hsv_mask = cv2.inRange(hsv, BLUE_HSV_LO, BLUE_HSV_HI)
        mask = cv2.bitwise_or(hsv_mask, rgb_mask)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        c_res = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = c_res[0] if len(c_res) == 2 else c_res[1]

        candidates = []
        n_bad_depth = 0
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

            x0, y0, bw, bh = cv2.boundingRect(c)
            sub = np.zeros((bh, bw), np.uint8)
            cv2.drawContours(sub, [c - np.array([x0, y0])], -1, 255, -1)
            sub = cv2.erode(sub, erode_k)
            d = depth[y0:y0 + bh, x0:x0 + bw][sub > 0]
            d = d[np.isfinite(d) & (d >= d_min) & (d <= d_max)]
            if d.size >= 5:
                candidates.append((u, v, float(np.median(d)), float(area)))
            else:
                n_bad_depth += 1
        if n_bad_depth:
            self.get_logger().info(
                f'{n_bad_depth} blue blob(s) had no depth sample inside '
                f'[{d_min:.2f}, {d_max:.2f}] m — discarded',
                throttle_duration_sec=3.0)
        return candidates

    def _publish_debug_image(self, bgr, header, best, rejected):
        """Overlay the selection on /box_detection_debug."""
        vis = bgr.copy()

        for (pt_base, u, v, _z, area) in rejected:
            p = pt_base.point
            cv2.drawMarker(vis, (u, v), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
            cv2.putText(vis, f'X ({p.x:.2f},{p.y:+.2f},{p.z:+.2f}) {area:.0f}px',
                        (u + 14, v + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 0, 255), 1, cv2.LINE_AA)

        if best is not None:
            pt_base, u, v, _z, area = best
            p = pt_base.point
            # Draw prominent green bounding box and crosshair
            bx_size = int(max(15, min(60, 40 / max(0.2, _z))))
            cv2.rectangle(vis, (u - bx_size, v - bx_size), (u + bx_size, v + bx_size), (0, 255, 0), 3)
            cv2.drawMarker(vis, (u, v), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(vis, f'TARGET: x={p.x:.2f} y={p.y:.2f} z={p.z:.2f}m',
                        (max(10, u - 60), max(25, v - bx_size - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(vis, "SEARCHING FOR BLUE CUBE...", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

        if self.get_parameter('search_enabled').value:
            banner = (f"win x[{self.get_parameter('search_x_min').value:.2f},"
                      f"{self.get_parameter('search_x_max').value:.2f}] "
                      f"|y|<{self.get_parameter('search_y_abs').value:.2f} "
                      f"z[{self.get_parameter('search_z_min').value:+.2f},"
                      f"{self.get_parameter('search_z_max').value:+.2f}]")
        else:
            banner = 'search window DISABLED'
        cv2.putText(vis, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 0), 1, cv2.LINE_AA)

        # Built by hand rather than via cv_bridge (see the note at the top).
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
        """Republish the last known box pose at 20 Hz, dynamically transformed from
        map to base_link so the coordinate remains accurate as the robot moves.

        Also publishes /box_detection_age — seconds since the last REAL detection —
        because this replay always uses the current timestamp, so a subscriber to
        /box_pose alone cannot tell a live sighting from an old one being repeated."""
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

            # The latch is a fixed world point, but base_link is not: as the robot
            # drives and turns, an old latch can transform to somewhere the box
            # cannot be. Both gates apply here as they do to live detections.
            if not self._in_search_window(pt_base.point):
                self.get_logger().info(
                    f'latched box now outside the search window '
                    f'({pt_base.point.x:.3f}, {pt_base.point.y:+.3f}, '
                    f'{pt_base.point.z:+.3f}) — not republishing',
                    throttle_duration_sec=5.0)
                return

            # Is the latched point still somewhere the camera can see? Checked in the
            # optical frame, because the camera sits 0.1 m ahead of and 0.065 m above
            # base_link, so "in front of the robot" and "in view" are not the same
            # test. Skipped silently if the depth frame is not up yet.
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
