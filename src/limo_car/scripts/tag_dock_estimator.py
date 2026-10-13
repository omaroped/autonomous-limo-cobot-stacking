#!/usr/bin/python3
"""
tag_dock_estimator.py — detect the AprilTag 36h11 on the place table (table B) and
publish its 6-DOF pose for the place dock.

  - Subscribe /rgb/image_raw + /depth_camera/depth/camera_info
  - Detect the largest visible AprilTag with cv2.aruco (no extra ROS package needed)
  - solvePnP → 6-DOF tag pose in the camera optical frame
  - TF: optical frame → map, latched; republished in base_link at 20 Hz

Published topics:
  /place_tag_pose   (geometry_msgs/PoseStamped, base_link)
      Pose of the tag face. Its +Z axis is the face normal, toward the robot.
      The orchestrator uses it for the station heading and the sideways centring.
  /place_tag_marker (visualization_msgs/Marker) for RViz.

Tag layout (IDs assigned in gen_apriltag.py):
    ID 0 → North face (+Y)    ID 1 → South face (-Y)
    ID 2 → East face  (+X)    ID 3 → West face  (-X)

Parameters:
    tag_size   (float, default 0.08) : physical tag side length in metres

Note on coordinate convention:
    solvePnP returns the tag pose in camera coordinates.  The +Z axis of the tag
    frame points OUT of the tag (toward the robot), so it is the face normal we need.
    After TF to base_link, the +Z column of the rotation matrix gives the outward normal.
"""

import math
from collections import deque
import numpy as np
import cv2
import cv2.aruco as aruco

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from visualization_msgs.msg import Marker

import tf2_ros
from tf2_geometry_msgs import do_transform_point

TARGET_FRAME = 'base_link'
TAG_IDS_VALID = {0, 1, 2, 3}   # IDs we placed on the four table faces

# cv_bridge is not used: its prebuilt extension is compiled against NumPy 1.x and
# fails to import with NumPy 2. The Image message is decoded directly.


def _imgmsg_to_array(msg):
    """sensor_msgs/Image -> numpy array, for the encodings this node sees."""
    if msg.encoding in ('bgr8', 'rgb8'):
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
    raise ValueError(f'_imgmsg_to_array: unsupported encoding {msg.encoding!r}')


class TagDockEstimator(Node):

    def __init__(self):
        super().__init__('tag_dock_estimator')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('tag_size',    0.08)

        # ── State ─────────────────────────────────────────────────────────────
        self._fx = self._fy = self._cx = self._cy = None
        self._cam_frame     = None
        self._cam_matrix    = None
        self._dist_coeffs   = np.zeros((4, 1))  # Gazebo camera: no distortion

        # OpenCV 5 ArucoDetector object API.
        self._dictionary    = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36H11)
        self._det_params    = aruco.DetectorParameters()
        self._detector      = aruco.ArucoDetector(self._dictionary, self._det_params)

        # Latched map-frame tag pose, as in box_pose_estimator
        self._last_tag_map   = None   # PointStamped in map
        # Tag face orientation in map, averaged over recent detections (see _rgb_cb)
        self._last_tag_R_map = None
        self._normals        = deque(maxlen=10)
        self._last_tag_id    = None

        # TF
        self._tf_buf = tf2_ros.Buffer()
        self._tf_lis = tf2_ros.TransformListener(self._tf_buf, self)

        # ── Subscriptions ─────────────────────────────────────────────────────
        self.create_subscription(CameraInfo, '/depth_camera/depth/camera_info',
                                 self._caminfo_cb, 10)
        self.create_subscription(Image, '/rgb/image_raw', self._rgb_cb, 10)

        # ── Publishers ────────────────────────────────────────────────────────
        self._tag_pub   = self.create_publisher(PoseStamped,  '/place_tag_pose',   10)
        self._mark_pub  = self.create_publisher(Marker,       '/place_tag_marker', 10)

        # Latch republish at 20 Hz
        self.create_timer(0.05, self._publish_latched)

        self.get_logger().info('tag_dock_estimator started — waiting for camera…')

    # ── Camera info ───────────────────────────────────────────────────────────

    def _caminfo_cb(self, msg: CameraInfo):
        self._fx, self._fy = msg.k[0], msg.k[4]
        self._cx, self._cy = msg.k[2], msg.k[5]
        self._cam_frame = msg.header.frame_id
        self._cam_matrix = np.array([[self._fx, 0, self._cx],
                                     [0, self._fy, self._cy],
                                     [0,         0,       1]], dtype=np.float64)

    # ── Main detection callback ────────────────────────────────────────────────

    def _rgb_cb(self, msg: Image):
        if self._cam_matrix is None:
            self.get_logger().info('Waiting for camera_info…', throttle_duration_sec=3.0)
            return

        try:
            bgr = _imgmsg_to_array(msg)
            if msg.encoding == 'rgb8':
                # the sim camera plugin (sensor.xacro, format R8G8B8) publishes rgb8
                bgr = bgr[:, :, ::-1]
        except Exception as e:
            self.get_logger().warn(f'rgb convert: {e}', throttle_duration_sec=5.0)
            return

        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(grey)

        if ids is None or len(ids) == 0:
            self.get_logger().info('No AprilTag visible.', throttle_duration_sec=3.0)
            return

        # ── Pick the tag with the largest visible area (closest / most reliable) ──
        best_idx = -1
        best_area = -1.0
        for i, tid in enumerate(ids.flatten()):
            if int(tid) not in TAG_IDS_VALID:
                continue
            c = corners[i][0]
            area = cv2.contourArea(c)
            if area > best_area:
                best_area, best_idx = area, i
        if best_idx < 0:
            return

        # flatten(): ids may be shaped (N,) or (N,1) depending on the OpenCV version.
        tag_id   = int(ids.flatten()[best_idx])
        tag_size = float(self.get_parameter('tag_size').value)

        # ── solvePnP: tag frame with +Z pointing out of the tag face ─────────
        half = tag_size / 2.0
        obj_pts = np.array([
            [-half,  half, 0],
            [ half,  half, 0],
            [ half, -half, 0],
            [-half, -half, 0],
        ], dtype=np.float64)
        img_pts = corners[best_idx][0].astype(np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, self._cam_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return

        # Tag +Z in camera frame = outward face normal toward robot
        R_tag, _ = cv2.Rodrigues(rvec)

        # Tag origin as PointStamped in camera frame
        pt_cam = PointStamped()
        pt_cam.header.frame_id = self._cam_frame
        pt_cam.header.stamp    = rclpy.time.Time().to_msg()
        # .item(): tvec is shaped (3,1), so tvec[0] is a 1-element array.
        pt_cam.point.x = float(tvec[0].item())
        pt_cam.point.y = float(tvec[1].item())
        pt_cam.point.z = float(tvec[2].item())

        # ── TF: camera → map (latch) ──────────────────────────────────────────
        try:
            tf_to_map = self._tf_buf.lookup_transform(
                'map', self._cam_frame,
                rclpy.time.Time(), timeout=Duration(seconds=0.2))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF cam→map: {e}', throttle_duration_sec=3.0)
            return

        pt_map = do_transform_point(pt_cam, tf_to_map)
        self._last_tag_map = pt_map

        # Transform the tag's rotation into map frame
        R_map = self._rot_from_tf(tf_to_map) @ R_tag

        # Average the face normal over recent detections. In the map frame the tag does
        # not move, so frame-to-frame variation is PnP noise only (up to ~18 deg seen).
        # Build the rotation from the averaged horizontal normal, tag upright (true for
        # every plate in this world): X = Y x Z, Y up, Z out.
        if tag_id != self._last_tag_id:
            self._normals.clear()
            self._last_tag_id = tag_id
        n = R_map[:, 2].copy()
        n[2] = 0.0
        self._normals.append(n / max(np.linalg.norm(n), 1e-9))
        z_avg = np.mean(self._normals, axis=0)
        z_avg /= max(np.linalg.norm(z_avg), 1e-9)
        up = np.array([0.0, 0.0, 1.0])
        self._last_tag_R_map = np.column_stack((np.cross(up, z_avg), up, z_avg))

        # Immediate publish + log
        self._publish_to_base_link(pt_map, tag_id)
        self.get_logger().info(
            f'Tag {tag_id} detected: '
            f'tag@map ({pt_map.point.x:.3f},{pt_map.point.y:.3f})',
            throttle_duration_sec=1.0)

    # ── Publish to base_link ──────────────────────────────────────────────────

    def _publish_to_base_link(self, pt_map, tag_id):
        try:
            tf_to_base = self._tf_buf.lookup_transform(
                TARGET_FRAME, 'map',
                rclpy.time.Time(), timeout=Duration(seconds=0.1))
        except Exception:
            return

        pt_base   = do_transform_point(pt_map,  tf_to_base)

        # Tag pose (full 6-DOF) in base_link; the orientation is rotated like the position.
        qx, qy, qz, qw = self._rot_to_quat(
            self._rot_from_tf(tf_to_base) @ self._last_tag_R_map)
        tag_ps = PoseStamped()
        tag_ps.header.frame_id = TARGET_FRAME
        tag_ps.header.stamp    = self.get_clock().now().to_msg()
        tag_ps.pose.position   = pt_base.point
        tag_ps.pose.orientation.x = qx
        tag_ps.pose.orientation.y = qy
        tag_ps.pose.orientation.z = qz
        tag_ps.pose.orientation.w = qw
        self._tag_pub.publish(tag_ps)
        self._publish_marker(tag_ps, tag_id)

    def _publish_latched(self):
        """Republish the last known detection at 20 Hz, dynamically in base_link."""
        if self._last_tag_map is None or self._last_tag_R_map is None:
            return
        try:
            tf = self._tf_buf.lookup_transform(
                TARGET_FRAME, 'map',
                rclpy.time.Time(), timeout=Duration(seconds=0.05))
        except Exception:
            return

        pt_base   = do_transform_point(self._last_tag_map,  tf)

        qx, qy, qz, qw = self._rot_to_quat(
            self._rot_from_tf(tf) @ self._last_tag_R_map)   # in base_link, as the header says
        tag_ps = PoseStamped()
        tag_ps.header.frame_id = TARGET_FRAME
        tag_ps.header.stamp    = self.get_clock().now().to_msg()
        tag_ps.pose.position   = pt_base.point
        tag_ps.pose.orientation.x = qx
        tag_ps.pose.orientation.y = qy
        tag_ps.pose.orientation.z = qz
        tag_ps.pose.orientation.w = qw
        self._tag_pub.publish(tag_ps)

    # ── Marker for RViz ──────────────────────────────────────────────────────

    def _publish_marker(self, tag_ps: PoseStamped, tag_id: int):
        # Tag face: cyan sphere
        m = Marker()
        m.header = tag_ps.header
        m.ns, m.id   = 'place_tag', tag_id
        m.type, m.action = Marker.SPHERE, Marker.ADD
        m.pose = tag_ps.pose
        m.scale.x = m.scale.y = m.scale.z = 0.04
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 1.0, 0.9
        self._mark_pub.publish(m)

    # ── Math helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _rot_from_tf(tf_stamped):
        """Extract 3×3 rotation matrix from a TransformStamped."""
        q = tf_stamped.transform.rotation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ])
        return R

    @staticmethod
    def _rot_to_quat(R):
        """3×3 rotation matrix → (qx, qy, qz, qw), Shepperd method."""
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            s = 0.5 / math.sqrt(tr + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return float(x), float(y), float(z), float(w)


def main(args=None):
    rclpy.init(args=args)
    node = TagDockEstimator()
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
