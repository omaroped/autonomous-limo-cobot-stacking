#!/usr/bin/python3
"""Publish the LIMO base (wheel/steer) joint states so MoveIt sees a COMPLETE robot state.

The Ackermann base is driven by a Gazebo plugin (NOT ros2_control), so the
joint_state_broadcaster never publishes these 7 joints. Without them, move_group
logs "The complete state of the robot is not yet known. Missing ..." and REFUSES
to plan. We publish them at 0.0 — correct enough for stationary arm planning (the
arm's plan doesn't depend on wheel angles). Purely additive: it only ever touches
these 7 joint names, so it never conflicts with the broadcaster's arm joints
(move_group's CurrentStateMonitor merges joint states by name).

Cosmetic caveat: RViz wheels won't reflect real spin. Irrelevant for picking.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

BASE_JOINTS = [
    'front_left_steer_joint', 'front_left_wheel_joint',
    'front_right_steer_joint', 'front_right_wheel_joint',
    'rear_left_wheel_joint', 'rear_right_wheel_joint',
    # 'steering_wheel_joint' removed — this joint doesn't exist in the diff-drive
    # URDF; publishing it caused MoveIt to log "Joint not found" 200+ times per run
    # at 30 Hz (once per CurrentStateMonitor update tick).
]


class BaseJointStatePub(Node):
    def __init__(self):
        super().__init__('base_joint_state_pub')
        self.pub = self.create_publisher(JointState, '/joint_states', 10)
        self.create_timer(1.0 / 30.0, self._tick)
        self.get_logger().info(
            f'publishing {len(BASE_JOINTS)} base joint states so MoveIt has a complete state')

    def _tick(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = BASE_JOINTS
        msg.position = [0.0] * len(BASE_JOINTS)
        msg.velocity = [0.0] * len(BASE_JOINTS)
        msg.effort = [0.0] * len(BASE_JOINTS)
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = BaseJointStatePub()
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
