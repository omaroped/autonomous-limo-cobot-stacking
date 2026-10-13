#!/usr/bin/env python3
import sys

import rclpy
from rclpy.utilities import remove_ros_args
from lifecycle_msgs.srv import GetState

WAIT_TIMEOUT_SEC = 60.0
POLL_SEC         = 1.0

def main():
    names = remove_ros_args(sys.argv)[1:]
    if not names:
        print('wait_for_lifecycle_node: no node names given', file=sys.stderr)
        return 1

    rclpy.init()
    node = rclpy.create_node('wait_for_lifecycle_node')
    clients = {n: node.create_client(GetState, f'/{n}/get_state') for n in names}

    remaining = set(names)
    waited = 0.0
    while remaining:
        for n in list(remaining):
            if clients[n].wait_for_service(timeout_sec=POLL_SEC):
                node.get_logger().info(f'{n}: get_state service is up')
                remaining.discard(n)
        if remaining:
            waited += POLL_SEC
            if waited >= WAIT_TIMEOUT_SEC:
                node.get_logger().error(
                    f'timed out after {WAIT_TIMEOUT_SEC:.0f} s waiting for: '
                    f'{sorted(remaining)} — check that these nodes actually started '
                    f'(colcon build clean? crashed on launch?)')
                node.destroy_node()
                rclpy.shutdown()
                return 1
            if int(waited) % 5 == 0:
                node.get_logger().info(
                    f'still waiting on {sorted(remaining)} ({waited:.0f}s)…')

    node.destroy_node()
    rclpy.shutdown()
    return 0

if __name__ == '__main__':
    sys.exit(main())
