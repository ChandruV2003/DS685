#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node


def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    # Z-axis rotation quaternion
    half = yaw * 0.5
    return (0.0, 0.0, math.sin(half), math.cos(half))


class InitialPosePublisher(Node):
    def __init__(self) -> None:
        super().__init__("ds685_publish_initialpose")
        self.pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)

    def publish(self, x: float, y: float, yaw: float, frame: str) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        qx, qy, qz, qw = _yaw_to_quat(float(yaw))
        msg.pose.pose.orientation.x = qx
        msg.pose.pose.orientation.y = qy
        msg.pose.pose.orientation.z = qz
        msg.pose.pose.orientation.w = qw

        # Covariance: keep it simple but non-zero.
        msg.pose.covariance[0] = 0.25  # x
        msg.pose.covariance[7] = 0.25  # y
        msg.pose.covariance[35] = 0.0685  # yaw

        # Publish a few times to improve chances AMCL receives it.
        for _ in range(3):
            self.pub.publish(msg)
            time.sleep(0.2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish /initialpose for Nav2/AMCL")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--yaw", type=float, required=True)
    parser.add_argument("--frame", default="map")
    args = parser.parse_args()

    rclpy.init()
    node = InitialPosePublisher()
    try:
        node.publish(args.x, args.y, args.yaw, args.frame)
        # Allow publish to flush.
        rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

