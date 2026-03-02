#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node


def _yaw_to_quat(yaw: float) -> Quaternion:
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    yaw: float
    label: str


DEFAULT_WAYPOINTS = [
    Waypoint(0.08, 4.10, 0.00, "bench_02_area"),
    Waypoint(5.60, 4.00, 1.57, "bench_03_area"),
    Waypoint(3.60, -1.00, 0.00, "bench_04_area"),
    Waypoint(-1.40, -1.00, 1.57, "bench_01_area"),
]


class Navigator(Node):
    def __init__(self) -> None:
        super().__init__("ds685_a3_navigate_waypoints")
        self.client = ActionClient(self, NavigateToPose, "navigate_to_pose")

    def wait_server(self, timeout_sec: float) -> bool:
        start = time.time()
        while rclpy.ok() and (time.time() - start) < timeout_sec:
            if self.client.wait_for_server(timeout_sec=1.0):
                return True
        return False

    def go(self, wp: Waypoint, timeout_sec: float) -> bool:
        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = "map"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(wp.x)
        ps.pose.position.y = float(wp.y)
        ps.pose.position.z = 0.0
        ps.pose.orientation = _yaw_to_quat(float(wp.yaw))
        goal.pose = ps

        self.get_logger().info(f"Sending goal {wp.label}: x={wp.x:.2f} y={wp.y:.2f} yaw={wp.yaw:.2f}")
        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or (not goal_handle.accepted):
            self.get_logger().warning(f"Goal rejected: {wp.label}")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout_sec)
        if not result_future.done():
            self.get_logger().warning(f"Timed out waiting for result; canceling: {wp.label}")
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=5.0)
            return False

        result = result_future.result()
        status = int(getattr(result, "status", -1))
        # nav2 action status: 4=SUCCEEDED, 6=ABORTED, 5=CANCELED
        if status == 4:
            self.get_logger().info(f"Reached: {wp.label}")
            return True
        self.get_logger().warning(f"Failed to reach {wp.label}: status={status}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Navigate a set of map-frame waypoints (Run A exploration helper)")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout per waypoint (sec)")
    parser.add_argument("--dwell", type=float, default=5.0, help="Wait at each waypoint (sec)")
    parser.add_argument("--loops", type=int, default=1, help="Repeat the waypoint list N times")
    args = parser.parse_args()

    waypoints = DEFAULT_WAYPOINTS

    rclpy.init()
    node = Navigator()
    try:
        if not node.wait_server(timeout_sec=60.0):
            raise SystemExit("Nav2 action server 'navigate_to_pose' not available")

        loops = max(1, int(args.loops))
        for i in range(loops):
            node.get_logger().info(f"Starting loop {i + 1}/{loops} (waypoints={len(waypoints)})")
            for wp in waypoints:
                node.go(wp, timeout_sec=float(args.timeout))
                t_end = time.time() + float(args.dwell)
                while rclpy.ok() and time.time() < t_end:
                    rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
