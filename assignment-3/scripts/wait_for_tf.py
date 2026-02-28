#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros


class TfWaiter(Node):
    def __init__(self) -> None:
        super().__init__("ds685_wait_for_tf")
        self.buf = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.buf, self)

    def wait_for(self, target: str, source: str, timeout_sec: float) -> None:
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() < deadline:
            try:
                self.buf.lookup_transform(
                    target_frame=target,
                    source_frame=source,
                    time=Time(),
                    timeout=Duration(seconds=0.2),
                )
                return
            except Exception:
                rclpy.spin_once(self, timeout_sec=0.1)
        raise TimeoutError(f"Timed out waiting for TF {target} <- {source}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for a TF transform to become available")
    parser.add_argument("--target", required=True, help="Target frame, e.g. map")
    parser.add_argument("--source", required=True, help="Source frame, e.g. base_footprint")
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    rclpy.init()
    node = TfWaiter()
    try:
        node.wait_for(args.target, args.source, args.timeout)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

