#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class OneShotImageSaver(Node):
    def __init__(self, topic: str, out_path: Path, desired_encoding: str) -> None:
        super().__init__("ds685_capture_camera_once")
        self._bridge = CvBridge()
        self._topic = topic
        self._out_path = out_path
        self._desired_encoding = desired_encoding
        self._done = False
        self._error: str | None = None

        self.create_subscription(Image, self._topic, self._on_image, 10)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def error(self) -> str | None:
        return self._error

    def _on_image(self, msg: Image) -> None:
        if self._done:
            return

        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding=self._desired_encoding)
            self._out_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(self._out_path), img)
            if not ok:
                raise RuntimeError("cv2.imwrite returned false")
            self.get_logger().info(
                f"Saved {self._out_path} (topic={self._topic}, size={img.shape[1]}x{img.shape[0]}, enc={self._desired_encoding})"
            )
        except Exception as e:
            self._error = str(e)
            self.get_logger().error(f"Failed to save image: {e}")
        finally:
            self._done = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture one ROS2 Image message to a PNG.")
    parser.add_argument("--topic", default=os.getenv("IMAGE_TOPIC", "/camera/image_raw"))
    parser.add_argument("--out", required=True, help="Output path (png).")
    parser.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for first frame.")
    parser.add_argument("--encoding", default="bgr8", help="cv_bridge desired_encoding (default: bgr8).")
    args = parser.parse_args()

    out_path = Path(args.out).expanduser()

    rclpy.init()
    node = OneShotImageSaver(args.topic, out_path, args.encoding)

    start = time.time()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
            if time.time() - start > args.timeout:
                raise TimeoutError(f"Timed out waiting for Image on {args.topic} after {args.timeout:.1f}s")

        if node.error:
            raise RuntimeError(node.error)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
