#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import time
import uuid
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from cv_bridge import CvBridge


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")


class CropCapture(Node):
    def __init__(self, topic: str, out_dir: Path, model_name: str, max_crops: int) -> None:
        super().__init__("ds685_a3_capture_query_crops")
        self.bridge = CvBridge()
        self.topic = topic
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.max_crops = max_crops

        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.done = False
        self.create_subscription(Image, topic, self.on_image, 10)

    def on_image(self, msg: Image) -> None:
        if self.done:
            return
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warning(f"cv_bridge conversion failed: {e}")
            return

        res = self.model(cv_img, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            self.get_logger().info("No detections in this frame; waiting...")
            return

        import cv2

        names = res.names
        saved = 0
        for b in res.boxes:
            if saved >= self.max_crops:
                break
            cls_id = int(b.cls[0].item())
            cls_name = str(names.get(cls_id, cls_id))
            conf = float(b.conf[0].item())
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            crop = cv_img[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
            if crop.size == 0:
                continue
            out = self.out_dir / f"q_{uuid.uuid4()}_{cls_name.replace(' ', '_')}.jpg"
            cv2.imwrite(str(out), crop)
            print(f"{out}  class={cls_name} conf={conf:.3f}")
            saved += 1

        self.done = True


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture 1–N YOLO crops from one camera frame")
    parser.add_argument("--topic", default="/camera/image_raw")
    parser.add_argument("--out", default="/data/query")
    parser.add_argument("--model", default=os.getenv("YOLO_MODEL", "yolov8n.pt"))
    parser.add_argument("--max", type=int, default=3, dest="max_crops")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    # Prefer local weights under DATA_DIR if available (avoids flaky downloads).
    data_dir = Path(os.getenv("DATA_DIR", "/data"))
    try:
        p = Path(args.model)
        if (not p.is_absolute()) and ("/" not in args.model):
            candidate = data_dir / args.model
            if candidate.exists():
                args.model = str(candidate)
    except Exception:
        pass

    rclpy.init()
    node = CropCapture(args.topic, Path(args.out), args.model, args.max_crops)
    start = time.time()
    try:
        while rclpy.ok() and not node.done:
            if time.time() - start > args.timeout:
                raise SystemExit("Timed out waiting for detections")
            rclpy.spin_once(node, timeout_sec=0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
