#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

import tf2_ros

import zenoh


def _q_to_yaw(x: float, y: float, z: float, w: float) -> float:
    # Yaw from quaternion (Z axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(__import__("math").atan2(siny_cosp, cosy_cosp))


def _quat_to_rot(x: float, y: float, z: float, w: float) -> list[list[float]]:
    # 3x3 rotation matrix from quaternion
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def _t_to_mat4(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> list[float]:
    rot = _quat_to_rot(qx, qy, qz, qw)
    # Row-major 4x4 homogeneous transform
    return [
        rot[0][0], rot[0][1], rot[0][2], tx,
        rot[1][0], rot[1][1], rot[1][2], ty,
        rot[2][0], rot[2][1], rot[2][2], tz,
        0.0, 0.0, 0.0, 1.0,
    ]


@dataclass
class OdomSample:
    stamp_sec: int
    stamp_nanosec: int
    msg: Odometry

    @property
    def t(self) -> float:
        return float(self.stamp_sec) + float(self.stamp_nanosec) * 1e-9


class DetectorPublisher(Node):
    def __init__(self) -> None:
        super().__init__("ds685_a2_detector")

        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")

        self.robot_id = os.getenv("ROBOT_ID", "tb3_sim")
        run_id_env = (os.getenv("RUN_ID", "") or "").strip()
        self.run_id = run_id_env or str(uuid.uuid4())
        self.image_topic = os.getenv("IMAGE_TOPIC", "/camera/image_raw")
        self.odom_topic = os.getenv("ODOM_TOPIC", "/odom")
        self.base_frame = os.getenv("BASE_FRAME", "base_footprint")
        self.camera_frame_override = os.getenv("CAMERA_FRAME", "")

        self.model_name = os.getenv("YOLO_MODEL", "yolov8n.pt")
        self.max_fps = float(os.getenv("MAX_FPS", "1.0"))

        self.zenoh_connect = os.getenv("ZENOH_CONNECT", "tcp/zenoh:7447")
        self.key_prefix = os.getenv("ZENOH_PREFIX", "maze")

        self._sequence = 0
        self._last_pub_wall = 0.0
        self._runmeta_published = False

        self._bridge = CvBridge()
        self._odom_buf: deque[OdomSample] = deque(maxlen=500)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        logging.info("Loading YOLO model: %s", self.model_name)
        from ultralytics import YOLO  # heavy import

        self._yolo = YOLO(self.model_name)

        logging.info("Opening Zenoh session (connect=%s)", self.zenoh_connect)
        cfg = zenoh.Config.from_json5(json.dumps({"connect": {"endpoints": [self.zenoh_connect]}}))
        self._zenoh = zenoh.open(cfg)

        self.create_subscription(Odometry, self.odom_topic, self._on_odom, 50)
        self.create_subscription(Image, self.image_topic, self._on_image, 10)

        logging.info("Ready: image=%s odom=%s run_id=%s", self.image_topic, self.odom_topic, self.run_id)

    def _on_odom(self, msg: Odometry) -> None:
        stamp = msg.header.stamp
        self._odom_buf.append(OdomSample(stamp.sec, stamp.nanosec, msg))

    def _closest_odom(self, image_stamp: Time) -> Odometry | None:
        if not self._odom_buf:
            return None
        t_img = image_stamp.nanoseconds * 1e-9
        best: OdomSample | None = None
        best_dt = 1e9
        for s in self._odom_buf:
            dt = abs(s.t - t_img)
            if dt < best_dt:
                best_dt = dt
                best = s
        return best.msg if best else None

    def _lookup_base_to_camera(self, camera_frame: str, stamp: Time) -> tuple[bool, list[float] | None]:
        try:
            tf = self._tf_buffer.lookup_transform(
                target_frame=camera_frame,
                source_frame=self.base_frame,
                time=stamp,
                timeout=Duration(seconds=0.2),
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            return True, _t_to_mat4(t.x, t.y, t.z, q.x, q.y, q.z, q.w)
        except Exception:
            return False, None

    def _publish_runmeta_once(self) -> None:
        if self._runmeta_published:
            return
        key = f"{self.key_prefix}/{self.robot_id}/{self.run_id}/runmeta/v1"
        payload = {
            "schema": "maze.runmeta.v1",
            "run_id": self.run_id,
            "robot_id": self.robot_id,
            "image_topic": self.image_topic,
            "odom_topic": self.odom_topic,
            "yolo_model": self.model_name,
            "started_at_unix": time.time(),
        }
        self._zenoh.put(key, json.dumps(payload).encode("utf-8"))
        self._runmeta_published = True

    def _on_image(self, msg: Image) -> None:
        now = time.time()
        if self.max_fps > 0:
            min_period = 1.0 / self.max_fps
            if now - self._last_pub_wall < min_period:
                return
        self._last_pub_wall = now

        self._publish_runmeta_once()

        stamp = Time.from_msg(msg.header.stamp)
        camera_frame = self.camera_frame_override or msg.header.frame_id or "camera_link"

        # Image sha256
        sha256 = hashlib.sha256(bytes(msg.data)).hexdigest()

        # Convert to OpenCV for YOLO
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warning(f"cv_bridge conversion failed: {e}")
            return

        # Odom aligned to image timestamp
        odom_msg = self._closest_odom(stamp)
        odom_payload: dict[str, Any] = {
            "topic": self.odom_topic,
            "frame_id": "",
            "x": None,
            "y": None,
            "yaw": None,
            "vx": None,
            "vy": None,
            "wz": None,
        }
        if odom_msg is not None:
            odom_payload["frame_id"] = odom_msg.header.frame_id
            p = odom_msg.pose.pose.position
            q = odom_msg.pose.pose.orientation
            odom_payload["x"] = float(p.x)
            odom_payload["y"] = float(p.y)
            odom_payload["yaw"] = _q_to_yaw(q.x, q.y, q.z, q.w)
            v = odom_msg.twist.twist.linear
            w = odom_msg.twist.twist.angular
            odom_payload["vx"] = float(v.x)
            odom_payload["vy"] = float(v.y)
            odom_payload["wz"] = float(w.z)

        tf_ok, t_base_camera = self._lookup_base_to_camera(camera_frame, stamp)

        # YOLO inference
        results = self._yolo(cv_img, verbose=False)
        r0 = results[0]
        names = getattr(r0, "names", None) or getattr(self._yolo, "names", {})

        det_list: list[dict[str, Any]] = []
        boxes = getattr(r0, "boxes", None)
        if boxes is not None:
            for b in boxes:
                cls_id = int(b.cls[0].item()) if hasattr(b.cls[0], "item") else int(b.cls[0])
                conf = float(b.conf[0].item()) if hasattr(b.conf[0], "item") else float(b.conf[0])
                xyxy = b.xyxy[0].tolist()
                det_list.append(
                    {
                        "det_id": str(uuid.uuid4()),
                        "class_id": cls_id,
                        "class_name": str(names.get(cls_id, str(cls_id))),
                        "confidence": conf,
                        "bbox_xyxy": [float(x) for x in xyxy],
                    }
                )

        event_id = str(uuid.uuid4())
        event = {
            "schema": "maze.detection.v1",
            "event_id": event_id,
            "run_id": self.run_id,
            "robot_id": self.robot_id,
            "sequence": self._sequence,
            "image": {
                "topic": self.image_topic,
                "stamp": {"sec": int(msg.header.stamp.sec), "nanosec": int(msg.header.stamp.nanosec)},
                "frame_id": msg.header.frame_id,
                "width": int(msg.width),
                "height": int(msg.height),
                "encoding": str(msg.encoding),
                "sha256": sha256,
            },
            "odometry": odom_payload,
            "tf": {
                "base_frame": self.base_frame,
                "camera_frame": camera_frame,
                "t_base_camera": t_base_camera if t_base_camera is not None else [0.0] * 16,
                "tf_ok": bool(tf_ok),
            },
            "detections": det_list,
        }

        key = f"{self.key_prefix}/{self.robot_id}/{self.run_id}/detections/v1/{event_id}"
        self._zenoh.put(key, json.dumps(event).encode("utf-8"))
        self._sequence += 1


def main() -> None:
    rclpy.init()
    node = DetectorPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
