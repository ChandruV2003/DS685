from __future__ import annotations

import json
import logging
import os
import queue
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
import zenoh


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _zenoh_open(connect_endpoint: str) -> zenoh.Session:
    cfg = zenoh.Config.from_json5(json.dumps({"connect": {"endpoints": [connect_endpoint]}}))
    return zenoh.open(cfg)


def _to_uuid(value: Any) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _stamp_to_datetime_utc(stamp: dict[str, Any]) -> datetime:
    sec = int(stamp.get("sec", 0))
    nanosec = int(stamp.get("nanosec", 0))
    return datetime.fromtimestamp(sec + nanosec * 1e-9, tz=timezone.utc)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def insert_event(conn: psycopg.Connection, event: dict[str, Any]) -> None:
    event_id = _to_uuid(event["event_id"])
    run_id = _to_uuid(event["run_id"])
    robot_id = str(event["robot_id"])
    sequence = int(event["sequence"])

    image = event.get("image") or {}
    odom = event.get("odometry") or {}
    tf = event.get("tf") or {}
    detections = event.get("detections") or []
    stamp = _stamp_to_datetime_utc(image.get("stamp") or {})

    t_base_camera = tf.get("t_base_camera")
    if isinstance(t_base_camera, list):
        t_base_camera = [float(x) for x in t_base_camera]
    else:
        t_base_camera = None

    map_pose = event.get("map_pose") or {}
    map_ok = bool(map_pose.get("map_ok", False))
    map_x = _safe_float(map_pose.get("x"))
    map_y = _safe_float(map_pose.get("y"))
    map_yaw = _safe_float(map_pose.get("yaw"))

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO detection_events (
                  event_id, run_id, robot_id, sequence, stamp,
                  image_frame_id, image_sha256, width, height, encoding,
                  x, y, yaw, vx, vy, wz,
                  tf_ok, t_base_camera, raw_event
                )
                VALUES (
                  %(event_id)s, %(run_id)s, %(robot_id)s, %(sequence)s, %(stamp)s,
                  %(image_frame_id)s, %(image_sha256)s, %(width)s, %(height)s, %(encoding)s,
                  %(x)s, %(y)s, %(yaw)s, %(vx)s, %(vy)s, %(wz)s,
                  %(tf_ok)s, %(t_base_camera)s, %(raw_event)s
                )
                ON CONFLICT DO NOTHING
                """,
                {
                    "event_id": event_id,
                    "run_id": run_id,
                    "robot_id": robot_id,
                    "sequence": sequence,
                    "stamp": stamp,
                    "image_frame_id": image.get("frame_id"),
                    "image_sha256": image.get("sha256"),
                    "width": image.get("width"),
                    "height": image.get("height"),
                    "encoding": image.get("encoding"),
                    "x": _safe_float(odom.get("x")),
                    "y": _safe_float(odom.get("y")),
                    "yaw": _safe_float(odom.get("yaw")),
                    "vx": _safe_float(odom.get("vx")),
                    "vy": _safe_float(odom.get("vy")),
                    "wz": _safe_float(odom.get("wz")),
                    "tf_ok": bool(tf.get("tf_ok", False)),
                    "t_base_camera": t_base_camera,
                    "raw_event": Jsonb(event),
                },
            )

            cur.execute(
                """
                INSERT INTO keyframes (
                  event_id, run_id, robot_id, sequence, stamp,
                  map_ok, map_x, map_y, map_yaw
                )
                VALUES (
                  %(event_id)s, %(run_id)s, %(robot_id)s, %(sequence)s, %(stamp)s,
                  %(map_ok)s, %(map_x)s, %(map_y)s, %(map_yaw)s
                )
                ON CONFLICT DO NOTHING
                """,
                {
                    "event_id": event_id,
                    "run_id": run_id,
                    "robot_id": robot_id,
                    "sequence": sequence,
                    "stamp": stamp,
                    "map_ok": map_ok,
                    "map_x": map_x,
                    "map_y": map_y,
                    "map_yaw": map_yaw,
                },
            )

            for det in detections:
                det_id_raw = det.get("det_id")
                det_id = _to_uuid(det_id_raw) if det_id_raw else None
                bbox = det.get("bbox_xyxy") or [None, None, None, None]
                x1, y1, x2, y2 = (bbox + [None, None, None, None])[:4]
                cur.execute(
                    """
                    INSERT INTO detections (
                      event_id, det_id, class_id, class_name, confidence,
                      x1, y1, x2, y2
                    )
                    VALUES (
                      %(event_id)s, %(det_id)s, %(class_id)s, %(class_name)s, %(confidence)s,
                      %(x1)s, %(y1)s, %(x2)s, %(y2)s
                    )
                    ON CONFLICT DO NOTHING
                    """,
                    {
                        "event_id": event_id,
                        "det_id": det_id,
                        "class_id": det.get("class_id"),
                        "class_name": det.get("class_name"),
                        "confidence": _safe_float(det.get("confidence")),
                        "x1": _safe_float(x1),
                        "y1": _safe_float(y1),
                        "x2": _safe_float(x2),
                        "y2": _safe_float(y2),
                    },
                )


def main() -> None:
    db_url = _require_env("DATABASE_URL")
    zenoh_connect = os.getenv("ZENOH_CONNECT", "tcp/zenoh:7447")
    key_expr = os.getenv("ZENOH_KEY_EXPR", "maze/**/detections/v1/*")

    logging.info("Connecting to Postgres...")
    conn = psycopg.connect(db_url, connect_timeout=10)

    logging.info("Connecting to Zenoh (%s)...", zenoh_connect)
    session = _zenoh_open(zenoh_connect)

    inserted_events = 0
    event_q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=2000)

    def handler(sample: zenoh.Sample) -> None:
        try:
            payload = sample.payload.to_string()
            event = json.loads(payload)
            if event.get("schema") != "maze.detection.v1":
                logging.warning("Ignoring unknown schema: %s", event.get("schema"))
                return
            try:
                event_q.put(event, timeout=0.2)
            except queue.Full:
                logging.warning(
                    "Dropping event due to full ingest queue (key=%s)",
                    getattr(sample, "key_expr", None),
                )
        except Exception:
            logging.exception("Failed to ingest sample on key=%s", getattr(sample, "key_expr", None))

    logging.info("Subscribing to: %s", key_expr)
    session.declare_subscriber(key_expr, handler)

    logging.info("Ready. Waiting for events...")
    try:
        while True:
            event = event_q.get()
            try:
                insert_event(conn, event)
                inserted_events += 1
                if inserted_events % 25 == 0:
                    logging.info("Ingested %d event messages so far", inserted_events)
            except Exception:
                logging.exception("DB insert failed (will reconnect and continue)")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = psycopg.connect(db_url, connect_timeout=10)
                try:
                    insert_event(conn, event)
                    inserted_events += 1
                except Exception:
                    logging.exception("DB insert failed after reconnect (dropping event)")
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        try:
            session.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

