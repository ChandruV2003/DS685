# DS685 Assignment 2 — Reproducibility

This repository implements the assignment pipeline:

TurtleBot3 (Gazebo Sim) → ROS 2 detector node (YOLO) → Zenoh events → PostgreSQL.

## Requirements

- Docker + Docker Compose

## Key expressions

- Run metadata (published once per run): `maze/<robot_id>/<run_id>/runmeta/v1`
- Detection events: `maze/<robot_id>/<run_id>/detections/v1/<event_id>`

`run_id` is provided via `RUN_ID` or generated on startup. `robot_id` defaults to `tb3_sim`.

## Bring up the stack

From `assignment-2/`:

```bash
docker compose up --build -d zenoh postgres ingest
docker compose up --build pipeline
```

Notes:
- The `pipeline` service launches the world headlessly and runs the detector publisher.
- On first run, Ultralytics may download `yolov8n.pt` inside the container.

## Detector metadata + time alignment

- The published event uses the **image message timestamp** (`sensor_msgs/Image.header.stamp`) as the frame timestamp.
- `/odom` is time-aligned by selecting the closest odometry sample to the image timestamp.
- The base→camera transform is queried from TF at the image timestamp; if unavailable, `tf_ok=false` is published.

## Database idempotency

The ingest worker inserts with `ON CONFLICT DO NOTHING` and relies on constraints for idempotency:

- `detection_events.event_id` primary key
- `UNIQUE(run_id, robot_id, sequence)`
- `UNIQUE(event_id, det_id)` in `detections`

## Generate world-assets screenshots

This captures 2 robot-camera images per bench into `assets/world/`:

```bash
./scripts/capture_world_assets.sh
```

The bench/object inventory and screenshots are documented in `world_assets.md`.

## Generate the run report

After running the pipeline for ~1–3 minutes:

```bash
docker compose run --rm ingest python /app/report.py --out /out/report.md
```

This writes `report/report.md` on the host.

## Stop / cleanup

```bash
docker compose down -v
```
