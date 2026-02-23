# DS685 — Assignment 2

Object Detection Events with ROS 2 + Zenoh + PostgreSQL.

This folder contains:
- `turtlebot-maze/`: forked maze sim with A2 benches + objects and a headless launch file.
- `ingest/`: Zenoh → PostgreSQL ingest worker (idempotent inserts).
- `sql/schema.sql`: required DB schema.
- `world_assets.md`: bench/object inventory + camera-view screenshots (populate).
- `report/`: generated run report (`report.md`).
- `reproducibility.md`: exact steps to reproduce the system.

## Run (Docker)

From this folder:

```bash
docker compose up --build -d zenoh postgres ingest
docker compose up --build pipeline
```

Notes:
- The `pipeline` service launches the A2 world headlessly and runs the ROS detector publisher.
- On first run, Ultralytics may download YOLO weights (`yolov8n.pt`) inside the container.

## Generate the report

After running the pipeline for ~1–3 minutes:

```bash
docker compose run --rm ingest python /app/report.py --out /out/report.md
```

This writes `report/report.md` on the host (mounted into the ingest container).

## Stop / cleanup

```bash
docker compose down -v
```

## Reproducibility report

See `reproducibility.md`.
