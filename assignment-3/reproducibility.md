# DS685 Assignment 3 — Reproducibility

This folder runs:

TurtleBot3 (Gazebo Sim) → ROS 2 detector → Zenoh → PostgreSQL (events + keyframes),
then builds embeddings + semantic structures using pgvector + Apache AGE.

## Requirements

- Docker + Docker Compose

## Bring up the live pipeline (Run A)

From `assignment-3/`:

```bash
docker compose up --build -d zenoh postgres ingest
docker compose up --build pipeline
```

This will:
- Launch the maze simulation headlessly
- Launch Nav2 bringup for localization against `sim_house_map.yaml` (to provide `map` frame)
- Publish an initial pose
- Run the detector node which publishes detection events over Zenoh

Images are saved under `data/images/` (named by `sha256`).

### Explore the maze (recommended)

To populate multiple places (grid cells) and observe semantic objects across the map, send a small set of Nav2 goals:

```bash
docker compose exec -T pipeline bash -lc '
  source /opt/ros/jazzy/setup.bash
  source /turtlebot_ws/install/setup.bash
  source /overlay_ws/install/setup.bash
  python3 /workspace/scripts/navigate_waypoints.py --loops 2 --timeout 180 --dwell 8
'
```

### Identify the Run A ID (recommended)

Each pipeline run uses a unique UUID `run_id`. Capture it so you can explicitly target Run A for embedding/graph/relocalization:

```bash
docker compose exec -T postgres psql -U ds685 -d ds685 -c "
  SELECT run_id, count(*) AS keyframes_map_ok
  FROM keyframes
  WHERE map_ok
  GROUP BY run_id
  ORDER BY keyframes_map_ok DESC
"
```

## Embed detections (pgvector)

```bash
docker compose run --rm --build semantics /app/embed_detections.py --run-id <RUN_A_UUID> --limit 2000
```

## Build places + landmarks + AGE graph

```bash
docker compose run --rm --build semantics /app/build_semantics.py --reset-graph --run-id <RUN_A_UUID>
```

## Run B (different start pose)

Edit `X_POSE`, `Y_POSE`, `YAW` in `docker-compose.yaml` and re-run:

```bash
docker compose up --build pipeline
```

Capture 1–3 query crops (manual, or save crops from any detector output).

To auto-capture crops from the current camera stream:

```bash
docker compose exec -T pipeline python3 /workspace/scripts/capture_query_crops.py --out /data/query --max 3
```

If you get repeated `No detections` messages, first navigate closer to a semantic object (e.g., a bench) using the exploration helper above, then re-run crop capture.

## Re-localize

```bash
docker compose run --rm --build semantics /app/relocalize.py --run-id <RUN_A_UUID> /data/query/crop1.jpg /data/query/crop2.jpg
```

## Required queries

### Vector (top-k similar detections)

```bash
docker compose run --rm --build semantics /app/vector_query.py --run-id <RUN_A_UUID> /data/query/crop1.jpg --topk 10
```

### Graph (reachable places within N hops with an object class)

```bash
docker compose run --rm --build semantics /app/graph_query.py --run-id <RUN_A_UUID> --start-place p_0_0 --class "stop sign" --hops 2
```

## Generate the report

```bash
docker compose run --rm --build ingest python /app/report.py --run-id <RUN_A_UUID> --out /out/report.md
```

## Stop / cleanup

```bash
docker compose down -v
```
