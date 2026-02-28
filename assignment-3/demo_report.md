# DS685 Assignment 3 — Demo Report

This report summarizes my end-to-end semantic mapping + re-localization pipeline:

TurtleBot3 (Gazebo Sim) → ROS 2 detector → Zenoh → PostgreSQL (events/keyframes) → pgvector embeddings → Apache AGE semantic graph → semantic re-localization.

## Runs

- Run A (map build): `<RUN_A_UUID>`
- Run B (query): `<RUN_B_UUID>`

## System design

### Data transport

- Zenoh router: `zenohd` (Compose service `zenoh`)
- Detection events key design: `maze/<robot_id>/<run_id>/detections/v1/<event_id>`
- Run metadata key: `maze/<robot_id>/<run_id>/runmeta/v1`

### Database schema

Relational tables (PostgreSQL):

- `detection_events` — raw detection event JSON + core metadata (idempotent via `PRIMARY KEY(event_id)` and `UNIQUE(run_id, robot_id, sequence)`)
- `detections` — per-event bounding boxes + class metadata (idempotent via `UNIQUE(event_id, det_id)`)
- `keyframes` — per-event map-frame pose (`map_x`, `map_y`, `map_yaw`) aligned to the image timestamp
- `detection_embeddings` — CLIP embeddings (pgvector `vector(512)`)
- `places` — grid-binned places from map poses
- `object_landmarks` — fused object landmarks (distance + embedding similarity thresholds)
- `detection_landmarks` — mapping from detections → fused landmarks (enables graph build)

Graph (Apache AGE):

- Graph name: `ds685_semantic` (env `AGE_GRAPH`)
- Node types: `Run`, `Keyframe`, `Pose`, `Place`, `Object`, `Observation`
- Edge types: `HAS_KEYFRAME`, `HAS_POSE`, `HAS_OBSERVATION`, `OBSERVES`, `LOCATED_IN`, `ADJ`

### Perception

- Detector: Ultralytics YOLO (`yolov8n.pt`)
- Camera topic: `/camera/image_raw`
- Odometry: `/odom`
- TF:
  - base frame: `base_footprint`
  - camera frame: `camera_link`
  - map pose: TF lookup `map <- base_footprint` at the image timestamp

### Embeddings

- Model: OpenCLIP `ViT-B-32` pretrained `laion2b_s34b_b79k`
- Dimension: 512
- Normalization: L2-normalized embeddings (cosine distance for KNN)

### Place construction

- Method: grid binning on `(map_x, map_y)`
- Cell size: `CELL_SIZE_M=1.0`

### Object landmark fusion

- Within each `(place_id, class_name)` bucket, merge detections into a landmark if:
  - distance ≤ `FUSE_DIST_M=0.75` meters, and
  - embedding cosine similarity ≥ `FUSE_SIM=0.80`

## Results

Generate counts + histogram:

```bash
docker compose run --rm ingest python /app/report.py
```

Fill in the latest numbers:

- Detection events: `<N_EVENTS>`
- Detections: `<N_DETECTIONS>`
- Keyframes with map pose: `<N_KEYFRAMES_MAP_OK>`
- Detection embeddings: `<N_EMBEDDINGS>`
- Places: `<N_PLACES>`
- Object landmarks: `<N_LANDMARKS>`

## Required queries (evidence)

### Vector: top-k similar detections for a query crop

```bash
docker compose run --rm semantics /app/vector_query.py --run-id <RUN_A_UUID> /data/query/<crop>.jpg --topk 10
```

Paste output snippet here:

```
<PASTE OUTPUT>
```

### Graph: reachable places within N hops containing an object class

```bash
docker compose run --rm semantics /app/graph_query.py --run-id <RUN_A_UUID> --start-place p_0_0 --class "stop sign" --hops 2
```

Paste output snippet here:

```
<PASTE OUTPUT>
```

### Re-localization: top-3 candidate places from query crops

```bash
docker compose run --rm semantics /app/relocalize.py --run-id <RUN_A_UUID> /data/query/<crop1>.jpg /data/query/<crop2>.jpg
```

Paste output snippet here:

```
<PASTE OUTPUT>
```

## Success + failure analysis

### What worked well

- Query crops of distinctive objects (e.g., a high-contrast sign) produced high similarity scores and concentrated place rankings.
- The Run A place bins were stable because localization was done in `map` frame with `use_sim_time`.

### Common failure modes

- Ambiguous objects (e.g., visually similar shapes/colors) produced diffuse place rankings.
- Early in a run, `map <- base_footprint` TF may not be available yet; those frames have `map_ok=false` and are excluded from global semantic mapping.
- Partial occlusions or low-confidence detections reduce embedding quality (bad crops → weak KNN matches).

### Improvements (if iterating)

- Use a larger YOLO model (or tune confidence threshold) to reduce false positives.
- Add per-landmark mean embedding update + cross-place fusion to handle boundary cases.
- Increase exploration coverage in Run A to densify the semantic map.

