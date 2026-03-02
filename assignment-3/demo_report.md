# DS685 Assignment 3 — Demo Report

This report summarizes my end-to-end semantic mapping + re-localization pipeline:

TurtleBot3 (Gazebo Sim) → ROS 2 detector → Zenoh → PostgreSQL (events/keyframes) → pgvector embeddings → Apache AGE semantic graph → semantic re-localization.

## Runs

- Run A (map build): `c3022101-c153-4a48-b100-d42d0080c611`
- Run B (query): `de1fec99-1142-47dd-a727-6cea247c89c1`

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
docker compose run --rm ingest python /app/report.py --run-id c3022101-c153-4a48-b100-d42d0080c611
```

Fill in the latest numbers:

- Detection events: `487`
- Detections: `309`
- Keyframes with map pose: `120`
- Detection embeddings: `80`
- Places: `13`
- Object landmarks: `18`

## Required queries (evidence)

### Vector: top-k similar detections for a query crop

```bash
docker compose run --rm semantics /app/vector_query.py --run-id c3022101-c153-4a48-b100-d42d0080c611 /data/query_run_b/q_04e59915-c5e1-4535-b3bc-1179b9fc2dff_bed.jpg --topk 10
```

Paste output snippet here:

```
Top matches:
- det_pk=144 sim=0.9648 class=bed conf=0.649 place=p_-2_-1 event=e7d268a4-1298-4310-8a8e-30ba64f3deef stamp=1970-01-01 00:05:38.652000+00:00
- det_pk=150 sim=0.9478 class=bed conf=0.732 place=p_-1_-1 event=1d2e606d-8e68-4e25-8995-5ea5c66bd92c stamp=1970-01-01 00:05:43.650000+00:00
- det_pk=204 sim=0.9452 class=bed conf=0.732 place=p_-1_-1 event=ce8b3c57-be95-4853-a763-5160c2925709 stamp=1970-01-01 00:06:36.750000+00:00
- det_pk=208 sim=0.9451 class=bed conf=0.744 place=p_-1_-1 event=3c048086-44b8-42cc-9a3d-f7200ce01ad8 stamp=1970-01-01 00:06:40.752000+00:00
- det_pk=156 sim=0.9445 class=bed conf=0.737 place=p_-1_-1 event=e17f8cad-e81a-4a77-bf9a-7cb0a913fca3 stamp=1970-01-01 00:05:49.701000+00:00
```

### Graph: reachable places within N hops containing an object class

```bash
docker compose run --rm semantics /app/graph_query.py --run-id c3022101-c153-4a48-b100-d42d0080c611 --start-place p_-1_-1 --class "bed" --hops 2
```

Paste output snippet here:

```
Reachable places:
- p_-1_-1
- p_-2_-1
```

### Re-localization: top-3 candidate places from query crops

```bash
docker compose run --rm semantics /app/relocalize.py --run-id c3022101-c153-4a48-b100-d42d0080c611 /data/query_run_b/q_04e59915-c5e1-4535-b3bc-1179b9fc2dff_bed.jpg
```

Paste output snippet here:

```
Top places:
- p_-1_-1: score=18.5369
- p_0_3: score=9.6395
- p_-2_-1: score=4.7174

Best pose hypothesis (mean of keyframes in best place):
- place_id=p_-1_-1
- map_x=-0.8566066512862444 map_y=-0.7592771923044432 map_yaw=0.9230464109692946
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
