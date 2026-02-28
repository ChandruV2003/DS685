-- DS685 Assignment 3 — PostgreSQL schema
-- Extensions: pgvector + Apache AGE

CREATE EXTENSION IF NOT EXISTS vector;

-- Base tables (from Assignment 2)
CREATE TABLE IF NOT EXISTS detection_events (
  event_id uuid PRIMARY KEY,
  run_id uuid NOT NULL,
  robot_id text NOT NULL,
  sequence bigint NOT NULL,
  stamp timestamptz NOT NULL,

  image_frame_id text,
  image_sha256 text,
  width int,
  height int,
  encoding text,

  x double precision,
  y double precision,
  yaw double precision,
  vx double precision,
  vy double precision,
  wz double precision,

  tf_ok boolean,
  t_base_camera double precision[],

  raw_event jsonb NOT NULL,

  UNIQUE(run_id, robot_id, sequence)
);

CREATE TABLE IF NOT EXISTS detections (
  det_pk bigserial PRIMARY KEY,
  event_id uuid REFERENCES detection_events(event_id) ON DELETE CASCADE,
  det_id uuid,
  class_id int,
  class_name text,
  confidence double precision,
  x1 double precision,
  y1 double precision,
  x2 double precision,
  y2 double precision,
  UNIQUE(event_id, det_id)
);

-- Keyframes (map-frame pose requirement)
CREATE TABLE IF NOT EXISTS keyframes (
  event_id uuid PRIMARY KEY REFERENCES detection_events(event_id) ON DELETE CASCADE,
  run_id uuid NOT NULL,
  robot_id text NOT NULL,
  sequence bigint NOT NULL,
  stamp timestamptz NOT NULL,
  map_ok boolean NOT NULL DEFAULT false,
  map_x double precision,
  map_y double precision,
  map_yaw double precision,
  place_id text,
  UNIQUE(run_id, robot_id, sequence)
);

-- pgvector embeddings for detection crops
CREATE TABLE IF NOT EXISTS detection_embeddings (
  det_pk bigint PRIMARY KEY REFERENCES detections(det_pk) ON DELETE CASCADE,
  model text,
  embedding vector(512)
);

-- Semantic places (constructed from map poses)
CREATE TABLE IF NOT EXISTS places (
  place_id text PRIMARY KEY,
  grid_x int NOT NULL,
  grid_y int NOT NULL,
  center_x double precision,
  center_y double precision,
  cell_size_m double precision NOT NULL
);

-- Object landmark fusion output (required fields + mean_embedding)
CREATE TABLE IF NOT EXISTS object_landmarks (
  landmark_id uuid PRIMARY KEY,
  run_id uuid NOT NULL,
  place_id text REFERENCES places(place_id) ON DELETE CASCADE,
  class_name text NOT NULL,
  mean_x double precision,
  mean_y double precision,
  first_seen timestamptz,
  last_seen timestamptz,
  observation_count bigint NOT NULL,
  mean_embedding vector(512)
);

-- Mapping: detection -> fused landmark
CREATE TABLE IF NOT EXISTS detection_landmarks (
  det_pk bigint PRIMARY KEY REFERENCES detections(det_pk) ON DELETE CASCADE,
  landmark_id uuid REFERENCES object_landmarks(landmark_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_keyframes_run ON keyframes(run_id);
CREATE INDEX IF NOT EXISTS idx_keyframes_place ON keyframes(place_id);
CREATE INDEX IF NOT EXISTS idx_detections_event ON detections(event_id);
CREATE INDEX IF NOT EXISTS idx_object_landmarks_run_place ON object_landmarks(run_id, place_id);
