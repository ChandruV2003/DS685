from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing env var: {name}")
    return v


def _ensure_age_loaded(cur: psycopg.Cursor) -> None:
    cur.execute("LOAD 'age';")
    cur.execute('SET search_path = ag_catalog, "$user", public;')


def _graph_exists(cur: psycopg.Cursor, graph: str) -> bool:
    cur.execute("SELECT 1 FROM ag_catalog.ag_graph WHERE name=%s", (graph,))
    return cur.fetchone() is not None


_GRAPH_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _graph_literal(graph: str) -> str:
    if not _GRAPH_NAME_RE.match(graph):
        raise SystemExit(f"Invalid AGE graph name: {graph!r}")
    return "'" + graph.replace("'", "''") + "'"


def _cypher(
    cur: psycopg.Cursor,
    graph: str,
    query: str,
    params: dict[str, Any] | None = None,
    out: str = "v agtype",
) -> None:
    graph_lit = _graph_literal(graph)
    dq_tag = "ds685"
    dq = f"${dq_tag}$"
    if dq in query:
        raise SystemExit("Internal error: cypher query contains an unexpected dollar-quote tag")
    query_lit = f"{dq}{query}{dq}"

    if params is None:
        cur.execute(f"SELECT * FROM cypher({graph_lit}, {query_lit}) AS ({out})")
        return

    cur.execute(
        f"SELECT * FROM cypher({graph_lit}, {query_lit}, %s::agtype) AS ({out})",
        (json.dumps(params),),
    )


def _pick_run_id(cur: psycopg.Cursor) -> uuid.UUID:
    cur.execute(
        """
        SELECT run_id, count(*) AS n
        FROM keyframes
        WHERE map_ok
        GROUP BY run_id
        ORDER BY n DESC, run_id ASC
        LIMIT 1
        """
    )
    row = cur.fetchone()
    if row:
        return uuid.UUID(str(row[0]))

    cur.execute("SELECT run_id FROM detection_events ORDER BY stamp DESC LIMIT 1")
    row = cur.fetchone()
    if row:
        return uuid.UUID(str(row[0]))

    raise SystemExit("No runs found in the database yet. Run the pipeline first.")


@dataclass(frozen=True)
class PlaceKey:
    gx: int
    gy: int

    @property
    def place_id(self) -> str:
        return f"p_{self.gx}_{self.gy}"


def _grid_key(x: float, y: float, cell: float) -> PlaceKey:
    return PlaceKey(int(math.floor(x / cell)), int(math.floor(y / cell)))


@dataclass
class Landmark:
    landmark_id: uuid.UUID
    run_id: uuid.UUID
    place_id: str
    class_name: str
    mean_x: float
    mean_y: float
    mean_embedding: np.ndarray
    first_seen: datetime
    last_seen: datetime
    observation_count: int

    def try_add(self, x: float, y: float, emb: np.ndarray, stamp: datetime, dist_m: float, sim: float) -> None:
        self.mean_x = (self.mean_x * self.observation_count + x) / (self.observation_count + 1)
        self.mean_y = (self.mean_y * self.observation_count + y) / (self.observation_count + 1)
        self.mean_embedding = (self.mean_embedding * self.observation_count + emb) / (self.observation_count + 1)
        n = float(np.linalg.norm(self.mean_embedding) + 1e-9)
        self.mean_embedding = self.mean_embedding / n
        self.observation_count += 1
        if stamp < self.first_seen:
            self.first_seen = stamp
        if stamp > self.last_seen:
            self.last_seen = stamp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DS685 A3: build places, fuse object landmarks, and build an Apache AGE semantic graph"
    )
    parser.add_argument("--run-id", default="", help="Run ID to build from (default: auto-pick best)")
    parser.add_argument("--cell-size-m", type=float, default=float(os.getenv("CELL_SIZE_M", "1.0")))
    parser.add_argument("--fuse-dist-m", type=float, default=float(os.getenv("FUSE_DIST_M", "0.75")))
    parser.add_argument("--fuse-sim", type=float, default=float(os.getenv("FUSE_SIM", "0.80")))
    parser.add_argument("--limit-keyframes", type=int, default=800, help="Max keyframes to graph (newest first)")
    parser.add_argument("--reset-relational", action="store_true", help="Clear places/landmarks and reassign")
    parser.add_argument("--reset-graph", action="store_true", help="Drop and recreate the AGE graph")
    args = parser.parse_args()

    db_url = _require_env("DATABASE_URL")
    graph = os.getenv("AGE_GRAPH", "ds685_semantic")

    logging.info("Connecting to Postgres...")
    with psycopg.connect(db_url, connect_timeout=10) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            run_id = uuid.UUID(args.run_id) if args.run_id else _pick_run_id(cur)

        logging.info("Using run_id=%s", run_id)

        if args.reset_relational:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute("UPDATE keyframes SET place_id=NULL WHERE run_id=%s", (run_id,))
                    cur.execute("DELETE FROM object_landmarks WHERE run_id=%s", (run_id,))

        # Build places + assign keyframes.place_id
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT event_id, map_x, map_y
                FROM keyframes
                WHERE run_id=%s
                  AND map_ok
                  AND map_x IS NOT NULL AND map_y IS NOT NULL
                ORDER BY stamp DESC
                LIMIT %s
                """,
                (run_id, args.limit_keyframes),
            )
            kf_rows = cur.fetchall()

        places: dict[PlaceKey, list[tuple[uuid.UUID, float, float]]] = {}
        for event_id, x, y in kf_rows:
            pk = _grid_key(float(x), float(y), args.cell_size_m)
            places.setdefault(pk, []).append((uuid.UUID(str(event_id)), float(x), float(y)))

        with conn.transaction():
            with conn.cursor() as cur:
                for pk, pts in places.items():
                    cx = sum(p[1] for p in pts) / len(pts)
                    cy = sum(p[2] for p in pts) / len(pts)
                    cur.execute(
                        """
                        INSERT INTO places (place_id, grid_x, grid_y, center_x, center_y, cell_size_m)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (place_id) DO UPDATE
                          SET grid_x = EXCLUDED.grid_x,
                              grid_y = EXCLUDED.grid_y,
                              center_x = EXCLUDED.center_x,
                              center_y = EXCLUDED.center_y,
                              cell_size_m = EXCLUDED.cell_size_m
                        """,
                        (pk.place_id, pk.gx, pk.gy, cx, cy, args.cell_size_m),
                    )
                    for event_id, _, _ in pts:
                        cur.execute(
                            "UPDATE keyframes SET place_id=%s WHERE event_id=%s",
                            (pk.place_id, event_id),
                        )

        logging.info("Built/updated places=%d", len(places))

        # Landmark fusion: distance + embedding similarity thresholds
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  d.det_pk,
                  d.event_id,
                  e.stamp,
                  d.class_name,
                  k.place_id,
                  k.map_x,
                  k.map_y,
                  de.embedding
                FROM detections d
                JOIN detection_events e ON e.event_id = d.event_id
                JOIN keyframes k ON k.event_id = e.event_id
                JOIN detection_embeddings de ON de.det_pk = d.det_pk
                WHERE e.run_id=%s
                  AND k.map_ok
                  AND k.place_id IS NOT NULL
                  AND k.map_x IS NOT NULL AND k.map_y IS NOT NULL
                  AND d.class_name IS NOT NULL
                ORDER BY e.stamp ASC, d.det_pk ASC
                """,
                (run_id,),
            )
            det_rows = cur.fetchall()

        if not det_rows:
            raise SystemExit("No detections with embeddings for this run. Run embed_detections.py first.")

        landmarks_by_key: dict[tuple[str, str], list[Landmark]] = {}
        det_to_landmark: dict[int, uuid.UUID] = {}

        for det_pk, _, stamp, class_name, place_id, map_x, map_y, embedding in det_rows:
            det_pk_i = int(det_pk)
            place_id_s = str(place_id)
            class_name_s = str(class_name)
            x = float(map_x)
            y = float(map_y)
            emb = np.asarray(embedding, dtype=np.float32)
            n = float(np.linalg.norm(emb) + 1e-9)
            emb = emb / n

            key = (place_id_s, class_name_s)
            candidates = landmarks_by_key.setdefault(key, [])

            best: Landmark | None = None
            best_sim = -1.0
            for lm in candidates:
                dx = x - lm.mean_x
                dy = y - lm.mean_y
                dist = float(math.sqrt(dx * dx + dy * dy))
                if dist > args.fuse_dist_m:
                    continue
                sim = float(np.dot(emb, lm.mean_embedding))
                if sim < args.fuse_sim:
                    continue
                if sim > best_sim:
                    best_sim = sim
                    best = lm

            if best is None:
                lm = Landmark(
                    landmark_id=uuid.uuid4(),
                    run_id=run_id,
                    place_id=place_id_s,
                    class_name=class_name_s,
                    mean_x=x,
                    mean_y=y,
                    mean_embedding=emb,
                    first_seen=stamp,
                    last_seen=stamp,
                    observation_count=1,
                )
                candidates.append(lm)
                det_to_landmark[det_pk_i] = lm.landmark_id
                continue

            best.try_add(x, y, emb, stamp, args.fuse_dist_m, best_sim)
            det_to_landmark[det_pk_i] = best.landmark_id

        landmarks: list[Landmark] = []
        for lms in landmarks_by_key.values():
            landmarks.extend(lms)

        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("DELETE FROM object_landmarks WHERE run_id=%s", (run_id,))
                for lm in landmarks:
                    cur.execute(
                        """
                        INSERT INTO object_landmarks (
                          landmark_id, run_id, place_id, class_name,
                          mean_x, mean_y, first_seen, last_seen, observation_count, mean_embedding
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            lm.landmark_id,
                            lm.run_id,
                            lm.place_id,
                            lm.class_name,
                            lm.mean_x,
                            lm.mean_y,
                            lm.first_seen,
                            lm.last_seen,
                            lm.observation_count,
                            lm.mean_embedding.tolist(),
                        ),
                    )

                for det_pk, landmark_id in det_to_landmark.items():
                    cur.execute(
                        """
                        INSERT INTO detection_landmarks (det_pk, landmark_id)
                        VALUES (%s, %s)
                        ON CONFLICT (det_pk) DO UPDATE SET landmark_id = EXCLUDED.landmark_id
                        """,
                        (det_pk, landmark_id),
                    )

        logging.info("Fused landmarks=%d (from %d detections)", len(landmarks), len(det_to_landmark))

        # AGE graph build
        with conn.transaction():
            with conn.cursor() as cur:
                _ensure_age_loaded(cur)

                if args.reset_graph and _graph_exists(cur, graph):
                    cur.execute("SELECT drop_graph(%s, true)", (graph,))

                if not _graph_exists(cur, graph):
                    cur.execute("SELECT create_graph(%s)", (graph,))

                _cypher(
                    cur,
                    graph,
                    "MERGE (r:Run {run_id: $run_id}) RETURN r",
                    {"run_id": str(run_id)},
                )

                cur.execute("SELECT place_id, grid_x, grid_y, center_x, center_y FROM places")
                place_rows = cur.fetchall()
                by_grid: dict[tuple[int, int], str] = {}
                for place_id, gx, gy, cx, cy in place_rows:
                    pid = str(place_id)
                    by_grid[(int(gx), int(gy))] = pid
                    _cypher(
                        cur,
                        graph,
                        """
                        MERGE (p:Place {place_id: $pid})
                        SET p.grid_x=$gx, p.grid_y=$gy, p.center_x=$cx, p.center_y=$cy
                        RETURN p
                        """,
                        {"pid": pid, "gx": int(gx), "gy": int(gy), "cx": float(cx or 0.0), "cy": float(cy or 0.0)},
                    )

                for (gx, gy), pid in by_grid.items():
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nb = by_grid.get((gx + dx, gy + dy))
                        if not nb:
                            continue
                        _cypher(
                            cur,
                            graph,
                            """
                            MATCH (a:Place {place_id: $a}), (b:Place {place_id: $b})
                            MERGE (a)-[:ADJ]->(b)
                            RETURN a
                            """,
                            {"a": pid, "b": nb},
                        )

                cur.execute(
                    """
                    SELECT landmark_id, place_id, class_name, mean_x, mean_y, observation_count
                    FROM object_landmarks
                    WHERE run_id=%s
                    """,
                    (run_id,),
                )
                for lid, place_id, class_name, mean_x, mean_y, n in cur.fetchall():
                    _cypher(
                        cur,
                        graph,
                        """
                        MERGE (o:Object {landmark_id: $lid})
                        SET o.class_name=$cn, o.mean_x=$mx, o.mean_y=$my, o.n=$n, o.run_id=$run_id
                        RETURN o
                        """,
                        {
                            "lid": str(lid),
                            "cn": str(class_name),
                            "mx": float(mean_x or 0.0),
                            "my": float(mean_y or 0.0),
                            "n": int(n),
                            "run_id": str(run_id),
                        },
                    )

                    _cypher(
                        cur,
                        graph,
                        """
                        MATCH (o:Object {landmark_id: $lid}), (p:Place {place_id: $pid})
                        MERGE (o)-[:LOCATED_IN]->(p)
                        RETURN o
                        """,
                        {"lid": str(lid), "pid": str(place_id)},
                    )

                cur.execute(
                    """
                    SELECT event_id, robot_id, sequence, stamp, map_x, map_y, map_yaw
                    FROM keyframes
                    WHERE run_id=%s AND map_ok AND place_id IS NOT NULL
                    ORDER BY stamp DESC
                    LIMIT %s
                    """,
                    (run_id, args.limit_keyframes),
                )
                keyframes = cur.fetchall()
                event_ids = [uuid.UUID(str(r[0])) for r in keyframes]

                for event_id, robot_id, seq, stamp, mx, my, myaw in keyframes:
                    eid = str(event_id)
                    pose_id = f"pose_{eid}"
                    _cypher(
                        cur,
                        graph,
                        """
                        MERGE (k:Keyframe {event_id: $eid})
                        SET k.run_id=$run_id, k.robot_id=$robot_id, k.sequence=$seq, k.stamp=$stamp
                        RETURN k
                        """,
                        {
                            "eid": eid,
                            "run_id": str(run_id),
                            "robot_id": str(robot_id),
                            "seq": int(seq),
                            "stamp": stamp.isoformat() if hasattr(stamp, "isoformat") else str(stamp),
                        },
                    )
                    _cypher(
                        cur,
                        graph,
                        """
                        MERGE (p:Pose {pose_id: $pid})
                        SET p.x=$x, p.y=$y, p.yaw=$yaw, p.stamp=$stamp, p.run_id=$run_id
                        RETURN p
                        """,
                        {
                            "pid": pose_id,
                            "x": float(mx or 0.0),
                            "y": float(my or 0.0),
                            "yaw": float(myaw or 0.0),
                            "stamp": stamp.isoformat() if hasattr(stamp, "isoformat") else str(stamp),
                            "run_id": str(run_id),
                        },
                    )
                    _cypher(
                        cur,
                        graph,
                        """
                        MATCH (r:Run {run_id: $run_id}), (k:Keyframe {event_id: $eid})
                        MERGE (r)-[:HAS_KEYFRAME]->(k)
                        RETURN r
                        """,
                        {"run_id": str(run_id), "eid": eid},
                    )
                    _cypher(
                        cur,
                        graph,
                        """
                        MATCH (k:Keyframe {event_id: $eid}), (p:Pose {pose_id: $pid})
                        MERGE (k)-[:HAS_POSE]->(p)
                        RETURN k
                        """,
                        {"eid": eid, "pid": pose_id},
                    )

                if event_ids:
                    cur.execute(
                        """
                        SELECT d.det_pk, d.event_id, d.class_name, d.confidence, dl.landmark_id
                        FROM detections d
                        JOIN detection_landmarks dl ON dl.det_pk = d.det_pk
                        WHERE d.event_id = ANY(%s)
                        ORDER BY d.det_pk ASC
                        """,
                        (event_ids,),
                    )
                    dets = cur.fetchall()
                else:
                    dets = []

                for det_pk, event_id, class_name, conf, landmark_id in dets:
                    det_pk_i = int(det_pk)
                    eid = str(event_id)
                    _cypher(
                        cur,
                        graph,
                        """
                        MERGE (o:Observation {det_pk: $det_pk})
                        SET o.event_id=$eid, o.class_name=$cn, o.confidence=$conf, o.run_id=$run_id
                        RETURN o
                        """,
                        {
                            "det_pk": det_pk_i,
                            "eid": eid,
                            "cn": str(class_name),
                            "conf": float(conf or 0.0),
                            "run_id": str(run_id),
                        },
                    )
                    _cypher(
                        cur,
                        graph,
                        """
                        MATCH (k:Keyframe {event_id: $eid}), (o:Observation {det_pk: $det_pk})
                        MERGE (k)-[:HAS_OBSERVATION]->(o)
                        RETURN k
                        """,
                        {"eid": eid, "det_pk": det_pk_i},
                    )
                    _cypher(
                        cur,
                        graph,
                        """
                        MATCH (o:Observation {det_pk: $det_pk}), (obj:Object {landmark_id: $lid})
                        MERGE (o)-[:OBSERVES]->(obj)
                        RETURN o
                        """,
                        {"det_pk": det_pk_i, "lid": str(landmark_id)},
                    )

        logging.info("Done. Graph=%s run_id=%s places=%d keyframes=%d landmarks=%d", graph, run_id, len(places), len(keyframes), len(landmarks))


if __name__ == "__main__":
    main()
