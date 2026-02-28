#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _require_file(path: Path, problems: list[str]) -> None:
    if not path.exists():
        problems.append(f"Missing file: {path.relative_to(ROOT)}")


def main() -> int:
    problems: list[str] = []

    _require_file(ROOT / "docker-compose.yaml", problems)
    _require_file(ROOT / "sql" / "schema.sql", problems)
    _require_file(ROOT / "reproducibility.md", problems)
    _require_file(ROOT / "demo_report.md", problems)

    _require_file(ROOT / "ingest" / "worker.py", problems)
    _require_file(ROOT / "ingest" / "report.py", problems)

    _require_file(ROOT / "semantics" / "embed_detections.py", problems)
    _require_file(ROOT / "semantics" / "build_semantics.py", problems)
    _require_file(ROOT / "semantics" / "vector_query.py", problems)
    _require_file(ROOT / "semantics" / "graph_query.py", problems)
    _require_file(ROOT / "semantics" / "relocalize.py", problems)

    _require_file(ROOT / "scripts" / "run_pipeline.sh", problems)
    _require_file(ROOT / "scripts" / "ds685_a3_detector.py", problems)

    schema = ROOT / "sql" / "schema.sql"
    if schema.exists():
        sql = schema.read_text(encoding="utf-8")
        for token in [
            "CREATE EXTENSION IF NOT EXISTS vector",
            "CREATE TABLE IF NOT EXISTS detection_events",
            "CREATE TABLE IF NOT EXISTS detections",
            "CREATE TABLE IF NOT EXISTS keyframes",
            "CREATE TABLE IF NOT EXISTS detection_embeddings",
            "CREATE TABLE IF NOT EXISTS places",
            "CREATE TABLE IF NOT EXISTS object_landmarks",
            "CREATE TABLE IF NOT EXISTS detection_landmarks",
            "raw_event jsonb NOT NULL",
            "UNIQUE(run_id, robot_id, sequence)",
        ]:
            if token not in sql:
                problems.append(f"`sql/schema.sql` missing expected token: {token!r}")

    build_semantics = ROOT / "semantics" / "build_semantics.py"
    if build_semantics.exists():
        text = build_semantics.read_text(encoding="utf-8")
        for token in ["LOAD 'age'", "cypher(", "Place", "Object", "Observation", "Keyframe", "Pose"]:
            if token not in text:
                problems.append(f"`semantics/build_semantics.py` missing expected token: {token!r}")

    repro = ROOT / "reproducibility.md"
    if repro.exists():
        text = repro.read_text(encoding="utf-8")
        for token in ["embed_detections.py", "build_semantics.py", "vector_query.py", "graph_query.py", "relocalize.py"]:
            if token not in text:
                problems.append(f"`reproducibility.md` missing expected reference: {token!r}")
        if len(re.findall(r"<RUN_A_UUID>", text)) == 0:
            problems.append("`reproducibility.md` should mention <RUN_A_UUID> placeholder (or real run id)")

    if problems:
        print("Submission check: FAIL\n")
        for p in problems:
            print(f"- {p}")
        return 1

    print("Submission check: OK")
    print("- Core files present")
    print("- Schema includes required tables for A3")
    print("- Reproducibility includes required commands/queries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

