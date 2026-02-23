#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _require_file(path: Path, problems: list[str]) -> None:
    if not path.exists():
        problems.append(f"Missing file: {path.relative_to(ROOT)}")


def main() -> int:
    problems: list[str] = []

    _require_file(ROOT / "docker-compose.yaml", problems)
    _require_file(ROOT / "sql" / "schema.sql", problems)
    _require_file(ROOT / "world_assets.md", problems)
    _require_file(ROOT / "reproducibility.md", problems)
    _require_file(ROOT / "ingest" / "worker.py", problems)
    _require_file(ROOT / "turtlebot-maze" / "tb_autonomy" / "scripts" / "ds685_a2_detector.py", problems)

    world_assets = ROOT / "world_assets.md"
    if world_assets.exists():
        text = world_assets.read_text(encoding="utf-8")
        refs = re.findall(r"\(assets/world/(bench_\d\d_view_\d\d\.png)\)", text)
        if len(refs) != 8:
            problems.append(
                f"`world_assets.md` should reference 8 bench screenshots; found {len(refs)}"
            )
        for name in sorted(set(refs)):
            _require_file(ROOT / "assets" / "world" / name, problems)

    schema = ROOT / "sql" / "schema.sql"
    if schema.exists():
        sql = schema.read_text(encoding="utf-8")
        for token in [
            "CREATE TABLE",
            "detection_events",
            "detections",
            "event_id uuid PRIMARY KEY",
            "raw_event jsonb NOT NULL",
            "UNIQUE(run_id, robot_id, sequence)",
            "UNIQUE(event_id, det_id)",
        ]:
            if token not in sql:
                problems.append(f"`sql/schema.sql` missing expected token: {token!r}")

    if problems:
        print("Submission check: FAIL\n")
        for p in problems:
            print(f"- {p}")
        return 1

    print("Submission check: OK")
    print("- Core files present")
    print("- `world_assets.md` references 8 screenshots (all found)")
    print("- `sql/schema.sql` contains required tables/constraints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
