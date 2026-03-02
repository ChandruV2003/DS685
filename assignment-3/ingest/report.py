from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import psycopg


def main() -> None:
    parser = argparse.ArgumentParser(description="DS685 A3: DB run report")
    parser.add_argument("--out", default="", help="Write markdown report to this path")
    parser.add_argument("--run-id", default="", help="Filter report to a single run_id (optional)")
    args = parser.parse_args()

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise SystemExit("Missing DATABASE_URL env var")

    now = datetime.now(tz=timezone.utc).isoformat()

    with psycopg.connect(db_url, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            where = ""
            params: tuple[object, ...] = ()
            if args.run_id:
                where = "WHERE run_id = %s"
                params = (args.run_id,)

            cur.execute(f"SELECT count(*) FROM detection_events {where}", params)
            event_count = cur.fetchone()[0]

            cur.execute(
                f"""
                SELECT count(*)
                FROM detections d
                JOIN detection_events e ON e.event_id = d.event_id
                {where.replace('run_id', 'e.run_id')}
                """,
                params,
            )
            det_count = cur.fetchone()[0]

            cur.execute(
                f"SELECT count(*) FROM keyframes WHERE map_ok {('AND run_id = %s' if args.run_id else '')}",
                params,
            )
            map_ok_count = cur.fetchone()[0]

            cur.execute(
                f"""
                SELECT count(*)
                FROM detection_embeddings de
                JOIN detections d ON d.det_pk = de.det_pk
                JOIN detection_events e ON e.event_id = d.event_id
                {where.replace('run_id', 'e.run_id')}
                """,
                params,
            )
            emb_count = cur.fetchone()[0]

            cur.execute("SELECT count(*) FROM places")
            place_count = cur.fetchone()[0]

            cur.execute(
                f"SELECT count(*) FROM object_landmarks {where}",
                params,
            )
            landmark_count = cur.fetchone()[0]

            cur.execute(
                f"""
                SELECT count(*)
                FROM detection_landmarks dl
                JOIN detections d ON d.det_pk = dl.det_pk
                JOIN detection_events e ON e.event_id = d.event_id
                {where.replace('run_id', 'e.run_id')}
                """,
                params,
            )
            det_landmark_count = cur.fetchone()[0]

            cur.execute(
                f"""
                SELECT d.class_name, count(*) AS n
                FROM detections d
                JOIN detection_events e ON e.event_id = d.event_id
                {where.replace('run_id', 'e.run_id')}
                GROUP BY d.class_name
                ORDER BY n DESC, d.class_name ASC
                """
            )
            rows = cur.fetchall()

    lines: list[str] = []
    lines.append("# DS685 Assignment 3 — Run Report")
    lines.append("")
    lines.append(f"Generated: `{now}`")
    lines.append("")
    lines.append("## Counts")
    lines.append(f"- Detection events: **{event_count}**")
    lines.append(f"- Detections: **{det_count}**")
    lines.append(f"- Keyframes with map pose: **{map_ok_count}**")
    lines.append(f"- Detection embeddings: **{emb_count}**")
    lines.append(f"- Places: **{place_count}**")
    lines.append(f"- Object landmarks: **{landmark_count}**")
    lines.append(f"- Detection→landmark links: **{det_landmark_count}**")
    lines.append("")
    lines.append("## Class Histogram")
    lines.append("")
    lines.append("| class_name | count |")
    lines.append("|---|---:|")
    for class_name, n in rows:
        lines.append(f"| {class_name or '(null)'} | {n} |")
    lines.append("")

    report_md = "\n".join(lines)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Wrote report: {args.out}")
    else:
        print(report_md)


if __name__ == "__main__":
    main()
