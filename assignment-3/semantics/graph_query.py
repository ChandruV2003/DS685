from __future__ import annotations

import argparse
import json
import logging
import os
import re
import uuid
from typing import Any

import psycopg


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


def _ag_to_py(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (dict, list, int, float, bool)):
        return v
    s = str(v)
    try:
        return json.loads(s)
    except Exception:
        return s.strip('"')


def _pick_run_id(cur: psycopg.Cursor) -> uuid.UUID | None:
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
    if not row:
        return None
    return uuid.UUID(str(row[0]))


_GRAPH_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _graph_literal(graph: str) -> str:
    if not _GRAPH_NAME_RE.match(graph):
        raise SystemExit(f"Invalid AGE graph name: {graph!r}")
    return "'" + graph.replace("'", "''") + "'"


def main() -> None:
    parser = argparse.ArgumentParser(description="DS685 A3: AGE graph query")
    parser.add_argument("--start-place", required=True, help="Start place_id, e.g. p_0_0")
    parser.add_argument("--class", required=True, dest="class_name", help="Object class_name, e.g. stop sign")
    parser.add_argument("--hops", type=int, default=2, help="Max adjacency hops from start")
    parser.add_argument("--run-id", default="", help="Restrict Object nodes to this run_id (default: auto)")
    args = parser.parse_args()

    db_url = _require_env("DATABASE_URL")
    graph = os.getenv("AGE_GRAPH", "ds685_semantic")
    graph_lit = _graph_literal(graph)

    with psycopg.connect(db_url, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            _ensure_age_loaded(cur)
            run_id = uuid.UUID(args.run_id) if args.run_id else _pick_run_id(cur)

            hops = max(0, int(args.hops))
            rows: list[Any] = []

            def run_query(query: str, params: dict[str, Any]) -> None:
                dq_tag = "ds685"
                dq = f"${dq_tag}$"
                if dq in query:
                    raise SystemExit("Internal error: cypher query contains an unexpected dollar-quote tag")
                query_lit = f"{dq}{query}{dq}"
                cur.execute(
                    f"SELECT * FROM cypher({graph_lit}, {query_lit}, %s::agtype) AS (place_id agtype)",
                    (json.dumps(params),),
                )
                rows.extend([_ag_to_py(r[0]) for r in cur.fetchall()])

            if run_id:
                common = {"start": args.start_place, "cn": args.class_name, "run_id": str(run_id)}
                run_query(
                    """
                    MATCH (p:Place {place_id: $start})<-[:LOCATED_IN]-(o:Object {class_name: $cn, run_id: $run_id})
                    RETURN DISTINCT p.place_id AS place_id
                    """,
                    common,
                )
                if hops > 0:
                    run_query(
                        f"""
                        MATCH (s:Place {{place_id: $start}})-[:ADJ*1..{hops}]->(p:Place)
                              <-[:LOCATED_IN]-(o:Object {{class_name: $cn, run_id: $run_id}})
                        RETURN DISTINCT p.place_id AS place_id
                        """,
                        common,
                    )
            else:
                common = {"start": args.start_place, "cn": args.class_name}
                run_query(
                    """
                    MATCH (p:Place {place_id: $start})<-[:LOCATED_IN]-(o:Object {class_name: $cn})
                    RETURN DISTINCT p.place_id AS place_id
                    """,
                    common,
                )
                if hops > 0:
                    run_query(
                        f"""
                        MATCH (s:Place {{place_id: $start}})-[:ADJ*1..{hops}]->(p:Place)
                              <-[:LOCATED_IN]-(o:Object {{class_name: $cn}})
                        RETURN DISTINCT p.place_id AS place_id
                        """,
                        common,
                    )

    rows = sorted({str(r) for r in rows if r is not None})
    print("Reachable places:")
    for r in rows:
        print(f"- {r}")


if __name__ == "__main__":
    main()
