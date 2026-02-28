from __future__ import annotations

import argparse
import logging
import os
import uuid
from pathlib import Path

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from PIL import Image


LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing env var: {name}")
    return v


def _embed_image(path: Path) -> list[float]:
    clip_model = os.getenv("CLIP_MODEL", "ViT-B-32")
    clip_pretrained = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")

    import torch
    import open_clip

    device = "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_pretrained, device=device
    )
    model.eval()

    img = Image.open(path).convert("RGB")
    img_tensor = torch.stack([preprocess(img)]).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_tensor).cpu().numpy().astype(np.float32)[0]
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb.tolist()


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


def main() -> None:
    parser = argparse.ArgumentParser(description="DS685 A3: vector query (top-k similar detections)")
    parser.add_argument("crop", help="Query crop image path")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--run-id", default="", help="Restrict search to this run_id (default: auto)")
    args = parser.parse_args()

    db_url = _require_env("DATABASE_URL")
    crop_path = Path(args.crop)
    if not crop_path.exists():
        raise SystemExit(f"File not found: {crop_path}")

    logging.info("Embedding query crop: %s", crop_path)
    q = _embed_image(crop_path)
    q_vec = "[" + ",".join(f"{float(x):.9f}" for x in q) + "]"

    with psycopg.connect(db_url, connect_timeout=10) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            run_id = uuid.UUID(args.run_id) if args.run_id else _pick_run_id(cur)

            if run_id:
                cur.execute(
                    """
                    SELECT
                      d.det_pk,
                      d.class_name,
                      d.confidence,
                      e.event_id,
                      e.stamp,
                      k.place_id,
                      (1.0 - (de.embedding <=> %s::vector)) AS sim
                    FROM detection_embeddings de
                    JOIN detections d ON d.det_pk = de.det_pk
                    JOIN detection_events e ON e.event_id = d.event_id
                    JOIN keyframes k ON k.event_id = e.event_id
                    WHERE e.run_id = %s
                      AND k.place_id IS NOT NULL
                    ORDER BY de.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (q_vec, run_id, q_vec, args.topk),
                )
            else:
                cur.execute(
                    """
                    SELECT
                      d.det_pk,
                      d.class_name,
                      d.confidence,
                      e.event_id,
                      e.stamp,
                      k.place_id,
                      (1.0 - (de.embedding <=> %s::vector)) AS sim
                    FROM detection_embeddings de
                    JOIN detections d ON d.det_pk = de.det_pk
                    JOIN detection_events e ON e.event_id = d.event_id
                    JOIN keyframes k ON k.event_id = e.event_id
                    WHERE k.place_id IS NOT NULL
                    ORDER BY de.embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (q_vec, q_vec, args.topk),
                )

            rows = cur.fetchall()

    print("Top matches:")
    for det_pk, class_name, conf, event_id, stamp, place_id, sim in rows:
        print(
            f"- det_pk={det_pk} sim={float(sim or 0.0):.4f} class={class_name} conf={float(conf or 0.0):.3f} "
            f"place={place_id} event={event_id} stamp={stamp}"
        )


if __name__ == "__main__":
    main()
