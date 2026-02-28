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


def _embed_images(paths: list[Path]) -> list[list[float]]:
    clip_model = os.getenv("CLIP_MODEL", "ViT-B-32")
    clip_pretrained = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")

    import torch
    import open_clip

    device = "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_pretrained, device=device
    )
    model.eval()

    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(preprocess(img))
    img_tensor = torch.stack(imgs).to(device)
    with torch.no_grad():
        emb = model.encode_image(img_tensor).cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
    emb = emb / norms
    return [e.tolist() for e in emb]


def main() -> None:
    parser = argparse.ArgumentParser(description="DS685 A3: semantic re-localization via pgvector KNN")
    parser.add_argument("crops", nargs="+", help="1–3 query crop image paths")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--top-places", type=int, default=3)
    parser.add_argument("--run-id", default="", help="Restrict search to this run_id (default: auto)")
    args = parser.parse_args()

    db_url = _require_env("DATABASE_URL")
    crop_paths = [Path(p) for p in args.crops]

    logging.info("Embedding %d query crops...", len(crop_paths))
    q_vecs = _embed_images(crop_paths)

    scores: dict[str, float] = {}
    with psycopg.connect(db_url, connect_timeout=10) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            run_id = uuid.UUID(args.run_id) if args.run_id else _pick_run_id(cur)
            if run_id:
                logging.info("Using run_id=%s", run_id)

            for q in q_vecs:
                q_vec = "[" + ",".join(f"{float(x):.9f}" for x in q) + "]"
                if run_id:
                    cur.execute(
                        """
                        SELECT k.place_id, (1.0 - (de.embedding <=> %s::vector)) AS sim
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
                        SELECT k.place_id, (1.0 - (de.embedding <=> %s::vector)) AS sim
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
                for place_id, sim in cur.fetchall():
                    scores[str(place_id)] = scores.get(str(place_id), 0.0) + float(sim or 0.0)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[: args.top_places]

        print("Top places:")
        for place_id, s in ranked:
            print(f"- {place_id}: score={s:.4f}")

        if ranked:
            best_place = ranked[0][0]
            with conn.cursor() as cur:
                if run_id:
                    cur.execute(
                        "SELECT avg(map_x), avg(map_y), avg(map_yaw) FROM keyframes WHERE run_id=%s AND place_id=%s",
                        (run_id, best_place),
                    )
                else:
                    cur.execute(
                        "SELECT avg(map_x), avg(map_y), avg(map_yaw) FROM keyframes WHERE place_id=%s",
                        (best_place,),
                    )
                mx, my, myaw = cur.fetchone()
            print("\nBest pose hypothesis (mean of keyframes in best place):")
            print(f"- place_id={best_place}")
            print(f"- map_x={mx} map_y={my} map_yaw={myaw}")


if __name__ == "__main__":
    main()
