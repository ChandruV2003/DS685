from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

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


def main() -> None:
    parser = argparse.ArgumentParser(description="DS685 A3: embed detection crops into pgvector")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--run-id", default="", help="Restrict embeddings to this run_id (default: all runs)")
    parser.add_argument(
        "--all-keyframes",
        action="store_true",
        help="Embed detections even when a keyframe has no map pose (default: map_ok only)",
    )
    args = parser.parse_args()

    db_url = _require_env("DATABASE_URL")
    data_dir = Path(os.getenv("DATA_DIR", "/data"))
    image_dir = data_dir / "images"

    clip_model = os.getenv("CLIP_MODEL", "ViT-B-32")
    clip_pretrained = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")
    model_tag = f"open_clip:{clip_model}:{clip_pretrained}"

    logging.info("Loading CLIP model (%s, %s)...", clip_model, clip_pretrained)
    import torch
    import open_clip

    device = "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model, pretrained=clip_pretrained, device=device
    )
    model.eval()

    logging.info("Connecting to Postgres...")
    with psycopg.connect(db_url, connect_timeout=10) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            run_filter_sql = ""
            run_filter_args: list[Any] = []
            if args.run_id:
                run_filter_sql = "AND e.run_id = %s"
                run_filter_args.append(args.run_id)

            pose_filter_sql = "" if args.all_keyframes else "AND k.map_ok"

            cur.execute(
                f"""
                SELECT d.det_pk, e.image_sha256, d.x1, d.y1, d.x2, d.y2
                FROM detections d
                JOIN detection_events e ON e.event_id = d.event_id
                JOIN keyframes k ON k.event_id = e.event_id
                LEFT JOIN detection_embeddings de ON de.det_pk = d.det_pk
                WHERE de.det_pk IS NULL
                  AND e.image_sha256 IS NOT NULL
                  AND d.x1 IS NOT NULL AND d.y1 IS NOT NULL AND d.x2 IS NOT NULL AND d.y2 IS NOT NULL
                  AND d.x2 > d.x1 AND d.y2 > d.y1
                  {pose_filter_sql}
                  {run_filter_sql}
                ORDER BY e.stamp DESC, d.det_pk ASC
                LIMIT %s
                """,
                (*run_filter_args, args.limit),
            )
            rows = cur.fetchall()

        if not rows:
            logging.info("No detections missing embeddings.")
            return

        logging.info("Embedding %d detections...", len(rows))

        inserted = 0
        batch: list[tuple[int, str, float, float, float, float]] = []

        def flush() -> None:
            nonlocal inserted
            if not batch:
                return

            det_pks = [r[0] for r in batch]
            imgs = []
            for _, sha, x1, y1, x2, y2 in batch:
                img_path = image_dir / f"{sha}.jpg"
                if not img_path.exists():
                    imgs.append(None)
                    continue
                img = Image.open(img_path).convert("RGB")
                w, h = img.size
                left = max(0, min(int(x1), w - 1))
                top = max(0, min(int(y1), h - 1))
                right = max(left + 1, min(int(x2), w))
                bottom = max(top + 1, min(int(y2), h))
                crop = img.crop((left, top, right, bottom))
                imgs.append(preprocess(crop))

            ok_idx = [i for i, t in enumerate(imgs) if t is not None]
            if not ok_idx:
                batch.clear()
                return

            import torch

            img_tensor = torch.stack([imgs[i] for i in ok_idx]).to(device)
            with torch.no_grad():
                emb = model.encode_image(img_tensor).cpu().numpy().astype(np.float32)
            # Normalize for cosine distance
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
            emb = emb / norms

            with conn.transaction():
                with conn.cursor() as cur:
                    for out_i, row_i in enumerate(ok_idx):
                        det_pk = det_pks[row_i]
                        cur.execute(
                            """
                            INSERT INTO detection_embeddings (det_pk, model, embedding)
                            VALUES (%s, %s, %s)
                            ON CONFLICT DO NOTHING
                            """,
                            (det_pk, model_tag, emb[out_i].tolist()),
                        )
                        inserted += 1

            batch.clear()

        for det_pk, sha, x1, y1, x2, y2 in rows:
            if None in (det_pk, sha, x1, y1, x2, y2):
                continue
            batch.append((int(det_pk), str(sha), float(x1), float(y1), float(x2), float(y2)))
            if len(batch) >= args.batch:
                flush()

        flush()

    logging.info("Done. Inserted ~%d embeddings.", inserted)


if __name__ == "__main__":
    main()
