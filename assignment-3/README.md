# DS685 — Assignment 3

Semantic graph + semantic re-localization using **pgvector** and **Apache AGE** on PostgreSQL.

This assignment builds on the same TurtleBot3 maze simulation + detection-event pipeline as Assignment 2, then:

- Stores CLIP embeddings for detection crops (pgvector)
- Builds a semantic graph (Apache AGE)
- Performs semantic re-localization via vector KNN + place ranking

See `reproducibility.md` for the full, step-by-step runbook.

Quick sanity check:

```bash
python3 check_submission.py
```
