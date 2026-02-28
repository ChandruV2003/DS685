FROM python:3.11-slim

WORKDIR /app

# CPU-only torch + open_clip for embeddings
RUN pip install --no-cache-dir \
      numpy==1.26.4 \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.3.1 torchvision==0.18.1 \
  && pip install --no-cache-dir \
      open_clip_torch==2.24.0 \
      Pillow==10.4.0 \
      psycopg[binary]==3.3.3 \
      pgvector==0.3.6

COPY *.py /app/

ENTRYPOINT ["python", "-u"]

