FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY worker.py /app/worker.py
COPY report.py /app/report.py

CMD ["python", "-u", "/app/worker.py"]

