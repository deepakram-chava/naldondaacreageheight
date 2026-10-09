FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y libgdal-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY models/ models/
# Note: In production, the SQLite DB should be on a persistent volume or PostgreSQL should be used.

ENV PYTHONPATH=/app
ENV PORT=8000

CMD uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
