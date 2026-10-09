FROM python:3.13-slim

WORKDIR /app
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1

# lightgbm and xgboost need this system library
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY artifacts ./artifacts
COPY templates ./templates
COPY app.py .

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]