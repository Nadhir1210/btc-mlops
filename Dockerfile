FROM python:3.11-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installation des dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements-api.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy application code
COPY api/ ./api/
COPY training/prepare_data.py ./training/
COPY training/catboost_best.pkl ./training/
COPY training/scaler.pkl ./training/
COPY training/mlflow.db ./training/
COPY training/mlruns/ ./training/mlruns/

# Port pour l'API FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
