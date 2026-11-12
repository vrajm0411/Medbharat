FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps for inference
COPY deploy/requirements_serve.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy model code
COPY model/ model/
COPY eval/ eval/
COPY deploy/server.py server.py

# Create directories for model weights (mounted at runtime)
RUN mkdir -p /app/weights

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "server.py"]
