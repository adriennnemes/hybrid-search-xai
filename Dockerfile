# Simple Python image for reproducible deployment
FROM python:3.11-slim

WORKDIR /app

# Small system deps (usually enough for pip builds)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better Docker cache)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the application
COPY app /app/app

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI (Gradio is mounted under /ui)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]