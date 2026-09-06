# Single image used by BOTH the one-shot `pipeline` (ETL + model training) and
# the long-running `api` service. They share dependencies, so one image keeps
# things simple and guarantees the API scores with the exact library versions
# the models were trained under. Entry point differs per service (set in
# docker-compose.yml).

FROM python:3.11-slim

# System deps: postgres client (pg_isready for the wait loop) + build basics.
RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-client \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the project. .dockerignore keeps out .venv, data CSVs, target/, etc.
COPY . .

# dbt reads its profile from the project dir (env vars supply the connection).
ENV DBT_PROFILES_DIR=/app/phase3_dbt

# The pipeline entrypoint script performs the one-time setup.
RUN chmod +x /app/docker/pipeline_entrypoint.sh

# Default command is the API; the pipeline service overrides it in compose.
# Bind to $PORT if the platform injects one (Render/Railway do), else 8000.
# Shell form so $PORT expands at runtime.
EXPOSE 8000
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --app-dir phase7_api
