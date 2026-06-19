FROM python:3.13-slim

ARG MLFLOW_VERSION=3.13.0

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install "mlflow[auth]==${MLFLOW_VERSION}" psycopg2-binary psutil nvidia-ml-py

RUN useradd -r -u 1001 -d /app -s /usr/sbin/nologin appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

USER appuser
