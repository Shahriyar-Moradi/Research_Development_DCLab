# The DCLab product in a container (package 12.1): the API server and, with another command, the job worker.
# Multi-stage: wheels are built in the first stage, the second runs as a user without privileges. No key is baked
# in: every key and connection string comes from the environment (docker-compose.yml, or the platform's secrets).

FROM python:3.11-slim AS build
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements/product.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r product.txt

FROM python:3.11-slim
# libgomp: LightGBM and XGBoost run their trees on OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 dclab
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index /wheels/* && rm -rf /wheels
WORKDIR /app
COPY --chown=dclab:dclab . /app
ENV PYTHONUNBUFFERED=1 \
    DCLAB_AGENT_HOME=/workspace \
    DCLAB_ML_PYTHON=/usr/local/bin/python
RUN mkdir -p /workspace && chown dclab:dclab /workspace
USER dclab
EXPOSE 8765
# 0.0.0.0 inside the container only; docker-compose.yml publishes it on 127.0.0.1
CMD ["sh", "-c", "python -m dclab_rnd.storage upgrade && exec python -m uvicorn dclab_rnd.agentic.server:app --host 0.0.0.0 --port 8765 --workers 1"]
