# dclabai in a container: the studio (server and its job worker) on port 8765, its workspace in /workspace.
#   docker run -p 127.0.0.1:8765:8765 -v dclabai:/workspace ghcr.io/shahriyar-moradi/dclabai
# No key is baked in: a model's settings come from the environment (-e DCLAB_LLM_BASE_URL=... -e OPENAI_MODEL=...).
FROM python:3.12-slim AS build
WORKDIR /src
COPY . .
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[notebook,postgres,anthropic,mcp]"

FROM python:3.12-slim
# libgomp: LightGBM and XGBoost run their trees on OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 dclab && mkdir -p /workspace && chown dclab:dclab /workspace
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir --no-index --find-links /wheels "dclabai[notebook,postgres,anthropic,mcp]" && rm -rf /wheels
ENV PYTHONUNBUFFERED=1 DCLAB_AGENT_HOME=/workspace
USER dclab
WORKDIR /workspace
EXPOSE 8765
# 0.0.0.0 inside the container only: publish it on 127.0.0.1 unless you turn sign-in on (DCLAB_AUTH=password, PostgreSQL)
CMD ["dclab", "start", "--host", "0.0.0.0", "--port", "8765", "--no-browser"]
