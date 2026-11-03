"""DCLab Watch (B4 of the 2026-11-03 plan): the spans DCLab already opens, sent to MLflow and Langfuse as OpenTelemetry traces.

``observe.span`` already gives every request, job, agent run, agent step, tool call and model request a W3C trace id and a
span id (23.11). This module turns each finished span into an OTLP span with the same ids and sends it, in batches from a
background thread, to the targets ``DCLAB_TRACE_EXPORT`` names:

- ``mlflow``: ``DCLAB_MLFLOW_OTLP`` (default ``http://127.0.0.1:5000/v1/traces``), experiment ``DCLAB_MLFLOW_EXPERIMENT`` (default 0);
- ``langfuse``: ``DCLAB_LANGFUSE_OTLP`` (default ``http://127.0.0.1:3000/api/public/otel/v1/traces``), with
  ``LANGFUSE_PUBLIC_KEY`` and ``LANGFUSE_SECRET_KEY`` as Basic auth.

``none`` (the default: tests and a clean checkout) sends nothing and costs nothing. What leaves is names, ids, counts,
models, prompt hashes, euro and seconds: an attribute is sent only when its key is in ``ALLOWED`` (or starts with ``dclab.``)
and its value is a number, a flag or a short word; never a table cell, a prompt or an answer. A span of a sealed set
(``dclab.split`` sealed) is never sent. A target that is down never slows DCLab: the queue is bounded, spans that do not fit
are dropped and counted, and a failed send is counted, never raised.
"""

from __future__ import annotations

import base64
import collections
import os
import threading
import time
import urllib.request
from typing import Any, Optional

TARGETS = ("mlflow", "langfuse")
ALLOWED = {"gen_ai.system", "gen_ai.operation.name", "gen_ai.request.model", "gen_ai.response.model", "gen_ai.usage.input_tokens",
           "gen_ai.usage.output_tokens", "gen_ai.usage.cost", "session.id", "langfuse.session.id", "langfuse.trace.name",
           "langfuse.observation.type", "langfuse.observation.model.name", "service.name"}
MAX_TEXT = 120  # a short word or an id; anything longer is not sent
QUEUE = 10_000
BATCH = 200
EVERY = 2.0  # seconds between sends
TIMEOUT = 3.0

_queue: collections.deque = collections.deque(maxlen=QUEUE)
_lock = threading.Lock()
_counts = {"queued": 0, "dropped": 0, "sent": 0, "failed": 0, "refused": 0}
_thread: Optional[threading.Thread] = None


def targets() -> list[str]:
    raw = (os.environ.get("DCLAB_TRACE_EXPORT") or "none").strip().lower()
    return [t for t in (x.strip() for x in raw.split(",")) if t in TARGETS]


def clean(attributes: dict[str, Any]) -> dict[str, Any]:
    """The attributes that may leave: allowed keys, numbers, flags and short words (no newline); the rest is dropped."""
    out: dict[str, Any] = {}
    for key, value in (attributes or {}).items():
        if not (key in ALLOWED or key.startswith("dclab.")):
            continue
        if isinstance(value, bool) or (isinstance(value, (int, float)) and value == value):
            out[key] = value
        elif isinstance(value, str) and len(value) <= MAX_TEXT and "\n" not in value:
            out[key] = value
    return out


def record(trace_id: str, span_id: str, parent: Optional[str], name: str, start_ns: int, end_ns: int,
           attributes: Optional[dict[str, Any]] = None, error: bool = False) -> bool:
    """Queue one finished span when a target is set; False when nothing is sent (no target, a sealed span, a full queue)."""
    if not targets():
        return False
    attrs = clean(attributes or {})
    if str(attrs.get("dclab.split", "")).startswith("sealed"):
        _counts["refused"] += 1
        return False
    with _lock:
        if len(_queue) >= QUEUE:
            _counts["dropped"] += 1
            return False
        _queue.append({"trace_id": trace_id, "span_id": span_id, "parent": parent, "name": name[:MAX_TEXT], "start": int(start_ns),
                       "end": int(end_ns), "attributes": attrs, "error": bool(error)})
        _counts["queued"] += 1
    _start()
    return True


def _start() -> None:
    global _thread
    if _thread is None or not _thread.is_alive():
        _thread = threading.Thread(target=_loop, name="dclab-trace-export", daemon=True)
        _thread.start()


def _loop() -> None:
    while True:
        time.sleep(EVERY)
        flush()


def flush() -> int:
    """Send what is queued, in batches, to every target; returns how many spans were taken from the queue."""
    taken = 0
    while True:
        with _lock:
            batch = [_queue.popleft() for _ in range(min(BATCH, len(_queue)))]
        if not batch:
            return taken
        taken += len(batch)
        body = encode(batch)
        for target in targets():
            try:
                _post(target, body)
                _counts["sent"] += len(batch)
            except Exception:  # noqa: BLE001 - a target that is down is counted, never raised into DCLab
                _counts["failed"] += len(batch)


def encode(spans: list[dict[str, Any]]) -> bytes:
    """An OTLP ExportTraceServiceRequest (protobuf) of the spans, with DCLab's own ids."""
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
    from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span, Status

    def value(v: Any) -> AnyValue:
        if isinstance(v, bool):
            return AnyValue(bool_value=v)
        if isinstance(v, int):
            return AnyValue(int_value=v)
        if isinstance(v, float):
            return AnyValue(double_value=v)
        return AnyValue(string_value=str(v))

    out = []
    for s in spans:
        out.append(Span(trace_id=bytes.fromhex(s["trace_id"]), span_id=bytes.fromhex(s["span_id"]),
                        parent_span_id=bytes.fromhex(s["parent"]) if s.get("parent") else b"", name=s["name"],
                        kind=Span.SPAN_KIND_INTERNAL, start_time_unix_nano=s["start"], end_time_unix_nano=s["end"],
                        attributes=[KeyValue(key=k, value=value(v)) for k, v in sorted(s["attributes"].items())],
                        status=Status(code=Status.STATUS_CODE_ERROR if s["error"] else Status.STATUS_CODE_OK)))
    from opentelemetry.proto.resource.v1.resource_pb2 import Resource

    request = ExportTraceServiceRequest(resource_spans=[ResourceSpans(
        resource=Resource(attributes=[KeyValue(key="service.name", value=AnyValue(string_value="dclab"))]),
        scope_spans=[ScopeSpans(spans=out)])])
    return request.SerializeToString()


def _post(target: str, body: bytes) -> None:
    headers = {"Content-Type": "application/x-protobuf"}
    if target == "mlflow":
        url = os.environ.get("DCLAB_MLFLOW_OTLP") or "http://127.0.0.1:5000/v1/traces"
        headers["x-mlflow-experiment-id"] = os.environ.get("DCLAB_MLFLOW_EXPERIMENT") or "0"
    else:
        url = os.environ.get("DCLAB_LANGFUSE_OTLP") or "http://127.0.0.1:3000/api/public/otel/v1/traces"
        pair = f"{os.environ.get('LANGFUSE_PUBLIC_KEY', '')}:{os.environ.get('LANGFUSE_SECRET_KEY', '')}"
        headers["Authorization"] = "Basic " + base64.b64encode(pair.encode()).decode()
        headers["x-langfuse-ingestion-version"] = "4"
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # noqa: S310 - a configured local endpoint
        response.read()


def stats() -> dict[str, Any]:
    return {**_counts, "waiting": len(_queue), "targets": targets()}


__all__ = ["targets", "clean", "record", "flush", "encode", "stats", "ALLOWED", "TARGETS"]
