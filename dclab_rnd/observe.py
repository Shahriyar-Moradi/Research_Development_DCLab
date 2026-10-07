"""Logs, health and metrics (package 12.4).

Logs: one JSON object per line on stderr when ``DCLAB_LOG_FORMAT=json`` (the containers set it; plain text otherwise),
with a request id that the response carries back (``X-Request-ID``). A request's line holds its method, its route
(the path pattern, never the values in it), its status, its time, and the ids of the person and the workspace: never
a secret, a prompt, a query string or a data value.

Metrics: counters and sums in Prometheus's text format, served on ``DCLAB_METRICS_PORT`` (off when unset) at
``DCLAB_METRICS_HOST`` (127.0.0.1 unless set): a port of its own, never the public one. Each process (an app
instance, a worker) serves its own; Prometheus adds them up.

    dclab_http_requests_total{method,route,status}        dclab_http_request_seconds_sum / _count {method,route}
    dclab_jobs_total{kind,status}                         dclab_job_seconds_sum / _count {kind}
    dclab_job_failures_total{kind,status}                 (failed or interrupted)
    dclab_model_requests_total{purpose,tier,outcome}      dclab_model_tokens_total{purpose,direction}   dclab_model_cost_eur_total{purpose}
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any

REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("dclab_request_id", default=None)
_VALID_ID = re.compile(r"[A-Za-z0-9._-]{8,64}")

log = logging.getLogger("dclab")


# ---------------------------------------------------------------------- logs

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {"at": datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds"), "level": record.levelname.lower(),
                 "logger": record.name, "message": record.getMessage()}
        request_id = REQUEST_ID.get()
        if request_id:
            entry["request_id"] = request_id
        entry.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            entry["error"] = record.exc_info[0].__name__ if record.exc_info[0] else "error"  # the type only: an exception's text can hold values
        return json.dumps(entry, ensure_ascii=False, default=str)


class PlainFormatter(logging.Formatter):
    """A plain line with the same fields as key=value (the default outside the containers)."""

    def __init__(self):
        super().__init__("%(asctime)s %(levelname)s %(name)s %(message)s")

    def format(self, record: logging.LogRecord) -> str:
        fields = {**({"request_id": REQUEST_ID.get()} if REQUEST_ID.get() else {}), **(getattr(record, "fields", {}) or {})}
        shown = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        line = logging.Formatter.format(self, logging.makeLogRecord({**record.__dict__, "exc_info": None, "exc_text": None}))
        return f"{line} {shown}" if shown else line


_CONFIGURED = [False]


def configure_logging() -> None:
    """Once per process: JSON lines on stderr with DCLAB_LOG_FORMAT=json, plain lines otherwise."""
    if _CONFIGURED[0]:
        return
    _CONFIGURED[0] = True
    handler = logging.StreamHandler(sys.stderr)
    if json_lines():
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(PlainFormatter())
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False


METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})


def method_label(method: str) -> str:
    return method if method in METHODS else "OTHER"


def request_id(offered: str | None) -> str:
    """The id a request is known by: the one the caller (a proxy) sent when it is a plain token, else a new one."""
    return offered if offered and _VALID_ID.fullmatch(offered) else secrets.token_hex(8)


def json_lines() -> bool:
    return (os.environ.get("DCLAB_LOG_FORMAT") or "").strip().lower() == "json"


def request_line(status: int, **fields: Any) -> None:
    """A request's line: every request with JSON lines (a log store filters them), only server errors in plain text
    (a developer's terminal and the test runner's output stay readable)."""
    if status >= 500 or json_lines():
        event("request", status=status, **fields)


def event(message: str, **fields: Any) -> None:
    """One log line with named fields (callers pass ids, counts, statuses: never a value a person or a model wrote)."""
    log.info(message, extra={"fields": fields})


# ---------------------------------------------------------------------- metrics

class Metrics:
    """Counters and sums by name and labels; thread-safe; rendered in Prometheus's text format."""

    def __init__(self):
        self._lock = threading.Lock()
        self._values: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._about: dict[str, tuple[str, str]] = {}  # family name → (type, help)

    def add(self, name: str, amount: float = 1.0, about: str = "", mtype: str = "counter", **labels: Any) -> None:
        key = (name, tuple(sorted((k, str(v)) for k, v in labels.items())))
        family = re.sub(r"_(sum|count)$", "", name) if mtype == "summary" else name
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount
            self._about.setdefault(family, (mtype, about))

    def value(self, name: str, **labels: Any) -> float:
        with self._lock:
            return self._values.get((name, tuple(sorted((k, str(v)) for k, v in labels.items()))), 0.0)

    def render(self) -> str:
        with self._lock:
            items = sorted(self._values.items())
            about = dict(self._about)
        out, said = [], set()
        for (name, labels), value in items:
            family = name if name in about else re.sub(r"_(sum|count)$", "", name)
            if family not in said:
                mtype, text = about.get(family, ("untyped", ""))
                out += [f"# HELP {family} {text}", f"# TYPE {family} {mtype}"]
                said.add(family)
            shown = ",".join(f'{k}="{_escape(v)}"' for k, v in labels)
            number = repr(float(value)) if value != int(value) or abs(value) >= 2**53 else str(int(value))  # every digit: rate() needs them
            out.append(f"{name}{{{shown}}} {number}" if shown else f"{name} {number}")
        return "\n".join(out) + "\n"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


METRICS = Metrics()


def observe_request(method: str, route: str, status: int, seconds: float) -> None:
    METRICS.add("dclab_http_requests_total", 1, "HTTP requests by route and status", method=method, route=route, status=status)
    METRICS.add("dclab_http_request_seconds_sum", seconds, "Time spent answering HTTP requests", "summary", method=method, route=route)
    METRICS.add("dclab_http_request_seconds_count", 1, "Time spent answering HTTP requests", "summary", method=method, route=route)


def observe_job(kind: str, status: str, seconds: float) -> None:
    METRICS.add("dclab_jobs_total", 1, "Jobs finished, by kind and how they ended", kind=kind, status=status)
    METRICS.add("dclab_job_seconds_sum", seconds, "Time jobs ran", "summary", kind=kind)
    METRICS.add("dclab_job_seconds_count", 1, "Time jobs ran", "summary", kind=kind)
    if status in ("failed", "interrupted"):
        METRICS.add("dclab_job_failures_total", 1, "Jobs that failed or were interrupted", kind=kind, status=status)


def observe_model(purpose: str, tier: str, outcome: str, input_tokens: int, output_tokens: int, cost_eur: float | None) -> None:
    METRICS.add("dclab_model_requests_total", 1, "Model requests by purpose, tier and outcome", purpose=purpose, tier=tier, outcome=outcome)
    METRICS.add("dclab_model_tokens_total", input_tokens or 0, "Model tokens", purpose=purpose, direction="in")
    METRICS.add("dclab_model_tokens_total", output_tokens or 0, "Model tokens", purpose=purpose, direction="out")
    if cost_eur:
        METRICS.add("dclab_model_cost_eur_total", cost_eur, "Model spend in euros (priced models)", purpose=purpose)


# ---------------------------------------------------------------------- the metrics port

_SERVER: list[Any] = [None]


def serve_metrics() -> int | None:
    """Start the metrics server once per process, when DCLAB_METRICS_PORT is set; returns the port (None when off)."""
    if _SERVER[0] is not None:
        return _SERVER[0].server_address[1]
    raw = (os.environ.get("DCLAB_METRICS_PORT") or "").strip()
    if not raw:
        return None
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            body = METRICS.render().encode() if self.path.split("?")[0] == "/metrics" else b"not found\n"
            self.send_response(200 if self.path.split("?")[0] == "/metrics" else 404)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # the scrapes are not worth a line each
            return

    try:
        server = ThreadingHTTPServer(((os.environ.get("DCLAB_METRICS_HOST") or "127.0.0.1").strip(), int(raw)), Handler)
    except (OSError, ValueError, OverflowError) as exc:  # a busy or bad port: the metrics are off, the app still serves
        log.warning("metrics off", extra={"fields": {"port": raw, "error": type(exc).__name__}})
        print(f"metrics off: DCLAB_METRICS_PORT={raw} could not be opened ({type(exc).__name__})", file=sys.stderr, flush=True)
        return None
    threading.Thread(target=server.serve_forever, name="dclab-metrics", daemon=True).start()
    _SERVER[0] = server
    return server.server_address[1]


def started_at() -> float:
    return _STARTED


_STARTED = time.time()
