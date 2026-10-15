"""Trace ids for a collector (package 23.11): the W3C ``traceparent`` of a request, and the span ids that join its events.

A request already has an id; a collector (OpenTelemetry, Tempo, a log store) joins what happened in one request by a *trace id*
(32 hex characters) and orders it by *span ids* (16 hex each, with the parent span). This module reads and writes the
``traceparent`` header (``00-<trace>-<parent span>-<flags>``) and derives a trace id from a request id when the caller sent
none, so every request has one and the same request id always gives the same trace id.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass
from typing import Optional

TRACEPARENT = re.compile(r"^([0-9a-f]{2})-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$")
ZERO_TRACE, ZERO_SPAN = "0" * 32, "0" * 16


@dataclass(frozen=True)
class Traceparent:
    trace_id: str
    parent_span: str
    sampled: bool


def new_trace_id() -> str:
    """32 hex characters, never all zero."""
    found = secrets.token_hex(16)
    return found if found != ZERO_TRACE else "1" * 32


def new_span_id() -> str:
    """16 hex characters, never all zero."""
    found = secrets.token_hex(8)
    return found if found != ZERO_SPAN else "1" * 16


def parse_traceparent(header: Optional[str]) -> Optional[Traceparent]:
    """The ids of a ``traceparent`` header, or None when it is absent or not valid W3C (version ``ff``, a zero trace or span id,
    upper-case or short hex). A later version with extra fields is read by its first four, as the specification asks."""
    if not header:
        return None
    text = header.strip()
    found = TRACEPARENT.match(text) or (TRACEPARENT.match("-".join(text.split("-")[:4])) if text.split("-")[0] not in ("00", "ff") else None)
    if not found:
        return None
    version, trace, span, flags = found.groups()
    if version == "ff" or trace == ZERO_TRACE or span == ZERO_SPAN:
        return None
    if version == "00" and len(text.split("-")) != 4:
        return None
    return Traceparent(trace, span, bool(int(flags, 16) & 1))


def format_traceparent(trace_id: str, span_id: str, sampled: bool = True) -> str:
    """``00-<trace>-<span>-01``: what a response carries so the next hop continues the trace."""
    if not (re.fullmatch(r"[0-9a-f]{32}", trace_id) and re.fullmatch(r"[0-9a-f]{16}", span_id)):
        raise ValueError("a trace id is 32 lower-case hex characters and a span id 16")
    return f"00-{trace_id}-{span_id}-{'01' if sampled else '00'}"


def trace_id_for(request_id: str) -> str:
    """The trace id of a request that came with none: the request id itself when it is already 32 hex characters, else the first
    32 hex characters of its SHA-256. The same request id always gives the same trace."""
    text = str(request_id or "").strip().lower()
    if re.fullmatch(r"[0-9a-f]{32}", text) and text != ZERO_TRACE:
        return text
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


__all__ = ["Traceparent", "new_trace_id", "new_span_id", "parse_traceparent", "format_traceparent", "trace_id_for"]
