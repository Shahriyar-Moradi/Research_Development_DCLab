"""Server-sent events for a numbered log (W3): the drafts' stream mechanism (``draft/api.py``, package 10.5) as a helper.

The browser opens an EventSource; the stream sends every entry after the number it was given (``Last-Event-ID`` when it
reconnects, else ``?after=``), each as ``id: <n>``, then waits. On PostgreSQL it sleeps until an instance commits a new
entry (LISTEN/NOTIFY through the server's one listener, ``storage/notify.py``), with a slow poll as the fallback; on files
it polls, as one process writes them. A missed notice only delays the next read: the entries are read from the store.

``draft/api.py`` keeps its own copy of this loop (written before this helper); a later change may move it here.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Callable, Optional

STREAM_SECONDS = 600  # the browser's EventSource reconnects and resumes from Last-Event-ID


class Closed(Exception):
    """Raised by ``read`` to end the stream now (the reader lost access): nothing more is sent."""


def start_from(request: Any, after: int = 0) -> int:
    """Where a stream starts: the browser's Last-Event-ID when it reconnects, else ``after``."""
    raw = request.headers.get("last-event-id") or after or 0
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


async def numbered_events(request: Any, read: Callable[[int], list[dict[str, Any]]], *, start: int, wait: float,
                          listening: Any = None, key: Optional[str] = None, kind: str = "message",
                          number: str = "seq") -> AsyncIterator[str]:
    """The SSE lines of every entry ``read(after)`` returns after ``start``, for ``wait`` seconds (at most STREAM_SECONDS).
    ``listening``/``key``: the listener and the key a commit notifies, to sleep until then instead of polling. ``read``
    raises :class:`Closed` to end the stream (a reader who lost access)."""
    limit = max(0.0, min(float(wait), STREAM_SECONDS))
    loop = asyncio.get_running_loop()
    seq, began, quiet = int(start), loop.time(), loop.time()
    yield "retry: 1500\n\n"
    # subscribed before the first read: an entry committed between a read and the wait still wakes the stream
    woken = listening.subscribe(key) if listening is not None and key else None
    try:
        first = True
        while first or loop.time() - began < limit:
            first = False
            if await request.is_disconnected():
                break
            if woken is not None:
                woken.clear()
            try:
                new = await asyncio.to_thread(read, seq)
            except Closed:
                break
            for entry in new:
                seq = int(entry[number])
                yield f"id: {seq}\nevent: {kind}\ndata: {json.dumps(entry, ensure_ascii=False, default=str)}\n\n"
            if new:
                quiet = loop.time()
                continue
            left = max(0.05, limit - (loop.time() - began))
            if woken is not None and listening.connected.is_set():
                try:
                    await asyncio.wait_for(woken.wait(), min(2.0, left))
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(min(0.4, left))
            if loop.time() - quiet >= 15:
                yield ": keep-alive\n\n"
                quiet = loop.time()
    finally:
        if woken is not None:
            listening.unsubscribe(key, woken)


__all__ = ["Closed", "STREAM_SECONDS", "numbered_events", "start_from"]
