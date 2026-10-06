"""Wake event streams across app instances (package 10.5): PostgreSQL LISTEN/NOTIFY.

``PgDrafts.emit`` sends ``pg_notify('dclab_draft_event', <draft id>)`` in the transaction that writes the event, so
the notice arrives when the row is committed, on every instance connected to the database. One ``Listener`` per
server process holds a connection that LISTENs, and wakes the streams waiting on that draft. A stream still reads
its events from the table (the notice carries only the draft id) and replays from Last-Event-ID, so a missed notice
only delays it to its next poll; nothing is lost.
"""

from __future__ import annotations

import asyncio
import threading

CHANNEL = "dclab_draft_event"


def dsn(url: str) -> str:
    """The libpq connection string of a SQLAlchemy URL (``postgresql+psycopg://…`` → ``postgresql://…``)."""
    from sqlalchemy.engine import make_url

    return make_url(url).set(drivername="postgresql").render_as_string(hide_password=False)


class Listener:
    """A daemon thread that LISTENs on ``CHANNEL`` and wakes asyncio waiters by draft id. Reconnects after an error."""

    def __init__(self, url: str, channel: str = CHANNEL):
        self.url, self.channel = url, channel
        self._waiters: dict[str, set[tuple[asyncio.AbstractEventLoop, asyncio.Event]]] = {}
        self._guard = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.connected = threading.Event()  # set while LISTENing (tests wait for it)

    def start(self) -> "Listener":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="dclab-listen", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        import psycopg

        while not self._stop.is_set():
            try:
                with psycopg.connect(dsn(self.url), autocommit=True) as conn:
                    conn.execute(f"LISTEN {self.channel}")
                    self.connected.set()
                    while not self._stop.is_set():
                        for notice in conn.notifies(timeout=1.0):
                            self._wake(notice.payload)
            except Exception as exc:  # noqa: BLE001 — the database went away: streams fall back to polling until it is back
                if self.connected.is_set() or not getattr(self, "_reported", False):
                    import sys

                    print(f"event listener: {type(exc).__name__}; streams poll until it reconnects", file=sys.stderr, flush=True)
                    self._reported = True
                self.connected.clear()
                self._stop.wait(2.0)
        self.connected.clear()

    def subscribe(self, key: str) -> asyncio.Event:
        """An event set each time ``key`` gets a notice; the caller clears it before reading and waits on it after."""
        entry = (asyncio.get_running_loop(), asyncio.Event())
        with self._guard:
            self._waiters.setdefault(key, set()).add(entry)
        return entry[1]

    def unsubscribe(self, key: str, event: asyncio.Event) -> None:
        with self._guard:
            waiting = self._waiters.get(key)
            if waiting is not None:
                waiting -= {e for e in waiting if e[1] is event}
                if not waiting:
                    self._waiters.pop(key, None)

    def _wake(self, key: str) -> None:
        with self._guard:
            waiters = list(self._waiters.get(key, ()))
        for loop, event in waiters:
            try:
                loop.call_soon_threadsafe(event.set)
            except RuntimeError:  # the waiter's loop has closed
                pass

    async def wait(self, key: str, timeout: float) -> bool:
        """Wait until an event for ``key`` is committed (True) or ``timeout`` seconds pass (False)."""
        entry = (asyncio.get_running_loop(), asyncio.Event())
        with self._guard:
            self._waiters.setdefault(key, set()).add(entry)
        try:
            await asyncio.wait_for(entry[1].wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False
        finally:
            with self._guard:
                waiting = self._waiters.get(key)
                if waiting is not None:
                    waiting.discard(entry)
                    if not waiting:
                        self._waiters.pop(key, None)


_LISTENERS: dict[str, Listener] = {}
_GUARD = threading.Lock()


def listener(url: str | None) -> Listener | None:
    """The process's listener for this database (started on first use), or None without one."""
    if not url:
        return None
    with _GUARD:
        found = _LISTENERS.get(url)
        if found is None:
            found = _LISTENERS[url] = Listener(url).start()
        return found

