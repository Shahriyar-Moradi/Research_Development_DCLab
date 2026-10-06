"""The running jobs by key, read from the job table (package 10.3).

The pages asked an in-memory dict of asyncio tasks "is a job running for this project?" (``key in jobs and not
jobs[key].done()``). ``Live`` answers the same question from the table, so it holds whichever process or worker runs
the job, and the pages keep their code. ``tasks`` adds in-process asyncio tasks that are not jobs (the Home agent's turns).
"""

from __future__ import annotations

from typing import Any, Iterator

from .store import Jobs


class Handle:
    def __init__(self, job: dict[str, Any] | None = None, task: Any = None):
        self.job, self.task = job, task

    def done(self) -> bool:
        return self.task.done() if self.task is not None else False


class Live:
    def __init__(self, store: Jobs, kinds: str | tuple[str, ...], tasks: dict[str, Any] | None = None):
        self.store, self.kinds = store, (kinds,) if isinstance(kinds, str) else tuple(kinds)
        self.tasks = tasks if tasks is not None else {}

    def _active(self) -> dict[str, Handle]:
        found: dict[str, Handle] = {k: Handle(task=t) for k, t in list(self.tasks.items()) if not t.done()}
        for kind in self.kinds:
            for job in self.store.list(kind=kind, active=True, limit=1000):
                found[job["key"]] = Handle(job)
        return found

    def get(self, key: str, default: Any = None) -> Handle | Any:
        task = self.tasks.get(key)
        if task is not None and not task.done():
            return Handle(task=task)
        for kind in self.kinds:
            job = self.store.active(kind, key)
            if job is not None:
                return Handle(job)
        return default

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self.get(key) is not None

    def __getitem__(self, key: str) -> Handle:
        found = self.get(key)
        if found is None:
            raise KeyError(key)
        return found

    def items(self) -> list[tuple[str, Handle]]:
        return list(self._active().items())

    def keys(self) -> list[str]:
        return list(self._active())

    def values(self) -> list[Handle]:
        return list(self._active().values())

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __len__(self) -> int:
        return len(self._active())
