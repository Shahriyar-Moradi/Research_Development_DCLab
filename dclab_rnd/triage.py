"""Delivering many projects in the studio (package 30.24): each project's last use read from the workspace's own records, and the ones
nobody used for a period flagged to retire (``dclab_core.triage.retire``); ``python -m dclab_rnd.triage [--days 90]`` lists them.

A project's use is its newest event in the experience log (a visit, a stage, a chat, an outcome observed: 23.5's ``outcome.observed``
is an event) or its own last change, whichever is later. Flagging decides nothing: a person retires a project.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import Any, Optional


def last_use(services: Any, log: Any = None) -> tuple[dict[str, Optional[str]], dict[str, Optional[str]]]:
    """Each project's last use and when it was made (ISO times)."""
    from . import experience

    log = log if log is not None else experience.current_log()
    used, created = {}, {}
    for p in services.projects.list():
        pid = p["id"]
        times = [t for t in (p.get("updated"),) if t]
        if log is not None:
            try:
                events = log.read(project=pid, limit=1)
            except Exception:  # noqa: BLE001 - a log that cannot be read gives no use, never a failure
                events = []
            times += [str(e.at) for e in events if getattr(e, "at", None)]
        used[pid] = max(times, key=lambda t: datetime.fromisoformat(t.replace("Z", "+00:00"))) if times else None
        created[pid] = p.get("created")
    return used, created


def unused(services: Any, *, days: int = 90, now: Optional[str] = None, log: Any = None) -> list[dict[str, Any]]:
    """The projects to retire: no use for ``days``, or never used and older than that."""
    from dclab_core import triage

    used, created = last_use(services, log)
    return triage.retire(used, now=now or datetime.now(timezone.utc).isoformat(timespec="seconds"), days=days, created=created)


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    from pathlib import Path

    from .agentic.services import Services
    from .settings import Settings

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.triage", description=__doc__.split("\n")[0])
    parser.add_argument("--home", default=None, help="the workspace folder (DCLAB_AGENT_HOME when not given)")
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args(argv)
    services = Services(Settings.load(Path(args.home) if args.home else None))
    print(json.dumps(unused(services, days=args.days, log=services.experience), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
