"""Find a session in the workspace's record (19.2; the Admin page and search come with 19.4).

    python -m dclab_rnd.experience sessions [--limit 20]           the latest visits: who, when, the problem, how it ended
    python -m dclab_rnd.experience show VISIT                      one visit's timeline, as it happened
    python -m dclab_rnd.experience show --project PROJECT_ID       everything that happened to a project
    python -m dclab_rnd.experience show --draft DRAFT_ID           a Home conversation, from its first sentence

The workspace is DCLAB_AGENT_HOME's (the server's own), on PostgreSQL when DCLAB_DATABASE_URL is set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import lines, open_log, sessions


def _log():
    from ..settings import Settings
    from ..storage import db

    home = Path(Settings.load().agent_home)
    workspace = db.workspace_for(home) if db.database_url(required=False) else None
    return open_log(home, workspace)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.experience", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("sessions")
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--json", action="store_true")
    show = sub.add_parser("show")
    show.add_argument("visit", nargs="?")
    show.add_argument("--project")
    show.add_argument("--draft")
    show.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    log = _log()
    if args.command == "sessions":
        cards = sessions(log, args.limit)
        if args.json:
            print(json.dumps([json.loads(c.model_dump_json()) for c in cards], indent=2, ensure_ascii=False))
            return 0
        for c in cards:
            cost = ", ".join(f"{v:.4f} {k}" for k, v in c.cost.items()) or "no model spend"
            print(f"{c.visit}  {c.started[:16]} → {c.ended[11:16]}  {c.outcome:<8} {c.events:>4} events  refused {c.refused}  errors {c.errors}  {cost}")
            if c.problem:
                print(f"    {c.problem[:110]}")
        return 0
    if not (args.visit or args.project or args.draft):
        parser.error("name a visit, --project or --draft")
    events = log.read(visit=args.visit, project=args.project, conversation=args.draft)
    if args.json:
        print(json.dumps([json.loads(e.model_dump_json()) for e in events], indent=2, ensure_ascii=False))
    else:
        print("\n".join(lines(events)) or "No events.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
