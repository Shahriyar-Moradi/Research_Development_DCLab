"""Find a session in the workspace's record (19.2, 19.4; the Admin page's Sessions tab reads the same).

    python -m dclab_rnd.experience sessions [--limit 20]           the latest visits: who, when, the problem, how it ended
    python -m dclab_rnd.experience show VISIT                      one visit's timeline, as it happened
    python -m dclab_rnd.experience show --project PROJECT_ID       everything that happened to a project
    python -m dclab_rnd.experience show --draft DRAFT_ID           a Home conversation, from its first sentence
                                                                   (a table's values never; --model-text adds prompts)
    python -m dclab_rnd.experience search --kind refused           events and their visits: --kind (a kind, a family,
                                                                   refused, errors), --project, --draft, --person,
                                                                   --text, --code, --since, --until (until excluded)
    python -m dclab_rnd.experience lineage PROJECT [--path P]      each stage's run, or where a report's number comes from
    python -m dclab_rnd.experience replay PROJECT STAGE            rerun a stage from its record (a scratch copy) and compare

The workspace is DCLAB_AGENT_HOME's (the server's own), on PostgreSQL when DCLAB_DATABASE_URL is set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import contract, finder, lines, open_log, sessions


def _log():
    from ..settings import Settings
    from ..storage import db

    home = Path(Settings.load().agent_home)
    workspace = db.workspace_for(home) if db.database_url(required=False) else None
    return open_log(home, workspace)


def _projects():
    from ..settings import Settings
    from ..storage import open_stores

    return open_stores(Path(Settings.load().agent_home))[0]


def _lineage(args, parser) -> int:
    """19.5: a run's lineage, from the project's stage records; a replay in a scratch copy."""
    from ..studio import lineage

    store = _projects()
    try:
        if args.command == "replay":
            with _workspace_log():
                out = lineage.replay(store, args.project, args.stage)
        else:
            out = lineage.trace(store, args.project, args.path) if args.path else lineage.runs(store, args.project)
    except (KeyError, ValueError) as exc:  # an unknown project, stage or path; a table that changed on disk (DataError)
        parser.error(str(exc.args[0]) if exc.args else "not found")
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    reproduced = all(out.get(k) for k in ("replayed", "same_inputs", "same_value", "same_numbers"))
    return 0 if args.command != "replay" or reproduced else 1


def _workspace_log():
    """The workspace's record installed for this process, so the replay's one run.recorded event is kept."""
    import contextlib

    from . import install

    @contextlib.contextmanager
    def installed():
        install(_log())
        try:
            yield
        finally:
            install(None)
    return installed()


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
    show.add_argument("--model-text", action="store_true", help="also the prompts and replies the workspace kept")
    find = sub.add_parser("search")
    for name in ("kind", "project", "draft", "person", "text", "code", "since", "until"):
        find.add_argument(f"--{name}")
    find.add_argument("--limit", type=int, default=50)
    find.add_argument("--json", action="store_true")
    trace = sub.add_parser("lineage")
    trace.add_argument("project")
    trace.add_argument("--path", help="a report's record path, e.g. final.evidence.holdout_metrics.roc_auc")
    again = sub.add_parser("replay")
    again.add_argument("project")
    again.add_argument("stage")
    args = parser.parse_args(argv)
    if args.command in ("lineage", "replay"):
        return _lineage(args, parser)
    from ..storage import db

    if db.database_url(required=False):  # the record lives in the database: say plainly when it cannot be read yet
        problem = db.reachable()
        if problem:
            print(f"The database did not answer ({problem}). Is PostgreSQL running? `make dev` checks it.", file=sys.stderr)
            return 2
        at, newest = db.current(), db.head()
        if at != newest:
            print(f"The database's schema is at {at or 'nothing'}; the record needs {newest}. Run `make dev` once (it migrates), "
                  "or: .venv/bin/python -m dclab_rnd.storage upgrade", file=sys.stderr)
            return 2
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
    if args.command == "search":
        query = {k: getattr(args, k) for k in ("kind", "project", "draft", "person", "text", "code", "since", "until")}
        if not any(query.values()):
            parser.error("name at least one of --kind, --project, --draft, --person, --text, --code, --since, --until")
        try:
            found = finder.search(log, **query, limit=args.limit)
        except ValueError as exc:  # an unknown kind, a date that is not one
            parser.error(str(exc))
        if args.json:
            print(json.dumps({**found, "events": [finder.view(e) for e in found["events"]],
                              "sessions": [json.loads(c.model_dump_json()) for c in found["sessions"]]}, indent=2, ensure_ascii=False))
            return 0
        more = f" (read the newest {found['scanned']} events only)" if found["capped"] else ""
        print(f"{found['total']} matching events in {found['visits']} sessions{more}")
        for c in found["sessions"]:
            print(f"{c.visit}  {c.started[:16]}  {c.outcome:<8} refused {c.refused}  errors {c.errors}  {(c.problem or '')[:80]}")
        print()
        shown = [contract.redact(e, finder.SHOWN) for e in reversed(found["events"])]  # what the page shows: no values, no prompts
        print("\n".join(lines(shown)) or "No events.")
        return 0
    if not (args.visit or args.project or args.draft):
        parser.error("name a visit, --project or --draft")
    events = log.read(visit=args.visit, project=args.project, conversation=args.draft)
    allowed = finder.SHOWN + (("model_text",) if args.model_text else ())  # a table's values never; prompts on request
    events = [contract.redact(e, allowed) for e in events]
    if args.json:
        print(json.dumps([json.loads(e.model_dump_json()) for e in events], indent=2, ensure_ascii=False))
    else:
        print("\n".join(lines(events)) or "No events.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
