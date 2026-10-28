"""``python -m dclab_rnd.store run|status|catalog|fixes|accept`` (the storing phase, docs/guides/STORING_PHASE_PLAN.md):

- ``run``: one whole run of the conductor (every keeper over ``agent_runs``, then the shapers), its record printed;
- ``shape``: ``records/`` rebuilt from ``raw/`` (every kind, unchanged or not);
- ``status``: what the store holds, kind by kind;
- ``catalog``: ``CATALOG.json``, ``README.md`` and the schemas rewritten;
- ``fixes``: every failing test linked to the run that passed it next (proposed) or not yet (open);
- ``accept FIX --by NAME``: a person's yes to one proposed fix;
- ``reason --cap-eur X [--yes]``: teacher reasoning for the exchanges with none, within one cap for the run;
- ``service``: the builds with their verdicts, the curves, the catalog, the engagements and the playbook (32.24 to 32.31);
- ``knowledge``: the proposed knowledge items and who accepted each; ``know-accept ITEM --by NAME`` a person's yes to one;
- ``guides``: a guide per problem kind from the accepted items and the judged builds, printed;
- ``curate-shadow --cap-eur X [--yes]``: the model curator in shadow over a sample the rules already placed, compared with them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional


def watch(folder: Optional[Path], parent: int, *, every: float = 5.0, stop: Callable[[int], None] = os._exit) -> threading.Thread:
    """A run started by a server's worker stops when that server is gone (it was given to another parent) or its folder of
    agent runs was removed (a test server's temporary home): 2026-10-25, five such runs outlived their test servers and
    kept writing into the owner's store. ``stop`` ends the process (the store's file locks go with it)."""
    def loop() -> None:
        while True:
            time.sleep(every)
            if os.getppid() != parent or (folder is not None and not folder.exists()):
                stop(3)
                return

    thread = threading.Thread(target=loop, name="store-run-watch", daemon=True)
    thread.start()
    return thread


def main(argv: Optional[list[str]] = None) -> int:
    from . import catalog, conductor, fixes
    from .layout import Layout
    from .store import Store

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.store", description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["run", "shape", "curate", "build", "reason", "status", "catalog", "fixes", "accept", "service", "knowledge", "know-accept", "guides", "curate-shadow", "backfill"])
    parser.add_argument("fix", nargs="?", help="accept: the fix's id (from the fixes command); know-accept: the item's id")
    parser.add_argument("--root", default=None, help="the store's root (DCLAB_STORE, else training_records/store)")
    parser.add_argument("--agent-runs", default=None, help="run: the workspaces' folder (default agent_runs; the worker's job names the server's)")
    parser.add_argument("--by", default=None, help="accept: who accepts it")
    parser.add_argument("--note", default="", help="accept: why")
    parser.add_argument("--cap-eur", type=float, default=None, help="reason, curate-shadow: the most the model may spend in this run")
    parser.add_argument("--limit", type=int, default=100, help="reason, curate-shadow: at most this many records")
    parser.add_argument("--yes", action="store_true", help="reason, curate-shadow: send (without it, the plan only)")
    parser.add_argument("--replay", action="store_true", help="backfill: ask the replayable calls again with thinking on (with --cap-eur; --yes sends)")
    parser.add_argument("--brief", action="store_true", help="status: counts only (how many workspaces, not their names), short enough for a chat")
    args = parser.parse_args(argv)
    layout = Layout(args.root)
    if args.command == "run":
        watch(Path(args.agent_runs) if args.agent_runs else None, os.getppid())
        record = conductor.conduct({"agent_runs": Path(args.agent_runs)} if args.agent_runs else None, layout)
        print(json.dumps({k: record[k] for k in ("id", "started", "ended", "failed", "source_errors")} |
                         {"steps": [{k: s[k] for k in ("name", "read", "kept", "unchanged", "error")} | {"errors": len(s["errors"])} for s in record["steps"]]},
                         indent=1))
        return 1 if record["failed"] else 0
    if args.command == "shape":
        from . import shapers

        print(json.dumps(shapers.shape(layout, force=True), indent=1))
        return 0
    if args.command == "reason":
        from . import reasoning

        if args.cap_eur is None:
            parser.error("reason needs --cap-eur: the teacher never spends without a cap")
        print(json.dumps(reasoning.run(layout, cap_eur=args.cap_eur, limit=args.limit, yes=args.yes), indent=1))
        return 0
    if args.command == "backfill":  # the owner (2026-10-28): thinking and reasoning for the past; what can be re-asked, planned
        from . import backfill

        if args.replay:  # paid: the plan without --yes; a cap is required either way
            if not args.cap_eur or args.cap_eur <= 0:
                print("refused: --replay needs --cap-eur above zero", file=sys.stderr)
                return 2
            print(json.dumps(backfill.replay(layout, cap_usd=args.cap_eur / 0.92, yes=args.yes), indent=1))
            return 0
        print(json.dumps(backfill.plan(layout), indent=1))
        return 0
    if args.command == "curate-shadow":
        from . import curators

        if args.cap_eur is None:
            parser.error("curate-shadow needs --cap-eur: the model curator never spends without a cap")
        print(json.dumps(curators.shadow_sample(layout, n=args.limit, cap_eur=args.cap_eur, yes=args.yes), indent=1, default=str))
        return 0
    if args.command == "curate":
        from . import curators

        print(json.dumps(curators.curate(layout), indent=1))
        return 0
    if args.command == "build":
        from .sets import decisions, experiments, llm, others

        print(json.dumps({"decisions": decisions.build(layout), "llm": llm.build(layout), "others": others.build(layout),
                          "experiments": experiments.build(layout)}, indent=1, default=str))
        return 0
    if args.command in ("service", "knowledge", "know-accept", "guides"):
        from . import service

        if args.command == "service":
            print(json.dumps(service.write(layout), indent=1))
        elif args.command == "knowledge":
            for item in service.knowledge():
                print(f"{item.id:22} {item.status:20} {item.stage:11} {item.accepted_by or 'proposed':10} {item.practice[:90]}")
        elif args.command == "know-accept":
            if not args.fix or not args.by:
                parser.error("know-accept needs the item's id and --by")
            print(service.accept(args.fix, args.by))
        else:
            for kind, text in service.guides(layout).items():
                print(text)
        return 0
    if args.command == "status":
        kinds = Store(layout).counts()
        if args.brief:  # every kind's counts and its last capture; the workspaces' names are the long part
            kinds = {k: {"entries": v["entries"], "sources": v["sources"], "bytes": v["bytes"], "workspaces": len(v["workspaces"]), "last": v["last"]}
                     for k, v in kinds.items()}
            total = {"entries": sum(v["entries"] for v in kinds.values()), "bytes": sum(v["bytes"] for v in kinds.values()),
                     "last": max((v["last"] for v in kinds.values() if v["last"]), default=None)}
            print(json.dumps({"root": str(layout.root), "total": total, "kinds": kinds}, indent=1))
        else:
            print(json.dumps({"root": str(layout.root), "kinds": kinds}, indent=1))
    elif args.command == "catalog":
        doc = catalog.write(layout)
        print(f"wrote {layout.catalog} and {layout.readme}: {sum(v['entries'] for v in doc['kinds'].values())} kept versions")
    elif args.command == "fixes":
        for fix in fixes.link():
            print(f"{fix['status']:8} {fix['id']}  {', '.join(fix['tests']) or '-'}  -> {fix['passing_run'] or 'not passed yet'}")
    else:
        if not args.fix or not args.by:
            parser.error("accept needs the fix's id and --by")
        print(fixes.accept(args.fix, args.by, args.note))
    return 0


if __name__ == "__main__":
    sys.exit(main())
