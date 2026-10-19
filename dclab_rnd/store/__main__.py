"""``python -m dclab_rnd.store run|status|catalog|fixes|accept`` (the storing phase, docs/guides/STORING_PHASE_PLAN.md):

- ``run``: one whole run of the conductor (every keeper over ``agent_runs``), its record printed;
- ``status``: what the store holds, kind by kind;
- ``catalog``: ``CATALOG.json``, ``README.md`` and the schemas rewritten;
- ``fixes``: every failing test linked to the run that passed it next (proposed) or not yet (open);
- ``accept FIX --by NAME``: a person's yes to one proposed fix.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> int:
    from . import catalog, conductor, fixes
    from .layout import Layout
    from .store import Store

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.store", description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["run", "status", "catalog", "fixes", "accept"])
    parser.add_argument("fix", nargs="?", help="accept: the fix's id (from the fixes command)")
    parser.add_argument("--root", default=None, help="the store's root (DCLAB_STORE, else training_records/store)")
    parser.add_argument("--by", default=None, help="accept: who accepts it")
    parser.add_argument("--note", default="", help="accept: why")
    args = parser.parse_args(argv)
    layout = Layout(args.root)
    if args.command == "run":
        record = conductor.conduct(None, layout)
        print(json.dumps({k: record[k] for k in ("id", "started", "ended", "failed", "source_errors")} |
                         {"steps": [{k: s[k] for k in ("name", "read", "kept", "unchanged", "error")} | {"errors": len(s["errors"])} for s in record["steps"]]},
                         indent=1))
        return 1 if record["failed"] else 0
    if args.command == "status":
        print(json.dumps({"root": str(layout.root), "kinds": Store(layout).counts()}, indent=1))
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
