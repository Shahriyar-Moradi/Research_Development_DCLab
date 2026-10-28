"""``python -m dclab_rnd.memory upkeep [--apply] [--workspace FOLDER]``: memory upkeep (35.6) on a workspace's memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.memory")
    sub = parser.add_subparsers(dest="step", required=True)
    up = sub.add_parser("upkeep", help="merge near-duplicates, retire unused or harmful lines, list conflicts (nothing deleted)")
    up.add_argument("--apply", action="store_true", help="write the merges and retirements (without it: the plan only)")
    up.add_argument("--workspace", type=Path, help="the workspace folder (default: the settings' agent home)")
    args = parser.parse_args(argv)
    from .. import experience
    from . import open_memory, upkeep

    home = args.workspace
    if home is None:
        from ..settings import Settings

        home = Path(Settings.load().agent_home)
    from ..storage import db

    workspace = db.workspace_for(home) if db.database_url(required=False) else None  # the log the agents write to
    store, log = open_memory(home), experience.open_log(home, workspace)
    todo = upkeep.plan(store, log)
    print(json.dumps({k: len(v) for k, v in todo.items()}))
    for c in todo["conflicts"]:
        print("conflict:", " | ".join(c["texts"]))
    if args.apply:
        print("retired", upkeep.apply(store, todo, log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
