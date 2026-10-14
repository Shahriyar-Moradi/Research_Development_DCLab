"""Twin runs from the command line (R&D only).

    python -m dclab_rnd.twin run --sample bank_marketing [--blind] [--yes] [--max-requests 120] [--cap-eur 2] [--max-steps 30]
    python -m dclab_rnd.twin run --project <id> | --draft <id>   [--source-home DIR]
        [--no-design] [--design-checkpoints data,leakage,models] [--no-steer]   (A6.7: Claude's solution design, watched)
    python -m dclab_rnd.twin list
    python -m dclab_rnd.twin compare RUN_DIR

Without ``--yes`` the plan is printed and nothing is sent to Claude. ``ANTHROPIC_API_KEY`` is the key; ``DCLAB_TWIN_MODEL``
(default claude-sonnet-5-5) the model; ``DCLAB_AGENT_HOME`` the workspace whose ``twin/`` folder holds the runs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import TASK, claude, compare_run, list_runs, run, twin_home


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.twin", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="the current system and Claude on one problem, at the same time")
    seed = p.add_mutually_exclusive_group(required=True)
    seed.add_argument("--sample", help="a studied sample's key (python -m dclab_rnd.twin samples lists them)")
    seed.add_argument("--project", help="a project id in the source workspace")
    seed.add_argument("--draft", help="a draft id in the source workspace: its chat is replayed")
    p.add_argument("--home", type=Path, default=None, help="the workspace folder (default DCLAB_AGENT_HOME or agent_runs/)")
    p.add_argument("--source-home", type=Path, default=None, help="where --project / --draft are read from (default: the workspace)")
    p.add_argument("--task", default=TASK)
    p.add_argument("--model", default=None, help=f"default DCLAB_TWIN_MODEL or {claude.DEFAULT_MODEL}")
    p.add_argument("--blind", action="store_true", help="strip the R&D's known solution from both lanes: each must find the leak itself")
    p.add_argument("--workflow-models", choices=("off", "configured"), default="off", help="the workflow lane with no model (default) or the workspace's configured one")
    p.add_argument("--max-requests", type=int, default=120)
    p.add_argument("--cap-eur", type=float, default=None, help="applied only when the model has a price (DCLAB_PRICES_FILE)")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--max-minutes", type=int, default=40)
    p.add_argument("--full", action="store_true", help="the engine's full profile instead of the quick one")
    p.add_argument("--roles", choices=("all", "judged"), default="all", help="every dclab_core role on for the AI-first lane (default), or the product's own gating")
    p.add_argument("--capture-cells", action="store_true", help="keep tool results verbatim in the transcript, cell examples included (off: the trace's scrub)")
    p.add_argument("--allow-private", action="store_true", help="run on a project marked private (its column names and previews then reach Claude)")
    p.add_argument("--no-design", action="store_true", help="skip Claude's solution design (A6.7): the AI-first lane starts from the task alone")
    p.add_argument("--design-checkpoints", default="data,leakage,models", help="the stages after which Claude revises its design (comma-separated; '' for none)")
    p.add_argument("--no-steer", action="store_true", help="keep each revision out of the agent's conversation (the design is only watched)")
    p.add_argument("--yes", action="store_true", help="send requests to Claude (without it, only the plan is printed)")
    sub.add_parser("list", help="the twin runs in the workspace").add_argument("--home", type=Path, default=None)
    sub.add_parser("samples", help="the studied samples a run can start from")
    c = sub.add_parser("compare", help="rebuild compare.json and COMPARE.md of a run")
    c.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)

    if args.command == "list":
        for r in list_runs(args.home):
            print(json.dumps(r, ensure_ascii=False))
        return 0
    if args.command == "samples":
        from ..studio import data as sd

        for s in sd.sample_catalog():
            print(f"{s['key']:<22} {s.get('name', '')}")
        return 0
    if args.command == "compare":
        comparison = compare_run(args.run_dir)
        print(json.dumps({"solution_differences": comparison["solution_differences"], "stage_differences": comparison["stage_differences"],
                          "holdout": comparison["holdout"]}, indent=2))
        print(f"written: {args.run_dir / 'COMPARE.md'}")
        return 0

    seed = f"sample:{args.sample}" if args.sample else f"project:{args.project}" if args.project else f"draft:{args.draft}"
    model = args.model or claude.model_name()
    plan = {"seed": seed, "model": model, "endpoint": claude.ANTHROPIC_OPENAI_URL, "key": "set" if claude.api_key() else "ANTHROPIC_API_KEY not set",
            "caps": {"max_requests": args.max_requests, "cap_eur": args.cap_eur, "max_steps": args.max_steps, "max_minutes": args.max_minutes},
            "design": {"on": not args.no_design, "checkpoints": [c for c in args.design_checkpoints.split(",") if c.strip()], "steer": not args.no_steer},
            "blind": args.blind, "workflow_models": args.workflow_models, "roles": args.roles, "capture_cells": args.capture_cells,
            "runs_folder": str(twin_home(args.home))}
    print(json.dumps(plan, indent=2))
    if not args.yes:
        print("Nothing was sent. Add --yes to run.")
        return 0
    if os.environ.get("DCLAB_NO_LIVE_MODELS") == "1":
        print("DCLAB_NO_LIVE_MODELS=1: no live model may be used in this environment.")
        return 2
    if not claude.api_key():
        print("ANTHROPIC_API_KEY is not set.")
        return 2
    manifest = run(seed, home=args.home, task=args.task, model=model, max_requests=args.max_requests, cap_eur=args.cap_eur, max_steps=args.max_steps,
                   max_minutes=args.max_minutes, blind=args.blind, workflow_models=args.workflow_models, quick=not args.full, roles=args.roles,
                   capture_cells=args.capture_cells, allow_private=args.allow_private, source_home=args.source_home, design=not args.no_design,
                   design_checkpoints=tuple(c.strip() for c in args.design_checkpoints.split(",") if c.strip()), design_steer=not args.no_steer)
    print(json.dumps({k: manifest[k] for k in ("run_id", "lanes", "differences", "model_usage")}, indent=2, ensure_ascii=False, default=str))
    return 0 if all(v.get("status") == "completed" for v in manifest["lanes"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
