"""Twin runs from the command line (R&D only).

    python -m dclab_rnd.twin run --sample bank_marketing [--blind] [--yes] [--max-requests 120] [--cap-eur 2] [--max-steps 30]
    python -m dclab_rnd.twin run --project <id> | --draft <id>   [--source-home DIR]
        [--no-design] [--design-checkpoints data,leakage,models] [--no-steer]   (25.3: Claude's expert record, watched)
    python -m dclab_rnd.twin list
    python -m dclab_rnd.twin compare RUN_DIR
    python -m dclab_rnd.twin eval [--run SAMPLE,SAMPLE --yes --cap-eur 1.5] [--no-store]   (25.8: the measurement, stored as evidence)

Without ``--yes`` the plan is printed and nothing is sent to Claude. ``ANTHROPIC_API_KEY`` is the key; ``DCLAB_TWIN_MODEL``
(default claude-sonnet-5-5) the model; ``DCLAB_AGENT_HOME`` the workspace whose ``twin/`` folder holds the runs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import TASK, claude, compare_run, list_runs, run, runs_home


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
    p.add_argument("--cap-eur", type=float, default=1.5, help="the run's euro cap (default 1.5, the owner's); applied when the model has a price")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--max-minutes", type=int, default=40)
    p.add_argument("--full", action="store_true", help="the engine's full profile instead of the quick one")
    p.add_argument("--roles", choices=("all", "judged"), default="all", help="every dclab_core role on for the AI-first lane (default), or the product's own gating")
    p.add_argument("--capture-cells", action="store_true", help="keep tool results verbatim in the transcript, cell examples included (off: the trace's scrub)")
    p.add_argument("--allow-private", action="store_true", help="run on a project marked private (its column names and previews then reach Claude)")
    p.add_argument("--no-design", action="store_true", help="skip Claude's expert record (25.3): the AI-first lane starts from the task alone")
    p.add_argument("--design-checkpoints", default="data,leakage,models", help="the stages after which Claude revises its record (comma-separated; '' for none)")
    p.add_argument("--no-steer", action="store_true", help="keep each revision out of the agent's conversation (the record is only watched)")
    p.add_argument("--yes", action="store_true", help="send requests to Claude (without it, only the plan is printed)")
    sub.add_parser("list", help="the twin runs in the workspace").add_argument("--home", type=Path, default=None)
    sub.add_parser("samples", help="the studied samples a run can start from")
    c = sub.add_parser("compare", help="rebuild compare.json and COMPARE.md of a run")
    c.add_argument("run_dir", type=Path)
    e = sub.add_parser("eval", help="25.8: the blind runs on studied samples, judged against the planted truth, stored as evidence")
    e.add_argument("--home", type=Path, default=None, help="the workspace whose twin/runs are read (default DCLAB_AGENT_HOME or agent_runs/)")
    e.add_argument("--run", default="", help="samples to run blind first (comma-separated), each under --cap-eur")
    e.add_argument("--cap-eur", type=float, default=1.5)
    e.add_argument("--yes", action="store_true", help="send the --run requests to Claude (without it, only their plans are printed)")
    e.add_argument("--no-store", action="store_true", help="print the matrix without storing a result")
    e.add_argument("--supersedes", default="", help="ID=why: the stored result this one corrects (a result is never edited)")
    args = parser.parse_args(argv)

    if args.command == "eval":
        from . import evaluate

        for key in [k.strip() for k in args.run.split(",") if k.strip()]:
            main(["run", "--sample", key, "--blind", "--cap-eur", str(args.cap_eur)] + (["--home", str(args.home)] if args.home else [])
                 + (["--yes"] if args.yes else []))
        measured = evaluate.measure(Path(args.home) if args.home else runs_home(None).parent.parent)
        print(evaluate.report(measured))
        if not args.no_store and measured["runs"]:
            ident, _, why = args.supersedes.partition("=")
            print(evaluate.store(measured, {"id": ident, "why": why} if ident else None))
        return 0

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
    plan = {"seed": seed, "model": model, "endpoint": "Anthropic Messages API (models.anthropic_client)", "key": "set" if claude.api_key() else "ANTHROPIC_API_KEY not set",
            "caps": {"max_requests": args.max_requests, "cap_eur": args.cap_eur, "max_steps": args.max_steps, "max_minutes": args.max_minutes},
            "design": {"on": not args.no_design, "checkpoints": [c for c in args.design_checkpoints.split(",") if c.strip()], "steer": not args.no_steer},
            "thinking": claude.thinking_setting() or "off",
            "blind": args.blind, "workflow_models": args.workflow_models, "roles": args.roles, "capture_cells": args.capture_cells,
            "runs_folder": str(runs_home(args.home))}
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
