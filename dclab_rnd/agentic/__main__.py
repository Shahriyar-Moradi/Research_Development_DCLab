import argparse
import asyncio
import json
import os
from pathlib import Path
import uuid
from dotenv import load_dotenv
from .catalog import ROOT
from .schemas import RunRequest, DEFAULT_GOAL
from .store import Store
from .engine import run_research, worker

def main():
    load_dotenv(ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description="DCLab agentic research studio")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve")
    sub.add_parser("status")
    run = sub.add_parser("run")
    run.add_argument("--project", choices=["general", "hyperack", "telco_churn"], default="general")
    run.add_argument("--datasets", nargs="+", default=["bank_marketing"])
    run.add_argument("--goal", default=DEFAULT_GOAL)
    run.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.6-terra"))
    run.add_argument("--experiments", type=int, default=4)
    run.add_argument("--rows", type=int, default=4000)
    run.add_argument("--repeats", type=int, default=2)
    run.add_argument("--minutes", type=int, default=30)
    resume = sub.add_parser("resume")
    resume.add_argument("run_id")
    archive = sub.add_parser(
        "archive",
        help="materialize every research artifact as clean on-disk folders (keeps SQLite)",
    )
    archive.add_argument("--all", action="store_true", help="archive every run in the studio home")
    archive.add_argument("--run-id", default=None, help="archive one run id")
    replay = sub.add_parser("replay")
    replay.add_argument("recipe", type=Path)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    store = Store(Path(os.environ.get("DCLAB_AGENT_HOME", ROOT / "agent_runs")))
    if args.command == "serve":
        import uvicorn
        from .server import app
        uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("DCLAB_PORT", "8765")))
    elif args.command == "status":
        print(json.dumps([{k: r[k] for k in ("id", "status", "config", "llm_calls", "usage")} for r in store.list()], indent=2))
    elif args.command == "archive":
        from .archive import archive_all, archive_run, write_studio_index

        if args.all:
            results = archive_all(store)
            print(json.dumps({"archived_runs": len(results), "index": str(store.home / "STUDIO_ARCHIVE_INDEX.json")}, indent=2))
        elif args.run_id:
            manifest = archive_run(store, args.run_id, reset_snapshots=True)
            write_studio_index(store)
            print(json.dumps({"run_id": args.run_id, "label": manifest.get("label"), "file_count": manifest.get("file_count")}, indent=2))
        else:
            parser.error("Pass --all or --run-id")
    elif args.command == "replay":
        import hashlib
        recipe = json.loads(args.recipe.read_text())
        from .schemas import Experiment
        Experiment.model_validate(recipe["plan"])
        if args.output.exists(): parser.error("Use a new output directory; evidence cannot be overwritten")
        dataset = recipe["plan"]["dataset"]
        if dataset == "hyperack": source_list = [ROOT / "hyper_ackt-dataset.csv"]
        elif dataset == "telco_churn": source_list = [ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"]
        else:
            directory = ROOT / "external_data" / dataset
            source_list = [directory / "X.parquet", directory / "y.parquet", directory / "meta.json"]
        allowed_sources = {path.name: path for path in source_list}
        for name, expected in recipe["data_hashes"].items():
            if name not in allowed_sources: parser.error("Invalid source name")
            actual = hashlib.sha256(allowed_sources[name].read_bytes()).hexdigest()
            if actual != expected: parser.error("Source fingerprint changed; replay is not equivalent")
        for name, field in [("worker.py", "worker_hash"), ("catalog.py", "catalog_hash")]:
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() != recipe[field]: parser.error("Executor code changed; use the original revision")
        result = asyncio.run(worker({"action": "experiment", "plan": recipe["plan"], "max_rows": recipe["max_rows"], "repeats": recipe["repeats"], "output": str(args.output.resolve())}))
        print(json.dumps(result["metrics"], indent=2))
    else:
        if not os.environ.get("OPENAI_API_KEY"): parser.error("OPENAI_API_KEY must be set; never pass a key as a command-line argument")
        if args.command == "run":
            request = RunRequest(project=args.project, goal=args.goal, datasets=args.datasets, model=args.model, max_experiments=args.experiments, max_rows=args.rows, repeats=args.repeats, max_minutes=args.minutes)
            run_id = uuid.uuid4().hex
            store.create(run_id, request.model_dump())
        else:
            run_id = args.run_id
            if store.get(run_id)["status"] not in ("paused", "interrupted", "failed"): parser.error("Run is not resumable")
        print(f"Research run: {run_id}", flush=True)
        asyncio.run(run_research(store, run_id, resume=args.command == "resume"))
        result = store.get(run_id)
        print(json.dumps(result, indent=2))
        if result["status"] != "completed": raise SystemExit(1)

if __name__ == "__main__": main()
