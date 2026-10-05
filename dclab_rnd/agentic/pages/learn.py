"""Policy model and Domain packs: what the future small model would learn from, and which packs are in use.

    GET /api/learn/policy   the SFT v3 corpus (manifest, line counts, one real record per task), the critic
                            gate recomputed over the campaigns, training runs found on disk, the examples the
                            notebook's projects yield today (``studio.sft.examples_from_project``), the graph's
                            transition logs counted as trajectories, and the curriculum's readiness
    GET /api/learn/packs    how many drafts and projects use each domain pack (``draft.pack.PACKS``)

Everything is read from files; nothing is trained, labelled or written. The corpus under ``out_v3/`` is a
generated file: it is only read here (CLAUDE.md rule 4).

Readiness of the curriculum is computed, never declared:

1. Declarative SFT has data when the v3 manifest exists and counts at least one example.
2. Trajectory SFT has data once ``TRAJECTORY_TARGET`` (50) trajectories exist. A trajectory is one project's
   transition log (state -> move -> verdict, ``transitions.jsonl``) with at least ``MIN_MOVES`` (3) moves;
   shorter logs count as moves but not as trajectories. Why 50: the v3 corpus validates on whole held-out
   datasets (3 of its datasets, roughly one in five). Holding out whole projects the same way, 50 logs leave
   about 10 unseen projects to measure transfer on; with fewer, the validation score is one or two projects'
   luck rather than a measurement.
3. Rejection sampling is planned: it needs a stage-2 model to sample moves from.
4. Reinforcement learning is research: it needs rewards from the validator and the judgment benchmark.

There is no store for expert review labels, so the review queue is always empty here; the route says so.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from ... import critic_gate
from ...draft.pack import PACKS
from ...studio.sft import examples_from_project
from ...studio.store import STAGE_KEYS

ROOT = Path(__file__).resolve().parents[3]
SFT_DIR = ROOT / "research" / "llm-fine-tuning" / "experiments" / "sft"
CORPUS_DIR = SFT_DIR / "out_v3"
# train_lora.py writes to sft/runs/<name> by default; its docstring also suggests sft/runs/ at the repository root.
RUN_DIRS = (SFT_DIR / "runs", ROOT / "sft" / "runs")

TRAJECTORY_TARGET = 50
MIN_MOVES = 3
EXAMPLE_CHARS = 4000  # per message; the longest v3 message is about 10 000 characters
TASK_ORDER = ("leakage_judgment", "explain_experiment", "critique_claim", "apply_selection_rule",
              "grounded_qa", "rule_reasoning", "workflow_steps")
STATUSES = ("allowed", "blocked", "needs_approval")


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).replace(microsecond=0).isoformat()


def _signature(paths: list[Path]) -> tuple:
    out = []
    for p in paths:
        try:
            st = p.stat()
            out.append((str(p), st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((str(p), None, None))
    return tuple(out)


# ---------------------------------------------------------------------------------------------- corpus
def _cut(text: str) -> dict[str, Any]:
    return {"content": text[:EXAMPLE_CHARS], "length": len(text), "truncated": len(text) > EXAMPLE_CHARS}


@lru_cache(maxsize=4)
def _corpus_cached(directory: str, _sig: tuple) -> dict[str, Any]:
    base = Path(directory)
    manifest_path = base / "MANIFEST.json"
    if not manifest_path.is_file():
        return {"corpus": {"available": False, "path": _rel(base),
                           "reason": "No MANIFEST.json: build the corpus with make knowledge."}, "examples": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lines: dict[str, int] = {}
    examples: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        path = base / f"{split}.chat.jsonl"
        if not path.is_file():
            continue
        n = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            n += 1
            row = json.loads(line)
            task = (row.get("metadata") or {}).get("task")
            if task and task not in examples:
                examples[task] = {"task": task, "split": split, "line": n, "file": _rel(path), "metadata": row.get("metadata") or {},
                                  "messages": [{"role": m.get("role"), **_cut(m.get("content") or "")} for m in row.get("messages", [])]}
        lines[f"{split}.chat.jsonl"] = n
    policy = manifest.get("validation_policy") or ""
    held_out = []
    if "held out entirely:" in policy:
        held_out = [d.strip() for d in policy.split("held out entirely:", 1)[1].split(";", 1)[0].split(",") if d.strip()]
    corpus = {
        "available": True, "path": _rel(base), "built": _iso(manifest_path.stat().st_mtime),
        "schema_version": manifest.get("schema_version"),
        "total": manifest.get("total", 0), "train": manifest.get("train", 0), "val": manifest.get("val", 0),
        "by_task": manifest.get("by_task") or {}, "by_split_and_task": manifest.get("by_split_and_task") or {},
        "by_campaign": manifest.get("by_campaign") or {}, "quality_gates": manifest.get("quality_gates") or {},
        "validation_policy": policy, "held_out_datasets": held_out, "formats": manifest.get("formats") or {},
        "lines": lines,
        "lines_match_manifest": lines.get("train.chat.jsonl") == manifest.get("train") and lines.get("val.chat.jsonl") == manifest.get("val"),
    }
    order = [t for t in TASK_ORDER if t in examples] + sorted(t for t in examples if t not in TASK_ORDER)
    return {"corpus": corpus, "examples": [examples[t] for t in order]}


def corpus(directory: Path = CORPUS_DIR) -> dict[str, Any]:
    """The v3 manifest, the real line counts of the chat files and the first record of each task."""
    files = [directory / "MANIFEST.json", directory / "train.chat.jsonl", directory / "val.chat.jsonl"]
    return _corpus_cached(str(directory), _signature(files))


# ---------------------------------------------------------------------------------------------- critic gate
@lru_cache(maxsize=2)
def _gate_cached(root: str, _sig: tuple) -> dict[str, Any]:
    return critic_gate.summary(critic_gate.gate_campaigns(Path(root)))


def gate(root: Path = ROOT) -> dict[str, Any]:
    """critic_gate.summary over every campaign result, recomputed when a result file changes."""
    files = sorted(root.glob("evidence/campaigns/*/results/EXP-*.json"))
    return _gate_cached(str(root), _signature(files))


# ---------------------------------------------------------------------------------------------- training runs
def training_runs(dirs: tuple[Path, ...] = RUN_DIRS) -> dict[str, Any]:
    """LoRA runs on disk: a folder under a runs directory with an adapter or a trainer state."""
    runs = []
    for base in dirs:
        if not base.is_dir():
            continue
        for run in sorted(p for p in base.iterdir() if p.is_dir()):
            adapter, state = run / "adapter_config.json", run / "trainer_state.json"
            checkpoints = sorted(run.glob("checkpoint-*/trainer_state.json"))
            if not state.is_file() and checkpoints:
                state = checkpoints[-1]
            if not adapter.is_file() and not state.is_file():
                continue
            item: dict[str, Any] = {"name": run.name, "path": _rel(run), "adapter": adapter.is_file(),
                                    "updated": _iso(max(p.stat().st_mtime for p in (adapter, state) if p.is_file()))}
            try:
                if adapter.is_file():
                    cfg = json.loads(adapter.read_text(encoding="utf-8"))
                    item["base_model"] = cfg.get("base_model_name_or_path")
                    item["rank"] = cfg.get("r")
                if state.is_file():
                    st = json.loads(state.read_text(encoding="utf-8"))
                    evals = [h for h in st.get("log_history", []) if "eval_loss" in h]
                    item["epochs"] = st.get("epoch")
                    item["steps"] = st.get("global_step")
                    if evals:
                        item["eval_loss"] = evals[-1]["eval_loss"]
            except (OSError, json.JSONDecodeError, TypeError, KeyError):
                item["unreadable"] = True
            runs.append(item)
    return {"runs": runs, "searched": [_rel(d) for d in dirs],
            "scripts": {name: (SFT_DIR / name).is_file() for name in ("train_lora.py", "eval_sft.py", "build_sft_dataset_v3.py")}}


# ---------------------------------------------------------------------------------------------- the notebook's own data
def project_examples(projects) -> dict[str, Any]:
    """How many SFT examples each project yields today (the same function the SFT export uses)."""
    items, by_stage, total = [], Counter(), 0
    for p in projects.list():
        item = {"id": p.get("id"), "name": p.get("name"), "examples": 0, "stages": []}
        if not p.get("solution"):
            item["why"] = "no saved solution"
        else:
            try:
                rows = examples_from_project(p, projects.records(p["id"]))
            except (KeyError, TypeError, ValueError, IndexError) as exc:  # a stage record from an older format
                rows, item["why"] = [], f"could not read its stage records ({type(exc).__name__})"
            item["examples"] = len(rows)
            item["stages"] = [r["metadata"]["stage"] for r in rows]
            by_stage.update(item["stages"])
            if not rows:
                item.setdefault("why", "no completed stage yet")
        total += item["examples"]
        items.append(item)
    return {"projects": len(items), "with_examples": sum(1 for i in items if i["examples"]), "examples": total,
            "by_stage": {s: by_stage[s] for s in (*STAGE_KEYS, "solution") if by_stage[s]}, "items": items}


def trajectories(projects) -> dict[str, Any]:
    """The graph's transition logs: moves per actor and status, and one real record to show."""
    by_status, by_move = Counter(), Counter()
    by_actor: dict[str, Counter] = {}
    logs, full, moves = 0, 0, 0
    pick, pick_key = None, None
    for p in projects.list():
        rows = projects.transitions(p["id"], 10**6)
        if not rows:
            continue
        logs += 1
        full += 1 if len(rows) >= MIN_MOVES else 0
        moves += len(rows)
        for i, row in enumerate(rows, start=1):
            status, actor = row.get("status") or "unknown", row.get("actor") or "unknown"
            by_status[status] += 1
            by_move[row.get("move") or "unknown"] += 1
            by_actor.setdefault(actor, Counter())[status] += 1
            # Show the latest blocked move if any (the boundary the model must learn), else the latest move.
            key = (status == "blocked", row.get("at") or "", i)
            if pick_key is None or key > pick_key:
                pick_key, pick = key, {"project": {"id": p["id"], "name": p.get("name")}, "index": i, "of": len(rows), "record": row}
    actors = {a: {"moves": sum(c.values()), **{s: c[s] for s in STATUSES}} for a, c in sorted(by_actor.items())}
    return {"projects_with_log": logs, "trajectories": full, "min_moves": MIN_MOVES, "target": TRAJECTORY_TARGET,
            "moves": moves, "by_status": {s: by_status[s] for s in by_status}, "by_actor": actors,
            "by_move": dict(by_move.most_common()), "sample": pick}


def curriculum(corp: dict[str, Any], runs: dict[str, Any], traj: dict[str, Any]) -> list[dict[str, Any]]:
    total = corp.get("total", 0) if corp.get("available") else 0
    n_runs = len(runs["runs"])
    sft_ready = total > 0
    traj_ready = traj["trajectories"] >= TRAJECTORY_TARGET
    return [
        {"stage": 1, "key": "sft", "name": "Declarative SFT", "ready": sft_ready,
         "status": "data ready" if sft_ready else "no corpus", "cls": "ok" if sft_ready else "warn",
         "detail": (f"Rules, experiments, critiques and leakage judgments as question-answer pairs. {total} examples; "
                    + (f"{n_runs} training run{'s' if n_runs != 1 else ''} found." if n_runs else "no training run yet."))
         if sft_ready else "The v3 corpus is missing; build it with make knowledge."},
        {"stage": 2, "key": "traj", "name": "Trajectory SFT", "ready": traj_ready,
         "status": "data ready" if traj_ready else "needs the graph log", "cls": "ok" if traj_ready else "outline",
         "detail": f"State → move → verdict from every project. {traj['trajectories']} of {TRAJECTORY_TARGET} project logs with "
                   f"{MIN_MOVES}+ moves; {traj['moves']} moves logged, {traj['by_status'].get('blocked', 0)} blocked."},
        {"stage": 3, "key": "rejection", "name": "Rejection sampling", "ready": False, "status": "planned", "cls": "outline",
         "detail": "Sample several moves per state; keep the ones the validator accepts and the benchmark scores well. Needs a stage-2 model to sample from."},
        {"stage": 4, "key": "rl", "name": "Reinforcement learning", "ready": False, "status": "research", "cls": "",
         "detail": "Rewards from the validator and the benchmark: valid moves, leaks caught, correct citations, no unsafe action."},
    ]


def policy(ctx) -> dict[str, Any]:
    c = corpus()
    runs = training_runs()
    traj = trajectories(ctx.projects)
    return {
        "corpus": c["corpus"], "examples": c["examples"], "critic_gate": gate(), "training": runs,
        "projects": project_examples(ctx.projects), "trajectories": traj,
        "curriculum": curriculum(c["corpus"], runs, traj),
        "review": {"queue": [], "store": None, "awaiting_person": traj["by_status"].get("needs_approval", 0),
                   "note": "There is no store for review labels yet."},
    }


def pack_usage(ctx) -> dict[str, Any]:
    """Drafts and projects per pack: ``draft.pack.key`` and ``project.draft.pack.key``."""
    usage = {p["key"]: {"drafts": 0, "open_drafts": 0, "projects": 0, "project_list": []} for p in PACKS}
    drafts = ctx.drafts.list(10**6)
    no_pack_drafts = 0
    for d in drafts:
        key = (d.get("pack") or {}).get("key")
        if key not in usage:
            no_pack_drafts += 1
            continue
        usage[key]["drafts"] += 1
        usage[key]["open_drafts"] += 1 if d.get("status") == "open" else 0
    projects = ctx.projects.list()
    no_pack_projects = 0
    for p in projects:
        key = ((p.get("draft") or {}).get("pack") or {}).get("key")
        if key not in usage:
            no_pack_projects += 1
            continue
        usage[key]["projects"] += 1
        usage[key]["project_list"].append({"id": p.get("id"), "name": p.get("name")})
    return {"packs": usage, "drafts": len(drafts), "projects": len(projects),
            "drafts_without_pack": no_pack_drafts, "projects_without_pack": no_pack_projects}


def register(app, ctx) -> None:
    @app.get("/api/learn/policy")
    async def learn_policy():
        return await asyncio.to_thread(policy, ctx)

    @app.get("/api/learn/packs")
    async def learn_packs():
        return await asyncio.to_thread(pack_usage, ctx)
