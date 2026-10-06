"""The corpus v4 recipe (package A6.2): the declarative v3 examples plus trajectory examples, held out by whole project.

A *trajectory* is one sequence of decisions on one opted-in project (``studio/trajectories.py``): the project's move
log, or one agent run on it (an intern session, the Home draft), counted when it holds at least ``MIN_DECISIONS``.
The project's own ids are masked (``<project>``, ``<run>``), and only distinct trajectories count: twenty identical
runs of the standard plan are one trajectory, not twenty. Nothing is generated until there are ``TARGET`` distinct
trajectories from at least ``MIN_PROJECTS`` different projects: a
policy model trained on a handful of projects learns those projects, and a validation split of one or two projects
measures nothing. Until then ``python -m dclab_rnd.studio.corpus_v4`` prints how many exist and writes nothing.

When the threshold is met, each trajectory gives two kinds of example:

- ``next_move``: the state, the moves or tools on offer and the last decisions; the answer is the move the run made
  (only moves the validator allowed and steps that worked: a policy model must not learn a refused move as a target);
- ``judge_move``: a proposed move in a state; the answer is the validator's verdict, the checks that failed and the
  rules it cited (refused moves are as useful here as allowed ones).

The split holds out whole projects: a project is in validation when its own id hashes there (about one in five, so a
project chosen by its hash never changes split as the workspace grows; when no project hashes there, the lowest hash
is held out as a recorded fallback that can move later) or when it is built on a dataset v3 holds out (heart_disease,
spambase, ecommerce_clothing_reviews; recognised by the sample key, the file name or the column set), so v4's training
never sees what v3 validates on. A project built on a judgment-suite table (A4.1, the benchmark A6.3 evaluates on) is
left out. The v3 examples keep their own split. The gates are v3's, applied within each split after the split: exact
duplicate prompts are removed, examples with an absolute path or a hash are rejected, and a training prompt that equals
or nearly equals a validation prompt (v4's or v3's; word 3-shingles, Jaccard at least ``NEAR``) is dropped from training.

The output goes to ``research/llm-fine-tuning/experiments/sft/out_v4/``, which is not under version control: unlike
v3, which is built from the repository's own evidence, v4 holds a workspace's projects.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SFT = ROOT / "research" / "llm-fine-tuning" / "experiments" / "sft"
V3 = SFT / "out_v3"
OUT = SFT / "out_v4"
TARGET = 50  # trajectories
MIN_PROJECTS = 10
MIN_DECISIONS = 3
VAL_EVERY = 5  # about one project in five is held out
NEAR = 0.9  # Jaccard of word 3-shingles above which two prompts count as the same prompt
V3_HELD_OUT = ("heart_disease", "spambase", "ecommerce_clothing_reviews")  # build_sft_dataset_v3.VAL_DATASETS
SYSTEM = ("You are DCLab's policy model. You choose the next move of a machine-learning project from its state, the moves the "
          "workflow graph allows and what happened so far, or you predict how the graph's validator judges a proposed move. "
          "Answer with JSON only. A move the validator would refuse is never the answer to which move comes next.")
_ABS_PATH = re.compile(r"(/Users/|/home/|C:\\\\|/tmp/)")  # build_sft_dataset_v3's gates, as they are
_SHA = re.compile(r"\b[0-9a-f]{40,64}\b")


# ---------------------------------------------------------------------- what a project is


@functools.lru_cache(maxsize=1)
def _held_out_columns() -> dict[frozenset[str], str]:
    """The column sets of the datasets v3 holds out, to recognise a project built on one whatever path built it."""
    from . import data as studio_data

    out = {}
    for key in V3_HELD_OUT:
        try:
            out[frozenset(map(str, studio_data.load_sample(key)[0].columns))] = key
        except Exception:  # noqa: BLE001 — a sample that is not on this machine cannot be matched by its columns
            continue
    return out


@functools.lru_cache(maxsize=1)
def _benchmark_columns() -> dict[frozenset[str], str]:
    """The column sets of the judgment suite's seeded tables (A4.1): the benchmark A6.3 evaluates on."""
    try:
        from ..agent_eval.cases import CASES
    except Exception:  # noqa: BLE001
        return {}
    out = {}
    for case in CASES:
        try:
            out[frozenset(map(str, case.table().columns))] = case.id
        except Exception:  # noqa: BLE001 — one case that cannot build its table must not stop the count
            continue
    return out


def project_meta(project: dict[str, Any]) -> dict[str, Any]:
    """Whether a project is built on a dataset v3 holds out (always validation) or on a judgment-suite table (never in v4)."""
    data = project.get("data") or {}
    columns = frozenset(map(str, data.get("columns") or []))
    stem = Path(str(data.get("filename") or "")).stem
    sample = (project.get("suggestion") or {}).get("sample")
    held = sample if sample in V3_HELD_OUT else stem if stem in V3_HELD_OUT else _held_out_columns().get(columns)
    return {"held_out_dataset": held, "benchmark_case": _benchmark_columns().get(columns) if columns else None}


# ---------------------------------------------------------------------- counting


def _mask(value: Any, ids: dict[str, str]) -> Any:
    """The project's own ids replaced by placeholders: two projects that make the same decisions make the same example."""
    if isinstance(value, str):
        return ids.get(value, value)
    if isinstance(value, list):
        return [_mask(v, ids) for v in value]
    if isinstance(value, dict):
        return {k: _mask(v, ids) for k, v in value.items()}
    return value


def trajectories_of(records: list[dict[str, Any]], meta: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Records grouped into trajectories (a project's move log, or one agent run on it), in a stable order, with the
    project's ids masked; only those with at least MIN_DECISIONS records, and none from a judgment-suite table."""
    meta = meta or {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        if (meta.get(r["project_id"]) or {}).get("benchmark_case"):
            continue  # the benchmark A6.3 measures on stays out of the training data
        key = (r["project_id"], "moves" if r["kind"] == "move" else f"run:{r.get('run')}")
        groups.setdefault(key, []).append(r)
    out = []
    for (pid, source), rs in sorted(groups.items()):
        if len(rs) < MIN_DECISIONS:
            continue
        ids = {pid: "<project>", **{r["run"]: "<run>" for r in rs if r.get("run")}}
        masked = [{**r, "arguments": _mask(r.get("arguments") or {}, ids)} for r in sorted(rs, key=lambda r: r.get("n") or 0)]
        signature = json.dumps([(r["kind"], r.get("state"), _decision(r), r.get("verdict")) for r in masked], sort_keys=True)
        out.append({"project_id": pid, "source": source, "records": masked, "signature": signature})
    return out


def status(trajs: list[dict[str, Any]], meta: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The count against the threshold. Only distinct trajectories count: twenty identical runs of the same plan teach
    one thing, not twenty."""
    distinct: dict[str, dict[str, Any]] = {}
    for t in trajs:
        distinct.setdefault(t["signature"], t)
    projects = sorted({t["project_id"] for t in distinct.values()})
    ready = len(distinct) >= TARGET and len(projects) >= MIN_PROJECTS
    return {"trajectories": len(distinct), "recorded": len(trajs), "projects": len(projects), "target": TARGET, "min_projects": MIN_PROJECTS,
            "min_decisions": MIN_DECISIONS, "ready": ready, "held_out_projects": len(held_out_projects(projects, meta or {}))}


def workspace_records(projects: Any, sessions: Any, traces: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """The opted-in projects' records, and what each project is built on (held-out dataset, benchmark table)."""
    from . import trajectories

    records, meta = [], {}
    for project in sorted(projects.list(), key=lambda p: p["id"]):
        if not trajectories.opted_in(project):
            continue
        records += trajectories.records(project, projects, sessions, traces)
        meta[project["id"]] = project_meta(project)
    return records, meta


def count(workspace: Path) -> dict[str, Any]:
    from ..agents.traces import open_traces
    from ..storage import open_stores

    projects, _, sessions = open_stores(Path(workspace))
    records, meta = workspace_records(projects, sessions, open_traces(Path(workspace)))
    return status(trajectories_of(records, meta), meta)


# ---------------------------------------------------------------------- examples


def _decision(r: dict[str, Any]) -> dict[str, Any]:
    if r["kind"] == "move":
        return {"move": r["move"], **({"arguments": r["arguments"]} if r.get("arguments") else {})}
    return {"tool": r["tool"], **({"arguments": r["arguments"]} if r.get("arguments") else {})}


def _history(records: list[dict[str, Any]], i: int, n: int = 3) -> str:
    done = records[max(0, i - n):i]
    return "; ".join(f"{json.dumps(_decision(r), sort_keys=True)} → {r['verdict']}" for r in done) or "nothing yet"


def examples_of(traj: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    rs = traj["records"]
    for i, r in enumerate(rs):
        who = r.get("actor") or r.get("agent")
        base = {"project_id": traj["project_id"], "source": traj["source"], "kind": r["kind"], "n": r.get("n")}
        context = (f"State: {r.get('state')}" + (f" (node {r['node']})" if r.get("node") else "") + f"\nActor: {who}\n"
                   f"On offer: {', '.join(r.get('allowed') or []) or 'not recorded'}\nLast decisions: {_history(rs, i)}")
        if r.get("verdict") in ("allowed", "ok"):
            out.append({"task": "next_move", "user": context + "\nWhich move comes next?",
                        "assistant": json.dumps(_decision(r), sort_keys=True), "metadata": {**base, "task": "next_move"}})
        if r["kind"] == "move":
            out.append({"task": "judge_move", "user": context + f"\nProposed: {json.dumps(_decision(r), sort_keys=True)}\nHow does the validator judge it?",
                        "assistant": json.dumps({"verdict": r["verdict"], "failed_checks": r.get("failed_checks") or [], "cited": r.get("cited") or []},
                                                sort_keys=True),
                        "metadata": {**base, "task": "judge_move"}})
    return out


# ---------------------------------------------------------------------- split and gates


def _bucket(project_id: str) -> int:
    return int(hashlib.sha256(project_id.encode()).hexdigest()[:8], 16)


def held_out_projects(project_ids: list[str], meta: dict[str, dict[str, Any]]) -> set[str]:
    """A project is in validation by its own hash (about one in five; such a project never changes split as the
    workspace grows) or because it is built on a dataset v3 holds out. When none is chosen that way, the lowest hash is
    held out as a fallback (``fallback_held_out``): that one can return to training once a project hashes to validation."""
    ids = sorted(set(project_ids))
    held = {p for p in ids if _bucket(p) % VAL_EVERY == 0 or (meta.get(p) or {}).get("held_out_dataset")}
    if not held and len(ids) > 1:
        held = {fallback_held_out(ids)}
    return held


def fallback_held_out(project_ids: list[str]) -> str | None:
    """The project held out only because no project hashes to validation (None when one does)."""
    ids = sorted(set(project_ids))
    if len(ids) < 2 or any(_bucket(p) % VAL_EVERY == 0 for p in ids):
        return None
    return min(ids, key=_bucket)


def _shingles(text: str) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + 3]) for i in range(max(1, len(words) - 2))}


def near_duplicate(a: set[str], b: set[str]) -> bool:
    return bool(a and b) and len(a & b) / len(a | b) >= NEAR


def _key(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()  # v3's duplicate key: the normalised user turn


def _clean(ex: dict[str, Any]) -> bool:
    blob = ex["user"] + "\n" + ex["assistant"]
    return not _ABS_PATH.search(blob) and not _SHA.search(blob)


def v3_examples(folder: Path = V3) -> dict[str, list[dict[str, Any]]]:
    """The v3 chat examples with the split v3 gave them."""
    out = {"train": [], "val": []}
    for split in out:
        path = folder / f"{split}.chat.jsonl"
        if path.is_file():
            out[split] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return out


def assemble(trajs: list[dict[str, Any]], meta: dict[str, dict[str, Any]], v3: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, list], dict[str, Any]]:
    held = held_out_projects([t["project_id"] for t in trajs], meta)
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for t in trajs:  # whole projects first, then the gates within each split: the result does not depend on list order
        by_split["val" if t["project_id"] in held else "train"] += examples_of(t)
    gates = Counter()
    kept: dict[str, list[dict[str, Any]]] = {}
    for split in ("val", "train"):
        seen, items = set(), []
        for ex in by_split[split]:
            k = _key(ex["user"])
            if k in seen:
                gates["duplicates_removed"] += 1
                continue
            seen.add(k)
            if not _clean(ex):
                gates["rejected_by_path_or_hash"] += 1
                continue
            items.append(ex)
        kept[split] = items
    v3_val_users = [next((m["content"] for m in e["messages"] if m["role"] == "user"), "") for e in v3["val"]]
    val_keys = {_key(ex["user"]) for ex in kept["val"]} | {_key(u) for u in v3_val_users}
    val_shingles = [_shingles(ex["user"]) for ex in kept["val"]] + [_shingles(u) for u in v3_val_users]
    train = []
    for ex in kept["train"]:
        if _key(ex["user"]) in val_keys:
            gates["same_prompt_as_validation_dropped_from_train"] += 1
            continue
        shingles = _shingles(ex["user"])
        if any(near_duplicate(shingles, s) for s in val_shingles):
            gates["near_duplicates_of_validation_dropped_from_train"] += 1  # the same prompt in both splits measures memory
            continue
        train.append(ex)
    val = kept["val"]

    def chat(ex: dict[str, Any]) -> dict[str, Any]:
        return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex["user"]},
                             {"role": "assistant", "content": ex["assistant"]}], "metadata": {**ex["metadata"], "corpus": "v4_trajectory"}}

    splits = {"train": v3["train"] + [chat(e) for e in train], "val": v3["val"] + [chat(e) for e in val]}
    projects_in = {s: sorted({e["metadata"]["project_id"] for e in items}) for s, items in (("train", train), ("val", val))}
    manifest = {"schema_version": 4, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), **status(trajs, meta),
                "total": sum(len(v) for v in splits.values()), "train": len(splits["train"]), "val": len(splits["val"]),
                "from_v3": {"train": len(v3["train"]), "val": len(v3["val"])},
                "trajectory_examples": {"train": len(train), "val": len(val), "by_task": dict(Counter(e["task"] for e in train + val))},
                "held_out": {"projects": sorted(held), "on_v3_held_out_datasets": sorted(p for p in held if (meta.get(p) or {}).get("held_out_dataset"))},
                "benchmark_projects_excluded": sorted(p for p, m in meta.items() if m.get("benchmark_case")),
                # held out only because no project hashed to validation: it may return to training in a later build
                "fallback_held_out": None if any((meta.get(p) or {}).get("held_out_dataset") for p in held)
                                     else fallback_held_out([t["project_id"] for t in trajs]),
                "projects_in_both_splits": sorted(set(projects_in["train"]) & set(projects_in["val"])),
                "quality_gates": {k: gates.get(k, 0) for k in ("duplicates_removed", "rejected_by_path_or_hash",
                                                                "same_prompt_as_validation_dropped_from_train",
                                                                "near_duplicates_of_validation_dropped_from_train")},
                "validation_policy": f"whole projects held out by their own hash (about one in {VAL_EVERY}) and every project on a dataset v3 holds "
                                     f"out ({', '.join(V3_HELD_OUT)}); projects on a judgment-suite table are left out; v3 examples keep v3's split"}
    return splits, manifest


def _inside(path: Path, folder: Path) -> bool:
    a, b = str(path.resolve()).casefold(), str(folder.resolve()).casefold().rstrip(os.sep)
    return a == b or a.startswith(b + os.sep)


def build(workspace: Path, out: Path = OUT, v3_dir: Path = V3) -> dict[str, Any]:
    """Write corpus v4 when the threshold is met; otherwise write nothing and say how many trajectories exist."""
    from ..agents.traces import open_traces
    from ..storage import open_stores

    if _inside(out, v3_dir):
        raise SystemExit(f"refusing to write into the v3 corpus ({v3_dir}); v4 has its own folder")
    projects, _, sessions = open_stores(Path(workspace))
    records, meta = workspace_records(projects, sessions, open_traces(Path(workspace)))
    trajs = trajectories_of(records, meta)
    state = status(trajs, meta)
    if not state["ready"]:
        return {**state, "written": False}
    splits, manifest = assemble(trajs, meta, v3_examples(v3_dir))
    out.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        (out / f"{split}.chat.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in items), encoding="utf-8")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**manifest, "written": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.studio.corpus_v4", description=__doc__.split("\n")[0])
    parser.add_argument("--workspace", type=Path, default=Path(os.environ.get("DCLAB_AGENT_HOME") or ROOT / "agent_runs"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    result = build(args.workspace, args.out)
    if not result["written"]:
        print(f"Not generated: {result['trajectories']} of {TARGET} trajectories from {result['projects']} of {MIN_PROJECTS} projects "
              f"(opted-in projects only; a trajectory has at least {MIN_DECISIONS} decisions). Nothing was written.")
        return 0
    print(f"Wrote corpus v4 to {args.out}: {result['train']} train, {result['val']} validation "
          f"({result['trajectory_examples']['train']} + {result['trajectory_examples']['val']} trajectory examples; "
          f"{len(result['held_out']['projects'])} projects held out).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
