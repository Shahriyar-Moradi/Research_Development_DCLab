"""Every decision DCLab's rules, Claude, DCLab's classifiers and people made, gathered whole (package 26.9).

The ledger writes each decision where it is made, in the workspace it is made in; the twin runs keep Claude's steps and
DCLab's beside them; the gateway logs every model request. They are scattered over many workspace folders (the
development workspace, every live test, every twin lane, every tester) and a database. ``harvest`` gathers all of it into
one structure under the lake's bronze layer, each record whole, nothing summarised away, duplicates (a twin run kept in
two places) once:

- ``bronze/sessions/ledgers/decisions.jsonl`` (and ``labels.jsonl``, ``outcomes.jsonl``): every decision record with every
  answer on it (the rule's, a model role's, a classifier's shadow, a person's), the workspace it came from (``found_in``)
  and the models that workspace's requests were served by;
- ``bronze/sessions/twin/<run>/``: each twin run's compared steps, Claude's expert record and design, the run's notes;
- ``bronze/sessions/model_requests.jsonl``: every model request (purpose, model, tokens, seconds, cost, outcome);
- per-answerer views in the answers' own layout (``bronze/answers/<answerer>/sessions/<point>/all.jsonl``):
  ``dclab-rules`` (the rule's answer of every recorded decision, ``use: train``, DCLab's own), ``claude`` (every answer a
  model role gave where Claude served it, and every twin step's AI-first answer: ``use: eval_only``, the owner's rule),
  ``dclab-classifiers`` (shadows) and ``people`` (a person's own answer, ``use: train`` under its workspace's consent).

Re-run any time (``make capture``): it is rebuilt whole from the workspaces, which keep everything themselves.

    python -m dclab_rnd.lake harvest
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from . import LAKE, ROOT

WORKSPACES = ROOT / "agent_runs"
BY = {"rule": ("dclab-rules", "train"), "classifier": ("dclab-classifiers", "train"), "person": ("people", "train")}


def _lines(path: Path) -> Iterator[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        try:
            yield json.loads(line)
        except ValueError:
            continue


def _key(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)) or "."
    except ValueError:
        return str(path)


def served_models(folder: Path) -> dict[str, str]:
    """The model each purpose was served by in a workspace (its model_requests.jsonl), the latest per purpose."""
    out: dict[str, str] = {}
    for r in _lines(folder / "model_requests.jsonl"):
        if r.get("purpose") and r.get("model") and r.get("outcome") in (None, "ok"):
            out[str(r["purpose"])] = str(r["model"])
    return out


def _database_ledger() -> tuple[list[dict[str, Any]], list[dict[str, Any]], Optional[str]]:
    """The development database's decision records and labels, when it answers (``DCLAB_DATABASE_URL`` or dclab_dev)."""
    url = os.environ.get("DCLAB_DATABASE_URL") or f"postgresql+psycopg://{os.environ.get('USER', '')}@/dclab_dev"
    try:
        import sqlalchemy as sa

        engine = sa.create_engine(url)
        with engine.connect() as c:
            records = [{**(json.loads(r[0]) if isinstance(r[0], str) else dict(r[0])), "_ws": r[1]}
                       for r in c.execute(sa.text("select doc, workspace_id from decision_record"))]
            labels = [{**(json.loads(r[0]) if isinstance(r[0], str) else dict(r[0])), "_ws": r[1]}
                      for r in c.execute(sa.text("select doc, workspace_id from decision_label"))]
        return records, labels, None
    except Exception as e:  # noqa: BLE001 - no database here: the folders alone, said so
        return [], [], f"{type(e).__name__}: {str(e)[:160]}"


def harvest(root: Path = WORKSPACES, lake: Path = LAKE, database: bool = True) -> dict[str, Any]:
    root, out = Path(root), Path(lake) / "bronze" / "sessions"
    shutil.rmtree(out, ignore_errors=True)  # rebuilt whole: the workspaces keep everything themselves
    (out / "ledgers").mkdir(parents=True)
    records: dict[str, dict[str, Any]] = {}
    labels: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[str, Any]] = {}
    ledgers = sorted(p for p in root.rglob("ledger") if p.is_dir() and (p / "decisions").is_dir()) if root.is_dir() else []
    for ledger in ledgers:
        workspace = ledger.parent
        where, models = _key(workspace, root), served_models(workspace)
        for kind, store in (("decisions", records), ("labels", labels), ("outcomes", outcomes)):
            for path in sorted((ledger / kind).rglob("*.jsonl")):
                for doc in _lines(path):
                    ident = str(doc.get("id") or doc.get("record") or json.dumps(doc, sort_keys=True)[:200])
                    if ident in store:
                        store[ident].setdefault("found_in", []).append(where)
                        continue
                    store[ident] = {**doc, "found_in": [where], **({"served_models": models} if kind == "decisions" else {})}
    database_note = None
    if database:
        db_records, db_labels, database_note = _database_ledger()
        for doc in db_records:
            ws = doc.pop("_ws", None)
            records.setdefault(str(doc.get("id")), {**doc, "found_in": [f"postgres:{ws}"], "served_models": {}})
        for doc in db_labels:
            ws = doc.pop("_ws", None)
            labels.setdefault(str(doc.get("id")), {**doc, "found_in": [f"postgres:{ws}"]})
    for name, store in (("decisions", records), ("labels", labels), ("outcomes", outcomes)):
        (out / "ledgers" / f"{name}.jsonl").write_text("".join(json.dumps(v, ensure_ascii=False, default=str) + "\n" for v in store.values()),
                                                      encoding="utf-8")
    # twin runs: Claude's lane beside DCLab's, step by step, its expert record and design kept
    runs: dict[str, Path] = {}
    for steps in sorted(root.rglob("steps.jsonl")) if root.is_dir() else []:
        if steps.parent.parent.name == "runs" and steps.parent.name not in runs:
            runs[steps.parent.name] = steps.parent
    for name, folder in runs.items():
        dest = out / "twin" / name
        dest.mkdir(parents=True, exist_ok=True)
        for item in ("steps.jsonl", "expert_steps.jsonl", "compare.json", "run.json", "COMPARE.md", "README.md"):
            if (folder / item).is_file():
                shutil.copy2(folder / item, dest / item)
        for lane in ("ai_first", "workflow"):
            for sub in ("design", "transcripts"):
                if (folder / lane / sub).is_dir():
                    shutil.copytree(folder / lane / sub, dest / lane / sub, dirs_exist_ok=True)
            if (folder / lane / "summary.json").is_file():
                (dest / lane).mkdir(parents=True, exist_ok=True)
                shutil.copy2(folder / lane / "summary.json", dest / lane / "summary.json")
    # every model request, once
    seen, requests = set(), []
    for path in sorted(root.rglob("model_requests.jsonl")) if root.is_dir() else []:
        for r in _lines(path):
            key = json.dumps([r.get(k) for k in ("at", "purpose", "model", "input_tokens", "output_tokens", "project_id", "draft_id")])
            if key not in seen:
                seen.add(key)
                requests.append({**r, "found_in": _key(path.parent, root)})
    (out / "model_requests.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in requests), encoding="utf-8")
    views = _views(records, runs, Path(lake))
    manifest = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "ledgers": len(ledgers), "decisions": len(records),
                "labels": len(labels), "outcomes": len(outcomes), "twin_runs": len(runs), "model_requests": len(requests),
                "decisions_by_point": dict(Counter(r.get("point") for r in records.values()).most_common()),
                "answers_by_answerer": views, "database": database_note or "read",
                "requests_by_model": dict(Counter(r.get("model") for r in requests).most_common())}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n", encoding="utf-8")
    return manifest


def _views(records: dict[str, dict[str, Any]], runs: dict[str, Path], lake: Path) -> dict[str, int]:
    """Each answer of each recorded decision under its answerer, in the answers' layout (source ``sessions``; twin steps
    under ``twin``), rebuilt whole."""
    base = lake / "bronze" / "answers"
    for answerer in ("dclab-rules", "dclab-classifiers", "people", "claude"):
        for source in ("sessions", "twin"):
            shutil.rmtree(base / answerer / source, ignore_errors=True)
    files: dict[Path, list[str]] = defaultdict(list)
    counts: Counter = Counter()
    for r in records.values():
        claude = any("claude" in str(m) for m in (r.get("served_models") or {}).values())
        for a in r.get("answers") or []:
            by = a.get("by")
            if by == "model_role":
                if not claude:
                    continue
                answerer, use = "claude", "eval_only"
            elif by in BY:
                answerer, use = BY[by]
            else:
                continue
            if a.get("source") == "shadow":
                answerer, use = "dclab-classifiers", "train"
            row = {"sample_id": r.get("id"), "source": "sessions", "decision_point": r.get("point"), "answerer": answerer,
                   "answered_by": a.get("answerer"), "value": a.get("value"), "confidence": a.get("confidence"), "reason": a.get("reason"),
                   "state": r.get("state"), "options": r.get("options"), "value_used": r.get("value_used"), "used": r.get("used"),
                   "at": r.get("at"), "workspace": (r.get("found_in") or [None])[0], "consent": r.get("consent"), "use": use}
            files[base / answerer / "sessions" / str(r.get("point")) / "all.jsonl"].append(json.dumps(row, ensure_ascii=False, default=str))
            counts[answerer] += 1
    for name, folder in runs.items():
        for step in _lines(folder / "steps.jsonl"):
            for arm, answerer, use in (("ai_first", "claude", "eval_only"), ("system", "dclab-rules", "train")):
                side = step.get(arm) or {}
                row = {"sample_id": f"{name}:{step.get('id')}", "source": "twin", "decision_point": step.get("step"), "answerer": answerer,
                       "answered_by": side.get("answerer"), "status": side.get("status"), "value": side.get("answer", side.get("value")),
                       "input_summary": step.get("input_summary"), "truth": step.get("truth"), "verdict": step.get("verdict"),
                       "at": step.get("at"), "run": name, "use": use, "detail": side}
                files[base / answerer / "twin" / str(step.get("step")) / "all.jsonl"].append(json.dumps(row, ensure_ascii=False, default=str))
                counts[f"{answerer} (twin)"] += 1
    for path, lines in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(x + "\n" for x in lines), encoding="utf-8")
    return dict(counts)


__all__ = ["harvest", "served_models"]
