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
  ``dclab-rules`` (the rule's answer of every recorded decision, ``use: train``, DCLab's own), one answerer per model a
  model role's answer came from (the model of the request that made it, read from its workspace's kept exchange, else the
  models that workspace was served by): ``claude`` (``use: eval_only``, the owner's rule, and every twin step's AI-first
  answer Claude gave), ``qwen3.8-flash`` for Qwen served through OpenRouter (``use: train``, the owner's decision of
  2026-11-11 in ``store/owner_decisions.json``), any other model under its own name (``use: eval_only`` until its terms
  are checked), ``dclab-classifiers`` (shadows) and ``people`` (a person's own answer, ``use: train`` under its workspace's
  consent).

Re-run any time (``make capture``): it is rebuilt whole from the workspaces, which keep everything themselves.

    python -m dclab_rnd.lake harvest
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from . import LAKE, ROOT

WORKSPACES = ROOT / "agent_runs"
BY = {"rule": ("dclab-rules", "train"), "classifier": ("dclab-classifiers", "train"), "person": ("people", "train")}
from ..store.keepers.workspaces import SCRIPTED, scripted  # noqa: E402,F401 - one rule for the lake and the store (S8): a workspace a
# script drove (the end-to-end and browser tests, written people): its "person" answers are a script's, never a person's


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


def served_times(folder: Path) -> list[tuple[float, str]]:
    """Every ok model request of a workspace as (when, model), oldest first: every model it was ever served by, with its time
    (the latest per purpose alone would file a purpose's pre-switch Claude answers under the model that serves it now)."""
    out = []
    for r in _lines(folder / "model_requests.jsonl"):
        when = _when(r.get("at"))
        if r.get("model") and r.get("outcome") in (None, "ok") and when is not None:
            out.append((when, str(r["model"])))
    return sorted(out)


def _when(text: Any) -> Optional[float]:
    try:
        when = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
        return (when if when.tzinfo else when.replace(tzinfo=timezone.utc)).timestamp()
    except (TypeError, ValueError):
        return None


WINDOW = 600.0  # seconds before a decision in which the request that made a model role's answer was sent
SEALED_PLACES = ("exam_vault", "sealed")  # a workspace under one of these is a sealed run's: its model answers never train


def sealed_place(where: Any) -> bool:
    return any(part in SEALED_PLACES for part in Path(str(where or "")).parts)


def served_models(folder: Path) -> dict[str, str]:
    """The model each purpose was served by in a workspace (its model_requests.jsonl), the latest per purpose."""
    out: dict[str, str] = {}
    for r in _lines(folder / "model_requests.jsonl"):
        if r.get("purpose") and r.get("model") and r.get("outcome") in (None, "ok"):
            out[str(r["purpose"])] = str(r["model"])
    return out


REQUEST_ID = re.compile(r'"model_request": "([^"\\]+)"')  # an exchange line's own request id, near its start (before the messages)


def request_models(folder: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    """The model that served each wanted request of a workspace, from its kept exchanges (``model_exchanges/``): the
    model's name, and the ``provider`` and ``source_class`` the gateway wrote when it did. Only a wanted line is parsed."""
    out: dict[str, dict[str, Any]] = {}
    exchanges = folder / "model_exchanges"
    if not wanted or not exchanges.is_dir():
        return out
    for path in sorted(exchanges.rglob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                m = REQUEST_ID.search(line, 0, 600)
                if m is None or m.group(1) not in wanted or m.group(1) in out:
                    continue
                try:
                    doc = json.loads(line)
                except ValueError:
                    continue
                if doc.get("model"):
                    out[m.group(1)] = {k: doc.get(k) for k in ("model", "provider", "source_class")}
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
        driven = scripted(workspace, root)
        added: list[str] = []
        for kind, store in (("decisions", records), ("labels", labels), ("outcomes", outcomes)):
            for path in sorted((ledger / kind).rglob("*.jsonl")):
                for doc in _lines(path):
                    ident = str(doc.get("id") or doc.get("record") or json.dumps(doc, sort_keys=True)[:200])
                    if ident in store:
                        store[ident].setdefault("found_in", []).append(where)
                        continue
                    store[ident] = {**doc, "found_in": [where], **({"served_models": models} if kind == "decisions" else {}),
                                    **({"scripted": True} if driven else {})}
                    if kind == "decisions":
                        added.append(ident)
        # the model of the request behind each model role's answer (its source), from this workspace's kept exchanges; when
        # none is kept (most roles pass no request id), the models served just before the decision, then every model ever
        # served there: never only the latest per purpose (review of step 3: that filed Claude's past answers as Qwen's)
        roled = [i for i in added if any(a.get("by") == "model_role" for a in records[i].get("answers") or [])]
        wanted = {str(a["source"]) for i in roled for a in records[i].get("answers") or [] if a.get("by") == "model_role" and a.get("source")}
        served = request_models(workspace, wanted)
        times = served_times(workspace) if roled else []
        ever = sorted({m for _, m in times})
        for i in roled:
            mine = {str(a["source"]): served[str(a["source"])] for a in records[i].get("answers") or []
                    if a.get("by") == "model_role" and str(a.get("source")) in served}
            if mine:
                records[i]["role_models"] = mine
            at = _when(records[i].get("at"))
            if at is not None:
                records[i]["models_before"] = sorted({m for t, m in times if at - WINDOW <= t <= at + 5})
            records[i]["ever_served"] = ever
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
    views = _views(records, runs, Path(lake), labels)
    manifest = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "ledgers": len(ledgers), "decisions": len(records),
                "labels": len(labels), "outcomes": len(outcomes), "twin_runs": len(runs), "model_requests": len(requests),
                "decisions_by_point": dict(Counter(r.get("point") for r in records.values()).most_common()),
                "answers_by_answerer": views, "database": database_note or "read",
                "requests_by_model": dict(Counter(r.get("model") for r in requests).most_common())}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n", encoding="utf-8")
    return manifest


SILVER_BAR = 0.8  # the judge's measured accuracy on a point below which its verdicts there are kept but not usable


def judge_accuracy(results: Optional[Path] = None) -> dict[str, float]:
    """29.18: the judge's accuracy per point, measured blind on homework with known truth (the latest judge run of
    ``evidence/campaigns/students_exam_v1``); a point it was never measured on is absent."""
    folder = results or ROOT / "evidence" / "campaigns" / "students_exam_v1" / "results"
    runs = []
    for path in sorted(folder.glob("SEX-*.json")) if folder.is_dir() else []:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if doc.get("kind_of_run") == "judge":
            runs.append(doc)
    latest = runs[-1] if runs else {}
    out: dict[str, float] = {}
    for p, j in (latest.get("judge") or {}).items():
        if j.get("accuracy") is not None:
            out[p] = float(j["accuracy"])
        for confidence, c in (j.get("by_confidence") or {}).items():  # "point|high": its verdicts at that confidence, when 10 or more were measured
            if int(c.get("judged") or 0) >= MIN_MEASURED:
                out[f"{p}|{confidence}"] = round(int(c.get("right") or 0) / int(c["judged"]), 4)
    return out


MIN_MEASURED = 10  # verdicts at one confidence on one point before that confidence gets its own weight


def model_of(r: dict[str, Any], a: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Which model gave a model role's answer: the model of its own request (its kept exchange) when known; else the models
    served in the ten minutes before the decision; else every model its workspace was ever served by. In either set Claude
    among them is Claude (evaluation only, never filed as another's), one model alone is that model, several others are not
    known. None when no model is known at all (a database record)."""
    info = (r.get("role_models") or {}).get(str(a.get("source") or ""))
    if info and info.get("model"):
        return {**info, "known_by": "its request's kept exchange"}
    for models, said in ((r.get("models_before"), "served in the ten minutes before the decision"),
                         (r.get("ever_served"), "its workspace was ever served by"),
                         (sorted({str(m) for m in (r.get("served_models") or {}).values() if m}), "its workspace was served by")):
        models = [str(m) for m in models or [] if m]
        claude = [m for m in models if "claude" in m.lower()]
        if claude:
            return {"model": claude[0], "known_by": f"a model {said} (Claude among them: evaluation only)"}
        if len(models) == 1:
            return {"model": models[0], "known_by": f"the one model {said}"}
        if models:
            return {"model": None, "known_by": f"{len(models)} models {said}; which one answered is not known"}
    return None


def _ai_first_model(side: dict[str, Any]) -> Any:
    """The model of a twin step's AI-first lane: its own ``model`` field (Home's lane), else its answerer without the expert's
    prefix (``ai_first@expert:qwen/qwen3.8-flash``), else the answerer as written (compare's lane names the model)."""
    answerer = str(side.get("answerer") or "")
    return side.get("model") or (answerer.split(":", 1)[1] if answerer.startswith("ai_first@expert:") else answerer)


def answerer_for(info: dict[str, Any]) -> tuple[str, str]:
    """The lake's answerer and use for a model: ``claude`` (eval_only, the owner's rule); Qwen served through OpenRouter under
    its own name (train, the owner's decision of 2026-11-11, while ``owner_decisions.json`` says yes); any other model under
    its own name, eval_only until its terms are checked; an unknown model ``unknown-model``, eval_only."""
    from ..store import provider_outputs as P

    model = info.get("model")
    if not model:
        return "unknown-model", "eval_only"
    if "claude" in str(model).lower():
        return "claude", "eval_only"
    if P.openrouter_qwen(model, info.get("provider"), info.get("source_class")):
        return P.answerer_name(model), "train" if P.for_training(decision=P.OPENROUTER_QWEN) else "eval_only"
    return P.answerer_name(model), "eval_only"


def _views(records: dict[str, dict[str, Any]], runs: dict[str, Path], lake: Path,
           labels: Optional[dict[str, dict[str, Any]]] = None) -> dict[str, int]:
    """Each answer of each recorded decision under its answerer, in the answers' layout (source ``sessions``; twin steps
    under ``twin``), rebuilt whole. 29.18: Claude's verdicts as the blind judge of real past decisions
    (``decision_labeler``, source ``model``) under ``claude/sessions/<point>/judge.jsonl``: silver (the owner's rule of
    2026-10-17), weighted by the judge's measured accuracy on the point, usable for training only at or above
    ``SILVER_BAR``, removable in one step (those files)."""
    from ..store import provider_outputs as P
    from .answers import ANSWERERS

    base = lake / "bronze" / "answers"
    for folder in (p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []:  # every answerer's harvested sources, rebuilt
        for source in ("sessions", "twin"):  # whole (only the harvest writes these two; capture's sources are never touched)
            shutil.rmtree(folder / source, ignore_errors=True)
    files: dict[Path, list[str]] = defaultdict(list)
    counts: Counter = Counter()
    for r in records.values():
        for a in r.get("answers") or []:
            by, served = a.get("by"), None
            if by == "model_role" and (a.get("source") == "shadow" or str(a.get("answerer") or "").startswith("shadow:")):
                # a shadow model's answer (logged, never used): under its own model, never the served model's
                served = {"model": str(a.get("answerer") or "").removeprefix("shadow:") or None, "known_by": "the shadow's own name"}
                answerer, use = answerer_for(served)
            elif by == "model_role":
                served = model_of(r, a)
                if served is None:  # no model known for it at all: not filed under a guessed answerer
                    continue
                answerer, use = answerer_for(served)
            elif by == "person" and r.get("scripted"):  # a test script's click, never a person's answer
                continue
            elif by in BY:
                answerer, use = BY[by]
            else:
                continue
            if a.get("source") == "shadow" and by != "model_role":
                answerer, use = "dclab-classifiers", "train"
            if served is not None and use == "train" and any(sealed_place(w) for w in r.get("found_in") or []):
                use = "eval_only"  # a sealed run's model answer: kept, never a training row
            row = {"sample_id": r.get("id"), "source": "sessions", "decision_point": r.get("point"), "answerer": answerer,
                   "answered_by": a.get("answerer"), "value": a.get("value"), "confidence": a.get("confidence"), "reason": a.get("reason"),
                   "state": r.get("state"), "options": r.get("options"), "value_used": r.get("value_used"), "used": r.get("used"),
                   "at": r.get("at"), "workspace": (r.get("found_in") or [None])[0], "consent": r.get("consent"), "use": use}
            if served is not None and answerer != "dclab-classifiers":
                row.update({"model": served.get("model"), "provider": served.get("provider"), "source_class": served.get("source_class"),
                            "model_known_by": served.get("known_by"), "licence": (ANSWERERS.get(answerer) or {}).get("licence")})
            files[base / answerer / "sessions" / str(r.get("point")) / "all.jsonl"].append(json.dumps(row, ensure_ascii=False, default=str))
            counts[answerer] += 1
    measured = judge_accuracy()
    for x in (labels or {}).values():
        if x.get("source") != "model" or "claude" not in str(x.get("by")):
            continue
        r = records.get(str(x.get("record")))
        if r is None:
            continue
        ref = str((x.get("evidence") or {}).get("ref") or "")
        confidence = ref.split(":")[0] if ":" in ref else None
        accuracy = measured.get(f"{x.get('point')}|{confidence}", measured.get(str(x.get("point"))))  # by its own confidence when measured
        claude_served = any("claude" in str(m) for m in (r.get("served_models") or {}).values())
        self_judged = claude_served and (r.get("used") == "model_role" or any(a.get("by") == "model_role" for a in r.get("answers") or []))
        row = {"sample_id": r.get("id"), "source": "sessions", "decision_point": x.get("point"), "answerer": "claude", "role": "judge",
               "answered_by": x.get("by"), "value": x.get("value"), "confidence": ref.split(":")[0] if ":" in ref else None,
               "reason": ref.split(":", 1)[1].strip() if ":" in ref else ref, "draw_weight": x.get("weight"), "state": r.get("state"),
               "options": r.get("options"), "value_used": r.get("value_used"), "used": r.get("used"), "at": x.get("at"),
               "workspace": (r.get("found_in") or [None])[0], "consent": r.get("consent"), "use": "silver",
               "silver_weight": accuracy, "usable": accuracy is not None and accuracy >= SILVER_BAR and not self_judged,
               "self_judged": self_judged,
               "why": ("the judge is the model that answered this decision: kept, not usable" if self_judged
                       else f"weighted by the judge's measured accuracy on this point at {confidence} confidence"
                       if f"{x.get('point')}|{confidence}" in measured
                       else "weighted by the judge's measured accuracy on this point" if accuracy is not None
                       else "the judge was never measured on this point: kept, not usable")}
        files[base / "claude" / "sessions" / str(x.get("point")) / "judge.jsonl"].append(json.dumps(row, ensure_ascii=False, default=str))
        counts["claude (judge, silver)"] += 1
    for name, folder in runs.items():
        for step in _lines(folder / "steps.jsonl"):
            ai = step.get("ai_first") or {}
            ai_model = _ai_first_model(ai)
            ai_by = (answerer_for({"model": ai_model, "provider": ai.get("provider"), "source_class": ai.get("source_class")})
                     if P.openrouter_qwen(ai_model, ai.get("provider"), ai.get("source_class")) else ("claude", "eval_only"))
            if ai_by[1] == "train" and sealed_place(folder):  # a sealed run's lane: kept, never a training row
                ai_by = (ai_by[0], "eval_only")
            # the AI-first lane was Claude's (eval_only) until Qwen served it through OpenRouter (2026-11-11: its own answerer)
            for arm, answerer, use in (("ai_first", *ai_by), ("system", "dclab-rules", "train")):
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
