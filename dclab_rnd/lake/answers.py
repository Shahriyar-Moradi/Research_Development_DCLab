"""Other answerers on every lake decision, captured whole (package 26.9): data for training DCLab's own models.

An answerer (first GLiNER2.5-Decide, Fastino's open-weight decision encoder run on this machine) answers every decision
of the lake's gold decision sets, and **everything** about each answer is kept, one JSON line per decision:

- what was asked: the decision's id and point, the text the model read (rendered from the decision's state: bands,
  name tokens and counts, never a table's value), the schema (the question and its labels), the render's version, and
  a fingerprint of all of it (a decision whose state changes is asked again);
- what came back: the raw output, the value as the decision's own answer (yes/no as true/false, a role, a column), the
  model's confidence;
- how: the model's id and the exact revision of its weights, the library's version, the machine, the time it took, when;
- what it may be used for: its licence and ``use`` (``train`` for an open-weight model run here, whose answers no
  service's terms restrict; a hosted model's answers are recorded ``eval_only`` until its terms are checked).

The labels are not copied in (they stay in gold, joined by id); the sealed split is answered too, so a later
measurement can read it once, but no score on it is ever computed here. Files: ``lake/bronze/answers/<answerer>/<point>/
<split>.jsonl``, appended batch by batch (a stopped run resumes where it stopped); the committed summary is
``research/llm-fine-tuning/experiments/lake/answers/MANIFEST.json``. ``lake build`` attaches each answer to its decision
as ``other_answers`` (never the label).

Nothing is trained or fine-tuned here: this only collects.

    python -m dclab_rnd.lake answers [--answerer gliner2.5-decide] [--points p1,p2] [--splits train,dev,sealed] [--limit N]
    python -m dclab_rnd.lake answers-summary
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from . import LAKE, ROOT, VERSION

RENDER = "render_v1"
SPLITS = ("train", "dev", "sealed")
MANIFEST = ROOT / "research" / "llm-fine-tuning" / "experiments" / "lake" / "answers" / "MANIFEST.json"
ANSWERERS: dict[str, dict[str, str]] = {
    "gliner2.5-decide": {"model": "fastino/GLiNER2.5-Decide", "licence": "Apache-2.0", "runs": "on this machine, open weights",
                         "use": "train", "why": "open weights under Apache 2.0, run locally: no service's terms restrict its answers"},
}  # a hosted answerer (Jev, OpenAI, GLiDE, Claude) is added with use "eval_only" until its terms are checked

PROMPTS = {
    "column.is_identifier": "Is this column an identifier: a key that names a row or an entity, never a property to learn from?",
    "column.time": "Does this column hold a date or a time?",
    "column.group": "Does this column name a group whose rows belong together (one customer's, one store's), so that the rows of "
                    "one value must stay on one side of a split?",
    "column.category_code": "Is this number a code for a category rather than a quantity?",
    "feature.leakage_suspect": "Is this column likely written only after the outcome is known, so that it would leak the outcome?",
}
ROLES = {"numeric": "a quantity or a measurement", "categorical_code": "a code or a label for a category",
         "identifier": "a key that names a row or an entity", "datetime": "a date or a time", "free_text": "text written by people",
         "other": "anything else"}
TASKS = {"binary": "two outcomes, a yes or a no", "multiclass": "one of several classes", "regression": "a number to predict"}


def _column(f: dict[str, Any]) -> str:
    parts = [f"Column {f['column']!r}" if f.get("column") else "A column"]
    if f.get("name_tokens"):
        parts.append(f"name words: {', '.join(map(str, f['name_tokens']))}")
    for key, said in (("dtype", "stored as"), ("uniqueness", "share of distinct values"), ("nulls", "share missing"),
                      ("cardinality", "distinct values"), ("value_pattern", "values look like")):
        if f.get(key) is not None:
            parts.append(f"{said} {f[key]}")
    return "; ".join(parts) + "."


def render(point: str, features: dict[str, Any], options: list[str]) -> tuple[str, dict[str, Any]]:
    """The text the model reads and the schema it answers, from a decision's state (bands and tokens only)."""
    f = {k: v for k, v in features.items() if v is not None}
    if point in PROMPTS:
        text = _column(f) + (f" The outcome predicted is {f['target']!r}." if f.get("target") else "")
        return text, {"answer": {"labels": ["yes", "no"], "prompt": PROMPTS[point]}}
    if point == "column.semantic_role":
        return _column(f), {"answer": {"labels": {k: v for k, v in ROLES.items() if k in options} or ROLES}}
    if point == "task.framing":
        t = f.get("target_state") or {}
        text = (f"A table of {f.get('rows', 'unknown')} size with {f.get('columns', 'some')} columns. The outcome is {f.get('target')!r}: "
                + _column(t))
        return text, {"answer": {"labels": {k: v for k, v in TASKS.items() if k in options} or TASKS,
                                 "prompt": "What kind of prediction is it?"}}
    if point == "target.column":
        cols = f.get("columns") if isinstance(f.get("columns"), list) else []
        listing = "; ".join(" ".join(map(str, c.get("name_tokens") or [])) + f" ({c.get('dtype')}, {c.get('cardinality')} values)" for c in cols[:60])
        return f"A table with these columns: {listing}.", {"answer": {"labels": list(options), "prompt": "Which column is the outcome to predict?"}}
    return json.dumps(f, sort_keys=True)[:4000], {"answer": {"labels": list(options)}}


def value_of(label: Any) -> Any:
    """The model's label as the decision's own answer: yes and no as true and false, anything else as it is."""
    return {"yes": True, "no": False}.get(label, label) if isinstance(label, str) else label


def fingerprint(text: str, schema: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps([RENDER, text, schema], sort_keys=True).encode()).hexdigest()[:16]


def path_of(answerer: str, point: str, split: str, lake: Path = LAKE) -> Path:
    return Path(lake) / "bronze" / "answers" / answerer / point / f"{split}.jsonl"


def stored(answerer: str, point: str, split: str, lake: Path = LAKE) -> list[dict[str, Any]]:
    path = path_of(answerer, point, split, lake)
    out = []
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        try:
            out.append(json.loads(line))
        except ValueError:  # a line cut by a stop mid-write: the rest stands, that decision is asked again
            continue
    return out


def decisions(point: str, split: str, lake: Path = LAKE) -> Iterator[dict[str, Any]]:
    path = Path(lake) / "gold" / "decisions" / point / VERSION / f"{split}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        if line.strip():
            yield json.loads(line)


def _load(answerer: str) -> tuple[Any, dict[str, Any]]:
    """The model and what identifies it exactly (the weights' revision, the library's version, the machine)."""
    info = ANSWERERS[answerer]
    from gliner2 import AutoExtractor  # the ai12 environment (torch): never imported by the studio itself

    model = AutoExtractor.from_pretrained(info["model"])
    revision = None
    try:
        from huggingface_hub import scan_cache_dir

        repo = next((r for r in scan_cache_dir().repos if r.repo_id == info["model"]), None)
        revision = sorted(rv.commit_hash for rv in repo.revisions)[0] if repo and repo.revisions else None
    except Exception:  # noqa: BLE001 - the revision is said unknown, never guessed
        revision = None
    import gliner2

    return model, {"revision": revision, "library": f"gliner2 {getattr(gliner2, '__version__', '?')}",
                   "machine": f"{platform.system()} {platform.machine()}, {getattr(getattr(model, 'device', None), 'type', 'cpu')}"}


def capture(answerer: str = "gliner2.5-decide", points: Optional[list[str]] = None, splits: tuple[str, ...] = SPLITS,
            limit: Optional[int] = None, batch: int = 32, lake: Path = LAKE, model: Any = None, identity: Optional[dict[str, Any]] = None,
            log: Any = print) -> dict[str, Any]:
    """Every decision of the gold sets asked once (resumable); returns what was asked now, per point and split."""
    info = ANSWERERS[answerer]
    if model is None:
        model, identity = _load(answerer)
    identity = identity or {"revision": None, "library": None, "machine": None}
    points = points or sorted(p.name for p in (Path(lake) / "gold" / "decisions").iterdir() if p.is_dir())
    asked: dict[str, int] = {}
    for point in points:
        for split in splits:
            done = {(r["sample_id"], r["input_fingerprint"]) for r in stored(answerer, point, split, lake)}
            todo = []
            for d in decisions(point, split, lake):
                text, schema = render(point, d.get("features") or {}, list(d.get("options") or []))
                if (d["sample_id"], fingerprint(text, schema)) not in done:
                    todo.append((d, text, schema))
            if limit is not None:
                todo = todo[:limit]
            path = path_of(answerer, point, split, lake)
            path.parent.mkdir(parents=True, exist_ok=True)
            groups: dict[str, list[tuple[dict[str, Any], str, dict[str, Any]]]] = defaultdict(list)
            for item in todo:  # one schema per call: a target column's options differ from table to table
                groups[json.dumps(item[2], sort_keys=True)].append(item)
            n = 0
            for key, items in groups.items():
                schema = items[0][2]
                for i in range(0, len(items), batch):
                    chunk = items[i:i + batch]
                    started = time.perf_counter()
                    outs = model.batch_classify_text([t for _, t, _ in chunk], schema, include_confidence=True)
                    seconds = (time.perf_counter() - started) / max(len(chunk), 1)
                    at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    lines = []
                    for (d, text, sch), out in zip(chunk, outs):
                        ans = (out or {}).get("answer")
                        label, confidence = (ans.get("label"), ans.get("confidence")) if isinstance(ans, dict) else (ans, None)
                        prov = d.get("provenance") or {}
                        lines.append(json.dumps({
                            "sample_id": d["sample_id"], "decision_point": point, "split": split, "group": prov.get("group"),
                            "label_origin": prov.get("origin"), "answerer": answerer, "model": info["model"], "revision": identity.get("revision"),
                            "library": identity.get("library"), "machine": identity.get("machine"), "render": RENDER,
                            "input_fingerprint": fingerprint(text, sch), "input_text": text, "schema": sch, "options": d.get("options"),
                            "output": out, "value": value_of(label), "confidence": confidence, "seconds": round(seconds, 4), "at": at,
                            "licence": info["licence"], "use": info["use"]}, ensure_ascii=False, default=str))
                    with path.open("a", encoding="utf-8") as fh:  # appended per batch: a stop loses at most one batch
                        fh.write("".join(x + "\n" for x in lines))
                    n += len(chunk)
            asked[f"{point}/{split}"] = n
            if n:
                log(f"{answerer} {point}/{split}: {n} asked")
    return {"answerer": answerer, "asked": asked}


def summary(lake: Path = LAKE, write: bool = True) -> dict[str, Any]:
    """What was captured, per answerer, point and split; agreement with the label and with the rule on dev only (the
    sealed split is never scored here: it is read once, by a measurement fixed before it)."""
    root = Path(lake) / "bronze" / "answers"
    out: dict[str, Any] = {}
    for answerer_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        per: dict[str, Any] = {}
        for point_dir in sorted(p for p in answerer_dir.iterdir() if p.is_dir()):
            point = point_dir.name
            entry: dict[str, Any] = {}
            for split in SPLITS:
                rows = stored(answerer_dir.name, point, split, lake)
                if not rows:
                    continue
                e: dict[str, Any] = {"answered": len(rows), "confidence_mean": round(sum(r["confidence"] or 0 for r in rows) / len(rows), 4),
                                     "seconds_mean": round(sum(r["seconds"] for r in rows) / len(rows), 4),
                                     "values": dict(Counter(str(r["value"]) for r in rows).most_common(8))}
                if split == "dev":
                    gold = {d["sample_id"]: d for d in decisions(point, "dev", lake)}
                    pairs = [(r, gold[r["sample_id"]]) for r in rows if r["sample_id"] in gold]
                    if pairs:
                        e["agrees_with_label"] = round(sum(str(r["value"]) == str(g["label"]) for r, g in pairs) / len(pairs), 4)
                        ruled = [(r, g) for r, g in pairs if g.get("rule_answer") is not None]
                        e["agrees_with_rule"] = round(sum(str(r["value"]) == str(g["rule_answer"]) for r, g in ruled) / len(ruled), 4) if ruled else None
                        e["rule_agrees_with_label"] = round(sum(str(g["rule_answer"]) == str(g["label"]) for _, g in ruled) / len(ruled), 4) if ruled else None
                entry[split] = e
            first = next((r for s in SPLITS for r in stored(answerer_dir.name, point, s, lake)[:1]), {})
            per[point] = {**entry, "model": first.get("model"), "revision": first.get("revision"), "render": first.get("render"),
                          "use": first.get("use"), "licence": first.get("licence")}
        out[answerer_dir.name] = per
    doc = {"what": "Other answerers' answers on the lake's decisions, each kept whole in lake/bronze/answers/ (not committed): the text and "
                   "schema asked, the raw output, the value, the confidence, the model's revision, the time; the labels stay in gold.",
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "answerers": out,
           "rule": "Agreement is computed on dev only and says how the answerer reads these decisions, not that it is ready to serve; the "
                   "sealed split is answered but never scored here."}
    if write:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return doc


def attach(rows: list[Any], lake: Path = LAKE) -> tuple[list[Any], Counter]:
    """Each decision sample with every captured answer on it as ``other_answers[answerer]`` (value, confidence, use, revision),
    when the answer was given on exactly this decision's state (its fingerprint); never as the label."""
    root = Path(lake) / "bronze" / "answers"
    if not root.is_dir() or not rows:
        return rows, Counter()
    by: dict[tuple[str, str], dict[str, Any]] = {}
    for answerer_dir in (p for p in root.iterdir() if p.is_dir()):
        for point_dir in (p for p in answerer_dir.iterdir() if p.is_dir()):
            for split in SPLITS:
                for r in stored(answerer_dir.name, point_dir.name, split, lake):
                    by[(answerer_dir.name, r["sample_id"])] = r
    answerers = sorted({a for a, _ in by})
    out, why = [], Counter()
    for row in rows:
        extra = {}
        for a in answerers:
            r = by.get((a, row.sample_id))
            if r is None:
                continue
            text, schema = render(row.decision_point, dict(row.features), list(row.options))
            if r["input_fingerprint"] != fingerprint(text, schema):
                why[f"{a}: answered on another state (ask again)"] += 1
                continue
            extra[a] = {"value": r["value"], "confidence": r["confidence"], "use": r["use"], "revision": r["revision"]}
            why[f"{a}: attached"] += 1
        out.append(row.model_copy(update={"other_answers": {**dict(row.other_answers), **extra}}) if extra else row)
    return out, why


__all__ = ["ANSWERERS", "RENDER", "render", "value_of", "fingerprint", "capture", "summary", "attach", "stored"]
