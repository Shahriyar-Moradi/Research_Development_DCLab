"""Other answerers on every DCLab test, captured whole (package 26.9): data for training DCLab's own models.

An answerer (first GLiNER2.5-Decide, Fastino's open-weight decision encoder run on this machine) answers every decision
of every test DCLab has, and **everything** about each answer is kept. Nothing is trained here: this only collects.

The tests, each a **source** of its own (``SOURCES``), each decision with its true answer from the test itself:

- ``lake``: the training lake's gold decisions (columns and tables: identifier, role, time, group, code, leak, the
  outcome column, the task), labelled by construction or by a public dataset's own marks;
- ``workflow``: the workflow decisions recorded while the written people of every journey set were played (26.3: the
  family, ready to build, the next question, the route), each labelled by the answers its journey accepts;
- ``journeys``: every written person's first sentence, in every journey set: which family of problem it is (a programme
  journey's several families at once);
- ``selfplay``: the verified invented problems (27.3): the problem's type, its time column, and for every column whether
  it is the planted leak or an identifier, labelled by what the replica planted, never by DCLab's answer.

Files (all under the lake's bronze layer, which no rebuild removes, and kept permanently by ``make archive``):

- ``bronze/decisions/<source>/<point>/<split>.jsonl``: each test's labelled decisions as they were asked (the lake's own
  are its gold sets): id, point, split, group, the text and schema, the options, the true answer (and every accepted one),
  the rule's answer, where the label comes from;
- ``bronze/answers/<answerer>/<source>/<point>/<split>.jsonl``: each answer whole: what was asked (the text, the schema,
  their fingerprint), the raw output, the value, the confidence, the model's id and weights' revision, the library, the
  machine, the time, the licence and ``use`` (``train`` for an open-weight model run here; a hosted model's answers are
  ``eval_only`` until its terms are checked). A decision whose text changes is asked again; a stopped run resumes.
- ``export_gliner2`` writes every source's labelled decisions in GLiNER2's own training format (``{"input", "output":
  {"classifications": [...]}}``, its tutorial's shape) to ``training_records/datasets/gliner2/<version>/``, the sealed
  decisions in their own files, never mixed into training.

The summary (``research/llm-fine-tuning/experiments/lake/answers/MANIFEST.json``, committed) gives agreement on dev only;
a sealed split is answered but never scored here.

    python -m dclab_rnd.lake answers [--sources lake,workflow,journeys,selfplay] [--points …] [--splits …] [--limit N]
    python -m dclab_rnd.lake answers-summary
    python -m dclab_rnd.lake answers-export
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from . import LAKE, ROOT, VERSION

RENDER = "render_v1"
SPLITS = ("train", "dev", "sealed")
SOURCES = ("lake", "workflow", "journeys", "selfplay")
MANIFEST = ROOT / "research" / "llm-fine-tuning" / "experiments" / "lake" / "answers" / "MANIFEST.json"
EXPORT = ROOT / "training_records" / "datasets" / "gliner2"
JOURNEYS = ROOT / "dclab_rnd" / "agent_eval" / "journeys"
WORKFLOW = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "workflow_v1" / "decisions"
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
QUESTIONS = {"target": "what exactly is predicted (the outcome)", "prediction_moment": "when each prediction is made and what is known then",
             "action": "what happens with each prediction and what a wrong one costs", "costs": "what each kind of error costs",
             "data_plan": "where the data comes from", "entity": "who or what each prediction is about",
             "labels": "how the known cases were found", "budget": "how many cases can be reviewed", "family": "which kind of problem it is",
             "none": "no further question: enough is known"}


def _families() -> dict[str, str]:
    from dclab_core.card import FAMILIES

    return dict(FAMILIES)


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


def _workflow_text(point: str, s: dict[str, Any]) -> str:
    said = [f"{k.replace('_', ' ')}: {v}" for k, v in sorted(s.items()) if v is not None]
    head = {"problem.family": "A conversation about a machine-learning problem, so far.",
            "problem.ready": "A problem card about to be built into a project.",
            "workflow.next_question": "A conversation that understands a machine-learning problem, so far.",
            "solution.route": "A part of a machine-learning problem to be solved."}.get(point, "A decision.")
    return head + " " + "; ".join(said) + "."


def value_of(label: Any) -> Any:
    """The model's label as the decision's own answer: yes and no as true and false, anything else as it is."""
    return {"yes": True, "no": False}.get(label, label) if isinstance(label, str) else label


def fingerprint(text: str, schema: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps([RENDER, text, schema], sort_keys=True).encode()).hexdigest()[:16]


def _split(group: str) -> str:
    from dclab_core import lake as L

    return L.split_of(group)


# ------------------------------------------------------------------------------------------------ the tests, as decisions

def _lake(lake: Path) -> Iterator[dict[str, Any]]:
    root = Path(lake) / "gold" / "decisions"
    for point_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        for split in SPLITS:
            path = point_dir / VERSION / f"{split}.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
                if not line.strip():
                    continue
                d = json.loads(line)
                text, schema = render(d["decision_point"], d.get("features") or {}, list(d.get("options") or []))
                prov = d.get("provenance") or {}
                yield {"sample_id": d["sample_id"], "decision_point": d["decision_point"], "split": split, "group": prov.get("group"),
                       "text": text, "schema": schema, "options": d.get("options"), "label": d.get("label"), "accepted": None,
                       "rule_answer": d.get("rule_answer"), "label_origin": prov.get("origin")}


def _workflow(_: Path) -> Iterator[dict[str, Any]]:
    from dclab_core.workflow_labels import QUESTIONS as ASKABLE

    for point_dir in sorted(p for p in WORKFLOW.iterdir() if p.is_dir()) if WORKFLOW.is_dir() else []:
        point = point_dir.name
        path = point_dir / "rows.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
            r = json.loads(line)
            if not r.get("label"):
                continue
            if point == "problem.family":
                labels: Any = _families()
            elif point == "problem.ready":
                labels = ["yes", "no"]
            elif point == "workflow.next_question":
                labels = {k: QUESTIONS[k] for k in ASKABLE}
            else:
                labels = list(r.get("options") or [])
            if not labels:
                continue
            prompt = {"problem.family": "Which family of machine-learning problem is it?", "problem.ready": "Is the problem understood enough to build?",
                      "workflow.next_question": "Which question should be asked next, if any?", "solution.route": "Which approach solves this part?"}[point]
            value = r["label"]["value"]
            yield {"sample_id": r["record"], "decision_point": point, "split": r.get("split") or "train", "group": r.get("group"),
                   "text": _workflow_text(point, r.get("state") or {}), "schema": {"answer": {"labels": labels, "prompt": prompt}},
                   "options": list(labels), "label": ("yes" if value is True else "no" if value is False else value),
                   "accepted": r["label"].get("accepted"), "rule_answer": r.get("value_used"), "label_origin": "journey_truth"}


def _journeys(_: Path) -> Iterator[dict[str, Any]]:
    families = _families()
    for path in sorted(JOURNEYS.glob("journeys_*.json")):
        name = path.stem.replace("journeys_", "")
        for j in json.loads(path.read_text(encoding="utf-8")):
            family = j.get("family")
            expected = (j.get("expect") or {}).get("families")
            multi = bool(expected) and not family
            accepted = [f for f in (expected if multi else (family if isinstance(family, list) else [family])) if f in families]
            if not accepted or not j.get("words"):
                continue
            group = f"journey:{name}:{j['id']}"
            schema = {"answer": {"labels": families, "prompt": "Which family of machine-learning problem does this person describe?",
                                 **({"multi_label": True, "cls_threshold": 0.4} if multi else {})}}
            yield {"sample_id": group, "decision_point": "problem.family_from_words", "split": _split(group), "group": group,
                   "text": str(j["words"]), "schema": schema, "options": list(families), "label": accepted if multi else accepted[0],
                   "accepted": accepted, "rule_answer": None, "label_origin": "journey_truth", "set": name}


def _selfplay(lake: Path) -> Iterator[dict[str, Any]]:
    from .benchmark import truth

    path = Path(lake) / "bronze" / "selfplay" / "examples.jsonl"
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        ex = json.loads(line)
        if ex["family"] not in ("problem_type", "split_design", "leakage_judgment"):
            continue
        try:
            given, answer = json.loads(ex["messages"][-2]["content"]), json.loads(ex["messages"][-1]["content"])
            t = truth(ex)
        except Exception:  # noqa: BLE001 - a problem whose replica cannot be rebuilt gives no test
            continue
        prov = ex["provenance"]
        masked = "col_01" in ex["messages"][-2]["content"]
        base = f"{prov['group']}:{'masked' if masked else 'named'}"
        cols = [c for c in given.get("columns") or [] if isinstance(c, dict) and c.get("column")]
        about = (f"A table of {given.get('rows')} rows. The outcome is {given.get('outcome')!r}. The prediction is made: "
                 f"{str(given.get('prediction_moment') or 'not said').rstrip('.')}.")
        common = {"split": prov["split"], "group": prov["group"], "label_origin": "replica_planted"}
        if ex["family"] == "problem_type" and f"{base}:task" not in seen:
            seen.add(f"{base}:task")
            listing = " ".join(_column(c) for c in cols[:40])
            yield {**common, "sample_id": f"{base}:task", "decision_point": "task.framing", "text": about + " " + listing,
                   "schema": {"answer": {"labels": TASKS, "prompt": "What kind of prediction is it?"}}, "options": list(TASKS),
                   "label": t["task"], "accepted": None, "rule_answer": answer.get("task")}
        if ex["family"] == "split_design" and f"{base}:time" not in seen:
            seen.add(f"{base}:time")
            options = [c["column"] for c in cols] + ["none"]
            yield {**common, "sample_id": f"{base}:time", "decision_point": "split.time_column", "text": about + " " + " ".join(_column(c) for c in cols[:40]),
                   "schema": {"answer": {"labels": options, "prompt": "Which column orders the rows in time, so that the test rows come after the "
                                                                      "training rows (none if no order matters)?"}},
                   "options": options, "label": t["time_column"] or "none", "accepted": None, "rule_answer": answer.get("time_column") or "none"}
        if ex["family"] == "leakage_judgment":
            forbidden = {str(f.get("column")) for f in answer.get("forbidden") or [] if isinstance(f, dict)}
            identifiers = {str(c) for c in answer.get("identifiers") or []}
            for c in cols:
                for point, truth_set, rule_set in (("feature.leakage_suspect", set(t["leak"]), forbidden),
                                                   ("column.is_identifier", set(t["identifier"]), identifiers)):
                    sid = f"{base}:{point}:{c['column']}"
                    if sid in seen:
                        continue
                    seen.add(sid)
                    text = _column(c) + f" The outcome predicted is {given.get('outcome')!r}; the prediction is made: " \
                                        f"{str(given.get('prediction_moment') or 'not said').rstrip('.')}."
                    yield {**common, "sample_id": sid, "decision_point": point, "text": text,
                           "schema": {"answer": {"labels": ["yes", "no"], "prompt": PROMPTS[point]}}, "options": ["yes", "no"],
                           "label": "yes" if c["column"] in truth_set else "no", "accepted": None,
                           "rule_answer": "yes" if c["column"] in rule_set else "no"}


PROVIDERS = {"lake": _lake, "workflow": _workflow, "journeys": _journeys, "selfplay": _selfplay}


def tests(source: str, lake: Path = LAKE) -> Iterator[dict[str, Any]]:
    """One test source's labelled decisions, as the model is asked them."""
    return PROVIDERS[source](lake)


def write_tests(sources: tuple[str, ...] = SOURCES, lake: Path = LAKE) -> dict[str, dict[str, int]]:
    """Each test source's labelled decisions kept as they were asked (the lake's own are its gold sets, not copied)."""
    counts: dict[str, dict[str, int]] = {}
    for source in sources:
        if source == "lake":
            continue
        root = Path(lake) / "bronze" / "decisions" / source
        shutil.rmtree(root, ignore_errors=True)  # rebuilt whole from its committed or bronze inputs
        files: dict[Path, list[str]] = defaultdict(list)
        for d in tests(source, lake):
            files[root / d["decision_point"] / f"{d['split']}.jsonl"].append(json.dumps(d, ensure_ascii=False, default=str))
        for path, lines in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(x + "\n" for x in lines), encoding="utf-8")
        counts[source] = {f"{p.parent.name}/{p.stem}": len(v) for p, v in sorted(files.items())}
    return counts


# ------------------------------------------------------------------------------------------------ the answers

def path_of(answerer: str, source: str, point: str, split: str, lake: Path = LAKE) -> Path:
    return Path(lake) / "bronze" / "answers" / answerer / source / point / f"{split}.jsonl"


def stored(answerer: str, source: str, point: str, split: str, lake: Path = LAKE) -> list[dict[str, Any]]:
    path = path_of(answerer, source, point, split, lake)
    out = []
    for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
        try:
            out.append(json.loads(line))
        except ValueError:  # a line cut by a stop mid-write: the rest stands, that decision is asked again
            continue
    return out


def migrate(lake: Path = LAKE) -> list[str]:
    """The first layout (answers/<answerer>/<point>/) moved under its source (answers/<answerer>/lake/<point>/)."""
    moved = []
    root = Path(lake) / "bronze" / "answers"
    for answerer_dir in (p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        for old in [p for p in answerer_dir.iterdir() if p.is_dir() and p.name not in SOURCES]:
            new = answerer_dir / "lake" / old.name
            new.parent.mkdir(parents=True, exist_ok=True)
            if new.exists():
                for f in old.glob("*.jsonl"):
                    with (new / f.name).open("a", encoding="utf-8") as out:
                        out.write(f.read_text(encoding="utf-8"))
                shutil.rmtree(old)
            else:
                old.rename(new)
            moved.append(old.name)
    return moved


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


def capture(answerer: str = "gliner2.5-decide", sources: tuple[str, ...] = SOURCES, points: Optional[list[str]] = None,
            splits: tuple[str, ...] = SPLITS, limit: Optional[int] = None, batch: int = 32, lake: Path = LAKE, model: Any = None,
            identity: Optional[dict[str, Any]] = None, log: Any = print) -> dict[str, Any]:
    """Every decision of every test source asked once (resumable); returns what was asked now, per source, point and split."""
    info = ANSWERERS[answerer]
    migrate(lake)
    write_tests(tuple(s for s in sources if s != "lake"), lake)
    if model is None:
        model, identity = _load(answerer)
    identity = identity or {"revision": None, "library": None, "machine": None}
    asked: dict[str, int] = {}
    for source in sources:
        by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for d in tests(source, lake):
            if (points is None or d["decision_point"] in points) and d["split"] in splits:
                by[(d["decision_point"], d["split"])].append(d)
        for (point, split), items in sorted(by.items()):
            done = {(r["sample_id"], r["input_fingerprint"]) for r in stored(answerer, source, point, split, lake)}
            todo = [d for d in items if (d["sample_id"], fingerprint(d["text"], d["schema"])) not in done]
            if limit is not None:
                todo = todo[:limit]
            path = path_of(answerer, source, point, split, lake)
            path.parent.mkdir(parents=True, exist_ok=True)
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for d in todo:  # one schema per call: options can differ from decision to decision
                groups[json.dumps(d["schema"], sort_keys=True)].append(d)
            n = 0
            for items_ in groups.values():
                schema = items_[0]["schema"]
                for i in range(0, len(items_), batch):
                    chunk = items_[i:i + batch]
                    started = time.perf_counter()
                    outs = model.batch_classify_text([d["text"] for d in chunk], schema, include_confidence=True)
                    seconds = (time.perf_counter() - started) / max(len(chunk), 1)
                    at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    lines = []
                    for d, out in zip(chunk, outs):
                        ans = (out or {}).get("answer")
                        if isinstance(ans, dict):
                            label, confidence = ans.get("label"), ans.get("confidence")
                        elif isinstance(ans, list):  # several labels (a multi-label question): each with its confidence
                            label = [a.get("label") if isinstance(a, dict) else a for a in ans]
                            confidence = [a.get("confidence") for a in ans if isinstance(a, dict)] or None
                        else:
                            label, confidence = ans, None
                        lines.append(json.dumps({
                            "sample_id": d["sample_id"], "source": source, "decision_point": point, "split": split, "group": d.get("group"),
                            "label_origin": d.get("label_origin"), "answerer": answerer, "model": info["model"], "revision": identity.get("revision"),
                            "library": identity.get("library"), "machine": identity.get("machine"), "render": RENDER,
                            "input_fingerprint": fingerprint(d["text"], d["schema"]), "input_text": d["text"], "schema": d["schema"],
                            "options": d.get("options"), "output": out, "value": value_of(label) if not isinstance(label, list) else label,
                            "confidence": confidence, "seconds": round(seconds, 4), "at": at, "licence": info["licence"], "use": info["use"]},
                            ensure_ascii=False, default=str))
                    with path.open("a", encoding="utf-8") as fh:  # appended per batch: a stop loses at most one batch
                        fh.write("".join(x + "\n" for x in lines))
                    n += len(chunk)
            asked[f"{source}/{point}/{split}"] = n
            if n:
                log(f"{answerer} {source}/{point}/{split}: {n} asked")
    return {"answerer": answerer, "asked": asked}


def _same(value: Any, label: Any, accepted: Any = None) -> bool:
    norm = lambda x: str({True: "yes", False: "no"}.get(x, x)) if isinstance(x, bool) else str(x)  # noqa: E731
    if isinstance(value, list) or isinstance(label, list):
        return sorted(map(norm, value if isinstance(value, list) else [value])) == sorted(map(norm, label if isinstance(label, list) else [label]))
    return norm(value) in {norm(a) for a in (accepted or [label])}


def summary(lake: Path = LAKE, write: bool = True) -> dict[str, Any]:
    """What was captured, per answerer, source, point and split; agreement with the true answer and with the rule on dev
    only (the sealed split is answered, never scored here)."""
    root = Path(lake) / "bronze" / "answers"
    out: dict[str, Any] = {}
    truths: dict[str, dict[str, dict[str, Any]]] = {}
    for answerer_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        per_source: dict[str, Any] = {}
        for source_dir in sorted(p for p in answerer_dir.iterdir() if p.is_dir()):
            source = source_dir.name
            if source not in truths:
                truths[source] = {d["sample_id"]: d for d in tests(source, lake) if d["split"] == "dev"} if source in PROVIDERS else {}
            per: dict[str, Any] = {}
            for point_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
                point, entry = point_dir.name, {}
                for split in [*SPLITS, *sorted({p.stem for p in point_dir.glob("*.jsonl")} - set(SPLITS))]:  # a session's answers: all.jsonl
                    rows = stored(answerer_dir.name, source, point, split, lake)
                    if not rows:
                        continue
                    conf = [c for r in rows for c in (r.get("confidence") if isinstance(r.get("confidence"), list) else [r.get("confidence")]) if isinstance(c, (int, float))]
                    secs = [r["seconds"] for r in rows if isinstance(r.get("seconds"), (int, float))]
                    e: dict[str, Any] = {"answered": len(rows), "confidence_mean": round(sum(conf) / len(conf), 4) if conf else None,
                                         "seconds_mean": round(sum(secs) / len(secs), 4) if secs else None}
                    if split == "dev":
                        pairs = [(r, truths[source][r["sample_id"]]) for r in rows if r["sample_id"] in truths[source]]
                        if pairs:
                            e["agrees_with_label"] = round(sum(_same(r["value"], d["label"], d.get("accepted")) for r, d in pairs) / len(pairs), 4)
                            ruled = [(r, d) for r, d in pairs if d.get("rule_answer") is not None]
                            if ruled:
                                e["agrees_with_rule"] = round(sum(_same(r["value"], d["rule_answer"]) for r, d in ruled) / len(ruled), 4)
                                e["rule_agrees_with_label"] = round(sum(_same(d["rule_answer"], d["label"], d.get("accepted")) for _, d in ruled) / len(ruled), 4)
                    entry[split] = e
                first = next((r for s in SPLITS for r in stored(answerer_dir.name, source, point, s, lake)[:1]), {})
                per[point] = {**entry, "model": first.get("model"), "revision": first.get("revision"), "render": first.get("render"),
                              "use": first.get("use"), "licence": first.get("licence")}
            per_source[source] = per
        out[answerer_dir.name] = per_source
    doc = {"what": "Other answerers' answers on every DCLab test, each kept whole in lake/bronze/answers/<answerer>/<source>/ (not committed; "
                   "kept permanently by make archive): the text and schema asked, the raw output, the value, the confidence, the model's "
                   "revision, the time; the true answers stay in the tests (lake/gold, lake/bronze/decisions).",
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "answerers": out,
           "rule": "Agreement is computed on dev only and says how the answerer reads these tests, not that it is ready to serve; the "
                   "sealed split is answered but never scored here."}
    if write:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return doc


def attach(rows: list[Any], lake: Path = LAKE) -> tuple[list[Any], Counter]:
    """Each lake decision sample with every captured answer on it as ``other_answers[answerer]`` (value, confidence, use,
    revision), when the answer was given on exactly this decision's state (its fingerprint); never as the label."""
    root = Path(lake) / "bronze" / "answers"
    if not root.is_dir() or not rows:
        return rows, Counter()
    migrate(lake)
    by: dict[tuple[str, str], dict[str, Any]] = {}
    for answerer_dir in (p for p in root.iterdir() if p.is_dir()):
        for point_dir in (p for p in (answerer_dir / "lake").iterdir() if p.is_dir()) if (answerer_dir / "lake").is_dir() else []:
            for split in SPLITS:
                for r in stored(answerer_dir.name, "lake", point_dir.name, split, lake):
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


def record_rules(sources: tuple[str, ...] = SOURCES, lake: Path = LAKE) -> dict[str, int]:
    """DCLab's own rules as an answerer of every test, in the same layout (``answers/dclab-rules/<source>/<point>/<split>.jsonl``):
    the rule's answer each test carries, beside the question it answered; ``use: train`` (DCLab's own). Rebuilt whole."""
    base = Path(lake) / "bronze" / "answers" / "dclab-rules"
    files: dict[Path, list[str]] = defaultdict(list)
    commit = None
    try:
        import subprocess

        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip() or None
    except Exception:  # noqa: BLE001
        commit = None
    for source in sources:
        shutil.rmtree(base / source, ignore_errors=True)
        for d in tests(source, lake):
            if d.get("rule_answer") is None:
                continue
            files[base / source / d["decision_point"] / f"{d['split']}.jsonl"].append(json.dumps({
                "sample_id": d["sample_id"], "source": source, "decision_point": d["decision_point"], "split": d["split"], "group": d.get("group"),
                "answerer": "dclab-rules", "model": "DCLab's rules", "revision": commit, "input_fingerprint": fingerprint(d["text"], d["schema"]),
                "input_text": d["text"], "schema": d["schema"], "options": d.get("options"), "value": d["rule_answer"], "confidence": None,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "licence": "DCLab's own", "use": "train"}, ensure_ascii=False, default=str))
    for path, lines in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(x + "\n" for x in lines), encoding="utf-8")
    return {str(p.relative_to(base)): len(v) for p, v in sorted(files.items())}


# ------------------------------------------------------------------------------------------------ GLiNER2's training format

def gliner2_row(d: dict[str, Any]) -> dict[str, Any]:
    """One labelled decision as a GLiNER2 training example (its tutorial's shape: ``input`` and ``output.classifications``)."""
    answer = d["schema"]["answer"]
    labels = answer["labels"]
    task: dict[str, Any] = {"task": d["decision_point"], "labels": list(labels)}
    if isinstance(labels, dict):
        task["label_descriptions"] = dict(labels)
    if answer.get("prompt"):
        task["prompt"] = answer["prompt"]
    if answer.get("multi_label"):
        task["multi_label"] = True
    label = d["label"]
    label = "yes" if label is True else "no" if label is False else label
    task["true_label"] = [str(x) for x in label] if isinstance(label, list) else [str(label)]
    return {"input": d["text"], "output": {"classifications": [task]}}


def export_gliner2(sources: tuple[str, ...] = SOURCES, lake: Path = LAKE, out: Path = EXPORT, version: str = "v1") -> dict[str, Any]:
    """Every source's labelled decisions in GLiNER2's training format, one file per source, point and split; the sealed
    decisions in files of their own (``sealed.jsonl``: evaluation, never training). Rebuilt whole."""
    root = Path(out) / version
    shutil.rmtree(root, ignore_errors=True)
    counts: dict[str, Any] = {}
    files: dict[Path, list[str]] = defaultdict(list)
    for source in sources:
        for d in tests(source, lake):
            if d.get("label") is None or str(d["label"]) not in {str(x) for x in (list(d["schema"]["answer"]["labels"]))} \
                    and not isinstance(d["label"], (list, bool)):
                counts.setdefault("refused", Counter())[f"{source}: a label outside its options"] += 1
                continue
            files[root / source / d["decision_point"] / f"{d['split']}.jsonl"].append(json.dumps(gliner2_row(d), ensure_ascii=False))
    for path, lines in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(x + "\n" for x in lines), encoding="utf-8")
    counts["files"] = {str(p.relative_to(root)): len(v) for p, v in sorted(files.items())}
    manifest = {"format": "GLiNER2 training JSONL (tutorial 8: input, output.classifications with task, labels, true_label, prompt, "
                          "label_descriptions, multi_label)", "version": version, "render": RENDER,
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "counts": {k: (dict(v) if isinstance(v, Counter) else v) for k, v in counts.items()},
                "rule": "train.jsonl may train; dev.jsonl validates; sealed.jsonl is evaluation only, never training or tuning. Labels come from "
                        "each test's own truth (construction, public datasets' marks, journeys' truth, replicas' planted facts), never from "
                        "a model's answer."}
    root.mkdir(parents=True, exist_ok=True)
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


__all__ = ["ANSWERERS", "SOURCES", "RENDER", "render", "value_of", "fingerprint", "tests", "write_tests", "capture", "summary",
           "attach", "stored", "migrate", "gliner2_row", "export_gliner2"]
