"""Real homework from public data (package 29.15): ``python -m dclab_rnd.real_homework fetch|build|score``.

The invented sets (``make decisions-gen``) say how a student reads planted kinds; this says how it reads real tables.
OpenML's tables, already profiled into the lake (26.2, the owner's yes on the download and the licences), give each
point a truth from the dataset's own declarations:

- ``column.is_identifier``: in a dataset that declares a row identifier, that column is one and every other declared
  feature is not; a dataset that declares none is left out (most OpenML keys are left as features, undeclared); an
  ignored column, whose role the dataset does not say, is left out, as the lake leaves it;
- ``column.semantic_role``: the row identifier, else the declared type (numeric, nominal as a category code, string
  as free text, date);
- ``task.framing``: the declared outcome's type (numeric: regression; nominal: binary with two values in the data, else
  multiclass). A numeric outcome with two values or fewer, or a nominal one with fewer than two, is left out: the
  declaration and the data disagree, and neither is a truth.

``fetch`` reads each profiled dataset's data file once more (in a temporary folder, deleted) and keeps only what the
decisions need: each column's state exactly as the invented sets build it (``datacard.card_of``, checked against the
lake's own state), DCLab's rule answers, the outcome's state (``ledger.states.task_state``) and its rule
(``studio.data.detect_task``); bands and counts, never a value, in ``data/downloads/openml/real_v1/`` (resumable).
``build`` writes ``research/llm-fine-tuning/experiments/decisions/real_v1/`` through the builder of every other set,
each table in the split the lake gives it (``dclab_core.lake.split_of``: the same table in the same split in the lake's
gold, the decisions export and here), the sealed partition in a file of its own and its fingerprint in the committed
``MANIFEST.json``. ``score`` measures, on the dev rows of both sets, the rule, the point's student (trained on invented
tables only) and a student trained on invented and real homework together (``models_real_v1/``); the sealed files are
never opened here.

Limits, said in every result (``LIMITS``): a dataset's declarations are its uploader's, not checked, and some differ from
the invented sets' definitions (an entity's key, a number used as a code, an outcome of a few whole values): a gain on the
real dev rows or a loss on the invented ones is partly that convention. Each score sits beside the commonest label's.
The lake's own OpenML samples (26.2) keep their first reading of these declarations (every dataset's columns for
identifiers, the declared levels for the task): aligning them changes the lake's gold and its export, left for later.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "downloads" / "openml"
DERIVED = CACHE / "real_v1"
DECISIONS = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions"
OUT = DECISIONS / "real_v1"
MODELS = DECISIONS / "models_real_v1"
CAMPAIGN = ROOT / "evidence" / "campaigns" / "real_homework_v1"
AT = "2026-10-19T00:00:00+00:00"  # one time for every record: the same cache builds the same files
# each point's invented homework and the student trained on it
POINTS = {"column.is_identifier": ("generated_v1", "models_v1"), "column.semantic_role": ("generated_v1", "models_v1"),
          "task.framing": ("generated_v2", "models_v2")}
TASK_OPTIONS = ["binary", "multiclass", "regression"]
LIMITS = ["A dataset's declarations are its uploader's, not checked.",
          "Identifiers: only datasets that declare a row identifier, few of them; an entity's key there is labelled not one, "
          "where the invented sets call it one.",
          "Roles: a number the uploader declared numeric is numeric, a code among them too, where the invented sets plant codes as "
          "categorical_code; an undeclared key keeps its declared type.",
          "Tasks: a numeric outcome of a few whole values (a rating, a count) is regression as declared; the invented sets plant "
          "integer codes as multiclass.",
          "GLiNER's real scores cover only the decisions it was asked with exactly this input (answered and of, beside each "
          "score): not a random part of the rows; states differing from the lake's own are counted in the set's manifest.",
          "Real names only: no hidden-name twin, as the invented sets have."]


# ------------------------------------------------------------------------------------------------ fetch

def derive(rec: dict[str, Any], frame: Any) -> dict[str, Any]:
    """What the decisions need from one dataset's table: each column's state and the rule's answers, the outcome's
    state, its rule and how many values it takes. Bands and counts only."""
    from dclab_core.bands import name_tokens
    from dclab_core.datacard import card_of

    from .ledger import column_state
    from .ledger import states as S
    from .studio.data import detect_task, profile_table

    frame = frame.rename(columns=str)
    target = rec.get("target") if isinstance(rec.get("target"), str) and "," not in str(rec.get("target")) else None
    if target is not None and target not in frame.columns:
        target = None
    name = f"openml-{rec['did']}"
    card = card_of({name: frame}, name=name, target=target)
    profile = profile_table(frame)
    by_name = {c.get("name"): c for c in profile.get("columns") or []}
    columns, differ = {}, 0
    for c in card.tables[0].columns:
        if c.name == target:
            continue
        state = {"column": c.name[:200], "name_tokens": name_tokens(c.name) or None, "dtype": c.dtype, "uniqueness": c.uniqueness,
                 "nulls": c.nulls, "cardinality": c.cardinality, "value_pattern": c.value_pattern}  # benchgen.decide's state
        state = {k: v for k, v in state.items() if v is not None}
        lake = {k: v for k, v in column_state(by_name[c.name], len(frame)).items() if v is not None} if c.name in by_name else None
        differ += lake is not None and lake != state
        columns[c.name] = {"state": state, "identifier_rule": c.purpose in ("identifier", "entity"), "role_rule": c.role}
    out: dict[str, Any] = {"did": rec["did"], "rows": int(len(frame)), "target": target, "columns": columns, "differ_from_lake": int(differ)}
    if target is not None:
        out["task"] = {"state": S.task_state(profile, target), "rule": detect_task(frame, target)["task"],
                       "values": int(frame[target].dropna().nunique())}
    return out


def fetch_one(rec: dict[str, Any], derived: Path = DERIVED) -> dict[str, Any]:
    """One dataset's derived record (written once; a later run reads it). The data file is read in a temporary folder."""
    path = Path(derived) / f"{rec['did']}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    import pandas as pd

    from .lake import openml

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        desc = openml._json(f"{openml.API}/data/{rec['did']}")["data_set_description"]
        if not desc.get("parquet_url"):
            raise ValueError("no parquet file")
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "data.parquet"
            file.write_bytes(openml._get(desc["parquet_url"], timeout=120))
            frame = pd.read_parquet(file)
        out = derive(rec, frame)
    except Exception as exc:  # noqa: BLE001 - one dataset that cannot be read costs itself only
        out = {"did": rec["did"], "failed": f"{type(exc).__name__}: {str(exc)[:200]}"}
    path.write_text(json.dumps(out) + "\n", encoding="utf-8")
    return out


def fetch(limit: Optional[int] = None, derived: Path = DERIVED, log=print) -> dict[str, int]:
    from .lake import openml

    counts: Counter = Counter()
    for i, rec in enumerate(openml.records()):
        if limit is not None and i >= limit:
            break
        had = (Path(derived) / f"{rec['did']}.json").is_file()
        got = fetch_one(rec, derived)
        counts["failed" if got.get("failed") else "derived"] += 1
        if not had:
            time.sleep(0.2)
        if i % 50 == 0:
            log(f"{i} datasets: {dict(counts)}")
    return dict(counts)


# ------------------------------------------------------------------------------------------------ build

def _task_truth(rec: dict[str, Any], derived: dict[str, Any]) -> Optional[str]:
    from dclab_core import declared

    target = next((c for c in rec.get("columns") or [] if c.get("is_target")), None)
    task = derived.get("task") or {}
    if target is None or not task:
        return None
    return declared.task(target.get("openml_type"), int(task.get("values") or 0))


def decisions_of(rec: dict[str, Any], derived: dict[str, Any]) -> tuple[list[Any], list[Any], Counter]:
    """One dataset's decisions with their declared truths (``dclab_core.declared``), recorded as the invented sets record
    theirs."""
    from dclab_core import declared
    from dclab_core.ledger import Answer, DecisionLabel, DecisionRecord

    records: list[Any] = []
    labels: list[Any] = []
    left: Counter = Counter()
    dataset = f"openml:{rec['did']}"
    target = derived.get("target")
    ignore = {s.strip() for s in str(rec.get("ignore") or "").split(",") if s.strip()}

    def add(point: str, subject: tuple[str, str], state: dict[str, Any], options: list[str], rule_id: str, rule: Any, truth: Any) -> None:
        digest = hashlib.sha256(f"{dataset}|{subject[1]}|{point}".encode()).hexdigest()[:20]
        record = DecisionRecord(id=f"rr{digest}", point=point, subject={"kind": subject[0], "key": subject[1][:200]}, state=state,
                                options=options, answers=[Answer(by="rule", answerer=rule_id, value=rule)], value_used=rule, used="rule",
                                dataset=dataset, at=AT)
        records.append(record)
        labels.append(DecisionLabel(id=f"rl{digest}", record=record.id, point=point, value=truth, source="benchmark",
                                    by="public:openml_declared", at=AT))

    marks = {str(c.get("name")): c for c in rec.get("columns") or []}
    declares_id = any(c.get("is_row_identifier") for c in rec.get("columns") or [])
    for name, col in (derived.get("columns") or {}).items():
        mark = marks.get(name)
        if mark is None or mark.get("is_target"):
            left["not a declared feature"] += 1
            continue
        if name in ignore and not mark.get("is_row_identifier"):
            left[declared.IGNORED] += 1
            continue
        state = {**col["state"], "target": (target or "")[:200]} if target else dict(col["state"])
        if declares_id:
            add("column.is_identifier", ("column", name), state, ["yes", "no"], "datacard:purposes", bool(col["identifier_rule"]),
                declared.identifier(bool(mark.get("is_row_identifier"))))
        else:
            left[declared.NO_IDENTIFIER] += 1
        role = declared.role(mark.get("openml_type"), bool(mark.get("is_row_identifier")))
        if role:
            add("column.semantic_role", ("column", name), state, ["numeric", "categorical_code", "identifier", "datetime", "free_text", "other"],
                "datacard:purposes", col["role_rule"], role)
        else:
            left[declared.NO_ROLE] += 1
    truth = _task_truth(rec, derived)
    if truth is not None:
        task = derived["task"]
        add("task.framing", ("draft", dataset), task["state"], TASK_OPTIONS, "rule:detect_task", task["rule"], truth)
    elif derived.get("task"):
        target_mark = next((c for c in rec.get("columns") or [] if c.get("is_target")), {})
        left[declared.why_no_task(target_mark.get("openml_type"))] += 1
    return records, labels, left


def build(out: Path = OUT, derived: Path = DERIVED, datasets: Optional[Any] = None, version: str = "real_v1", at: str = AT) -> dict[str, Any]:
    """``datasets``: the profiled datasets to build from (default: the lake's); ``version``, ``at``: the set's name and time
    (the collector builds real_v2 from its new tables only, so its sealed partition is fresh)."""
    from dclab_core import datasets as builders
    from dclab_core import lake as L

    from .lake import openml

    records, labels, left, tables, failed, differ = [], [], Counter(), 0, 0, 0
    for rec in (openml.records() if datasets is None else datasets):
        path = Path(derived) / f"{rec['did']}.json"
        if not path.is_file():
            continue
        got = json.loads(path.read_text(encoding="utf-8"))
        if got.get("failed"):
            failed += 1
            continue
        r, lab, why = decisions_of(rec, got)
        records += r
        labels += lab
        left.update(why)
        tables += 1
        differ += int(got.get("differ_from_lake") or 0)
    by_record: dict[str, list[dict[str, Any]]] = {}
    for label in labels:
        by_record.setdefault(label.record, []).append(json.loads(label.model_dump_json()))
    docs = [{**json.loads(r.model_dump_json()), "labels": by_record.get(r.id, [])} for r in records]
    out = Path(out)
    sets: dict[str, Any] = {}
    for point in sorted({d["point"] for d in docs}):
        rows, manifest = builders.decisions(docs, point, consent="public data under an open licence (OpenML)")
        for x in rows:  # the lake's split of the table, so a table sits in one split everywhere it is used
            x["split"] = L.split_of(x["group"].removeprefix("dataset:"))
        manifest = builders.refit(manifest, rows, note="Each table in the split the lake gives it (dclab_core.lake.split_of: 15% sealed, 15% dev).")
        manifest = manifest.model_copy(update={"policy": "One group per OpenML dataset; whole groups to sealed (15%), dev (15%) or train by the "
                                                         "lake's hash (dclab_core.lake.split_of), the same split as in the lake's gold. The "
                                                         "sealed partition is a file of its own, never used for tuning."})
        folder = out / "decisions" / point
        folder.mkdir(parents=True, exist_ok=True)
        sealed = [x for x in rows if x["split"] == "sealed"]
        for file, items in (("rows.jsonl", [x for x in rows if x["split"] != "sealed"]), ("sealed.jsonl", sealed)):
            (folder / file).write_text("".join(json.dumps(x, ensure_ascii=False, default=str) + "\n" for x in items), encoding="utf-8")
        (folder / "manifest.json").write_text(manifest.model_dump_json(indent=1) + "\n", encoding="utf-8")
        right = sum(1 for x in rows if x["label"] and x["value_used"] == x["label"]["value"])
        positive = sum(1 for x in rows if x["label"] and x["label"]["value"] is True)
        content = sorted(json.dumps({k: x[k] for k in ("record", "state", "label")}, sort_keys=True, default=str) for x in sealed)
        sets[point] = {"decisions": len(rows), "tables": len({x["group"] for x in rows}), "splits": manifest.splits, "rule_right": right,
                       **({"label_yes": positive} if point == "column.is_identifier" else {}),
                       "sealed_fingerprint": hashlib.sha256("\n".join(content).encode()).hexdigest()}
    summary = {"version": version, "at": at, "source": "OpenML (26.2's profiled datasets, open licences), each table's data file read once more",
               "grouping": "one group per OpenML dataset; its split from dclab_core.lake.split_of, the same as in the lake's gold",
               "tables": tables, "failed": failed, "decisions": len(docs), "left_out": dict(left),
               "states_differing_from_the_lake": differ, "sets": sets, "limits": LIMITS,
               "rule": "Real tables, each truth the dataset's own declaration: scores say how an answerer reads real columns as their "
                       "uploaders declared them. The sealed partition is never used for tuning."}
    out.mkdir(parents=True, exist_ok=True)
    (out / "MANIFEST.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


# ------------------------------------------------------------------------------------------------ score

def _rows(folder: Path, point: str) -> list[dict[str, Any]]:
    """A set's training and dev rows (``rows.jsonl``); the sealed file is not opened."""
    path = Path(folder) / "decisions" / point / "rows.jsonl"
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []
    return [r for r in rows if r.get("split") != "sealed" and (r.get("label") or {}).get("value") is not None]


def _student(folder: Path, point: str) -> Any:
    from .ledger.shadow import _load

    return next((m for m in _load(Path(folder)) if m.point == point), None)


def _score(model: Any, point: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from dclab_core.decision_models import metrics

    truth = [r["label"]["value"] for r in rows]
    groups = [r["group"] for r in rows]
    if model is None:
        answers = [r.get("value_used") for r in rows]
    else:
        answers = []
        for r in rows:
            got = model.answer(point, r["state"], list(r.get("options") or []))
            answers.append(None if got is None else got.value)
    score = json.loads(metrics(truth, answers, groups=groups).model_dump_json())
    return {k: score.get(k) for k in ("n", "accuracy", "accuracy_ci", "answered", "precision", "precision_ci", "recall", "recall_ci", "per_class")
            if score.get(k) not in (None, {})}


def baseline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The rows' labels counted, and the score of always answering the commonest one: every score sits beside it."""
    from dclab_core.answerers import _key

    counts = Counter(_key(r["label"]["value"]) for r in rows)
    common, n = counts.most_common(1)[0] if counts else (None, 0)
    return {"classes": dict(counts), "commonest": common, "commonest_accuracy": round(n / len(rows), 4) if rows else None}


GLINER = "gliner2.5-decide"
SUFFIX = {"column.is_identifier": "id", "column.semantic_role": "role"}  # the lake's sample ids of OpenML columns (26.2)


def gliner(point: str, rows: list[dict[str, Any]], lake: Optional[Path] = None) -> Optional[list[Any]]:
    """GLiNER2.5-Decide's captured answers (26.9) to these real decisions, where it was asked exactly this: the lake asked
    it the same OpenML columns, and an answer counts only when its input (the text rendered from the state, and the
    options) is this decision's; None for a point whose lake state differs (the task's)."""
    from dclab_core import lake as L

    from .lake import answers

    if point not in SUFFIX:
        return None
    kw = {"lake": lake} if lake is not None else {}
    got = {a["sample_id"]: (a.get("input_fingerprint"), a.get("value")) for split in ("train", "dev", "sealed")
           for a in answers.stored(GLINER, "lake", point, split, **kw)}
    did = lambda r: r["group"].removeprefix("dataset:openml:")  # noqa: E731
    out = []
    for r in rows:
        found = got.get(f"openml{did(r)}:{L.fingerprint(str((r.get('subject') or {}).get('key')))[:12]}:{SUFFIX[point]}")
        asked = answers.fingerprint(*answers.render(point, r["state"], list(r.get("options") or []))) if found else None
        out.append(found[1] if found and found[0] == asked else None)
    return out


def train_mixed(point: str, invented: list[dict[str, Any]], real: list[dict[str, Any]], models: Path = MODELS, fit=None) -> tuple[Any, Any]:
    """The point's student trained on invented and real homework together, chosen as the generation's were (table-grouped
    folds of the training rows), its card and fitted file in ``models_real_v1/``."""
    import joblib

    from dclab_core.decision_models import train

    from .decision_models_build import record_fitted

    rows = [r for r in invented if r["split"] == "train"] + [r for r in real if r["split"] == "train"]
    dev = [r for r in real if r["split"] == "dev"]
    limits = ["Trained on invented tables (planted kinds) and real OpenML tables (their declared truths); measured here on the real "
              "tables' dev rows. The two sets' definitions differ in places (real_homework.LIMITS): not evidence about a project's data."]
    model, card = (fit or train)(point, rows, dev, limits=limits)
    models = Path(models)
    models.mkdir(parents=True, exist_ok=True)
    (models / f"{point}.json").write_text(card.model_dump_json(indent=1) + "\n", encoding="utf-8")
    joblib.dump(model, models / f"{point}.joblib")
    record_fitted(models)
    return model, card


def score(points: Iterable[str] = tuple(POINTS), real: Path = OUT, decisions: Path = DECISIONS, models: Path = MODELS,
          fit=None, log=print, lake: Optional[Path] = None) -> dict[str, Any]:
    """Each point's rule, its student and the student with real homework, on the dev rows of the invented set and of the
    real one."""
    out: dict[str, Any] = {}
    for point in points:
        invented_folder, student_folder = POINTS[point]
        invented = _rows(Path(decisions) / invented_folder, point)
        reals = _rows(Path(real), point)
        if not invented or not reals:
            out[point] = {"skipped": "no invented set" if not invented else "no real set"}
            continue
        started = time.perf_counter()
        student = _student(Path(decisions) / student_folder, point)
        mixed, card = train_mixed(point, invented, reals, models, fit)
        devs = {"invented": [r for r in invented if r["split"] == "dev"], "real": [r for r in reals if r["split"] == "dev"]}
        entry: dict[str, Any] = {"homework": {"invented_train": sum(r["split"] == "train" for r in invented),
                                              "real_train": sum(r["split"] == "train" for r in reals)},
                                 "student": f"{student.name}@{student.version}" if student else None,
                                 "with_real_homework": f"{card.name}@{card.version}", "family": card.family}
        for name, rows in devs.items():
            entry[name] = {"tables": len({r["group"] for r in rows}), "baseline": baseline(rows), "rule": _score(None, point, rows),
                           "student": _score(student, point, rows) if student else None, "with_real_homework": _score(mixed, point, rows)}
        said = gliner(point, devs["real"], lake)
        if said is not None and any(a is not None for a in said):  # over the decisions it answered, its coverage beside it
            idx = [i for i, a in enumerate(said) if a is not None]
            from dclab_core.decision_models import metrics

            sc = metrics([devs["real"][i]["label"]["value"] for i in idx], [said[i] for i in idx], groups=[devs["real"][i]["group"] for i in idx])
            entry["real"]["gliner"] = {"answered": len(idx), "of": len(said), "accuracy": sc.accuracy, "accuracy_ci": sc.accuracy_ci}
        entry["seconds"] = round(time.perf_counter() - started, 1)
        out[point] = entry
        log(f"{point}: invented dev rule {entry['invented']['rule']['accuracy']} student {(entry['invented']['student'] or {}).get('accuracy')} "
            f"real-homework {entry['invented']['with_real_homework']['accuracy']} (commonest {entry['invented']['baseline']['commonest_accuracy']}) | "
            f"real dev rule {entry['real']['rule']['accuracy']} student {(entry['real']['student'] or {}).get('accuracy')} "
            f"real-homework {entry['real']['with_real_homework']['accuracy']} (commonest {entry['real']['baseline']['commonest_accuracy']})")
    return {"points": out, "limits": LIMITS}


def write(doc: dict[str, Any], campaign: Path = CAMPAIGN) -> Path:
    from datetime import datetime, timezone

    results = Path(campaign) / "results"
    results.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in results.glob("RHW-*.json")] or [0])
    ident = f"RHW-{n:03d}"
    doc = {"id": ident, "experiment_id": ident, "campaign_id": "real_homework_v1", "kind": "real_homework", "status": "completed",
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **doc,
           "rule": "Each point's rule, its student (trained on invented tables) and a student trained on invented and real homework, "
                   "scored on the dev rows of the invented set and of the real set (OpenML tables, truths as declared); the sealed "
                   "partitions unread. Evidence about reading declared columns of public tables, not about any project's data."}
    path = results / f"{ident}_real_v1.json"
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.real_homework", description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("fetch", "build", "score"))
    parser.add_argument("--limit", type=int, default=None, help="fetch: at most this many datasets")
    parser.add_argument("--write", action="store_true", help="score: store the result in evidence/campaigns/real_homework_v1")
    args = parser.parse_args(argv)
    if args.command == "fetch":
        print(json.dumps(fetch(args.limit), indent=1))
    elif args.command == "build":
        summary = build()
        print(json.dumps({k: summary[k] for k in ("tables", "failed", "decisions", "left_out", "states_differing_from_the_lake")}, indent=1))
        for point, s in summary["sets"].items():
            print(f"{point:<24} {s['decisions']:>6} decisions, {s['tables']:>5} tables, {s['splits']}, rule right {s['rule_right']}")
    else:
        doc = score()
        if args.write:
            print(write(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
