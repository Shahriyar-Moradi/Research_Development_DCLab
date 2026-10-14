"""The decision classifiers in shadow (package 22.4): each one answers beside the rule at its decision point, at level 0.

Every decision the ledger records at a point that has a fitted classifier (22.2, ``make decision-models``: the fitted
models stay on this machine, in ``models_v1/<point>.joblib``, each unpickled only when its bytes match the SHA-256 in
the committed ``fitted.json``) carries the classifier's answer as a shadow: recorded, never used, never an answer of the
record (``DecisionRecord.answer``, labels and exports set it aside). A decision whose state holds no band (only a
name) gets no shadow: the classifier would answer from nothing. Inside ``ledger.batch()`` the shadows of every record
are answered together, one prediction per point. A shadow that differs from the rule puts the decision in the blind
draw's "disagree" stratum (the most drawn), and the blind labelling page shows those first.

``measure`` counts, per classifier: the decisions at its point, how many it answered, its agreement with the rule, its
disagreements, and its accuracy on the labelled ones beside the rule's, with the labels by source and blind labels
weighted by their draw (an estimate per stratum stays possible). On a page it reads the shadows already recorded; a run
measuring new models answers again.

``replicas`` builds a ledger from every replica's project run by the engine (each planted trap a benchmark label): not
a replay of recorded sessions, which ``recorded`` measures (the current workspace's own ledger). ``python -m
dclab_rnd.ledger.shadow replicas --write`` stores the measurement in ``evidence/campaigns/decision_shadow_v1``.

``DCLAB_DECISION_SHADOW=0`` turns shadows off (the test runner does, unless a test turns them on).
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "research/llm-fine-tuning/experiments/decisions/models_v1"
RESULTS = ROOT / "evidence/campaigns/decision_shadow_v1/results"
GATE = ROOT / "evidence/campaigns/decision_models_v1/results"
BANDS = ("dtype", "uniqueness", "nulls", "cardinality", "value_pattern")  # a state with none of these holds only a name


def on() -> bool:
    return os.environ.get("DCLAB_DECISION_SHADOW", "1") not in ("0", "false", "off")


@functools.lru_cache(maxsize=4)
def answerers(models: str = str(MODELS)) -> tuple[Any, ...]:
    """The fitted classifiers whose bytes match ``fitted.json`` and whose card describes them; none when none is here."""
    try:
        import joblib
    except ImportError:
        return ()
    folder = Path(models)
    try:
        hashes = json.loads((folder / "fitted.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    out = []
    for card_path in sorted(folder.glob("*.json")):
        fitted = card_path.with_suffix(".joblib")
        if card_path.name == "fitted.json" or not fitted.exists():
            continue
        try:
            if hashlib.sha256(fitted.read_bytes()).hexdigest() != hashes.get(card_path.stem):
                continue  # not the file that was built: never unpickled
            card = json.loads(card_path.read_text(encoding="utf-8"))
            model = joblib.load(fitted)
        except Exception:  # noqa: BLE001 - a model that cannot be read answers nothing
            continue
        if getattr(model, "version", None) == card.get("version") and getattr(model, "point", None) == card.get("point"):
            model.name = card.get("name") or model.name
            out.append(model)
    return tuple(out)


def warm() -> None:
    """Load the classifiers off the request path (the first load takes seconds)."""
    if on():
        threading.Thread(target=answerers, name="dclab-shadow-warm", daemon=True).start()


def banded(state: dict[str, Any]) -> bool:
    return any(state.get(k) is not None for k in BANDS)


def _answers(model: Any, records: list[Any]) -> list[Any]:
    """The classifier's answers for many records of its point, in one prediction: as ``ClassifierAnswerer.answer`` gives
    them one by one (renormalised over each decision's options; None below its confidence or with no option it knows)."""
    from dclab_core.answerers import _key, _option, features
    from dclab_core.ledger import Answer

    if model.model is None or not records:
        return [None] * len(records)
    probas = model.model.predict_proba([features(r.state, model.exclude) for r in records])
    classes = list(model.model.classes_)
    out = []
    for r, proba in zip(records, probas):
        allowed = {_key(_option(o)) for o in r.options} if r.options else set(classes)
        keep = [i for i, c in enumerate(classes) if c in allowed]
        if not keep:
            out.append(None)
            continue
        total = float(sum(proba[i] for i in keep)) or 1.0
        best = max(keep, key=lambda i: proba[i])
        confidence = float(proba[best]) / total
        out.append(None if confidence < model.abstain_below else
                   Answer(by="classifier", answerer=f"{model.name}@{model.version}", value=model._values.get(classes[best], classes[best]),
                          raw=round(confidence, 6), confidence=round(confidence, 4)))
    return out


def add_many(records: list[Any], models: Optional[Iterable[Any]] = None) -> list[Any]:
    """The records with their classifiers' shadow answers, one prediction per point (never used: level and ``used``
    stay). A record whose state holds no band gets none."""
    from dclab_core.answerers import SHADOW
    from dclab_core.ledger import POINTS, answer_problem

    if models is None and not on():
        return records
    models = list(answerers() if models is None else models)
    if not models:
        return records
    extra: dict[int, list[Any]] = {}
    for m in models:
        name = f"{m.name}@{m.version}"
        mine = [(i, r) for i, r in enumerate(records) if r.point in m.points and banded(r.state) and not any(a.answerer == name for a in r.answers)]
        try:
            got = _answers(m, [r for _, r in mine])
        except Exception:  # noqa: BLE001 - a model that fails answers nothing; the decisions are recorded all the same
            continue
        for (i, r), a in zip(mine, got):
            if a is not None and not answer_problem(POINTS[r.point], a.value):
                extra.setdefault(i, []).append(a.model_copy(update={"source": SHADOW}))
    out = []
    for i, r in enumerate(records):
        if i in extra:
            r = type(r).model_validate({**r.model_dump(), "answers": [a.model_dump() for a in [*r.answers, *extra[i]][:20]]})
        out.append(r)
    return out


def add(record: Any, models: Optional[Iterable[Any]] = None) -> Any:
    return add_many([record], models)[0]


def gate_results() -> dict[str, str]:
    """Each classifier's verdict from its sealed gate (22.3), by ``name@version``."""
    out = {}
    for path in sorted(GATE.glob("DMG-*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        out[f"{doc['model']['name']}@{doc['model']['version']}"] = doc.get("result")
    return out


def _recorded(record: Any, name: str) -> Any:
    return next((a for a in record.answers if a.answerer == name), None)  # a shadow, or an answer a person was shown (22.8)


def measure(records: Iterable[Any], labels: Iterable[Any], models: Optional[Iterable[Any]] = None, answer: bool = False) -> dict[str, Any]:
    """Per classifier (``name@version``): decisions at its point, answered, agreement with the rule (its first answer
    with a value), disagreements, and accuracy on the labelled disagreements beside the rule's, with the labels' sources
    and the blind ones weighted by their draw. ``answer``: answer again where no shadow of this version is recorded
    (a run measuring models); otherwise only the recorded shadows count (a page)."""
    from dclab_core.answerers import RecordedRule, _key
    from dclab_core.ledger import current_labels, effective_label

    models = list(answerers() if models is None else models)
    records = list(records)
    by_record: dict[str, list[Any]] = {}
    for x in current_labels(list(labels)):  # a label another one supersedes (on another record too) no longer counts
        by_record.setdefault(x.record, []).append(x)
    gates = gate_results()
    rule_of = RecordedRule()
    out: dict[str, Any] = {}
    for m in models:
        name = f"{m.name}@{m.version}"
        mine = [r for r in records if r.point in m.points]
        missing = [r for r in mine if _recorded(r, name) is None and banded(r.state)] if answer else []
        fresh = dict(zip((r.id for r in missing), _answers(m, missing))) if missing else {}
        row = {"point": next(iter(m.points)), "gate": gates.get(name), "decisions": len(mine), "without_bands": sum(not banded(r.state) for r in mine),
               "answered": 0, "with_rule": 0, "agree": 0, "disagree": 0, "labelled_disagreements": 0, "right_on_disagreements": 0,
               "rule_right_on_disagreements": 0, "weighted_disagreements": 0.0, "weighted_right": 0.0, "weighted_rule_right": 0.0,
               "labels_by_source": {}, "labelled": 0, "right": 0}
        for r in mine:
            got = _recorded(r, name) or fresh.get(r.id)
            if got is None or got.value is None:
                continue
            row["answered"] += 1
            rule = rule_of.of(r)
            truth = effective_label(r, by_record.get(r.id, []))
            if truth is not None:
                row["labelled"] += 1
                row["right"] += _key(got.value) == _key(truth.value)
            if rule is None or rule.value is None:
                continue
            row["with_rule"] += 1
            if _key(got.value) == _key(rule.value):
                row["agree"] += 1
                continue
            row["disagree"] += 1
            if truth is not None:
                weight = float(truth.weight or 1.0) if truth.source == "blind" else 1.0
                row["labelled_disagreements"] += 1
                row["labels_by_source"][truth.source] = row["labels_by_source"].get(truth.source, 0) + 1
                row["right_on_disagreements"] += _key(got.value) == _key(truth.value)
                row["rule_right_on_disagreements"] += _key(rule.value) == _key(truth.value)
                row["weighted_disagreements"] += weight
                row["weighted_right"] += weight * (_key(got.value) == _key(truth.value))
                row["weighted_rule_right"] += weight * (_key(rule.value) == _key(truth.value))
        share = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
        row.update(agreement=share(row["agree"], row["with_rule"]), coverage=share(row["answered"], row["decisions"]),
                   accuracy_on_disagreements=share(row["right_on_disagreements"], row["labelled_disagreements"]),
                   rule_accuracy_on_disagreements=share(row["rule_right_on_disagreements"], row["labelled_disagreements"]),
                   weighted_accuracy_on_disagreements=share(row["weighted_right"], row["weighted_disagreements"]),
                   accuracy=share(row["right"], row["labelled"]))
        for k in ("weighted_disagreements", "weighted_right", "weighted_rule_right"):
            row[k] = round(row[k], 4)
        out[name] = row
    return out


def replicas(keys: Optional[list[str]] = None, seed: int = 0) -> tuple[list[Any], list[Any], list[str]]:
    """Every replica (or ``keys``) built into a project and run by the engine with a ledger of its own: its records and
    labels (the planted traps as benchmark labels). The records are written without shadows (they are answered when
    measured), so the runs measure whatever classifiers are fitted now."""
    import tempfile
    from unittest import mock

    from dclab_rnd import kinds
    from dclab_rnd.expansion import runner
    from dclab_rnd.kinds import acceptance
    from dclab_rnd.models.gateway import private_scope

    from . import _INSTALLED, FileLedger, install

    done = []
    previous = _INSTALLED.get("ledger")
    with tempfile.TemporaryDirectory(prefix="dclab-shadow-") as tmp:
        ledger = FileLedger(Path(tmp) / "ledger", "shadow-replicas")
        install(ledger)
        quiet = mock.patch.dict(os.environ, {"DCLAB_DECISION_SHADOW": "0"})
        quiet.start()
        try:
            for kind in kinds.all_kinds():
                if keys and kind.key not in keys:
                    continue
                case = kind.case(seed)
                if case is None or not getattr(kind, "runs_in_engine", True):
                    continue
                with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
                    acceptance.engine_run(case, kind.key)
                done.append(kind.key)
        finally:
            quiet.stop()
            install(previous)
        return ledger.records(), ledger.labels(), done


def write(result: dict[str, Any], kinds_run: list[str], decisions: int, labels: int, source: str) -> Path:
    from datetime import datetime, timezone

    RESULTS.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in RESULTS.glob("DSH-*.json")] or [0])
    ident = f"DSH-{n:03d}"
    doc = {"id": ident, "experiment_id": ident, "campaign_id": "decision_shadow_v1", "kind": "decision_shadow", "status": "completed",
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": source, "replicas": kinds_run, "decisions": decisions,
           "labels": labels, "classifiers": result,
           "rule": "Each classifier answered in shadow beside the rule; labels are the replicas' planted traps (benchmark) unless the "
                   "source is a recorded ledger. Evidence about these decisions, not about production readiness; nothing a shadow "
                   "answered was used."}
    path = RESULTS / f"{ident}_{source}.json"
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.ledger.shadow")
    parser.add_argument("command", choices=["replicas", "recorded"], help="replicas: run every replica; recorded: this workspace's ledger")
    parser.add_argument("--kinds", nargs="*", default=None)
    parser.add_argument("--write", action="store_true", help="store the measurement as evidence")
    args = parser.parse_args(argv)
    if not answerers():
        print("no fitted decision classifier on this machine (or none matching fitted.json): run make decisions-gen and make decision-models")
        return 2
    if args.command == "replicas":
        records, labels, done = replicas(args.kinds)
    else:
        from . import current_ledger

        store = current_ledger()
        if store is None:
            print("no ledger for this workspace")
            return 2
        records, labels, done = store.records(), store.labels(), []
    result = measure(records, labels, answer=True)
    print(json.dumps({"source": args.command, "replicas": done, "decisions": len(records), "labels": len(labels), "classifiers": result}, indent=1))
    if args.write:
        print(write(result, done, len(records), len(labels), args.command))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("DCLAB_NO_LIVE_MODELS", "1")
    raise SystemExit(main())
