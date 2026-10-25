"""The understanding students with more homework (S8 B2 of the 2026-10-25 plan; the owner: "the student model must have
more ability"): ``python -m dclab_rnd.understand_students collect|train|report``.

The four workflow points of the first conversation (``problem.family``, ``problem.ready``, ``workflow.next_question``,
``solution.route``) failed their exam (26.3) on tiny sets of written people the rule reads right 95% of the time. This
adds the 600 people of ``journeys_homework_v1.json`` (S8 B1, written through the gateway, each person's truth the
code's) and trains two kinds of student:

- **State students**, as 26.3's: a classifier per point on the ledger's state (counts, flags and codes, never the
  person's words), ``dclab_core.decision_models.train``.
- **A text student** for ``problem.family``: the 26.3 gate's own note says a state student cannot read the family
  (the state holds no words; it scored 0.526 against the rules' 0.895). This one reads the written person's words and
  answers (invented, so they may be read), character n-grams and a logistic regression. In the product it would read the
  person's own words in shadow, as the rule does.

Splits: one per person, the store's (``dclab_core.datasets.split_of`` of ``dataset:journey:<set>:<id>``), its sealed
share used as dev (never trained on): the plays' kept records (S8 A2) carry the same key, so the store and these sets
agree. The older journeys' rows come from ``workflow_v1`` (not played again); the new people are played once, at seed 0,
on the rules alone, each play kept whole. The sealed exam (``journeys_understand_sealed.json``, written by a separate
agent, pinned by hash) is read once, by a gate fixed before it, in a later command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from . import workflow_models as W

ROOT = Path(__file__).resolve().parents[1]
SETS = W.DECISIONS / "understand_v1"
MODELS = W.DECISIONS / "understand_models_v1"
HOMEWORK = W.JOURNEYS / "journeys_homework_v1.json"
SEEDS = (0,)


def person_split(group: str) -> str:
    """A person's split, the store's: its sealed share is dev here (measured on, never trained on)."""
    from dclab_core.datasets import split_of

    split = split_of(f"dataset:journey:{group}"[:88])
    return "dev" if split == "sealed" else split


def homework() -> list[tuple[str, dict[str, Any]]]:
    """The new people, each saying "I don't know" to what they cannot answer (they were written with answers left out on
    purpose; S8 B3); the sealed people are played the same way."""
    return [("homework_v1", {**j, "dont_know": True}) for j in json.loads(HOMEWORK.read_text(encoding="utf-8"))] if HOMEWORK.is_file() else []


def items() -> list[tuple[str, dict[str, Any]]]:
    return W.journeys() + homework()


def _resplit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**r, "split": person_split(r["group"])} for r in rows]


def collect(workers: int = 3, out: Path = SETS) -> dict[str, Any]:
    """The new people played once (kept whole), the older rows merged from workflow_v1, every row in its person's split."""
    os.environ.setdefault("DCLAB_JOURNEY_PLAYS", str(W.PLAYS))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    played = W._played([(s, j, seed) for seed in SEEDS for s, j in homework()], workers)
    raw = [r for p in played for r in p["rows"]]
    old = []
    for point in _points():
        old += [{k: v for k, v in r.items() if k not in ("label", "split", "value_used")} for r in W.rows(point)]
    (out / "raw.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in _resplit(raw + old)), encoding="utf-8")
    (out / "PLAYED.json").write_text(json.dumps({"seeds": list(SEEDS), "new_people": len(homework()), "plays": len(played),
                                                 "failed_plays": sorted({f"{p['source']}:{p['id']}" for p in played if p["error"]}),
                                                 "older_rows_from": "workflow_v1", "older_rows": len(old)}, indent=1) + "\n", encoding="utf-8")
    manifest = W.relabel(out, items())
    manifest.update(version="understand_v1", split_rule="one split per person: dclab_core.datasets.split_of('dataset:journey:<set>:<id>'), its sealed share dev")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def _points() -> tuple[str, ...]:
    from dclab_core import workflow_labels as wl

    return tuple(wl.POINTS)


def _texts() -> dict[str, str]:
    """Each person's words and answers, by group (``<set>:<id>``)."""
    out = {}
    for source, j in items():
        answers = " ".join(str(v) for v in (j.get("answers") or {}).values())
        out[f"{source}:{j['id']}"] = f"{j.get('words', '')} || {answers}"
    return out


def text_student(rows: list[dict[str, Any]]) -> tuple[Any, dict[str, Any]]:
    """problem.family from the person's words: one example per person (its family label), trained on train people,
    measured on dev people beside the rule's answer on the same rows."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    texts = _texts()
    labelled = [r for r in rows if r.get("label") and r["group"] in texts]
    by_person: dict[str, dict[str, Any]] = {}
    for r in labelled:
        by_person.setdefault(r["group"], r)
    train = [r for r in by_person.values() if r["split"] == "train"]
    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
                          LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced"))
    model.fit([texts[r["group"]] for r in train], [str(r["label"]["value"]) for r in train])
    dev = [r for r in labelled if r["split"] == "dev"]
    picks = model.predict([texts[r["group"]] for r in dev]) if dev else []
    right = sum(str(p) in [str(a) for a in r["label"]["accepted"]] for p, r in zip(picks, dev))
    rule = sum(str(r["value_used"]) == str(r["label"]["value"]) for r in dev)
    return model, {"point": "problem.family", "kind": "text", "train_people": len(train), "dev_rows": len(dev),
                   "dev_people": len({r["group"] for r in dev}), "student_dev_accuracy": round(right / len(dev), 4) if dev else None,
                   "rule_dev_accuracy": round(rule / len(dev), 4) if dev else None,
                   "features": "character n-grams (2 to 5) of the person's words and answers, tf-idf; logistic regression, balanced"}


def train(sets: Path = SETS, out: Path = MODELS) -> dict[str, Any]:
    """A state student per point (26.3's trainer) and the text student for problem.family; their dev figures beside
    the rule's on the same rows (``MEASURED.json``). The sealed people are not read here."""
    import joblib

    from dclab_core.decision_models import train as fit

    from .decision_models_build import record_fitted

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    measured, skipped = {}, {}
    for point in _points():
        data = [r for r in W.rows(point, sets) if r.get("label")]
        train_rows, dev_rows = [r for r in data if r["split"] == "train"], [r for r in data if r["split"] == "dev"]
        try:
            model, card = fit(point, train_rows, dev_rows, limits=W.LIMITS)
        except ValueError as e:
            skipped[point] = str(e)
            continue
        (out / f"{point}.json").write_text(card.model_dump_json(indent=1) + "\n", encoding="utf-8")
        joblib.dump(model, out / f"{point}.joblib")
        measured[point] = {"kind": "state", "train_rows": len(train_rows), "dev_rows": len(dev_rows),
                           "student_dev_accuracy": card.dev.accuracy if card.dev else None, "student_dev_ci": card.dev.accuracy_ci if card.dev else None,
                           "rule_dev_accuracy": card.rule_dev.accuracy if card.rule_dev else None,
                           "rule_dev_ci": card.rule_dev.accuracy_ci if card.rule_dev else None, "family": card.family}
    rows = W.rows("problem.family", sets)
    if rows:
        model, m = text_student(rows)
        joblib.dump(model, out / "problem.family.text.joblib")
        measured["problem.family:text"] = m
    record_fitted(out)
    doc = {"sets": str(Path(sets).relative_to(ROOT)), "measured": measured, "skipped": skipped,
           "rule_on_new_people": rule_on_new_people(sets), "limits": W.LIMITS + [
               "The new people are written by Claude through the gateway (S8 B1): invented, varied on purpose; a student may learn the "
               "writer's style; the sealed people are written by a separate agent to test that."]}
    (out / "MEASURED.json").write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    return doc


def rule_on_new_people(sets: Path = SETS) -> dict[str, Any]:
    """The plan's 'too easy' check: the rule's accuracy per point on the new people alone (it was 95% on the old ones)."""
    out = {}
    for point in _points():
        new = [r for r in W.rows(point, sets) if r.get("label") and r.get("source") == "homework_v1"]
        old = [r for r in W.rows(point, sets) if r.get("label") and r.get("source") != "homework_v1"]
        acc = lambda xs: round(sum(str(r["value_used"]) == str(r["label"]["value"]) for r in xs) / len(xs), 4) if xs else None  # noqa: E731
        out[point] = {"new_rows": len(new), "rule_on_new": acc(new), "old_rows": len(old), "rule_on_old": acc(old)}
    return out


SEALED_FILE, SEALED_HASH = W.JOURNEYS / "journeys_understand_sealed.json", W.JOURNEYS / "sealed_understand.sha256"
CAMPAIGN = ROOT / "evidence" / "campaigns" / "understand_v1"
MARK = "sealed_read.json"  # written before the sealed set is opened: a crash afterwards still spends it
GATE = {"version": "understand_gate_v1", "decision_gate": "decision_gate_v1", "seed": 5,
        "set": "journeys_understand_sealed.json (150 people, ten per kind, written by a separate agent told only the format, the kinds "
               "and their columns, never the homework or its writer's prompt; pinned by sealed_understand.sha256)",
        "played": "seed 5, on the rules alone, each person saying \"I don't know\" to what they cannot answer, as the homework was played",
        "rule": "Each point's state student, and the text student for problem.family, judged once by decision_gate_v1 (the column "
                "points' thresholds, unchanged) beside the rule on the same sealed decisions, by person, an answer right when the "
                "person's journey accepts it; a model that does not pass stays in shadow.",
        # written after the dev figures (plays of the fixed product, 2026-10-25) and before the sealed people were read
        "seen_before": "Dev, each beside the rule on the same rows: problem.family text student 0.966 against the rule's 0.821 (235 "
                       "decisions); problem.family state student 0.498 (the state holds no words): expected to fail; problem.ready "
                       "0.887 [0.844, 0.920] against 0.827: its accuracy's lower bound is under 0.90, expected to fail; "
                       "workflow.next_question 0.876 [0.859, 0.891] against 0.855: expected to fail on the same bar; solution.route "
                       "0.820 against 0.826: no margin, expected to fail. The text student is the one candidate; its risk is the "
                       "writer's style (the homework was written by one writer, the sealed people by another), and the rule's right "
                       "answers it would turn wrong were not measured on dev. The rule on the new people: 0.71 (family), 0.71 (ready), "
                       "0.80 (next question), 0.84 (route), against 0.95, 0.96, 0.99 and 0.89 on the older ones."}


def spent(campaign: Path = CAMPAIGN) -> bool:
    fingerprint = SEALED_HASH.read_text(encoding="utf-8").strip()
    mark = Path(campaign) / MARK
    return mark.is_file() and json.loads(mark.read_text(encoding="utf-8")).get("fingerprint") == fingerprint


def gate(models: Path = MODELS, campaign: Path = CAMPAIGN, workers: int = 3) -> list[Path]:
    """The sealed people played once (their plays never kept), every student judged on them beside the rule. The set is
    marked read before it is opened; a person whose play failed fails every point."""
    import hashlib

    import joblib

    from dclab_core import decision_gate
    from dclab_core import workflow_labels as wl

    if not GATE["seen_before"]:
        raise ValueError("the gate's dev figures are not written: the gate is fixed (and committed) before the sealed set is read")
    if spent(campaign):
        raise ValueError("the understanding students' sealed people were read: a sealed set is read once")
    fingerprint = SEALED_HASH.read_text(encoding="utf-8").strip()
    if hashlib.sha256(SEALED_FILE.read_bytes()).hexdigest() != fingerprint:
        raise ValueError("the sealed people changed since they were sealed: a sealed set is never edited")
    models = Path(models)
    loaded = {p: (json.loads((models / f"{p}.json").read_text(encoding="utf-8")), joblib.load(models / f"{p}.joblib"))
              for p in _points() if (models / f"{p}.json").is_file()}
    text = joblib.load(models / "problem.family.text.joblib") if (models / "problem.family.text.joblib").is_file() else None
    campaign = Path(campaign)
    campaign.mkdir(parents=True, exist_ok=True)
    (campaign / MARK).write_text(json.dumps({"fingerprint": fingerprint, "at": W.datetime.now(W.timezone.utc).isoformat(timespec="seconds"),
                                             "models": {p: c["version"] for p, (c, _) in loaded.items()}}, indent=1) + "\n", encoding="utf-8")
    os.environ.pop("DCLAB_JOURNEY_PLAYS", None)  # a sealed person's play is never kept where the store would read it
    people = [("understand_sealed", {**j, "dont_know": True}) for j in json.loads(SEALED_FILE.read_text(encoding="utf-8"))]  # as the homework is played
    played = W._played([(s, j, GATE["seed"]) for s, j in people], workers)
    labelled = W.label_rows([r for p in played for r in p["rows"]], W.truths(people))
    failed = sorted({f"{p['source']}:{p['id']}" for p in played if p.get("error")})
    texts = {f"{s}:{j['id']}": f"{j.get('words', '')} || " + " ".join(str(v) for v in (j.get("answers") or {}).values()) for s, j in people}
    judged = []
    students = [(p, c, m) for p, (c, m) in sorted(loaded.items())] + ([("problem.family", {"name": "problem.family.text", "version": "text_v1"}, text)] if text else [])
    for point, card, model in students:
        rows = [r for r in labelled if r["point"] == point and r.get("label")]
        answers, errors = [], 0
        for r in rows:
            label = wl.Label(r["label"]["value"], tuple(r["label"].get("accepted") or [r["label"]["value"]]), "")
            try:
                got = model.predict([texts[r["group"]]])[0] if card["name"].endswith(".text") else getattr(model.answer(point, r["state"], list(r.get("options") or [])), "value", None)
                answers.append(None if got is None else wl.canonical(got, label))
            except Exception:  # noqa: BLE001 - an error is counted, never skipped
                answers.append(None)
                errors += 1
        verdict = decision_gate.verdict([r["label"]["value"] for r in rows], answers, [r["value_used"] for r in rows], [r["group"] for r in rows],
                                        errors=errors) if rows else {"pass": False, "criteria": [], "note": "no labelled sealed decision"}
        if failed:
            verdict = {**verdict, "pass": False, "criteria": [*verdict.get("criteria", []), {"criterion": "every sealed person played",
                       "value": f"{len(played) - len(failed)}/{len(played)}", "need": "all", "pass": False}]}
        judged.append((point, card, rows, verdict))
    folder = campaign / "results"
    folder.mkdir(parents=True, exist_ok=True)
    if not (campaign / "manifest.json").exists():
        (campaign / "manifest.json").write_text(json.dumps({"campaign_id": "understand_v1", "objective": "S8 B2: the understanding students "
                                                            "with more homework (600 written people), state students and a text student, judged once on "
                                                            "150 sealed people written apart.", "gate": GATE, "rule": GATE["rule"],
                                                            "command": "python -m dclab_rnd.understand_students gate --yes"}, indent=2) + "\n", encoding="utf-8")
    paths = []
    for point, card, rows, verdict in judged:
        n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in folder.glob("UND-*.json")] or [0])
        ident = f"UND-{n:03d}"
        doc = {"id": ident, "experiment_id": ident, "campaign_id": "understand_v1", "kind": "understand_student_gate", "status": "completed",
               "at": W.datetime.now(W.timezone.utc).isoformat(timespec="seconds"), "point": point, "student": card["name"], "version": card["version"],
               "gate": {**GATE, "thresholds": decision_gate.THRESHOLDS},
               "sealed": {"decisions": len(rows), "people": len({r["group"] for r in rows}), "fingerprint": fingerprint, "plays": len(played),
                          "failed_plays": failed},
               "verdict": verdict, "result": "pass" if verdict["pass"] else "fail",
               "rule": f"One run of {card['name']} on {len(rows)} sealed decisions of written people, judged by a gate fixed before it; evidence "
                       "about these written conversations with no model, not about real sessions nor production readiness."}
        path = folder / f"{ident}_{card['name']}.json"
        path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


# ------------------------------------------------------------------------------------------------ v2: the first message
# (S8 B2 v2, the owner's "do all", 2026-10-25): the family read from the person's first message alone, where the rules read
# nothing for 76% of the new people and are wrong on 10%. Three answerers on the same people: the rules, the text student
# alone, and the text student only where the rules read nothing (the rules' right answers kept by construction). Judged
# once on a second sealed set (journeys_understand_sealed_v2.json, a third writer: channels such as a WhatsApp message, a
# form field, a voice note), under decision_gate_v1 unchanged, the gate fixed and committed first.
HOMEWORK_V2 = W.JOURNEYS / "journeys_homework_v2.json"
SEALED_V2_FILE, SEALED_V2_HASH = W.JOURNEYS / "journeys_understand_sealed_v2.json", W.JOURNEYS / "sealed_understand_v2.sha256"
MODELS_V2 = W.DECISIONS / "understand_models_v2"
CAMPAIGN_V2 = ROOT / "evidence" / "campaigns" / "understand_v2"
POINT_V2 = "problem.family_from_words"
GATE_V2 = {"version": "understand_gate_v2", "decision_gate": "decision_gate_v1", "point": POINT_V2,
           "set": "journeys_understand_sealed_v2.json (150 people, ten per kind, a third writer, pinned by sealed_understand_v2.sha256)",
           "answerers": "the rules (dclab_core.conversation.card_for on the first message); the text student alone; the text student only "
                        "where the rules read no family",
           "rule": "Each of the two student answerers judged once by decision_gate_v1 (the thresholds unchanged) beside the rules on the "
                   "same sealed first messages, by person, an answer right when it is one of the person's families.",
           "seen_before": None}  # the dev figures, written before the sealed set is read (committed first)


def _families(j: dict[str, Any]) -> list[str]:
    f = j.get("family")
    return [str(x) for x in (f if isinstance(f, list) else [f]) if x]


def first_words_rows(people: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    """One row per person: the first message, the families it may be (the truth), the rules' reading (None: unread)."""
    from dclab_core.conversation import card_for

    rows = []
    for source, j in people:
        families = _families(j)
        if not families or not str(j.get("words") or "").strip():
            continue
        group = f"{source}:{j['id']}"
        rule = card_for(str(j["words"]), {}).family.value
        rows.append({"group": group, "words": str(j["words"]), "accepted": families, "label": families[0], "rule": rule,
                     "split": person_split(group), "source": source})
    return rows


def homework_v2_people() -> list[tuple[str, dict[str, Any]]]:
    out = homework()
    if HOMEWORK_V2.is_file():
        out += [("homework_v2", j) for j in json.loads(HOMEWORK_V2.read_text(encoding="utf-8"))]
    return out


def _right(pick: Any, row: dict[str, Any]) -> bool:
    return pick is not None and str(pick) in row["accepted"]


def text_model_v2(train_rows: list[dict[str, Any]]) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
                          LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced"))
    return model.fit([r["words"] for r in train_rows], [r["label"] for r in train_rows])


def answers_v2(model: Any, rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    picks = list(model.predict([r["words"] for r in rows])) if rows else []
    return {"rule": [r["rule"] for r in rows], "student": picks,
            "student_where_rule_unread": [r["rule"] if r["rule"] is not None else p for r, p in zip(rows, picks)]}


def train_v2(out: Path = MODELS_V2) -> dict[str, Any]:
    """The text student on the first messages of the homework people (train split), measured on the dev people beside the
    rules; the sealed people are never read here."""
    import joblib

    rows = first_words_rows(homework_v2_people())
    train_rows, dev = [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "dev"]
    model = text_model_v2(train_rows)
    got = answers_v2(model, dev)
    acc = {k: round(sum(_right(a, r) for a, r in zip(v, dev)) / len(dev), 4) if dev else None for k, v in got.items()}
    unread = [i for i, r in enumerate(dev) if r["rule"] is None]
    rule_right = [i for i, r in enumerate(dev) if _right(r["rule"], r)]
    doc = {"point": POINT_V2, "train_people": len(train_rows), "dev_people": len(dev), "by_source": dict(Counter(r["source"] for r in rows)),
           "dev_accuracy": acc, "rule_unread_share": round(len(unread) / len(dev), 4) if dev else None,
           "student_on_unread": round(sum(_right(got["student"][i], dev[i]) for i in unread) / len(unread), 4) if unread else None,
           "student_turns_rule_right": round(sum(not _right(got["student"][i], dev[i]) for i in rule_right) / len(rule_right), 4) if rule_right else None,
           "features": "character n-grams (2 to 5) of the first message only, tf-idf; logistic regression, balanced"}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / f"{POINT_V2}.text.joblib")
    (out / "MEASURED.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def gate_v2(models: Path = MODELS_V2, campaign: Path = CAMPAIGN_V2) -> list[Path]:
    """The sealed v2 people's first messages read once; both student answerers judged beside the rules."""
    import hashlib

    import joblib

    from dclab_core import decision_gate

    if not GATE_V2["seen_before"]:
        raise ValueError("the v2 gate's dev figures are not written: it is fixed (and committed) before the sealed set is read")
    fingerprint = SEALED_V2_HASH.read_text(encoding="utf-8").strip()
    campaign = Path(campaign)
    mark = campaign / MARK
    if mark.is_file() and json.loads(mark.read_text(encoding="utf-8")).get("fingerprint") == fingerprint:
        raise ValueError("the v2 sealed people were read: a sealed set is read once")
    if hashlib.sha256(SEALED_V2_FILE.read_bytes()).hexdigest() != fingerprint:
        raise ValueError("the v2 sealed people changed since they were sealed")
    model = joblib.load(Path(models) / f"{POINT_V2}.text.joblib")
    campaign.mkdir(parents=True, exist_ok=True)
    mark.write_text(json.dumps({"fingerprint": fingerprint, "at": W.datetime.now(W.timezone.utc).isoformat(timespec="seconds")}, indent=1) + "\n",
                    encoding="utf-8")
    rows = first_words_rows([("understand_sealed_v2", j) for j in json.loads(SEALED_V2_FILE.read_text(encoding="utf-8"))])
    got = answers_v2(model, rows)
    truth = [r["label"] for r in rows]
    canon = {k: [r["label"] if _right(a, r) else a for a, r in zip(v, rows)] for k, v in got.items()}  # an accepted family reads as the label
    groups = [r["group"] for r in rows]
    folder = campaign / "results"
    folder.mkdir(parents=True, exist_ok=True)
    if not (campaign / "manifest.json").exists():
        (campaign / "manifest.json").write_text(json.dumps({"campaign_id": "understand_v2", "objective": "S8 B2 v2: the family from the first "
                                                            "message, the text student alone and where the rules read none, judged once.",
                                                            "gate": GATE_V2, "rule": GATE_V2["rule"],
                                                            "command": "python -m dclab_rnd.understand_students gate-v2 --yes"}, indent=2) + "\n",
                                                 encoding="utf-8")
    paths = []
    for n, name in enumerate(("student", "student_where_rule_unread"), start=1):
        verdict = decision_gate.verdict(truth, canon[name], canon["rule"], groups)
        ident = f"UNW-{n:03d}"
        doc = {"id": ident, "experiment_id": ident, "campaign_id": "understand_v2", "kind": "understand_student_gate", "status": "completed",
               "at": W.datetime.now(W.timezone.utc).isoformat(timespec="seconds"), "point": POINT_V2, "student": name,
               "gate": {**GATE_V2, "thresholds": decision_gate.THRESHOLDS},
               "sealed": {"decisions": len(rows), "people": len(rows), "fingerprint": fingerprint,
                          "rule_unread": sum(r["rule"] is None for r in rows)},
               "verdict": verdict, "result": "pass" if verdict["pass"] else "fail",
               "rule": f"One run of the {name.replace('_', ' ')} on {len(rows)} sealed first messages of written people, judged by a gate "
                       "fixed before it; evidence about these written messages, not about real sessions nor production readiness."}
        path = folder / f"{ident}_{name}.json"
        path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dclab_rnd.understand_students", description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("collect", "train", "report", "gate", "train-v2", "gate-v2"))
    ap.add_argument("--yes", action="store_true", help="gate: read the sealed people (once: they are then spent)")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(argv)
    if args.command == "collect":
        m = collect(args.workers)
        for point, s in m["sets"].items():
            print(f"{point:<24} {s['labelled']:>5} labelled, {s['people']} people, rule right {s['rule_right']}, {s['by_split']}")
        return 0
    if args.command == "train":
        print(json.dumps(train(), indent=1, default=str)[:4000])
        return 0
    if args.command == "train-v2":
        print(json.dumps(train_v2(), indent=1))
        return 0
    if args.command == "gate-v2":
        if not args.yes:
            print("Would read the v2 sealed people's first messages once; add --yes (the set is then spent).")
            return 0
        for path in gate_v2():
            doc = json.loads(path.read_text(encoding="utf-8"))
            print(f"{doc['student']:<28} {doc['result']:<4} " + "  ".join(f"{c['criterion'].split(',')[0]}={c['value']}" for c in doc["verdict"].get("criteria", [])))
        return 0
    if args.command == "gate":
        if not args.yes:
            print("Would play the sealed people once and judge every student on them; add --yes (the set is then spent).")
            return 0
        for path in gate(workers=args.workers):
            doc = json.loads(path.read_text(encoding="utf-8"))
            print(f"{doc['student']:<32} {doc['result']:<4} " + "  ".join(f"{c['criterion'].split(',')[0]}={c['value']}" for c in doc["verdict"].get("criteria", [])))
        return 0
    print(json.dumps(json.loads((MODELS / "MEASURED.json").read_text(encoding="utf-8")), indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
