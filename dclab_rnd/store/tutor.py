"""The tutor (32.14): the students' agent under the hood. After each conductor run it reads the store's student sets
(``sets/m1_students/<point>/v1``) and, for every point with enough labelled rows, fits a student
(``dclab_core.decision_models.train``: table-grouped folds, the model chosen on them, calibrated, measured on dev beside the
rule), and keeps the whole student version under ``raw/students/``: the fitted model, its card, its training snapshot (the
records and labels it was fitted on, by hash) and its measurement. A version already kept is not fitted again; a point with
too few rows says how many it has. The tutor never promotes, routes or serves a student: that stays the owner's audited
switch; it only says, with its numbers, which students measure above the rule.
"""

from __future__ import annotations

import io
import json
from typing import Any, Optional

from .layout import Layout
from .store import Store, now

MIN_TRAIN, MIN_DEV, MIN_GROUPS = 20, 5, 2
CONFIRMED = ("person", "measured", "benchmark", "blind")


def _rows(path: Any) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        prov = ex.get("provenance") or {}
        group = ex.get("split_group") or (f"project:{prov['project']}" if prov.get("project") else f"draft:{prov['draft']}" if prov.get("draft") else ex["id"])
        rule = next((a.get("value") for a in ex.get("answers") or [] if a.get("by") == "rule"), None)
        rows.append({"record": ex["id"], "point": ex["point"], "state": ex.get("state") or {}, "options": ex.get("options") or [], "group": group,
                     "split": ex.get("split"), "value_used": rule, "label": {"value": ex.get("label"), "source": ex.get("label_source")} if ex.get("label") is not None else None,
                     "weight": ex.get("weight", 1.0)})
    return rows


def teach(layout: Optional[Layout] = None, run: Optional[str] = None) -> dict[str, Any]:
    """Every point's student fitted where its rows allow; the new versions kept whole; what each point lacks said."""
    import pickle

    from dclab_core import decision_models

    from ..agents import harness

    layout = layout or Layout()
    store = Store(layout)
    kept = {e.source for e in store.entries("students")}
    report: dict[str, Any] = {"students": {}, "waiting": {}, "at": now()}
    base = layout.sets("m1_students")
    with harness.session("tutor", run_id=run) as session:
        for folder in sorted(p for p in base.iterdir() if p.is_dir()) if base.exists() else []:
            point = folder.name
            labelled = [r for r in _rows(folder / "v1" / "train.jsonl") if r["label"]]
            confirmed = [r for r in labelled if (r["label"] or {}).get("source") in CONFIRMED]
            train_rows = confirmed if len(confirmed) >= MIN_TRAIN else labelled  # silver only when confirmed labels are too few (said)
            silver = len(train_rows) - len([r for r in train_rows if (r["label"] or {}).get("source") in CONFIRMED])
            dev_rows = [r for r in _rows(folder / "v1" / "dev.jsonl") if r["label"] and (r["label"] or {}).get("source") in CONFIRMED]  # measured on confirmed only
            groups = len({r["group"] for r in train_rows})
            if len(train_rows) < MIN_TRAIN or len(dev_rows) < MIN_DEV or groups < MIN_GROUPS:
                report["waiting"][point] = {"train": len(train_rows), "dev": len(dev_rows), "groups": groups,
                                            "needs": {"train": MIN_TRAIN, "dev": MIN_DEV, "groups": MIN_GROUPS}}
                continue
            try:
                answerer, card = decision_models.train(point, train_rows, dev_rows, limits=["fitted by the tutor on the store's student set"])
            except Exception as exc:  # noqa: BLE001 - one point's failure leaves the others
                report["waiting"][point] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                session.step(f"teach:{point}", {"train": len(train_rows)}, {"error": str(exc)[:200]}, 0.0, verdict="failed")
                continue
            prefix = f"students/{point}/{card.name}/{card.version}/"
            measurement = {"point": point, "version": card.version, "silver_in_training": silver, "dev": card.dev.model_dump(mode="json") if card.dev else None,
                           "rule_dev": card.rule_dev.model_dump(mode="json") if card.rule_dev else None,
                           # Only a rule that answered can be beaten: a point with no rule (DCLab Code's) scores the rule 0, not lower.
                           "above_rule": bool(card.dev and card.rule_dev and card.rule_dev.answered > 0 and card.dev.accuracy > card.rule_dev.accuracy),
                           "rule_answered": card.rule_dev.answered if card.rule_dev else 0, "promoted": False,
                           "why_not_promoted": "the tutor never promotes: the owner's audited switch does, on these numbers"}
            if prefix + "card.json" not in kept:
                buf = io.BytesIO()
                pickle.dump(answerer, buf)
                store.keep("students", "rnd", prefix + "model.pkl", buf.getvalue(), consent="train", note="the fitted student, whole")
                store.keep("students", "rnd", prefix + "card.json", card.model_dump_json(indent=1).encode(), consent="train")
                store.keep("students", "rnd", prefix + "training_snapshot.json", json.dumps({"trained_set": card.trained_set,
                           "records": sorted(r["record"] for r in train_rows)}).encode(), consent="train")
                store.keep("students", "rnd", prefix + "measurement.json", json.dumps(measurement, indent=1).encode(), consent="train")
                store.keep("students", "rnd", f"students/{point}/shadow.json", json.dumps({"point": point, "student": card.name, "version": card.version,
                           "level": "shadow", "model": prefix + "model.pkl", "at": now(),
                           "how": "answers in shadow where a workspace's ledger.shadow reads the store's students; never served"}).encode(),
                           consent="train")  # the point's student in shadow: its newest version
            report["students"][point] = {"version": card.version, **{k: measurement[k] for k in ("above_rule", "promoted")},
                                         "dev_accuracy": card.dev.accuracy if card.dev else None, "rule_accuracy": card.rule_dev.accuracy if card.rule_dev else None}
            session.step(f"teach:{point}", {"train": len(train_rows), "dev": len(dev_rows)}, report["students"][point], 0.0, verdict="ok")
    return report


__all__ = ["teach", "MIN_TRAIN", "MIN_DEV"]
