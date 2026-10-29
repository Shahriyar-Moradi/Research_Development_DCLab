"""The intent classifier (package P6 of the 2026-10-29 plan): the first candidate for the 7th model, in shadow only.

It reads a person's first request to Home and names the kind of help (a track of ``draft.intent``) with a calibrated
confidence. It is the text model of ``understand_students`` v2 (``text_model_v2``: character 2–5-gram tf-idf with
balanced logistic regression), Platt-calibrated (sigmoid, one class against the rest) on folds grouped by writer batch,
with a ``ModelCard`` of ``dclab_core.decision_models``.

- **Trained on written labels only** (``agent_eval/home/intent_requests_v1.json``: the requests whose plain reading
  agreed with their code-assigned kind; the flagged ones never). Written labels are kept apart from real ones.
- **Measured across writers**: trained on writer A and measured on writer B, then the reverse, so it cannot pass by
  learning one writer's style; per track and per language, beside the rules (``intent.read`` with no model) on the same
  requests. Also measured on Home's own requests (dev and the two spent sealed sets, dev material since they were run).
- **The gate** (``intent_gate_v1.json``) is written from the dev figures before any sealed read; the sealed set
  (``intent_sealed_v1.json``, 180 requests by an independent writer, pinned by SHA-256) is read once by :func:`gate`.
- It answers nothing in the product: the shadow wiring (``ledger/capture.py``) and the answerer order (own model →
  Claude → rules) belong to other packages, and it may answer alone only after passing its gate.

``python -m dclab_rnd.intent_classifier train`` measures and writes the card; ``write-gate`` fixes the gate; ``gate
--yes`` reads the sealed set once (not run in P6).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
HOME_EVAL = ROOT / "dclab_rnd" / "agent_eval" / "home"
REQUESTS = HOME_EVAL / "intent_requests_v1.json"
MODELS = HOME_EVAL / "intent_model_v1"
GATE_FILE = HOME_EVAL / "intent_gate_v1.json"
SEALED = HOME_EVAL / "intent_sealed_v1.json"
SEALED_SHA = HOME_EVAL / "intent_sealed_v1.sha256"
CAMPAIGN = ROOT / "evidence" / "campaigns" / "intent_model_v1"
MARK = "sealed_read.json"  # written before the sealed set is opened: a crash afterwards still spends it
POINT = "request.intent"
NAME = "request.intent.text_v2"
TRACKS = ("prediction", "analysis", "preparation", "reporting", "business", "advice")
TARGET_PRECISION = 0.95  # the dev precision the "answer alone" confidence threshold is chosen for
MIN_ALONE = 20  # a threshold is chosen only where at least this many dev answers are above it

GATE = {
    "version": "intent_gate_v1",
    "set": "intent_sealed_v1: 180 first requests, 30 per kind of help (main track), written by an independent agent told only "
           "the format and the six kinds; read once",
    "right": "the classifier's track equals the request's main track (tracks[0]); a two-kind request is right only on its main one",
    # the owner (2026-10-29): "for now you can skip persian". The gate judges the English requests (about 108 of 180); the
    # Persian and mixed ones are measured and reported, never gating, and a non-English request is never answered alone
    "scope": "en",
    # every number below was set after the dev measurements (MEASURED.json) and before any sealed read
    "accuracy_min": 0.75, "accuracy_ci_low_min": 0.66,
    "per_track_recall_min": 0.60,
    "margin_over_rules_min": 0.10,  # paired, on the same English requests: the rules are intent.read with no model, its version recorded
    # set after seeing dev (cross-writer 0.19-0.22, Home's requests 0.10) and before any sealed read: the precision above the
    # threshold, gated directly below, is what answering alone needs; this bounds how far the confidences may drift
    "calibration_error_max": 0.15,
    # answering alone is the safety bar: 95% right above the threshold; about 20 answers above it on ~108 requests, so the
    # interval's floor of 0.80 asks for nearly all of them right. Coverage is a usefulness bar: a model that answers 15% of
    # first requests alone at that precision already spares Claude those calls
    "alone": {"precision_min": 0.95, "precision_ci_low_min": 0.80, "coverage_min": 0.15,
              "threshold": "the stricter of two dev thresholds, each the smallest confidence with 95% right above it: the held-out "
                           "writers' English answers, and Home's 72 English requests (another writer); never the sealed set"},
    "reported_not_gating": ["Persian and mixed requests (the owner skipped Persian for now)", "segments (about 8 requests: too few "
                            "to gate)", "each style", "the rules' default share"],
    "pass_means": "it may answer a first request alone above the threshold, with Claude as the fallback, once P1 wires the "
                  "answerer order; written labels alone never promote it (plan decision 3): only this independent set does",
}


# ------------------------------------------------------------------------------------------------ rows and the model

def load_rows(path: Path = REQUESTS) -> list[dict[str, Any]]:
    """The kept written requests as training rows (the flagged file is never read here)."""
    if path.name.endswith("_flagged.json"):
        raise ValueError("flagged requests are never trained on")
    out = []
    for r in json.loads(Path(path).read_text(encoding="utf-8")):
        if r.get("track") not in TRACKS:
            continue
        out.append({"id": r["id"], "words": r["text"], "label": r["track"], "tracks": r.get("tracks") or [r["track"]],
                    "lang": r.get("lang"), "style": r.get("style"), "need": r.get("need"), "writer": r.get("writer"),
                    "group": _group(r), "source": r.get("label_source", "written")})
    return out


def _group(r: dict[str, Any]) -> str:
    """A writer's batch number across the kinds ("a-3"): one writer batch holds one kind only, so folds grouped by batch
    would leave whole kinds out of a fold (measured 2026-10-29: calibration error 0.33); grouped by number, a fold keeps
    a writer's sitting together and holds every kind."""
    batch = str(r.get("batch") or "")
    parts = batch.split("-")
    return "-".join(parts[1:]) if len(parts) == 3 and parts[0] in TRACKS else (batch or r["id"])


def _pipeline() -> Any:
    """``understand_students.text_model_v2``'s pipeline, unfitted (its parameters, never a copy of them)."""
    from sklearn.base import clone

    from dclab_rnd.understand_students import text_model_v2

    probe = [{"words": "predict who leaves next month", "label": "prediction"}, {"words": "predict churn next month", "label": "prediction"},
             {"words": "average sales per region", "label": "analysis"}, {"words": "average sales per month", "label": "analysis"}]
    return clone(text_model_v2(probe))


def fit(rows: list[dict[str, Any]], calibrate: bool = True, folds: int = 5) -> Any:
    """The text model on ``rows``; calibrated with Platt scaling on folds grouped by writer batch (a batch never sits in
    the fold it is calibrated on)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GroupKFold

    if any(r.get("split") == "sealed" for r in rows):
        raise ValueError("sealed rows are never trained on")
    words, labels = [r["words"] for r in rows], [r["label"] for r in rows]
    pipe = _pipeline()
    if not calibrate:
        return pipe.fit(words, labels)
    groups = [r["group"] for r in rows]
    splits = list(GroupKFold(n_splits=min(folds, len(set(groups)))).split(words, labels, groups))
    return CalibratedClassifierCV(pipe, method="sigmoid", cv=splits).fit(words, labels)


def answer(model: Any, texts: list[str]) -> list[tuple[str, float]]:
    if not texts:
        return []
    proba = model.predict_proba(texts)
    classes = list(model.classes_)
    return [(str(classes[int(p.argmax())]), round(float(p.max()), 4)) for p in proba]


def rules_track(texts: list[str]) -> list[dict[str, Any]]:
    """The rules' reading of each request (``intent.read`` with no model): its track and whether it was a default."""
    from dclab_rnd.draft import intent

    out = []
    for t in texts:
        r = intent.read(t)
        out.append({"track": r.get("track"), "source": r.get("source"), "status": r.get("status")})
    return out


def _rules_version() -> str:
    from dclab_rnd.draft import intent

    return str(getattr(intent, "RULES_VERSION", "unknown"))


# ------------------------------------------------------------------------------------------------ measuring

def measure(model: Any, rows: list[dict[str, Any]], with_rules: bool = True) -> dict[str, Any]:
    """Accuracy (a pick right when it is the main track) with Wilson intervals, per track (recall and precision), per
    language, per style, on the segments requests and two-kind requests, the calibration error; the rules beside it."""
    from dclab_core.decision_models import calibration_error, metrics, wilson

    texts = [r["words"] for r in rows]
    got = answer(model, texts)
    truth = [r["label"] for r in rows]
    picks = [g[0] for g in got]
    right = [p == t for p, t in zip(picks, truth)]
    score = metrics(truth, picks, groups=[r.get("group") for r in rows])

    def acc(idx: list[int], flags: list[bool]) -> dict[str, Any]:
        k = sum(flags[i] for i in idx)
        return {"n": len(idx), "accuracy": round(k / len(idx), 4) if idx else None, "ci": list(wilson(k, len(idx)))}

    def by(key: str, flags: list[bool]) -> dict[str, Any]:
        return {str(v): acc([i for i, r in enumerate(rows) if r.get(key) == v], flags) for v in sorted({str(r.get(key)) for r in rows}) if v != "None"}

    out: dict[str, Any] = {
        "n": len(rows), "accuracy": score.accuracy, "accuracy_ci": score.accuracy_ci,
        "macro_precision": score.precision, "macro_recall": score.recall,
        "per_track": {t: {"n": score.per_class.get(t, {}).get("n", 0), "recall": score.per_class.get(t, {}).get("recall"),
                          "recall_ci": score.per_class.get(t, {}).get("recall_ci"), "precision": score.per_class.get(t, {}).get("precision")}
                      for t in TRACKS},
        "by_lang": by("lang", right), "by_style": by("style", right),
        "segments": acc([i for i, r in enumerate(rows) if r.get("need") == "segments"], right),
        "two_kinds_any_track": acc([i for i, r in enumerate(rows) if len(r.get("tracks") or []) > 1],
                                   [p in (r.get("tracks") or [r["label"]]) for p, r in zip(picks, rows)]),
        "calibration_error": calibration_error([g[1] for g in got], right),
        "confusions": dict(Counter(f"{t}->{p}" for t, p in zip(truth, picks) if t != p).most_common(12)),
    }
    if with_rules:
        rules = rules_track(texts)
        r_right = [x["track"] == t for x, t in zip(rules, truth)]
        rs = metrics(truth, [x["track"] for x in rules], groups=[r.get("group") for r in rows])
        out["rules"] = {"version": _rules_version(), "accuracy": rs.accuracy, "accuracy_ci": rs.accuracy_ci,
                        "per_track_recall": {t: rs.per_class.get(t, {}).get("recall") for t in TRACKS},
                        "by_lang": by("lang", r_right), "default_share": round(sum(x["source"] == "default" for x in rules) / len(rows), 4) if rows else None,
                        "paired": {"model_only_right": sum(a and not b for a, b in zip(right, r_right)),
                                   "rules_only_right": sum(b and not a for a, b in zip(right, r_right))}}
    out["_answers"] = [{"id": r["id"], "pick": p, "confidence": c, "right": ok} for r, (p, c), ok in zip(rows, got, right)]
    return out


def choose_threshold(answers: list[dict[str, Any]], target: float = TARGET_PRECISION, min_n: int = MIN_ALONE) -> dict[str, Any]:
    """The smallest confidence at which the answers above it are right at least ``target`` of the time (at least
    ``min_n`` of them); None when no threshold reaches it."""
    for step in range(30, 100):
        t = step / 100
        above = [a for a in answers if a["confidence"] >= t]
        if len(above) >= min_n and sum(a["right"] for a in above) / len(above) >= target:
            return {"threshold": t, "coverage": round(len(above) / len(answers), 4), "precision": round(sum(a["right"] for a in above) / len(above), 4),
                    "n_above": len(above)}
    return {"threshold": None, "coverage": 0.0, "precision": None, "n_above": 0}


def home_rows() -> list[dict[str, Any]]:
    """Home's own written requests: dev and the two spent sealed sets (dev material since HOME-011 and HOME-013 ran them)."""
    from dclab_rnd.agent_eval import home

    out = []
    for split in ("dev", "sealed", "sealed_v2"):
        for c in home.load(split):
            tracks = list(c["gold"]["tracks"])
            out.append({"id": f"{split}:{c['id']}", "words": c["text"], "label": tracks[0], "tracks": tracks, "lang": c.get("lang"),
                        "style": None, "need": None, "group": split, "source": f"home_{split}"})
    return out


def _strip(doc: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in doc.items() if not k.startswith("_")}


def train(requests: Path = REQUESTS, out: Path = MODELS) -> dict[str, Any]:
    """The cross-writer measurements (A→B, B→A), the dev threshold, then the model on every kept request with its card;
    the fitted model is saved (not committed), the card and the measurements are."""
    import joblib

    from dclab_core.decision_models import ModelCard, Score, calibration_error, metrics

    rows = load_rows(requests)
    writers = sorted({r["writer"] for r in rows})
    by_writer = {w: [r for r in rows if r["writer"] == w] for w in writers}
    cross = {}
    for train_w, test_w in (("a", "b"), ("b", "a")):  # the pair the plan asks for
        model = fit(by_writer[train_w])
        cross[f"train_{train_w}_test_{test_w}"] = measure(model, by_writer[test_w])
    if len(writers) > 2:  # with writer C: each writer held out, trained on the others; every row answered once
        loo = {}
        for w in writers:
            others = [x for x in writers if x != w]
            loo[f"train_{''.join(others)}_test_{w}"] = measure(fit([r for r in rows if r["writer"] != w]), by_writer[w])
        cross.update(loo)
        held = list(loo)
        test_rows = [r for w in writers for r in by_writer[w]]
    else:
        held = ["train_b_test_a", "train_a_test_b"]
        test_rows = by_writer["a"] + by_writer["b"]
    both = [a for k in held for a in cross[k]["_answers"]]
    alone = choose_threshold(both)
    alone_en = choose_threshold([a for a, r in zip(both, test_rows) if r["lang"] == "en"])
    final = fit(rows)
    home = measure(final, home_rows())
    home_en = [a for a, r in zip(home["_answers"], home_rows()) if r["lang"] == "en"]
    alone_home_en = choose_threshold(home_en, min_n=10)  # 72 requests: at least 10 above
    plain = fit(rows, calibrate=False)
    features = len(plain.steps[0][1].vocabulary_)
    trained = sorted(f"{r['id']}={r['label']}" for r in rows)
    trained_set = hashlib.sha256("\n".join(trained).encode()).hexdigest()
    version = hashlib.sha256(f"{NAME}|{trained_set}".encode()).hexdigest()[:12]
    truth = [r["label"] for r in test_rows]
    picks = [a["pick"] for a in both]
    groups = [r["group"] for r in test_rows]
    rules = rules_track([r["words"] for r in test_rows])
    by_lang: dict[str, Score] = {}
    for lang in sorted({str(r["lang"]) for r in test_rows}):
        idx = [i for i, r in enumerate(test_rows) if str(r["lang"]) == lang]
        by_lang[f"written:{lang}"] = metrics([truth[i] for i in idx], [picks[i] for i in idx], groups=[groups[i] for i in idx])
    by_lang["home_dev_and_spent_sealed"] = metrics([r["label"] for r in home_rows()], [a["pick"] for a in home["_answers"]])
    card = ModelCard(point=POINT, name=NAME, version=version, family="logistic", strength=4.0, features=features, fields=["words"],
                     trained_on=len(rows), trained_set=trained_set, groups=len({r["group"] for r in rows}),
                     label_sources=dict(Counter(r["source"] for r in rows)),
                     chosen_by="fixed before any measurement: understand_students.text_model_v2 (character 2-5-gram tf-idf, balanced "
                               "logistic regression, C=4); no selection",
                     oof_accuracy={k: v["accuracy"] for k, v in cross.items()},
                     calibration="Platt scaling (sigmoid, one class against the rest) on 5 folds grouped by writer batch number (each fold holds every kind)",
                     dev=metrics(truth, picks, groups=groups), rule_dev=metrics(truth, [x["track"] for x in rules], groups=groups),
                     dev_by_source=by_lang, dev_calibration_error=calibration_error([a["confidence"] for a in both], [a["right"] for a in both]),
                     limits=["Trained and measured on written requests (a model wrote them from code-assigned kinds): not real people.",
                             "Dev is the other writer's requests (cross-writer) and Home's 108 written requests; the independent sealed "
                             "set is read once, under intent_gate_v1.",
                             "Requests a plain model reading disagreed with were left out, so the hardest requests are under-represented.",
                             "Shadow only: it answers nothing in the product until it passes its gate and P1 wires the answerer order."])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out / f"{NAME}.joblib")
    (out / "card.json").write_text(card.model_dump_json(indent=1) + "\n", encoding="utf-8")
    doc = {"point": POINT, "name": NAME, "version": version, "trained_on": len(rows),
           "by_writer": {w: len(v) for w, v in by_writer.items()},
           "cross_writer": {k: _strip(v) for k, v in cross.items()}, "held_out": held, "alone_on_dev": alone,
           "alone_on_dev_english": alone_en, "alone_on_home_english": alone_home_en,
           "held_out_by_lang": {l: {"n": len(idx), "accuracy": round(sum(both[i]["right"] for i in idx) / len(idx), 4)}
                                for l in sorted({str(r["lang"]) for r in test_rows})
                                for idx in [[i for i, r in enumerate(test_rows) if str(r["lang"]) == l]]},
           "home_requests": _strip(home), "rules_version": _rules_version(),
           "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "MEASURED.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return doc


# ------------------------------------------------------------------------------------------------ the gate

def fingerprint(path: Path = SEALED) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_gate(measured: Path = MODELS / "MEASURED.json", path: Path = GATE_FILE) -> dict[str, Any]:
    """The gate, fixed from the dev figures before any sealed read: written once, never rewritten."""
    if Path(path).is_file():
        raise SystemExit(f"{path.name} is already written; a gate is fixed once")
    m = json.loads(Path(measured).read_text(encoding="utf-8"))
    found = [m.get("alone_on_dev_english") or m["alone_on_dev"], m.get("alone_on_home_english") or {"threshold": None}]
    threshold = None if any(f["threshold"] is None for f in found) else max(f["threshold"] for f in found)  # none found: it never answers alone
    doc = {**GATE, "alone": {**GATE["alone"], "threshold": threshold, "dev_held_out_writers_english": found[0], "dev_home_english": found[1]},
           "sealed_sha256": SEALED_SHA.read_text(encoding="utf-8").strip(), "model_version": m["version"],
           "seen_before": {k: {"accuracy": v["accuracy"], "rules": v["rules"]["accuracy"], "calibration_error": v["calibration_error"]}
                           for k, v in m["cross_writer"].items()} | {"home_requests": {"accuracy": m["home_requests"]["accuracy"],
                                                                                        "rules": m["home_requests"]["rules"]["accuracy"]}},
           "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    Path(path).write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return doc


def spent(campaign: Path = CAMPAIGN) -> bool:
    return (Path(campaign) / MARK).is_file()


def gate(campaign: Path = CAMPAIGN, models: Path = MODELS, sealed: Path = SEALED, sealed_sha: Path = SEALED_SHA,
         gate_file: Path = GATE_FILE) -> dict[str, Any]:
    """The sealed set read once under the written gate: refused when already read, when the gate is not written, or when
    the set is not the pinned one. Not run in P6."""
    import joblib

    from dclab_core.decision_models import wilson

    if spent(campaign):
        raise SystemExit("the sealed set was already read; it is read once")
    if not Path(gate_file).is_file():
        raise SystemExit("write the gate first (write-gate): thresholds are fixed before the sealed read")
    g = json.loads(Path(gate_file).read_text(encoding="utf-8"))
    if fingerprint(sealed) != Path(sealed_sha).read_text(encoding="utf-8").strip() or g["sealed_sha256"] != fingerprint(sealed):
        raise SystemExit("the sealed set is not the pinned one: refused")
    Path(campaign).mkdir(parents=True, exist_ok=True)
    (Path(campaign) / MARK).write_text(json.dumps({"read_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "gate": g["version"]}) + "\n",
                                       encoding="utf-8")
    model = joblib.load(Path(models) / f"{NAME}.joblib")
    rows = [{"id": c["id"], "words": c["text"], "label": c["tracks"][0], "tracks": c["tracks"], "lang": c.get("lang"), "style": c.get("style"),
             "need": c.get("need"), "group": c["id"], "split": "sealed"} for c in json.loads(Path(sealed).read_text(encoding="utf-8"))]
    every = measure(model, rows)  # every language, reported
    rows = [r for r in rows if g.get("scope") in (None, "all") or r["lang"] == g["scope"]]
    m = measure(model, rows)  # the gate's scope
    t = g["alone"]["threshold"]
    above = [a for a in m["_answers"] if t is not None and a["confidence"] >= t]
    k = sum(a["right"] for a in above)
    checks = {
        "accuracy": (m["accuracy"] or 0) >= g["accuracy_min"],
        "accuracy_ci_low": (m["accuracy_ci"][0] or 0) >= g["accuracy_ci_low_min"],
        "per_track_recall": all((m["per_track"][tr]["recall"] or 0) >= g["per_track_recall_min"] for tr in TRACKS),
        "margin_over_rules": (m["accuracy"] or 0) - (m["rules"]["accuracy"] or 0) >= g["margin_over_rules_min"],
        "calibration": m["calibration_error"] is not None and m["calibration_error"] <= g["calibration_error_max"],
        "alone_precision": bool(above) and k / len(above) >= g["alone"]["precision_min"],
        "alone_precision_ci_low": bool(above) and (wilson(k, len(above))[0] or 0) >= g["alone"]["precision_ci_low_min"],
        "alone_coverage": len(above) / len(rows) >= g["alone"]["coverage_min"],
    }
    result = {"gate": g["version"], "passed": all(checks.values()), "checks": checks, "measured": _strip(m), "every_language": _strip(every),
              "alone": {"threshold": t, "n_above": len(above), "right": k}}
    (Path(campaign) / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return result


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dclab_rnd.intent_classifier", description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("train", "write-gate", "gate"))
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args(argv)
    if args.command == "train":
        doc = train()
        print(json.dumps({k: doc[k] for k in ("version", "trained_on", "by_writer", "alone_on_dev")}, indent=1))
        for k, v in doc["cross_writer"].items():
            print(k, v["accuracy"], "rules", v["rules"]["accuracy"], {t: x["recall"] for t, x in v["per_track"].items()},
                  {l: x["accuracy"] for l, x in v["by_lang"].items()})
        print("home requests", doc["home_requests"]["accuracy"], "rules", doc["home_requests"]["rules"]["accuracy"])
        return 0
    if args.command == "write-gate":
        print(json.dumps(write_gate(), ensure_ascii=False, indent=1))
        return 0
    if not args.yes:
        print("The sealed set is read once. Add --yes to read it under the written gate.")
        return 0
    print(json.dumps(gate(), ensure_ascii=False, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
