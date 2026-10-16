"""Decision models for the workflow (package 26.3): collected from journeys, trained, judged once.

The workflow's own decision points (the problem's family, whether it is ready to build, the next question, the route of
a sub-problem) are learned from the journeys: each written person is played through the Home conversation with a ledger
of its own, every workflow decision it leaves is labelled from the journey's truth (``dclab_core.workflow_labels``), and
one generation-1 classifier per point is trained on them (``dclab_core.decision_models.train``, the column points' code).

- ``collect``: the dev journeys and the sealed sets already spent (v1 to v4, read once each and dev material since),
  standard and programme, at the replicas' seeds 0 and 1; a person's rows stay in one split (train or dev, by person).
  The rows go to ``decisions/workflow_v1/decisions/<point>/rows.jsonl`` (rebuilt, not committed); ``MANIFEST.json`` is.
- ``train``: a card per point in ``decisions/workflow_models_v1/<point>.json`` (committed) beside the rule's dev
  score; the fitted models (``.joblib``) are not committed, their SHA-256s are (``fitted.json``).
- ``gate --yes``: the sealed journeys written for this package (``journeys_workflow_sealed.json``, hash pinned) played
  once at seed 5, and each model judged by ``decision_gate_v1`` (the column points' gate, unchanged) beside the rule on
  the same decisions; ``evidence/campaigns/workflow_models_v1/results/DWM-NNN_<point>.json``. A second run is refused.

A point whose model does not pass stays in shadow (22.4); a point that passes is only a candidate for the audited switch
(22.6). Nothing here stops the stages: a journey is played up to its build, the engine's stages are not needed by the
Home's decisions.

    python -m dclab_rnd.workflow_models collect|relabel|train|gate [--yes] [--workers 3]
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions"
SETS = DECISIONS / "workflow_v1"
MODELS = DECISIONS / "workflow_models_v1"
CAMPAIGN = ROOT / "evidence" / "campaigns" / "workflow_models_v1"
JOURNEYS = ROOT / "dclab_rnd" / "agent_eval" / "journeys"
SEALED_FILE, SEALED_HASH = JOURNEYS / "journeys_workflow_sealed.json", JOURNEYS / "sealed_workflow.sha256"
SEEDS = (0, 1)
DEV_SHARE = 0.3
GATE = {"version": "workflow_models_gate_v1", "decision_gate": "decision_gate_v1", "seed": 5,
        "set": "journeys_workflow_sealed.json (30 people, two per kind, written by an agent that read only the format, the kinds and their replicas)",
        "rule": "Each point's model judged once by decision_gate_v1 (the column points' thresholds, unchanged) beside the rule on the same "
                "sealed decisions, by person; a model that does not pass stays in shadow.",
        # written after the dev figures of the cards and before the sealed journeys were read (2026-10-17)
        "seen_before": "Labels v2 (accepted sets, after the review of 26.3); dev figures of the cards: problem.family 0.526 against the "
                       "rules' 0.895 (the recorded state holds no words): expected to fail; problem.ready 0.905 against 0.905: no margin, "
                       "expected to fail; workflow.next_question 0.995 against 0.979 (third collection, after the second review: the "
                       "rules' 18 misses on dev rows are all one behaviour, a rare-event conversation with a file of cases that stops "
                       "after the budget while the action is open and the person could answer it): it passes only if that behaviour "
                       "appears for enough sealed people for the margin's lower bound to stay above zero, unlikely; solution.route 0.909 against 0.879 "
                       "(66 decisions): the margin's lower bound is unlikely to be above zero, expected to fail."
        }
LIMITS = ["Measured on dev rows of journeys: written people played through the Home conversation on the rules alone, each "
          "label the set of answers its journey accepts (workflow_labels_v2); the sealed journeys are read once, under the gate "
          "fixed before.",
          "A group is a person (a set and an id): dev holds other people of the same problem kinds, so dev measures reading "
          "these kinds of conversation, not real sessions, whose words and order vary more.",
          "On dev the rules' answer counts right when the journey accepts it, the classifier's only when it is the label it "
          "learned (the first accepted); the sealed gate counts every accepted answer for both.",
          "The state is what the ledger records (counts, flags and codes, never the person's words): a classifier sees less "
          "than the rules, which read the words; matching the rules is the expected result until real sessions accumulate."]


def _split(group: str) -> str:
    h = int(hashlib.sha256(f"workflow-v1:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "dev" if h < DEV_SHARE else "train"


def journeys(sealed: bool = False) -> list[tuple[str, dict[str, Any]]]:
    """(source, journey) pairs: the dev and spent sets, or the workflow models' sealed set (checked against its hash)."""
    from .agent_eval import journeys as J
    from .agent_eval.journeys import programme as P

    if sealed:
        if hashlib.sha256(SEALED_FILE.read_bytes()).hexdigest() != SEALED_HASH.read_text(encoding="utf-8").strip():
            raise ValueError("the workflow models' sealed journeys changed since they were sealed: a sealed set is never edited")
        return [("workflow_sealed", j) for j in json.loads(SEALED_FILE.read_text(encoding="utf-8"))]
    out = [("dev", j) for j in J.load("dev")]
    for split in J.SEALED:  # each spent: read once by its own gate, dev material since
        out += [(split, j) for j in J.load(split)]
    out += [("programme_dev", j) for j in P.load("dev")] + [("programme_sealed", j) for j in P.load("sealed")]
    return out


def play(item: tuple[str, dict[str, Any], int]) -> dict[str, Any]:
    """One journey played up to its build with a ledger of its own, on the rules alone (an empty routing scope, no
    shadow): its workflow decisions at the learned points, unlabelled (``label_rows`` labels them)."""
    import os

    from dclab_core import workflow_labels as wl
    from dclab_core.ledger import POINTS

    from . import ledger
    from .agent_eval import journeys as J
    from .agent_eval.journeys import programme as P
    from .ledger import use

    source, journey, seed = item
    person = f"{source}:{journey['id']}"  # a person is a set and an id: two sets may reuse an id
    shadow_before = os.environ.get("DCLAB_DECISION_SHADOW")
    os.environ["DCLAB_DECISION_SHADOW"] = "0"
    with tempfile.TemporaryDirectory() as tmp:
        book = ledger.FileLedger(Path(tmp) / "ledger", "workflow-models")
        book.home = Path(tmp)
        before = ledger._INSTALLED.get("ledger")
        ledger.install(book)
        stop = RuntimeError("played up to the build: the stages are not needed by the workflow's decisions")
        error = None
        try:
            with use.scope({"decisions": {}}), mock.patch("dclab_rnd.studio.engine.run_all", side_effect=stop), \
                    mock.patch("dclab_rnd.studio.programme.run", side_effect=stop):
                (P.play if "programme" in journey else J.play)(journey, seed)
        except Exception as e:  # noqa: BLE001 - a journey that fails is counted, its decisions so far kept
            if e is not stop:  # the stop after the build is the play's planned end (a programme journey does not catch it)
                error = f"{type(e).__name__}: {str(e)[:160]}"
        finally:
            ledger.install(before)
            if shadow_before is None:
                os.environ.pop("DCLAB_DECISION_SHADOW", None)
            else:
                os.environ["DCLAB_DECISION_SHADOW"] = shadow_before
        records = book.records()
    seen, rows = set(), []
    workflow = Counter(r.point for r in records if POINTS[r.point].kind == "workflow")
    for r in records:
        if r.point not in wl.POINTS:
            continue
        key = json.dumps([r.point, r.subject.key, r.state, r.options, r.value_used], sort_keys=True, default=str)
        if key in seen:  # the same decision written again by a second call: one row
            continue
        seen.add(key)
        options = list(wl.QUESTIONS) if r.point == "workflow.next_question" else list(r.options)  # records before 26.3 listed five
        rows.append({"record": f"{person}:s{seed}:{len(rows)}", "point": r.point, "state": r.state, "options": options,
                     "value_raw": r.value_used, "used": r.used, "group": person, "journey": journey["id"], "source": source, "seed": seed})
    return {"id": journey["id"], "source": source, "seed": seed, "rows": rows, "workflow_records": dict(workflow), "error": error}


def label_rows(raw: list[dict[str, Any]], truths: dict[str, Any]) -> list[dict[str, Any]]:
    """Each raw decision labelled from its journey's truth (``truths`` by person): ``label`` the accepted answers and the
    one learned, ``value_used`` the rules' answer as it is scored (an accepted one reads as the label's value; the
    recorded one stays in ``value_raw``). A family the person stated is not the rules' decision: left out."""
    from dclab_core import workflow_labels as wl

    out = []
    for r in raw:
        if r["point"] == "problem.family" and r.get("used") == "person":
            continue
        truth = truths.get(r["group"])
        got = None if truth is None else wl.label(r["point"], r["value_raw"], r["state"], r["options"], truth)
        row = {**r, "split": r.get("split") or _split(r["group"]), "label": None, "value_used": r["value_raw"]}
        if got is not None:
            row["label"] = {"value": got.value, "accepted": list(got.accepted), "source": "benchmark", "why": got.why, "by": wl.VERSION}
            row["value_used"] = wl.canonical(r["value_raw"], got)
        out.append(row)
    return out


def truths(items: list[tuple[str, dict[str, Any]]]) -> dict[str, Any]:
    from dclab_core import workflow_labels as wl

    return {f"{s}:{j['id']}": wl.truth_of(j) for s, j in items}


def _played(items: list[tuple[str, dict[str, Any], int]], workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [play(i) for i in items]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(play, items))


def _fingerprint(rows: list[dict[str, Any]]) -> str:
    content = sorted(json.dumps({k: r.get(k) for k in ("record", "state", "label", "value_used")}, sort_keys=True, default=str) for r in rows)
    return hashlib.sha256("\n".join(content).encode()).hexdigest()


def collect(workers: int = 3, seeds: tuple[int, ...] = SEEDS, out: Path = SETS, items: Optional[list[tuple[str, dict[str, Any]]]] = None) -> dict[str, Any]:
    """The journeys played (the raw decisions to ``raw.jsonl``), then labelled (``relabel``)."""
    items = items if items is not None else journeys()
    played = _played([(s, j, seed) for seed in seeds for s, j in items], workers)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    raw = [r for p in played for r in p["rows"]]
    (out / "raw.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in raw), encoding="utf-8")
    run = {"seeds": list(seeds), "journeys": dict(Counter(s for s, _ in items)), "plays": len(played),
           "failed_plays": sorted({f"{p['source']}:{p['id']}" for p in played if p["error"]}),
           "workflow_records": dict(sum((Counter(p["workflow_records"]) for p in played), Counter()))}
    (out / "PLAYED.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
    return relabel(out, items)


def relabel(out: Path = SETS, items: Optional[list[tuple[str, dict[str, Any]]]] = None) -> dict[str, Any]:
    """The played decisions (``raw.jsonl``) labelled again from the journeys, per point, with the manifest: what
    ``collect`` does after its plays, without replaying them."""
    from dclab_core import workflow_labels as wl

    out = Path(out)
    raw = [json.loads(x) for x in (out / "raw.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = label_rows(raw, truths(items if items is not None else journeys()))
    by_point: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_point.setdefault(r["point"], []).append(r)
    sets = {}
    for point, mine in sorted(by_point.items()):
        folder = out / "decisions" / point
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "rows.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in mine), encoding="utf-8")
        labelled = [r for r in mine if r["label"]]
        sets[point] = {"rows": len(mine), "labelled": len(labelled), "by_split": dict(Counter(r["split"] for r in labelled)),
                       "labels": dict(Counter(str(r["label"]["value"]) for r in labelled)),
                       "rule_right": sum(1 for r in labelled if str(r["value_used"]) == str(r["label"]["value"])),
                       "people": len({r["group"] for r in labelled}), "fingerprint": _fingerprint(mine)}
    played = json.loads((out / "PLAYED.json").read_text(encoding="utf-8")) if (out / "PLAYED.json").is_file() else {}
    manifest = {"name": "workflow decisions from journeys", "version": "workflow_v1", "labels": wl.VERSION,
                "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **played,
                "left_out": {"problem.family stated by the person": sum(1 for r in raw if r["point"] == "problem.family" and r.get("used") == "person")},
                "sets": sets, "not_learned": wl.NOT_LEARNED,
                "rule": "Rows of written journeys played on the rules alone (no model, no routed model, no shadow); each label is the set "
                        "of answers its journey accepts, never DCLab's answer, and the rules are scored right when their answer is in it. A "
                        "person (a set and an id) is in one split. Rebuilt by `python -m dclab_rnd.workflow_models collect` (and `relabel` "
                        "from its plays); the rows are not committed."}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def rows(point: str, sets: Path = SETS) -> list[dict[str, Any]]:
    path = Path(sets) / "decisions" / point / "rows.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []


def train(points: Optional[list[str]] = None, sets: Path = SETS, out: Path = MODELS) -> dict[str, Any]:
    """A card per point; a point with too few people or one label value gets none (said in ``skipped``)."""
    import joblib

    from dclab_core import workflow_labels as wl
    from dclab_core.decision_models import train as fit

    from .decision_models_build import record_fitted

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cards, skipped = {}, {}
    for point in points or list(wl.POINTS):
        data = [r for r in rows(point, sets) if r.get("label")]
        train_rows, dev_rows = [r for r in data if r["split"] == "train"], [r for r in data if r["split"] == "dev"]
        try:
            model, card = fit(point, train_rows, dev_rows, limits=LIMITS)
        except ValueError as e:
            skipped[point] = str(e)
            continue
        (out / f"{point}.json").write_text(card.model_dump_json(indent=1) + "\n", encoding="utf-8")
        joblib.dump(model, out / f"{point}.joblib")
        cards[point] = card
    record_fitted(out)
    (out / "SKIPPED.json").write_text(json.dumps(skipped, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return {"cards": cards, "skipped": skipped}


def results(campaign: Path = CAMPAIGN) -> list[dict[str, Any]]:
    folder = Path(campaign) / "results"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("DWM-*.json"))] if folder.is_dir() else []


MARK = "sealed_read.json"  # written before the sealed set is opened: a crash afterwards still spends it


def spent(campaign: Path = CAMPAIGN) -> bool:
    fingerprint = SEALED_HASH.read_text(encoding="utf-8").strip()
    mark = Path(campaign) / MARK
    if mark.is_file() and json.loads(mark.read_text(encoding="utf-8")).get("fingerprint") == fingerprint:
        return True
    return any(d.get("sealed", {}).get("fingerprint") == fingerprint for d in results(campaign))


def judge(models: Path = MODELS, campaign: Path = CAMPAIGN, workers: int = 3, played: Optional[list[dict[str, Any]]] = None) -> list[Path]:
    """The sealed journeys played once and every trained point judged on them. The set is marked read before it is
    opened; every verdict is computed before any result is written; a sealed journey that failed to play fails every
    point (its decisions are missing)."""
    import joblib

    from dclab_core import decision_gate
    from dclab_core import workflow_labels as wl

    if spent(campaign):
        raise ValueError("the workflow models' sealed journeys were read: a sealed set is read once")
    if decision_gate.VERSION != GATE["decision_gate"]:
        raise ValueError(f"the gate fixed before the run is {GATE['decision_gate']}, not {decision_gate.VERSION}")
    models = Path(models)
    fitted = json.loads((models / "fitted.json").read_text(encoding="utf-8"))
    loaded = {}
    for point in wl.POINTS:
        card_path, model_path = models / f"{point}.json", models / f"{point}.joblib"
        if not card_path.is_file():
            continue
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != fitted.get(point):
            raise ValueError(f"{point}: the fitted model's bytes are not those recorded in fitted.json")
        card, model = json.loads(card_path.read_text(encoding="utf-8")), joblib.load(model_path)
        if model.version != card["version"]:
            raise ValueError(f"{point}: the fitted model ({model.version}) is not the one its card describes ({card['version']})")
        loaded[point] = (card, model)
    if not loaded:
        raise ValueError("no workflow model to judge: run train first")
    fingerprint = SEALED_HASH.read_text(encoding="utf-8").strip()
    campaign = Path(campaign)
    campaign.mkdir(parents=True, exist_ok=True)
    (campaign / MARK).write_text(json.dumps({"fingerprint": fingerprint, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                             "models": {p: c["version"] for p, (c, _) in loaded.items()}}, indent=1) + "\n", encoding="utf-8")
    if played is None:
        items = journeys(sealed=True)
        played = _played([(s, j, GATE["seed"]) for s, j in items], workers)
        labelled = label_rows([r for p in played for r in p["rows"]], truths(items))
    else:  # a test's plays, labelled already
        labelled = [r for p in played for r in p["rows"]]
    folder = campaign / "results"
    folder.mkdir(parents=True, exist_ok=True)
    if not (campaign / "manifest.json").exists():
        (campaign / "manifest.json").write_text(json.dumps({
            "campaign_id": "workflow_models_v1",
            "objective": "Package 26.3: generation-1 classifiers for the workflow's decision points (the family, ready to build, the next question, "
                         "the route), trained on journeys and judged once on a sealed set of journeys by decision_gate_v1 beside the rules.",
            "gate": GATE, "rule": GATE["rule"], "command": "python -m dclab_rnd.workflow_models gate --yes"}, indent=2) + "\n", encoding="utf-8")
    failed = sorted({f"{p.get('source', '')}:{p['id']}" for p in played if p.get("error")})
    docs = []
    for point, (card, model) in sorted(loaded.items()):
        sealed_rows = [r for r in labelled if r["point"] == point and r.get("label")]
        answers, errors = [], 0
        for r in sealed_rows:
            try:
                got = model.answer(point, r["state"], list(r.get("options") or []))
                answers.append(None if got is None else wl.canonical(got.value, wl.Label(r["label"]["value"], tuple(r["label"].get("accepted") or [r["label"]["value"]]), "")))
            except Exception:  # noqa: BLE001 - an error is counted, never skipped
                answers.append(None)
                errors += 1
        truth = [r["label"]["value"] for r in sealed_rows]
        verdict = decision_gate.verdict(truth, answers, [r["value_used"] for r in sealed_rows], [r["group"] for r in sealed_rows], errors=errors) \
            if sealed_rows else {"pass": False, "criteria": [], "note": "no labelled sealed decision at this point"}
        if failed:  # a journey that did not play hides its decisions: the set was not judged whole
            verdict = {**verdict, "pass": False, "criteria": [*verdict.get("criteria", []), {"criterion": "every sealed journey played",
                       "value": f"{len(played) - len(failed)}/{len(played)}", "need": "all", "pass": False}]}
        docs.append((point, card, sealed_rows, verdict))
    paths = []
    for point, card, sealed_rows, verdict in docs:
        n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in folder.glob("DWM-*.json")] or [0])
        ident = f"DWM-{n:03d}"
        doc = {"id": ident, "experiment_id": ident, "campaign_id": "workflow_models_v1", "kind": "workflow_model_gate", "status": "completed",
               "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "point": point,
               "model": {k: card[k] for k in ("name", "version", "family", "strength", "trained_set")},
               "gate": {**GATE, "thresholds": decision_gate.THRESHOLDS},
               "sealed": {"decisions": len(sealed_rows), "people": len({r["group"] for r in sealed_rows}), "fingerprint": fingerprint,
                          "plays": len(played), "failed_plays": failed},
               "verdict": verdict, "result": "pass" if verdict["pass"] else "fail",
               "rule": f"One run of {card['name']} on {len(sealed_rows)} sealed decisions of written journeys, judged by a gate fixed before it, "
                       "an answer counted right when the journey accepts it; evidence about these conversations with no model, not about real "
                       "sessions nor production readiness."}
        path = folder / f"{ident}_{point}.json"
        path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.workflow_models", description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("collect", "relabel", "train", "gate"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--yes", action="store_true", help="gate: read the sealed journeys (once: they are then spent)")
    args = parser.parse_args(argv)
    if args.command in ("collect", "relabel"):
        m = collect(args.workers) if args.command == "collect" else relabel()
        for point, s in m["sets"].items():
            print(f"{point:<24} {s['labelled']:>5} labelled of {s['rows']:>5}, {s['people']} people, rule right {s['rule_right']}  {s['labels']}")
        print(f"{m.get('plays')} plays, failed: {m.get('failed_plays') or 'none'}")
        return 0
    if args.command == "train":
        got = train()
        for point, c in got["cards"].items():
            d, r = c.dev, c.rule_dev
            print(f"{point:<24} {c.family}:{c.strength:<5} dev accuracy {d.accuracy} {d.accuracy_ci} | rule {r.accuracy} {r.accuracy_ci}")
        for point, why in got["skipped"].items():
            print(f"{point:<24} no model: {why}")
        return 0
    if not args.yes:
        print("Would play the sealed workflow journeys once and judge every trained point on them; add --yes (the set is then spent).")
        return 0
    for path in judge(workers=args.workers):
        doc = json.loads(path.read_text(encoding="utf-8"))
        print(f"{doc['point']:<24} {doc['result']:<4} " + "  ".join(f"{c['criterion'].split(',')[0]}={c['value']}" for c in doc["verdict"].get("criteria", [])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
