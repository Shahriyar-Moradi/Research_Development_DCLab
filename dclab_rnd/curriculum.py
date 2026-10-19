"""Harder homework where a student fails (package 29.12): ``python -m dclab_rnd.curriculum [--write]``.

A round reads which kinds of table the point's student gets wrong on dev (the outcome's kind of each dev table, recovered
from its group's seed), asks the generator (``dclab_core.benchgen2.task_table(kind=)``) for more tables of every failing
kind, in equal numbers so a kind confused with another is taught beside it, keeps only the new tables whose group the
datasets' own hash puts in train (so none reaches dev or sealed), and fits the student again on the parent's training rows
and every round's new ones, as the generation's students were fitted (``decision_models.train``, every candidate). Round 0
fits on the parent's training rows alone: the baseline in the same code.

The dev rows and the sealed file are the parent's, byte for byte (their SHA-256 checked before and after); the sealed
file is never opened. Each round is a version of its own (``decisions/generated_v2_curriculum/round_<n>/``: the new rows and
a manifest naming the parent, the failing kinds, the seeds and the fingerprints), its card in
``decisions/models_v2_curriculum/round_<n>/``, and a line in the register's history. ``GATE`` was fixed before round 1.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "research/llm-fine-tuning/experiments/decisions"
PARENT = DECISIONS / "generated_v2"
OUT = DECISIONS / "generated_v2_curriculum"
MODELS = DECISIONS / "models_v2_curriculum"
CAMPAIGN = ROOT / "evidence/campaigns/curriculum_v1"
POINT = "task.framing"
SEED_BASE = 100_000  # a round's seeds: far from the parent's 0..1199 and from every other set's
GATE = {"version": "curriculum_gate_v1", "fixed": "2026-10-19, before round 1", "point": POINT, "rounds": 2, "tables_per_kind": 120,
        "pass": "round 2's student scores at least 0.008 above round 0's on the parent's dev rows (round 0: the parent's training rows "
                "alone, fitted the same way), neither round scores more than 0.004 below the one before, and the dev rows and the sealed "
                "file are byte-identical to the parent's"}


def _sha(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def parent_rows(point: str = POINT) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [json.loads(x) for x in (PARENT / "decisions" / point / "rows.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = [r for r in rows if (r.get("label") or {}).get("value") is not None]
    return [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "dev"]


def fingerprints(point: str = POINT) -> dict[str, Optional[str]]:
    """The parent's files a round must never change: its rows (train and dev) and its sealed file, by their bytes."""
    folder = PARENT / "decisions" / point
    return {"rows": _sha(folder / "rows.jsonl"), "sealed": _sha(folder / "sealed.jsonl")}


def kind_of(row: dict[str, Any]) -> Optional[str]:
    """The outcome's kind a parent table was built with, from its group's seed (benchgen2: ``TASK_KINDS[seed % 10]``); a
    round's table carries its own."""
    from dclab_core import benchgen2 as g

    if row.get("kind"):
        return str(row["kind"])
    group = str(row.get("group") or "")
    try:
        seed = int(group.rsplit("-s", 1)[1])
    except (IndexError, ValueError):
        return None
    return g.TASK_KINDS[seed % len(g.TASK_KINDS)]


def errors_by_kind(model: Any, dev: list[dict[str, Any]], point: str = POINT) -> dict[str, dict[str, int]]:
    from dclab_core.answerers import _key

    out: dict[str, dict[str, int]] = {}
    for r in dev:
        got = model.answer(point, r["state"], list(r.get("options") or []))
        c = out.setdefault(kind_of(r) or "unknown", {"rows": 0, "wrong": 0})
        c["rows"] += 1
        c["wrong"] += int(got is None or _key(got.value) != _key(r["label"]["value"]))
    return out


def accuracy(model: Any, dev: list[dict[str, Any]], point: str = POINT) -> float:
    found = errors_by_kind(model, dev, point)
    rows = sum(c["rows"] for c in found.values())
    return round(1 - sum(c["wrong"] for c in found.values()) / rows, 4) if rows else 0.0


def new_rows(kinds: list[str], round_no: int, per_kind: int) -> list[dict[str, Any]]:
    """``per_kind`` new tables of each failing kind (each with its hidden-name twin, one group), their task decisions built as
    the parent's were (``decisions_gen``), only the groups the datasets' hash puts in train."""
    from dclab_core import benchgen2 as g
    from dclab_core.datasets import split_of

    from .decisions_gen import _hide, _record
    from .ledger import states
    from .studio.data import detect_task, profile_table

    rows = []
    for k, kind in enumerate(kinds):
        made = 0
        start = seed = SEED_BASE + round_no * 10_000 + k * 1_000
        while made < per_kind:
            seed += 1
            if seed - start >= 1_000:
                raise ValueError("a kind's seeds would run into the next kind's: ask for fewer tables per kind")
            group = f"task-c{round_no}-s{seed}"
            if split_of(f"dataset:{group}") != "train":
                continue  # a new table only ever teaches: dev and sealed stay the parent's
            frame, target, facts = g.task_table(seed, kind=kind)
            hidden, names = _hide(frame, seed * 104729 + 1)
            for suffix, f, n in (("", frame, {str(c): str(c) for c in frame.columns}), ("-ob", hidden, names)):
                t = n[target]
                record, label = _record(POINT, ("draft", group + suffix), states.task_state(profile_table(f), t), ["binary", "multiclass", "regression"],
                                        "rule:detect_task", detect_task(f, t)["task"], facts["task"], group + suffix, t, group)
                rows.append({"record": record.id, "point": POINT, "subject": record.subject.model_dump(), "state": record.state,
                             "options": record.options, "answers": [a.model_dump(mode="json") for a in record.answers], "value_used": record.value_used,
                             "used": "rule", "label": {"value": label.value, "source": "benchmark"}, "group": f"dataset:{group}", "split": "train",
                             "kind": kind, "round": round_no})
            made += 1
    return rows


def fit(rows: list[dict[str, Any]], dev: list[dict[str, Any]]) -> tuple[Any, Any]:
    from dclab_core.decision_models import train

    return train(POINT, rows, dev)  # every candidate, as decision_models_build fits the generation's students


def save(round_no: int, model: Any, card: Any, added: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    import joblib

    data = OUT / f"round_{round_no}"
    data.mkdir(parents=True, exist_ok=True)
    (data / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in added), encoding="utf-8")
    (data / "MANIFEST.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    folder = MODELS / f"round_{round_no}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{POINT}.json").write_text(card.model_dump_json(indent=1) + "\n", encoding="utf-8")
    joblib.dump(model, folder / f"{POINT}.joblib")
    (folder / "fitted.json").write_text(json.dumps({POINT: _sha(folder / f"{POINT}.joblib")}, indent=1) + "\n", encoding="utf-8")


def publish(round_no: int) -> Path:
    """The last round's student kept as ``models_v2_curriculum/latest/``, for a later decision: neither the register nor
    the shadows read it, since CUR-001's gain on the parent's dev rows did not hold on a fresh draw (CUR-002)."""
    import shutil

    source, latest = MODELS / f"round_{round_no}", MODELS / "latest"
    shutil.rmtree(latest, ignore_errors=True)
    shutil.copytree(source, latest)
    (latest / "ROUND").write_text(f"{round_no}\n", encoding="utf-8")
    return latest


def register_line(home: Optional[Path], card: Any) -> None:
    """The round's card in the register's history (``students.record``): the point's row with this round's student."""
    if home is None:
        return
    from . import students
    from .experience import file_workspace_id
    from .ledger import open_ledger

    try:  # the workspace's own ledger and routing, as the register's command reads them (else every point would look changed)
        ledger = open_ledger(Path(home), file_workspace_id(Path(home)))
    except Exception:  # noqa: BLE001
        ledger = None
    try:
        from .models.routing import open_routing

        routing_doc = open_routing(Path(home)).get()
    except Exception:  # noqa: BLE001
        routing_doc = None
    reg = students.read(ledger, routing_doc, measure=False, home=Path(home))
    name = f"{card.name}@{card.version}"
    exam = students.verdicts().get(POINT)
    for row in reg["items"]:
        if row["point"] == POINT:
            row["student"] = {**(row.get("student") or {}), "answerer": name, "dev_accuracy": card.dev.accuracy,
                              "rule_accuracy": card.rule_dev.accuracy, "family": card.family, "trained_on": card.trained_on}
            row["verdict"] = exam if exam and exam.get("answerer") == name else None  # an exam belongs to the version that sat it
            row["earlier_exam"] = exam if exam and row["verdict"] is None else None
            row["status"] = students.status_of(row)
    students.record(Path(home), reg)


def run(rounds: int = GATE["rounds"], per_kind: int = GATE["tables_per_kind"], home: Optional[Path] = None, log=print, persist: bool = True) -> dict[str, Any]:
    before = fingerprints()
    train, dev = parent_rows()
    started = time.perf_counter()
    model, card = fit(train, dev)
    history = [{"round": 0, "added": 0, "trained_on": card.trained_on, "dev_accuracy": accuracy(model, dev), "card_dev_accuracy": card.dev.accuracy,
                "student": f"{card.name}@{card.version}", "errors_by_kind": errors_by_kind(model, dev), "seconds": round(time.perf_counter() - started, 1)}]
    log(f"round 0: dev {history[-1]['dev_accuracy']} on {card.trained_on} rows")
    if persist:
        save(0, model, card, [], {"parent": "generated_v2", "round": 0, "point": POINT, "added": 0, "fingerprints": before})
        register_line(home, card)
    taught: list[dict[str, Any]] = []
    for n in range(1, rounds + 1):
        failing = sorted(k for k, c in history[-1]["errors_by_kind"].items() if c["wrong"] and k != "unknown")
        if not failing:
            log(f"round {n}: nothing failing on dev: stopped")
            break
        started = time.perf_counter()
        added = new_rows(failing, n, per_kind)
        taught += added
        model, card = fit(train + taught, dev)
        history.append({"round": n, "failing_kinds": failing, "added": len(added), "trained_on": card.trained_on, "dev_accuracy": accuracy(model, dev),
                        "card_dev_accuracy": card.dev.accuracy, "student": f"{card.name}@{card.version}", "errors_by_kind": errors_by_kind(model, dev),
                        "seconds": round(time.perf_counter() - started, 1)})
        log(f"round {n}: {failing}, +{len(added)} rows, dev {history[-1]['dev_accuracy']}")
        if persist:
            save(n, model, card, added, {"parent": "generated_v2", "round": n, "point": POINT, "failing_kinds": failing, "tables_per_kind": per_kind,
                                         "seeds_from": SEED_BASE + n * 10_000, "added": len(added), "fingerprints": before,
                                         "rule": "New tables of the kinds the student failed on dev, only groups the datasets' hash puts in train; the "
                                                 "parent's dev rows and sealed file unchanged."})
            register_line(home, card)
    if persist and len(history) > 1:
        publish(history[-1]["round"])
    after = fingerprints()
    accs = [h["dev_accuracy"] for h in history]
    held = before == after and all(v for v in after.values())
    steady = all(b >= a - 0.004 for a, b in zip(accs, accs[1:]))
    gain = round(accs[-1] - accs[0], 4) if len(accs) > 1 else 0.0
    return {"gate": GATE, "history": history, "fingerprints": {"before": before, "after": after, "unchanged": held},
            "verdict": {"gain": gain, "steady": steady, "untouched": held, "pass": bool(len(accs) == GATE["rounds"] + 1 and gain >= 0.008 and steady and held)}}


def fresh_check(tables: int = 2000, rounds: tuple[str, ...] = ("round_0", "latest"), seed_start: int = 300_000, draws: int = 2000) -> dict[str, Any]:
    """An independent check run after CUR-001's verdict: the rounds chose their kinds from the parent's dev errors and were
    scored on those dev rows, so a gain there may be the choice's own. ``tables`` new task tables in the generator's natural
    mix of kinds (seeds no set uses), each with its hidden-name twin; each student scored on them, and the last round
    against the first paired by table (a bootstrap over tables, 95%)."""
    import random

    import joblib
    from dclab_core import benchgen2 as g
    from dclab_core.answerers import _key

    from .decisions_gen import _hide, _record
    from .ledger import states
    from .studio.data import detect_task, profile_table

    rows = []
    for seed in range(seed_start, seed_start + tables):
        group = f"task-fresh-s{seed}"
        frame, target, facts = g.task_table(seed)
        hidden, names = _hide(frame, seed * 104729 + 1)
        for suffix, f, n in (("", frame, {str(c): str(c) for c in frame.columns}), ("-ob", hidden, names)):
            t = n[target]
            record, label = _record(POINT, ("draft", group + suffix), states.task_state(profile_table(f), t), ["binary", "multiclass", "regression"],
                                    "rule:detect_task", detect_task(f, t)["task"], facts["task"], group + suffix, t, group)
            rows.append({"state": record.state, "options": record.options, "label": {"value": label.value}, "kind": facts["kind"],
                         "rule": record.value_used, "table": group})
    out: dict[str, Any] = {"tables": tables, "decisions": len(rows), "seeds": [seed_start, seed_start + tables - 1],
                           "rule": round(sum(r["rule"] == r["label"]["value"] for r in rows) / len(rows), 4)}
    right: dict[str, list[int]] = {}
    for name in rounds:
        model = joblib.load(MODELS / name / f"{POINT}.joblib")
        marks = []
        for r in rows:
            got = model.answer(POINT, r["state"], list(r["options"]))
            marks.append(int(got is not None and _key(got.value) == _key(r["label"]["value"])))
        right[name] = marks
        out[name] = {"accuracy": round(sum(marks) / len(marks), 4), "errors_by_kind": errors_by_kind(model, rows)}
    first, last = rounds[0], rounds[-1]
    by_table: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_table.setdefault(r["table"], []).append(right[last][i] - right[first][i])
    groups = list(by_table.values())
    rng, diffs = random.Random(0), []
    for _ in range(draws):
        pick = [groups[rng.randrange(len(groups))] for _ in groups]
        diffs.append(sum(sum(x) for x in pick) / sum(len(x) for x in pick))
    diffs.sort()
    out["difference"] = {"of": f"{last} - {first}", "point": round(sum(map(sum, groups)) / len(rows), 4),
                         "ci95": [round(diffs[int(0.025 * draws)], 4), round(diffs[int(0.975 * draws) - 1], 4)], "method": "paired bootstrap over tables"}
    return out


def write(doc: dict[str, Any]) -> Path:
    from datetime import datetime, timezone

    results = CAMPAIGN / "results"
    results.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in results.glob("CUR-*.json")] or [0])
    ident = f"CUR-{n:03d}"
    doc = {"id": ident, "experiment_id": ident, "campaign_id": "curriculum_v1", "kind": doc.pop("kind", "curriculum"), "status": "completed",
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **doc,
           "rule": "Two curriculum rounds on task.framing: each adds tables of the kinds its student failed on dev (train groups only) and fits "
                   "the student again; scored on the parent's dev rows, never on its sealed file. Invented tables: evidence about teaching "
                   "a student, not about real projects."}
    path = results / f"{ident}_{POINT}.json"
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    from .settings import Settings

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.curriculum", description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help="store the result in evidence/campaigns/curriculum_v1 and the rounds' lines in the register")
    parser.add_argument("--fresh", action="store_true", help="the independent check on a fresh draw (after a run)")
    args = parser.parse_args(argv)
    if args.fresh:
        check = fresh_check()
        print(json.dumps({k: (v if not isinstance(v, dict) else v.get("accuracy")) for k, v in check.items()}, indent=1))
        if args.write:
            print(write({"kind": "curriculum_fresh_check", "check": check,
                         "note": "Run after CUR-001's verdict, not part of its gate: a fresh draw of task tables in the generator's natural mix "
                                 "(never trained on) scores round 0's and the last round's students."}))
        return 0
    if not (PARENT / "decisions" / POINT / "rows.jsonl").is_file():
        print("no generated_v2 sets: run make decisions-gen first")
        return 2
    doc = run(home=Path(Settings().agent_home) if args.write else None, persist=args.write)
    print(json.dumps(doc["verdict"], indent=1))
    if args.write:
        print(write(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
