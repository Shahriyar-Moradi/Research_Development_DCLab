"""Challenge packs as decision data (2026-10-24): the homework's keys label every column's decisions, and every answerer
DCLab has is measured on them.

The H3 pilot showed where the local solver fails: it copies the studio's proposal, which never offers a plain-named
group key (``account``, ``borrower``) as a group and calls it a leak when it predicts well. The decision benchmark
(``generated_v1``) scores the rules 1.00 on ``column.group``: it holds no such column. This module turns the challenge
packs into labelled decisions of the same shape as ``generated_v1``'s rows, so the rule, the students and GLiNER are
measured on the trap that broke the pilot, and a student can learn it.

* Labels come from each variant's key, never from a model: the group column is the group and an identifier; the row key
  an identifier; the leaks leak; the decoys, the honest columns and the time column do not. A column the key does not
  name is not labelled.
* Splits follow the pack's partition: **learning** rows train, **check** rows measure. A **sealed** variant is never
  read here: it is the learning loop's exam, read once.
* Answerers: the rules as the decision benchmark has them (``datacard:purposes``), the studio's proposal the intern
  copies (``studio.solution.propose``: group candidates, keep-outs, identifiers), each point's student (``models_v1``)
  and GLiNER2.5-Decide (asked in the owner's ``ai12`` environment by ``gliner_run.py``, answers kept whole).

    python -m dclab_rnd.learning decisions build      # the rows, learning and check, into decisions/challenge_v1/
    python -m dclab_rnd.learning decisions gliner     # GLiNER's answers (ai12), kept beside the rows
    python -m dclab_rnd.learning decisions measure [--write]
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "challenge_v1"
POINTS = ("column.group", "feature.leakage_suspect", "column.is_identifier")
SPLIT = {"learning": "train", "check": "dev"}  # a sealed variant is never read here
AI12 = Path("/opt/miniconda3/envs/ai12/bin/python")


def roles_of(key: Any) -> dict[str, str]:
    """Each column the key names, by its part in the trap: group, row_key, leak, decoy, honest, time."""
    roles: dict[str, str] = {}
    for c in key.known_before:
        roles[c] = "honest"
    for c in key.decoys:
        roles[c] = "decoy"
    if key.time_column:
        roles[key.time_column] = "time"
    for c in key.leaks:
        roles[c] = "leak"
    if key.row_key:
        roles[key.row_key] = "row_key"
    if key.group_column:
        roles[key.group_column] = "group"
    return roles


def truths_of(roles: dict[str, str]) -> dict[str, dict[str, bool]]:
    return {c: {"column.group": r == "group", "feature.leakage_suspect": r == "leak", "column.is_identifier": r in ("group", "row_key")}
            for c, r in roles.items()}


def _proposal(frame: Any, target: str, task: str) -> dict[str, set[str]]:
    """What the studio proposes for this table, as the intern sees it from ``propose_solution``."""
    from dclab_rnd.studio import data as studio_data, solution as studio_solution

    proposal = studio_solution.propose(frame, studio_data.profile_table(frame), target, task)
    return {"group": set(proposal.get("group_candidates") or []), "forbidden": {f["column"] for f in proposal.get("forbidden") or []},
            "identifiers": set(proposal.get("identifiers") or [])}


def rows_of(family: str, seed: int, partition: str) -> list[dict[str, Any]]:
    from dclab_core import benchgen
    from dclab_core import variants as V

    v = V.generate(family, seed)
    key, target = v.key, v.pack.brief.target
    roles = roles_of(key)
    roles.pop(target, None)
    at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    group = f"challenge:{family}:{seed}"
    records, labels = benchgen.decide(v.table, {c: "challenge" for c in roles}, target, group, at, group=group, truths=truths_of(roles),
                                      label_by=f"challenge:{family}_key")
    proposal = _proposal(v.table, target, v.pack.brief.task)
    by_record = {lab.record: lab for lab in labels}
    out = []
    for r in records:
        if r.point not in POINTS:
            continue
        column = r.subject.key
        said = {"column.group": column in proposal["group"], "feature.leakage_suspect": column in proposal["forbidden"],
                "column.is_identifier": column in proposal["identifiers"] or column in proposal["group"]}[r.point]
        label = by_record[r.id]
        out.append({"record": r.id, "point": r.point, "subject": r.subject.model_dump(), "state": r.state, "options": r.options,
                    "answers": [a.model_dump() for a in r.answers], "value_used": r.value_used, "used": r.used,
                    "proposal": said, "label": {"value": label.value, "source": "benchmark"}, "role": roles[column],
                    "skin": (key.knobs or {}).get("skin"), "group": f"dataset:{group}", "split": SPLIT[partition], "family": family})
    return out


def build(family: str = "groups", n: int = 20, out: Path = OUT, log=print) -> dict[str, Any]:
    """Every point's rows from the first ``n`` learning and ``n`` check variants (never a sealed one)."""
    from dclab_core import variants as V

    by_point: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for partition in SPLIT:
        for seed in V.seeds(family, partition, n):
            for row in rows_of(family, seed, partition):
                by_point[row["point"]].append(row)
    counts = {}
    for point, rows in sorted(by_point.items()):
        folder = Path(out) / "decisions" / point
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "rows.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        counts[point] = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev")} | {"yes": sum(bool(r["label"]["value"]) for r in rows)}
        log(f"{point}: {counts[point]}")
    manifest = {"version": "challenge_v1", "family": family, "variants_per_partition": n, "partitions": dict(SPLIT), "counts": counts,
                "rule": "labels from each variant's key; learning rows train, check rows measure; a sealed variant is never read",
                "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (Path(out) / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def read(point: str, out: Path = OUT) -> list[dict[str, Any]]:
    path = Path(out) / "decisions" / point / "rows.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.is_file() else []


def gliner(points: tuple[str, ...] = POINTS, out: Path = OUT, python: Path = AI12, log=print) -> dict[str, int]:
    """GLiNER2.5-Decide's answer to every row, asked in the ``ai12`` environment (torch, gliner2) with the lake's own
    rendering (``lake.answers.render``), kept whole beside the rows (``gliner.jsonl``: the text, the schema, the output)."""
    from dclab_rnd.lake.answers import render

    asked = {}
    for point in points:
        rows = read(point, out)
        if not rows:
            continue
        todo = []
        for r in rows:
            text, schema = render(point, r["state"], list(r["options"]))
            todo.append({"record": r["record"], "text": text, "schema": schema})
        src = Path(out) / "decisions" / point / "gliner_in.jsonl"
        dst = Path(out) / "decisions" / point / "gliner.jsonl"
        src.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in todo), encoding="utf-8")
        subprocess.run([str(python), "-I", str(Path(__file__).with_name("gliner_run.py")), str(src), str(dst)], check=True)
        asked[point] = len(todo)
        log(f"gliner {point}: {len(todo)} asked")
    return asked


def _gliner_answers(point: str, out: Path = OUT) -> dict[str, Any]:
    path = Path(out) / "decisions" / point / "gliner.jsonl"
    if not path.is_file():
        return {}
    got = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            a = json.loads(line)
            got[a["record"]] = {"yes": True, "no": False}.get(a.get("label"), a.get("label"))
    return got


def answers_of(point: str, rows: list[dict[str, Any]], out: Path = OUT, students: Optional[Path] = None) -> dict[str, list[Any]]:
    from dclab_rnd.committee import STUDENTS, student_of
    from dclab_rnd.students_exam import student_values

    g = _gliner_answers(point, out)
    return {"rule": [r["value_used"] for r in rows], "proposal": [r["proposal"] for r in rows],
            "student": student_values(student_of(point, students or STUDENTS), point, rows), "gliner": [g.get(r["record"]) for r in rows]}


def score(rows: list[dict[str, Any]], answers: list[Any]) -> dict[str, Any]:
    """Accuracy with its interval, and how each kind of column was answered (the share answered yes by role)."""
    from dclab_core.decision_models import wilson

    pairs = [(r, a) for r, a in zip(rows, answers)]
    right = sum(a is not None and bool(a) == bool(r["label"]["value"]) for r, a in pairs)
    answered = sum(a is not None for _, a in pairs)
    by_role: dict[str, dict[str, Any]] = {}
    for role in sorted({r.get("role", "generated") for r in rows}):  # a generated row has no challenge role
        sub = [(r, a) for r, a in pairs if r.get("role", "generated") == role]
        yes = sum(bool(a) for _, a in sub if a is not None)
        by_role[role] = {"n": len(sub), "said_yes": yes, "right": sum(a is not None and bool(a) == bool(r["label"]["value"]) for r, a in sub)}
    if not answered:  # an answerer that never ran is not measured, never scored as wrong on every row
        return {"n": len(pairs), "answered": 0, "accuracy": None, "interval": None, "by_role": by_role, "why": "no answer: not measured"}
    low, high = wilson(right, len(pairs))
    return {"n": len(pairs), "answered": answered, "accuracy": round(right / len(pairs), 4) if pairs else None,
            "interval": [round(low, 4), round(high, 4)] if low is not None else None, "by_role": by_role}


def measure(points: tuple[str, ...] = POINTS, splits: tuple[str, ...] = ("train", "dev"), out: Path = OUT,
            students: Optional[Path] = None) -> dict[str, Any]:
    """Every answerer on every point, by split (none of them was trained on these rows: both splits are a fair test)."""
    result: dict[str, Any] = {}
    for point in points:
        rows = read(point, out)
        result[point] = {}
        for split in splits:
            sub = [r for r in rows if r["split"] == split]
            got = answers_of(point, sub, out, students)
            result[point][split] = {name: score(sub, vals) for name, vals in got.items()}
    return result


def held_out(generated_train: list[dict[str, Any]], rows: list[dict[str, Any]], skin: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The rows to train on with ``skin`` held out (the generated train rows and the other skins' learning rows) and the
    rows to test on (that skin's check rows): no row of the skin is trained on, and no sealed row anywhere."""
    fit = [r for r in generated_train if r.get("split") == "train"] + [r for r in rows if r["split"] == "train" and r.get("skin") != skin]
    test = [r for r in rows if r["split"] == "dev" and r.get("skin") == skin]
    return fit, test


def retrain(points: tuple[str, ...] = POINTS, out: Path = OUT, cards: Optional[Path] = None, log=print) -> dict[str, Any]:
    """Each point's student trained again on the generated train rows **and** the challenge learning rows, measured on
    the challenge check rows and the generated dev rows beside the old student, and once more with each skin held out
    (trained without its learning rows, tested on its check rows): the student reads column names, and the skins share
    them across partitions, so only a held-out skin shows whether it learned the trap or memorised the names. The new
    students are measured, not served: serving one is a routing switch with its own gate. Its card is kept."""
    from dclab_core.decision_models import train

    from dclab_rnd.committee import STUDENTS, student_of
    from dclab_rnd.decision_models_build import rows as generated
    from dclab_rnd.students_exam import student_values

    result: dict[str, Any] = {}
    for point in points:
        gen = generated(point)
        gtrain = [r for r in gen if r["split"] == "train"]
        gdev = [r for r in gen if r["split"] == "dev" and (r.get("label") or {}).get("value") is not None]
        rows = read(point, out)
        ctrain, cdev = [r for r in rows if r["split"] == "train"], [r for r in rows if r["split"] == "dev"]
        old = student_of(point, STUDENTS)
        new, card = train(point, gtrain + ctrain, cdev)
        held = {}
        for skin in sorted({r["skin"] for r in rows if r.get("skin")}):
            fit_rows, test = held_out(gtrain, rows, skin)
            if not test:
                continue
            model, _ = train(point, fit_rows, [])
            held[skin] = {"new": score(test, student_values(model, point, test)), "old": score(test, student_values(old, point, test))}
        result[point] = {"check": {"old": score(cdev, student_values(old, point, cdev)), "new": score(cdev, student_values(new, point, cdev))},
                         "generated_dev": {"old": score(gdev, student_values(old, point, gdev))["accuracy"],
                                           "new": score(gdev, student_values(new, point, gdev))["accuracy"]},
                         "held_out_skin": held, "card": {"version": card.version, "family": card.family, "strength": card.strength,
                                                         "trained_on": card.trained_on, "groups": card.groups}}
        if cards is not None:
            Path(cards).mkdir(parents=True, exist_ok=True)
            (Path(cards) / f"{point}.json").write_text(card.model_dump_json(indent=1) + "\n", encoding="utf-8")
        log(f"{point}: check {result[point]['check']['old']['accuracy']} -> {result[point]['check']['new']['accuracy']}")
    return result


__all__ = ["OUT", "POINTS", "SPLIT", "retrain", "held_out", "roles_of", "truths_of", "rows_of", "build", "read", "gliner", "answers_of", "score", "measure"]
