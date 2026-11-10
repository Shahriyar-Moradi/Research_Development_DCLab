"""A committee for each table point, examined like any student (package 29.16): ``python -m dclab_rnd.committee``.

Each of the six table points (the generated set v1's) gets a committee of its rule, its student (``models_v1``) and
GLiNER2.5-Decide (26.9's captured answers), each voting with a weight from its accuracy on the v1 dev rows
(``dclab_core.committee.weight``). Claude is a member only in evaluation (the owner's rule: its answers never train a
student): ``--claude N`` asks it N sealed decisions per point, and a second committee with it is scored on those.

The exam is a fresh set (``generated_v1_committee``: twelve new seeds of the same ten bases, ``decisions_gen.build(first=24)``):
the v1 sealed sets were read once by the students' exams (DMG-001 to DMG-006) and are spent. The weights, the members'
dev scores and the gate (``GATE``) are written to the campaign's manifest before the sealed rows are read
(``plan --write``); ``exam --yes`` reads each point's fresh sealed rows once (their fingerprint against the committed
manifest, refused when that point's set was examined) and judges every answerer by ``dclab_core.decision_gate`` on the
same decisions: the rule, the student, GLiNER and the committee, with the committee's paired difference against the
member best on dev.

    python -m dclab_rnd.committee plan [--write]                 # the weights from dev, nothing sealed read
    python -m dclab_rnd.committee exam --yes [--points p1,p2]    # the sealed verdicts, once per point
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions"
DEV = DECISIONS / "generated_v1"
FRESH = DECISIONS / "generated_v1_committee"
STUDENTS = DECISIONS / "models_v1"
CAMPAIGN = ROOT / "evidence" / "campaigns" / "committee_v1"
POINTS = ("column.category_code", "column.group", "column.is_identifier", "column.semantic_role", "column.time", "feature.leakage_suspect")
MEMBERS = ("rule", "student", "gliner")  # the committee's order: a tie goes to the heaviest member, then to the rule
GLINER = "gliner2.5-decide"
CLAUDE = "claude"
GATE = {"version": "committee_gate_v1", "fixed": "2026-10-19, before any fresh sealed row was read",
        "members": list(MEMBERS), "evaluation_only": [CLAUDE],
        "weights": "dclab_core.committee.weight of each member's accuracy on the v1 dev rows (log((k-1)a/(1-a)), 0 at chance; an "
                   "accuracy of 1.0 read as 0.995)",
        "exam": "each point's fresh sealed rows read once; every answerer judged by dclab_core.decision_gate (decision_gate_v1's "
                "thresholds) on the same decisions",
        "pass": "the committee passes decision_gate_v1 on a point, as a student must",
        "compared": "the committee's accuracy minus that of the member with the best dev accuracy, paired by decision, with a 95% "
                    "interval from a bootstrap drawing whole tables (reported, not a criterion)"}


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()] if Path(path).is_file() else []


def dev_rows(point: str, sets: Path = DEV) -> list[dict[str, Any]]:
    """The v1 dev rows (``rows.jsonl``; the sealed file is not opened)."""
    return [r for r in _read(Path(sets) / "decisions" / point / "rows.jsonl") if r.get("split") == "dev" and (r.get("label") or {}).get("value") is not None]


def captured(answerer: str, source: str, point: str, splits: tuple[str, ...] = ("train", "dev", "sealed"), lake: Optional[Path] = None) -> dict[str, Any]:
    """Another model's captured answers to a point's decisions, by record, whatever split the lake filed them under (the
    last answer kept when one was asked twice)."""
    from .lake import answers

    kw = {"lake": lake} if lake is not None else {}
    out: dict[str, Any] = {}
    for split in splits:
        for a in answers.stored(answerer, source, point, split, **kw):
            out[a["sample_id"]] = a.get("value") if not a.get("abstained") else None
    return out


def student_of(point: str, folder: Path = STUDENTS) -> Any:
    from .ledger.shadow import _load

    return next((m for m in _load(Path(folder)) if m.point == point), None)


def member_answers(point: str, rows: list[dict[str, Any]], student: Any, gliner: dict[str, Any]) -> dict[str, list[Any]]:
    from .students_exam import student_values

    return {"rule": [r.get("value_used") for r in rows], "student": student_values(student, point, rows),
            "gliner": [gliner.get(r["record"]) for r in rows]}


def _accuracy(truth: list[Any], answers: list[Any]) -> Optional[float]:
    from dclab_core.answerers import _key

    return round(sum(a is not None and _key(a) == _key(t) for t, a in zip(truth, answers)) / len(truth), 4) if truth else None


def plan(points: tuple[str, ...] = POINTS, sets: Path = DEV, students: Path = STUDENTS, lake: Optional[Path] = None,
         gliner: Optional[dict[str, dict[str, Any]]] = None) -> dict[str, Any]:
    """Each point's members measured on the v1 dev rows, their weights, and the committee's own dev score (in-sample for
    its weights: said so). GLiNER's accuracy is over the dev rows it answered (its coverage beside it); Claude's, from the
    hard-first sample 29.17 asked it, is reported and never weights a homework committee."""
    from dclab_core import committee as C

    out: dict[str, Any] = {}
    for point in points:
        rows = dev_rows(point, sets)
        student = student_of(point, students)
        glin = (gliner or {}).get(point) if gliner is not None else captured(GLINER, "lake", point, lake=lake)
        answers = member_answers(point, rows, student, glin or {})
        truth = [r["label"]["value"] for r in rows]
        k = len(rows[0].get("options") or []) if rows else 2
        answered = [i for i, a in enumerate(answers["gliner"]) if a is not None]
        acc = {"rule": _accuracy(truth, answers["rule"]), "student": _accuracy(truth, answers["student"]),
               "gliner": _accuracy([truth[i] for i in answered], [answers["gliner"][i] for i in answered]) if answered else None}
        weights = {m: C.weight(acc[m], k) for m in MEMBERS}
        versions = {"rule": "recorded", "student": f"{student.name}@{student.version}" if student else "", "gliner": GLINER}
        committee = C.Committee(point, MEMBERS, weights, versions=versions)
        votes = [C.vote({m: answers[m][i] for m in MEMBERS}, weights, MEMBERS)[0] for i in range(len(rows))]
        claude = captured(CLAUDE, "lake", point, ("dev",), lake=lake)
        sample = [i for i, r in enumerate(rows) if r["record"] in claude]
        claude_acc = _accuracy([truth[i] for i in sample], [claude[rows[i]["record"]] for i in sample]) if sample else None
        with_claude = {**weights, CLAUDE: C.weight(claude_acc, k)}  # evaluation only: a second committee, never homework
        out[point] = {"dev_rows": len(rows), "options": k, "dev_accuracy": acc, "weights": weights, "versions": versions,
                      "gliner_answered": len(answered), "committee": f"committee@{committee.version}",
                      "committee_dev_accuracy": _accuracy(truth, votes), "best_member": max(MEMBERS, key=lambda m: (acc[m] or 0, -MEMBERS.index(m))),
                      "claude_dev_accuracy_on_its_sample": claude_acc, "claude_sample": len(sample), "weights_with_claude": with_claude,
                      "committee_with_claude": f"committee+claude@{C.fingerprint([*((m, versions[m]) for m in MEMBERS), (CLAUDE, 'decision_answer')], with_claude)}"}
    return {"gate": GATE, "points": out,
            "limits": ["The committee's dev accuracy is measured on the rows its weights came from.",
                       "Claude's dev accuracy comes from 29.17's sample, chosen hard first: lower than its rate."]}


def fingerprint(rows: list[dict[str, Any]]) -> str:
    content = sorted(json.dumps({k: x[k] for k in ("record", "state", "label")}, sort_keys=True, default=str) for x in rows)
    return hashlib.sha256("\n".join(content).encode()).hexdigest()


def sealed(point: str, fresh: Path = FRESH) -> list[dict[str, Any]]:
    rows = _read(Path(fresh) / "decisions" / point / "sealed.jsonl")
    committed = json.loads((Path(fresh) / "MANIFEST.json").read_text(encoding="utf-8"))["sets"][point]["sealed_fingerprint"]
    if fingerprint(rows) != committed:
        raise ValueError(f"{point}: the fresh sealed decisions differ from the committed fingerprint")
    return [r for r in rows if (r.get("label") or {}).get("value") is not None]


def results(campaign: Path = CAMPAIGN) -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((Path(campaign) / "results").glob("CMT-*.json"))]


def plan_hash(manifest: dict[str, Any]) -> str:
    """The fingerprint of the plan an exam read its weights from, kept in each result."""
    return hashlib.sha256(json.dumps({k: manifest.get(k) for k in ("gate", "points")}, sort_keys=True, default=str).encode()).hexdigest()[:16]


def exam(point: str, manifest: dict[str, Any], fresh: Path = FRESH, students: Path = STUDENTS, campaign: Path = CAMPAIGN,
         lake: Optional[Path] = None) -> Path:
    """One point's fresh sealed rows read once: every answerer judged on the same decisions by decision_gate_v1."""
    from dclab_core import committee as C
    from dclab_core import decision_gate

    committed = json.loads((Path(fresh) / "MANIFEST.json").read_text(encoding="utf-8"))["sets"][point]["sealed_fingerprint"]
    if any(d.get("point") == point and d.get("sealed", {}).get("fingerprint") == committed for d in results(campaign)):
        raise ValueError(f"{point}: its fresh sealed set was examined; it is spent")
    entry = manifest["points"][point]
    weights = entry["weights"]
    student = student_of(point, students)
    if (f"{student.name}@{student.version}" if student else "") != entry["versions"]["student"]:
        raise ValueError(f"{point}: the student is not the one the committee's weights were measured for")
    rows = sealed(point, fresh)
    glin = captured(GLINER, "committee", point, ("sealed",), lake=lake)
    if float(weights.get("gliner") or 0) > 0 and any(r["record"] not in glin for r in rows):  # never spend a set on a committee short of a member
        raise ValueError(f"{point}: GLiNER, a member with a weight, has not answered every fresh sealed decision "
                         "(python -m dclab_rnd.lake answers --sources committee --splits sealed, in the ai12 env)")
    answers = member_answers(point, rows, student, glin)
    answers["committee"] = [C.vote({m: answers[m][i] for m in MEMBERS}, weights, MEMBERS)[0] for i in range(len(rows))]
    truth, groups, rule = [r["label"]["value"] for r in rows], [r["group"] for r in rows], answers["rule"]
    verdicts = {name: decision_gate.verdict(truth, answers[name], rule, groups) for name in ("rule", "student", "gliner", "committee")}
    best = entry["best_member"]
    doc = {"point": point, "committee": entry["committee"], "weights": weights, "versions": entry["versions"], "gate": GATE, "plan": plan_hash(manifest),
           "decision_gate": {"version": decision_gate.VERSION, "thresholds": decision_gate.THRESHOLDS},
           "sealed": {"decisions": len(rows), "tables": len(set(groups)), "fingerprint": committed, "set": Path(fresh).name},
           "gliner_answered": sum(a is not None for a in answers["gliner"]),
           "verdicts": verdicts, "accuracy": {k: _accuracy(truth, v) for k, v in answers.items()},
           "against_best_member": {"member": best, **decision_gate.margin(truth, answers["committee"], answers[best], groups)},
           "result": "pass" if verdicts["committee"]["pass"] else "fail"}
    out = Path(campaign) / "results"
    out.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in out.glob("CMT-*.json")] or [0])
    ident = f"CMT-{n:03d}"
    doc = {"id": ident, "experiment_id": ident, "campaign_id": "committee_v1", "kind": "committee_exam", "status": "completed",
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **doc,
           "rule": "One read of a point's fresh sealed decisions (new seeds of invented tables): the rule, the student, GLiNER and their "
                   "committee judged by a gate fixed before; evidence about planted kinds, not about real tables nor readiness."}
    path = out / f"{ident}_{point}.json"
    path.write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    return path


def _spent(advisor: Any) -> Optional[float]:
    spent = getattr(advisor, "spent_eur", None)
    try:
        return round(float(spent() if callable(spent) else spent), 4) if spent is not None else None
    except (TypeError, ValueError):
        return None


def claude_sample(rows: list[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """The first ``n`` sealed rows in a seeded hash order: a fair sample, whatever the answers."""
    return sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:committee:{r['record']}".encode()).hexdigest())[:n]


def keep_claude(point: str, rows: list[dict[str, Any]], got: dict[str, Any], model: str, prompt: str, lake: Optional[Path] = None,
                seed: int = 0) -> int:
    """Claude's answers to the fresh sealed sample, whole, in the answers' layout (``claude/committee/<point>/sealed.jsonl``):
    evaluation only, never a label (the truths are planted)."""
    from .lake import answers

    at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [json.dumps({"sample_id": r["record"], "source": "committee", "decision_point": point, "split": "sealed", "group": r["group"],
                         "label_origin": "construction", "answerer": CLAUDE, "model": model, "prompt": prompt, "options": r.get("options"),
                         "value": None if got.get(r["record"]) is None else got[r["record"]].value,
                         "reason": None if got.get(r["record"]) is None else got[r["record"]].reason, "abstained": got.get(r["record"]) is None,
                         "at": at, "use": "eval_only", "silver": False, "package": "29.16", "seed": seed}, ensure_ascii=False) for r in rows]
    path = answers.path_of(CLAUDE, "committee", point, "sealed", **({"lake": lake} if lake is not None else {}))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as out:
        out.write("".join(x + "\n" for x in lines))
    return len(lines)


def exam_claude(points: tuple[str, ...], per_point: int, advisor: Any, manifest: dict[str, Any], seed: int = 0, workers: int = 6,
                fresh: Path = FRESH, students: Path = STUDENTS, campaign: Path = CAMPAIGN, lake: Optional[Path] = None,
                # full: Optional[Path] = None, model: str = "claude-sonnet-5-5") -> Path:
                full: Optional[Path] = None, model: str = "qwen/qwen3.8-flash") -> Path:
    """Evaluation only: Claude asked a fair sample of each point's fresh sealed decisions; on that sample, every member,
    the committee and the committee with Claude (its weight fixed in the manifest before any sealed row was read)."""
    from concurrent.futures import ThreadPoolExecutor

    from dclab_core import committee as C
    from dclab_core import prompts
    from dclab_core.answerers import ModelAnswerer
    from dclab_core.decision_models import metrics

    from .students_exam import Whole

    from .lake import answers as lake_answers

    done = results(campaign)
    samples: dict[str, list[dict[str, Any]]] = {}
    for point in points:  # every point checked and drawn before the first request: a refusal never follows a spend
        committed = json.loads((Path(fresh) / "MANIFEST.json").read_text(encoding="utf-8"))["sets"][point]["sealed_fingerprint"]
        if not any(d.get("kind") == "committee_exam" and d.get("point") == point and d.get("sealed", {}).get("fingerprint") == committed for d in done):
            raise ValueError(f"{point}: its committee exam comes first (the weights are fixed and examined before Claude reads the set)")
        if any(d.get("kind") == "committee_claude_sample" and point in (d.get("points") or {}) and d.get("seed") == seed for d in done):
            raise ValueError(f"{point}: Claude was asked this point's sample (seed {seed}) already")
        samples[point] = claude_sample(sealed(point, fresh), per_point, seed)
        kept = {a["sample_id"] for a in lake_answers.stored(CLAUDE, "committee", point, "sealed", **({"lake": lake} if lake is not None else {}))
                if a.get("seed", 0) == seed}
        if kept & {r["record"] for r in samples[point]}:  # answers kept by a run that wrote no result: never asked and paid twice
            raise ValueError(f"{point}: Claude's answers to this sample (seed {seed}) are kept already")
    prompt = prompts.fingerprints()["decision_answer"]
    asker = ModelAnswerer(Whole(advisor, full), set(points), name=f"{CLAUDE}:{model}", version=prompt)
    out_points: dict[str, Any] = {}
    for point in points:
        entry = manifest["points"][point]
        rows = samples[point]
        student = student_of(point, students)
        answers = member_answers(point, rows, student, captured(GLINER, "committee", point, ("sealed",), lake=lake))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            got = dict(zip([r["record"] for r in rows], pool.map(lambda r: asker.answer(point, r["state"], list(r.get("options") or [])), rows)))
        keep_claude(point, rows, got, model, prompt, lake, seed=seed)
        answers[CLAUDE] = [None if got.get(r["record"]) is None else got[r["record"]].value for r in rows]
        answers["committee"] = [C.vote({m: answers[m][i] for m in MEMBERS}, entry["weights"], MEMBERS)[0] for i in range(len(rows))]
        order = (*MEMBERS, CLAUDE)
        answers["committee_with_claude"] = [C.vote({m: answers[m][i] for m in order}, entry["weights_with_claude"], order)[0] for i in range(len(rows))]
        truth, groups = [r["label"]["value"] for r in rows], [r["group"] for r in rows]
        out_points[point] = {"decisions": len(rows), "tables": len(set(groups)), "claude_abstained": sum(a is None for a in answers[CLAUDE]),
                             "sealed_fingerprint": json.loads((Path(fresh) / "MANIFEST.json").read_text(encoding="utf-8"))["sets"][point]["sealed_fingerprint"],
                             "scores": {k: {f: getattr(sc, f) for f in ("accuracy", "accuracy_ci")}
                                        for k, v in answers.items() for sc in [metrics(truth, v, groups=groups)]}}
    if out_points and all(p["claude_abstained"] == p["decisions"] for p in out_points.values()) and asker.errors == 0:
        raise RuntimeError("the model answered none of the decisions and no request failed: no model was asked")
    results_dir = Path(campaign) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in results_dir.glob("CMT-*.json")] or [0])
    ident = f"CMT-{n:03d}"
    doc = {"id": ident, "experiment_id": ident, "campaign_id": "committee_v1", "kind": "committee_claude_sample", "status": "completed",
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": model, "prompt": prompt, "per_point": per_point, "seed": seed,
           "claude_errors": asker.errors, "points": out_points, "gate": GATE, "plan": plan_hash(manifest),
           "cap_eur": getattr(advisor, "cap_eur", None), "spent_eur": _spent(advisor),
           "rule": "Evaluation only: Claude asked a seeded sample of each point's fresh sealed decisions; every answerer, the committee and the "
                   "committee with Claude scored on that sample. Claude's answers are never homework."}
    path = results_dir / f"{ident}_claude_sample.json"
    path.write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    return path


def write_plan(doc: dict[str, Any], campaign: Path = CAMPAIGN) -> Path:
    """The weights and the gate, written before any exam: refused once any result exists (weights set after a sealed
    read would be chosen on it)."""
    if results(campaign):
        raise ValueError("the committee was examined: its plan is fixed (a new plan needs a new fresh set and campaign)")
    Path(campaign).mkdir(parents=True, exist_ok=True)
    path = Path(campaign) / "manifest.json"
    about = {"objective": "29.16: does a vote of the rule, the student and GLiNER, each weighted by its dev accuracy, pass the students' sealed "
                          "gate, and beat its best member, on the six table points?",
             "rule": "Weights from the v1 dev rows, written with the gate before any fresh sealed row is read; one read of each point's fresh "
                     "sealed set; every answerer judged by decision_gate_v1 on the same decisions; Claude only in a second committee, for evaluation.",
             "command": "python -m dclab_rnd.committee plan --write | exam --yes | claude --yes --cap-eur N"}
    path.write_text(json.dumps({"campaign_id": "committee_v1", **about, "written": datetime.now(timezone.utc).isoformat(timespec="seconds"), **doc},
                               indent=1, default=str) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.committee", description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("plan", "exam", "claude"))
    parser.add_argument("--write", action="store_true", help="plan: write the weights and the gate to the campaign's manifest")
    parser.add_argument("--yes", action="store_true", help="exam: read the fresh sealed sets (once: each is then spent)")
    parser.add_argument("--points", default="")
    parser.add_argument("--per-point", type=int, default=60, help="claude: sealed decisions asked per point")
    parser.add_argument("--cap-eur", type=float, default=None, help="claude: the most the run may spend (required with --yes)")
    args = parser.parse_args(argv)
    points = tuple(p for p in args.points.split(",") if p) or POINTS
    if args.command == "plan":
        doc = plan(points)
        for point, e in doc["points"].items():
            print(f"{point:<24} dev {e['dev_accuracy']} weights {e['weights']} committee {e['committee_dev_accuracy']} claude {e['claude_dev_accuracy_on_its_sample']}")
        if args.write:
            print(write_plan(doc))
        return 0
    manifest = json.loads((CAMPAIGN / "manifest.json").read_text(encoding="utf-8"))
    if args.command == "claude":
        import os

        from .students_exam import cost_estimate

        n = args.per_point * len(points)
        print(f"Plan: Claude Sonnet 5.5 through decision_answer on {args.per_point} fresh sealed decisions of each of {len(points)} points "
              f"({n} in all), about EUR {cost_estimate(n)}; evaluation only, every reply kept whole.")
        if not args.yes or args.cap_eur is None:
            print("Add --yes --cap-eur N to send the requests.")
            return 0
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
        from .core_advisor import GatewayAdvisor
        from .models.gateway import for_workspace

        gateway = for_workspace()
        if gateway.client("decision_answer") is None:
            print("no model serves decision_answer here: nothing sent, nothing written")
            return 2
        home = Path(os.environ.get("DCLAB_AGENT_HOME") or ROOT / "agent_runs")
        path = exam_claude(points, args.per_point, GatewayAdvisor(gateway, max_tokens=300, cap_eur=args.cap_eur), manifest,
                           full=home / "answers" / "committee_answers.jsonl")
        doc = json.loads(path.read_text(encoding="utf-8"))
        for point, e in doc["points"].items():
            print(f"{point:<24} " + " ".join(f"{k}={v['accuracy']}" for k, v in e["scores"].items()))
        print(path)
        return 0
    if not args.yes:
        print(f"Would read the fresh sealed decisions of {', '.join(points)} once; add --yes (each set is then spent).")
        return 0
    for point in points:
        doc = json.loads(exam(point, manifest).read_text(encoding="utf-8"))
        print(f"{point:<24} {doc['result']:<4} " + " ".join(f"{k}={v}" for k, v in doc["accuracy"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
