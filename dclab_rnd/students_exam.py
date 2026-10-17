"""Claude and GLiNER beside every student (package 29.17, ``docs/guides/PHASE7_STUDENTS_PLAN.md``): the same dev
decisions answered by the rule, the student, Claude (Sonnet 5.5, through the gateway's ``decision_answer`` purpose:
bands and name tokens, never a value) and GLiNER2.5-Decide (its answers already captured by 26.9, read, not re-asked),
scored against the truth planted in the homework.

What it says, per decision point: each examinee's accuracy, precision and recall on the decisions all of them answered,
and where the student and the rule disagree, who was right. The disagreements come first in the sample: they are where
an examinee can teach something.

Every answer is kept (the owner's capture rule): Claude's in ``lake/bronze/answers/claude/lake/<point>/dev.jsonl`` with
its reason, the prompt's fingerprint and ``use: eval_only`` (homework: the truth is known by construction, so its answer
is never a label; silver labels are for real projects only, the owner's rule of 2026-10-17); the run's result in
``evidence/campaigns/students_exam_v1/results/SEX-NNN.json``; every model request in the run's workspace (cost, tokens).

    python -m dclab_rnd.students_exam --per-point 300                      # the plan: decisions and the cost, nothing sent
    python -m dclab_rnd.students_exam --per-point 300 --yes --cap-eur 6    # the run, capped
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
SETS = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions" / "generated_v1" / "decisions"
RESULTS = ROOT / "evidence" / "campaigns" / "students_exam_v1" / "results"
GLINER = "gliner2.5-decide"
CLAUDE = "claude"
# a decision_answer request, measured on the prompt file and a rendered state: about this many tokens each way
TOKENS_IN, TOKENS_OUT = 900, 60


def _rows(point: str, sets: Optional[Path] = None) -> list[dict[str, Any]]:
    path = Path(sets or SETS) / point / "rows.jsonl"  # train and dev only: the sealed file is a file of its own, never opened here
    with path.open(encoding="utf-8") as handle:
        return [r for r in (json.loads(x) for x in handle if x.strip()) if r.get("split") == "dev" and (r.get("label") or {}).get("value") is not None]


def students_by_point() -> dict[str, Any]:
    from .ledger import shadow

    return {m.point: m for m in shadow.answerers()}


def student_values(student: Any, point: str, rows: list[dict[str, Any]]) -> list[Any]:
    """The student's answer to each row, in one batch: the same choice as ``ClassifierAnswerer.answer`` (the likeliest
    of the row's own options, renormalised; None below its confidence), without one model call per row."""
    if student is None:
        return [None] * len(rows)
    model = getattr(student, "model", None)
    if model is None or not hasattr(model, "predict_proba"):  # an answerer of another kind: asked row by row
        return [None if (a := student.answer(point, r["state"], list(r.get("options") or []))) is None else a.value for r in rows]
    from dclab_core.answerers import _key, _option, features

    probas = model.predict_proba([features(r["state"], getattr(student, "exclude", ())) for r in rows]) if rows else []
    classes = list(model.classes_)
    out = []
    for r, proba in zip(rows, probas):
        options = list(r.get("options") or [])
        allowed = {_key(_option(o)) for o in options} if options else set(classes)
        keep = [i for i, c in enumerate(classes) if c in allowed]
        if not keep:
            out.append(None)
            continue
        best = max(keep, key=lambda i: proba[i])
        confidence = float(proba[best]) / (float(sum(proba[i] for i in keep)) or 1.0)
        out.append(None if confidence < getattr(student, "abstain_below", 0.0) else student._values.get(classes[best], classes[best]))
    return out


def gliner(point: str, lake: Optional[Path] = None) -> dict[str, dict[str, Any]]:
    """GLiNER's captured dev answers of a point, by decision id."""
    from .lake import answers

    kw = {"lake": lake} if lake is not None else {}
    return {a["sample_id"]: a for a in answers.stored(GLINER, "lake", point, "dev", **kw)}


def sample(per_point: int, points: Optional[list[str]] = None, seed: int = 0, sets: Optional[Path] = None,
           students: Optional[dict[str, Any]] = None, lake: Optional[Path] = None) -> list[dict[str, Any]]:
    """``per_point`` dev decisions of each point: every one where the student and the rule disagree first, then those
    GLiNER answered, then the rest, each part drawn with ``seed``."""
    students = students_by_point() if students is None else students
    sets = Path(sets or SETS)
    out = []
    for point in points or sorted(p.name for p in sets.iterdir() if p.is_dir()):
        rows, student, captured = _rows(point, sets), students.get(point), gliner(point, lake)
        rng = random.Random(f"{seed}:{point}")

        for r, v in zip(rows, student_values(student, point, rows)):
            r["_student"] = v
        apart = [r for r in rows if r["_student"] is not None and r["_student"] != r.get("value_used")]
        known = [r for r in rows if r not in apart and r["record"] in captured]
        rest = [r for r in rows if r not in apart and r["record"] not in captured]
        picked: list[dict[str, Any]] = []
        for part in (apart, known, rest):
            rng.shuffle(part)
            picked += part[: max(0, per_point - len(picked))]
        out += picked
    return out


def cost_estimate(n: int) -> float:
    from .models import prices

    price = prices.price_of("claude-sonnet-5-5")
    return round(n * (TOKENS_IN * price.input + TOKENS_OUT * price.output) / 1e6, 2)


def _score(truth: list[Any], answers: list[Any], groups: list[Any]) -> dict[str, Any]:
    from dclab_core.decision_models import metrics

    s = metrics(truth, answers, groups=groups)
    return {k: getattr(s, k) for k in ("n", "accuracy", "accuracy_ci", "precision", "recall") if hasattr(s, k)}


def score(rows: list[dict[str, Any]], answers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per point: each examinee on the decisions every examinee answered, and on the student's disagreements with the
    rule who was right. ``answers``: examinee -> decision id -> value (None: abstained or not asked)."""
    out: dict[str, Any] = {}
    for point in sorted({r["point"] for r in rows}):
        mine = [r for r in rows if r["point"] == point]
        active = [e for e in answers if any(answers[e].get(r["record"]) is not None for r in mine)]  # an examinee with no answer here is left out, and said
        both = [r for r in mine if all(answers[e].get(r["record"]) is not None for e in active)]
        truth, groups = [r["label"]["value"] for r in both], [r["group"] for r in both]
        row: dict[str, Any] = {"decisions": len(mine), "answered_by": {e: sum(answers[e].get(r["record"]) is not None for r in mine) for e in answers},
                               "compared": active, "common": len(both), "on_common": {}}
        if both:
            row["on_common"] = {e: _score(truth, [answers[e][r["record"]] for r in both], groups) for e in active}
        apart = [r for r in mine if answers["student"].get(r["record"]) is not None and answers["student"][r["record"]] != answers["rule"].get(r["record"])]
        row["student_vs_rule"] = {"disagreements": len(apart),
                                  "right": {e: sum(answers[e].get(r["record"]) == r["label"]["value"] for r in apart) for e in answers},
                                  "answered": {e: sum(answers[e].get(r["record"]) is not None for r in apart) for e in answers}}
        out[point] = row
    return out


def keep_claude(rows: list[dict[str, Any]], got: dict[str, Any], model: str, prompt: str, lake: Optional[Path] = None) -> int:
    """Claude's answers, whole, beside GLiNER's (26.9's layout): one line per decision, appended."""
    from .lake import answers

    at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    by_point: dict[str, list[str]] = {}
    for r in rows:
        a = got.get(r["record"])
        line = {"sample_id": r["record"], "source": "lake", "decision_point": r["point"], "split": "dev", "group": r["group"],
                "label_origin": "construction", "answerer": CLAUDE, "model": model, "prompt": prompt, "options": r.get("options"),
                "value": None if a is None else a.value, "reason": None if a is None else a.reason, "abstained": a is None, "at": at,
                "use": "eval_only", "silver": False,  # homework: the truth is planted, so an answer is never a label here
                "package": "29.17"}
        by_point.setdefault(r["point"], []).append(json.dumps(line, ensure_ascii=False))
    kw = {"lake": lake} if lake is not None else {}
    for point, lines in by_point.items():
        path = answers.path_of(CLAUDE, "lake", point, "dev", **kw)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as out:
            out.write("".join(x + "\n" for x in lines))
    return sum(len(v) for v in by_point.values())


def run(rows: list[dict[str, Any]], advisor: Any, students: Optional[dict[str, Any]] = None, model: str = "claude-sonnet-5-5",
        lake: Optional[Path] = None, workers: int = 1) -> dict[str, Any]:
    """Ask Claude every sampled decision through ``advisor`` (the gateway's, capped), read GLiNER's captured answers,
    score all four, keep Claude's answers."""
    from dclab_core import prompts
    from dclab_core.answerers import ModelAnswerer

    students = students_by_point() if students is None else students
    prompt = prompts.fingerprints()["decision_answer"]
    claude = ModelAnswerer(advisor, {r["point"] for r in rows}, name=f"{CLAUDE}:{model}", version=prompt)
    from concurrent.futures import ThreadPoolExecutor

    def ask(r: dict[str, Any]) -> Any:
        return claude.answer(r["point"], r["state"], list(r.get("options") or []))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:  # Claude's calls in parallel (the owner's rule); the gateway's cap holds them
        got = dict(zip([r["record"] for r in rows], pool.map(ask, rows)))
    captured: dict[str, dict[str, Any]] = {}
    for point in {r["point"] for r in rows}:
        captured.update(gliner(point, lake))
    answers = {"rule": {r["record"]: r.get("value_used") for r in rows},
               "student": {r["record"]: r.get("_student") for r in rows},
               CLAUDE: {k: (None if a is None else a.value) for k, a in got.items()},
               GLINER: {r["record"]: (captured.get(r["record"]) or {}).get("value") for r in rows}}
    if rows and all(a is None for a in got.values()) and claude.errors == 0:
        raise RuntimeError("the model answered none of the decisions and no request failed: no model was asked; nothing kept")
    kept = keep_claude(rows, got, model, prompt, lake)
    return {"scores": score(rows, answers), "claude_errors": claude.errors, "kept": kept, "prompt": prompt,
            "abstained": dict(Counter(r["point"] for r in rows if got.get(r["record"]) is None))}


def rescore(rows: list[dict[str, Any]], lake: Optional[Path] = None, package: str = "29.17") -> dict[str, Any]:
    """The same scores from the answers already kept (Claude's from the lake, GLiNER's as captured): no request is sent."""
    from .lake import answers as A

    kw = {"lake": lake} if lake is not None else {}
    kept: dict[str, Any] = {}
    for point in {r["point"] for r in rows}:
        for a in A.stored(CLAUDE, "lake", point, "dev", **kw):
            if a.get("package") == package:
                kept[a["sample_id"]] = a.get("value")
    captured: dict[str, dict[str, Any]] = {}
    for point in {r["point"] for r in rows}:
        captured.update(gliner(point, lake))
    answers = {"rule": {r["record"]: r.get("value_used") for r in rows}, "student": {r["record"]: r.get("_student") for r in rows},
               CLAUDE: {r["record"]: kept.get(r["record"]) for r in rows},
               GLINER: {r["record"]: (captured.get(r["record"]) or {}).get("value") for r in rows}}
    return {"scores": score(rows, answers), "requests": 0}


def write(doc: dict[str, Any], results: Path = RESULTS) -> Path:
    results.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.stem.split("_")[0].split("-")[1]) for p in results.glob("SEX-*.json")] or [0])
    ident = f"SEX-{n:03d}"
    doc = {"id": ident, "experiment_id": ident, "campaign_id": "students_exam_v1", "kind": "students_exam", "status": "completed",
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **doc,
           "rule": "The generated sets' dev decisions (never the sealed partition), answered by the recorded rule, the student, "
                   "Claude through decision_answer (bands and name tokens, never values) and GLiNER2.5-Decide (captured by 26.9), "
                   "against the planted truth. Dev only: nothing here is a sealed verdict, and nothing is used or trained on."}
    path = results / f"{ident}.json"
    path.write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.students_exam", description=__doc__.split("\n")[0])
    parser.add_argument("--per-point", type=int, default=300)
    parser.add_argument("--points", default="", help="comma-separated; default every point with a generated set")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--yes", action="store_true", help="send the requests (without it: the plan only)")
    parser.add_argument("--cap-eur", type=float, default=None, help="the most this run may spend (required with --yes)")
    parser.add_argument("--workers", type=int, default=6, help="Claude's requests at once")
    parser.add_argument("--rescore", metavar="SEX-NNN", help="score again from the answers kept by that run: nothing is sent")
    args = parser.parse_args(argv)
    points = [p for p in args.points.split(",") if p] or None
    rows = sample(args.per_point, points, args.seed)
    by_point = Counter(r["point"] for r in rows)
    apart = Counter(r["point"] for r in rows if r["_student"] is not None and r["_student"] != r.get("value_used"))
    estimate = cost_estimate(len(rows))
    print(f"Plan: {len(rows)} dev decisions ({', '.join(f'{p} {n} ({apart[p]} where the student and the rule disagree)' for p, n in sorted(by_point.items()))}); "
          f"Claude Sonnet 5.5 through decision_answer, about EUR {estimate}; GLiNER's captured answers read, not re-asked.")
    if args.rescore:
        doc = {"model": "claude-sonnet-5-5", "per_point": args.per_point, "seed": args.seed, "decisions": len(rows), "rescored_from": args.rescore,
               **rescore(rows)}
        print(json.dumps(doc["scores"], indent=1, default=str))
        print(write(doc))
        return 0
    if not args.yes:
        print("Add --yes --cap-eur N to send the requests.")
        return 0
    if args.cap_eur is None:
        print("--cap-eur is required with --yes")
        return 2
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)  # the product's model, as the live journeys read it
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    from .core_advisor import GatewayAdvisor
    from .models.gateway import for_workspace

    gateway = for_workspace()  # every request, its tokens and cost, in this workspace's usage log (DCLAB_AGENT_HOME)
    if gateway.client("decision_answer") is None:  # nothing would be asked: say so before anything is written
        print("no model serves decision_answer here (no key, or DCLAB_NO_LIVE_MODELS): nothing sent, nothing written")
        return 2
    result = run(rows, GatewayAdvisor(gateway, max_tokens=300, cap_eur=args.cap_eur), workers=args.workers)
    doc = {"model": "claude-sonnet-5-5", "cap_eur": args.cap_eur, "estimate_eur": estimate, "per_point": args.per_point, "seed": args.seed,
           "decisions": len(rows), **result}
    print(json.dumps(doc["scores"], indent=1, default=str))
    print(write(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
