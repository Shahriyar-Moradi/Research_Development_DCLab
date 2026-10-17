"""The students' register (package 29.1, ``docs/guides/PHASE7_STUDENTS_PLAN.md``): one row per decision point, with
everything the loop knows about its student.

A student is the small decision model behind one choice (22.2 the column points, 26.3 the workflow points, 26.8 the
image points): trained on homework whose answers are known, examined once on a sealed set, kept in whisper beside
the rule until it proved better. The register reads, never computes: the ledger (decisions recorded, labels by source),
the generated sets' manifests (homework, sealed rows), the students' cards (family, dev score beside the rule's), the
sealed verdicts (``evidence/campaigns``), the routing document (the level each answerer serves at) and the shadow
measurement (labelled disagreements so far, what is still needed to act).

    python -m dclab_rnd.students [--json]       # the register of this workspace, as a table or as JSON
    GET /api/students                            # the same, on the Admin page (the owner's)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
DECISIONS = ROOT / "research" / "llm-fine-tuning" / "experiments" / "decisions"
CAMPAIGNS = ROOT / "evidence" / "campaigns"
# the homework: where each generated set's manifest is, and the shape of its per-point entry
SETS = (("generated_v1", "benchgen_v1"), ("generated_images_v1", "image_decisions_v1"), ("workflow_v1", "workflow_v1"))
# the students: the folders of cards, and the campaign that examined them (None: not examined yet)
STUDENTS = (("models_v1", "decision_models_v1"), ("models_images_v1", None), ("workflow_models_v1", "workflow_models_v1"))


def _json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def homework() -> dict[str, dict[str, Any]]:
    """Per point: the generated set's counts (decisions, sealed, dev, train, the rules right), from the manifests."""
    out: dict[str, dict[str, Any]] = {}
    for folder, _ in SETS:
        manifest = _json(DECISIONS / folder / "MANIFEST.json") or {}
        for point, s in (manifest.get("sets") or {}).items():
            splits = s.get("splits") or s.get("by_split") or {}
            rows = int(s.get("decisions") or s.get("labelled") or s.get("rows") or 0)  # the labelled decisions: what rule_right counts over
            out[point] = {"set": folder, "decisions": rows, "played": s.get("rows") if "labelled" in s else None,
                          "tables": s.get("tables", s.get("people")),
                          "sealed": int(splits.get("sealed") or 0), "dev": int(splits.get("dev") or 0), "train": int(splits.get("train") or 0),
                          "rule_right": s.get("rule_right"), "rule_accuracy": round(s["rule_right"] / rows, 4) if rows and isinstance(s.get("rule_right"), int) else None}
    return out


def students() -> dict[str, dict[str, Any]]:
    """Per point: the student's card (its name, family, how much it trained on, its dev score beside the rule's), or why
    it was skipped."""
    out: dict[str, dict[str, Any]] = {}
    for folder, _ in STUDENTS:
        here = DECISIONS / folder
        for path in sorted(here.glob("*.json")):
            if path.name in ("fitted.json", "SKIPPED.json"):
                continue
            card = _json(path) or {}
            if not card.get("point"):
                continue
            dev, rule = card.get("dev") or {}, card.get("rule_dev") or {}
            out[card["point"]] = {"folder": folder, "answerer": f"{card.get('name')}@{card.get('version')}", "family": card.get("family"),
                                  "trained_on": card.get("trained_on"), "dev_accuracy": dev.get("accuracy"), "dev_precision": dev.get("precision"),
                                  "dev_recall": dev.get("recall"), "rule_accuracy": rule.get("accuracy"), "rule_precision": rule.get("precision"),
                                  "rule_recall": rule.get("recall")}
        for point, why in (_json(here / "SKIPPED.json") or {}).items():
            out.setdefault(point, {"folder": folder, "skipped": str(why)})
    return out


def verdicts() -> dict[str, dict[str, Any]]:
    """Per point: the sealed exam's verdict (pass or fail, and the criteria it failed), read once from the campaigns."""
    out: dict[str, dict[str, Any]] = {}
    for _, campaign in STUDENTS:
        if campaign is None:
            continue
        for path in sorted((CAMPAIGNS / campaign / "results").glob("*.json")):
            doc = _json(path) or {}
            v = doc.get("verdict") or {}
            if not doc.get("point") or not isinstance(v, dict):
                continue
            failed = [c.get("criterion", "").split(",")[0] for c in v.get("criteria") or [] if not c.get("pass")]
            out[doc["point"]] = {"id": doc.get("id"), "pass": bool(v.get("pass")), "failed": failed,
                                 "sealed_decisions": (doc.get("sealed") or {}).get("decisions"),
                                 "sealed_accuracy": (v.get("model") or {}).get("accuracy"), "rule_sealed_accuracy": (v.get("rule") or {}).get("accuracy"),
                                 "answerer": f"{(doc.get('model') or {}).get('name')}@{(doc.get('model') or {}).get('version')}"}
    return out


def _ledger_counts(ledger: Any) -> tuple[Counter, dict[str, Counter]]:
    """Decisions per point, and labels per point by source, from a workspace's ledger (empty without one)."""
    from dclab_core.ledger import current_labels

    if ledger is None:
        return Counter(), {}
    try:
        records, labels = list(ledger.records()), current_labels(list(ledger.labels()))
    except Exception:  # noqa: BLE001 - a ledger that cannot be read counts nothing
        return Counter(), {}
    by_source: dict[str, Counter] = {}
    for x in labels:
        by_source.setdefault(x.point, Counter())[x.source] += 1
    return Counter(r.point for r in records), by_source


def status_of(row: dict[str, Any]) -> str:
    """One phrase per row: where the student stands in the loop."""
    from .ledger.routing import MIN_LABELS

    if row.get("not_decided"):
        return "not decided"
    level = row.get("level")
    if isinstance(level, int) and level >= 2:
        return "acts"
    if level == 1:
        return "flags"
    student, verdict = row.get("student") or {}, row.get("verdict") or {}
    if student.get("skipped"):
        return "no student"
    if not student:
        return "homework, no student" if row.get("homework") else "no homework"
    if verdict.get("pass"):
        needed = max(0, MIN_LABELS - int(row.get("labelled_disagreements") or 0))
        return f"whisper, ready to act after {needed} labels" if needed else "whisper, may act"
    if verdict:
        return "whisper, failed its exam"
    earlier = row.get("earlier_exam")
    if earlier:  # the exam was sat by another version: this one never saw it
        return f"whisper, not examined (an earlier version {'passed' if earlier.get('pass') else 'failed'})"
    return "whisper, not examined"


def register(ledger: Any = None, routing_doc: Optional[dict[str, Any]] = None, measured: Optional[dict[str, Any]] = None,
             warnings: Optional[list[str]] = None) -> dict[str, Any]:
    """The register: a row per point of ``dclab_core.ledger.POINTS`` and the counts over them."""
    from dclab_core.ledger import POINTS

    from .ledger import CAPTURED, NOT_DECIDED
    from .ledger.routing import MIN_LABELS, entry

    hw, st, vd = homework(), students(), verdicts()
    recorded, labelled = _ledger_counts(ledger)
    rows = []
    for key, point in POINTS.items():
        student = st.get(key) or {}
        name = student.get("answerer")
        exam = vd.get(key)
        verdict = exam if exam and exam.get("answerer") == name else None  # an exam belongs to the version that sat it
        work = dict(hw[key]) if key in hw else None
        if work is not None and not work.get("sealed") and exam and exam.get("sealed_decisions"):
            work["sealed"], work["sealed_from"] = int(exam["sealed_decisions"]), "the exam's own sealed set"  # 26.3: sealed journeys apart
        route = entry(routing_doc or {}, key) if routing_doc else {}
        shadow = (measured or {}).get(name) or {} if name else {}
        row = {"point": key, "kind": point.kind, "answer": point.answer, "about": point.about,
               "captured": CAPTURED.get(key), "not_decided": NOT_DECIDED.get(key),
               "recorded": int(recorded.get(key, 0)), "labels": dict(labelled.get(key) or {}),
               "homework": work, "student": student, "verdict": verdict, "earlier_exam": exam if exam and verdict is None else None,
               "level": route.get("level") if route.get("answerer") == name and name else None, "serving": route.get("answerer"),
               "labelled_disagreements": shadow.get("labelled_disagreements"), "shadow_agreement": shadow.get("agreement"),
               "needed_to_act": max(0, MIN_LABELS - int(shadow.get("labelled_disagreements") or 0)) if name else None}
        row["status"] = status_of(row)
        rows.append(row)
    counts = Counter(r["status"].split(",")[0] for r in rows)
    return {"items": rows, "total": len(rows), "warnings": list(warnings or []),
            "summary": {"points": len(rows), "technical": sum(r["kind"] == "technical" for r in rows), "workflow": sum(r["kind"] == "workflow" for r in rows),
                        "captured": sum(bool(r["captured"]) for r in rows), "not_decided": sum(bool(r["not_decided"]) for r in rows),
                        "with_homework": sum(bool(r["homework"]) for r in rows), "with_student": sum(bool(r["student"]) and not r["student"].get("skipped") for r in rows),
                        "passed": sum(bool((r["verdict"] or {}).get("pass")) for r in rows), "serving": sum(isinstance(r["level"], int) and r["level"] >= 1 for r in rows),
                        "recorded": sum(r["recorded"] for r in rows), "by_status": dict(counts), "min_labels_to_act": MIN_LABELS}}


def read(ledger: Any = None, routing_doc: Optional[dict[str, Any]] = None, measure: bool = True) -> dict[str, Any]:
    """The register as the page and the command both show it: the shadow measurement taken when shadows are on (a
    failure is said in ``warnings``, never shown as a number)."""
    measured, warnings = None, []
    if measure and ledger is not None:
        from .ledger import shadow

        if shadow.on():
            try:
                measured = shadow.measure(ledger.records(), ledger.labels())
            except Exception as error:  # noqa: BLE001 - a classifier that cannot be read measures nothing
                warnings.append(f"the shadow measurement failed ({type(error).__name__}): labelled disagreements are unknown")
        else:
            warnings.append("shadows are off here (DCLAB_DECISION_SHADOW=0): labelled disagreements are not measured")
    return register(ledger, routing_doc, measured, warnings)


def table(reg: dict[str, Any]) -> str:
    lines = [f"{'point':<28} {'kind':<9} {'recorded':>8} {'homework':>8} {'sealed':>6} {'student':<9} {'dev':>6} {'rule':>6} {'exam':<5} status"]
    for r in reg["items"]:
        hw, s, v = r.get("homework") or {}, r.get("student") or {}, r.get("verdict") or {}
        acc = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
        lines.append(f"{r['point']:<28} {r['kind']:<9} {r['recorded']:>8} {hw.get('decisions', 0):>8} {hw.get('sealed', 0):>6} {s.get('family') or '—':<9} "
                     f"{acc(s.get('dev_accuracy')):>6} {acc(s.get('rule_accuracy')):>6} {('pass' if v.get('pass') else 'fail') if v else '—':<5} {r['status']}")
    c = reg["summary"]
    lines.append(f"{c['points']} points: {c['with_homework']} with homework, {c['with_student']} with a student, {c['passed']} passed their exam, "
                 f"{c['serving']} serving above whisper, {c['not_decided']} not decided")
    lines += [f"note: {w}" for w in reg.get("warnings") or []]
    return "\n".join(lines)


def record(home: Path, reg: dict[str, Any]) -> Path:
    """The register's summary and every row's status, appended to ``<home>/students/register.jsonl`` with the time: what
    was known of every student on each run (the owner's rule: every result kept; ``make archive`` snapshots the folder)."""
    from datetime import datetime, timezone

    path = Path(home) / "students" / "register.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "summary": reg["summary"],
            "points": {r["point"]: {"status": r["status"], "recorded": r["recorded"], "student": (r.get("student") or {}).get("answerer"),
                                    "dev_accuracy": (r.get("student") or {}).get("dev_accuracy"), "exam": (r.get("verdict") or {}).get("pass"),
                                    "level": r.get("level")} for r in reg["items"]}}
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    from .ledger import open_ledger
    from .settings import Settings

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.students", description=__doc__.split("\n")[0])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-shadow", action="store_true", help="skip the shadow measurement (fast; the labels still needed are not counted)")
    args = parser.parse_args(argv)
    home = Path(Settings().agent_home)
    try:
        ledger = open_ledger(home)
    except Exception:  # noqa: BLE001 - no workspace here: the register still reads the files
        ledger = None
    routing_doc = None
    try:
        from .models.routing import open_routing

        routing_doc = open_routing(home).get()
    except Exception:  # noqa: BLE001
        routing_doc = None
    reg = read(ledger, routing_doc, measure=not args.no_shadow)
    record(home, reg)
    print(json.dumps(reg, indent=1, ensure_ascii=False) if args.json else table(reg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
