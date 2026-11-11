"""One model against another across runs (step 6 of the 2026-11-11 plan: every exam that ran on Claude rerun on Qwen).

``python -m dclab_rnd.bench compare`` pairs two arms inside one result. Here the two sides come from different runs: for each
suite (``SUITES``: one reader per result shape), every result is read into arm records (suite, split, arm, the served model,
the judge, whether it is a first run or a rerun, and each case's numbers). Records on the same suite, split and arm are put
side by side, the baseline model's (Claude) against the target's (Qwen), per metric:

- **paired** when both sides answered at least two of the same cases (people, journeys, requests): ``dclab_core.bench.paired``,
  the bootstrap interval of the mean difference and, for yes/no metrics, the exact McNemar p-value;
- **unpaired** otherwise: each side's mean with its own 95% interval (Wilson's for yes/no, a bootstrap of the mean else),
  and the row says so. A suite that keeps no per-case rows (the students' exam) is always unpaired.

When a side has several runs on one split, each case keeps its latest run's value (newest by id, then date); the runs that
fed a side are listed. A **rerun** (``"unseen": false``, or ``_rerun`` in its name: an old sealed set run again on a new
model) is never mixed with a first run: it is its own side, and its row says the set was seen before. Every row carries the
judge of each side (Claude's exams were judged by Opus, Qwen's by Qwen: a gap mixes the student and the judge), and the
written people's model where the suite has one. Arms that call no model (the rules) are left out; a model never recorded is
"unknown", never guessed from a date.

Each report is written once, as new dated files ``evidence/bench/models/QWEN_VS_CLAUDE_<time>.{json,md}`` (never over one).

    python -m dclab_rnd.bench models [--target qwen/qwen3.8-flash] [--baseline claude] [--suites a,b] [--json] [--dry-run]
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from dclab_core import bench as core

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS = ROOT / "evidence" / "campaigns"
OUT = ROOT / "evidence" / "bench" / "models"
TARGET = "qwen/qwen3.8-flash"
BASELINE = "claude"  # a prefix: every Claude model is a baseline (Sonnet served the exams, Opus some Home dev runs)
NO_MODEL = "none"  # an arm that calls no model (the rules); left out of the comparison
UNKNOWN = "unknown"
CODE_JUDGE = "code (scored against the case's truth)"


@dataclass
class Arm:
    """One arm of one result: who answered, who judged, and each case's numbers."""

    suite: str
    ref: str
    split: str
    arm: str
    model: str
    model_source: str
    judge: str = CODE_JUDGE
    person: str = ""
    rerun: bool = False
    at: str = ""
    spent_eur: Optional[float] = None
    cases: dict[str, dict[str, float]] = field(default_factory=dict)  # metric -> case id -> value
    totals: dict[str, dict[str, Any]] = field(default_factory=dict)  # metric -> {k, n, value, lower, upper}: no per-case rows

    def header(self) -> dict[str, Any]:
        return {"suite": self.suite, "ref": self.ref, "split": self.split, "arm": self.arm, "model": self.model,
                "model_source": self.model_source, "judge": self.judge, "person": self.person,
                "run": "rerun (not unseen)" if self.rerun else "first run", "at": self.at, "spent_eur": self.spent_eur,
                "cases": max([len(v) for v in self.cases.values()] + [int(t.get("n") or 0) for t in self.totals.values()] + [0])}


# ---- reading a result's model -----------------------------------------------------------------------------------------

_SUFFIX = re.compile(r"\s*\(.*\)\s*$")


def clean(name: Any) -> str:
    """``"claude-sonnet-5-5 (the product's)"`` -> ``claude-sonnet-5-5``; an empty or non-text value -> ''."""
    return _SUFFIX.sub("", str(name)).strip() if isinstance(name, str) else ""


def _named(values: Iterable[Any]) -> str:
    names = sorted({clean(v) for v in values if clean(v)})
    return " + ".join(names)


def model_of(doc: Mapping[str, Any], arm: Optional[str] = None) -> tuple[str, str]:
    """The model that answered in ``arm`` of a result, and where it was read: the exam's per-arm field (``models.arms`` as
    text for every arm or a mapping by arm, or ``models.<arm>``), then the run's own ``model``, its plan's, a mapping of
    purposes to models (``models``), the rows' ``models``. Nothing recorded: ``unknown``."""
    models = doc.get("models")
    if isinstance(models, Mapping):
        arms = models.get("arms")
        if isinstance(arms, Mapping) and arm and clean(arms.get(arm)):
            return clean(arms[arm]), f"models.arms.{arm}"
        if isinstance(arms, Mapping) and not arm and _named(arms.values()):
            return _named(arms.values()), "models.arms"
        if isinstance(arms, str) and clean(arms):
            return clean(arms), "models.arms"
        if arm and clean(models.get(arm)):
            return clean(models[arm]), f"models.{arm}"
    if clean(doc.get("model")):
        return clean(doc["model"]), "model"
    plan = doc.get("plan")
    if isinstance(plan, Mapping) and clean(plan.get("model")):
        return clean(plan["model"]), "plan.model"
    if isinstance(models, Mapping):
        purposes = _named(v for k, v in models.items() if k not in ("person", "judge", "arms"))
        if purposes:
            return purposes, "models"
    if isinstance(models, str) and clean(models):
        return clean(models), "models"
    rows = doc.get("rows")
    if isinstance(rows, list):
        found = _named(v for r in rows if isinstance(r, Mapping) and isinstance(r.get("models"), Mapping) for v in r["models"].values())
        if found:
            return found, "rows[].models"
    return UNKNOWN, ""


def judge_of(doc: Mapping[str, Any]) -> str:
    models = doc.get("models")
    if isinstance(models, Mapping) and clean(models.get("judge")):
        return clean(models["judge"])
    if clean(doc.get("judge")):
        return clean(doc["judge"])
    return CODE_JUDGE


def person_of(doc: Mapping[str, Any]) -> str:
    models = doc.get("models")
    return clean(models.get("person")) if isinstance(models, Mapping) else ""


def is_rerun(doc: Mapping[str, Any], ref: str) -> bool:
    """A result run on a set its model's predecessor had already seen: ``"unseen": false`` or ``_rerun`` in its name."""
    return doc.get("unseen") is False or "_rerun" in ref


_ID = re.compile(r"^[A-Za-z]+-\d+_?")


def split_name(doc: Mapping[str, Any], ref: str) -> str:
    """The exact split (``sealed_v3``, ``dev_route``), not the coarse one: the result's own, else its name after the id,
    without the rerun words; a result with neither is ``-``."""
    if isinstance(doc.get("split"), str) and doc["split"]:
        return doc["split"]
    rest = re.sub(r"_rerun.*$", "", _ID.sub("", ref, count=1))
    return rest or "-"


def _num(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


def _put(arm: Arm, metric: str, case: Any, value: Any) -> None:
    v = _num(value)
    if v is not None and case not in (None, ""):
        arm.cases.setdefault(metric, {})[str(case)] = v


def _base(doc: Mapping[str, Any], suite: str, ref: str, arm: str, model: tuple[str, str]) -> Arm:
    spent = _num(doc.get("spent_eur"))
    return Arm(suite=suite, ref=ref, split=split_name(doc, ref), arm=arm, model=model[0], model_source=model[1],
               judge=judge_of(doc), person=person_of(doc), rerun=is_rerun(doc, ref),
               at=str(doc.get("at") or doc.get("completed_at") or "")[:25], spent_eur=spent)


# ---- one reader per result shape ---------------------------------------------------------------------------------------

CEX_METRICS = {"understanding": True, "options_fit": True, "route_right": True, "route_reached": True, "trap_pass": True,
               "safety_failures": False, "already_answered": False, "asked": False, "seconds_per_turn": False, "spent_eur": False}


def read_exam(doc: Mapping[str, Any], ref: str, suite: str) -> list[Arm]:
    """The conversation exam: ``scores.<arm>`` per person; rows that failed or that the judge did not score are left out."""
    out = []
    scores = doc.get("scores")
    if not isinstance(scores, Mapping) or "_owner_labels" in ref or "_second_labeller" in ref or "_pairs" in ref:
        return out
    for name, rows in scores.items():
        if not isinstance(rows, list):
            continue
        a = _base(doc, suite, ref, name, model_of(doc, name))
        for r in rows:
            if not isinstance(r, Mapping) or r.get("failed") or r.get("unjudged") or r.get("judged") is False:
                continue
            for m in CEX_METRICS:
                _put(a, m, r.get("id"), r.get(m))
        out.append(a)
    return out


HOME_METRICS = {"track_right": True, "exit_right": True, "fields_right": True, "breaches": False, "questions": False}
HOMECONV_METRICS = {"right": True, "right_offered": True, "turns_before_options": False, "silent": False, "repeated": False}
_BREACHES = ("quote_invalid", "uncited", "rows_exposed", "phantom", "destructive_unconfirmed", "lost")


def read_home(doc: Mapping[str, Any], ref: str, suite: str) -> list[Arm]:
    """Home: the request runs (``rows[].scores.<arm>``; the rules arm calls no model) and the conversations (HOMECONV:
    one row per person)."""
    rows = doc.get("rows") if isinstance(doc.get("rows"), list) else []
    if ref.startswith("HOMECONV"):
        a = _base(doc, suite, ref, "home", model_of(doc))
        for r in rows:
            if isinstance(r, Mapping):
                for m in HOMECONV_METRICS:
                    _put(a, m, r.get("id"), r.get(m))
        return [a]
    out = []
    for name in doc.get("arms") or []:
        a = _base(doc, suite, ref, str(name), (NO_MODEL, "a rules arm") if name == "rules" else model_of(doc, str(name)))
        for r in rows:
            s = ((r or {}).get("scores") or {}).get(name) if isinstance(r, Mapping) else None
            if not isinstance(s, Mapping):
                continue
            case = r.get("case")
            if s.get("track") is not None:
                _put(a, "track_right", case, s["track"] == "right")
            if s.get("exit") is not None:
                _put(a, "exit_right", case, s["exit"] == "right")
            fields = [v for v in (s.get("fields") or {}).values() if v != "abstained"] if isinstance(s.get("fields"), Mapping) else []
            if fields:
                _put(a, "fields_right", case, sum(v == "right" for v in fields) / len(fields))
            _put(a, "breaches", case, sum(int(_num(s.get(k)) or 0) for k in _BREACHES))
            _put(a, "questions", case, s.get("questions"))
        out.append(a)
    return out


JOURNEY_METRICS = {"right": True, "part.card": True, "part.workflow": True, "part.pipeline": True, "part.holdout": True,
                   "seconds": False, "spent_eur": False}
NO_MODEL_ARMS = {"rules", "rules_graph"}


def read_journeys(doc: Mapping[str, Any], ref: str, suite: str) -> list[Arm]:
    name = str(doc.get("arm") or "journeys")
    a = _base(doc, suite, ref, name, (NO_MODEL, "a rules arm") if name in NO_MODEL_ARMS else model_of(doc, name))
    if isinstance(doc.get("journeys"), str) and doc["journeys"]:
        a.split = f"{a.split}@{doc['journeys'][:12]}"  # the set's hash: an edited set is another split, its ids other people
    for r in doc.get("rows") or []:
        if not isinstance(r, Mapping):
            continue
        _put(a, "right", r.get("id"), r.get("right"))
        for part, ok in (r.get("parts") or {}).items():
            _put(a, f"part.{part}", r.get("id"), ok)
        _put(a, "seconds", r.get("id"), r.get("seconds"))
        _put(a, "spent_eur", r.get("id"), r.get("spent_eur"))
    return [a]


def read_harness(doc: Mapping[str, Any], ref: str, suite: str) -> list[Arm]:
    """The harness suite calls a model only when ``live`` (the scripted runs answer with the rules)."""
    a = _base(doc, suite, ref, "harness", model_of(doc) if doc.get("live") else (NO_MODEL, "scripted (live: false)"))
    for r in doc.get("rows") or []:
        if isinstance(r, Mapping):
            _put(a, "passed", r.get("id"), r.get("passed"))
    return [a]


AEV_METRICS = {"leak_caught": True, "false_alarm": False, "valid_moves": True, "citations_exist": True, "eur": False, "seconds": False}


def read_agent_eval(doc: Mapping[str, Any], ref: str, suite: str) -> list[Arm]:
    """The judgment suite: only its live runs call a model; a case's repeats are averaged."""
    if doc.get("scripted") or "scripted" in ref or not isinstance(doc.get("runs"), list):
        return []
    a = _base(doc, suite, ref, "live", model_of(doc))
    a.split = str(doc.get("suite") or a.split)
    sums: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in doc["runs"]:
        if not isinstance(r, Mapping) or not r.get("ran"):
            continue
        for m in AEV_METRICS:
            v = _num(r.get(m))
            if v is not None and r.get("case"):
                sums[(m, str(r["case"]))].append(v)
    for (m, case), vals in sums.items():
        a.cases.setdefault(m, {})[case] = sum(vals) / len(vals)
    a.spent_eur = a.spent_eur if a.spent_eur is not None else _num((doc.get("spent") or {}).get("eur") if isinstance(doc.get("spent"), Mapping) else doc.get("spent"))
    return [a]


MODEL_ANSWERER = ("model", "claude", "qwen")  # the students' exam named the served model's answerer "claude" before 2026-11-11


def read_students(doc: Mapping[str, Any], ref: str, suite: str) -> list[Arm]:
    """The students' exam keeps each decision point's accuracy, not each decision: always unpaired. The model answerer's
    accuracy on the decisions every compared answerer answered, with the interval the exam stored."""
    a = _base(doc, suite, ref, "decision_answer", model_of(doc))
    a.split = f"per_point_{doc.get('per_point')}_seed_{doc.get('seed')}"  # the same decisions only with the same size and seed
    for point, s in (doc.get("scores") or {}).items():
        common = (s or {}).get("on_common") if isinstance(s, Mapping) else None
        key = next((k for k in MODEL_ANSWERER if isinstance(common, Mapping) and isinstance(common.get(k), Mapping)), None)
        if not key:
            continue
        c = common[key]
        n, acc = int(c.get("n") or 0), _num(c.get("accuracy"))
        if n and acc is not None:
            ci = c.get("accuracy_ci") if isinstance(c.get("accuracy_ci"), list) else [None, None]
            a.totals[f"{point}.accuracy"] = {"k": round(acc * n), "n": n, "value": acc, "lower": ci[0], "upper": ci[1]}
    return [a]


Reader = Callable[[Mapping[str, Any], str, str], list[Arm]]
# a suite here is compared; a new one needs its reader and its metrics' direction
SUITES: dict[str, tuple[Reader, dict[str, bool]]] = {
    "conversation_exam_v1": (read_exam, CEX_METRICS),
    "home_v1": (read_home, {**HOME_METRICS, **HOMECONV_METRICS}),
    "journeys_v1": (read_journeys, JOURNEY_METRICS),
    "harness_v1": (read_harness, {"passed": True}),
    "students_exam_v1": (read_students, {}),
    "agent_eval_v1": (read_agent_eval, AEV_METRICS),
}
# arms compared inside one model across runs (the report's "lead agent against the model alone")
ARM_PAIRS: dict[str, list[tuple[str, str]]] = {"conversation_exam_v1": [("plain", "case")], "home_v1": [("plain", "model")],
                                               "journeys_v1": [("single_live", "team_live")]}


def arms(campaigns: Path = CAMPAIGNS, suites: Optional[Iterable[str]] = None) -> list[Arm]:
    out: list[Arm] = []
    for suite in suites or SUITES:
        if suite not in SUITES:
            raise ValueError(f"no reader for {suite}: choose from {', '.join(SUITES)}")
        reader = SUITES[suite][0]
        for path in sorted((Path(campaigns) / suite / "results").glob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(doc, dict):
                out += reader(doc, path.stem, suite)
    return out


# ---- the comparison ---------------------------------------------------------------------------------------------------

def _order(a: Arm) -> tuple[int, str, str]:
    m = re.match(r"^[A-Za-z]+-(\d+)", a.ref)
    return (int(m.group(1)) if m else -1, a.at, a.ref)


def side(records: list[Arm], metric: str) -> dict[str, Any]:
    """One side of a row from its runs: each case's latest value, the runs that gave a value, their judges and people."""
    values: dict[str, float] = {}
    used: list[str] = []
    total = None
    for a in sorted(records, key=_order):
        got = a.cases.get(metric) or {}
        if got:
            values.update(got)
            used.append(a.ref)
        if metric in a.totals:
            total, used = a.totals[metric], [a.ref]  # a suite with totals only: the latest run's, and only it is listed
    return {"values": values, "total": total, "runs": sorted(set(used)), "model": records[0].model,
            "run": "rerun (not unseen)" if records[0].rerun else "first run",
            "judge": _named(a.judge for a in records if a.ref in used) or records[0].judge,
            "person": _named(a.person for a in records if a.ref in used)}


def _binary(values: Iterable[float]) -> bool:
    return set(values) <= {0.0, 1.0}


def interval(values: dict[str, float], draws: int = 2000, seed: int = 0) -> dict[str, Any]:
    """A side's own mean and 95% interval: Wilson's for yes/no values, a bootstrap of the mean else (fewer than 2: none)."""
    vals = list(values.values())
    if not vals:
        return {"n": 0, "mean": None, "interval": [None, None]}
    mean = sum(vals) / len(vals)
    if _binary(vals):
        lo, hi = core.wilson(int(sum(vals)), len(vals))
    elif len(vals) >= 2:
        import numpy as np

        x = np.array(vals)
        means = x[np.random.default_rng(seed).integers(0, len(x), (draws, len(x)))].mean(axis=1)
        lo, hi = round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)
    else:
        lo = hi = None
    return {"n": len(vals), "mean": round(mean, 4), "interval": [lo, hi]}


def _summary(s: dict[str, Any]) -> dict[str, Any]:
    own = interval(s["values"]) if s["values"] else (
        {"n": s["total"]["n"], "mean": s["total"]["value"], "interval": [s["total"].get("lower"), s["total"].get("upper")]}
        if s["total"] else {"n": 0, "mean": None, "interval": [None, None]})
    return {"model": s["model"], "run": s["run"], "runs": s["runs"], "judge": s["judge"], "person": s["person"], **own}


def row(suite: str, split: str, arm: str, metric: str, higher: bool, a: dict[str, Any], b: dict[str, Any],
        label_a: str = "baseline", label_b: str = "target") -> Optional[dict[str, Any]]:
    """B against A on one metric: paired on their shared cases when there are two or more, unpaired else."""
    if not (a["values"] or a["total"]) or not (b["values"] or b["total"]):
        return None
    shared = set(a["values"]) & set(b["values"])
    out: dict[str, Any] = {"suite": suite, "split": split, "arm": arm, "metric": metric, "higher_is_better": higher,
                           label_a: _summary(a), label_b: _summary(b), "judges_differ": a["judge"] != b["judge"]}
    if len(shared) >= 2:
        out["mode"] = "paired"
        out["paired"] = core.paired(a["values"], b["values"], higher_is_better=higher)
    else:
        out["mode"] = "unpaired"
        out["why_unpaired"] = ("the suite keeps no per-case rows" if not (a["values"] and b["values"])
                               else f"{len(shared)} shared case{'s' if len(shared) != 1 else ''}: fewer than 2")
    notes = []
    if out["judges_differ"]:
        notes.append(f"judged by {a['judge']} against {b['judge']}: a gap mixes the answerer and the judge")
    if b["run"] != "first run" or a["run"] != "first run":
        notes.append("a rerun of a set seen before: not unseen")
    if out["mode"] == "unpaired":
        notes.append("unpaired: " + out["why_unpaired"])
    out["note"] = "; ".join(notes)
    return out


MIXED = " + "  # a run whose purposes were answered by several models (``_named``)


def _is(model: str, who: str) -> bool:
    """The model is ``who`` or starts with it; a mixed run (half Claude, half Qwen) is neither side."""
    return MIXED not in model and (model == who or model.startswith(who))


def compare(records: list[Arm], target: str = TARGET, baseline: str = BASELINE) -> dict[str, Any]:
    """Every (suite, split, arm, metric) where the baseline and the target both ran: a row per baseline model and per
    target side (first runs and reruns apart); and inside each model, its arm pairs (``ARM_PAIRS``) across runs."""
    groups: dict[tuple[str, str, str], dict[tuple[str, bool], list[Arm]]] = defaultdict(lambda: defaultdict(list))
    for a in records:
        if a.model in (NO_MODEL,):
            continue
        groups[(a.suite, a.split, a.arm)][(a.model, a.rerun)].append(a)
    rows_out, arms_out = [], []
    for (suite, split, arm), by_model in sorted(groups.items()):
        metrics = SUITES[suite][1]
        names = sorted({m for recs in by_model.values() for a in recs for m in (*a.cases, *a.totals)})
        for (bm, brerun), brecs in sorted(by_model.items()):
            if not _is(bm, baseline) or brerun:
                continue  # the baseline is the model's first runs
            for (tm, trerun), trecs in sorted(by_model.items()):
                if not _is(tm, target):
                    continue
                for metric in names:
                    r = row(suite, split, arm, metric, metrics.get(metric, True), side(brecs, metric), side(trecs, metric))
                    if r:
                        rows_out.append(r)
    for suite, pairs in ARM_PAIRS.items():
        for (a_arm, b_arm) in pairs:
            for (s, split, arm), by_model in sorted(groups.items()):
                if s != suite or arm != a_arm:
                    continue
                other = groups.get((suite, split, b_arm), {})
                for key, arecs in sorted(by_model.items()):
                    if key not in other or not (_is(key[0], target) or _is(key[0], baseline)):
                        continue
                    for metric in sorted({m for x in arecs + other[key] for m in x.cases}):
                        r = row(suite, split, f"{b_arm} vs {a_arm}", metric, SUITES[suite][1].get(metric, True),
                                side(arecs, metric), side(other[key], metric), "a", "b")
                        if r:  # sides under fixed keys: Home has an arm called "model"
                            arms_out.append({**r, "arms": [a_arm, b_arm], "answered_by": key[0]})
    return {"rows": rows_out, "arm_pairs": arms_out}


LIMITS = ["Qwen's exams are judged by Qwen and Claude's by Opus: a gap mixes the answerer and the judge (each row names both).",
          "A rerun of an old sealed set is not unseen: its people were written, run and read before (marked in every row).",
          "Small sets have wide intervals; an unpaired row compares two groups of cases, not the same cases.",
          "A side with several runs keeps each case's latest value; the runs listed are the ones that gave a value."]


def build(campaigns: Path = CAMPAIGNS, target: str = TARGET, baseline: str = BASELINE,
          suites: Optional[Iterable[str]] = None) -> dict[str, Any]:
    records = arms(campaigns, suites)
    found = compare(records, target, baseline)
    covered = {(r["suite"], r["split"], r["arm"]) for r in found["rows"]}
    runs = [a.header() for a in records if a.model != NO_MODEL]
    missing = sorted({(a.suite, a.split, a.arm) for a in records if _is(a.model, target)} - covered)
    return {"schema": "dclab_rnd.bench_models.v1", "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "target": target, "baseline": baseline, "suites": list(suites or SUITES), "runs": runs, **found,
            "target_alone": [{"suite": s, "split": sp, "arm": a} for s, sp, a in missing], "limits": LIMITS}


def _f(v: Any) -> str:
    return "–" if v is None else (f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v))


def _ci(i: list) -> str:
    return "" if not i or i[0] is None else f" [{_f(i[0])}, {_f(i[1])}]"


def _cell(s: dict[str, Any]) -> str:
    return f"{_f(s['mean'])}{_ci(s['interval'])} (n {s['n']})"


def render(doc: Mapping[str, Any]) -> str:
    out = [f"# {doc['target']} against {doc['baseline']}: every suite, every arm", "",
           f"Written by `python -m dclab_rnd.bench models` at {doc['at']} from the results in `evidence/campaigns`; nothing was run.",
           "A and B show each side's own mean over all its cases. A paired row adds the same cases only: B's mean difference (signed",
           "so that positive is better) with its bootstrap interval and McNemar's p for yes/no metrics; an unpaired row has no",
           "difference, only the two sides' own means and intervals.", "",
           "## Limits", ""] + [f"- {x}" for x in doc["limits"]] + [""]
    for title, key, la, lb in (("Target against baseline", "rows", "baseline", "target"), ("Arms inside one model, across runs", "arm_pairs", "a", "b")):
        rows = doc[key]
        out += [f"## {title}", ""]
        if not rows:
            out += ["None: no split where both sides ran.", ""]
            continue
        out += ["| suite | split | arm | metric | A | A model · judge · runs | B | B model · judge · runs | mode | B better by (paired) | McNemar p | note |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            sa, sb = r[la], r[lb]
            p = r.get("paired") or {}
            diff = (f"{_f(p.get('difference'))}{_ci(p.get('interval'))} on {p.get('n')} shared (A {_f(p.get('mean_a'))}, "
                    f"B {_f(p.get('mean_b'))}; wins {p.get('wins_b')}, losses {p.get('losses_b')})") if p else "–"
            out.append(f"| {r['suite']} | {r['split']} | {r['arm']} | {r['metric']}{'' if r['higher_is_better'] else ' (lower better)'} | "
                       f"{_cell(sa)} | {sa['model']} ({sa['run']}) · {sa['judge']} · {', '.join(sa['runs'])} | {_cell(sb)} | "
                       f"{sb['model']} ({sb['run']}) · {sb['judge']} · {', '.join(sb['runs'])} | {r['mode']} | {diff} | "
                       f"{_f(p.get('mcnemar_p'))} | {r['note']} |")
        out.append("")
    if doc["target_alone"]:
        out += ["## The target with no baseline on the same split", ""] + [f"- {x['suite']} {x['split']} {x['arm']}" for x in doc["target_alone"]] + [""]
    out += ["## Runs read", "", "| suite | result | split | arm | model (read from) | judge | people | run | cases | € |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in doc["runs"]:
        out.append(f"| {r['suite']} | {r['ref']} | {r['split']} | {r['arm']} | {r['model']} ({r['model_source'] or 'not recorded'}) | "
                   f"{r['judge']} | {r['person'] or '–'} | {r['run']} | {r['cases']} | {_f(r['spent_eur'])} |")
    return "\n".join(out) + "\n"


def write(doc: Mapping[str, Any], out: Path = OUT, stem: str = "QWEN_VS_CLAUDE") -> tuple[Path, Path]:
    """The report as new dated files: never over an existing one (``x`` mode; the pid keeps two runs in a second apart)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = str(doc["at"])[:19].replace("-", "").replace(":", "")
    base = out / f"{stem}_{stamp}Z_{os.getpid()}"
    paths = (base.with_suffix(".json"), base.with_suffix(".md"))
    if any(p.exists() for p in paths):
        raise FileExistsError(f"{base.name} exists: a report is never rewritten")
    text, body = render(doc), json.dumps(doc, indent=1, ensure_ascii=False)  # both made before a file is opened: no half report
    with open(paths[0], "x", encoding="utf-8") as fh:
        fh.write(body)
    with open(paths[1], "x", encoding="utf-8") as fh:
        fh.write(text)
    return paths


__all__ = ["Arm", "SUITES", "ARM_PAIRS", "arms", "build", "compare", "model_of", "judge_of", "is_rerun", "render", "write"]
