"""Harness evaluation from real sessions (package 21.5): the agents and the coordinator graded step by step, and the gate
every change to a prompt, a model or a gate passes before it is on.

The suite is three sets played through the same journeys (20.10, ``journeys.play``: a draft, the coordinator, the build,
the engine, with the play's own session record):

- **the regression cases** (``cases.json``): people written from failures, the owner's morning first, each with the
  trace it must follow: the questions asked (each once, in the expected order), the coordinator's steps (the expected
  ones in order), the hand-offs (allowed or refused, and which columns), and the failure flags that must not come back
  (``dclab_core.signals``);
- **cases from recorded sessions**: a flagged session a reviewer accepted (19.12's failure candidates, in the
  workspace's ``datasets/failures/benchmark_candidates/accepted.jsonl``) is replayed with the person's own answers on a
  table invented from its data card, and is right when none of its flags comes back and its card's traps stay out;
- **the dev journeys** (20.10), every part right as in the last run that passed.

``GATE`` is fixed before any run. A run stores the fingerprints of what it ran with (every prompt, every tool scope, the
graph's validator, dclab_core's role prompts and every suite's gate) and the model serving each purpose: a change to
any of them is not on until the suite passed on it (``tests/test_harness_gate.py`` for prompts and gates,
``models.routing.change`` for a model, ``ledger.routing.switch`` with 22.9's measurement for a decision model). A run
with served models (``--live``, capped) records which model answered each purpose and what it spent; only such a run
sends the prompts, so a prompt changed since the latest live run is listed by ``prompts_unmeasured``. A model set in
the environment (a tier's own model) is the owner's configuration, outside these gates.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "evidence" / "campaigns" / "harness_v1" / "results"
GATE = {
    "version": "harness_gate_v1",
    "cases": "every regression case and every accepted session case right: each step check and each part of its journey",
    "journeys": "every dev journey right",
    "flags": "no failure flag a case was made of comes back",
}
SESSION_CASES = Path("datasets") / "failures" / "benchmark_candidates" / "accepted.jsonl"


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def fingerprints() -> dict[str, str]:
    """Everything a run ran with that a change could make different: the studio's prompts, tool schemas and graph
    validator, dclab_core's role prompts, and every suite's gate."""
    from dclab_rnd.prompts import fingerprints as studio

    from ..conversation import GATES as conversation_gates
    from ..journeys import GATE as journeys_gate

    out = dict(studio())
    hashes = ROOT / "dclab_core" / "prompts" / "HASHES.json"
    if hashes.exists():
        out["core_prompts"] = _hash(json.loads(hashes.read_text(encoding="utf-8")))
    out["gate:harness"] = _hash(GATE)
    out["gate:journeys"] = _hash(journeys_gate)
    out["gate:conversation"] = _hash(conversation_gates)
    out["cases:regression"] = _hash(load())  # the cases themselves: a weakened expectation is a change too
    out["cases:journeys"] = _hash(json.loads((HERE.parent / "journeys" / "journeys_dev.json").read_text(encoding="utf-8")))
    import importlib

    for name, module, attrs in (("judgment", "dclab_rnd.agent_eval.gate", ("VERSION", "THRESHOLDS")), ("roles", "dclab_rnd.agent_eval.roles", ("GATES",)),
                                ("team", "dclab_rnd.agent_eval.team", ("GATE",)), ("layer", "dclab_rnd.agent_eval.layer", ("GATE",)),
                                ("analyst", "dclab_rnd.agent_eval.analyst", ("GATE",)), ("investigator", "dclab_rnd.agent_eval.investigator", ("GATE",)),
                                ("decision_models", "dclab_core.decision_gate", ("VERSION", "THRESHOLDS", "GATE")),
                                ("decision_routing", "dclab_rnd.ledger.routing", ("BAR", "MIN_LABELS"))):
        try:
            found = importlib.import_module(module)
        except ImportError:
            continue
        values = {a: getattr(found, a) for a in attrs if hasattr(found, a)}
        if values:
            out[f"gate:{name}"] = _hash(values)
    return out


PROMPT_KEYS = ("prompt:", "core_prompts")  # what only a run with a served model sends


def load() -> list[dict[str, Any]]:
    return json.loads((HERE / "cases.json").read_text(encoding="utf-8"))


def session_cases(home: Optional[Path]) -> list[dict[str, Any]]:
    """The flagged sessions a reviewer accepted, as cases: the problem, the person's answers, the data card, and the
    flags that must not come back. None without a workspace folder or an accepted case."""
    path = Path(home) / SESSION_CASES if home else None
    if path is None or not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line) if line.strip() else None
        if not row or not row.get("data_card") or not row.get("problem"):
            continue  # a session without its data card cannot be replayed on an invented table
        out.append({"id": row["id"], "source": "session", "kind": None, "family": None, "words": row["problem"],
                    "answers": row.get("answers") or {}, "data_card": row["data_card"],
                    "expect": {"no_flags": list(row.get("detectors") or [])}})
    return out


def grade(case: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    """The step checks of one case against its played row: each a {step, ok, detail}."""
    expect = case.get("expect") or {}
    checks = []

    def check(step: str, ok: bool, detail: str) -> None:
        checks.append({"step": step, "ok": bool(ok), "detail": detail[:300]})

    asked = list(row.get("asked") or [])
    check("each question asked once", len(asked) == len(set(asked)), f"asked {asked}")
    if "questions" in expect:
        wanted = list(expect["questions"])
        positions = [asked.index(q) if q in asked else -1 for q in wanted]
        if expect.get("exact"):  # no question more, none less, in this order
            check("the right questions, in order", asked == wanted, f"asked {asked}, expected exactly {wanted}")
        else:
            check("the right questions, in order", -1 not in positions and positions == sorted(positions), f"asked {asked}, expected {wanted}")
    steps = list(row.get("coordinator") or row.get("steps") or [])
    if "build" in steps:  # never a build tried before the data was analysed
        check("no build before the analysis", "analyse" in steps and steps.index("analyse") < steps.index("build"), f"steps {steps}")
    if "steps" in expect:
        if expect.get("exact"):
            check("the coordinator's steps, in order", steps == list(expect["steps"]), f"steps {steps}, expected exactly {expect['steps']}")
        else:
            at = 0
            for s in steps:
                if at < len(expect["steps"]) and s == expect["steps"][at]:
                    at += 1
            check("the coordinator's steps, in order", at == len(expect["steps"]), f"steps {steps}, expected in order {expect['steps']}")
    if "handoffs" in expect:
        got = [{k: h.get(k) for k in ("role", "allowed", "refused_columns")} for h in row.get("handoffs") or []]
        check("the right hand-offs", got == expect["handoffs"], f"hand-offs {got}")
    back = sorted(set(expect.get("no_flags") or []) & set(row.get("flags") or []))
    check("no failure flag comes back", not back, f"flags {row.get('flags')}" + (f"; back: {back}" if back else ""))
    return checks


def run(home: Optional[Path] = None, seed: int = 0, workers: int = 4, live: Optional[dict[str, Any]] = None,
        only: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """Every case and every dev journey played and graded; one row each."""
    from concurrent.futures import ProcessPoolExecutor

    from .. import journeys

    cases = [{**c, "source": c.get("source") or "written"} for c in load()] + session_cases(home)
    cases += [{**j, "source": "journey"} for j in journeys.load("dev")]
    cases = [c for c in cases if not only or c["id"] in only]
    if live is not None or workers <= 1:  # a served model's client stays in this process
        played = [_one((c, seed, live)) for c in cases]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            played = list(pool.map(_one, [(c, seed, None) for c in cases]))
    return played


def _one(args: tuple[dict[str, Any], int, Optional[dict[str, Any]]]) -> dict[str, Any]:
    from .. import journeys

    case, seed, live = args
    try:
        row = journeys.play(case, seed, None, live)
    except Exception as error:  # noqa: BLE001 - a case that breaks is wrong, with its reason
        row = {"id": case["id"], "kind": case.get("kind"), "right": False, "parts": {}, "why": [f"broke: {type(error).__name__}: {str(error)[:300]}"]}
    row["source"] = case["source"]
    row["checks"] = grade(case, row) if "parts" in row and row.get("parts") else []
    row["passed"] = bool(row.get("right")) and all(c["ok"] for c in row["checks"])
    return row


def verdict(rows: list[dict[str, Any]]) -> str:
    wrong = [r["id"] for r in rows if not r["passed"]]
    if not rows:
        return "fail: nothing played"
    return f"{'pass' if not wrong else 'fail'}: {len(rows) - len(wrong)} of {len(rows)} right on every step and part" + (
        f"; wrong: {', '.join(wrong)}" if wrong else "")


def exercised(rows: list[dict[str, Any]]) -> dict[str, Optional[str]]:
    """The model that answered each purpose the run's plays sent a request for (from their own session records); a
    purpose two models answered is recorded as None (no one model passed it). Empty: the rules alone."""
    seen: dict[str, set[str]] = {}
    for r in rows:
        for purpose, model in (r.get("models") or {}).items():
            seen.setdefault(purpose, set()).add(str(model))
    return {p: next(iter(m)) if len(m) == 1 else None for p, m in sorted(seen.items())}


def _kept(r: dict[str, Any]) -> dict[str, Any]:
    out = {k: r.get(k) for k in ("id", "source", "kind", "passed", "right", "parts", "checks", "why", "asked", "coordinator", "handoffs",
                                 "flags", "seconds", "models")}
    if r.get("source") == "session":  # a workspace's session: its column names never reach a committed file, only what was checked
        out["checks"] = [{"step": c["step"], "ok": c["ok"]} for c in r.get("checks") or []]
        out["why"] = [w.split(":")[0] for w in r.get("why") or []]
        out["handoffs"] = [{"role": h.get("role"), "allowed": h.get("allowed"), "refused": len(h.get("refused_columns") or [])}
                           for h in r.get("handoffs") or []]
    return out


def store(rows: list[dict[str, Any]], result: str, models: dict[str, Optional[str]], seed: int = 0, spent_eur: float = 0.0,
          live: bool = False) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    n = 1 + max([_number(p) for p in RESULTS.glob("HRN-*.json")] or [0])
    ident = f"HRN-{n:03d}"
    doc = {"id": ident, "campaign_id": "harness_v1", "kind": "harness_regression", "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "gate": GATE, "fingerprints": fingerprints(), "models": models, "live": live, "seed": seed, "spent_eur": round(spent_eur, 4),
           "rows": [_kept(r) for r in rows],
           "summary": {"cases": len(rows), "passed": sum(r["passed"] for r in rows),
                       "by_source": {s: [sum(r["passed"] for r in rows if r["source"] == s), sum(1 for r in rows if r["source"] == s)]
                                     for s in sorted({r["source"] for r in rows})}},
           "verdict": result, "status": "completed", "experiment_id": ident,
           "rule": "The suite says these written and recorded cases still go as they should with what this run ran; it is not a "
                   "claim about any other session or about production readiness."}
    path = RESULTS / f"{ident}.json"
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _number(path: Path) -> int:
    return int(path.name.split("_")[0].split(".")[0].split("-")[1])


def runs() -> list[dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(RESULTS.glob("HRN-*.json"), key=_number)] if RESULTS.exists() else []


def latest() -> Optional[dict[str, Any]]:
    done = runs()
    return done[-1] if done else None


def stale(run: Optional[dict[str, Any]] = None) -> list[str]:
    """What changed since the last run (or ``run``): the fingerprints that differ. Empty when it ran on what is here now."""
    run = run if run is not None else latest()
    if run is None:
        return ["no run of the suite"]
    now, then = fingerprints(), run.get("fingerprints") or {}
    return sorted(k for k in set(now) | set(then) if now.get(k) != then.get(k))


def prompts_unmeasured() -> list[str]:
    """The prompts changed since the latest run with a served model (only such a run sends them); empty when none
    changed, or when no model was ever measured here (then no prompt is served through the suite's gate either: a
    purpose is moved to a model only after a live run, ``passed_with``)."""
    live = [r for r in runs() if r.get("live")]
    if not live:
        return []
    now, then = fingerprints(), live[-1].get("fingerprints") or {}
    return sorted(k for k in set(now) | set(then) if k.startswith(PROMPT_KEYS) and now.get(k) != then.get(k))


def passed_with(purpose: str, model: str) -> Optional[dict[str, Any]]:
    """The latest passing run in which ``model`` served ``purpose`` on what is here now (its prompts and gates), or None."""
    for run in reversed(runs()):
        if (run.get("models") or {}).get(purpose) == model and str(run.get("verdict", "")).startswith("pass") and not stale(run):
            return run
    return None
