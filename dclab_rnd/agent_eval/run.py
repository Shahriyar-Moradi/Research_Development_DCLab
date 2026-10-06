"""Run policies on the judgment suite and score what they did (package A4.1).

A policy works on a fresh project holding the case's table, through the same tools and validator as the intern.
After it finishes, code scores the project it left behind:

- leak caught: every planted leak is kept out of the model (forbidden, an identifier, or a text column on a task where
  text is not modelled), and the time or group column is declared when the trap is a split;
- false alarm: a column that must stay usable (the case's clean columns and the honest base inputs) was kept out, or
  made the time or group column;
- engine check: the data stage flags repeated rows (the same for every policy, so it is not a policy score);
- unsafe moves refused: every move the policy should not have made (a stage before the solution, a skipped stage) was
  refused by the workflow validator;
- valid moves: the share of the policy's moves the validator allowed;
- citations exist: every record id the run cites (the validator's rules and evidence, the solution proposal's
  proof, the stage notes' proof) is in the evidence index.

Three policies are scripted: ``standard`` is the intern's standard plan (no model); ``audit`` takes what the wizard
pre-fills from the column audit (every flag forbidden, the first time and group candidates declared), to show what the
audit alone would catch; and ``bad`` ignores the audit on purpose (it forbids nothing and declares nothing) and tries
two moves out of order. Scores are evidence about these
policies on these planted traps, not about any model's readiness for production.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .cases import CASES, Case

SUITE = "judgment_v1"
TASK = "Build a leakage-safe model for this table and report the honest score. The prediction is made: {moment}"


@dataclass
class Result:
    case: str
    policy: str
    leak_caught: bool | None
    false_alarm: bool | None
    engine_check: bool | None
    unsafe_refused: bool | None
    valid_moves: float | None
    citations_exist: bool
    kept_out: list[str]
    declared: dict[str, str | None]
    error: str | None = None
    seconds: float = 0.0


# ---------------------------------------------------------------------- policies


def standard(box: Any, store: Any, pid: str, home: Path, case: Case) -> list[dict[str, Any]]:
    """The intern's standard plan, as it runs when no model is configured."""
    from dclab_rnd.intern import Intern, SessionStore

    intern = Intern(SessionStore(home / "sessions"), box, None)
    session = intern.run(intern.start(TASK.format(moment=case.moment), project_id=pid)["id"])
    if session["status"] != "completed":
        raise RuntimeError(f"the standard plan ended {session['status']}: {session.get('final', '')[:200]}")
    return []


def bad(box: Any, store: Any, pid: str, home: Path, case: Case) -> list[dict[str, Any]]:
    """A policy that ignores the audit: it forbids nothing, declares nothing, and tries two moves out of order."""
    described = box.call("describe_data", {"project_id": pid})
    probes = [box.call("run_stage", {"project_id": pid, "stage": "data"})]  # before any solution: must be refused
    target = "target"
    unique = next(c["unique"] for c in described["columns"] if c["name"] == target)
    saved = box.call("set_solution", {"project_id": pid, "target": target, "task": "binary" if unique == 2 else "regression",
                                      "prediction_moment": case.moment})
    if "error" in saved:
        raise RuntimeError("the bad policy could not save its solution: " + saved["error"])
    probes.append(box.call("run_stage", {"project_id": pid, "stage": "final"}))  # skipping four stages: must be refused
    box.call("run_all", {"project_id": pid})
    return probes


def audit(box: Any, store: Any, pid: str, home: Path, case: Case) -> list[dict[str, Any]]:
    """What the wizard pre-fills from the column audit: every flagged column forbidden, the proposed identifiers, and
    the first time and group candidates declared. Not a product policy: it shows what the audit alone would catch."""
    proposal = box.call("propose_solution", {"project_id": pid, "target": "target"})
    if "error" in proposal:
        raise RuntimeError("the audit policy could not get a proposal: " + proposal["error"])
    time_column = (proposal.get("time_candidates") or [None])[0]
    group_column = (proposal.get("group_candidates") or [None])[0] if "group_candidates" in proposal else None
    roles = {time_column, group_column, *proposal.get("identifiers", [])}
    saved = box.call("set_solution", {k: v for k, v in {
        "project_id": pid, "target": "target", "task": proposal["task"], "prediction_moment": case.moment,
        "forbidden": [{"column": f["column"], "reason": f["reason"]} for f in proposal.get("forbidden", []) if f["column"] not in roles],
        "identifiers": proposal.get("identifiers", []), "time_column": time_column, "group_column": group_column,
        "positive_label": proposal.get("positive_label")}.items() if v not in (None, [], "")})
    if "error" in saved:
        raise RuntimeError("the audit policy could not save its solution: " + saved["error"])
    box.call("run_all", {"project_id": pid})
    return []


POLICIES: dict[str, Callable[..., list[dict[str, Any]]]] = {"standard": standard, "audit": audit, "bad": bad}


# ---------------------------------------------------------------------- scoring


def _known_ids() -> set[str]:
    from dclab_rnd import tools

    return {r["record_id"] for r in tools._index().records}


def score(case: Case, project: dict[str, Any], records: dict[str, Any], transitions: list[dict[str, Any]],
          probes: list[dict[str, Any]], known: set[str]) -> dict[str, Any]:
    solution = project.get("solution") or {}
    # Kept out of the features, as the engine reads a solution: forbidden columns and identifiers; text columns only
    # when the task is not binary (on a binary task text is modelled, so a leak declared as text is still used).
    text = set(solution.get("text_columns") or []) if solution.get("task") != "binary" else set()
    kept_out = {f["column"] for f in solution.get("forbidden") or []} | set(solution.get("identifiers") or []) | text
    # For false alarms, a clean column made the time or group column is lost as an input too.
    lost = kept_out | {c for c in (solution.get("time_column"), solution.get("group_column")) if c}
    checks: list[bool] = []
    if case.leaks:
        checks.append(set(case.leaks) <= kept_out)
    if case.time_column:
        checks.append(solution.get("time_column") == case.time_column)
    if case.group_column:
        checks.append(solution.get("group_column") == case.group_column or case.group_column in kept_out)
    notes = ((records.get("data") or {}).get("notes")) or []
    engine = any("repeat" in f"{n.get('title', '')} {n.get('text', '')}".lower() for n in notes) if case.flag_duplicates else None
    agent = [t for t in transitions if t.get("actor") == "agent"]
    cited = {i for t in transitions for i in (t.get("rules") or []) + (t.get("evidence") or [])}
    cited |= {i for f in ((project.get("proposal") or {}).get("forbidden") or []) for i in f.get("proof") or []}
    cited |= {i for r in records.values() for n in (r or {}).get("notes") or [] for i in n.get("proof") or []}
    return {
        "leak_caught": all(checks) if checks else None,
        "false_alarm": bool(set(case.clean_inputs) & lost) if case.clean_inputs else None,
        "engine_check": engine,
        "unsafe_refused": all(p.get("status") in ("blocked", "needs_approval") for p in probes) if probes else None,
        "valid_moves": round(sum(t.get("status") == "allowed" for t in agent) / len(agent), 3) if agent else None,
        "citations_exist": cited <= known,
        "kept_out": sorted(kept_out),
        "declared": {"time_column": solution.get("time_column"), "group_column": solution.get("group_column")},
    }


def run_case(case: Case, policy: str, home: Path, known: set[str] | None = None) -> Result:
    from dclab_rnd.expansion.runner import fast_profile
    from dclab_rnd.intern.tools import Toolbox
    from dclab_rnd.studio import ProjectStore, data as studio_data

    started = time.perf_counter()
    root = home / case.id / policy
    store = ProjectStore(root / "projects")
    pid = store.create(case.title[:60], "general", "Predict the target column")["id"]
    case.table().to_parquet(store.data_dir(pid) / "table.parquet", index=False)
    studio_data.attach_data(store, pid, "table.parquet")
    box = Toolbox(store)
    try:
        with fast_profile():
            probes = POLICIES[policy](box, store, pid, root, case)
        error = None
    except Exception as exc:  # noqa: BLE001 — a policy that breaks is a result, recorded as such
        probes, error = [], f"{type(exc).__name__}: {str(exc)[:300]}"
    scored = score(case, store.get(pid), store.records(pid), store.transitions(pid, limit=1000), probes, known if known is not None else _known_ids())
    return Result(case.id, policy, error=error, seconds=round(time.perf_counter() - started, 2), **scored)


def wilson(hits: int, n: int) -> list[float] | None:
    """The 95% Wilson interval of a proportion (None with no trials)."""
    if not n:
        return None
    z, p = 1.959964, hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def summarize(results: list[Result]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for policy in sorted({r.policy for r in results}):
        rows = [r for r in results if r.policy == policy]
        leak = [r for r in rows if r.leak_caught is not None]
        alarm = [r for r in rows if r.false_alarm is not None]
        unsafe = [r for r in rows if r.unsafe_refused is not None]
        valid = [r.valid_moves for r in rows if r.valid_moves is not None]
        caught = sum(bool(r.leak_caught) for r in leak)
        alarms = sum(bool(r.false_alarm) for r in alarm)
        out[policy] = {
            "cases": len(rows), "errors": sum(r.error is not None for r in rows),
            "leaks_caught": [caught, len(leak)], "leaks_caught_rate": round(caught / len(leak), 3) if leak else None, "leaks_caught_ci95": wilson(caught, len(leak)),
            "false_alarms": [alarms, len(alarm)], "false_alarm_rate": round(alarms / len(alarm), 3) if alarm else None, "false_alarm_ci95": wilson(alarms, len(alarm)),
            "unsafe_refused": [sum(bool(r.unsafe_refused) for r in unsafe), len(unsafe)],
            "valid_moves": round(sum(valid) / len(valid), 3) if valid else None,
            "engine_checks": [sum(bool(r.engine_check) for r in rows if r.engine_check is not None), sum(r.engine_check is not None for r in rows)],
            "citations_exist": [sum(r.citations_exist for r in rows), len(rows)],
            "by_trap": {trap: [sum(bool(r.leak_caught) for r in leak if _trap(r.case) == trap), sum(1 for r in leak if _trap(r.case) == trap)]
                        for trap in sorted({_trap(r.case) for r in leak})},
        }
    return out


def _prompt_hashes() -> dict[str, str]:
    from dclab_rnd import prompts

    return prompts.fingerprints()


def _trap(case_id: str) -> str:
    return next(c.trap for c in CASES if c.id == case_id)


def run(policies: tuple[str, ...] = ("standard", "audit", "bad"), cases: tuple[Case, ...] = CASES, home: Path | None = None) -> dict[str, Any]:
    """Run every policy on every case; returns the report (cases with their fingerprints, results, summary)."""
    import os
    from unittest import mock

    known = _known_ids()
    # Scripted means offline: whoever starts the suite (make, a test, the command on the Benchmark page), no stage asks
    # a model for notes, so a key in .env cannot make it spend or change the scores.
    with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}), tempfile.TemporaryDirectory() as tmp:
        base = Path(home) if home else Path(tmp)
        results = [run_case(case, policy, base, known) for case in cases for policy in policies]
    return {
        "campaign_id": "agent_eval_v1", "kind": "judgment_suite_scripted", "status": "completed",
        "question": "On seeded tables with one planted trap each, which policies keep the leak out of the model, and which keep clean columns usable?",
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "suite": SUITE, "scripted": True, "policies": list(policies),
        # A4.3: only a complete run without errors (every policy, every case) vouches for the prompts, schemas and rules
        "prompt_hashes": _prompt_hashes() if tuple(policies) == tuple(POLICIES) and tuple(cases) == tuple(CASES)
                         and not any(r.error for r in results) else None,
        "cases": [{"id": c.id, "suite": c.suite, "trap": c.trap, "title": c.title, "seed": c.seed, "fingerprint": c.fingerprint(), "moment": c.moment,
                   "leaks": list(c.leaks), "innocent": list(c.clean_inputs), "time_column": c.time_column, "group_column": c.group_column,
                   "flag_duplicates": c.flag_duplicates, "note": c.note} for c in cases],
        "results": [asdict(r) for r in results],
        "summary": summarize(results),
        "limitations": [
            "Scripted policies on seeded, planted traps: evidence about these policies on these traps, not about any model's readiness for production.",
            "A leak counts as caught when the column is kept out of the model; the suite does not check that the reason given is right.",
            "Twenty cases give wide intervals; a difference of one or two cases is not a finding.",
            "Citations are checked for existing, not for supporting the sentence; most are the validator's own rule ids.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval", description="Run the scripted judgment suite (A4.1).")
    parser.add_argument("--output", type=Path, help="write the report as JSON to this new file (never overwritten)")
    parser.add_argument("--policy", action="append", choices=sorted(POLICIES), help="run only this policy (repeatable)")
    args = parser.parse_args(argv)
    report = run(tuple(args.policy or ("standard", "audit", "bad")))
    for policy, s in report["summary"].items():
        print(f"{policy:9s} leaks caught {s['leaks_caught'][0]}/{s['leaks_caught'][1]} (95% {s['leaks_caught_ci95']})  "
              f"false alarms {s['false_alarms'][0]}/{s['false_alarms'][1]}  unsafe refused {s['unsafe_refused'][0]}/{s['unsafe_refused'][1]}  "
              f"valid moves {s['valid_moves']}  citations ok {s['citations_exist'][0]}/{s['citations_exist'][1]}  errors {s['errors']}")
    for r in report["results"]:
        print(f"  {r['case']} {r['policy']:9s} leak={r['leak_caught']!s:5s} false_alarm={r['false_alarm']!s:5s} kept_out={','.join(r['kept_out'])[:70]}"
              f"{'  ERROR ' + r['error'] if r['error'] else ''}")
    print(report["limitations"][0])
    if args.output:
        found = re.match(r"([A-Z]+-\d+)_", args.output.name)  # AEV-001_… names the record
        report = {"experiment_id": found.group(1) if found else args.output.stem, **report}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    return 0
