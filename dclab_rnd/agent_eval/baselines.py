"""Baselines on the frozen benchmark, judged by the gate (package 14.5).

The scripted policies (standard plan, audit, bad) are run by ``benchmark.py``. Here: a prompted base model (one request
per case, no tools, no audit flags: it sees column names, kinds, missing shares and distinct counts, as the leakage
review may, never a value), the current agent run (``agent_eval.live --split test``), and the report that sets every
policy against ``gate.py``. Every result is a file in evidence/campaigns/benchmark_v2/results; a run on the sealed test
set is always recorded, and a failure stays in the file.

    python -m dclab_rnd.agent_eval.baselines base-model --split test --cap-eur 0.5 --yes --output evidence/campaigns/benchmark_v2/results/BEN-003_….json
    python -m dclab_rnd.agent_eval.baselines report --scripted BEN-002… --base-model BEN-003… --agent BEN-004… --output …
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import benchmark, gate

RESULTS = Path(__file__).resolve().parents[2] / "evidence" / "campaigns" / benchmark.VERSION / "results"
SCOPE = "Evidence about these policies on these seeded cases, not about any model's readiness for production."


def _suite():
    return importlib.import_module("dclab_rnd.agent_eval.run")  # the package exports a function named run


def rows_summary(rows: list[dict[str, Any]], total: int, traps: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """One policy's summary in the shape the gate reads, and its per-family counts, from one row per scored case.
    A case with no row, or with an error, is counted as not scored (the gate fails it)."""
    suite = _suite()
    scored = [r for r in rows if not r.get("error") and r.get("ran", True)]
    leak = [r for r in scored if r.get("leak_caught") is not None]
    alarm = [r for r in scored if r.get("false_alarm") is not None]
    valid = [r["valid_moves"] for r in scored if r.get("valid_moves") is not None]
    caught, alarms = sum(bool(r["leak_caught"]) for r in leak), sum(bool(r["false_alarm"]) for r in alarm)
    summary = {"cases": total, "errors": total - len(scored),
               "leaks_caught": [caught, len(leak)], "leaks_caught_rate": round(caught / len(leak), 3) if leak else None, "leaks_caught_ci95": suite.wilson(caught, len(leak)),
               "false_alarms": [alarms, len(alarm)], "false_alarm_rate": round(alarms / len(alarm), 3) if alarm else None, "false_alarm_ci95": suite.wilson(alarms, len(alarm)),
               "valid_moves": round(sum(valid) / len(valid), 3) if valid else None,
               "citations_exist": [sum(bool(r.get("citations_exist")) for r in scored), len(scored)]}
    families = suite.by_trap_families([{**r, "policy": "p"} for r in scored], traps)["p"] if scored else {}
    return summary, families


def base_model_policy(client: Any, counter: dict[str, Any], max_requests: int, max_eur: float | None):
    """The prompted base model as a policy of the suite: one request, then the same moves the audit policy makes with
    its own list (set the solution, run the stages), so the engine's checks apply to it too."""
    from dclab_rnd import prompts

    def policy(box: Any, store: Any, pid: str, home: Path, case: Any) -> list[dict[str, Any]]:
        if counter["requests"] >= max_requests:
            raise RuntimeError(f"CapReached: the request cap ({max_requests}) was reached")
        if max_eur is not None and counter["eur"] >= max_eur:
            raise RuntimeError(f"CapReached: the euro cap ({max_eur}) was reached")
        described = box.call("describe_data", {"project_id": pid})
        shown = [{k: c.get(k) for k in ("name", "kind", "missing", "unique")} for c in described["columns"] if c["name"] != "target"]
        message = f"The prediction is made: {case.moment}\n\nColumns:\n" + json.dumps(shown)
        before = client.spent_eur
        counter["requests"] += 1
        try:
            reply = client.complete([{"role": "system", "content": prompts.text("base_audit")}, {"role": "user", "content": message}], None, 1500,
                                    response_format={"type": "json_object"})
        finally:
            counter["eur"] += client.spent_eur - before
        answer = json.loads(re.search(r"\{.*\}", reply.get("content") or "", re.S).group(0))
        names = {c["name"] for c in described["columns"]} - {"target"}
        forbidden = [{"column": f["column"], "reason": str(f.get("reason") or "kept out by the base model")[:200]}
                     for f in answer.get("forbidden") or [] if isinstance(f, dict) and f.get("column") in names]
        identifiers = [c for c in answer.get("identifiers") or [] if c in names]
        time_column = answer.get("time_column") if answer.get("time_column") in names else None
        group_column = answer.get("group_column") if answer.get("group_column") in names else None
        roles = {time_column, group_column, *identifiers}
        unique = next(c["unique"] for c in described["columns"] if c["name"] == "target")
        saved = box.call("set_solution", {k: v for k, v in {
            "project_id": pid, "target": "target", "task": "binary" if unique == 2 else "regression", "prediction_moment": case.moment,
            "forbidden": [f for f in forbidden if f["column"] not in roles], "identifiers": identifiers,
            "time_column": time_column, "group_column": group_column}.items() if v not in (None, [], "")})
        if "error" in saved:
            raise RuntimeError("the base model's solution was refused: " + str(saved["error"])[:200])
        box.call("run_all", {"project_id": pid})
        return []
    return policy


def run_base_model(gateway: Any, split_name: str, max_requests: int, max_eur: float | None, log=print) -> dict[str, Any]:
    from unittest import mock

    suite = _suite()
    cases = benchmark.split(split_name)
    counter = {"requests": 0, "eur": 0.0}
    client = gateway.client("leakage_review")
    suite.POLICIES["base_model"] = base_model_policy(client, counter, max_requests, max_eur)
    known = suite._known_ids()
    results = []
    try:  # the stages' notes and explanations would cost requests and do not change a move: off, as in the live agent run
        with mock.patch("dclab_rnd.studio.agent._chat", lambda *a, **k: None), mock.patch("dclab_rnd.studio.agent.explain_stage", lambda *a, **k: None), \
                tempfile.TemporaryDirectory() as tmp:
            for case in cases:
                result = suite.run_case(case, "base_model", Path(tmp), known)
                results.append(result)
                log(f"  {case.id} leak={result.leak_caught!s:5s} false_alarm={result.false_alarm!s:5s} kept_out={','.join(result.kept_out)[:60]}"
                    + (f"  ERROR {result.error[:80]}" if result.error else ""))
    finally:
        suite.POLICIES.pop("base_model", None)
    rows = [asdict(r) for r in results]
    summary, families = rows_summary(rows, len(cases), {c.id: c.trap for c in cases})
    return {"campaign_id": benchmark.VERSION, "kind": "benchmark_base_model", "status": "completed", "suite": f"{benchmark.VERSION}:{split_name}",
            "split": split_name, "sealed": split_name == "test", "model": client.model if hasattr(client, "model") else None,
            "prompt_sha256": __import__("hashlib").sha256((Path(__file__).resolve().parents[1] / "prompts" / "base_audit.md").read_bytes()).hexdigest()[:16],
            "question": "A base model, prompted once per case with the column names and summaries and no tools, on the benchmark: what does it keep out?",
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "requests": counter["requests"], "spent_eur": round(counter["eur"], 6),
            "cases": [{"id": c.id, "trap": c.trap, "fingerprint": c.fingerprint()} for c in cases], "results": rows,
            "summary": {"base_model": summary}, "by_family": {"base_model": families},
            "limitations": [SCOPE, "One request per case, no repeats: a model's answers vary, and the interval is over cases, not over repeats.",
                            "The base model sees names, kinds, missing shares and distinct counts, never values or the audit's flags.",
                            "A leak counts as caught when the column is kept out of the model; the reason given is not checked."]}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def live_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    """The live agent run's rows (one per case and repeat) in the shape the gate reads."""
    return [{"case": r["case"], "ran": r.get("ran"), "error": None if r.get("ran") else r.get("why"), **{k: r.get(k) for k in
            ("leak_caught", "false_alarm", "valid_moves", "citations_exist")}} for r in report["runs"]]


def _sealed(report: dict[str, Any], label: str) -> None:
    """A result is judged as the sealed test set's only if it is a run on exactly that set: the split says test and its
    cases (ids and fingerprints) are the frozen ones. A dev run, a subset or a changed table is refused, not judged."""
    expected = {c.id: c.fingerprint() for c in benchmark.split("test")}
    got = {c["id"]: c.get("fingerprint") for c in report.get("cases") or []}
    if report.get("split") != "test" or got != expected:
        raise ValueError(f"{label} is not a run on the sealed test set (split {report.get('split')!r}, {len(got)} cases of {len(expected)}, "
                         f"{'the fingerprints differ' if set(got) == set(expected) else 'other cases'}): it is not judged as one")


def _control_alarms(rows: list[dict[str, Any]], controls: set[str]) -> list[int]:
    """False alarms over the control cases only (no leak, no time or group column to declare): [alarms, controls scored]."""
    scored = [r for r in rows if r["case"] in controls and not r.get("error") and r.get("ran", True) and r.get("false_alarm") is not None]
    return [sum(bool(r["false_alarm"]) for r in scored), len(scored)]


def build_report(scripted: dict[str, Any], base_model: dict[str, Any] | None, agent: dict[str, Any] | None) -> dict[str, Any]:
    """Every policy against the gate, per family, with intervals; each failure stays in its own file. Every input must be
    a run on the whole sealed test set, and a case that did not run counts against its policy."""
    import hashlib

    test = benchmark.split("test")
    traps = {c.id: c.trap for c in test}
    controls = {c.id for c in test if not c.leaks and not c.time_column and not c.group_column}
    total = len(traps)
    _sealed(scripted, "the scripted result")
    policies: dict[str, Any] = {}
    for name in ("standard", "audit", "bad"):
        rows = [r for r in scripted["results"] if r["policy"] == name]
        policies[name] = {"summary": {**scripted["summary"][name], "cases": total}, "families": scripted["by_family"][name], "source": scripted["experiment_id"],
                          "false_alarms_on_controls": _control_alarms(rows, controls)}
    if base_model:
        _sealed(base_model, "the base-model result")
        policies["base_model"] = {"summary": {**base_model["summary"]["base_model"], "cases": total}, "families": base_model["by_family"]["base_model"],
                                  "source": base_model["experiment_id"], "false_alarms_on_controls": _control_alarms(base_model["results"], controls),
                                  "prompt_sha256": base_model.get("prompt_sha256") or "not recorded in the result; the prompt file now has " +
                                  hashlib.sha256((Path(__file__).resolve().parents[1] / "prompts" / "base_audit.md").read_bytes()).hexdigest()[:16]}
    repeats = 1
    if agent:
        _sealed(agent, "the live agent result")
        repeats = max(1, int((agent.get("plan") or {}).get("repeats", 1)))
        rows = live_rows(agent)
        summary, families = rows_summary(rows, total * repeats, traps)
        policies["agent"] = {"summary": summary, "families": families, "source": agent["experiment_id"], "model": agent.get("model"),
                             "stopped": agent.get("stopped"), "steps_ended": _ended(agent), "false_alarms_on_controls": _control_alarms(rows, controls),
                             "tools_reasoning_effort": agent.get("tools_reasoning_effort")}
    for p in policies.values():
        p["verdict"] = gate.verdict(p["summary"], p["families"], p["summary"]["cases"])
    return {"campaign_id": benchmark.VERSION, "kind": "benchmark_gate", "gate": gate.VERSION, "thresholds": gate.THRESHOLDS, "split": "test", "sealed": True,
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "policies": policies,
            "passes": sorted(n for n, p in policies.items() if p["verdict"]["pass"]),
            "limitations": [SCOPE, "The labels are the author's until package 14.4.",
                            "The scripted policies' sealed-set numbers (BEN-002) were recorded before the gate was written; the gate's thresholds come from the package text.",
                            "Cases from one base share its columns and generator, so they are not independent: the intervals are wider than they look.",
                            "The gate counts a false alarm over every case that has a clean column to lose (70 of 70); `false_alarms_on_controls` gives the same count over the "
                            "control cases only (no leak, no time or group column), which is what the package text literally says. The verdict uses the first.",
                            "The base model's prompt describes the kinds of leak in general words (a copy of the target in another unit, a sum, a value that exists only when the "
                            "outcome happened): it is a prompted baseline with hints, not a bare one, and it has no tools; the agent has its tools and 16 steps.",
                            *([f"{repeats} repeats per case are pooled as independent trials in the intervals, which they are not: read them as too narrow."] if repeats > 1 else [])]}


def _ended(agent: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in agent.get("runs") or []:
        key = r.get("status") or ("not run" if not r.get("ran") else "?")
        out[key] = out.get(key, 0) + 1
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.baselines", description="Baselines on the frozen benchmark (14.5).")
    sub = parser.add_subparsers(dest="command", required=True)
    base = sub.add_parser("base-model", help="a prompted base model, one request per case")
    base.add_argument("--split", choices=("dev", "test"), required=True)
    base.add_argument("--cap-eur", type=float, required=True)
    base.add_argument("--max-requests", type=int, default=None)
    base.add_argument("--yes", action="store_true")
    base.add_argument("--output", type=Path, required=True)
    rep = sub.add_parser("report", help="every policy against the gate")
    rep.add_argument("--scripted", type=Path, required=True)
    rep.add_argument("--base-model", type=Path)
    rep.add_argument("--agent", type=Path)
    rep.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"{args.output} exists; results are never overwritten")
    if args.command == "base-model":
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
        from dclab_rnd.models.gateway import for_workspace

        gateway = for_workspace()
        gateway.shadows = None
        cases = benchmark.split(args.split)
        limit = args.max_requests or len(cases)
        print(f"A prompted base model on {len(cases)} {args.split} cases: at most {limit} requests, no tools, cap €{args.cap_eur}. {SCOPE}")
        if not gateway.available("leakage_review"):
            print("No model serves the leakage review purpose; nothing was sent.", file=sys.stderr)
            return 2
        from dclab_rnd.models import prices, settings

        probe = gateway.client("leakage_review")
        if not settings.public(probe.tier.get("name") or "standard")["local"] and prices.load().get(probe.model) is None:
            print(f"The model {probe.model} has no price in the price table, so a euro cap could not stop anything; nothing was sent.", file=sys.stderr)
            return 2
        if not args.yes:
            print("Nothing was sent. Run again with --yes to start.", file=sys.stderr)
            return 2
        report = run_base_model(gateway, args.split, limit, args.cap_eur)
        s = report["summary"]["base_model"]
        print(f"base_model leaks caught {s['leaks_caught']} (95% {s['leaks_caught_ci95']}) false alarms {s['false_alarms']} errors {s['errors']} · "
              f"{report['requests']} requests · €{report['spent_eur']}")
    else:
        scripted = _load(args.scripted)
        report = build_report(scripted, _load(args.base_model) if args.base_model else None, _load(args.agent) if args.agent else None)
        for name, p in report["policies"].items():
            failed = [c["criterion"] for c in p["verdict"]["criteria"] if not c["pass"]]
            print(f"{name:11s} {'PASS' if p['verdict']['pass'] else 'FAIL'}" + (f"  failed: {'; '.join(failed)}" if failed else ""))
        print(SCOPE)
    found = re.match(r"([A-Z]+-\d+)_", args.output.name)
    report = {"experiment_id": found.group(1) if found else args.output.stem, **report}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
        handle.write("\n")
    print(f"Stored {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
