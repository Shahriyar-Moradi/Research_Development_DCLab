"""The judgment suite with a live model (package A4.2): the intern, driven by the configured model, on the A4.1 cases.

A model's answer varies, so every case runs several times (``--repeats``, 5 by default) and each score is reported
as a mean over the repeats with a 95% interval. Nothing starts without ``--yes``: first the run prints what it will
do (the model, the number of runs, an upper bound on requests, the caps) and stops. It refuses to exceed its caps:
``--max-requests`` (always set; the default is the upper bound it printed) counts every request of the suite. Only the
intern talks to the model: for the suite, the tools that would ask a model of their own (the leakage review, a project
answer, the stages' notes and explanations) take their deterministic path, so the intern's session caps and its
accounting cover every request. ``--cap-eur`` is required when the intern's model is a priced remote one, and refused
for an unpriced one (it could not stop anything). A request past a
cap is never sent. A run that a cap or an error cut short is recorded as not run, never scored, and the suite stops.

The result is stored under evidence/campaigns/agent_eval_v1/results with a run id and the model's name, so later runs
can be compared with it. Scores on these cases are evidence about this model on these planted traps, not about any
model's readiness for production.

    python -m dclab_rnd.agent_eval.live                       # prints the plan and stops
    python -m dclab_rnd.agent_eval.live --yes --cap-eur 2      # runs it
    make agent-eval-live ARGS="--yes --repeats 3 --cases AE-01,AE-11"
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cases import CASES, Case
from .run import SUITE, TASK, _known_ids, score

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evidence" / "campaigns" / "agent_eval_v1" / "results"
PURPOSES = ("intern",)  # the only purpose the suite sends for: every other model use is off for the suite
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}  # t quantiles by degrees of freedom
SCOPE = "Scores on these cases are evidence about this model on these planted traps, not about any model's readiness for production."


class CapReached(RuntimeError):
    pass


def interval(values: list[float]) -> dict[str, Any]:
    """Mean and 95% t-interval over repeats (the spread a model's varying answers give)."""
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": None, "ci95": None, "n": 0}
    mean = statistics.fmean(values)
    if len(values) < 2:
        return {"mean": round(mean, 3), "ci95": None, "n": 1}
    half = T95.get(len(values) - 1, 1.96) * statistics.stdev(values) / math.sqrt(len(values))
    return {"mean": round(mean, 3), "ci95": [round(max(0.0, mean - half), 3), round(min(1.0, mean + half), 3)], "n": len(values)}


def _run_chars(max_steps: int) -> int:
    """An upper bound on the characters one run sends: request k carries the policy, the tool schemas, the task and k
    tool results of at most RESULT_CHARS each (the conversation grows), for k = 0 … max_steps + 1."""
    from dclab_rnd.agents import default_registry
    from dclab_rnd.intern.loop import POLICY, SESSION_TOOLS
    from dclab_rnd.intern.tools import RESULT_CHARS

    registry = default_registry()
    schemas = len(json.dumps(registry.schemas(names=[*registry.names("project"), *SESSION_TOOLS])))
    base = len(POLICY) + schemas + 1_000
    return sum(base + k * (RESULT_CHARS + 400) for k in range(max_steps + 2))


def plan(gateway: Any, cases: list[Case], repeats: int, max_steps: int) -> dict[str, Any]:
    """What a run would do, before anything is sent: the model, the runs and upper bounds on requests, tokens and euros."""
    from dclab_rnd.models import prices, settings

    tiers = {}
    for purpose in PURPOSES:
        t = settings.public(settings.purpose(purpose).tier)
        tiers[purpose] = {"tier": settings.purpose(purpose).tier, "model": t.get("model"), "local": bool(t.get("local"))}
    intern = tiers["intern"]
    runs = len(cases) * repeats
    per_run_requests = max_steps + 2  # each tool call needs a reply, plus the first reply and the last answer
    requests = runs * per_run_requests
    chars = _run_chars(max_steps)
    try:
        per_run_eur = prices.estimate(intern["model"], intern["local"], chars, 1800 * per_run_requests)
        priced = {p: prices.estimate(t["model"], t["local"], 1000, 1) is not None for p, t in tiers.items()}
    except ValueError:
        per_run_eur, priced = None, {p: False for p in tiers}
    return {"model": intern["model"], "tier": intern["tier"], "local": intern["local"], "available": gateway.available("intern"),
            "tiers": tiers, "remote_priced": sorted(p for p, t in tiers.items() if priced[p] and not t["local"]),
            "remote_unpriced": sorted(p for p, t in tiers.items() if not priced[p] and not t["local"]),
            "cases": [c.id for c in cases], "repeats": repeats, "runs": runs, "max_steps": max_steps, "max_requests": requests,
            "max_input_tokens": runs * chars // 3, "max_eur": round(per_run_eur * runs, 2) if per_run_eur is not None else None,
            "priced": per_run_eur is not None}


def _counted(gateway: Any, counter: dict[str, int], max_requests: int) -> None:
    """Every request this gateway sends, from any purpose, is counted at the transport (so after the gateway's own
    budget check accepted it) and stopped once the suite's cap is reached."""
    make = gateway.transport

    class Counted:
        def __init__(self, inner: Any):
            self.inner = inner

        def complete(self, *args: Any, **kwargs: Any) -> Any:
            if counter["requests"] >= max_requests:
                error = RuntimeError("BudgetExceeded: the suite's request cap is reached")
                error.transient = False  # type: ignore[attr-defined]
                raise error
            counter["requests"] += 1
            return self.inner.complete(*args, **kwargs)

    gateway.transport = lambda tier, purpose: Counted(make(tier, purpose))


class _NoModels:
    """The installed gateway during the suite: every purpose answers "no model", so only the intern (which holds its
    own client from the suite's gateway) sends requests, and its session's caps and accounting cover them all."""

    def __init__(self, gateway: Any):
        self._gateway = gateway

    def client(self, *args: Any, **kwargs: Any) -> None:
        return None

    def available(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._gateway, name)


def _cut_short(session: dict[str, Any]) -> str | None:
    """Why a run cannot be scored: an error, or a cap that stopped it (a run that used its own tool-call or minute
    budget is a real result: the model spent it)."""
    if session["status"] == "failed":
        return "the run failed: " + str(session.get("error") or session.get("final") or "")[:160]
    final = str(session.get("final") or "")
    if session["status"] == "budget_exhausted" and "tool calls is used up" not in final and "minutes is used up" not in final:
        return "a cap stopped it: " + final[:160]
    return None


def run_live(gateway: Any, cases: list[Case], repeats: int, max_requests: int, cap_eur: float | None, max_steps: int,
             home: Path | None = None, log=print) -> dict[str, Any]:
    from dclab_rnd.expansion.runner import fast_profile
    from dclab_rnd.intern import Intern, SessionStore
    from dclab_rnd.intern.tools import Toolbox
    from unittest import mock

    from dclab_rnd.models import gateway as gateway_module, install
    from dclab_rnd.studio import ProjectStore, data as studio_data

    known = _known_ids()
    counter = {"requests": 0}
    spent = {"eur": 0.0, "input_tokens": 0, "output_tokens": 0}
    runs: list[dict[str, Any]] = []
    stopped = None
    _counted(gateway, counter, max_requests)
    previous = gateway_module._INSTALLED
    install(_NoModels(gateway))  # tools that ask a model of their own (installed()) get none: their deterministic path
    # the stages' notes and explanations are switched off too: they would cost requests and do not change a move
    quiet = (mock.patch("dclab_rnd.studio.agent._chat", lambda *a, **k: None), mock.patch("dclab_rnd.studio.agent.explain_stage", lambda *a, **k: None))
    for patch in quiet:
        patch.start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(home) if home else Path(tmp)
            for repeat in range(repeats):
                for case in cases:
                    if stopped:
                        runs.append({"case": case.id, "repeat": repeat, "ran": False, "why": stopped})
                        continue
                    root = base / f"{case.id}-{repeat}"
                    store = ProjectStore(root / "projects")
                    pid = store.create(case.title[:60], "general", "Predict the target column")["id"]
                    case.table().to_parquet(store.data_dir(pid) / "table.parquet", index=False)
                    studio_data.attach_data(store, pid, "table.parquet")
                    client = gateway.client("intern", project_id=pid)
                    budget: dict[str, Any] = {"max_steps": max_steps, "max_minutes": 10}
                    if cap_eur is not None:
                        budget["max_eur"] = max(0.0, cap_eur - spent["eur"])  # what is left of the suite's euro cap
                    intern = Intern(SessionStore(root / "sessions"), Toolbox(store), client)
                    started = time.perf_counter()
                    with fast_profile():
                        session = intern.run(intern.start(TASK.format(moment=case.moment), budget, project_id=pid)["id"])
                    used = session.get("used") or {}
                    spent["eur"] += float(used.get("eur") or 0.0)
                    spent["input_tokens"] += int(used.get("input_tokens") or 0)
                    spent["output_tokens"] += int(used.get("output_tokens") or 0)
                    why = _cut_short(session)
                    if why:  # never scored: a cap or an error is not the model's judgment
                        runs.append({"case": case.id, "repeat": repeat, "ran": False, "why": why, "status": session["status"], "tokens": [used.get("input_tokens"), used.get("output_tokens")]})
                        stopped = stopped or why
                        log(f"  {case.id} #{repeat + 1} not scored: {why[:100]}")
                        continue
                    scored = score(case, store.get(pid), store.records(pid), store.transitions(pid, limit=1000), [], known)
                    runs.append({"case": case.id, "repeat": repeat, "ran": True, "status": session["status"], "steps": used.get("steps"),
                                 "tokens": [used.get("input_tokens"), used.get("output_tokens")], "eur": used.get("eur"),
                                 "seconds": round(time.perf_counter() - started, 1), **{k: scored[k] for k in ("leak_caught", "false_alarm", "valid_moves",
                                                                                                                "citations_exist", "kept_out", "declared")}})
                    log(f"  {case.id} #{repeat + 1} {session['status']:16s} leak={scored['leak_caught']!s:5s} false_alarm={scored['false_alarm']!s:5s} "
                        f"kept_out={','.join(scored['kept_out'])[:60]}")
                    if counter["requests"] >= max_requests:
                        stopped = f"the request cap ({max_requests}) was reached"
                    elif cap_eur is not None and spent["eur"] >= cap_eur:
                        stopped = f"the euro cap ({cap_eur}) was reached"
    finally:
        for patch in quiet:
            patch.stop()
        install(previous)
    return {"runs": runs, "requests": counter["requests"], "spent": {**spent, "eur": round(spent["eur"], 6)}, "stopped": stopped}


def summarize(runs: list[dict[str, Any]], cases: list[Case], repeats: int) -> dict[str, Any]:
    done = [r for r in runs if r.get("ran")]
    per_repeat = []
    complete = [k for k in range(repeats) if sum(1 for r in done if r["repeat"] == k) == len(cases)]
    for repeat in complete:  # a repeat with a case not run would average a different set of cases
        rows = [r for r in done if r["repeat"] == repeat]
        leak = [r for r in rows if r["leak_caught"] is not None]
        alarm = [r for r in rows if r["false_alarm"] is not None]
        valid = [r["valid_moves"] for r in rows if r["valid_moves"] is not None]
        per_repeat.append({"leaks_caught": sum(bool(r["leak_caught"]) for r in leak) / len(leak) if leak else None,
                           "false_alarms": sum(bool(r["false_alarm"]) for r in alarm) / len(alarm) if alarm else None,
                           "valid_moves": statistics.fmean(valid) if valid else None,
                           "citations_exist": sum(bool(r["citations_exist"]) for r in rows) / len(rows)})
    scores = {k: interval([p[k] for p in per_repeat]) for k in ("leaks_caught", "false_alarms", "valid_moves", "citations_exist")}
    by_case = {c.id: {"leak_caught": [sum(bool(r["leak_caught"]) for r in done if r["case"] == c.id and r["leak_caught"] is not None),
                                      sum(1 for r in done if r["case"] == c.id and r["leak_caught"] is not None)],
                      "false_alarm": [sum(bool(r["false_alarm"]) for r in done if r["case"] == c.id and r["false_alarm"] is not None),
                                      sum(1 for r in done if r["case"] == c.id and r["false_alarm"] is not None)]} for c in cases}
    return {"scores": scores, "by_case": by_case, "runs": len(done), "not_run": len(runs) - len(done), "complete_repeats": len(complete),
            "note": "Intervals are over complete repeats only; per-case counts include every scored run."}


def next_path(model: str) -> Path:
    numbers = [int(m.group(1)) for p in RESULTS.glob("AEV-*.json") if (m := re.match(r"AEV-(\d+)_", p.name))]
    slug = re.sub(r"[^a-z0-9]+", "-", str(model or "model").lower()).strip("-")[:40]
    return RESULTS / f"AEV-{max(numbers, default=0) + 1:03d}_{SUITE}_live_{slug}.json"


def main(argv: list[str] | None = None, gateway: Any = None) -> int:
    import argparse

    from dclab_rnd.models.gateway import for_workspace

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.live", description="Run the judgment suite with the configured model (A4.2).")
    parser.add_argument("--yes", action="store_true", help="run it; without this flag the plan is printed and nothing is sent")
    parser.add_argument("--repeats", type=int, default=5, help="runs per case (a model's answers vary; default 5)")
    parser.add_argument("--cases", help="comma-separated case ids (default: all)")
    parser.add_argument("--max-requests", type=int, help="stop before this many model requests (default: the printed upper bound)")
    parser.add_argument("--cap-eur", type=float, help="stop before the suite costs more than this (required for a priced remote model)")
    parser.add_argument("--max-steps", type=int, default=16, help="tool calls per run (default 16)")
    parser.add_argument("--output", type=Path, help="where to store the result (default: the next AEV file of the campaign)")
    args = parser.parse_args(argv)
    gateway = gateway or for_workspace()  # the workspace usage log and its monthly cap, as the other command-line tools
    wanted = {c.strip() for c in (args.cases or "").split(",") if c.strip()}
    cases = [c for c in CASES if not wanted or c.id in wanted]
    if wanted - {c.id for c in cases}:
        parser.error(f"unknown case(s): {', '.join(sorted(wanted - {c.id for c in cases}))}")
    if not 1 <= args.repeats <= 10:
        parser.error("--repeats must be between 1 and 10")
    if not 3 <= args.max_steps <= 80:
        parser.error("--max-steps must be between 3 and 80 (the session's own limits)")
    if args.max_requests is not None and args.max_requests < 1:
        parser.error("--max-requests must be at least 1")
    p = plan(gateway, cases, args.repeats, args.max_steps)
    max_requests = args.max_requests if args.max_requests is not None else p["max_requests"]
    print(f"Model {p['model']} ({'local' if p['local'] else 'remote'}, tier {p['tier']}) · {len(cases)} cases × {args.repeats} repeats = {p['runs']} runs")
    print(f"At most {p['max_requests']} requests, all by the intern ({p['max_steps']} tool calls a run), "
          f"at most about {p['max_input_tokens']:,} input tokens; "
          + (f"at most €{p['max_eur']} by the price table" if p["priced"] else "no price configured for the intern's model: cost is counted in tokens only"))
    print("Only the intern asks the model: the leakage review, project answers and stage notes use their deterministic path for the suite.")
    print(f"Caps: {max_requests} requests" + (f", €{args.cap_eur}" if args.cap_eur is not None else ""))
    print(SCOPE)
    if not p["available"]:
        print("No model serves the intern (see GET /api/models, or DCLAB_NO_LIVE_MODELS is set); nothing was sent.", file=sys.stderr)
        return 2
    if p["remote_priced"] and args.cap_eur is None:
        print("The intern's model is a priced remote one: give --cap-eur; nothing was sent.", file=sys.stderr)
        return 2
    if args.cap_eur is not None and not p["remote_priced"]:
        print("No model these runs use has a price, so a euro cap could not stop anything: use --max-requests; nothing was sent.", file=sys.stderr)
        return 2
    if not args.yes:
        print("Nothing was sent. Run again with --yes to start.", file=sys.stderr)
        return 2
    started = datetime.now(timezone.utc)
    live = run_live(gateway, cases, args.repeats, max_requests, args.cap_eur, args.max_steps)
    summary = summarize(live["runs"], cases, args.repeats)
    print("Score (mean over repeats, 95% interval):")
    for name, s in summary["scores"].items():
        print(f"  {name:16s} {s['mean']!s:6s} {s['ci95']}  (n={s['n']})")
    print(f"Requests {live['requests']} · tokens in {live['spent']['input_tokens']}, out {live['spent']['output_tokens']} · €{live['spent']['eur']}"
          + (f" · stopped: {live['stopped']}" if live["stopped"] else ""))
    print(SCOPE)
    report = {"experiment_id": None, "campaign_id": "agent_eval_v1", "kind": "judgment_suite_live", "status": "completed" if not live["stopped"] else "stopped",
              "run_id": uuid.uuid4().hex[:12], "model": p["model"], "tier": p["tier"], "local": p["local"], "suite": SUITE,
              "question": "On the judgment suite's planted traps, how often does the intern, driven by this model, keep the leak out and the clean columns in?",
              "started_at": started.isoformat(timespec="seconds"), "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "plan": {**p, "max_requests_cap": max_requests, "cap_eur": args.cap_eur}, "requests": live["requests"], "spent": live["spent"],
              "stopped": live["stopped"], "summary": summary, "runs": live["runs"],
              "cases": [{"id": c.id, "fingerprint": c.fingerprint()} for c in cases], "limitations": [SCOPE,
              f"{args.repeats} repeats per case: the interval is about this model's varying answers on these cases, not about other tables."]}
    path = args.output or next_path(p["model"])
    report["experiment_id"] = (re.match(r"([A-Z]+-\d+)_", path.name) or re.match(r"(.*)", path.stem)).group(1)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(f"Stored {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
