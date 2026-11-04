"""Agent metrics (the owner, 2026-10-20: keep track of all the metrics of the agents, under the hood).

One append-only ledger per workspace, ``<workspace>/metrics/agent_metrics.jsonl`` (the store keeps it whole with the rest
of the workspace), one line per measurement::

    {at, agent, metric, value, n, source, prompt, model, run, ref, extra}

``agent`` is a model purpose or role (``home_agent``, ``notebook_cell``, ``intern``), a tool-using agent, or an evaluation
suite; ``prompt`` is the version of the prompt it ran with (its content hash), so a score is never read across prompts;
``source`` says where the measurement came from: ``live`` (what happened to a real request, step or proposal) or ``eval``
(a result file of ``evidence/campaigns``, read back by :func:`sync_eval`).

The live measurements, each written where it is made (never a reason for the work to fail):

- ``request_ok`` (1 or 0): a model request answered or not, with its seconds, cost and tokens;
- ``output_check`` (1 or 0): the model's output passed the code that checks it;
- ``step_ok`` (1 or 0): an agent's tool step that worked, not an error or a refusal;
- ``proposal_accepted`` (1 or 0): a person's answer to what an agent proposed (the notebook's cells);
- ``correction`` (1): a person changed what DCLab proposed;
- ``feedback_helpful`` (1 or 0): a person's rating of a reply.

:func:`scorecard` joins them per agent with intervals, the prompt versions seen, latency, cost and the latest result of every
evaluation suite; nothing here switches a role on or off (the gates and ``roles_judged.json`` do, from the evaluations).
"""

from __future__ import annotations

import json
import math
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

FILE = "agent_metrics.jsonl"
RATES = ("request_ok", "output_check", "step_ok", "proposal_accepted", "feedback_helpful")  # metrics whose value is 1 or 0
VERDICTS = ("eval_pass", "eval_unscored")
READER = 2  # 2026-11-03: dclab_core.bench.verdict_of; a ledger's verdict rows read before it are read again once
_LOCK = threading.Lock()
_PROMPTS: dict[str, Optional[str]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _home(home: Any = None) -> Optional[Path]:
    if home:
        return Path(home)
    try:
        from .corrections import _home as current

        return current()
    except Exception:  # noqa: BLE001
        return None


def path_of(home: Any = None) -> Optional[Path]:
    found = _home(home)
    return found / "metrics" / FILE if found else None


def prompt_version(agent: str) -> Optional[str]:
    """The version of the prompt an agent runs with (its content hash), once per process."""
    if agent not in _PROMPTS:
        try:
            from .models import routing

            _PROMPTS[agent] = routing.prompt_of(agent)
        except Exception:  # noqa: BLE001 - an agent that is not a model purpose has none
            _PROMPTS[agent] = None
    return _PROMPTS[agent]


def emit(agent: str, metric: str, value: float, *, source: str = "live", n: int = 1, prompt: Optional[str] = None, model: Optional[str] = None,
         run: Optional[str] = None, ref: Optional[str] = None, extra: Optional[dict[str, Any]] = None, home: Any = None) -> bool:
    """One measurement kept. False when no workspace is known or it could not be written: never a reason for the work to fail."""
    try:
        target = path_of(home)
        if target is None:
            return False
        row = {"at": _now(), "agent": agent, "metric": metric, "value": float(value), "n": int(n), "source": source,
               "prompt": prompt if prompt is not None else prompt_version(agent), "model": model, "run": run, "ref": ref}
        if extra:
            row["extra"] = extra
        line = json.dumps(row, ensure_ascii=False, default=str, sort_keys=True) + "\n"
        with _LOCK:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a", encoding="utf-8") as fh:
                fh.write(line)
        return True
    except Exception:  # noqa: BLE001
        return False


def series(home: Any = None, *, agent: Optional[str] = None, since: Optional[str] = None) -> list[dict[str, Any]]:
    """The measurements, oldest first, of one agent and since a time (ISO) when given."""
    target = path_of(home)
    if target is None or not target.exists():
        return []
    out = []
    for line in target.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if (agent is None or row.get("agent") == agent) and (since is None or str(row.get("at") or "") >= since):
            out.append(row)
    return out


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson interval of a rate of k in n."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)


def _pct(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1) + 0.5))], 3)


# ------------------------------------------------------------------------------------------------ evaluation results
def eval_rows(results: Path) -> list[dict[str, Any]]:
    """Every result file of ``evidence/campaigns/*/results`` as measurements: its verdict (1 pass, 0 else), what it spent, and
    each numeric count or rate of its summary. The agent is the suite (its campaign), the ref the result's id."""
    rows = []
    for path in sorted(Path(results).glob("*/results/*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        suite, ref = path.parent.parent.name, path.stem
        at = str(doc.get("at") or doc.get("created") or datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"))
        from dclab_core import bench  # here, not at the top: the campaign agent's environment has no numpy, which dclab_core loads

        passed, verdict = bench.verdict_of(doc)  # every verdict shape; a dict verdict used to read as a failure
        base = {"at": at, "agent": suite, "source": "eval", "ref": ref, "n": 1, "model": None, "run": None, "prompt": None}
        if passed is not None:
            rows.append({**base, "metric": "eval_pass", "value": 1.0 if passed else 0.0, "extra": {"verdict": verdict.lower()[:40], "reader": READER}})
        elif verdict or doc.get("verdict") not in (None, ""):  # any verdict the reader cannot score replaces an older misread
            words = verdict or json.dumps(doc.get("verdict"), ensure_ascii=False, default=str)
            rows.append({**base, "metric": "eval_unscored", "value": 1.0, "extra": {"verdict": words.lower()[:40], "reader": READER}})
        if isinstance(doc.get("spent_eur"), (int, float)):
            rows.append({**base, "metric": "eval_spent_eur", "value": float(doc["spent_eur"])})
        summary = doc.get("summary")
        for key, value in (summary.items() if isinstance(summary, dict) else []):
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                rows.append({**base, "metric": f"summary.{str(key)[:60]}", "value": float(value)})
    return rows


def sync_eval(home: Any = None, results: Optional[Path] = None) -> int:
    """Read the evaluation results into the ledger once each (a result already in it is never added again). Returns how many
    measurements were added."""
    from .settings import ROOT  # the repository (the evidence folder is the full edition's)

    target = path_of(home)
    if target is None:
        return 0
    results = results or ROOT / "evidence" / "campaigns"
    if not Path(results).is_dir():
        return 0  # a deployment without the repository's evidence: nothing to read
    have = {(r.get("ref"), r.get("metric")) for r in series(home) if r.get("source") == "eval"
            and (r.get("metric") not in VERDICTS or (r.get("extra") or {}).get("reader") == READER)}  # an older verdict is read again; the later row wins
    fresh = [r for r in eval_rows(results) if (r["ref"], r["metric"]) not in have]
    if not fresh:
        return 0
    with _LOCK:
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as fh:
            for r in fresh:
                fh.write(json.dumps(r, ensure_ascii=False, default=str, sort_keys=True) + "\n")
    return len(fresh)


# ------------------------------------------------------------------------------------------------ the scorecard
def scorecard(home: Any = None, *, days: int = 30, rows: Optional[Iterable[dict[str, Any]]] = None) -> dict[str, Any]:
    """Per agent: the rates (with their Wilson intervals) of what is measured live, latency, tokens and cost, the prompt versions and
    models seen, and a daily count of requests; and, per evaluation suite, its latest result. ``rows`` replaces the ledger (tests)."""
    all_rows = list(rows) if rows is not None else series(home)
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
    agents: dict[str, dict[str, Any]] = {}
    suites: dict[str, dict[str, Any]] = {}
    for r in all_rows:
        if r.get("source") == "eval":
            s = suites.setdefault(str(r["agent"]), {"suite": r["agent"], "results": {}})
            res = s["results"].setdefault(str(r.get("ref")), {"ref": r.get("ref"), "at": r.get("at"), "pass": None, "spent_eur": None, "summary": {}})
            if r["metric"] == "eval_pass":
                res["pass"] = bool(r["value"])
            elif r["metric"] == "eval_unscored":
                res["pass"] = None
            elif r["metric"] == "eval_spent_eur":
                res["spent_eur"] = r["value"]
            elif str(r["metric"]).startswith("summary."):
                res["summary"][str(r["metric"])[8:]] = r["value"]
            continue
        a = agents.setdefault(str(r.get("agent")), {"agent": r.get("agent"), "rates": {}, "seconds": [], "eur": 0.0, "tokens": 0, "prompts": {}, "models": {},
                                                    "daily": {}, "corrections": 0, "last": None})
        metric, value, n = r.get("metric"), float(r.get("value") or 0), int(r.get("n") or 1)
        if metric in RATES:
            k = a["rates"].setdefault(metric, [0.0, 0])
            k[0] += value * n
            k[1] += n
        if metric == "correction":
            a["corrections"] += n
        if metric == "request_ok":
            extra = r.get("extra") or {}
            if isinstance(extra.get("seconds"), (int, float)):
                a["seconds"].append(float(extra["seconds"]))
            a["eur"] += float(extra.get("cost_eur") or 0)
            a["tokens"] += int(extra.get("tokens_in") or 0) + int(extra.get("tokens_out") or 0)
            if str(r.get("at") or "") >= since:
                day = str(r["at"])[:10]
                a["daily"][day] = a["daily"].get(day, 0) + 1
        if r.get("prompt"):
            a["prompts"][r["prompt"]] = a["prompts"].get(r["prompt"], 0) + 1
        if r.get("model"):
            a["models"][r["model"]] = a["models"].get(r["model"], 0) + 1
        a["last"] = max(filter(None, (a["last"], r.get("at"))), default=a["last"])
    out_agents = []
    for name, a in sorted(agents.items()):
        rates = {m: {"n": n, "k": k, "rate": round(k / n, 4) if n else None, "interval": list(wilson(k, n))} for m, (k, n) in a["rates"].items()}
        out_agents.append({"agent": name, "rates": rates, "p50_seconds": _pct(a["seconds"], 0.5), "p95_seconds": _pct(a["seconds"], 0.95), "requests": len(a["seconds"]),
                           "eur": round(a["eur"], 4), "tokens": a["tokens"], "prompts": a["prompts"], "models": a["models"], "corrections": a["corrections"],
                           "daily": [{"day": d, "requests": c} for d, c in sorted(a["daily"].items())], "last": a["last"]})
    latest = []
    for s in sorted(suites.values(), key=lambda s: str(s["suite"])):
        results = sorted(s["results"].values(), key=lambda r: (str(r["at"]), str(r["ref"])))
        latest.append({"suite": s["suite"], "runs": len(results), "latest": results[-1], "passed": sum(1 for r in results if r["pass"]),
                       "failed": sum(1 for r in results if r["pass"] is False)})
    return {"agents": out_agents, "suites": latest, "rows": len(all_rows),
            "note": "Rates are per request, step, proposal or rating, with Wilson intervals; a suite's latest result is its newest file. "
                    "This is evidence about what ran: it never switches an agent on or off (the gates do, from the evaluations)."}


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_metrics", description="the agents' metrics ledger")
    parser.add_argument("command", choices=["sync", "show"])
    parser.add_argument("--home", default=os.environ.get("DCLAB_AGENT_HOME"))
    args = parser.parse_args(argv)
    from .settings import ROOT, Settings

    home = Path(args.home) if args.home else Path(Settings.load().agent_home)
    if args.command == "sync":
        print(f"{sync_eval(home)} evaluation measurements added to {path_of(home)}")
        return 0
    card = scorecard(home)
    for a in card["agents"]:
        print(f"{a['agent']:<22} requests {a['requests']:>5}  p50 {a['p50_seconds']}s  eur {a['eur']}  rates "
              + ", ".join(f"{m} {v['rate']} ({v['n']})" for m, v in a["rates"].items()))
    for s in card["suites"]:
        print(f"suite {s['suite']:<28} runs {s['runs']:>3}  passed {s['passed']}  latest {s['latest']['ref']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
