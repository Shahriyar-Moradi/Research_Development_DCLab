"""The agentic layer judged on the sealed benchmark (package 18.9): rules only, a plain prompted model, and the roles.

The three arms solve the same sealed cases of benchmark v3 (15.9) and are scored by the same ``run.score``:

- **rules**: the roles' pipeline with no model (``understand`` → ``analyse`` → ``plan``, the plan saved as the solution,
  the stages run): what DCLab does when every role abstains; no request;
- **plain**: a base model prompted once per case with the column names and summaries (14.5's ``base_model`` policy);
- **roles**: the same pipeline with the problem analyst, the data investigator and the solution planner answering,
  each checked by code.

Both pipeline arms read partial labels from the moment's words (``label_coverage``), as the product's plan adoption
does. The gate (``GATE``) was written before any of these runs. A run on the sealed set happens once per arm: it is
refused when that arm has a run already (a crashed one included), when the gate, the harness, the pipeline or the
prompts have uncommitted changes, or when a model is unpriced (a euro cap could not stop it). Each run is recorded as a
new ``BEN3`` result with every case's row as it finishes. ``judge`` gives "not scored", never pass, when the roles did
not really answer (a request failed, or a role never answered), or when any arm had errors.

    python -m dclab_rnd.agent_eval.layer rules                         # free
    python -m dclab_rnd.agent_eval.layer plain --yes --cap-eur 0.5
    python -m dclab_rnd.agent_eval.layer roles --yes --cap-eur 0.5
    python -m dclab_rnd.agent_eval.layer judge                         # the three against the gate -> LAYER-001
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import benchmark_v3 as v3

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "evidence/campaigns/benchmark_v3/results"
LAYER_RESULTS = ROOT / "evidence/campaigns/team_v1/results"
GATE = {"version": "layer_gate_v1", "fixed": "2026-10-13, before any of its runs",
        "rule": "on benchmark v3's sealed set: the roles catch at least as many traps as the rules-only pipeline, with no more false "
                "alarms, and at least as many traps as a plain prompted model; no case errors in any arm; the roles really answered "
                "(no failed request, and the analyst, the investigator and the planner each answered at least once)"}
def _provisional_rows_v1(index: Any, times: Any = None) -> Any:
    """The provisional training rows exactly as this sealed run read them (the product's rule moved to 75 % in 16.15):
    the earliest 80 % by time, else a seeded 80 %."""
    import numpy as np
    import pandas as pd

    if times is not None:
        stamps = pd.to_datetime(pd.Series(list(times), index=index), errors="coerce")
        if stamps.notna().mean() > 0.9:
            order = stamps.rank(method="first").fillna(len(index))
            return index[(order <= int(0.8 * len(index))).to_numpy()]
    keep = np.random.default_rng(42).random(len(index)) < 0.8
    return index[keep]



ROLE_PURPOSES = ("problem_analyst", "data_investigator", "solution_planner")
# what a sealed run depends on: committed and unchanged before it may start
FROZEN = (*v3.GUARDED, "dclab_rnd/agent_eval/layer.py", "dclab_rnd/agent_eval/baselines.py", "dclab_rnd/core_adapter.py",
          "dclab_core/understand.py", "dclab_core/analyse.py", "dclab_core/plan.py", "dclab_core/prompts", "dclab_rnd/prompts/base_audit.md")


class Counting:
    """An advisor that counts, per purpose, the answers it got, the requests that failed and those it abstained on."""

    def __init__(self, inner: Any):
        self.inner, self.counts = inner, {}

    def _count(self, purpose: str, what: str) -> None:
        self.counts.setdefault(purpose, {"answered": 0, "failed": 0, "abstained": 0})[what] += 1

    def ask(self, purpose: str, evidence: Any) -> Any:
        try:
            answer = self.inner.ask(purpose, evidence)
        except Exception:
            self._count(purpose, "failed")
            raise
        self._count(purpose, "answered" if answer is not None else "abstained")
        return answer

    def spent_eur(self) -> float:
        return float(self.inner.spent_eur()) if hasattr(self.inner, "spent_eur") else 0.0


def _suite():
    return importlib.import_module("dclab_rnd.agent_eval.run")


def pipeline_policy(advisor: Any, counter: dict[str, Any]):
    """understand → analyse → plan with ``advisor`` for the roles (None: the rules alone), saved through set_solution."""

    def policy(box: Any, store: Any, pid: str, home: Path, case: Any) -> list[dict[str, Any]]:
        from dclab_core.analyse import analyse
        from dclab_core.plan import plan as rule_plan, plan_with
        from dclab_core.understand import label_coverage, understand

        from dclab_rnd.core_adapter import solution_fields_from_plan
        from dclab_rnd.studio import data as sd

        project = store.get(pid)
        frame = sd.load_table(sd.data_path(store, project))
        description = f"Predict the target column. The prediction is made: {case.moment}"
        brief = understand(description, {"table.parquet": [str(c) for c in frame.columns]}, advisor=advisor)
        coverage = label_coverage(description)  # the same words rule in both pipeline arms, as the product's adopt_plan
        if coverage and not brief.label_coverage.known:
            brief = brief.model_copy(update={"label_coverage": type(brief.label_coverage)(value="partial", status="stated",
                                                                                          quote=coverage[1], source="person")})
        task = sd.detect_task(frame, "target")["task"]
        times = next((c for c in (project["data"].get("profile") or {}).get("time_candidates") or [] if c in frame.columns), None)
        train = _provisional_rows_v1(frame.index, frame[times] if times else None)
        findings = analyse({"table": frame}, target="target", task=task, brief=brief, train_index=train, advisor=advisor, moment=case.moment)
        plan_ = plan_with(advisor, brief, findings)["plan"] if advisor is not None else rule_plan(brief, findings)
        fields = solution_fields_from_plan(plan_, "target", case.moment)
        saved = box.call("set_solution", {"project_id": pid, **{k: v for k, v in fields.items() if v not in (None, "", [])}})
        if "error" in saved:
            raise RuntimeError("the solution was refused: " + str(saved["error"])[:200])
        box.call("run_all", {"project_id": pid})
        return []
    return policy


def run_arm(arm: str, split_name: str, advisor: Any = None, client: Any = None, cap_eur: Optional[float] = None,
            log=print, rows_file: Optional[Path] = None) -> dict[str, Any]:
    from unittest import mock

    from . import baselines

    suite = _suite()
    cases = v3.split(split_name)
    counter = {"requests": 0, "eur": 0.0}
    name = f"layer_{arm}"
    counting = Counting(advisor) if arm == "roles" and advisor is not None else None
    if arm == "plain":
        suite.POLICIES[name] = baselines.base_model_policy(client, counter, max_requests=len(cases), max_eur=cap_eur)
    else:
        suite.POLICIES[name] = pipeline_policy(counting, counter)
    known = suite._known_ids()
    results = []
    try:  # the stages' notes ask no model: they do not change a move, and would cost requests outside the cap
        with mock.patch("dclab_rnd.studio.agent._chat", lambda *a, **k: None), \
                mock.patch("dclab_rnd.studio.agent.explain_stage", lambda *a, **k: None), tempfile.TemporaryDirectory() as tmp:
            for case in cases:
                result = suite.run_case(case, name, Path(tmp), known)
                results.append(result)
                if rows_file is not None:  # every paid row on disk as it finishes
                    with rows_file.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(asdict(result), default=str) + "\n")
                log(f"  {case.id} {arm} caught={result.leak_caught!s:5s} alarm={result.false_alarm!s:5s}" + (f" ERROR {result.error[:80]}" if result.error else ""))
    finally:
        suite.POLICIES.pop(name, None)
    rows = [asdict(r) for r in results]
    summary, families = baselines.rows_summary(rows, len(cases), {c.id: c.trap for c in cases})
    spent = counter["eur"] if arm == "plain" else (counting.spent_eur() if counting is not None else 0.0)
    models = {}
    if client is not None:
        models["leakage_review"] = getattr(client, "model", None)
    if advisor is not None and hasattr(advisor, "_clients"):
        models.update({p: getattr(c, "model", None) for p, c in advisor._clients.items() if c is not None})
    try:
        core_prompts = json.loads((ROOT / "dclab_core/prompts/HASHES.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        core_prompts = {}
    return {"campaign_id": v3.VERSION, "kind": f"benchmark_layer_{arm}", "status": "completed", "suite": f"{v3.VERSION}:{split_name}",
            "split": split_name, "sealed": split_name == "test", "arm": arm, "gate": GATE, "code": v3._provenance(),
            "models": models, "core_prompt_hashes": {p: core_prompts.get(p) for p in ROLE_PURPOSES},
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "spent_eur": round(spent, 6),
            "requests": counter.get("requests") if arm == "plain" else (counting.counts if counting is not None else {}),
            "cases": [{"id": c.id, "trap": c.trap, "fingerprint": c.fingerprint()} for c in cases], "results": rows,
            "summary": {arm: summary}, "by_family": {arm: families},
            "limitations": ["Evidence about these policies on these seeded cases, not about any model's readiness for production.",
                            "One run per arm, one model: answers vary between runs, and the interval is over cases, not repeats."]}


def judge(rules: dict[str, Any], plain: dict[str, Any], roles: dict[str, Any]) -> dict[str, Any]:
    r, p, x = rules["summary"]["rules"], plain["summary"]["plain"], roles["summary"]["roles"]
    caught = lambda s: s["leaks_caught"][0]  # noqa: E731
    alarms = lambda s: s["false_alarms"][0]  # noqa: E731
    base = {"gate": GATE["version"], "caught": {"rules": r["leaks_caught"], "plain": p["leaks_caught"], "roles": x["leaks_caught"]},
            "false_alarms": {"rules": r["false_alarms"], "plain": p["false_alarms"], "roles": x["false_alarms"]},
            "ci95": {"rules": r.get("leaks_caught_ci95"), "plain": p.get("leaks_caught_ci95"), "roles": x.get("leaks_caught_ci95")}}
    eur = {arm: d.get("spent_eur", 0.0) for arm, d in (("rules", rules), ("plain", plain), ("roles", roles))}
    base["eur"] = eur
    base["eur_per_caught_trap"] = {arm: (round(eur[arm] / caught(s), 5) if caught(s) else None) for arm, s in (("rules", r), ("plain", p), ("roles", x))}
    errors = {arm: s.get("errors", 0) for arm, s in (("rules", r), ("plain", p), ("roles", x))}
    if any(errors.values()):
        return {**base, "status": "not scored", "why": f"errors in an arm: {errors}"}
    counts = roles.get("requests") if isinstance(roles.get("requests"), dict) else None
    if counts is not None:  # a real run records its counts; the roles must have answered, never silently fallen back
        failed = sum(c.get("failed", 0) for c in counts.values())
        silent = [p for p in ROLE_PURPOSES if (counts.get(p) or {}).get("answered", 0) == 0]
        if failed or silent:
            return {**base, "status": "not scored", "why": f"{failed} requests failed; never answered: {silent}"}
    checks = {"at_least_rules": caught(x) >= caught(r), "no_more_alarms": alarms(x) <= alarms(r), "at_least_plain": caught(x) >= caught(p)}
    return {**base, "status": "pass" if all(checks.values()) else "fail", "checks": checks}


def _git(*args: str) -> int:
    import subprocess

    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, timeout=30).returncode
    except (OSError, subprocess.SubprocessError):
        return 1


def frozen_problem() -> Optional[str]:
    """Why the sealed comparison may not start: the benchmark not frozen, or a file it depends on untracked or changed."""
    problem = v3.frozen_problem()
    if problem:
        return problem
    for path in FROZEN:
        if _git("ls-files", "--error-unmatch", path) != 0:
            return f"{path} is not committed"
    if _git("diff", "--quiet", "HEAD", "--", *FROZEN) != 0:
        return "the gate, the harness, the pipeline or the prompts have uncommitted changes"
    return None


def _runs(arm: str, split: str) -> list[Path]:
    """Every run of this arm on this split, a crashed one included (its stub)."""
    out = []
    for path in sorted(RESULTS.glob("BEN3-*")):
        if path.suffix not in (".json", ".jsonl"):
            continue
        if path.suffix == ".jsonl":
            if f"_{split}_layer_{arm}" in path.name:
                out.append(path)
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if doc.get("arm") == arm and doc.get("split") == split:
            out.append(path)
    return out


def _store(path: Path, doc: dict[str, Any]) -> None:
    import re

    found = re.match(r"([A-Z0-9]+-\d+)_", path.name)
    doc = {"experiment_id": found.group(1) if found else path.stem, **doc}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)  # the stub becomes the record in one step: a crash never leaves neither


def _judge_main(split: str) -> int:
    latest: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(RESULTS.glob("BEN3-*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("kind", "").startswith("benchmark_layer_") and doc.get("split") == split:
            latest.setdefault(doc["arm"], []).append(doc)
    if set(latest) != {"rules", "plain", "roles"}:
        print(f"missing arms: {sorted({'rules', 'plain', 'roles'} - set(latest))}")
        return 1
    if split == "test":
        if any(len(v) != 1 for v in latest.values()):
            print("an arm has more than one sealed run: refusing to judge (best-of-n is not a measurement)")
            return 1
        frozen = json.loads(v3.SEALED.read_text(encoding="utf-8"))["cases"]
        for arm, (doc,) in latest.items():
            if {c["id"]: c["fingerprint"] for c in doc["cases"]} != frozen:
                print(f"the {arm} arm did not run the frozen sealed cases: refusing to judge")
                return 1
            if doc.get("gate") != GATE:
                print(f"the {arm} arm ran under another gate: refusing to judge")
                return 1
    verdict = judge(latest["rules"][-1], latest["plain"][-1], latest["roles"][-1])
    LAYER_RESULTS.mkdir(parents=True, exist_ok=True)
    n = len(list(LAYER_RESULTS.glob("LAYER-*.json"))) + 1
    path = LAYER_RESULTS / f"LAYER-{n:03d}_judged.json"
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _store(path, {"id": f"LAYER-{n:03d}", "campaign_id": "team_v1", "kind": "layer_judged", "status": "completed", "at": now,
                  "completed_at": now, "gate": GATE, "verdict": verdict, "split": split,
                  "inputs": {arm: docs[-1].get("experiment_id") for arm, docs in latest.items()}})
    print(json.dumps(verdict, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.layer")
    parser.add_argument("arm", choices=("rules", "plain", "roles", "judge"))
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--cap-eur", type=float, default=None)
    args = parser.parse_args(argv)
    print(json.dumps({"gate": GATE, "arm": args.arm, "split": args.split, "cases": len(v3.split(args.split))}, indent=1))
    if args.arm == "judge":
        return _judge_main(args.split)
    if args.split == "test":
        problem = frozen_problem()
        if problem:
            parser.error(f"refusing to run the sealed set: {problem}")
        if _runs(args.arm, "test"):
            parser.error(f"the {args.arm} arm has run on the sealed set already: a sealed look happens once")
    advisor = client = None
    if args.arm in ("plain", "roles"):
        if not args.yes or args.cap_eur is None:
            print("nothing was sent: add --yes and --cap-eur to run it")
            return 0
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        from dclab_rnd.core_advisor import GatewayAdvisor
        from dclab_rnd.models import prices, settings
        from dclab_rnd.models.gateway import for_workspace

        gateway = for_workspace()
        gateway.shadows = None
        purposes = ("leakage_review",) if args.arm == "plain" else ROLE_PURPOSES
        for purpose in purposes:  # every purpose served by a priced model, or nothing is sent
            if not gateway.available(purpose):
                parser.error(f"no model serves {purpose}: nothing was sent")
            probe = gateway.client(purpose)
            if not settings.public(probe.tier.get("name") or "standard")["local"] and prices.load().get(probe.model) is None:
                parser.error(f"the model {probe.model} has no price, so a euro cap could not stop it: nothing was sent")
        if args.arm == "plain":
            client = gateway.client("leakage_review")
            client.run_limit_eur = args.cap_eur
        else:
            advisor = GatewayAdvisor(gateway, max_tokens=3000, cap_eur=args.cap_eur)
    RESULTS.mkdir(parents=True, exist_ok=True)
    n = len([p for p in RESULTS.glob("BEN3-*") if p.suffix == ".json"]) + 1
    path = RESULTS / f"BEN3-{n:03d}_{args.split}_layer_{args.arm}.json"
    rows_file = path.with_suffix(".jsonl")
    with path.open("x", encoding="utf-8") as handle:  # the look is on record before it happens
        json.dump({"status": "started", "arm": args.arm, "split": args.split, "gate": GATE}, handle)
    report = run_arm(args.arm, args.split, advisor, client, args.cap_eur, rows_file=rows_file)
    _store(path, report)
    rows_file.unlink(missing_ok=True)  # the record holds every row now
    s = report["summary"][args.arm]
    print(f"{args.arm}: caught {s['leaks_caught']} alarms {s['false_alarms']} errors {s.get('errors')} spent EUR {report['spent_eur']}, stored {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
