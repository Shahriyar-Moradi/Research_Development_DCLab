"""Versioned TP-06 release evaluation campaign (task_planning_v2) with live AI Arm C & USD ceiling."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

from .gate import (
    CASE_FILE,
    CASE_SHA256,
    CASES,
    EXPECTED,
    GATE,
    MAX_TOTAL_USD,
    SEEDS,
    BudgetGateway,
    DollarCeiling,
    judge,
    snapshot_openrouter_prices,
)
from .runner import run_case

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "evidence/campaigns/task_planning_v2"
FREEZE_NAME = "freeze_v1.json"
CODE_FILES = (
    "dclab_rnd/agent_eval/task_planning_v2/gate.py",
    "dclab_rnd/agent_eval/task_planning_v2/runner.py",
    "dclab_rnd/agent_eval/task_planning_v2/__init__.py",
    "dclab_rnd/tasks/dispatch.py",
    "dclab_rnd/tasks/decomposer.py",
    "dclab_rnd/tasks/planner.py",
    "dclab_rnd/orchestra/solution.py",
)
PLANNING_BOUNDS = {"cap_eur": 0.08, "max_seconds": 60.0, "max_tokens": 2400, "max_total_tokens": 12000}


def code_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CODE_FILES}


def load_dotenv(env_path: Path | None = None) -> None:
    target = env_path or (ROOT / ".env")
    if not target.exists():
        return
    for raw in target.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


def freeze(campaign: Path = CAMPAIGN, *, live_model: str | None = None, cap_usd: str = "3.80") -> dict[str, object]:
    """Freeze cases, code hashes, thresholds, and OpenRouter price snapshot prior to live runs."""
    target = campaign / FREEZE_NAME
    if target.exists():
        raise FileExistsError(str(target))
    cap = Decimal(str(cap_usd))
    if cap <= 0 or cap > MAX_TOTAL_USD:
        raise ValueError("usd_ceiling_out_of_authorized_bounds")
    (campaign / "results").mkdir(parents=True, exist_ok=True)
    price_info = None
    if live_model:
        price_info = snapshot_openrouter_prices(live_model, campaign / "price_snapshot_usd.json")
    doc = {
        "schema": "dclab_rnd.task_planning_v2_freeze.v1",
        "campaign_id": "task_planning_v2",
        "evaluation_only": True,
        "case_file": str(CASE_FILE.relative_to(ROOT)),
        "case_sha256": CASE_SHA256,
        "code_sha256": code_hashes(),
        "seeds": list(SEEDS),
        "cases": [case.__dict__ for case in CASES],
        "expected": sorted(EXPECTED),
        "gate": {key: list(val) if isinstance(val, tuple) else val for key, val in GATE.items()},
        "cap_usd": str(cap),
        "live_model": live_model,
        "price_snapshot": price_info,
    }
    target.write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def verify_freeze(campaign: Path = CAMPAIGN) -> dict[str, object]:
    path = campaign / FREEZE_NAME
    if not path.exists():
        raise RuntimeError(f"missing_freeze:{path}")
    frozen = json.loads(path.read_text())
    if frozen["case_sha256"] != CASE_SHA256 or frozen["code_sha256"] != code_hashes():
        raise RuntimeError("frozen_cases_or_code_modified_after_freeze")
    return frozen


def _scripted_server():
    class Scripted:
        thinking = None

        def complete(self, messages, tools=None, max_tokens=None, **options):
            body = json.loads(messages[-1]["content"])
            baseline = body.get("baseline") or body["deterministic_graph"]
            if "deterministic_graph" in body:
                leaves = [n for n in baseline["nodes"] if n["kind"] != "group"]
                ans = {
                    "version": "tp02.v1",
                    "baseline_fingerprint": baseline["fingerprint"],
                    "groups": [{"id": "group_0", "parent_id": None, "members": [n["id"] for n in leaves]}],
                    "dependencies": [{"task_id": n["id"], "requires": n["dependencies"]} for n in leaves],
                }
            else:
                ans = {"version": "tp03.v1", "baseline_fingerprint": baseline["fingerprint"], **{k: baseline[k] for k in ("actions", "budget", "policy")}}
            return {
                "content": json.dumps(ans),
                "finish_reason": "stop",
                "tool_calls": [],
                "thinking": [{"text": "SCRIPTED V2 DRY-RUN: preserve immutable operations."}],
                "usage": {"input_tokens": 12, "output_tokens": 36},
            }

    return Scripted()


def run_offline(*, campaign: Path | None = None, limit: int | None = None) -> dict[str, object]:
    """Run $0 offline scripted end-to-end validation of Arms A, B, and C across v2 cases."""
    from dclab_rnd.agent_eval.task_planning.probe import offline_environment, require_standalone
    from dclab_rnd.agent_eval.task_planning.probe import run_probe
    from dclab_rnd.models import FileUsage

    require_standalone()
    probe = run_probe()
    selected = CASES[:limit] if limit else CASES
    scores = {arm: [] for arm in ("A", "B", "C")}
    server = _scripted_server()
    with tempfile.TemporaryDirectory(prefix="tpv2-offline-") as tmp, offline_environment():
        usage = FileUsage(Path(tmp) / "usage.jsonl")
        ceiling = DollarCeiling("4.00", max_calls=1000)
        gateway = BudgetGateway(usage, ceiling, "claude-sonnet-5-5", home=Path(tmp), transport=lambda tier, purpose: server)
        for case in selected:
            for seed in SEEDS:
                scores["A"].append(run_case(case, seed, "A"))
                scores["B"].append(run_case(case, seed, "B"))
                scores["C"].append(run_case(case, seed, "C", gateway=gateway, planning_budget=PLANNING_BOUNDS, experience_home=Path(tmp) / "exp"))
    judgement = judge(scores, configured_model=True, c_integrated=True, proposal_probe=probe) if limit is None else None
    result = {
        "schema": "dclab_rnd.task_planning_v2_result.v1",
        "mode": "offline_scripted",
        "cases_run": len(selected),
        "scores": scores,
        "judgment": judgement,
    }
    if campaign is not None:
        (campaign / "results").mkdir(parents=True, exist_ok=True)
        (campaign / "results" / "TP2-001_offline.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _load_price_table(snap_path: Path) -> dict[str, object]:
    from dclab_rnd.models import prices

    if not snap_path.exists():
        return {}
    snap = json.loads(snap_path.read_text())
    return {
        model_id: prices.Price(
            input=float(row["input"]),
            output=float(row["output"]),
            as_of=str(row.get("as_of") or row.get("updated") or "openrouter_live_snapshot"),
            source=str(row.get("source") or "openrouter:/api/v1/models"),
        )
        for model_id, row in (snap.get("prices") or {}).items()
    }


def _run_shard_worker(args: tuple[list[dict[str, object]], str, str, str, str]) -> list[dict[str, dict[str, object]]]:
    case_dicts, seed_list_json, model, shard_cap, work_dir = args
    from .gate import Case
    from dclab_rnd import control
    from dclab_rnd.models import FileUsage, prices, settings

    load_dotenv()
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    if not control.live_allowed():
        raise RuntimeError("owner_has_stopped_live_calls")
    for purpose in ("task_decomposer", "task_planner"):
        if settings.served_model(purpose) != model:
            raise RuntimeError(f"model_mismatch:{purpose}:{settings.served_model(purpose)}!={model}")
    home = Path(work_dir)
    home.mkdir(parents=True, exist_ok=True)
    price_table = _load_price_table(home.parent.parent / "price_snapshot_usd.json")
    seeds = json.loads(seed_list_json)
    usage = FileUsage(home / "usage.jsonl")
    ceiling = DollarCeiling(shard_cap, max_calls=300)
    out = []
    with patch.dict(prices.PRICES, price_table) if price_table else ExitStack():
        gateway = BudgetGateway(usage, ceiling, model, home=home)
        shard_name = home.name
        for idx, body in enumerate(case_dicts, 1):
            case = Case(**body)
            for seed in seeds:
                row_a = run_case(case, seed, "A")
                row_b = run_case(case, seed, "B")
                row_c = run_case(case, seed, "C", gateway=gateway, planning_budget=PLANNING_BOUNDS, experience_home=home / "experience")
                print(
                    f"[{shard_name} {idx}/{len(case_dicts)}] {case.id} A={row_a['status']} B={row_b['status']} C={row_c['status']} "
                    f"cost=${row_c['cost_eur']:.5f} reserved=${ceiling.committed}",
                    file=sys.stderr,
                    flush=True,
                )
                out.append({"A": row_a, "B": row_b, "C": row_c})
    return out


def run_live_campaign(campaign: Path = CAMPAIGN, *, workers: int = 8) -> dict[str, object]:
    """Run frozen Arms A, B, and live-AI Arm C in sharded subprocesses within the frozen USD ceiling."""
    from dclab_rnd.agent_eval.task_planning.probe import require_standalone, run_probe

    require_standalone()
    frozen = verify_freeze(campaign)
    model = frozen.get("live_model")
    if not model:
        raise RuntimeError("freeze_missing_live_model")
    cap = Decimal(str(frozen["cap_usd"]))
    if cap > MAX_TOTAL_USD:
        raise ValueError("usd_ceiling_out_of_authorized_bounds")
    result_path = campaign / "results" / "TP2-001_live.json"
    if result_path.exists():
        raise FileExistsError(str(result_path))

    probe = run_probe()
    cases_dicts = frozen["cases"]
    n_workers = max(1, min(workers, len(cases_dicts)))
    shards: list[list[dict[str, object]]] = [[] for _ in range(n_workers)]
    for idx, body in enumerate(cases_dicts):
        shards[idx % n_workers].append(body)
    shard_cap = (cap / Decimal(n_workers)).quantize(Decimal("0.01"))
    if shard_cap * n_workers > cap:
        shard_cap -= Decimal("0.01")
    assert shard_cap > 0 and shard_cap * n_workers <= cap

    live_dir = campaign / "live_run_artifacts"
    live_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (shard, json.dumps(frozen["seeds"]), str(model), str(shard_cap), str(live_dir / f"shard_{idx}"))
        for idx, shard in enumerate(shards)
        if shard
    ]
    scores = {"A": [], "B": [], "C": []}
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(_run_shard_worker, job): idx for idx, job in enumerate(jobs)}
        for fut in as_completed(futures):
            for entry in fut.result():
                for arm in ("A", "B", "C"):
                    scores[arm].append(entry[arm])

    order = {f"{c['id']}:{s}": i for i, c in enumerate(cases_dicts) for s in frozen["seeds"]}
    for arm in ("A", "B", "C"):
        scores[arm].sort(key=lambda r: order[r["id"]])

    judgement = judge(scores, configured_model=True, c_integrated=True, proposal_probe=probe)
    summary = {
        "total_cost_usd": round(sum(r.get("cost_eur") or 0.0 for r in scores["C"]), 6),
        "total_tokens": sum(r.get("tokens") or 0 for r in scores["C"]),
        "total_provider_requests": sum(r.get("provider_requests") or 0 for r in scores["C"]),
        "model_graphs_admitted": sum(bool(r.get("model_graph_admitted")) for r in scores["C"] if r.get("phase") == "execution"),
        "execution_cases": sum(1 for r in scores["C"] if r.get("phase") == "execution"),
    }
    doc = {
        "schema": "dclab_rnd.task_planning_v2_result.v1",
        "campaign_id": "task_planning_v2",
        "evaluation_only": True,
        "freeze": FREEZE_NAME,
        "live_model": model,
        "cap_usd": str(cap),
        "summary": summary,
        "judgment": judgement,
        "scores": scores,
        "proposal_probe": probe,
    }
    result_path.write_text(json.dumps(doc, indent=2) + "\n")
    (campaign / "REPORT.md").write_text(_format_report(doc))
    return doc


def _format_report(doc: dict[str, object]) -> str:
    j = doc["judgment"]
    s = doc["summary"]
    comp = j["comparisons"]["C"]
    lines = [
        "# Task Planning v2 Release Evaluation Report",
        "",
        f"- **Model:** `{doc['live_model']}`",
        f"- **Authorized evaluation cap:** `${doc['cap_usd']}` USD",
        f"- **Measured live cost:** `${s['total_cost_usd']:.6f}` USD across `{s['total_provider_requests']}` requests (`{s['total_tokens']}` tokens)",
        f"- **Model graphs admitted:** `{s['model_graphs_admitted']} / {s['execution_cases']}` execution cases",
        f"- **Gate Verdict:** **`{j['verdict']}`** (`promoted={j['promoted']}`)",
        f"- **Reasons:** `{', '.join(j['reasons']) or 'none'}`",
        "",
        "## Paired Metrics (Arm C vs Arm A)",
        "",
        "| Metric | Difference (C - A) | 95% Bootstrap CI | Matched Rows |",
        "|---|---:|---|---:|",
    ]
    for metric in ("correct", "completed", "recovered", "proper_abstention", "seconds", "compute_seconds", "cost_eur", "extra_prompts"):
        m = comp[metric]
        lines.append(f"| {metric} | {m['difference']} | {m['interval']} | {m['matched_case_seed_rows']} |")
    lines.extend(["", f"- Resource checks: `{j['resource_checks']}`", f"- Critical safety sums: `{j['critical']}`", ""])
    return "\n".join(lines)
