"""Versioned TP-06 release evaluation campaign (task_planning_v3) with dual-arm Gateway capture."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from dclab_rnd.agent_eval.task_planning_v2 import PLANNING_BOUNDS, _load_price_table, _scripted_server, load_dotenv
from dclab_rnd.agent_eval.task_planning_v2.runner import run_case
from .gate import (
    CASE_FILE,
    CASE_SHA256,
    CASES,
    EXPECTED,
    GATE,
    GATE_V2,
    MAX_TOTAL_USD,
    SEEDS,
    BudgetGateway,
    DollarCeiling,
    judge_v2,
    snapshot_openrouter_prices,
)

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN = ROOT / "evidence/campaigns/task_planning_v3"
PROTOCOL_FILE = ROOT / "docs/guides/TASK_PLANNING_V3_PROTOCOL.md"
LIVE_ARTIFACTS_DIR = ROOT / "agent_runs/live_tests/task_planning_v3_20261112"
FREEZE_NAME = "freeze_v1.json"
CODE_FILES = (
    "docs/guides/TASK_PLANNING_V3_PROTOCOL.md",
    "dclab_rnd/agent_eval/task_planning_v3/gate.py",
    "dclab_rnd/agent_eval/task_planning_v3/__init__.py",
    "dclab_rnd/agent_eval/task_planning_v2/runner.py",
    "dclab_rnd/tasks/dispatch.py",
    "dclab_rnd/tasks/decomposer.py",
    "dclab_rnd/tasks/planner.py",
    "dclab_rnd/orchestra/solution.py",
)


def code_hashes() -> dict[str, str]:
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CODE_FILES}


def freeze(campaign: Path = CAMPAIGN, *, live_model: str | None = None, cap_usd: str = "3.50") -> dict[str, object]:
    """Freeze cases, protocol, code hashes, thresholds, and OpenRouter price snapshot prior to live runs."""
    target = campaign / FREEZE_NAME
    if target.exists():
        raise FileExistsError(str(target))
    cap = Decimal(str(cap_usd))
    if cap <= 0 or cap > MAX_TOTAL_USD:
        raise ValueError("usd_ceiling_out_of_authorized_bounds")
    (campaign / "results").mkdir(parents=True, exist_ok=True)
    price_info = snapshot_openrouter_prices(live_model, campaign / "price_snapshot_usd.json") if live_model else None
    doc = {
        "schema": "dclab_rnd.task_planning_v3_freeze.v1",
        "campaign_id": "task_planning_v3",
        "evaluation_only": True,
        "case_file": str(CASE_FILE.relative_to(ROOT)),
        "case_sha256": CASE_SHA256,
        "protocol_file": str(PROTOCOL_FILE.relative_to(ROOT)),
        "protocol_sha256": hashlib.sha256(PROTOCOL_FILE.read_bytes()).hexdigest(),
        "code_sha256": code_hashes(),
        "seeds": list(SEEDS),
        "cases": [case.__dict__ for case in CASES],
        "expected": sorted(EXPECTED),
        "gate_v1": {k: list(v) if isinstance(v, tuple) else v for k, v in GATE.items()},
        "gate_v2": {k: list(v) if isinstance(v, tuple) else v for k, v in GATE_V2.items()},
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


def _read_usage_lines(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run_case_with_capture(case, seed, arm, *, gateway=None, planning_budget=None, experience_home=None):
    from dclab_rnd.models import gateway as gateway_mod

    usage_path = Path(gateway.usage.path) if gateway is not None and getattr(gateway, "usage", None) is not None else None
    before_count = len(_read_usage_lines(usage_path)) if (arm == "A" and usage_path is not None) else 0
    prev_installed = gateway_mod._INSTALLED
    with ExitStack() as stack:
        if gateway is not None:
            gateway_mod.install(gateway)
            stack.callback(gateway_mod.install, prev_installed)
            if arm == "A":
                stack.enter_context(patch("dclab_rnd.experience.model_text_allowed", return_value=True))
        row = run_case(case, seed, arm, gateway=gateway, planning_budget=planning_budget, experience_home=experience_home)
    if arm == "A" and usage_path is not None:
        new_rows = _read_usage_lines(usage_path)[before_count:]
        row["cost_eur"] = round(sum(float(r.get("cost_eur") or 0.0) for r in new_rows), 8)
        row["tokens"] = sum(
            int(r.get("input_tokens") or r.get("tokens_in") or 0) + int(r.get("output_tokens") or r.get("tokens_out") or 0)
            for r in new_rows
        )
        row["provider_requests"] = len(new_rows)
        row["extra_prompts"] = len(new_rows)
    return row


def _offline_server():
    base = _scripted_server()

    class OfflineServer:
        thinking = None

        def complete(self, messages, tools=None, max_tokens=None, **options):
            raw = str((messages[-1] if messages else {}).get("content") or "")
            if raw.lstrip().startswith("{"):
                try:
                    return base.complete(messages, tools=tools, max_tokens=max_tokens, **options)
                except (KeyError, ValueError, TypeError):
                    pass
            return {
                "content": "Holdout score 0.720 with 95% interval 0.650 to 0.790.",
                "finish_reason": "stop",
                "tool_calls": [],
                "thinking": [{"text": "SCRIPTED V3 OFFLINE PROSE."}],
                "usage": {"input_tokens": 10, "output_tokens": 14},
            }

    return OfflineServer()


def run_offline(*, campaign: Path | None = None, limit: int | None = None) -> dict[str, object]:
    """Run $0 offline scripted end-to-end validation of Arms A, B, and C with dual Gateway capture."""
    from dclab_rnd.agent_eval.task_planning.probe import offline_environment, require_standalone, run_probe
    from dclab_rnd.models import FileUsage, Gateway
    from dclab_rnd.models.gateway import mark_invented_tables

    require_standalone()
    probe = run_probe()
    selected = CASES[:limit] if limit else CASES
    scores = {arm: [] for arm in ("A", "B", "C")}
    server = _offline_server()
    with tempfile.TemporaryDirectory(prefix="tpv3-offline-") as tmp, offline_environment():
        root = Path(tmp)
        gw_a = mark_invented_tables(
            Gateway(FileUsage(root / "arm_a/usage.jsonl"), transport=lambda tier, purpose: server),
            "invented_accounts_v2",
        )
        gw_a.home = root / "arm_a"
        ceiling = DollarCeiling("4.00", max_calls=1000)
        gw_c = BudgetGateway(
            FileUsage(root / "arm_c/usage.jsonl"),
            ceiling,
            "claude-sonnet-5-5",
            home=root / "arm_c",
            transport=lambda tier, purpose: server,
        )
        for case in selected:
            for seed in SEEDS:
                scores["A"].append(_run_case_with_capture(case, seed, "A", gateway=gw_a, experience_home=root / "arm_a/exp"))
                scores["B"].append(run_case(case, seed, "B"))
                scores["C"].append(
                    _run_case_with_capture(
                        case, seed, "C", gateway=gw_c, planning_budget=PLANNING_BOUNDS, experience_home=root / "arm_c/exp"
                    )
                )
    judgment = judge_v2(scores, configured_model=True, c_integrated=True, proposal_probe=probe) if limit is None else None
    result = {
        "schema": "dclab_rnd.task_planning_v3_result.v1",
        "mode": "offline_scripted",
        "cases_run": len(selected),
        "scores": scores,
        "judgment": judgment,
    }
    if campaign is not None:
        (campaign / "results").mkdir(parents=True, exist_ok=True)
        (campaign / "results" / "TP3-001_offline.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _run_shard_worker(args: tuple[list[dict[str, object]], str, str, str, str, str]) -> list[dict[str, dict[str, object]]]:
    case_dicts, seed_list_json, model, shard_cap, work_dir, price_snap_path = args
    from dclab_rnd import control
    from dclab_rnd.agent_eval.task_planning_v2.gate import Case
    from dclab_rnd.models import FileUsage, Gateway, prices, settings
    from dclab_rnd.models.gateway import mark_invented_tables

    load_dotenv()
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    if not control.live_allowed():
        raise RuntimeError("owner_has_stopped_live_calls")
    for purpose in ("task_decomposer", "task_planner"):
        if settings.served_model(purpose) != model:
            raise RuntimeError(f"model_mismatch:{purpose}:{settings.served_model(purpose)}!={model}")
    home = Path(work_dir)
    arm_a_home, arm_c_home = home / "arm_a", home / "arm_c"
    arm_a_home.mkdir(parents=True, exist_ok=True)
    arm_c_home.mkdir(parents=True, exist_ok=True)
    price_table = _load_price_table(Path(price_snap_path))
    seeds = json.loads(seed_list_json)
    gw_a = mark_invented_tables(Gateway(FileUsage(arm_a_home / "usage.jsonl")), "invented_accounts_v2")
    gw_a.home = arm_a_home
    ceiling = DollarCeiling(shard_cap, max_calls=300)
    out = []
    with patch.dict(prices.PRICES, price_table) if price_table else ExitStack():
        gw_c = BudgetGateway(FileUsage(arm_c_home / "usage.jsonl"), ceiling, model, home=arm_c_home)
        shard_name = home.name
        for idx, body in enumerate(case_dicts, 1):
            case = Case(**body)
            for seed in seeds:
                row_a = _run_case_with_capture(case, seed, "A", gateway=gw_a, experience_home=arm_a_home / "experience")
                row_b = run_case(case, seed, "B")
                row_c = _run_case_with_capture(
                    case, seed, "C", gateway=gw_c, planning_budget=PLANNING_BOUNDS, experience_home=arm_c_home / "experience"
                )
                print(
                    f"[{shard_name} {idx}/{len(case_dicts)}] {case.id} A={row_a['status']}({row_a['seconds']:.1f}s) "
                    f"B={row_b['status']} C={row_c['status']}({row_c['seconds']:.1f}s) "
                    f"p={row_c['extra_prompts']} cost=${row_c['cost_eur']:.5f}",
                    file=sys.stderr,
                    flush=True,
                )
                out.append({"A": row_a, "B": row_b, "C": row_c})
    return out


def summarize_exchange_capture(root: Path) -> dict[str, object]:
    """Verify captured live exchanges under root retain whole prompts and raw reasoning."""
    summary: dict[str, object] = {}
    for arm_name in ("arm_a", "arm_c"):
        rows = []
        for path in sorted(root.glob(f"shard_*/{arm_name}/model_exchanges/**/*.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        ok_rows = [r for r in rows if r.get("outcome") == "ok"]
        summary[arm_name] = {
            "total": len(rows),
            "ok": len(ok_rows),
            "whole_true": sum(1 for r in ok_rows if r.get("whole") is True),
            "raw_reasoning_present": sum(1 for r in ok_rows if r.get("reasoning") or r.get("reasoning_details") or r.get("thinking")),
            "purposes": sorted({str(r.get("purpose")) for r in rows}),
        }
    return summary


def run_live_campaign(campaign: Path = CAMPAIGN, *, workers: int = 8, live_dir: Path = LIVE_ARTIFACTS_DIR) -> dict[str, object]:
    """Run frozen Arms A, B, and live-AI Arm C in sharded subprocesses with full capture on A and C."""
    from dclab_rnd.agent_eval.task_planning.probe import require_standalone, run_probe

    require_standalone()
    frozen = verify_freeze(campaign)
    model = frozen.get("live_model")
    if not model:
        raise RuntimeError("freeze_missing_live_model")
    cap = Decimal(str(frozen["cap_usd"]))
    if cap > MAX_TOTAL_USD:
        raise ValueError("usd_ceiling_out_of_authorized_bounds")
    result_path = campaign / "results" / "TP3-001_live.json"
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

    live_dir.mkdir(parents=True, exist_ok=True)
    snap_path = campaign / "price_snapshot_usd.json"
    jobs = [
        (shard, json.dumps(frozen["seeds"]), str(model), str(shard_cap), str(live_dir / f"shard_{idx}"), str(snap_path))
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

    judgment = judge_v2(scores, configured_model=True, c_integrated=True, proposal_probe=probe)
    capture = summarize_exchange_capture(live_dir)
    summary = {
        "arm_c_cost_usd": round(sum(r.get("cost_eur") or 0.0 for r in scores["C"]), 6),
        "arm_c_tokens": sum(r.get("tokens") or 0 for r in scores["C"]),
        "arm_c_provider_requests": sum(r.get("provider_requests") or 0 for r in scores["C"]),
        "arm_c_max_extra_prompts": max((r.get("extra_prompts") or 0) for r in scores["C"]),
        "arm_a_cost_usd": round(sum(r.get("cost_eur") or 0.0 for r in scores["A"]), 6),
        "arm_a_tokens": sum(r.get("tokens") or 0 for r in scores["A"]),
        "arm_a_provider_requests": sum(r.get("provider_requests") or 0 for r in scores["A"]),
        "total_live_cost_usd": round(
            sum(r.get("cost_eur") or 0.0 for r in scores["A"]) + sum(r.get("cost_eur") or 0.0 for r in scores["C"]), 6
        ),
        "model_graphs_admitted": sum(bool(r.get("model_graph_admitted")) for r in scores["C"] if r.get("phase") == "execution"),
        "execution_cases": sum(1 for r in scores["C"] if r.get("phase") == "execution"),
        "exchange_capture": capture,
        "live_artifacts_dir": str(live_dir.relative_to(ROOT)) if live_dir.is_relative_to(ROOT) else str(live_dir),
    }
    doc = {
        "schema": "dclab_rnd.task_planning_v3_result.v1",
        "campaign_id": "task_planning_v3",
        "evaluation_only": True,
        "freeze": FREEZE_NAME,
        "live_model": model,
        "cap_usd": str(cap),
        "summary": summary,
        "judgment": judgment,
        "scores": scores,
        "proposal_probe": probe,
    }
    result_path.write_text(json.dumps(doc, indent=2) + "\n")
    (campaign / "REPORT.md").write_text(_format_report(doc))
    return doc


def _format_report(doc: dict[str, object]) -> str:
    j = doc["judgment"]
    s = doc["summary"]
    g2 = j["gate_v2"]
    comp = j["comparisons"]["C"]
    lines = [
        "# Task Planning v3 Release Evaluation Report",
        "",
        f"- **Model:** `{doc['live_model']}`",
        f"- **Authorized Arm C ceiling:** `${doc['cap_usd']}` USD",
        f"- **Arm C measured usage:** `${s['arm_c_cost_usd']:.6f}` USD across `{s['arm_c_provider_requests']}` requests (`{s['arm_c_tokens']}` tokens, max `{s['arm_c_max_extra_prompts']}` prompts/case)",
        f"- **Arm A measured baseline usage:** `${s['arm_a_cost_usd']:.6f}` USD across `{s['arm_a_provider_requests']}` requests (`{s['arm_a_tokens']}` tokens)",
        f"- **Model graphs admitted:** `{s['model_graphs_admitted']} / {s['execution_cases']}` execution cases",
        f"- **Strict `tp06.gate.v1` Verdict:** **`{j['verdict']}`** (`reasons={j['reasons'] or ['none']}`)",
        f"- **Pre-declared `tp06.gate.v2` Verdict:** **`{g2['verdict']}`** (`promoted={g2['promoted']}`, `reasons={g2['reasons'] or ['none']}`)",
        f"- **Exchange Capture (`agent_runs/live_tests/task_planning_v3_20261112/`):** `{json.dumps(s['exchange_capture'])}`",
        "",
        "## Paired Metrics (Arm C vs Arm A)",
        "",
        "| Metric | Difference (C - A) | 95% Bootstrap CI | Matched Rows |",
        "|---|---:|---|---:|",
    ]
    for metric in ("correct", "completed", "recovered", "proper_abstention", "seconds", "compute_seconds", "cost_eur", "extra_prompts"):
        m = comp[metric]
        lines.append(f"| {metric} | {m['difference']} | {m['interval']} | {m['matched_case_seed_rows']} |")
    lines.extend(["", f"- Resource checks v1: `{j['resource_checks']}`", f"- Resource checks v2: `{g2['resource_checks']}`", f"- Critical safety sums: `{j['critical']}`", ""])
    return "\n".join(lines)
