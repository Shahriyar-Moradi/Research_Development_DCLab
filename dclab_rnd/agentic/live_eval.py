"""Small, frozen pilot of real specialist calls; never a release certification."""
import argparse
import asyncio
import time
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from nooa.unifiedllm import ResponsesClient, RetryConfig

from . import agents
from .catalog import ROOT
from .schemas import Experiment


AGENTS = {name: getattr(agents, name) for name in (
    "ResearchPlanner", "DataScientist", "ExperimentDesigner", "ResultsCritic", "KnowledgeCurator"
)}


class MeteredClient(ResponsesClient):
    def __init__(self, model):
        super().__init__(model="openai/" + model, store=False, max_tokens=6000,
                         retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0), num_retries=0)
        self.calls = 0
        self.usage = {}

    async def acall(self, messages, **kwargs):
        from ..models import installed  # NOOA keeps its own HTTP client; the gateway still counts every request
        self.calls += 1
        started = time.monotonic()
        try:
            response = await asyncio.wait_for(super().acall(messages, **kwargs), timeout=180)
        except Exception as exc:
            installed().record("campaign", self.model, None, time.monotonic() - started, 1, f"{type(exc).__name__}: the model request failed")
            raise
        usage = response.usage or {}
        installed().record("campaign", self.model, {"input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
                                                     "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens")},
                           time.monotonic() - started, 1, "ok")
        for key, value in (response.usage or {}).items():
            if isinstance(value, (int, float)):
                self.usage[key] = self.usage.get(key, 0) + value
        return response


def grade(case, answer):
    """Conservative hard checks. Passing does not replace domain-expert review."""
    name = case["hard_check"]
    lower = json.dumps(answer, ensure_ascii=False).lower()
    if name == "mentions_prediction_time_gap":
        return any(x in lower for x in ("before receipt", "pre-receipt", "prediction time", "not available", "unavailable"))
    if name == "flags_duration":
        return any("duration" in x.lower() for x in answer["leakage_risks"])
    if name == "valid_first_baseline":
        proposal = Experiment.model_validate(answer)
        return (proposal.dataset == "bank_marketing" and proposal.model in ("dummy", "logistic_regression")
                and not proposal.features and not proposal.evidence_ids and proposal.reference_evidence_id is None
                and "duration" not in proposal.stress_columns
                and all("duration" not in f.inputs for f in proposal.features))
    if name == "cites_both_and_mentions_tradeoff":
        ids = set(answer["evidence_ids"])
        return ({"eval:trial-001", "eval:trial-002"} <= ids
                and ("auc" in lower or "roc" in lower)
                and ("precision" in lower or "ap" in lower)
                and ("brier" in lower or "log loss" in lower or "calibrat" in lower))
    if name == "no_empirical_lesson_from_failure":
        return answer["lessons"] == []
    raise ValueError("Unknown hard check")


async def evaluate(manifest, model, limit=None):
    records = []
    for case in manifest["cases"][:limit]:
        client = MeteredClient(model)
        record = {"id": case["id"], "agent": case["agent"], "hard_check": case["hard_check"],
                  "expert_question": case["expert_question"], "human_review": None,
                  "context_sha256": hashlib.sha256(json.dumps(case["context"], sort_keys=True).encode()).hexdigest()}
        try:
            answer = await asyncio.wait_for(agents.ask(AGENTS[case["agent"]], case["method"], client, case["context"]), timeout=190)
            record.update(status="completed", schema_valid=True, hard_check_pass=bool(grade(case, answer)), answer=answer)
        except Exception as exc:
            # Error text may contain request details; retain only the safe type.
            record.update(status="failed", schema_valid=False, hard_check_pass=False, error_type=type(exc).__name__)
        record.update(model_calls=client.calls, usage=client.usage)
        records.append(record)
        if record.get("error_type") in ("RateLimitError", "AuthenticationError", "PermissionDeniedError", "InternalServerError"):
            break
    return records


def main():
    from ..models import for_workspace, install
    install(for_workspace())  # count this tool's model requests in the workspace's usage log
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "evaluation/live_cases_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path; evaluations are immutable snapshots")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("OPENAI_API_KEY is missing; do not pass it on the command line")
    model = args.model or os.environ.get("OPENAI_MODEL", "gpt-5.6-terra")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    records = asyncio.run(evaluate(manifest, model, args.limit))
    result = {"suite_id": manifest["suite_id"], "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "created_at": datetime.now(timezone.utc).isoformat(), "model": model,
              "cases_requested": len(manifest["cases"][:args.limit]), "cases_run": len(records),
              "cases_not_run": len(manifest["cases"][:args.limit]) - len(records),
              "hard_check_passed": sum(r["hard_check_pass"] for r in records),
              "schema_valid": sum(r["schema_valid"] for r in records), "records": records,
              "overall_release_ready": False,
              "limitation": "Researcher-authored development pilot; lexical checks and no independent expert adjudication."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"output": str(args.output), "cases_run": len(records),
                      "hard_check_passed": result["hard_check_passed"],
                      "failures": [r["error_type"] for r in records if r["status"] == "failed"]}))
    if any(r["status"] == "failed" for r in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
