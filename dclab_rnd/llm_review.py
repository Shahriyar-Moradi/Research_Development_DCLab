"""Run pending campaign LLM critic reviews via OpenAI.

The LLM may challenge claims and propose next experiments. It must not rewrite
observed metrics or approve deployment. Deterministic evidence stays authoritative.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

from .campaign import CAMPAIGN_DIR, CAMPAIGN_ID, sync_campaign_outputs

DEFAULT_MODEL = "gpt-5.6-terra"
MEMORY_NAME = "llm_review_memory.jsonl"


def memory_name_for_model(model: str) -> str:
    """Stable filename for a model-specific review memory (e.g. gpt-5.6-terra)."""
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model.strip())
    return f"llm_review_memory__{safe}.jsonl"
SYSTEM_PROMPT = """You are the DCLab evidence critic for tabular ML research.

Role: critic and hypothesis generator ONLY.
You may challenge unsupported claims, flag leakage/validation risks, and propose the smallest falsifiable next experiment.
You may NOT rewrite observed metrics, invent numbers, approve deployment, or claim causality from predictive scores.

Mandatory answer format (return valid JSON only):
{
  "experiment_id": "...",
  "observed_evidence": [{"statement": "...", "citation": "EXPERIMENT_ID / evidence.path"}],
  "interpretation": "...",
  "decision": "...",
  "risks_and_missing_evidence": ["..."],
  "challenges_to_claims": [{"claim_id": "...", "issue": "...", "severity": "low|medium|high"}],
  "next_experiment": {
    "title": "...",
    "hypothesis": "...",
    "success_gate": "...",
    "why_smallest": "..."
  },
  "overall_confidence": "low|medium|high"
}

Rules:
- Separate facts, inferences, hypotheses, and recommendations.
- Cite experiment_id and exact evidence paths for every factual statement.
- Never infer causal effects from predictive evidence.
- Never approve a feature without decision-time and production-availability review.
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_env(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        # Minimal parser if python-dotenv is unavailable
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _compact_evidence(result: dict[str, Any]) -> dict[str, Any]:
    """Keep prompt-sized evidence; drop bulky fold matrices."""
    evidence = result.get("evidence", {})
    keep_keys = [
        "source_rows",
        "feature_count",
        "positive_rate_train",
        "detector_canary_passed",
        "declared_leakage_features",
        "safe_vs_unsafe_training_cv",
        "stage_results",
        "selection_rule",
        "model_results",
        "holdout_metrics",
        "holdout_roc_auc_bootstrap_ci",
        "holdout_consumed",
        "top_final_features",
        "cv_feature_stability",
        "production_feature_rule",
        "feature_profiles",
    ]
    compact: dict[str, Any] = {}
    for key in keep_keys:
        if key not in evidence:
            continue
        value = evidence[key]
        if key == "feature_profiles" and isinstance(value, list):
            # Keep only flagged / top rows if present
            flagged = [row for row in value if isinstance(row, dict) and row.get("review_flags")]
            compact[key] = flagged[:20] if flagged else value[:10]
        elif key in {"stage_results", "model_results"} and isinstance(value, list):
            compact[key] = value[:12]
        elif key == "top_final_features" and isinstance(value, list):
            compact[key] = value[:15]
        else:
            compact[key] = value
    return {
        "experiment_id": result.get("experiment_id"),
        "dataset": result.get("dataset"),
        "kind": result.get("kind"),
        "question": result.get("question"),
        "status": result.get("status"),
        "claims": result.get("claims", []),
        "evidence": compact,
        "limitations": result.get("limitations", []),
    }


def _parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


def _client(root: Path) -> OpenAI:
    _load_env(root)
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY missing. Add it to .env at the repo root, then retry."
        )
    return OpenAI(api_key=key)


def load_queue(root: Path) -> list[dict[str, Any]]:
    path = root / CAMPAIGN_DIR / "llm_review_queue.jsonl"
    if not path.exists():
        return []
    tasks = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            tasks.append(json.loads(line))
    return tasks


def _task_from_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "DCLab evidence critic",
        "experiment_id": result.get("experiment_id"),
        "dataset": result.get("dataset"),
        "question": result.get("question"),
        "claims": result.get("claims", []),
        "evidence_path": result.get("_path")
        or f"{CAMPAIGN_DIR}/results/{result.get('experiment_id')}_{result.get('dataset')}_{result.get('kind')}.json",
        "instructions": result.get("llm_review", {}).get("required_outputs", []),
        "rules": [
            "Separate observed facts, inferences, hypotheses, and recommendations.",
            "Cite the experiment ID and exact evidence path for every factual statement.",
            "Never infer causal effects from predictive evidence.",
            "Never approve a feature without decision-time and production-availability review.",
        ],
    }


def load_all_review_tasks(root: Path) -> list[dict[str, Any]]:
    """Build critic tasks from every completed experiment result (for force re-review)."""
    results_dir = root / CAMPAIGN_DIR / "results"
    tasks: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("EXP-*.json")):
        result = json.loads(path.read_text())
        if result.get("status") != "completed":
            continue
        result["_path"] = path.relative_to(root).as_posix()
        tasks.append(_task_from_result(result))
    return tasks


def review_one(
    client: OpenAI,
    root: Path,
    task: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    evidence_rel = task["evidence_path"]
    evidence_path = root / evidence_rel
    result = json.loads(evidence_path.read_text())
    payload = {
        "task": task.get("task"),
        "question": task.get("question"),
        "instructions": task.get("instructions"),
        "rules": task.get("rules"),
        "claims": task.get("claims"),
        "evidence_bundle": _compact_evidence(result),
    }
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Critique this experiment evidence package and return JSON only.\n\n"
                    + json.dumps(payload, indent=2)
                ),
            },
        ],
    )
    content = response.choices[0].message.content or "{}"
    review = _parse_json_response(content)
    review["experiment_id"] = task.get("experiment_id")
    review["dataset"] = task.get("dataset")
    review["evidence_path"] = evidence_rel
    review["model"] = model
    review["reviewed_at"] = _utc_now()
    review["role"] = "critic_and_hypothesis_generator_only"
    usage = getattr(response, "usage", None)
    if usage is not None:
        review["usage"] = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    # Persist onto the experiment result
    result.setdefault("llm_review", {})
    result["llm_review"].update(
        {
            "status": "completed",
            "role": "critic_and_hypothesis_generator_only",
            "required_outputs": task.get("instructions", []),
            "model": model,
            "reviewed_at": review["reviewed_at"],
            "review": review,
        }
    )
    evidence_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return review


def _memory_record(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "experiment_id": review.get("experiment_id"),
        "dataset": review.get("dataset"),
        "model": review.get("model"),
        "reviewed_at": review.get("reviewed_at"),
        "decision": review.get("decision"),
        "next_experiment": review.get("next_experiment"),
        "challenges_to_claims": review.get("challenges_to_claims", []),
        "risks_and_missing_evidence": review.get("risks_and_missing_evidence", []),
        "overall_confidence": review.get("overall_confidence"),
        "evidence_path": review.get("evidence_path"),
    }


def append_memory(
    root: Path,
    review: dict[str, Any],
    *,
    memory_filename: str = MEMORY_NAME,
) -> Path:
    path = root / CAMPAIGN_DIR / memory_filename
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_memory_record(review), sort_keys=True) + "\n")
    return path


def run_llm_reviews(
    root: Path,
    *,
    model: str = DEFAULT_MODEL,
    limit: int | None = None,
    experiment_ids: list[str] | None = None,
    sleep_s: float = 0.4,
    force: bool = False,
) -> int:
    client = _client(root)
    tasks = load_all_review_tasks(root) if force else load_queue(root)
    if experiment_ids:
        wanted = {eid.strip() for eid in experiment_ids}
        tasks = [t for t in tasks if t.get("experiment_id") in wanted]
    if limit is not None:
        tasks = tasks[: max(0, limit)]
    if not tasks:
        print("No pending LLM review tasks." + (" Use --force to re-review completed ones." if not force else ""))
        return 0

    model_memory = memory_name_for_model(model)
    model_memory_path = root / CAMPAIGN_DIR / model_memory
    if force and model_memory_path.exists():
        model_memory_path.unlink()
    # Keep the shared memory file aligned with the latest critic pass.
    shared_memory = root / CAMPAIGN_DIR / MEMORY_NAME
    if force and shared_memory.exists():
        archive = root / CAMPAIGN_DIR / "llm_review_memory__previous.jsonl"
        if not archive.exists():
            shared_memory.replace(archive)
        else:
            shared_memory.unlink()

    print(f"Running {len(tasks)} LLM review(s) with model={model}")
    print(f"Model memory → {CAMPAIGN_DIR}/{model_memory}")
    failures = 0
    for idx, task in enumerate(tasks, start=1):
        eid = task.get("experiment_id")
        print(f"[{idx}/{len(tasks)}] {eid} ({task.get('dataset')}) ...", flush=True)
        try:
            review = review_one(client, root, task, model=model)
            append_memory(root, review, memory_filename=model_memory)
            append_memory(root, review, memory_filename=MEMORY_NAME)
            decision = (review.get("decision") or "")[:120]
            print(f"  done | confidence={review.get('overall_confidence')} | {decision}")
        except Exception as exc:  # noqa: BLE001 - keep batch going
            failures += 1
            print(f"  FAILED: {exc}")
        if sleep_s > 0 and idx < len(tasks):
            time.sleep(sleep_s)

    changed = sync_campaign_outputs(root)
    print("Refreshed campaign artifacts:" + (", ".join(changed) if changed else " none"))
    print(f"Memory: {CAMPAIGN_DIR}/{model_memory}")
    print(f"Shared memory: {CAMPAIGN_DIR}/{MEMORY_NAME}")
    if failures:
        print(f"Completed with {failures} failure(s).")
        return 1
    return 0
