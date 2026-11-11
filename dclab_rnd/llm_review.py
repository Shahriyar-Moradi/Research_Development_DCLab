"""Run pending campaign LLM critic reviews through the model gateway (purpose ``campaign_review``, the strong tier).

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

from .campaign import CAMPAIGN_DIR, CAMPAIGN_ID, sync_campaign_outputs

def default_model() -> str:
    """The campaign reviewer's model: the one the strong tier serves under the current settings (2026-11-11: never a name
    written here, so a Qwen name never reaches Anthropic's endpoint nor a Claude name OpenRouter)."""
    from .models import settings

    return settings.served_model("campaign_review") or "qwen/qwen3.8-flash"
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


def _client(root: Path, model: str):
    """The gateway's client for campaign reviews; each request is counted in the workspace's usage log."""
    from .models import for_workspace

    _load_env(root)
    client = for_workspace().client("campaign_review", model=model)
    if client is None:
        raise RuntimeError(
            "No model is configured. Add OPENAI_API_KEY (or DCLAB_TIER_STRONG_*) to .env at the repo root, then retry."
        )
    return client


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
    client: Any,
    root: Path,
    task: dict[str, Any],
    *,
    model: str,
    persist: bool = True,
    keep: Path | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    """One critique. ``persist`` writes it onto the experiment's result (the campaign's own reviewer); ``keep`` (29.18,
    a second reviewer beside the first) writes exactly what was sent and the whole raw reply to ``keep/<id>.json`` before
    anything is parsed or cut, and never touches the result."""
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
    response = client.complete(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Critique this experiment evidence package and return JSON only.\n\n"
                    + json.dumps(payload, indent=2)
                ),
            },
        ],
        max_tokens=max_tokens,  # the gateway's default unless asked again with more room (a cut-off JSON critique cannot be parsed)
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    content = response.get("content") or "{}"
    if keep is not None:  # whole, first: the owner's rule (2026-10-18)
        keep.mkdir(parents=True, exist_ok=True)
        (keep / f"{task.get('experiment_id')}.json").write_text(json.dumps(
            {"experiment_id": task.get("experiment_id"), "model": model, "at": _utc_now(), "system": SYSTEM_PROMPT, "sent": payload,
             "raw_reply": content, "usage": response.get("usage"), "finish_reason": response.get("finish_reason")},
            indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    report = getattr(client, "output", None) or (lambda passed, reason="": None)
    try:
        review = _parse_json_response(content)
    except Exception as error:
        report(False, f"the critique is not valid JSON ({type(error).__name__})")
        raise
    if not isinstance(review, dict):
        report(False, "the critique is not a JSON object")
        raise ValueError("The critique is not a JSON object")
    report(True)
    review["experiment_id"] = task.get("experiment_id")
    review["dataset"] = task.get("dataset")
    review["evidence_path"] = evidence_rel
    review["model"] = model
    review["reviewed_at"] = _utc_now()
    review["role"] = "critic_and_hypothesis_generator_only"
    usage = response.get("usage") or {}
    if usage:
        review["usage"] = {
            "prompt_tokens": usage.get("input_tokens"),
            "completion_tokens": usage.get("output_tokens"),
            "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
        }

    if not persist:  # a second reviewer: its review lives in its own files, the result keeps the campaign's
        return review
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
    model: str | None = None,
    limit: int | None = None,
    experiment_ids: list[str] | None = None,
    sleep_s: float = 0.4,
    force: bool = False,
    beside: bool = False,
    cap_eur: float | None = None,
) -> int:
    """``beside`` (29.18): review every completed experiment as a second reviewer: the reviews go to this model's own
    memory and to ``reviews/<model>/<id>.json`` (what was sent and the whole raw reply); no result, no shared memory and
    no campaign report is changed, and nothing earlier is deleted."""
    model = model or default_model()
    client = _client(root, model)
    if cap_eur is not None:  # the most this run may spend; the gateway stops the request that would pass it
        client.run_limit_eur = cap_eur
    if beside:
        return _beside(client, root, model, experiment_ids, limit, sleep_s)
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


RETRY_MAX_TOKENS = 24_000  # a reply cut at the default limit (thinking took most of it) is asked once more with this room


def _cut(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("finish_reason") == "length"
    except ValueError:
        return False


def _supersede(keep: Path, eid: str) -> None:
    """Move a cut review aside, whole and unchanged, before it is asked again: nothing earlier is deleted."""
    old = keep / f"{eid}.json"
    aside = keep / "superseded"
    aside.mkdir(exist_ok=True)
    n = 1 + sum(1 for _ in aside.glob(f"{eid}.*.json"))
    old.replace(aside / f"{eid}.{n}.json")


def _beside(client: Any, root: Path, model: str, experiment_ids: list[str] | None, limit: int | None, sleep_s: float) -> int:
    model_memory = memory_name_for_model(model)
    keep = root / CAMPAIGN_DIR / "reviews" / "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model)
    kept = list(keep.glob("*.json")) if keep.is_dir() else []
    cut = {p.stem for p in kept if _cut(p)}
    done = {p.stem for p in kept} - cut
    tasks = [t for t in load_all_review_tasks(root) if t.get("experiment_id") not in done]  # a whole review kept once is not asked again
    if experiment_ids:
        wanted = {eid.strip() for eid in experiment_ids}
        tasks = [t for t in tasks if t.get("experiment_id") in wanted]
    if limit is not None:
        tasks = tasks[: max(0, limit)]
    print(f"Reviewing {len(tasks)} experiment(s) beside the campaign's reviewer with model={model}; kept in {keep.relative_to(root)}")
    failures = 0
    for idx, task in enumerate(tasks, start=1):
        eid = task.get("experiment_id")
        try:
            if eid in cut:
                _supersede(keep, eid)
            review = review_one(client, root, task, model=model, persist=False, keep=keep,
                                max_tokens=RETRY_MAX_TOKENS if eid in cut else None)
            append_memory(root, review, memory_filename=model_memory)
            print(f"[{idx}/{len(tasks)}] {task.get('experiment_id')} | confidence={review.get('overall_confidence')}", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep the batch going; the raw reply is already kept
            failures += 1
            print(f"[{idx}/{len(tasks)}] {task.get('experiment_id')} FAILED: {exc}", flush=True)
            if "cap" in str(exc).lower():
                break
        if sleep_s > 0 and idx < len(tasks):
            time.sleep(sleep_s)
    return 1 if failures else 0
