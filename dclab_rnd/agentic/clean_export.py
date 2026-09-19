"""Additive clean views + SLM fine-tuning corpus from Research Studio archives.

Does **not** delete or rewrite existing trial-*/events/reports artifacts.
Writes beside them:

  <run_id>/clean/          human + machine-simple cards
  agent_runs/clean_exports/ studio-wide leaderboard + SFT JSONL
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from .archive import run_label, _write_json, _now

SCHEMA_VERSION = 1

SYSTEM_ML_ENGINEER = (
    "You are a DCLab tabular ML research specialist. Propose and interpret "
    "experiments only from provided measurements and IDs. Never fabricate "
    "metrics, citations, or production approval. Distinguish leakage risks "
    "from importance scores. Development CV is adaptive reuse, not an "
    "untouched test set. Prefer paired comparisons and clear limitations."
)


def _round_metrics(metrics: dict[str, Any] | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in (metrics or {}).items():
        if isinstance(value, (int, float)):
            out[key] = round(float(value), 6)
    return out


def compact_trial_from_disk(run_id: str, trial_dir: Path) -> dict[str, Any] | None:
    """One simple trial card from recipe/result/META — no OOF rows."""
    result_path = trial_dir / "result.json"
    recipe_path = trial_dir / "recipe.json"
    meta_path = trial_dir / "META.json"
    if not result_path.exists() and not meta_path.exists():
        return None

    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else {}
    recipe = json.loads(recipe_path.read_text(encoding="utf-8")) if recipe_path.exists() else {}
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    plan = result.get("plan") or recipe.get("plan") or meta.get("plan") or {}
    metrics = _round_metrics(result.get("metrics") or {})
    features = plan.get("features") or []
    feature_names = [
        f.get("name") if isinstance(f, dict) else str(f) for f in features
    ]
    stress = result.get("stress_tests") or []
    stress_brief = []
    for item in stress[:6]:
        if not isinstance(item, dict):
            continue
        stress_brief.append(
            {
                "columns": item.get("columns") or item.get("masked") or item.get("stress_columns"),
                "roc_auc": _round_metrics({"roc_auc": item.get("roc_auc") or (item.get("metrics") or {}).get("roc_auc")}).get("roc_auc"),
                "auc_drop": item.get("auc_drop") or item.get("delta_auc"),
            }
        )
    sensitivity = []
    for item in (result.get("input_sensitivity") or [])[:5]:
        if isinstance(item, dict):
            sensitivity.append(
                {
                    "column": item.get("column") or item.get("feature"),
                    "auc_drop": item.get("auc_drop"),
                }
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "trial_id": meta.get("trial_id") or f"{run_id}:{trial_dir.name}",
        "trial_dir": trial_dir.name,
        "status": meta.get("status") or ("completed" if metrics else "unknown"),
        "dataset": plan.get("dataset") or result.get("dataset"),
        "title": plan.get("title") or meta.get("title"),
        "hypothesis": plan.get("hypothesis"),
        "model": plan.get("model"),
        "parameters": {
            k: v for k, v in (plan.get("parameters") or {}).items() if v is not None
        },
        "engineered_features": feature_names,
        "drop_columns": plan.get("drop_columns") or [],
        "stress_columns": plan.get("stress_columns") or [],
        "reference_evidence_id": plan.get("reference_evidence_id"),
        "evidence_ids": plan.get("evidence_ids") or [],
        "rows": result.get("rows"),
        "feature_count": result.get("feature_count"),
        "metrics": metrics,
        "fold_auc_sd": (result.get("fold_standard_deviation") or {}).get("roc_auc"),
        "top_sensitivity": sensitivity,
        "stress_brief": [s for s in stress_brief if s.get("columns") is not None or s.get("roc_auc") is not None],
        "limitations": (result.get("limitations") or [])[:8],
        "production_approved": result.get("production_approved", False),
        "wall_seconds": result.get("wall_seconds"),
        "artifacts": {
            "recipe": f"{trial_dir.name}/recipe.json",
            "result": f"{trial_dir.name}/result.json",
            "oof": f"{trial_dir.name}/oof_predictions.jsonl",
        },
    }


def build_run_card(store, run_id: str) -> dict[str, Any]:
    run = store.get(run_id)
    cfg = run.get("config") or {}
    root = store.home / run_id
    trials = []
    for trial_dir in sorted(root.glob("trial-*")):
        if trial_dir.is_dir():
            card = compact_trial_from_disk(run_id, trial_dir)
            if card:
                trials.append(card)

    leaderboard = sorted(
        trials,
        key=lambda t: (t.get("metrics") or {}).get("roc_auc") or -1.0,
        reverse=True,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "run_id": run_id,
        "label": run_label(run),
        "status": run.get("status"),
        "phase": run.get("phase"),
        "error": run.get("error"),
        "project": cfg.get("project"),
        "datasets": cfg.get("datasets"),
        "llm_model": cfg.get("model"),
        "goal": cfg.get("goal"),
        "max_experiments": cfg.get("max_experiments"),
        "max_rows": cfg.get("max_rows"),
        "repeats": cfg.get("repeats"),
        "llm_calls": run.get("llm_calls"),
        "usage": run.get("usage") or {},
        "trial_count": len(trials),
        "best_cv_auc": (leaderboard[0].get("metrics") or {}).get("roc_auc") if leaderboard else None,
        "leaderboard": [
            {
                "rank": i + 1,
                "trial_dir": t["trial_dir"],
                "title": t.get("title"),
                "model": t.get("model"),
                "roc_auc": (t.get("metrics") or {}).get("roc_auc"),
                "average_precision": (t.get("metrics") or {}).get("average_precision"),
                "log_loss": (t.get("metrics") or {}).get("log_loss"),
            }
            for i, t in enumerate(leaderboard)
        ],
        "trials": trials,
        "how_to_read": {
            "prefer": [
                "clean/run_card.md — one-page story",
                "clean/trials.jsonl — one compact trial per line",
                "clean/leaderboard.json — sorted AUC table",
            ],
            "raw_kept": [
                "trial-*/result.json — full diagnostics",
                "trial-*/oof_predictions.jsonl — row-level CV probs",
                "reports/STUDIO_RUN_REPORT.md — full UI narrative",
            ],
            "slm": "Studio-wide fine-tuning corpus: clean_exports/sft_chat.jsonl",
        },
    }


def _run_card_markdown(card: dict[str, Any]) -> str:
    lines = [
        f"# {card.get('label')}",
        "",
        f"**Status:** {card.get('status')} · **Project:** {card.get('project') or '—'}  ",
        f"**Datasets:** {', '.join(card.get('datasets') or [])}  ",
        f"**LLM:** `{card.get('llm_model')}` · calls: {card.get('llm_calls')}  ",
        f"**Run ID:** `{card.get('run_id')}`  ",
        f"**Best CV AUC:** {card.get('best_cv_auc') if card.get('best_cv_auc') is not None else '—'}  ",
        f"**Trials on disk:** {card.get('trial_count')}",
        "",
        "## Goal",
        "",
        card.get("goal") or "—",
        "",
        "## Leaderboard (development CV)",
        "",
        "| Rank | Trial | Model | ROC-AUC | AP | Log loss |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in card.get("leaderboard") or []:
        lines.append(
            f"| {row['rank']} | {row.get('trial_dir')} · {(row.get('title') or '')[:48]} | "
            f"{row.get('model')} | {row.get('roc_auc') if row.get('roc_auc') is not None else '—'} | "
            f"{row.get('average_precision') if row.get('average_precision') is not None else '—'} | "
            f"{row.get('log_loss') if row.get('log_loss') is not None else '—'} |"
        )
    if not card.get("leaderboard"):
        lines.append("| — | no trials | — | — | — | — |")
    lines += [
        "",
        "## Clean vs raw",
        "",
        "- Use this folder for scanning and SLM prep.",
        "- Keep `trial-*/` and `reports/` as authoritative raw evidence.",
        "- OOF predictions stay in `trial-*/oof_predictions.jsonl` (not copied here).",
        "",
    ]
    return "\n".join(lines)


def write_run_clean(store, run_id: str) -> Path:
    """Write <run_id>/clean/ beside existing artifacts."""
    root = store.home / run_id
    clean = root / "clean"
    if clean.exists():
        shutil.rmtree(clean)
    clean.mkdir(parents=True, exist_ok=True)

    card = build_run_card(store, run_id)
    _write_json(clean / "run_card.json", card)
    (clean / "run_card.md").write_text(_run_card_markdown(card), encoding="utf-8")
    _write_json(clean / "leaderboard.json", card.get("leaderboard") or [])

    trials_path = clean / "trials.jsonl"
    with trials_path.open("w", encoding="utf-8") as handle:
        for trial in card.get("trials") or []:
            handle.write(json.dumps(trial, ensure_ascii=False, default=str) + "\n")

    (clean / "README.md").write_text(
        "\n".join(
            [
                "# Clean view (additive)",
                "",
                "This folder is a **simplified projection** of the run.",
                "Nothing under `trial-*/`, `events/`, or `reports/` was removed.",
                "",
                "| File | Use |",
                "|---|---|",
                "| `run_card.md` | Human one-pager |",
                "| `run_card.json` | Full clean machine card |",
                "| `leaderboard.json` | Sorted CV metrics only |",
                "| `trials.jsonl` | One compact trial per line (no OOF) |",
                "",
                "Studio-wide SLM corpus: `../../clean_exports/`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return clean


def _event_payloads(store, run_id: str, kind: str) -> list[dict[str, Any]]:
    items = []
    for event in store.events(run_id):
        if event.get("kind") == kind:
            payload = event.get("payload")
            if isinstance(payload, dict):
                items.append({"seq": event.get("seq"), "payload": payload})
    return items


def _compact_context_trials(trials: list[dict[str, Any]], *, limit: int = 6) -> list[dict[str, Any]]:
    out = []
    for trial in trials[-limit:]:
        out.append(
            {
                "trial_id": trial.get("trial_id"),
                "title": trial.get("title"),
                "model": trial.get("model"),
                "parameters": trial.get("parameters"),
                "engineered_features": trial.get("engineered_features"),
                "metrics": trial.get("metrics"),
                "stress_columns": trial.get("stress_columns"),
                "top_sensitivity": trial.get("top_sensitivity"),
            }
        )
    return out


def build_sft_records_for_run(store, run_id: str) -> list[dict[str, Any]]:
    """Chat-style SFT examples for DS/ML-engineering SLM fine-tuning."""
    run = store.get(run_id)
    cfg = run.get("config") or {}
    label = run_label(run)
    card = build_run_card(store, run_id)
    trials = card.get("trials") or []
    records: list[dict[str, Any]] = []
    base_meta = {
        "run_id": run_id,
        "label": label,
        "datasets": cfg.get("datasets"),
        "project": cfg.get("project"),
        "llm_model_source": cfg.get("model"),
        "domain": "tabular_ml_engineering",
    }

    def add(task: str, user: Any, assistant: Any, *, seq: Any = None) -> None:
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "id": f"{run_id}:{task}:{len(records):04d}",
                "task": task,
                "source": {**base_meta, "event_seq": seq},
                "messages": [
                    {"role": "system", "content": SYSTEM_ML_ENGINEER},
                    {
                        "role": "user",
                        "content": json.dumps(user, ensure_ascii=False, default=str),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(assistant, ensure_ascii=False, default=str)
                        if not isinstance(assistant, str)
                        else assistant,
                    },
                ],
            }
        )

    for item in _event_payloads(store, run_id, "agenda"):
        add(
            "plan_agenda",
            {
                "task": "Write a bounded research agenda.",
                "goal": cfg.get("goal"),
                "datasets": cfg.get("datasets"),
                "constraints": {
                    "max_experiments": cfg.get("max_experiments"),
                    "max_rows": cfg.get("max_rows"),
                    "repeats": cfg.get("repeats"),
                },
            },
            item["payload"],
            seq=item["seq"],
        )

    for item in _event_payloads(store, run_id, "data_review"):
        add(
            "review_data_leakage",
            {
                "task": "Review leakage risks and prediction-time availability.",
                "datasets": cfg.get("datasets"),
                "note": "Profiles are summarized elsewhere; reason from stated findings.",
            },
            item["payload"],
            seq=item["seq"],
        )

    proposals = _event_payloads(store, run_id, "proposal")
    critiques = _event_payloads(store, run_id, "critique")
    for index, item in enumerate(proposals):
        prior = _compact_context_trials(trials[:index])
        prior_critique = critiques[index - 1]["payload"] if index > 0 and index - 1 < len(critiques) else None
        add(
            "propose_experiment",
            {
                "task": "Propose ONE next executable tabular experiment.",
                "datasets": cfg.get("datasets"),
                "attempt_index": index,
                "prior_trials": prior,
                "latest_critique": prior_critique,
            },
            item["payload"],
            seq=item["seq"],
        )

    for index, item in enumerate(critiques):
        trial = trials[index] if index < len(trials) else None
        add(
            "critique_experiment",
            {
                "task": "Critique this development-CV experiment scientifically.",
                "trial": trial,
                "all_trial_metrics": [
                    {
                        "trial_dir": t.get("trial_dir"),
                        "model": t.get("model"),
                        "metrics": t.get("metrics"),
                    }
                    for t in trials[: index + 1]
                ],
            },
            item["payload"],
            seq=item["seq"],
        )

    for item in _event_payloads(store, run_id, "synthesis"):
        add(
            "synthesize_lessons",
            {
                "task": "Synthesize provisional lessons with evidence IDs and limitations.",
                "leaderboard": card.get("leaderboard"),
                "trials": _compact_context_trials(trials, limit=12),
            },
            item["payload"],
            seq=item["seq"],
        )

    return records


def write_studio_clean_exports(store) -> Path:
    """Studio-wide clean leaderboard + SFT corpus (additive)."""
    export_root = store.home / "clean_exports"
    export_root.mkdir(parents=True, exist_ok=True)

    all_trials: list[dict[str, Any]] = []
    all_sft: list[dict[str, Any]] = []
    run_summaries: list[dict[str, Any]] = []

    for run in store.list():
        run_id = run["id"]
        if not (store.home / run_id).exists():
            continue
        write_run_clean(store, run_id)
        card = build_run_card(store, run_id)
        run_summaries.append(
            {
                "run_id": run_id,
                "label": card.get("label"),
                "status": card.get("status"),
                "datasets": card.get("datasets"),
                "trial_count": card.get("trial_count"),
                "best_cv_auc": card.get("best_cv_auc"),
                "llm_model": card.get("llm_model"),
                "clean_path": f"{run_id}/clean/",
            }
        )
        for trial in card.get("trials") or []:
            all_trials.append({**trial, "run_id": run_id, "label": card.get("label")})
        all_sft.extend(build_sft_records_for_run(store, run_id))

    trials_path = export_root / "trials_all.jsonl"
    with trials_path.open("w", encoding="utf-8") as handle:
        for trial in all_trials:
            handle.write(json.dumps(trial, ensure_ascii=False, default=str) + "\n")

    sft_path = export_root / "sft_chat.jsonl"
    with sft_path.open("w", encoding="utf-8") as handle:
        for record in all_sft:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    # Split by task for selective fine-tuning.
    by_task = export_root / "sft_by_task"
    if by_task.exists():
        shutil.rmtree(by_task)
    by_task.mkdir(parents=True, exist_ok=True)
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in all_sft:
        buckets.setdefault(record["task"], []).append(record)
    for task, items in buckets.items():
        path = by_task / f"{task}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in items:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    csv_path = export_root / "leaderboard.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank_global",
                "run_id",
                "label",
                "dataset",
                "trial_dir",
                "title",
                "model",
                "roc_auc",
                "average_precision",
                "log_loss",
                "brier",
            ],
        )
        writer.writeheader()
        ranked = sorted(
            all_trials,
            key=lambda t: (t.get("metrics") or {}).get("roc_auc") or -1.0,
            reverse=True,
        )
        for i, trial in enumerate(ranked, start=1):
            metrics = trial.get("metrics") or {}
            writer.writerow(
                {
                    "rank_global": i,
                    "run_id": trial.get("run_id"),
                    "label": trial.get("label"),
                    "dataset": trial.get("dataset"),
                    "trial_dir": trial.get("trial_dir"),
                    "title": trial.get("title"),
                    "model": trial.get("model"),
                    "roc_auc": metrics.get("roc_auc"),
                    "average_precision": metrics.get("average_precision"),
                    "log_loss": metrics.get("log_loss"),
                    "brier": metrics.get("brier"),
                }
            )

    task_counts = {task: len(items) for task, items in buckets.items()}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "studio_home": str(store.home),
        "run_count": len(run_summaries),
        "trial_count": len(all_trials),
        "sft_record_count": len(all_sft),
        "sft_task_counts": task_counts,
        "runs": run_summaries,
        "files": {
            "trials_all.jsonl": "Compact trials across all runs (no OOF)",
            "leaderboard.csv": "Sorted development-CV metrics",
            "sft_chat.jsonl": "OpenAI-style chat SFT for DS/ML engineering SLM",
            "sft_by_task/": "Same SFT records split by task name",
        },
    }
    _write_json(export_root / "MANIFEST.json", manifest)

    (export_root / "README.md").write_text(
        "\n".join(
            [
                "# Clean exports (additive)",
                "",
                "Generated beside existing Research Studio archives. **Nothing raw was deleted.**",
                "",
                "## For humans",
                "",
                "1. Open `leaderboard.csv` for a simple cross-run score table.",
                "2. Open each run’s `../<run_id>/clean/run_card.md` for a one-pager.",
                "3. Use `trials_all.jsonl` when you want metrics without OOF noise.",
                "",
                "## For SLM fine-tuning (data science / ML engineering)",
                "",
                "| File | Format |",
                "|---|---|",
                "| `sft_chat.jsonl` | `{id, task, source, messages[system,user,assistant]}` |",
                "| `sft_by_task/*.jsonl` | Same records filtered by skill |",
                "",
                "Tasks:",
                "",
                "- `plan_agenda`",
                "- `review_data_leakage`",
                "- `propose_experiment`",
                "- `critique_experiment`",
                "- `synthesize_lessons`",
                "",
                "User/assistant contents are JSON strings so the SLM learns structured "
                "scientific decisions. OOF predictions and huge fold matrices are excluded "
                "from prompts to keep examples clean.",
                "",
                "## Regenerate",
                "",
                "```bash",
                ".venv-agent/bin/python -m dclab_rnd.agentic archive --all",
                "# or only clean exports:",
                ".venv-agent/bin/python -m dclab_rnd.agentic export-clean",
                "```",
                "",
                f"Last generated: `{manifest['generated_at']}` · "
                f"trials={manifest['trial_count']} · sft={manifest['sft_record_count']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return export_root
