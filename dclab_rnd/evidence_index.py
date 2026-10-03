"""Structured, citable evidence index for DCLab agents, copilots and SFT builders.

Every piece of R&D output is turned into a *self-contained record*: a rule, a
workflow block, a dataset card, an experiment card (dataset + stage + setup +
findings + critic review), a leakage precedent, or a cross-study finding.

Design choices (see docs/guides/AGENT_KNOWLEDGE_ARCHITECTURE.md):

* Records are never chunked like a PDF. A rule keeps its statement, its "why" and
  its failure signal together; an experiment card always says what the dataset
  predicts and what the stage does, so a reader (human, RAG prompt or SFT
  example) never sees a bare "EXP-023 found X".
* Retrieval is **filter first, rank second**: exact metadata filters (type,
  dataset, stage, task type, category) narrow the candidates for free, then a
  small BM25 ranker orders what survives. Swap BM25 for embeddings later without
  changing the record contract.
* Pure standard library and deterministic, so CI can verify that the committed
  ``knowledge/rag/records.jsonl`` is current.

CLI::

    python -m dclab_rnd.evidence_index build          # write knowledge/rag/records.jsonl
    python -m dclab_rnd.evidence_index check          # fail if the committed index is stale
    python -m dclab_rnd.evidence_index search "is call duration safe" --dataset bank_marketing
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = Path("knowledge") / "rag" / "records.jsonl"
SCHEMA_VERSION = 1

STAGE_CARDS: dict[str, dict[str, str]] = {
    "data_understanding": {
        "title": "Data understanding",
        "workflow": "WF-04",
        "method": (
            "Profile the TRAINING split only: schema and types, target balance, missingness, "
            "duplicates, identifier-like columns and a random-split drift smoke check. The goal is "
            "to surface reliability and production-availability risks a leaderboard would hide; "
            "associations found here are hypotheses, never promoted features."
        ),
    },
    "leakage_audit": {
        "title": "Leakage audit",
        "workflow": "WF-05",
        "method": (
            "Decide which columns would NOT exist at the declared prediction moment (semantic "
            "availability), add heuristic review candidates (very high single-feature predictive "
            "power, identifier-like uniqueness, suspicious names), and measure a safe-vs-unsafe "
            "ablation on identical folds. A large score gap shows how much a leak would inflate the "
            "result; only decision-time semantics can confirm the leak."
        ),
    },
    "feature_engineering": {
        "title": "Feature engineering ablation",
        "workflow": "WF-06",
        "method": (
            "Climb a feature ladder (raw, single transforms, ratios, bounded interactions, "
            "selection), fitting every transform inside each training fold. Choose the SMALLEST "
            "feature matrix whose mean CV score is within a small tolerance of the best, because "
            "extra features that do not clearly help dilute signal and add production surface."
        ),
    },
    "model_selection": {
        "title": "Algorithm screening",
        "workflow": "WF-07",
        "method": (
            "Screen several model families (regularized linear, bagged trees, boosting) on the "
            "selected features and IDENTICAL folds. Rank by mean score penalized by fold-to-fold "
            "spread, then runtime. A CV leader is a candidate, not a universal winner."
        ),
    },
    "optimization_reliability": {
        "title": "Optimization, reliability and final holdout",
        "workflow": "WF-08/WF-09",
        "method": (
            "Try a small explicit parameter search for the selected model on training CV and keep "
            "it only if it beats the baseline by a declared margin. Then evaluate ONCE on the "
            "locked holdout with a bootstrap interval, calibration (Brier, ECE) and threshold "
            "metrics. The holdout is consumed: no further tuning against it."
        ),
    },
}

STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how i in is it its of on or our "
    "should that the their this to was we what when which with you your into than then them "
    "they not no use used using".split()
)


# --------------------------------------------------------------------------- helpers


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, bool) or value is None:
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.{digits}f}"
    return str(value)


def _metric_mean(row: dict[str, Any], name: str) -> float | None:
    metric = (row.get("metrics") or {}).get(name)
    if isinstance(metric, dict):
        return metric.get("mean")
    if isinstance(metric, (int, float)):
        return float(metric)
    return None


def _metric_std(row: dict[str, Any], name: str) -> float | None:
    metric = (row.get("metrics") or {}).get(name)
    return metric.get("std") if isinstance(metric, dict) else None


def _humanize(key: str) -> str:
    return key.replace("_", " ")


# --------------------------------------------------------------------------- dataset cards


def _external_descriptions(root: Path) -> dict[str, dict[str, str]]:
    """Read DatasetSpec(...) literals from general_pipeline/external_catalog.py without importing it."""
    path = root / "general_pipeline" / "external_catalog.py"
    if not path.exists():
        return {}
    found: dict[str, dict[str, str]] = {}
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "DatasetSpec":
            values = {
                kw.arg: kw.value.value
                for kw in node.keywords
                if kw.arg and isinstance(kw.value, ast.Constant)
            }
            if "key" in values:
                found[values["key"]] = values
    return found


def _agentic_policies() -> dict[str, dict[str, Any]]:
    try:
        from dclab_rnd.agentic.catalog import DATASETS  # stdlib-only module
    except Exception:  # pragma: no cover - defensive
        return {}
    return DATASETS


def _expansion_cards() -> dict[str, dict[str, Any]]:
    """Dataset cards for the task-type expansion campaign, if its adapters are present."""
    try:
        from dclab_rnd.expansion import datasets as expansion  # noqa: WPS433
    except Exception:
        return {}
    getter = getattr(expansion, "dataset_cards", None)
    if callable(getter):
        try:
            return {card["key"]: card for card in getter()}
        except Exception:
            return {}
    return {}


PROJECT_CARDS = {
    "hyperack": {
        "name": "HyperAck delivery acceptance (DCLab project data)",
        "description": "Predict whether a delivery order is accepted (hyper_ack 0/1) at order time, before dispatch.",
        "task_type": "binary",
        "rows": 11118,
    },
    "telco_churn": {
        "name": "Telco customer churn (IBM sample)",
        "description": "Predict whether a telecom customer churns (Churn Yes/No) from account, service and billing fields.",
        "task_type": "binary",
        "rows": 7043,
    },
}


def dataset_cards(root: Path) -> dict[str, dict[str, Any]]:
    descriptions = _external_descriptions(root)
    policies = _agentic_policies()
    cards: dict[str, dict[str, Any]] = {}
    for meta_path in sorted((root / "external_data").glob("*/meta.json")):
        meta = _read_json(meta_path)
        key = meta.get("key", meta_path.parent.name)
        policy = policies.get(key, {})
        cards[key] = {
            "key": key,
            "name": meta.get("name", key),
            "description": descriptions.get(key, {}).get("description", ""),
            "task_type": "binary",
            "rows": meta.get("n_rows"),
            "features": meta.get("n_features"),
            "positive_rate": meta.get("pos_rate"),
            "source": meta.get("url", ""),
            "decision_time_contract": policy.get("decision", ""),
            "blocked_features": list(policy.get("blocked", [])),
        }
    for key, card in PROJECT_CARDS.items():
        policy = policies.get(key, {})
        cards[key] = {
            "key": key,
            **card,
            "decision_time_contract": policy.get("decision", ""),
            "blocked_features": list(policy.get("blocked", [])),
        }
    for key, card in _expansion_cards().items():
        cards[key] = {**cards.get(key, {}), **card, "key": key}
    return cards


def dataset_card_text(card: dict[str, Any]) -> str:
    parts = [f"Dataset `{card['key']}`: {card.get('name', card['key'])}."]
    if card.get("description"):
        parts.append(card["description"].rstrip(".") + ".")
    facts = []
    if card.get("task_type"):
        facts.append(f"task type {card['task_type']}")
    if card.get("rows"):
        facts.append(f"{_fmt(card['rows'])} rows")
    if card.get("features"):
        facts.append(f"{_fmt(card['features'])} features")
    if card.get("positive_rate") is not None:
        facts.append(f"positive rate {card['positive_rate']:.1%}")
    if facts:
        parts.append("Facts: " + ", ".join(facts) + ".")
    if card.get("decision_time_contract"):
        parts.append("Decision-time contract: " + card["decision_time_contract"])
    if card.get("blocked_features"):
        parts.append("Blocked (post-outcome or forbidden) columns: " + ", ".join(card["blocked_features"]) + ".")
    return " ".join(parts)


# --------------------------------------------------------------------------- setup summaries


def _summarize_data_understanding(ev: dict[str, Any]) -> str:
    bits = []
    for key in ("source_rows", "analyzed_rows", "train_rows", "holdout_rows", "feature_count"):
        if key in ev:
            bits.append(f"{_humanize(key)} {_fmt(ev[key])}")
    for key in ("positive_rate_train", "missing_cell_rate_train", "duplicate_row_rate_train"):
        if key in ev:
            bits.append(f"{_humanize(key)} {ev[key]:.3f}")
    risky = [
        f"{p['feature']} ({', '.join(p['production_risks'])})"
        for p in ev.get("feature_profiles", [])
        if p.get("production_risks")
    ]
    text = "Profiled " + "; ".join(bits) + "."
    text += " Features with production risks: " + (", ".join(risky) if risky else "none flagged") + "."
    return text


def _summarize_leakage(ev: dict[str, Any]) -> str:
    parts = []
    declared = ev.get("declared_leakage_features") or []
    parts.append("Declared decision-time exclusions: " + (", ".join(declared) if declared else "none") + ".")
    if ev.get("policy_rationale"):
        parts.append("Rationale: " + ev["policy_rationale"])
    candidates = ev.get("heuristic_review_candidates") or []
    if candidates:
        rendered = []
        for c in candidates[:6]:
            extra = []
            if c.get("univariate_auc") is not None:
                extra.append(f"single-feature AUC {c['univariate_auc']:.3f}")
            if c.get("reasons"):
                extra.append("reasons: " + ", ".join(c["reasons"]))
            rendered.append(f"{c.get('feature')} ({'; '.join(extra)})")
        parts.append("Heuristic review candidates: " + "; ".join(rendered) + ".")
    else:
        parts.append("Heuristic review candidates: none.")
    comparison = ev.get("safe_vs_unsafe_training_cv") or {}
    if comparison:
        safe = _metric_mean(comparison.get("safe", {}), "roc_auc")
        unsafe = _metric_mean(comparison.get("unsafe", {}), "roc_auc")
        lift = comparison.get("apparent_auc_lift")
        if safe is not None and unsafe is not None:
            parts.append(
                f"Safe-vs-unsafe training CV ROC-AUC: safe {safe:.4f}, with blocked columns {unsafe:.4f}, apparent lift {lift:+.4f}."
            )
    if "detector_canary_passed" in ev:
        parts.append(f"Synthetic exact-target canary detected: {ev['detector_canary_passed']}.")
    return " ".join(parts)


def _ladder(rows: list[dict[str, Any]], label_key: str) -> str:
    rendered = []
    for row in rows:
        mean = _metric_mean(row, "roc_auc")
        std = _metric_std(row, "roc_auc")
        label = row.get(label_key)
        count = row.get("feature_count_mean")
        text = f"{label}"
        if count is not None:
            text += f" ({count:.0f} features"
            text += f", ROC-AUC {mean:.4f}±{std:.4f})" if mean is not None and std is not None else ")"
        elif mean is not None:
            text += f" (ROC-AUC {mean:.4f}" + (f"±{std:.4f})" if std is not None else ")")
        if row.get("elapsed_seconds") is not None and label_key == "model":
            text = text.rstrip(")") + f", {row['elapsed_seconds']:.1f}s)"
        rendered.append(text)
    return "; ".join(rendered)


def _summarize_feature_engineering(ev: dict[str, Any]) -> str:
    rows = ev.get("stage_results") or []
    model = rows[0].get("model") if rows else "?"
    folds = rows[0].get("folds") if rows else "?"
    text = f"{len(rows)} feature recipes compared with {model}, {folds}-fold training CV: {_ladder(rows, 'stage')}."
    if ev.get("selection_rule"):
        text += " Selection rule: " + ev["selection_rule"]
    if ev.get("selected_stage"):
        text += f" Selected: {ev['selected_stage']}"
        if ev.get("selected_vs_raw_auc") is not None:
            text += f" ({ev['selected_vs_raw_auc']:+.4f} ROC-AUC vs raw)."
    return text


def _summarize_model_selection(ev: dict[str, Any]) -> str:
    rows = ev.get("model_results") or []
    text = f"{len(rows)} model families screened on the '{ev.get('feature_stage', '?')}' features: {_ladder(rows, 'model')}."
    if ev.get("selection_rule"):
        text += " Selection rule: " + ev["selection_rule"]
    if ev.get("selected_model"):
        text += f" Selected: {ev['selected_model']}."
    return text


def _summarize_optimization(ev: dict[str, Any]) -> str:
    parts = [f"Model {ev.get('model', '?')} on '{ev.get('feature_stage', '?')}' features."]
    configs = ev.get("configuration_results") or []
    if configs:
        parts.append(
            "Configurations on training CV: "
            + "; ".join(
                f"{c.get('optimization')} (ROC-AUC {(_metric_mean(c, 'roc_auc') or 0):.4f})" for c in configs
            )
            + "."
        )
    if ev.get("selected_optimization"):
        parts.append(f"Kept: {ev['selected_optimization']}")
    if ev.get("optimized_vs_baseline_cv_auc") is not None:
        parts.append(f"(CV gain vs baseline {ev['optimized_vs_baseline_cv_auc']:+.4f}).")
    hold = ev.get("holdout_metrics") or {}
    if hold:
        keys = [k for k in ("roc_auc", "avg_precision", "precision", "recall", "f1", "brier", "ece_10") if k in hold]
        parts.append("Holdout (used once): " + ", ".join(f"{k} {hold[k]:.4f}" for k in keys) + ".")
    ci = ev.get("holdout_roc_auc_bootstrap_ci") or {}
    if ci:
        parts.append(f"Bootstrap 95% interval for holdout ROC-AUC: {ci.get('low', 0):.4f}–{ci.get('high', 0):.4f}.")
    top = ev.get("top_final_features") or []
    if top:
        names = [t.get("feature") for t in top[:5] if isinstance(t, dict)]
        parts.append("Top final features: " + ", ".join(n for n in names if n) + ".")
    return " ".join(parts)


def _summarize_generic(ev: dict[str, Any]) -> str:
    bits = []
    for key in sorted(ev):
        value = ev[key]
        if isinstance(value, (int, float, bool)) and not isinstance(value, bool) or isinstance(value, bool):
            bits.append(f"{_humanize(key)} {_fmt(value)}")
        elif isinstance(value, str) and len(value) <= 240:
            bits.append(f"{_humanize(key)}: {value}")
        elif isinstance(value, list) and value and all(isinstance(v, str) for v in value) and len(value) <= 8:
            bits.append(f"{_humanize(key)}: {', '.join(value)}")
        elif isinstance(value, dict) and value and all(isinstance(v, (int, float)) for v in value.values()):
            inner = ", ".join(f"{k} {_fmt(v)}" for k, v in sorted(value.items())[:8])
            bits.append(f"{_humanize(key)} ({inner})")
    return "; ".join(bits[:24]) + ("." if bits else "")


SUMMARIZERS = {
    "data_understanding": _summarize_data_understanding,
    "leakage_audit": _summarize_leakage,
    "feature_engineering": _summarize_feature_engineering,
    "model_selection": _summarize_model_selection,
    "optimization_reliability": _summarize_optimization,
}


def setup_summary(result: dict[str, Any]) -> str:
    """Human-readable description of what a run actually compared, from its evidence block."""
    ev = result.get("evidence") or {}
    custom = result.get("setup_summary")
    if isinstance(custom, str) and custom.strip():
        return custom.strip()
    summarizer = SUMMARIZERS.get(result.get("kind", ""))
    if summarizer is not None:
        try:
            return summarizer(ev)
        except (KeyError, TypeError, ValueError, AttributeError):
            pass
    return _summarize_generic(ev)


# --------------------------------------------------------------------------- records


@dataclass
class Record:
    record_id: str
    type: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    citations: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "type": self.type,
            "title": self.title,
            "text": self.text,
            "metadata": self.metadata,
            "citations": self.citations,
        }


def _rule_records(root: Path) -> list[Record]:
    out = []
    for rule in _read_jsonl(root / "knowledge" / "model_building_rules.jsonl"):
        text = (
            f"Rule {rule['rule_id']} ({rule['category']}): {rule['statement']} "
            f"Why: {rule['why']} Failure signal: {rule.get('failure_signal', '')} "
            f"Next test: {rule.get('next_test', '')} Confidence: {rule.get('confidence', '')}."
        )
        out.append(
            Record(
                record_id=rule["rule_id"],
                type="rule",
                title=rule["statement"],
                text=text,
                metadata={"category": rule["category"], "confidence": rule.get("confidence", "")},
                citations=["knowledge/model_building_rules.jsonl", *rule.get("evidence", [])],
            )
        )
    return out


def _workflow_records(root: Path) -> list[Record]:
    path = root / "knowledge" / "workflow_blocks.json"
    if not path.exists():
        return []
    stage_for_block = {card["workflow"]: kind for kind, card in STAGE_CARDS.items()}
    out = []
    for block in _read_json(path):
        stages = [kind for wf, kind in stage_for_block.items() if block["id"] in wf.split("/")]
        out.append(
            Record(
                record_id=block["id"],
                type="workflow",
                title=block["name"],
                text=f"Workflow block {block['id']} — {block['name']}: " + " → ".join(block["flow"]) + ".",
                metadata={"stage": stages[0] if stages else None},
                citations=["knowledge/workflow_blocks.json"],
            )
        )
    return out


def _dataset_records(cards: dict[str, dict[str, Any]]) -> list[Record]:
    return [
        Record(
            record_id=f"DATASET-{key}",
            type="dataset",
            title=card.get("name", key),
            text=dataset_card_text(card),
            metadata={"dataset": key, "task_type": card.get("task_type")},
            citations=[card["source"]] if card.get("source") else [],
        )
        for key, card in sorted(cards.items())
    ]


def experiment_files(root: Path) -> list[Path]:
    return sorted(root.glob("campaigns/*/results/EXP-*.json"))


def experiment_context(result: dict[str, Any], cards: dict[str, dict[str, Any]]) -> dict[str, str]:
    """The three self-contained context blocks every experiment-grounded example carries."""
    dataset = result.get("dataset", "")
    card = cards.get(dataset, {"key": dataset, "name": dataset})
    kind = result.get("kind", "")
    stage = STAGE_CARDS.get(kind, {"title": _humanize(kind), "workflow": "", "method": ""})
    stage_text = f"Stage: {stage['title']} ({stage['workflow']}). {stage['method']}"
    if result.get("question"):
        stage_text += f" Question: {result['question']}"
    if result.get("hypothesis"):
        stage_text += f" Hypothesis: {result['hypothesis']}"
    return {
        "dataset_card": dataset_card_text(card),
        "stage_card": stage_text,
        "setup_summary": setup_summary(result),
    }


def _review(result: dict[str, Any]) -> dict[str, Any]:
    review = (result.get("llm_review") or {}).get("review")
    return review if isinstance(review, dict) else {}


def _experiment_records(root: Path, cards: dict[str, dict[str, Any]]) -> list[Record]:
    out = []
    for path in experiment_files(root):
        result = _read_json(path)
        context = experiment_context(result, cards)
        claims = result.get("claims") or []
        findings = " ".join(f"[{c.get('kind')}] {c.get('statement')}" for c in claims)
        limitations = sorted({lim for c in claims for lim in c.get("limitations", [])})
        review = _review(result)
        critic = ""
        if review:
            challenges = "; ".join(
                f"{c.get('severity', '?')}: {c.get('issue', '')}" for c in review.get("challenges_to_claims", [])[:4]
            )
            critic = f" Critic review ({review.get('model', 'LLM')}): decision — {review.get('decision', '')}"
            if challenges:
                critic += f" Challenges — {challenges}"
        text = (
            f"Experiment {result.get('experiment_id')} ({result.get('campaign_id')}). "
            f"{context['dataset_card']} {context['stage_card']} Setup: {context['setup_summary']} "
            f"Findings: {findings or 'none recorded.'}"
            + (f" Limitations: {' '.join(limitations)}" if limitations else "")
            + critic
        )
        out.append(
            Record(
                record_id=result.get("experiment_id", path.stem),
                type="experiment",
                title=f"{result.get('experiment_id')} · {result.get('dataset')} · {STAGE_CARDS.get(result.get('kind', ''), {}).get('title', result.get('kind'))}",
                text=text,
                metadata={
                    "dataset": result.get("dataset"),
                    "stage": result.get("kind"),
                    "campaign": result.get("campaign_id"),
                    "task_type": result.get("task_type") or cards.get(result.get("dataset", ""), {}).get("task_type"),
                    "has_critic_review": bool(review),
                },
                citations=[path.relative_to(root).as_posix()],
            )
        )
    return out


def _leakage_records(root: Path, cards: dict[str, dict[str, Any]]) -> list[Record]:
    """Stack-Overflow-style precedents: a concrete leakage question, its evidence and resolution."""
    out = []
    for path in experiment_files(root):
        result = _read_json(path)
        if result.get("kind") != "leakage_audit":
            continue
        ev = result.get("evidence") or {}
        declared = ev.get("declared_leakage_features") or []
        if not declared:
            continue
        dataset = result.get("dataset", "")
        card = cards.get(dataset, {"key": dataset})
        lift = (ev.get("safe_vs_unsafe_training_cv") or {}).get("apparent_auc_lift")
        if lift is None:
            lift = ev.get("apparent_lift")
        review = _review(result)
        columns = ", ".join(f"`{c}`" for c in declared)
        text = (
            f"Precedent: Is it safe to use {columns} as model input for {card.get('name', dataset)}? "
            f"Context: {card.get('description', '')} "
            f"Evidence: {setup_summary(result)} "
            f"Resolution: exclude {columns} from deployable models because "
            f"{(ev.get('policy_rationale') or 'it is not available at the declared prediction moment').rstrip('. ')}. "
            + (f"Including it inflated the validation score by {lift:+.4f}. " if isinstance(lift, (int, float)) else "")
            + (f"Critic caveat: {review.get('decision')}" if review.get("decision") else "")
        )
        out.append(
            Record(
                record_id=f"LEAK-{dataset}",
                type="leakage_precedent",
                title=f"Leakage precedent: {', '.join(declared)} in {dataset}",
                text=text,
                metadata={"dataset": dataset, "stage": "leakage_audit", "columns": declared,
                          "task_type": result.get("task_type") or card.get("task_type"),
                          "apparent_lift": round(lift, 6) if isinstance(lift, (int, float)) else None,
                          "rationale": ev.get("policy_rationale") or ""},
                citations=[path.relative_to(root).as_posix()],
            )
        )
    pack_path = root / "knowledge" / "master_evidence_pack.json"
    if pack_path.exists():
        hyper = _read_json(pack_path).get("hyperack") or {}
        if hyper:
            out.append(
                Record(
                    record_id="LEAK-hyperack",
                    type="leakage_precedent",
                    title="Leakage precedent: final fares in HyperAck",
                    text=(
                        "Precedent: Can final_customer_fare and final_biker_fare be used to predict HyperAck order "
                        "acceptance? Context: HyperAck predicts at order time whether a delivery order is accepted. "
                        f"Evidence: the best model WITH final fares reached ROC-AUC {hyper['unsafe_auc']:.4f}; the best "
                        f"leakage-safe ensemble reached {hyper['safe_champion_auc']:.4f} (apparent lift "
                        f"{hyper['unsafe_lift']:+.4f}). Resolution: final fares are only known after the dispatch outcome, "
                        "so they are post-outcome leakage and are blocked in every deployable or agentic run."
                    ),
                    metadata={"dataset": "hyperack", "stage": "leakage_audit", "columns": hyper.get("blocked_features", []),
                              "task_type": "binary", "apparent_lift": round(hyper["unsafe_lift"], 6),
                              "rationale": "Final fares are only known after the dispatch outcome."},
                    citations=[hyper.get("safe_source", ""), hyper.get("unsafe_source", "")],
                )
            )
    return out


def _pitfall_records(root: Path) -> list[Record]:
    """Measured cost of common notebook mistakes (wrong way vs right way on identical data)."""
    out = []
    for path in sorted(root.glob("campaigns/*/results/PIT-*.json")):
        result = _read_json(path)
        claims = " ".join(c.get("statement", "") for c in result.get("claims", []))
        limits = sorted({lim for c in result.get("claims", []) for lim in c.get("limitations", [])})
        out.append(
            Record(
                record_id=result["experiment_id"],
                type="pitfall",
                title=f"{result['experiment_id']} · measured cost of: {result['pitfall'].replace('_', ' ')}",
                text=(
                    f"Pitfall experiment {result['experiment_id']}: {result['question']} "
                    f"Setup: {result.get('setup_summary', '')} Findings: {claims}"
                    + (f" Limitations: {' '.join(limits)}" if limits else "")
                ),
                metadata={"pitfall": result["pitfall"], "datasets": result.get("datasets", []),
                          "campaign": result.get("campaign_id")},
                citations=[path.relative_to(root).as_posix()],
            )
        )
    return out


def _finding_records(root: Path) -> list[Record]:
    path = root / "knowledge" / "master_evidence_pack.json"
    if not path.exists():
        return []
    pack = _read_json(path)
    out = []
    models = pack.get("cross_dataset_models") or []
    if models:
        ranking = "; ".join(
            f"{m['model_family']} mean rank {m['mean_rank']:.2f}, wins {m['wins']}/{m['dataset_count']}, "
            f"mean ROC-AUC {m['mean_roc_auc']:.4f}"
            for m in models[:6]
        )
        out.append(
            Record(
                record_id="FINDING-model-families",
                type="finding",
                title="No universal best algorithm across datasets",
                text=(
                    "Cross-dataset algorithm evidence from the historical registry: " + ranking + ". "
                    "Winners differ by dataset, so screen several families on identical folds instead of "
                    "defaulting to one."
                ),
                metadata={"stage": "model_selection"},
                citations=["knowledge/master_evidence_pack.json", "knowledge/evidence.json"],
            )
        )
    gaps = pack.get("registry_leakage_gaps") or []
    if gaps:
        out.append(
            Record(
                record_id="FINDING-leakage-gaps",
                type="finding",
                title="Leakage produces the largest apparent score jumps in the registry",
                text="Safe-vs-unsafe registry gaps: "
                + "; ".join(
                    f"{g['dataset']} safe {g['safe_roc_auc']:.4f} vs unsafe {g['unsafe_roc_auc']:.4f} "
                    f"(lift {g['apparent_leakage_lift']:+.4f})"
                    for g in gaps
                )
                + ". A sudden large improvement is a reason to audit for leakage before celebrating.",
                metadata={"stage": "leakage_audit"},
                citations=["knowledge/master_evidence_pack.json"],
            )
        )
    churn = (pack.get("churn") or {}).get("best") or {}
    if churn:
        metrics = churn.get("metrics", {})
        out.append(
            Record(
                record_id="FINDING-churn-linear",
                type="finding",
                title="A regularized linear model won the Telco churn campaign",
                text=(
                    f"On Telco churn, {churn.get('title')} ({churn.get('model')}) led the 15-experiment development "
                    f"campaign with ROC-AUC {metrics.get('roc_auc', 0):.4f} and average precision "
                    f"{metrics.get('average_precision', 0):.4f}; boosting did not beat it. Simple models deserve a "
                    "fair screen, and adaptive development CV is not independent confirmation."
                ),
                metadata={"dataset": "telco_churn", "stage": "model_selection"},
                citations=["research/churn-prediction/churn_exp/CHURN_BENCHMARK.md", "knowledge/master_evidence_pack.json"],
            )
        )
    return out


def build_records(root: Path = ROOT) -> list[dict[str, Any]]:
    root = root.resolve()
    cards = dataset_cards(root)
    records: list[Record] = []
    records += _rule_records(root)
    records += _workflow_records(root)
    records += _dataset_records(cards)
    records += _experiment_records(root, cards)
    records += _leakage_records(root, cards)
    records += _finding_records(root)
    records += _pitfall_records(root)
    seen: set[str] = set()
    unique = []
    for record in records:
        if record.record_id in seen:
            continue
        seen.add(record.record_id)
        unique.append(record.to_json())
    return unique


def render_index(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)


def write_index(root: Path = ROOT, *, check: bool = False) -> bool:
    target = root / INDEX_PATH
    content = render_index(build_records(root))
    current = target.read_text(encoding="utf-8") if target.exists() else None
    if check:
        return current == content
    target.parent.mkdir(parents=True, exist_ok=True)
    if current != content:
        target.write_text(content, encoding="utf-8")
    return True


# --------------------------------------------------------------------------- retrieval


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower().replace("_", " ").replace("-", " "))
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class EvidenceIndex:
    """Filter-then-rank retrieval (BM25) over evidence records."""

    def __init__(self, records: Iterable[dict[str, Any]], k1: float = 1.4, b: float = 0.75):
        self.records = list(records)
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(r["title"] + " " + r["text"])) for r in self.records]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    @classmethod
    def load(cls, root: Path = ROOT) -> "EvidenceIndex":
        path = root / INDEX_PATH
        records = _read_jsonl(path) if path.exists() else build_records(root)
        return cls(records)

    def get(self, record_id: str) -> dict[str, Any] | None:
        return next((r for r in self.records if r["record_id"] == record_id), None)

    @staticmethod
    def _matches(record: dict[str, Any], filters: dict[str, Any]) -> bool:
        for key, wanted in filters.items():
            if wanted in (None, "", []):
                continue
            value = record.get(key) if key == "type" else record["metadata"].get(key)
            wanted_set = set(wanted) if isinstance(wanted, (list, tuple, set)) else {wanted}
            if value is None:
                return False
            if isinstance(value, list):
                if not wanted_set.intersection(value):
                    return False
            elif value not in wanted_set:
                return False
        return True

    def search(self, query: str, *, k: int = 5, **filters: Any) -> list[dict[str, Any]]:
        terms = tokenize(query)
        scored = []
        for i, record in enumerate(self.records):
            if not self._matches(record, filters):
                continue
            doc, length = self.docs[i], self.lengths[i]
            score = 0.0
            for term in terms:
                freq = doc.get(term, 0)
                if freq:
                    norm = freq * (self.k1 + 1) / (freq + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1)))
                    score += self.idf.get(term, 0.0) * norm
            if score > 0 or not terms:
                scored.append((score, record["record_id"], record))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [{**record, "score": round(score, 4)} for score, _, record in scored[:k]]


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.evidence_index", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="write knowledge/rag/records.jsonl")
    sub.add_parser("check", help="exit 1 when the committed index is stale")
    search = sub.add_parser("search", help="filter-then-rank search")
    search.add_argument("query")
    search.add_argument("--type", dest="record_type")
    search.add_argument("--dataset")
    search.add_argument("--stage")
    search.add_argument("--task-type")
    search.add_argument("--category")
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "build":
        write_index(ROOT)
        count = len(_read_jsonl(ROOT / INDEX_PATH))
        print(f"Wrote {INDEX_PATH} ({count} records)")
        return 0
    if args.command == "check":
        if write_index(ROOT, check=True):
            print("Evidence index is current.")
            return 0
        print(f"Stale: {INDEX_PATH}. Run: python -m dclab_rnd.evidence_index build")
        return 1
    index = EvidenceIndex.load(ROOT)
    hits = index.search(
        args.query,
        k=args.k,
        type=args.record_type,
        dataset=args.dataset,
        stage=args.stage,
        task_type=args.task_type,
        category=args.category,
    )
    if args.json:
        print(json.dumps(hits, indent=2, ensure_ascii=False))
        return 0
    for hit in hits:
        print(f"[{hit['score']:.3f}] {hit['record_id']} ({hit['type']}) — {hit['title']}")
        print("   " + hit["text"][:400] + ("…" if len(hit["text"]) > 400 else ""))
        print("   cites: " + ", ".join(c for c in hit["citations"] if c))
    return 0


if __name__ == "__main__":
    sys.exit(main())
