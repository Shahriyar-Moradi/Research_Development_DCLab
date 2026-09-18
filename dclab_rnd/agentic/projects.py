"""First-class research projects and their non-production evidence status."""
import json
from pathlib import Path
from .catalog import ROOT

PROJECTS = {
    "general": {
        "name": "Open tabular lab",
        "summary": "Ten public binary-classification datasets for broad workflow discovery.",
        "datasets": ["adult", "bank_marketing", "breast_cancer", "heart_disease", "credit_default", "german_credit", "mushroom", "spambase", "online_shoppers", "wine_quality"],
        "default_dataset": "bank_marketing",
        "status": "Ready for agentic experiments",
    },
    "hyperack": {
        "name": "HyperAck",
        "summary": "Delivery acceptance R&D with an existing 83-experiment lifecycle: 15 original, 15 leakage-safe and 53 optimized-safe tests.",
        "datasets": ["hyperack"],
        "default_dataset": "hyperack",
        "status": "83 historical experiments completed",
    },
    "telco_churn": {
        "name": "Telco Churn",
        "summary": "Customer churn R&D with a reproducible 15-experiment development-CV benchmark and an agentic follow-up loop.",
        "datasets": ["telco_churn"],
        "default_dataset": "telco_churn",
        "status": "15-experiment campaign available",
    },
}

def _read(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None

def historical_context(project, root=ROOT):
    """Curated context only: old reused holdouts are hypotheses, not live evidence IDs."""
    if project != "hyperack":
        return []
    unsafe = _read(root / "hyperack_exp/results/benchmark_summary.json") or {}
    safe = _read(root / "hyperack_exp/results/14_leakage_safe_features.json") or {}
    champion = _read(root / "optimized_safe_model/results/53_softvote_etbag_lgbmwinner_xgb.json") or {}
    return [
        {"record": "hyperack-history-original", "experiments": 15, "best_reported_auc": 0.979256, "status": "leakage-unsafe ceiling", "warning": "Final fare fields are post-outcome proxies and are forbidden in new research.", "source": "hyperack_exp/results/benchmark_summary.json", "raw_summary_present": bool(unsafe)},
        {"record": "hyperack-history-safe", "experiments": 15, "reported_auc": safe.get("metrics", {}).get("roc_auc", 0.938655), "status": "leakage-safe historical holdout", "warning": "The repeatedly consulted historical holdout is development evidence, not independent confirmation.", "source": "hyperack_exp/results/14_leakage_safe_features.json"},
        {"record": "hyperack-history-optimized-safe", "experiments": 53, "reported_auc": champion.get("metrics", {}).get("roc_auc", 0.945461), "reported_recall": champion.get("metrics", {}).get("recall", 0.77842), "status": "best optimized-safe historical result", "warning": "Use as a hypothesis/reference only; new agentic trials use separate group-CV evidence IDs.", "source": "optimized_safe_model/results/53_softvote_etbag_lgbmwinner_xgb.json"},
    ]

def project_catalog(root=ROOT):
    result = []
    for key, value in PROJECTS.items():
        item = {"key": key, **value}
        if key == "telco_churn":
            summary = _read(root / "churn_exp/results/summary.json")
            item["campaign"] = summary or {"completed": 0, "planned": 15}
            if summary: item["status"] = f"{summary['completed']}-experiment campaign completed"
        elif key == "hyperack":
            item["campaign"] = {"completed": 83, "planned": 83, "safe_champion_auc": 0.945461, "unsafe_ceiling_auc": 0.979256}
        result.append(item)
    return result
