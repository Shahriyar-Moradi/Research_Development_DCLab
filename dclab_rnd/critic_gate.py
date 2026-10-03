"""Deterministic gate for LLM critic output before it becomes knowledge or training data.

The campaign's LLM critic is useful but not ground truth. Each campaign stage
records its own selection rule next to the numbers it was applied to, so the rule
can be *recomputed*. A critic challenge that disputes a selection the recomputation
confirms is marked ``contradicted`` and must not be used as a teaching target.

Known failure this catches: the feature ladder contains a recipe literally named
``selected`` (a feature-selection subset). Critics repeatedly confused it with the
recipe the rule *selected* and reported rule violations that do not exist.

CLI::

    python -m dclab_rnd.critic_gate            # per-experiment report
    python -m dclab_rnd.critic_gate --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

FE_TOLERANCE = 0.002
STD_PENALTY = 0.25
OPT_MIN_GAIN = 0.001

_RULE_DISPUTE = re.compile(
    r"(not within|would (instead|not) (choose|select)|not satisfied|rule is not|"
    r"is not the smallest|smaller than raw yet|much lower mean|not supported by the (stated )?(selection|ranking) rule|"
    r"appears pote|highest reported mean|inconsistent with its fold)",
    re.IGNORECASE,
)
_RULE_TOPIC = re.compile(r"(selection rule|ranking rule|within 0\.002|0\.25\s*[×x*]|selected stage|fold std)", re.IGNORECASE)


def _mean(row: dict[str, Any], metric: str = "roc_auc") -> float:
    return float(row["metrics"][metric]["mean"])


def _std(row: dict[str, Any], metric: str = "roc_auc") -> float:
    return float(row["metrics"][metric]["std"])


def recompute(result: dict[str, Any]) -> dict[str, Any] | None:
    """Re-apply the stage's declared selection rule to its own evidence.

    Returns ``None`` when the stage has no recomputable rule.
    """
    kind = result.get("kind")
    ev = result.get("evidence") or {}
    try:
        if kind == "feature_engineering" and ev.get("stage_results"):
            rows = ev["stage_results"]
            best = max(_mean(r) for r in rows)
            eligible = [r for r in rows if _mean(r) >= best - FE_TOLERANCE]
            chosen = min(eligible, key=lambda r: (r["feature_count_mean"], r["elapsed_seconds"]))["stage"]
            return {
                "rule": f"smallest feature matrix within {FE_TOLERANCE} mean CV ROC-AUC of the best stage",
                "recomputed": chosen,
                "recorded": ev.get("selected_stage"),
                "consistent": chosen == ev.get("selected_stage"),
                "eligible": sorted(r["stage"] for r in eligible),
            }
        if kind == "model_selection" and ev.get("model_results"):
            rows = ev["model_results"]
            ranked = sorted(rows, key=lambda r: (-(_mean(r) - STD_PENALTY * _std(r)), r["elapsed_seconds"]))
            return {
                "rule": f"rank by mean CV ROC-AUC minus {STD_PENALTY}×fold std, then runtime",
                "recomputed": ranked[0]["model"],
                "recorded": ev.get("selected_model"),
                "consistent": ranked[0]["model"] == ev.get("selected_model"),
                "scores": {r["model"]: round(_mean(r) - STD_PENALTY * _std(r), 6) for r in rows},
            }
        if kind == "optimization_reliability" and ev.get("configuration_results"):
            rows = ev["configuration_results"]
            baseline = rows[0]
            tuned = rows[1:] or [baseline]
            best = max(tuned, key=lambda r: _mean(r) - STD_PENALTY * _std(r))
            chosen = best if _mean(best) - _mean(baseline) >= OPT_MIN_GAIN else baseline
            return {
                "rule": f"keep the best tuned candidate only if mean CV ROC-AUC improves by at least {OPT_MIN_GAIN}",
                "recomputed": chosen.get("optimization"),
                "recorded": ev.get("selected_optimization"),
                "consistent": chosen.get("optimization") == ev.get("selected_optimization"),
            }
    except (KeyError, TypeError, ValueError):
        return None
    return None


def classify_challenge(challenge: dict[str, Any], check: dict[str, Any] | None) -> str:
    """``contradicted`` when the critic disputes a rule the recomputation confirms; else ``unverified``."""
    issue = challenge.get("issue", "")
    if check and check.get("consistent") and _RULE_TOPIC.search(issue) and _RULE_DISPUTE.search(issue):
        return "contradicted"
    return "unverified"


def gate_result(result: dict[str, Any]) -> dict[str, Any]:
    review = ((result.get("llm_review") or {}).get("review")) or {}
    check = recompute(result)
    challenges = []
    for challenge in review.get("challenges_to_claims", []) or []:
        challenges.append({**challenge, "gate": classify_challenge(challenge, check)})
    return {
        "experiment_id": result.get("experiment_id"),
        "dataset": result.get("dataset"),
        "kind": result.get("kind"),
        "rule_check": check,
        "challenges": challenges,
        "contradicted": sum(c["gate"] == "contradicted" for c in challenges),
        "kept": sum(c["gate"] != "contradicted" for c in challenges),
    }


def gate_campaigns(root: Path = ROOT) -> list[dict[str, Any]]:
    gated = []
    for path in sorted(root.glob("campaigns/*/results/EXP-*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        gated.append({**gate_result(result), "path": path.relative_to(root).as_posix()})
    return gated


def summary(gated: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [g["rule_check"] for g in gated if g["rule_check"]]
    return {
        "experiments": len(gated),
        "recomputable_rules": len(checks),
        "rule_inconsistencies": sum(not c["consistent"] for c in checks),
        "critic_challenges": sum(len(g["challenges"]) for g in gated),
        "contradicted_challenges": sum(g["contradicted"] for g in gated),
        "experiments_with_contradicted_challenges": sorted(g["experiment_id"] for g in gated if g["contradicted"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.critic_gate", description=__doc__.split("\n")[0])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    gated = gate_campaigns(ROOT)
    stats = summary(gated)
    if args.json:
        print(json.dumps({"summary": stats, "experiments": gated}, indent=2, ensure_ascii=False))
        return 0
    print(json.dumps(stats, indent=2))
    for g in gated:
        if g["contradicted"] or (g["rule_check"] and not g["rule_check"]["consistent"]):
            check = g["rule_check"] or {}
            print(f"\n{g['experiment_id']} {g['dataset']} {g['kind']}: recorded={check.get('recorded')} "
                  f"recomputed={check.get('recomputed')} consistent={check.get('consistent')}")
            for c in g["challenges"]:
                if c["gate"] == "contradicted":
                    print(f"  contradicted [{c.get('severity')}]: {c.get('issue', '')[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
