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


LOWER_IS_BETTER = {"mae", "rmse", "mape", "log_loss", "brier", "mse"}


def _mean(row: dict[str, Any], metric: str = "roc_auc") -> float:
    return float(row["metrics"][metric]["mean"])


def _std(row: dict[str, Any], metric: str = "roc_auc") -> float:
    return float(row["metrics"][metric]["std"])


def _rule_params(result: dict[str, Any]) -> dict[str, Any]:
    """Read metric, direction, tolerance, std penalty and acceptance margin from the recorded rule text."""
    ev = result.get("evidence") or {}
    rule = ev.get("selection_rule") or ""
    metric = result.get("primary_metric") or "roc_auc"
    lower = metric in LOWER_IS_BETTER or "lower is better" in rule.lower()
    tol = re.search(r"within ([0-9.]+)(\s*%)?", rule)
    margin = re.search(r"at least ([0-9.]+)(\s*%)?", rule)
    penalty = re.search(r"([0-9.]+)\s*[×x*]\s*(fold )?std", rule)
    return {
        "metric": metric,
        "lower": lower,
        "tolerance": float(tol.group(1)) if tol else FE_TOLERANCE,
        "tolerance_relative": bool(tol and tol.group(2)),
        "margin": float(margin.group(1)) if margin else OPT_MIN_GAIN,
        "margin_relative": bool(margin and margin.group(2)),
        "penalty": float(penalty.group(1)) if penalty else STD_PENALTY,
    }


def _label(row: dict[str, Any], *keys: str) -> Any:
    return next((row[k] for k in keys if row.get(k) is not None), None)


def recompute(result: dict[str, Any]) -> dict[str, Any] | None:
    """Re-apply the stage's declared selection rule to its own evidence.

    The metric, its direction, the tolerance, the std penalty and the acceptance margin are read
    from the result (``primary_metric``) and its recorded rule text, so binary, multiclass,
    regression and text campaigns are all checked against their own rules.
    Returns ``None`` when the stage has no recomputable rule.
    """
    kind = result.get("kind")
    ev = result.get("evidence") or {}
    pr = _rule_params(result)
    metric, lower, k = pr["metric"], pr["lower"], pr["penalty"]
    sign = 1.0 if lower else -1.0  # sort key: smaller is better
    unit = "" if not (pr["tolerance_relative"]) else "% relative"
    try:
        if kind == "feature_engineering" and ev.get("stage_results"):
            rows = ev["stage_results"]
            means = [_mean(r, metric) for r in rows]
            best = min(means) if lower else max(means)
            tol = abs(best) * pr["tolerance"] / 100 if pr["tolerance_relative"] else pr["tolerance"]
            eligible = [r for r, m in zip(rows, means) if (m <= best + tol if lower else m >= best - tol)]
            chosen = _label(min(eligible, key=lambda r: (r["feature_count_mean"], r["elapsed_seconds"])), "stage", "recipe")
            return {
                "rule": f"smallest feature matrix within {pr['tolerance']}{unit} mean CV {metric} of the best stage",
                "metric": metric,
                "recomputed": chosen,
                "recorded": ev.get("selected_stage"),
                "consistent": chosen == ev.get("selected_stage"),
                "eligible": sorted(str(_label(r, "stage", "recipe")) for r in eligible),
            }
        if kind == "model_selection" and ev.get("model_results"):
            rows = ev["model_results"]
            score = lambda r: _mean(r, metric) + (k if lower else -k) * _std(r, metric)  # noqa: E731
            ranked = sorted(rows, key=lambda r: (sign * score(r), r["elapsed_seconds"]))
            return {
                "rule": f"rank by mean CV {metric} {'+' if lower else '-'} {k}×fold std ({'lower' if lower else 'higher'} is better), then runtime",
                "metric": metric,
                "recomputed": ranked[0]["model"],
                "recorded": ev.get("selected_model"),
                "consistent": ranked[0]["model"] == ev.get("selected_model"),
                "scores": {r["model"]: round(score(r), 6) for r in rows},
            }
        if kind == "optimization_reliability" and ev.get("configuration_results"):
            rows = ev["configuration_results"]
            baseline = rows[0]
            tuned = rows[1:] or [baseline]
            score = lambda r: _mean(r, metric) + (k if lower else -k) * _std(r, metric)  # noqa: E731
            best = min(tuned, key=lambda r: sign * score(r))
            gain = (_mean(baseline, metric) - _mean(best, metric)) if lower else (_mean(best, metric) - _mean(baseline, metric))
            needed = abs(_mean(baseline, metric)) * pr["margin"] / 100 if pr["margin_relative"] else pr["margin"]
            chosen = best if gain >= needed - 1e-12 else baseline
            return {
                "rule": f"keep the best tuned candidate only if mean CV {metric} improves by at least {pr['margin']}{'%' if pr['margin_relative'] else ''}",
                "metric": metric,
                "recomputed": chosen.get("optimization"),
                "recorded": ev.get("selected_optimization"),
                "consistent": chosen.get("optimization") == ev.get("selected_optimization"),
                "gain": round(gain, 6),
                "required": round(needed, 6),
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
