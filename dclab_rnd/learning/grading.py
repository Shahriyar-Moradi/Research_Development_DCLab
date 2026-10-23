"""A solve of a challenge pack and its grade (H0).

``solve(folder, statements)`` runs the studio's engine on a pack's public folder as a person would bring it: the
table, the request's outcome, task and moment, the proposal accepted, and ``statements`` on top (none for today's
DCLab; the key's reference statements to prove a variant solvable). ``observe(run)`` reads the finished run into
:class:`dclab_core.challenge_pack.Observed`; :func:`dclab_core.challenge_pack.grade` does the rest.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from dclab_core.challenge_pack import ChallengeKey, Grade, Observed, grade

from .packs import read_public


def solve(folder: Path, statements: dict[str, Any] | None = None) -> Any:
    """The engine on a pack's public folder (``kinds.acceptance.engine_run``: a temporary project, the quick profile, no
    model asked anything). Returns its ``Run``."""
    from dclab_rnd.kinds.acceptance import Case, engine_run

    pack, table = read_public(folder)
    case = Case(table=table, target=pack.brief.target, task=pack.brief.task, moment=pack.brief.moment, solution=dict(statements or {}))
    return engine_run(case, replica=pack.brief.target)


def _baseline(evidence: dict[str, Any], metric: str | None, higher: bool) -> float | None:
    """The best simple baseline's holdout score on the primary metric (a random ranking's for average precision)."""
    values = []
    for scores in (evidence.get("baselines") or {}).values():
        if not isinstance(scores, dict) or metric is None:
            continue
        value = scores.get(metric, scores.get(f"{metric}_random_ranking"))
        if isinstance(value, (int, float)):
            values.append(float(value))
    if not values:
        return None
    return max(values) if higher else min(values)


def observe(run: Any) -> Observed:
    """What a finished run did, as the grader reads it: the columns kept out (blocked or identifiers, and not modelled as
    text), the group and split, the final model's top features, the validation and holdout scores, the baseline's."""
    from dclab_rnd.expansion.runner import higher_is_better

    data, final = run.evidence("data"), run.evidence("final")
    roles = data.get("column_roles") or {}
    modelled_text = set((data.get("text_stats") or {}).keys())
    kept_out = sorted(c for c, r in roles.items() if r in ("blocked", "identifier") and c not in modelled_text)
    metric = (run.records.get("final") or {}).get("primary_metric")
    try:
        higher = higher_is_better(metric) if metric else True
    except KeyError:
        higher = True
    holdout = (final.get("holdout_metrics") or {}).get(metric) if metric else None
    ci = final.get("holdout_primary_metric_ci") or {}
    interval = [float(ci["low"]), float(ci["high"])] if isinstance(ci.get("low"), (int, float)) and isinstance(ci.get("high"), (int, float)) else None
    models = run.evidence("models")
    std = models.get("selected_metric_std")
    folds = re.search(r"\((\d+)\)|(\d+)-fold|(\d+) folds", str(models.get("cv_protocol") or data.get("cv_protocol") or ""))
    return Observed(kept_out=kept_out, group_column=(run.solution or {}).get("group_column"), split=data.get("split_strategy"),
                    validation_std=float(std) if isinstance(std, (int, float)) else None,
                    folds=int(next(g for g in folds.groups() if g)) if folds else 5,
                    leaned_on=[str(f.get("feature")) for f in final.get("top_final_features") or [] if isinstance(f, dict) and f.get("feature")],
                    metric=metric, validation=final.get("cv_selected_metric_mean"),
                    holdout=float(holdout) if isinstance(holdout, (int, float)) else None, holdout_interval=interval,
                    gap=final.get("cv_to_holdout_gap"), baseline=_baseline(final, metric, higher), higher_is_better=higher)


def grade_run(key: ChallengeKey, run: Any) -> tuple[Grade, Observed]:
    observed = observe(run)
    return grade(key, observed), observed


__all__ = ["solve", "observe", "grade_run"]
