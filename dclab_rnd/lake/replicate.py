"""A tuning judged on a fresh draw (package 29.3): self-play's truth for ``tuning.accept``, never a holdout read again.

The final stage keeps a tuning when its validation gain beats a margin, choosing the best of several candidates on the
same folds: optimistic by construction, and nothing measured afterwards in the project can say whether the gain was real
(the holdout scores only the configuration kept; scoring the default there too would be a second use of it, the owner's
call). A replica can say it: its generator draws new rows of the same process (``kind.case(seed + OFFSET)``: the same
columns, traps and coefficients, other rows). Both configurations (the default and the best candidate, the parameters
the stage measured) are fitted on the fresh draw's training rows and scored on its own holdout rows, as the stage would;
a paired bootstrap of the difference (whole groups when the split keeps groups apart) says which is better beyond noise:

- the candidate clearly better: keeping it was right (a rule that kept it was right; one that declined it was wrong);
- the candidate clearly worse: keeping it was wrong;
- neither: nothing is labelled.

Only self-play's replicas have a generator, so only their solves carry this label; a real project's tuning stays
unlabelled. The original project's rows, its holdout and its records are not read beyond the stage's own record.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

OFFSET = 100_000  # a fresh draw's seed: far from every seed self-play or an evaluation reads
KIND = "fresh_draw"


def tuning_on_fresh_draw(kind: Any, seed: int, run: Any) -> Optional[tuple[bool, dict[str, Any]]]:
    """(keeping the tuning was right, evidence) for a solve whose final stage measured a candidate, or None."""
    from ..expansion import runner as rn
    from ..kinds import acceptance as A
    from ..studio import engine

    final = run.evidence("final")
    tuning = final.get("tuning_decision") or {}
    configs = {c.get("config_id"): c for c in final.get("configuration_results") or [] if isinstance(c, dict)}
    default = next((c for k, c in configs.items() if str(k).endswith("_baseline")), None)
    candidate = configs.get(tuning.get("best_candidate"))
    recipe, family = final.get("feature_recipe"), final.get("model")
    if default is None or candidate is None or not recipe or not family:
        return None  # no candidate measured (a kind with its own machine tunes nothing): nothing to replicate
    fresh = kind.case(seed + OFFSET)
    if fresh is None or list(fresh.table.columns) != list(run.case.table.columns):
        return None  # not the same process: no replicate
    with tempfile.TemporaryDirectory() as tmp, rn.fast_profile():
        store, project_id = A.build(fresh, Path(tmp))
        project = store.get(project_id)
        project["solution"] = dict(run.solution)  # the solve's own solution: the same columns kept out, the same split
        store.save(project)
        p = engine.prepare(store, store.get(project_id))
        bundle, spec = p.bundle, p.bundle.spec
        metric, higher = spec.primary_metric, spec.higher_is_better
        if metric != run.records["final"].get("primary_metric"):
            return None
        fits = [rn.fit_and_predict(bundle, recipe, family, c.get("model_params") or None, bundle.X_train, bundle.y_train, bundle.X_test)
                for c in (candidate, default)]
        y = bundle.y_test.to_numpy()
        metric_fn = rn.primary_metric_fn(bundle.task_type, metric, rn._n_classes(bundle))
        groups = bundle.groups("test") if spec.split_strategy in ("group", "time_group") else None
    sign = 1.0 if higher else -1.0
    paired = np.stack([np.asarray(fits[0]["predictions"]), np.asarray(fits[1]["predictions"])], axis=1)
    # outside the fast profile: its 40 draws are too few for an interval of a difference
    ci = rn.bootstrap_ci(y, paired, lambda yy, pp: sign * (metric_fn(yy, pp[:, 0]) - metric_fn(yy, pp[:, 1])), groups=groups, draws=400)
    low, high = ci.get("low"), ci.get("high")
    if not ci.get("draws") or low is None or high is None:
        return None
    evidence = {"kind": KIND, "value": round(float(ci["point"]), 6), "ref": f"replicate:{seed + OFFSET}:{candidate.get('config_id')}"[:200]}
    if low > 0:
        return True, evidence
    if high < 0:
        return False, evidence
    return None


__all__ = ["OFFSET", "KIND", "tuning_on_fresh_draw"]
