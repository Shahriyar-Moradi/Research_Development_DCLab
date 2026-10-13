"""Time to event (package 16.4): a duration, an event flag (rows not ended yet are censored) and a start moment. The kind
runs its own machine (``dclab_core.survival``), not the five tabular stages: Kaplan-Meier, Cox and discrete-time
models scored by C-index, the integrated Brier score and time-dependent AUC, with censored rows kept as censored and
columns counted during the follow-up kept out."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .base import ProblemKind


class TimeToEventFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    duration_column: str
    event_column: str
    start_column: str
    horizons: list[int] = Field(default_factory=lambda: [90, 180, 365], min_length=1, max_length=10)
    group_column: str | None = None


class TimeToEvent(ProblemKind):
    key = "time_to_event"
    title = "Time to event"
    description = "How long until something happens (a cancellation, a failure), when for some rows it has not happened yet."
    base_task = "regression"
    task_type = "survival"
    specificity = 15
    packs = ()
    # "how long until" alone is a delivery time (a regression, nothing censored): only with an ending that can still be to come
    words = (r"\b(time[- ]to[- ](?:event|cancel\w*|failure|churn)|survival (?:analysis|curves?|time)|censor\w*"
             r"|(?:how long|time) (?:until|before) (?:\w+ ){0,3}(?:cancel\w*|churn\w*|leav\w*|quit\w*|fail\w*|die\w*|defaults?|unsubscrib\w*|lapse\w*))\b")
    questions = ("When does a row start, and which column says how long it lasted or has lasted so far?",
                 "Which column says whether it ended (cancelled, failed) or is still going?",
                 "At which horizons does the answer matter (90, 180, 365 days)?")
    metrics = (("c_index", "Concordance (C-index)"), ("integrated_brier", "Integrated Brier score (censoring-weighted)"),
               ("auc_at_horizon", "Time-dependent AUC at a horizon"))
    baselines = ("kaplan_meier",)
    checks = ("PIT-005",)
    reliability = ("censored share", "horizons the follow-up reaches", "survival per group", "columns counted during the follow-up")
    evidence_track = None  # no research track yet (research/<track>/): nothing for agents to cite
    stratify = False
    runs_in_engine = False  # the five-stage engine has no censoring yet: dclab_core.survival runs it as a library
    Fields = TimeToEventFields

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return "integrated_brier"

    def target(self, frame: pd.DataFrame, solution: Any):
        return pd.to_numeric(frame[solution.target], errors="coerce").astype(float), None

    def summarise_target(self, y, class_labels, solution):
        return {"mean_train": float(y.mean()), "std_train": float(y.std())}, f"a mean duration of {y.mean():.4g}"

    def target_claim(self, summary, metric_label, spec):
        return ("fact", "A duration is judged with its censoring: rows still going count as still going.", ["evidence.target_summary"])

    def applies(self, solution: Any) -> bool:
        from .base import _get

        return _get(solution, "kind") == self.key  # only declared: the engine would read a censored duration as a finished one

    # ------------------------------------------------------------------ acceptance (16.2's harness, its own machine)
    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import subscription_lifetimes

        from .acceptance import Case

        replica = subscription_lifetimes.generate(seed)
        f = replica.facts
        return Case(replica.subscribers, f["duration"], "regression", "At signup, before the first bill: how long will this subscriber stay?",
                    solution={"kind_fields": {"duration_column": f["duration"], "event_column": f["event"], "start_column": f["start"],
                                              "horizons": f["horizons"], "group_column": f["group"]}}, facts=f)

    def solve(self, case: Any) -> Any:
        from dclab_core import survival

        from dclab_rnd.studio.leakage_review import from_moment

        fields = self.fields(case.solution.get("kind_fields"))  # what the person states; never which column leaks
        table = case.table
        d, e, s = fields["duration_column"], fields["event_column"], fields["start_column"]
        censoring = survival.censoring_check(table[e])
        if not censoring["ok"]:  # an export of ended rows only: the durations cannot be read honestly, so nothing is fitted
            return {"refused": censoring["why"], "censoring": censoring, "facts": case.facts}
        identifiers = [c for c in table.columns if c not in (d, e, s) and table[c].nunique() == len(table)]
        counted = survival.duration_check(table, duration=d, event=e, exclude=[s, *identifiers])
        said = {i["column"]: i["reason"] for i in from_moment([{"name": c} for c in table.columns], case.moment, {d, e, s})}
        out = set(counted["past_the_start"]) | set(said) | set(identifiers)
        features = [c for c in table.columns if c not in (d, e, s) and c not in out]
        result = survival.evaluate(table, duration=d, event=e, start=s, features=features, horizons=fields["horizons"],
                                   group=fields["group_column"])
        return {"censoring": censoring, "counted": counted, "moment_says": said, "identifiers": identifiers, "features": features,
                "evaluation": result, "facts": case.facts}

    def accept(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check

        if run.get("refused"):
            return [check("the data can hold censoring: rows still going are in it", False, run["refused"])]
        r, f = run["evaluation"], run["facts"]
        truth = f.get("true_survival") or {}
        bias = r["censoring"]["survival"]
        errors = {h: {k: abs(v[k] - truth[h]) for k in ("kaplan_meier", "censored_dropped", "censored_as_events")} for h, v in bias.items() if h in truth}
        # never worse than either shortcut, and both shortcuts clearly off where the follow-up censors (the longer horizons)
        honest = bool(errors) and all(e["kaplan_meier"] <= min(e["censored_dropped"], e["censored_as_events"]) for e in errors.values()) \
            and any(min(e["censored_dropped"], e["censored_as_events"]) - e["kaplan_meier"] >= 0.05 for e in errors.values())
        chosen = r["holdout"].get(r["chosen"]) or {}
        sc = r["shortcuts"]["mean_survival"]
        observed = r["shortcuts"]["observed_on_holdout"]
        gaps = [(abs(sc[w][h] - observed[h]), abs(sc["honest"][h] - observed[h])) for h in sc["honest"] for w in ("censored_dropped", "censored_as_events")
                if observed.get(h) is not None and None not in (sc[w].get(h), sc["honest"].get(h))]
        # each shortcut further from what the held-out rows show than the honest fit, by 0.05 at some horizon
        shortcut_gap = bool(gaps) and all(any(g[0] - g[1] >= 0.05 for g in gaps[i::2]) for i in (0, 1))
        groups = f.get("group")
        return [check("columns counted during the follow-up are kept out", all(c not in run["features"] for c in f["leaks"])
                      and set(f["leaks"]) <= set(run["counted"]["past_the_start"]),
                      "; ".join(f"{c}: {run['counted']['why'].get(c, 'not found')}" for c in f["leaks"])),
                check("censored rows stay censored: dropping them or calling them events is shown to be biased", bool(truth) and honest,
                      "; ".join(f"{h} days: Kaplan-Meier {v['kaplan_meier']}, dropped {v['censored_dropped']}, as events "
                                f"{v['censored_as_events']}, true {truth.get(h)}" for h, v in bias.items())),
                check("rows split by start day, training censored at the cut, the model chosen before the holdout",
                      r["split"]["holdout_from"] > r["split"]["train_until"] and r["split"]["censored_at_cut"] and r["chosen"] == min(
                          (n for n, v in r["validation"].items() if n != "kaplan_meier" and v.get("integrated_brier") is not None),
                          key=lambda n: r["validation"][n]["integrated_brier"], default="kaplan_meier"),
                      f"{r['split']['train']} training rows until {r['split']['train_until']}, {r['split']['holdout']} held out from "
                      f"{r['split']['holdout_from']}; {r['chosen']} chosen on the validation rows"),
                check("the chosen model fitted with censored rows dropped, or counted as events, is shown to be biased", shortcut_gap,
                      "; ".join(f"{h}: honest {sc['honest'].get(h)}, dropped {sc['censored_dropped'].get(h)}, as events "
                                f"{sc['censored_as_events'].get(h)}, held-out rows {r['shortcuts']['observed_on_holdout'].get(h)}"
                                for h in sc["honest"])),
                check("every model compared with Kaplan-Meier, the chosen one beats it", r["chosen"] != "kaplan_meier"
                      and r["beats_kaplan_meier"].get(r["chosen"], False),
                      f"{r['chosen']}: integrated Brier {chosen.get('integrated_brier')} against {r['holdout']['kaplan_meier'].get('integrated_brier')}"),
                check("C-index, integrated Brier and time-dependent AUC where the follow-up reaches", chosen.get("c_index") is not None
                      and chosen.get("integrated_brier") is not None and bool(chosen.get("auc")),
                      f"C-index {chosen.get('c_index')}, AUC {chosen.get('auc')}, horizons reached {r['horizons_reached']}"),
                check("survival curves per group in the final record", bool(groups) and len(r["curves"]) >= 2,
                      ", ".join(f"{k}: {v['survival']}" for k, v in r["curves"].items()))]


KIND = TimeToEvent()
