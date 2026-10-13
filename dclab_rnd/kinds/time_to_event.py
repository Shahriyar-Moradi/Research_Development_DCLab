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
    evidence_track = "churn-prediction"  # 16.14: the research track whose evidence agents cite for this kind
    aliases = ('survival analysis', 'churn timing', 'censoring', 'time until cancellation')  # 16.14: the names this kind of problem goes by (read into its evidence record)
    stratify = False
    runs_in_engine = True  # 16.15: the engine runs its own machine (SurvivalMachine), censoring and all
    Fields = TimeToEventFields

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return "integrated_brier"

    def target(self, frame: pd.DataFrame, solution: Any):
        return pd.to_numeric(frame[solution.target], errors="coerce").astype(float), None

    def summarise_target(self, y, class_labels, solution):
        return {"mean_train": float(y.mean()), "std_train": float(y.std())}, f"a mean duration of {y.mean():.4g}"

    def target_claim(self, summary, metric_label, spec):
        return ("fact", "A duration is judged with its censoring: rows still going count as still going.", ["evidence.target_summary"])

    export = {"notebook": False, "service": False}  # 16.15: the notebook and the service know rows with outcomes, not censoring

    def own_machine(self) -> Any:
        return SurvivalMachine

    def columns_of(self, solution: Any) -> list[str]:
        from .base import _get

        f = _get(solution, "kind_fields") or {}
        return [c for c in (f.get("duration_column"), f.get("event_column"), f.get("start_column"), f.get("group_column")) if c]

    def check_solution(self, solution: Any) -> None:
        k = solution.kind_fields
        if k.get("duration_column") != solution.target:
            raise ValueError("The target of a time-to-event problem is its duration column")
        named = [k.get("duration_column"), k.get("event_column"), k.get("start_column")]
        if len(set(named)) < len(named):
            raise ValueError("The duration, event and start columns must be three different columns")
        if solution.metric and solution.metric != SurvivalMachine.primary_metric:  # the machine judges by the integrated Brier score only
            raise ValueError(f"The survival machine chooses and reports by {SurvivalMachine.primary_metric}: {solution.metric!r} would be ignored")

    def accept_engine(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check

        records, f = run.records, run.case.facts
        leak, models, final = run.evidence("leakage"), run.records["models"], run.evidence("final")
        sc = final.get("shortcuts") or {}
        honest, observed = (sc.get("mean_survival") or {}).get("honest", {}), sc.get("observed_on_holdout") or {}
        gaps = [abs(sc["mean_survival"][w][h] - observed[h]) - abs(honest[h] - observed[h]) for h in honest
                for w in ("censored_dropped", "censored_as_events") if observed.get(h) is not None and sc["mean_survival"][w].get(h) is not None]
        return [check("the engine runs every stage as the survival machine", all((records[s] or {}).get("machine") == self.key for s in records),
                      ", ".join(f"{s}: {(records[s] or {}).get('machine')}" for s in records)),
                check("in the engine the columns counted during the follow-up are kept out at the leakage stage",
                      set(f["leaks"]) <= set(leak.get("kept_out", [])) and not set(f["leaks"]) & set(run.evidence("features").get("features", [])),
                      f"kept out {leak.get('kept_out')}"),
                check("in the engine the model chosen on validation is the one the holdout scores, once", final.get("model") == (models.get("decision") or {}).get("chosen")
                      and final.get("holdout_uses_in_this_project") == 1, f"{final.get('model')}, holdout uses {final.get('holdout_uses_in_this_project')}"),
                check("in the engine the chosen model beats Kaplan-Meier on the holdout, and the wrong fits are shown biased",
                      bool(final.get("beats_kaplan_meier")) and bool(gaps) and max(gaps) >= 0.05,
                      f"integrated Brier {final.get('holdout_metrics', {}).get('integrated_brier')} against "
                      f"{final.get('baselines', {}).get('kaplan_meier', {}).get('integrated_brier')}")]

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
                    solution={"kind": "time_to_event", "time_column": None,
                              "kind_fields": {"duration_column": f["duration"], "event_column": f["event"], "start_column": f["start"],
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



from .machine import Machine, claim  # noqa: E402


class SurvivalMachine(Machine):
    """Time to event in the engine (16.15). The holdout is the latest starts; training rows are censored at its first day;
    the choice is made on a validation cut inside the training rows; the holdout is read once in the final stage."""

    task_type = "survival"
    primary_metric = "integrated_brier"
    higher_is_better = False

    def __init__(self, kind: Any, solution: Any, frame: Any, project: dict[str, Any]):
        from dclab_core import survival

        super().__init__(kind, solution, frame, project)
        f = self.fields
        self.d, self.e, self.s = f["duration_column"], f["event_column"], f["start_column"]
        whole = survival.censoring_check(frame[self.e])  # the flag's form only: two values
        if not whole["ok"]:
            raise ValueError(f"Time to event cannot be read from this table: {whole['why']}")
        self.parts = survival.split(frame, duration=self.d, event=self.e, start=self.s)
        self.check = survival.censoring_check(self.parts["train"][self.e])  # reported as the training rows show it, censored at the holdout's day
        if not self.check["ok"]:
            raise ValueError(f"Time to event cannot be learned from the training rows: {self.check['why']}")

    def training_rows(self) -> Any:
        return self.parts["train"]

    def holdout_description(self) -> str:
        sp = self.parts["split"]
        return f"the {sp['holdout']} latest starts, from {sp['holdout_from']} (training rows censored at that day)"

    def cv_description(self) -> str:
        sp = self.parts["split"]
        return f"a validation cut inside the training rows from {sp['validation_from']} ({sp['validation_rows']} rows), the rows fitted censored at it"

    def _features(self, previous: dict) -> list[str]:
        return list(((previous.get("leakage") or {}).get("evidence") or {}).get("features") or [])

    def data(self, eid: str, previous: dict) -> tuple:
        sp, train = self.parts["split"], self.parts["train"]
        f = self.fields
        evidence = {"train_rows": sp["train"], "holdout_rows": sp["holdout"], "holdout_split": self.holdout_description(),
                    "cv_protocol": self.cv_description(), "split": sp, "censoring": self.check, "horizons": f["horizons"],
                    "events_in_training": int(train[self.e].sum()), "usable_feature_count": int(len(self.frame.columns) - 3),
                    "source_rows": int(len(self.frame)),
                    "target_summary": {"mean_train": round(float(train[self.d].mean()), 4), "censored_share": self.check.get("censored_share")},
                    "notes_for_person": [{"title": "Rows still going are censored, not missing", "text": self.check["why"] + ". The latest starts "
                                          f"are the holdout ({self.holdout_description()}).", "proof": ["DCLAB-R02", "DCLAB-R17"]}]}
        claims = [claim(f"{eid}-C1", "fact", f"{sp['train']} training rows ({evidence['events_in_training']} ended before the holdout's first day), "
                        f"{sp['holdout']} held out from {sp['holdout_from']}; {self.check['why']}.", ["evidence.split", "evidence.censoring"])]
        return evidence, claims, f"{sp['train']} training rows, holdout {self.holdout_description()}", None

    def leakage(self, eid: str, previous: dict) -> tuple:
        from dclab_core import survival

        from dclab_rnd.studio.leakage_review import from_moment

        train = self.parts["train"]
        identifiers = [c for c in self.frame.columns if c not in (self.d, self.e, self.s) and self.frame[c].nunique() == len(self.frame)]
        counted = survival.duration_check(train, duration=self.d, event=self.e, exclude=[self.s, *identifiers])  # training rows only
        said = {i["column"]: i["reason"] for i in from_moment([{"name": c} for c in self.frame.columns], self.solution.prediction_moment,
                                                               {self.d, self.e, self.s})}
        forbidden = [x.column for x in self.solution.forbidden if x.column not in (self.d, self.e, self.s)]  # the kind's own columns are not inputs
        out = sorted(set(counted["past_the_start"]) | set(said) | set(identifiers) | set(forbidden))
        features = [c for c in self.frame.columns if c not in (self.d, self.e, self.s, "_day", "_start") and c not in out
                    and c not in self.solution.identifiers]
        why = {**{c: "a different value on every row: an identifier" for c in identifiers}, **counted["why"],
               **{c: said[c] for c in said}, **{c: "forbidden in the solution" for c in forbidden}}
        evidence = {"kept_out": out, "why": why, "features": features, "declared_leakage_features": out,
                    "heuristic_review_candidates": [{"feature": c, "reasons": ["counted_during_the_follow_up"]} for c in counted["past_the_start"]],
                    "notes_for_person": [{"level": "warning", "title": f"{c} is kept out", "text": why[c], "proof": ["DCLAB-R01", "DCLAB-R05"]}
                                         for c in counted["past_the_start"]]}
        claims = [claim(f"{eid}-C1", "fact", f"Kept out: {', '.join(f'{c} ({why[c][:80]})' for c in out) or 'none'}. Read: {', '.join(features)}.",
                        ["evidence.kept_out", "evidence.features"])]
        return evidence, claims, f"{len(out)} columns kept out, {len(features)} read", None

    def features(self, eid: str, previous: dict) -> tuple:
        features = self._features(previous)
        recipe = "standardised_numbers_onehot_categories"
        about = f"{len(features)} columns known at the start: numbers standardised and categories one-hot, both fitted on the training rows"
        evidence = {"selected_recipe": recipe, "best_recipe": recipe, "selected_recipe_description": about, "features": features,
                    "stage_results": [{"recipe": recipe, "description": about, "feature_count_mean": len(features),
                                       "metrics": {self.primary_metric: {"mean": None, "std": None}}}],
                    "selection_rule": "one design for every model"}
        decision = {"kind": "recipe", "selected": recipe, "chosen": recipe, "rule": evidence["selection_rule"],
                    "options": [{"id": recipe, "label": recipe, "description": about, "features": len(features)}]}
        return evidence, [claim(f"{eid}-C1", "fact", f"The models read {about}.", ["evidence.features"])], about, decision

    def models(self, eid: str, previous: dict) -> tuple:
        from dclab_core import survival

        f = self.fields
        picked = survival.choose(self.parts, duration=self.d, event=self.e, features=self._features(previous), horizons=f["horizons"])
        v = picked["validation"]
        ranking = sorted(({"model": n, "mean": r.get("integrated_brier"), "std": 0.0, "c_index": r.get("c_index")} for n, r in v.items()),
                         key=lambda r: (r["mean"] is None, r["mean"]))
        best = picked["chosen"]
        evidence = {"selected_model": best, "ranking": ranking, "validation": v, "selection_rule": picked["chosen_by"],
                    "notes_for_person": [{"title": f"{best} chosen on the validation rows", "proof": ["DCLAB-R13", "DCLAB-R17"],
                                          "text": "Integrated Brier score by model: " + "; ".join(f"{r['model']} {r['mean']}" for r in ranking)
                                                  + ". The holdout is still sealed."}]}
        decision = {"kind": "model", "selected": best, "chosen": best, "rule": picked["chosen_by"],
                    "options": [{"id": r["model"], "label": r["model"], "mean": r["mean"], "std": r["std"]} for r in ranking]}
        claims = [claim(f"{eid}-C1", "fact", "Validation integrated Brier score by model: " + "; ".join(f"{r['model']} {r['mean']}" for r in ranking)
                        + f". Chosen: {best}.", ["evidence.ranking", "evidence.selected_model"])]
        return evidence, claims, f"{best} chosen on the validation rows", decision

    def final(self, eid: str, previous: dict) -> tuple:
        from dclab_core import survival

        f = self.fields
        sp = self.parts["split"]
        locked = ((previous.get("data") or {}).get("evidence") or {}).get("split") or {}
        if (locked.get("holdout_from"), locked.get("holdout")) != (sp["holdout_from"], sp["holdout"]):  # the table changed since the lock
            raise ValueError(f"The holdout was locked from {locked.get('holdout_from')} ({locked.get('holdout')} rows), but the table now puts it "
                             f"from {sp['holdout_from']} ({sp['holdout']} rows): rerun the data stage")
        chosen = self.chosen(previous.get("models"))
        scored = survival.score(self.parts, chosen, duration=self.d, event=self.e, features=self._features(previous), horizons=f["horizons"],
                                group=f["group_column"])
        mine, km = scored["holdout"].get(chosen) or {}, scored["holdout"].get("kaplan_meier") or {}
        cv = next((r["mean"] for r in ((previous.get("models") or {}).get("evidence") or {}).get("ranking", []) if r["model"] == chosen), None)
        metrics = {"integrated_brier": mine.get("integrated_brier"), "c_index": mine.get("c_index"),
                   **{f"auc_{h}": v for h, v in (mine.get("auc") or {}).items()}}
        evidence = {"model": chosen, "feature_recipe": self.chosen(previous.get("features")), "holdout_metrics": metrics,
                    "holdout_primary_metric_ci": {"low": None, "high": None, "point": mine.get("integrated_brier"),
                                                  "method": "not computed: one holdout of the latest starts"},
                    "cv_selected_metric_mean": cv, "cv_to_holdout_gap": round(mine["integrated_brier"] - cv, 4)
                    if mine.get("integrated_brier") is not None and cv is not None else None,
                    "baselines": {"kaplan_meier": {"integrated_brier": km.get("integrated_brier"), "c_index": km.get("c_index")}},
                    "beats_kaplan_meier": scored["beats_kaplan_meier"].get(chosen), "horizons_reached": scored["horizons_reached"],
                    "shortcuts": scored["shortcuts"], "censoring_bias": scored["censoring"], "curves": scored["curves"],
                    "holdout_split": self.holdout_description(), "holdout_rows": self.parts["split"]["holdout"], "train_rows": self.parts["split"]["train"],
                    "selection_rule": "the model chosen on validation, read once here",
                    "notes_for_person": [{"title": f"Holdout integrated Brier {mine.get('integrated_brier')} for {chosen}", "proof": ["DCLAB-R17", "DCLAB-R18"],
                                          "text": f"Kaplan-Meier {km.get('integrated_brier')}; C-index {mine.get('c_index')}; time-dependent AUC "
                                                  f"{mine.get('auc')} at the horizons the follow-up reaches."}]}
        claims = [claim(f"{eid}-C1", "fact", f"On the holdout ({self.holdout_description()}) {chosen} scores integrated Brier {mine.get('integrated_brier')} "
                        f"against Kaplan-Meier's {km.get('integrated_brier')}, C-index {mine.get('c_index')}.",
                        ["evidence.holdout_metrics", "evidence.baselines"],
                        ["One holdout of the latest starts, scored where their follow-up reaches.", "Not approved for production from this evidence alone."]),
                  claim(f"{eid}-C2", "fact", f"Fitted with censored rows dropped or counted as events, {chosen}'s mean survival moves away from what the "
                        f"held-out rows show: {scored['shortcuts']['mean_survival']} against {scored['shortcuts']['observed_on_holdout']}.",
                        ["evidence.shortcuts"])]
        return evidence, claims, f"{chosen}: holdout integrated Brier {mine.get('integrated_brier')}", {
            "kind": "final", "selected": "defaults", "chosen": "defaults", "rule": evidence["selection_rule"], "options": []}


KIND = TimeToEvent()
