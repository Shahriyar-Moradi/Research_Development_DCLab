"""Forecasting (package 16.3): many series, a horizon, a forecast moment; rolling-origin backtests that never see the
future. The kind runs its own machine (``dclab_core.forecast``), not the five tabular stages: a forecast is judged on
origins in time, against the seasonal naive, with covariates kept to those known at the moment."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .base import ProblemKind


AHEAD = re.compile(r"\b(?:ahead|in advance|beforehand|planned|scheduled|booked|announced|"
                   r"(?:sets?|fix(?:es|ed)?|decide[sd]?|agree[sd]?|publish(?:es|ed)?|chos(?:e|en)|plan(?:s|ned)?) (?:\w+ ){0,2}(?:weeks?|days?|months?) (?:before|earlier))\b", re.I)  # 20.10 v2: "set weeks before"
ON_THE_DAY = re.compile(r"\b(?:on the day|measured|observed|recorded|afterwards|later|not known|unknown)\b", re.I)


def known_from_moment(columns: list[str], moment: str | None) -> dict[str, str]:
    """The columns the person's moment says are known ahead, each with the words that say so: a clause that says
    something is planned or known ahead (and nothing measured on the day) names the columns whose words it contains.
    The leakage reviewer's word matching (``studio.leakage_review``), turned round: what the moment allows, not what it
    excludes. A column the moment does not name stays unknown ahead."""
    from dclab_rnd.studio.leakage_review import WEAK, _column_stems, _stems

    known: dict[str, str] = {}
    for clause in re.split(r"[.;:,]\s*", str(moment or "")):
        if not AHEAD.search(clause) or ON_THE_DAY.search(clause):
            continue
        words = _stems(clause)
        for name in columns:
            shared = _column_stems(name) & words
            if name not in known and shared and not shared <= WEAK:
                known[name] = clause.strip()
    return known


class ForecastFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    series_column: str
    horizon: int = Field(ge=1, le=366)
    season: int = Field(default=7, ge=1, le=366)
    gap: int = Field(default=0, ge=0, le=366)
    known_ahead: list[str] = Field(default_factory=list, max_length=50)


class Forecasting(ProblemKind):
    key = "forecasting"
    title = "Forecasting"
    description = "Many series forecast a horizon ahead from a forecast moment: demand per store and product, load per region."
    base_task = "regression"
    task_type = "timeseries_regression"
    specificity = 15
    packs = ("timeseries",)
    words = r"\b(forecast\w*|demand|per (?:day|week|hour)|next (?:week|month|day)s? (?:sales|demand|load))\b"
    questions = ("What is one series (a store and a product, a region), and how far ahead must the forecast look?",
                 "When is the forecast made, and which inputs are known by then for the days ahead (planned promotions, prices)?",
                 "Is there a gap between the last day known and the first day forecast?")
    metrics = (("mase", "Mean absolute scaled error (against the seasonal naive)"), ("smape", "Symmetric mean absolute percentage error"),
               ("mae", "Mean absolute error"))
    baselines = ("naive", "seasonal_naive", "moving_average")
    checks = ("PIT-005", "PIT-006")
    reliability = ("errors by horizon step", "intervals from backtest residuals", "short series", "covariates known ahead")
    evidence_track = "time-series-forecasting"
    aliases = ('time series forecasting', 'demand forecast', 'sales forecast', 'backtesting')  # 16.14: the names this kind of problem goes by (read into its evidence record)
    stratify = False
    runs_in_engine = True  # 16.15: the engine runs its own machine (ForecastMachine), rolling origins and all
    Fields = ForecastFields

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return "mase"

    def target(self, frame: pd.DataFrame, solution: Any):
        return pd.to_numeric(frame[solution.target], errors="coerce").astype(float), None

    def summarise_target(self, y, class_labels, solution):
        return {"mean_train": float(y.mean()), "std_train": float(y.std())}, f"a target mean of {y.mean():.4g}"

    def target_claim(self, summary, metric_label, spec):
        return ("fact", "A forecast is judged on rolling origins in time, against the seasonal naive.", ["evidence.target_summary"])

    export = {"notebook": False, "service": False}  # 16.15: the notebook and the service know rows, not origins

    def own_machine(self) -> Any:
        return ForecastMachine

    def accept_engine(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check

        records, f = run.records, run.case.facts
        leak, models, final = run.evidence("leakage"), run.records["models"], run.evidence("final")
        ci = final.get("holdout_primary_metric_ci") or {}
        return [check("the engine runs every stage as the forecast machine", all((records[s] or {}).get("machine") == self.key for s in records),
                      ", ".join(f"{s}: {(records[s] or {}).get('machine')}" for s in records)),
                check(f"in the engine `{f['leak']}` is kept out at the leakage stage and the promotion read as known ahead",
                      f["leak"] in leak.get("kept_out", []) and "on_promotion" in run.evidence("features").get("known_ahead", []),
                      f"kept out {leak.get('kept_out')}; known ahead {run.evidence('features').get('known_ahead')}"),
                check("in the engine the model chosen on validation is the one the holdout scores, once", final.get("model") == (models.get("decision") or {}).get("chosen")
                      and final.get("holdout_uses_in_this_project") == 1, f"{final.get('model')}, holdout uses {final.get('holdout_uses_in_this_project')}"),
                check("in the engine the chosen model beats the seasonal naive on the holdout, with an interval over series",
                      bool(final.get("beats_seasonal_naive")) and ci.get("low") is not None,
                      f"MASE {final.get('holdout_metrics', {}).get('mase')} (95% {ci.get('low')} to {ci.get('high')}) against "
                      f"{final.get('baselines', {}).get('seasonal_naive', {}).get('mase')}")]

    def columns_of(self, solution: Any) -> list[str]:
        from .base import _get

        f = _get(solution, "kind_fields") or {}
        return [c for c in [f.get("series_column"), *(f.get("known_ahead") or [])] if c]

    def check_solution(self, solution: Any) -> None:
        if not solution.time_column:
            raise ValueError("A forecast needs its time column: the column that orders each series' days")
        if solution.kind_fields.get("series_column") in (solution.target, solution.time_column):
            raise ValueError("The series column names a series: it cannot be the target or the time column")
        if solution.time_column == solution.target:
            raise ValueError("The time column orders the days: it cannot be the target")
        if solution.metric and solution.metric != ForecastMachine.primary_metric:  # the machine judges by MASE only
            raise ValueError(f"The forecast machine chooses and reports by {ForecastMachine.primary_metric}: {solution.metric!r} would be ignored")

    def applies(self, solution: Any) -> bool:
        """Only a declared forecast: a project with the time-series pack keeps running as regression until it declares this
        kind and its fields (the series, the horizon), which the engine's forecast machine needs."""
        from .base import _get

        return _get(solution, "kind") == self.key

    def detect(self, problem: str, profile: dict[str, Any] | None = None) -> str | None:
        found = super().detect(problem, profile)
        if found and not ((profile or {}).get("time_candidates") or profile is None):
            return None  # forecasting needs rows in time
        return found

    # ------------------------------------------------------------------ acceptance (16.2's harness, its own machine)
    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import store_demand

        from .acceptance import Case

        replica = store_demand.generate(seed)
        f = replica.facts
        return Case(replica.sales, "units", "regression", "Each Sunday evening, for the next 7 days of each store and product; promotions "
                    "are planned weeks ahead, the day's weather is measured on the day.",
                    solution={"kind": "forecasting", "time_column": f["time"],  # what the person states (16.15: the engine runs it)
                              "kind_fields": {"series_column": f["series"], "horizon": f["horizon"], "season": f["season"]}}, facts=f)

    def solve(self, case: Any) -> Any:
        from dclab_core import forecast

        f = case.facts  # the problem's shape as the person states it (series, time, target, horizon, season); never which column leaks
        columns = [c for c in case.table.columns if c not in (f["series"], f["time"], f["target"])]
        declared = known_from_moment(columns, case.moment)
        check = forecast.covariate_check(case.table, series=f["series"], time=f["time"], target=f["target"], known_ahead=list(declared))
        result = forecast.backtest(case.table, series=f["series"], time=f["time"], target=f["target"], horizon=f["horizon"],
                                   known_ahead=check["kept"], season=f["season"], origins=3)
        return {"covariates": check, "declared": declared, "backtest": result, "facts": f}

    def accept(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check

        b, c, f = run["backtest"], run["covariates"], run["facts"]
        seasonal = (b["scores"].get("seasonal_naive") or {}).get("mase")
        best = b["best"]
        seen = b["target_seen_until"]
        measured = b["intervals"]["measured"].get("global_gbm") or {}
        return [check("the backtests never read a target after their origin", bool(seen) and all(d <= o for o, d in seen.items()),
                      f"{len(b['origins'])} origins, gap {b['gap']}, last target read: {sorted(seen.values())[-1] if seen else None}"),
                check(f"`{f['leak']}` is kept out: not known at the forecast moment", f["leak"] in c["kept_out"] and f["leak"] not in b["known_ahead"],
                      c["why"].get(f["leak"], "")),
                check("the planned promotion is used: known ahead, as the moment says", "on_promotion" in b["known_ahead"],
                      run["declared"].get("on_promotion", "the moment does not say it is known ahead")),
                check("every model is compared with the seasonal naive", seasonal is not None and set(b["beats_seasonal_naive"]) >= {"naive", "global_gbm"},
                      f"seasonal naive MASE {seasonal}"),
                check("the chosen model beats the seasonal naive", best is not None and b["beats_seasonal_naive"].get(best, False),
                      f"{best}: MASE {(b['scores'].get(best) or {}).get('mase')} against {seasonal}"),
                check("intervals by horizon step, their coverage measured on a later origin", len(b["intervals"]["by_step"]) == f["horizon"]
                      and measured.get("coverage") is not None,
                      f"{len(b['intervals']['by_step'])} steps; coverage {measured.get('coverage')} of {b['intervals']['coverage']} "
                      f"on {measured.get('origin')}")]



from .machine import Machine, bootstrap_mean, claim  # noqa: E402


class ForecastMachine(Machine):
    """Forecasting in the engine (16.15). The holdout is the last forecast: the ``horizon`` dates after the last origin.
    Validation is rolling origins before it; the choice is made there; the holdout is read once in the final stage."""

    task_type = "forecasting"
    primary_metric = "mase"
    higher_is_better = False
    VALIDATION_ORIGINS = 3

    def __init__(self, kind: Any, solution: Any, frame: pd.DataFrame, project: dict[str, Any]):
        from dclab_core import forecast

        super().__init__(kind, solution, frame, project)
        f = self.fields
        self.time, self.series, self.target = solution.time_column, f["series_column"], solution.target
        self.frame = frame.assign(**{self.time: pd.to_datetime(frame[self.time], errors="coerce")}).dropna(subset=[self.time, self.target])
        self.origin = forecast.holdout_origin(self.frame, time=self.time, horizon=f["horizon"], gap=f["gap"])
        self.train = self.frame[self.frame[self.time] <= self.origin]  # every target the stages before the final may read

    def training_rows(self) -> pd.DataFrame:
        return self.train

    def holdout_description(self) -> str:
        f = self.fields
        return f"the {f['horizon']} dates after {self.origin.date()}" + (f" (after a gap of {f['gap']})" if f["gap"] else "") + ", every series"

    def cv_description(self) -> str:
        return f"{self.VALIDATION_ORIGINS} rolling origins of {self.fields['horizon']} dates before the holdout, each fitted on its past only"

    def _base(self) -> dict[str, Any]:
        return {"series_column": self.series, "time_column": self.time, "holdout_origin": str(self.origin.date())}

    def data(self, eid: str, previous: dict) -> tuple:
        f, train = self.fields, self.train
        held = self.frame[self.frame[self.time] > self.origin]
        sizes = train.groupby(self.series).size()
        from dclab_core.forecast import MIN_CLASSICAL

        others = [c for c in self.frame.columns if c not in (self.series, self.time, self.target)]
        evidence = {**self._base(), "train_rows": int(len(train)), "holdout_rows": int(len(held)), "holdout_split": self.holdout_description(),
                    "cv_protocol": self.cv_description(), "series": int(train[self.series].nunique()),
                    "dates": [str(train[self.time].min().date()), str(train[self.time].max().date())], "horizon": f["horizon"], "gap": f["gap"],
                    "season": f["season"], "short_series": int((sizes < MIN_CLASSICAL).sum()), "usable_feature_count": len(others),
                    "source_rows": int(len(self.frame)),
                    "target_summary": {"mean_train": round(float(train[self.target].mean()), 4), "std_train": round(float(train[self.target].std()), 4)},
                    "notes_for_person": [{"title": "The holdout is the last forecast", "proof": ["DCLAB-R02", "DCLAB-R17"],
                                          "text": f"The {f['horizon']} dates after {self.origin.date()} stay sealed until the final stage; every choice "
                                                  f"before it is made on {self.cv_description()}."}]}
        claims = [claim(f"{eid}-C1", "fact", f"{len(train)} training rows over {evidence['series']} series up to {self.origin.date()}; the holdout is "
                        f"{self.holdout_description()} ({len(held)} rows); {evidence['short_series']} series are too short for a model of their own.",
                        ["evidence.train_rows", "evidence.series", "evidence.holdout_split", "evidence.short_series"])]
        return evidence, claims, f"{evidence['series']} series, holdout {self.holdout_description()}", None

    def _known(self, previous: dict) -> list[str]:
        forbidden = {x.column for x in self.solution.forbidden}
        kept = ((previous.get("leakage") or {}).get("evidence") or {}).get("kept") or []
        return [c for c in kept if c not in forbidden]

    def leakage(self, eid: str, previous: dict) -> tuple:
        from dclab_core import forecast

        from .forecasting import known_from_moment

        columns = [c for c in self.frame.columns if c not in (self.series, self.time, self.target)]
        said = known_from_moment(columns, self.solution.prediction_moment)
        declared = sorted(set(self.fields["known_ahead"]) | set(said))
        check = forecast.covariate_check(self.train, series=self.series, time=self.time, target=self.target, known_ahead=declared)
        forbidden = [x.column for x in self.solution.forbidden if x.column in self.frame.columns]
        evidence = {**self._base(), "declared_known_ahead": declared, "moment_says": said, "kept": check["kept"], "kept_out": check["kept_out"],
                    "to_confirm": check["to_confirm"], "why": check["why"], "declared_leakage_features": sorted(set(check["kept_out"]) | set(forbidden)),
                    "heuristic_review_candidates": [{"feature": c, "reasons": ["declared_known_ahead_but_moves_with_the_target" if c in check["kept"]
                                                                               else "constant_in_its_series_but_follows_the_series_level_of_the_target"]}
                                                    for c in check["to_confirm"]],
                    "notes_for_person": [{"level": "warning", "title": f"Confirm {c} is set before the forecast moment", "text": check["why"][c],
                                          "proof": ["DCLAB-R01", "DCLAB-R05"], "action": "Confirm or forbid this column in the solution"}
                                         for c in check["to_confirm"]]}
        claims = [claim(f"{eid}-C1", "fact", f"Known ahead and used: {', '.join(check['kept']) or 'none'}; kept out (a future value is not known at the "
                        f"moment): {', '.join(check['kept_out']) or 'none'}.", ["evidence.kept", "evidence.kept_out"])]
        if check["to_confirm"]:
            claims.append(claim(f"{eid}-C2", "risk", "To confirm before the forecast moment: " + "; ".join(f"{c} ({check['why'][c]})" for c in check["to_confirm"])
                                + ".", ["evidence.to_confirm", "evidence.why"]))
        return evidence, claims, f"{len(check['kept'])} covariates known ahead, {len(check['kept_out'])} kept out", None

    def features(self, eid: str, previous: dict) -> tuple:
        from dclab_core.forecast import LAGS, WINDOWS

        known = self._known(previous)
        recipe = "lags_windows_calendar" + ("_known_ahead" if known else "")
        about = (f"lags {', '.join(map(str, LAGS))} and means over {', '.join(map(str, WINDOWS))} days of each series' past, the day of week and "
                 f"month, the value one season back" + (f", and the covariates known ahead: {', '.join(known)}" if known else ""))
        count = len(LAGS) + len(WINDOWS) + 5 + len(known)
        evidence = {**self._base(), "selected_recipe": recipe, "best_recipe": recipe, "selected_recipe_description": about, "known_ahead": known,
                    "stage_results": [{"recipe": recipe, "description": about, "feature_count_mean": count, "metrics": {"mase": {"mean": None, "std": None}}}],
                    "selection_rule": "one recipe: each forecast date's features come from the past of its origin only"}
        decision = {"kind": "recipe", "selected": recipe, "chosen": recipe, "rule": evidence["selection_rule"],
                    "options": [{"id": recipe, "label": recipe, "description": about, "features": count}]}
        claims = [claim(f"{eid}-C1", "fact", f"The global model reads {about}.", ["evidence.selected_recipe_description"])]
        return evidence, claims, about, decision

    def models(self, eid: str, previous: dict) -> tuple:
        from dclab_core import forecast

        f, known = self.fields, self._known(previous)
        result = forecast.backtest(self.train, series=self.series, time=self.time, target=self.target, horizon=f["horizon"],
                                   known_ahead=known, season=f["season"], origins=self.VALIDATION_ORIGINS, gap=f["gap"])
        ranking = []
        for name, sc in result["scores"].items():
            per_origin = [forecast.origin_scores(result, o).get(name, {}).get("mase") for o in result["origins"]]
            per_origin = [x for x in per_origin if x is not None]
            ranking.append({"model": name, "mean": sc["mase"], "std": round(float(pd.Series(per_origin).std()), 4) if len(per_origin) > 1 else 0.0,
                            "smape": sc["smape"], "beats_seasonal_naive": result["beats_seasonal_naive"].get(name), "forecasts": sc["forecasts"],
                            "covers_every_series": sc["covers_every_series"]})
        ranking.sort(key=lambda r: (r["mean"] is None, r["mean"]))
        # a model that skips the short series is scored on fewer, easier forecasts: it may not be chosen over the others
        best = next((r["model"] for r in ranking if r["model"] != "seasonal_naive" and r["mean"] is not None and r["covers_every_series"]),
                    "seasonal_naive")
        evidence = {**self._base(), "selected_model": best, "ranking": ranking, "validation_origins": result["origins"],
                    "intervals": result["intervals"], "selection_rule": "the lowest validation MASE among the models that forecast every series, "
                                                                       "other than the seasonal naive baseline they are judged against",
                    "notes_for_person": [{"title": f"{best} chosen on the validation origins",
                                          "text": f"MASE {next(r['mean'] for r in ranking if r['model'] == best)} against the seasonal naive's "
                                                  f"{next((r['mean'] for r in ranking if r['model'] == 'seasonal_naive'), None)}, over origins "
                                                  f"{', '.join(result['origins'])}; the holdout is still sealed.", "proof": ["DCLAB-R13", "DCLAB-R17"]}]}
        decision = {"kind": "model", "selected": best, "chosen": best, "rule": evidence["selection_rule"],
                    "options": [{"id": r["model"], "label": r["model"], "mean": r["mean"], "std": r["std"]} for r in ranking]}
        claims = [claim(f"{eid}-C1", "fact", "Validation MASE by model: " + "; ".join(f"{r['model']} {r['mean']}" for r in ranking) + f". Chosen: {best}.",
                        ["evidence.ranking", "evidence.selected_model"])]
        return evidence, claims, f"{best} chosen on {len(result['origins'])} validation origins", decision

    def final(self, eid: str, previous: dict) -> tuple:
        from dclab_core import forecast

        f, known = self.fields, self._known(previous)
        locked = ((previous.get("data") or {}).get("evidence") or {}).get("holdout_origin")
        if locked != str(self.origin.date()):  # the table changed since the holdout was locked: rerun from the data stage
            raise ValueError(f"The holdout was locked at {locked}, but the table now puts it at {self.origin.date()}: rerun the data stage")
        chosen = self.chosen(previous.get("models"))
        recipe = self.chosen(previous.get("features"))
        result = forecast.backtest(self.frame, series=self.series, time=self.time, target=self.target, horizon=f["horizon"],
                                   known_ahead=known, season=f["season"], origins=self.VALIDATION_ORIGINS + 1, gap=f["gap"])
        last = str(self.origin.date())
        mine = forecast.origin_scores(result, last).get(chosen) or {}  # the holdout: the last origin's forecasts, read here only
        held = forecast.origin_scores(result, last, series=mine.get("series"))  # the baselines on the same series as the chosen model
        low, high = bootstrap_mean(mine.get("mase_by_series") or [])
        cv = next((r["mean"] for r in ((previous.get("models") or {}).get("evidence") or {}).get("ranking", []) if r["model"] == chosen), None)
        measured = (result["intervals"].get("measured") or {}).get("global_gbm") if chosen == "global_gbm" else None
        baselines = {b: {"mase": held[b]["mase"], "smape": held[b]["smape"]} for b in ("seasonal_naive", "naive", "moving_average") if b in held}
        evidence = {**self._base(), "model": chosen, "feature_recipe": recipe, "selected_model_params": {},
                    "holdout_metrics": {"mase": mine.get("mase"), "smape": mine.get("smape"), "mae": mine.get("mae")},
                    "holdout_primary_metric_ci": {"low": low, "high": high, "point": mine.get("mase"), "method": "bootstrap over series, 95%"},
                    "cv_selected_metric_mean": cv, "cv_to_holdout_gap": round(mine["mase"] - cv, 4) if mine.get("mase") is not None and cv is not None else None,
                    "baselines": baselines, "beats_seasonal_naive": (mine.get("mase") is not None and "seasonal_naive" in held
                                                                     and mine["mase"] < held["seasonal_naive"]["mase"]),
                    "holdout_split": self.holdout_description(), "holdout_rows": int(mine.get("forecasts", 0)) * f["horizon"],
                    "train_rows": int(len(self.train)), "intervals_measured": measured, "selection_rule": "the model chosen on validation, read once here",
                    "notes_for_person": [{"title": f"Holdout MASE {mine.get('mase')} for {chosen}",
                                          "text": f"Against the seasonal naive's {baselines.get('seasonal_naive', {}).get('mase')} on the same last forecast "
                                                  f"({self.holdout_description()}); 95% interval over series {low} to {high}.",
                                          "proof": ["DCLAB-R17", "DCLAB-R18"]}]}
        claims = [claim(f"{eid}-C1", "fact", f"On the holdout ({self.holdout_description()}) {chosen} scores MASE {mine.get('mase')} (95% interval over "
                        f"series {low} to {high}); the seasonal naive {baselines.get('seasonal_naive', {}).get('mase')}.",
                        ["evidence.holdout_metrics", "evidence.baselines", "evidence.holdout_primary_metric_ci"],
                        ["One last forecast window: a holdout of one origin.", "Not approved for production from this evidence alone."])]
        if measured:
            claims.append(claim(f"{eid}-C2", "fact", f"Intervals set on the earlier origins covered {measured.get('coverage')} of the holdout's values.",
                                ["evidence.intervals_measured"]))
        return evidence, claims, f"{chosen}: holdout MASE {mine.get('mase')}", {"kind": "final", "selected": "defaults", "chosen": "defaults",
                                                                                  "rule": evidence["selection_rule"], "options": []}


KIND = Forecasting()
