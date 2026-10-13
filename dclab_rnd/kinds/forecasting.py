"""Forecasting (package 16.3): many series, a horizon, a forecast moment; rolling-origin backtests that never see the
future. The kind runs its own machine (``dclab_core.forecast``), not the five tabular stages: a forecast is judged on
origins in time, against the seasonal naive, with covariates kept to those known at the moment."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .base import ProblemKind


AHEAD = re.compile(r"\b(?:ahead|in advance|beforehand|planned|scheduled|booked|announced)\b", re.I)
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
    stratify = False
    runs_in_engine = False  # the five-stage engine has no rolling origins yet: dclab_core.forecast runs it as a library
    Fields = ForecastFields

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return "mase"

    def target(self, frame: pd.DataFrame, solution: Any):
        return pd.to_numeric(frame[solution.target], errors="coerce").astype(float), None

    def summarise_target(self, y, class_labels, solution):
        return {"mean_train": float(y.mean()), "std_train": float(y.std())}, f"a target mean of {y.mean():.4g}"

    def target_claim(self, summary, metric_label, spec):
        return ("fact", "A forecast is judged on rolling origins in time, against the seasonal naive.", ["evidence.target_summary"])

    def applies(self, solution: Any) -> bool:
        """Only a declared forecast: the five-stage engine has no MASE or rolling origins, so a project with the time-series
        pack keeps running as regression until it declares this kind (and its machine runs it)."""
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
        return Case(replica.sales, "units", "regression", "Each Sunday evening, for the next 7 days of each store and product; promotions "
                    "are planned weeks ahead, the day's weather is measured on the day.", facts=replica.facts)

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


KIND = Forecasting()
