"""Streams and sensor anomalies (package 16.11): readings over time per unit (machines, sessions), rare incidents, and
alerts judged by the incidents they catch. The kind runs its own machine (``dclab_core.streams``), not the five tabular
stages: windows over each unit's past, labels from the incidents after them, detectors and a classifier with an alert
budget, and metrics per incident."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .base import ProblemKind


class StreamFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    unit_column: str
    time_column: str
    sensor_columns: list[str] = Field(min_length=1, max_length=50)
    incident_time_column: str  # in the incidents table the person brings
    window_hours: float = Field(default=24, gt=0, le=24 * 60)
    horizon_hours: float = Field(default=48, gt=0, le=24 * 60)
    budget_per_unit_day: float = Field(default=0.2, gt=0, le=50)
    label_delay_hours: float = Field(default=0, ge=0, le=24 * 90)  # how long after an incident it is known (a delayed label)


class Streams(ProblemKind):
    key = "streams"
    title = "Streams: sensor anomalies and incidents"
    description = "Readings over time from machines or sessions: alert before an incident, within an alert budget."
    base_task = "binary"
    task_type = "streams"
    specificity = 14  # below time to event: 'time to failure, with censored machines' is a survival brief
    packs = ()
    words = r"\b(sensors?|telemetry|predictive maintenance|before (?:a |the )?(?:failure|breakdown|outage)|machine failures?|anomal\w* in (?:the )?(?:stream|logs|readings))\b"
    questions = ("Which column names the unit (a machine), which the time, and which hold the readings?",
                 "Where are the incidents (failures) and their times, and how long before one is an alert still useful?",
                 "How many alerts per machine a day can the team look at?")
    metrics = (("incident_recall", "Incidents caught within the warning horizon"), ("lead_time", "Hours of warning before an incident"),
               ("alerts_per_unit_day", "Alerts per unit-day"))
    baselines = ("isolation_forest", "own_change")
    checks = ("PIT-005",)
    reliability = ("incidents in the holdout period", "units never seen", "false alerts per unit-day")
    evidence_track = "anomaly-and-fraud-detection"  # 16.14: the research track whose evidence agents cite for this kind
    stratify = False
    runs_in_engine = False  # windows and incidents, not rows: dclab_core.streams runs it as a library
    Fields = StreamFields

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return "incident_recall"

    def applies(self, solution: Any) -> bool:
        from .base import _get

        return _get(solution, "kind") == self.key

    # ------------------------------------------------------------------ acceptance (16.2's harness, its own machine)
    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import machine_sensors

        from .acceptance import Case

        r = machine_sensors.generate(seed)
        f = r.facts
        return Case(r.readings, "incident", "binary", "Every six hours, for each machine: raise an alert now?",
                    solution={"kind_fields": {"unit_column": f["unit"], "time_column": f["time"], "sensor_columns": f["sensors"],
                                              "incident_time_column": f["incident_time"]},
                              "incidents": r.incidents}, facts=f)

    def solve(self, case: Any) -> Any:
        from dclab_core import streams

        from dclab_rnd.expansion import detectors

        f = self.fields(case.solution.get("kind_fields"))  # what the person states; never the warning signs
        result = streams.evaluate(case.table, case.solution["incidents"], unit=f["unit_column"], time=f["time_column"], sensors=f["sensor_columns"],
                                  incident_time=f["incident_time_column"], window_hours=f["window_hours"], horizon_hours=f["horizon_hours"],
                                  label_delay_hours=f["label_delay_hours"], budget_per_unit_day=f["budget_per_unit_day"],
                                  detectors={"percentile_rules": detectors.PercentileRules, "ecod": detectors.ECOD})  # 15.5's, per window
        return {"evaluation": result, "facts": case.facts}

    def accept(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check

        r = run["evaluation"]
        best = r["methods"][r["best"]]
        unseen = r["unseen_units"] or {}
        fc = r["future_check"]
        return [check("no window moves when every reading after a cut is changed", fc["windows_before_the_cut"] > 0 and fc["changed_by_the_future"] == 0,
                      f"{fc['windows_before_the_cut']} windows end before {fc['cut']}; changed by the later readings: {fc['changed_by_the_future']}"),
                check("training labels are complete and known before the holdout period",
                      pd.Timestamp(r["split"]["train_labels_until"]) <= pd.Timestamp(r["split"]["cut"]),
                      f"split by time at {r['split']['cut']}; training labels read until {r['split']['train_labels_until']}"),
                check("thresholds and the choice of method are set on the validation period, not the holdout", r["best"] in r["validation"]
                      and r["split"]["thresholds_set_on"].startswith("the validation period"), r["chosen_by"]),
                check("most incidents are caught, with hours of warning, within the alert budget", (best["incident_recall"] or 0) >= 0.75
                      and (best["lead_time_hours_median"] or 0) >= 6 and (best["alerts_per_unit_day"] or 99) <= 1.5 * r["budget_per_unit_day"],
                      f"{r['best']}: {best['caught']} of {best['incidents']} incidents, median {best['lead_time_hours_median']} h before, "
                      f"{best['alerts_per_unit_day']} alerts per machine-day ({best['false_alerts_per_unit_day']} false)"),
                check("every method, 15.5's detectors included, is scored by incidents, not rows", {"percentile_rules", "ecod", "supervised"} <= set(r["methods"])
                      and all("incident_recall" in m and "alerts_per_unit_day" in m for m in r["methods"].values()),
                      ", ".join(f"{n}: {m['incident_recall']}" for n, m in r["methods"].items())),
                check("machines never seen are scored on the holdout period", (unseen.get("incidents") or 0) > 0,
                      f"{unseen.get('caught')} of {unseen.get('incidents')} incidents on {len(r['held_out_units'])} unseen machines")]


KIND = Streams()
