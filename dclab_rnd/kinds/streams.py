"""Streams and sensor anomalies (package 16.11): readings over time per unit (machines, sessions), rare incidents, and
alerts judged by the incidents they catch. The kind runs its own machine in the engine since 16.15 C
(``dclab_core.streams``): the readings are the project's table and the incidents a related table; windows over each
unit's past, labels from the incidents after them, detectors and a classifier with an alert budget, and metrics per
incident. The holdout is the latest windows, cut from their dates alone; training reads only the readings before the cut
and the incidents known by it."""

from __future__ import annotations

from typing import Any

import numpy as np
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
    incidents_table: str = "incidents"  # 16.15 C: the project's related table of incidents
    confirmed_sensors: list[str] = Field(default_factory=list, max_length=50)  # sensors a person confirmed are read when the window ends


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
    aliases = ('predictive maintenance', 'sensor anomaly detection', 'machine alerts', 'time series anomalies')  # 16.14: the names this kind of problem goes by (read into its evidence record)
    stratify = False
    runs_in_engine = True  # 16.15 C: the engine runs its own machine, the incidents a related table
    needs_target = False  # the target is built from the incidents, per window
    Fields = StreamFields
    export = {"notebook": False, "service": False}

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return "incident_recall"

    def applies(self, solution: Any) -> bool:
        from .base import _get

        return _get(solution, "kind") == self.key

    def own_machine(self) -> Any:
        return StreamMachine

    def columns_of(self, solution: Any) -> list[str]:
        from .base import _get

        f = _get(solution, "kind_fields") or {}
        return [c for c in [f.get("unit_column"), f.get("time_column"), *(f.get("sensor_columns") or [])] if c]

    def tables_of(self, solution: Any) -> list[str]:
        from .base import _get

        return [((_get(solution, "kind_fields") or {}).get("incidents_table") or "incidents")]

    def table_columns_of(self, solution: Any) -> dict[str, list[str]]:
        from .base import _get

        f = _get(solution, "kind_fields") or {}
        return {f.get("incidents_table") or "incidents": [f["unit_column"], f["incident_time_column"]]}

    def check_solution(self, solution: Any) -> None:
        k = solution.kind_fields
        sensors = k["sensor_columns"]
        if len(set(sensors)) != len(sensors) or {k["unit_column"], k["time_column"]} & set(sensors) or k["unit_column"] == k["time_column"]:
            raise ValueError("The unit, the time and the sensor columns must be distinct")
        bad = set(sensors) & ({f.column for f in solution.forbidden} | set(solution.identifiers))
        if bad:
            raise ValueError(f"A forbidden column cannot be a sensor the windows read: {', '.join(sorted(bad))}")
        if solution.metric and solution.metric != StreamMachine.primary_metric:
            raise ValueError(f"The streams machine chooses and reports by {StreamMachine.primary_metric}: {solution.metric!r} would be ignored")

    def accept_engine(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check

        records = run.records
        feats, models, final = run.evidence("features"), records["models"], run.evidence("final")
        fc = feats.get("future_check") or {}
        best = (final.get("holdout_metrics") or {})
        budget = final.get("budget_per_unit_day") or 0
        unseen = final.get("unseen_units") or {}
        return [check("the engine runs every stage as the streams machine", all((records[s] or {}).get("machine") == self.key for s in records),
                      ", ".join(f"{s}: {(records[s] or {}).get('machine')}" for s in records)),
                check("in the engine no training window reads a reading after the holdout's start, and none moves when later readings change",
                      feats.get("readings_after_the_cut_read") == 0 and fc.get("windows_before_the_cut", 0) > 0 and fc.get("changed_by_the_future") == 0,
                      f"{feats.get('readings_after_the_cut_read')} windows read after the cut; {fc.get('changed_by_the_future')} of "
                      f"{fc.get('windows_before_the_cut')} windows moved"),
                check("in the engine thresholds and the method are set on validation, and the holdout scores that method once",
                      final.get("model") == (models.get("decision") or {}).get("chosen") and final.get("holdout_uses_in_this_project") == 1,
                      f"chosen {(models.get('decision') or {}).get('chosen')}, scored {final.get('model')}"),
                check("in the engine most holdout incidents are caught, with hours of warning, within the alert budget",
                      (best.get("incident_recall") or 0) >= 0.75 and (best.get("lead_time_hours_median") or 0) >= 6
                      and (best.get("alerts_per_unit_day") or 99) <= 1.5 * budget,
                      f"{best.get('caught')} of {best.get('incidents')} incidents, median {best.get('lead_time_hours_median')} h before, "
                      f"{best.get('alerts_per_unit_day')} alerts per machine-day"),
                check("in the engine machines never seen are scored on the holdout period", (unseen.get("incidents") or 0) > 0,
                      f"{unseen.get('caught')} of {unseen.get('incidents')} incidents on unseen machines")]

    # ------------------------------------------------------------------ acceptance (16.2's harness, its own machine)
    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import machine_sensors

        from .acceptance import Case

        r = machine_sensors.generate(seed)
        f = r.facts
        return Case(r.readings, None, "binary", "Every six hours, for each machine: raise an alert now?",
                    solution={"kind": "streams", "kind_fields": {"unit_column": f["unit"], "time_column": f["time"], "sensor_columns": f["sensors"],
                                                                 "incident_time_column": f["incident_time"]}},
                    tables={"incidents": r.incidents}, facts=f)

    def solve(self, case: Any) -> Any:
        from dclab_core import streams

        from dclab_rnd.expansion import detectors

        f = self.fields(case.solution.get("kind_fields"))  # what the person states; never the warning signs
        result = streams.evaluate(case.table, case.tables[f["incidents_table"]], unit=f["unit_column"], time=f["time_column"], sensors=f["sensor_columns"],
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


from .machine import Machine, claim  # noqa: E402


def _detectors() -> dict[str, Any]:
    from dclab_rnd.expansion import detectors

    return {"percentile_rules": detectors.PercentileRules, "ecod": detectors.ECOD}  # 15.5's, per window


class StreamMachine(Machine):
    """Streams in the engine (16.15 C). The holdout is the latest windows, cut from their dates in the data stage; every
    training window reads the readings before the cut and is labelled by the incidents known by it; the method and its
    alert threshold are set on the validation period; the holdout's windows are built and scored once, in the final stage."""

    task_type = "streams"
    primary_metric = "incident_recall"
    higher_is_better = True
    uses_tables = True

    def __init__(self, kind: Any, solution: Any, frame: Any, project: dict[str, Any], tables: dict[str, Any] | None = None):
        from dclab_core import streams

        super().__init__(kind, solution, frame, project)
        f = self.fields
        self.incidents = (tables or {})[f["incidents_table"]]
        self.unit, self.time, self.sensors, self.incident_time = f["unit_column"], f["time_column"], f["sensor_columns"], f["incident_time_column"]
        self.parts = streams.split(frame, unit=self.unit, time=self.time, sensors=self.sensors, window_hours=f["window_hours"],
                                   horizon_hours=f["horizon_hours"], label_delay_hours=f["label_delay_hours"])
        self._view = None

    def view(self) -> dict[str, Any]:
        from dclab_core import streams

        if self._view is None:
            self._view = streams.training_view(self.frame, self.incidents, self.parts, unit=self.unit, time=self.time, sensors=self.sensors,
                                               incident_time=self.incident_time)
        return self._view

    def known_incidents(self) -> Any:
        from dclab_core import streams

        return streams.known_by(self.frame, self.incidents, self.parts, time=self.time, incident_time=self.incident_time)[1]

    def training_rows(self) -> Any:
        v = self.view()
        return pd.concat([v["fit"], v["validation"]], ignore_index=True)

    def holdout_description(self) -> str:
        return f"the windows ending after {self.parts['cut']}, the latest 30% of window ends"

    def cv_description(self) -> str:
        return (f"fitted on the windows whose label is known by {self.parts['validation_from']}, thresholds and the method set on the windows "
                f"after it, up to the holdout's start")

    def data(self, eid: str, previous: dict) -> tuple:
        v, p = self.view(), self.parts
        rows = len(v["fit"]) + len(v["validation"])
        evidence = {"split": p, "train_rows": rows, "fit_windows": len(v["fit"]), "validation_windows": len(v["validation"]), "holdout_rows": None,
                    "holdout_split": self.holdout_description(), "cv_protocol": self.cv_description(), "source_rows": int(len(self.frame)),
                    "usable_feature_count": len(v["features"]), "incidents_known_by_the_cut": v["incidents_known"],
                    "units": int(self.frame[self.unit].nunique()),
                    "notes_for_person": [{"title": "Windows over each machine's past, the latest held out", "proof": ["DCLAB-R02"],
                                          "text": f"{rows} training windows; the holdout is {self.holdout_description()}. "
                                                  f"{v['incidents_known']} incidents are known by its start."}]}
        claims = [claim(f"{eid}-C1", "fact", f"{rows} training windows before {p['cut']}; {v['incidents_known']} incidents known by then.", ["evidence.split"])]
        return evidence, claims, f"{rows} training windows, holdout after {p['cut']}", None

    def leakage(self, eid: str, previous: dict) -> tuple:
        from dclab_core import streams

        not_read = [c for c in self.frame.columns if c not in (self.unit, self.time, *self.sensors)]
        forbidden = [x.column for x in self.solution.forbidden]
        delay = self.parts["label_delay_hours"]
        ahead = streams.written_ahead(self.view(), self.known_incidents(), unit=self.unit, incident_time=self.incident_time, sensors=self.sensors,
                                      horizon_hours=self.parts["horizon_hours"])
        held = [c for c in ahead["to_confirm"] if c not in self.fields["confirmed_sensors"]]  # kept out until a person confirms them
        evidence = {"kept_out": sorted(set(not_read) | set(forbidden) | set(held)), "declared_leakage_features": sorted(set(forbidden) & set(self.frame.columns)),
                    "sensors_kept_out": held, "sensors_to_confirm": ahead["to_confirm"],
                    "why": {**{c: "not a sensor the person named: the windows never read it" for c in not_read}, **ahead["why"]},
                    "label_delay_hours": delay, "train_labels_until": self.parts["train_labels_until"],
                    "notes_for_person": [{"title": "Only the named sensors are read, each window from its own past", "proof": ["DCLAB-R01"],
                                          "text": f"Not read: {', '.join(not_read) or 'nothing else'}. A training window counts only when its "
                                                  f"{self.parts['horizon_hours']:g}-hour horizon and the {delay:g}-hour delay before an incident is "
                                                  "known end by the holdout's start."}]
                    + [{"level": "warning", "title": f"{c} is kept out until confirmed", "proof": ["DCLAB-R01"],
                        "text": ahead["why"][c] + ". Confirm it is read when the window ends (confirmed_sensors) to use it."} for c in held]}
        claims = [claim(f"{eid}-C1", "fact", f"The windows read {', '.join(c for c in self.sensors if c not in held)} only; kept out until confirmed: "
                        f"{', '.join(held) or 'none'}; training labels read until {self.parts['train_labels_until']}.",
                        ["evidence.kept_out", "evidence.sensors_kept_out", "evidence.train_labels_until"])]
        return evidence, claims, f"{len(self.sensors) - len(held)} sensors read, {len(held)} kept out to confirm", None

    def used(self, previous: dict) -> dict[str, Any]:
        """The training view without the sensors the leakage stage kept out."""
        from dclab_core import streams

        held = ((previous.get("leakage") or {}).get("evidence") or {}).get("sensors_kept_out") or []
        return streams.without(self.view(), held)

    def features(self, eid: str, previous: dict) -> tuple:
        from dclab_core import streams

        v = self.used(previous)
        readings, _ = streams.known_by(self.frame, self.incidents, self.parts, time=self.time, incident_time=self.incident_time)
        rows = self.training_rows()
        fc = streams._future_check(readings, rows, unit=self.unit, time=self.time, sensors=self.sensors,
                                   window_hours=self.parts["window_hours"], step_hours=self.parts["step_hours"])
        recipe = "windows_mean_spread_last_slope_vs_own"
        about = f"{len(v['features'])} features per window: mean, spread, last value and slope per sensor, and the change against the machine's own past"
        evidence = {"selected_recipe": recipe, "best_recipe": recipe, "selected_recipe_description": about, "features": v["features"],
                    "readings_after_the_cut_read": v["readings_after_the_cut_read"], "future_check": fc,
                    "stage_results": [{"recipe": recipe, "description": about, "feature_count_mean": len(v["features"]),
                                       "metrics": {self.primary_metric: {"mean": None, "std": None}}}], "selection_rule": "one design for every method"}
        decision = {"kind": "recipe", "selected": recipe, "chosen": recipe, "rule": evidence["selection_rule"],
                    "options": [{"id": recipe, "label": recipe, "description": about, "features": len(v["features"])}]}
        return evidence, [claim(f"{eid}-C1", "fact", f"{about}; with every reading after {fc['cut']} changed, {fc['changed_by_the_future']} of "
                                f"{fc['windows_before_the_cut']} earlier windows move.", ["evidence.future_check"])], about, decision

    def models(self, eid: str, previous: dict) -> tuple:
        from dclab_core import streams

        f = self.fields
        picked = streams.choose(self.used(previous), self.known_incidents(), self.parts, unit=self.unit, incident_time=self.incident_time,
                                budget_per_unit_day=f["budget_per_unit_day"], detectors=_detectors())
        v = picked["validation"]
        ranking = sorted(({"model": n, "mean": m["incident_recall"], "std": 0.0, "false_alerts_per_unit_day": m["false_alerts_per_unit_day"],
                           "lead_time_hours_median": m["lead_time_hours_median"]} for n, m in v.items()),
                         key=lambda r: (-(r["mean"] or 0), r["false_alerts_per_unit_day"] or 0))
        best = picked["chosen"]
        evidence = {"selected_model": best, "ranking": ranking, "validation": v, "selection_rule": picked["chosen_by"],
                    "fit_windows": picked["fit_windows"], "validation_windows": picked["validation_windows"],
                    "notes_for_person": [{"title": f"{best} chosen on the validation period", "proof": ["DCLAB-R13"],
                                          "text": "Incident recall by method: " + "; ".join(f"{r['model']} {r['mean']}" for r in ranking) + ". The holdout is unread."}]}
        decision = {"kind": "model", "selected": best, "chosen": best, "rule": picked["chosen_by"],
                    "options": [{"id": r["model"], "label": r["model"], "mean": r["mean"], "std": r["std"]} for r in ranking]}
        return evidence, [claim(f"{eid}-C1", "fact", "Validation incident recall: " + "; ".join(f"{r['model']} {r['mean']}" for r in ranking)
                                + f". Chosen: {best}.", ["evidence.ranking"])], f"{best} chosen on the validation period", decision

    def final(self, eid: str, previous: dict) -> tuple:
        from dclab_core import streams

        f = self.fields
        locked = ((previous.get("data") or {}).get("evidence") or {}).get("split") or {}
        if (locked.get("cut"), locked.get("validation_from")) != (self.parts["cut"], self.parts["validation_from"]):
            raise ValueError(f"The holdout was locked at {locked.get('cut')}, but the readings now put it at {self.parts['cut']}: rerun the data stage")
        chosen = self.chosen(previous.get("models"))
        scored = streams.score(self.frame, self.incidents, self.used(previous), self.parts, chosen, unit=self.unit, time=self.time, sensors=self.sensors,
                               incident_time=self.incident_time, budget_per_unit_day=f["budget_per_unit_day"], detectors=_detectors())
        mine = scored["holdout"][chosen]
        base = {n: m for n, m in scored["holdout"].items() if n != chosen}
        cv = next((r["mean"] for r in ((previous.get("models") or {}).get("evidence") or {}).get("ranking", []) if r["model"] == chosen), None)
        unseen = scored["unseen_units"] or {}
        low, high = streams.wilson(mine["caught"], mine["incidents"])
        recall = mine["incident_recall"]
        evidence = {"model": chosen, "feature_recipe": self.chosen(previous.get("features")), "holdout_metrics": mine,
                    "holdout_primary_metric_ci": {"low": low, "high": high, "point": recall, "method": f"Wilson 95% on {mine['incidents']} incidents"},
                    "cv_selected_metric_mean": cv, "cv_to_holdout_gap": round(recall - cv, 4) if cv is not None and recall is not None else None,
                    "baselines": base, "budget_per_unit_day": f["budget_per_unit_day"], "unseen_units": scored["unseen_units"],
                    "held_out_units": scored["held_out_units"], "unseen_fitted_on_units": scored["unseen_fitted_on_units"], "holdout_split": self.holdout_description(), "holdout_rows": scored["holdout_windows"],
                    "train_rows": len(self.training_rows()), "selection_rule": "the method and threshold set on validation, read once here",
                    "notes_for_person": [{"title": f"{mine['caught']} of {mine['incidents']} holdout incidents caught by {chosen}", "proof": ["DCLAB-R17"],
                                          "text": f"Median {mine['lead_time_hours_median']} hours of warning, {mine['alerts_per_unit_day']} alerts per machine-day "
                                                  f"({mine['false_alerts_per_unit_day']} false) against a budget of {f['budget_per_unit_day']:g}. "
                                                  f"Machines never seen: {unseen.get('caught')} of {unseen.get('incidents')} incidents."}]}
        said = (f"On the holdout period {chosen} caught {mine['caught']} of {mine['incidents']} incidents (95% {low} to {high}), median "
                f"{mine['lead_time_hours_median']} hours before, at {mine['alerts_per_unit_day']} alerts per machine-day." if mine["incidents"] else
                f"No holdout incident has its whole warning horizon in the holdout period: {chosen} raised {mine['alerts_per_unit_day']} alerts "
                "per machine-day, and its recall cannot be read.")
        claims = [claim(f"{eid}-C1", "fact", said, ["evidence.holdout_metrics", "evidence.holdout_primary_metric_ci"],
                        [f"{mine['incidents']} incidents: a recall read on few events.", "How long after an incident starts it is detected is not measured.",
                         "Not approved for production from this evidence alone."])]
        return evidence, claims, f"{chosen}: {mine['caught']} of {mine['incidents']} incidents" if mine["incidents"] else f"{chosen}: no holdout incident to score", {
            "kind": "final", "selected": "defaults", "chosen": "defaults", "rule": evidence["selection_rule"], "options": []}


KIND = Streams()
