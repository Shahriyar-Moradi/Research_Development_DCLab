"""Several tables (package 16.8, in the engine since 16.15 C): an entity table, event tables and tables joined as columns,
related by keys; the target and the features built per entity as of each cutoff. The kind runs its own machine
(``dclab_core.entities``): the project's main table is the entity table, its event tables are related tables attached
by name; rows are (entity, cutoff) pairs, features come only from events before a row's cutoff, the holdout is the
latest cutoff whose horizon the log reaches, and a join that repeats rows is refused."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .base import ProblemKind


class EventTable(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str  # a related table of the project
    time_column: str
    sum_columns: list[str] = Field(default_factory=list, max_length=20)


class EntityTablesFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key_column: str
    target_table: str  # the event table whose absence (or presence) in the horizon is the target
    horizon_days: int = Field(ge=1, le=730)
    absent: bool = True  # True: the target is "no event in the horizon" (churn); False: "an event in the horizon"
    cutoffs: list[str] = Field(min_length=2, max_length=60)
    event_tables: list[EventTable] = Field(default_factory=list, max_length=10)  # 16.15 C: the related tables of events
    joined_tables: list[str] = Field(default_factory=list, max_length=10)  # related tables the person adds as columns (checked)
    data_end: str | None = None  # where the log ends, when the person knows it (else its last event)


class EntityTables(ProblemKind):
    key = "entity_tables"
    title = "Several tables: entity targets as of a cutoff"
    description = "Customers with their orders and tickets in separate tables: will a customer stop, buy or call in the days after a date?"
    base_task = "binary"
    task_type = "entity_tables"
    specificity = 15
    packs = ()
    # several tables or a cutoff must be said: "from the orders table" alone is any other kind's problem
    words = r"\b((?:several|related|separate|multiple) tables|as of (?:a|the|each) (?:cutoff|date)|(?:at|on) each cutoff(?: date)?)\b"
    questions = ("Which column links the tables (a customer id), and which table holds the events that decide the outcome?",
                 "On which dates is the question asked, and how many days after each date decide the answer?",
                 "Does the outcome mean an event happens in those days, or that none does?")
    metrics = (("roc_auc", "ROC-AUC"), ("average_precision", "Average precision"))
    baselines = ("recency_only",)
    checks = ("PIT-005",)
    reliability = ("cutoffs per row", "rows left out because the horizon passes the data's end", "joins checked for repeats")
    evidence_track = "churn-prediction"  # 16.14: the research track whose evidence agents cite for this kind
    aliases = ('relational tables', 'feature aggregation per customer', 'transactions per customer', 'point-in-time features')  # 16.14: the names this kind of problem goes by (read into its evidence record)
    stratify = False
    runs_in_engine = True  # 16.15 C: the engine runs its own machine on the project's related tables
    needs_target = False  # the target is built from the events, per cutoff
    Fields = EntityTablesFields
    export = {"notebook": False, "service": False}

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return "roc_auc"

    def applies(self, solution: Any) -> bool:
        from .base import _get

        return _get(solution, "kind") == self.key

    def own_machine(self) -> Any:
        return EntityMachine

    def columns_of(self, solution: Any) -> list[str]:
        from .base import _get

        return [c for c in [(_get(solution, "kind_fields") or {}).get("key_column")] if c]

    def tables_of(self, solution: Any) -> list[str]:
        from .base import _get

        f = _get(solution, "kind_fields") or {}
        return [*(t["name"] for t in f.get("event_tables") or []), *(f.get("joined_tables") or [])]

    def table_columns_of(self, solution: Any) -> dict[str, list[str]]:
        from .base import _get

        f = _get(solution, "kind_fields") or {}
        return {t["name"]: [f["key_column"], t["time_column"], *t.get("sum_columns", [])] for t in f.get("event_tables") or []} | {
            n: [f["key_column"]] for n in f.get("joined_tables") or []}

    def check_solution(self, solution: Any) -> None:
        k = solution.kind_fields
        names = [t["name"] for t in k.get("event_tables") or []]
        if not names:
            raise ValueError("Name the event tables (orders, tickets...) the features and the target come from")
        if k["target_table"] not in names:
            raise ValueError(f"The target table {k['target_table']!r} must be one of the event tables")
        if set(names) & set(k.get("joined_tables") or []):
            raise ValueError("A table is either events or columns, not both")
        if solution.metric and solution.metric != EntityMachine.primary_metric:
            raise ValueError(f"The entity machine chooses and reports by {EntityMachine.primary_metric}: {solution.metric!r} would be ignored")

    def accept_engine(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check

        records, f = run.records, run.case.facts
        leak, models, final = run.evidence("leakage"), run.records["models"], run.evidence("final")
        return [check("the engine runs every stage as the entity machine", all((records[s] or {}).get("machine") == self.key for s in records),
                      ", ".join(f"{s}: {(records[s] or {}).get('machine')}" for s in records)),
                check("in the engine the entity columns written after the cutoff are kept out and the repeating join refused",
                      set(f["written_later"]) <= set(leak.get("kept_out", [])) and f["repeating_table"] in (leak.get("refused_joins") or {}),
                      f"kept out {leak.get('kept_out')}; refused {list((leak.get('refused_joins') or {}))}"),
                check("in the engine no training feature reads an event after its cutoff, nor anything after the holdout's",
                      run.evidence("features").get("events_after_cutoff_read") == 0 and run.evidence("features").get("rows_that_read_after_the_holdout") == 0,
                      f"{run.evidence('features').get('events_after_cutoff_read')} rows read an event at or after their cutoff; "
                      f"{run.evidence('features').get('rows_that_read_after_the_holdout')} rows differ between the whole log and the log before the holdout"),
                check("in the engine the model chosen on validation is the one the holdout scores, once, and it beats recency",
                      final.get("model") == (models.get("decision") or {}).get("chosen") and final.get("holdout_uses_in_this_project") == 1
                      and bool(final.get("beats_recency")), f"{final.get('model')}: ROC-AUC {final.get('holdout_metrics', {}).get('roc_auc')} "
                      f"against recency's {final.get('baselines', {}).get('recency_only', {}).get('roc_auc')}")]

    # ------------------------------------------------------------------ acceptance (16.2's harness, its own machine)
    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import shop_customers

        from .acceptance import Case

        r = shop_customers.generate(seed)
        f = r.facts
        late = f["late_cutoff"]  # 10 days before the data ends: its 30-day horizon is not known yet
        return Case(r.customers, None, "binary", "On each cutoff date: will the customer place no order in the next 30 days?",
                    solution={"kind": "entity_tables", "kind_fields": {
                        "key_column": f["key"], "target_table": "orders", "horizon_days": f["horizon_days"], "absent": True,
                        "cutoffs": [*f["cutoffs"], late],
                        "event_tables": [{"name": "orders", "time_column": "ordered_at", "sum_columns": ["amount"]},
                                         {"name": "tickets", "time_column": "opened_at"}],
                        "joined_tables": ["addresses"]}},
                    tables={"orders": r.orders, "tickets": r.tickets, "addresses": r.addresses}, facts=f)

    @staticmethod
    def event_specs(fields: dict[str, Any], tables: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {t["name"]: {"frame": tables[t["name"]], "time": t["time_column"], "sums": t.get("sum_columns", [])} for t in fields["event_tables"]}

    def solve(self, case: Any) -> Any:
        from dclab_core import entities

        f = self.fields(case.solution.get("kind_fields"))  # what the person states; never which column leaks
        specs = self.event_specs(f, case.tables or {})
        built = entities.build(case.table, specs, key=f["key_column"], cutoffs=f["cutoffs"], target_table=f["target_table"],
                               horizon_days=f["horizon_days"], absent=f["absent"])
        refused = {}
        for name in f["joined_tables"]:  # a table the person adds as columns: checked first
            try:
                entities.check_join(built["table"], case.tables[name], f["key_column"], name=name)
            except entities.DuplicatingJoin as error:
                refused[name] = str(error)
        result = entities.evaluate(built, key=f["key_column"])
        return {"built": {k: v for k, v in built.items() if k not in ("table", "last_read")}, "evaluation": result, "refused_joins": refused,
                "rows": int(len(built["table"])), "facts": case.facts}

    def accept(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check

        b, e, f = run["built"], run["evaluation"], run["facts"]
        kept_out = b["entity_columns_kept_out"]["after_the_cutoff"]
        return [check("no event after a row's cutoff is read", b["events_after_cutoff_read"] == 0,
                      f"{run['rows']} rows over {len(b['cutoffs'])} cutoffs; events read after a cutoff: {b['events_after_cutoff_read']}"),
                check("entity columns written after the cutoff are kept out", set(f["written_later"]) <= set(kept_out)
                      and not set(f["written_later"]) & set(e["features"]), "; ".join(f"{c}: {b['entity_columns_kept_out']['why'][c]}" for c in kept_out)),
                check("a join that repeats rows is refused, with the numbers", f["repeating_table"] in run["refused_joins"],
                      run["refused_joins"].get(f["repeating_table"], "not refused")),
                check("rows whose horizon passes the data's end are left out, not called negative", b["rows_left_out_unknown"] > 0
                      and e["holdout_cutoff"] < f["late_cutoff"], f"{b['rows_left_out_unknown']} rows at {f['late_cutoff']} left out; "
                      f"the holdout is the latest cutoff whose horizon is known, {e['holdout_cutoff']}"),
                check("training targets read nothing at or after the holdout's cutoff; scored once on the latest", e["train_targets_read_until"] is not None
                      and pd.Timestamp(e["train_targets_read_until"]) < pd.Timestamp(e["holdout_cutoff"]),
                      f"{e['train_rows']} training rows at {', '.join(e['train_cutoffs'])}, their targets read until {e['train_targets_read_until']}; "
                      f"{e['holdout_rows']} held out at {e['holdout_cutoff']}"),
                check("the window features beat recency alone", e["model"]["roc_auc"] > e["recency_only"]["roc_auc"],
                      f"ROC-AUC {e['model']['roc_auc']} against {e['recency_only']['roc_auc']} for days since the last order")]


from .machine import Machine, claim  # noqa: E402


class EntityMachine(Machine):
    """Several tables in the engine (16.15 C). The holdout is the latest cutoff whose horizon the log reaches (locked in
    the data stage from dates alone); every training feature and target is built from the events before it; the model
    is chosen on the latest training cutoff; the holdout's rows are built and scored once, in the final stage."""

    task_type = "entity_tables"
    primary_metric = "roc_auc"
    higher_is_better = True
    uses_tables = True

    def __init__(self, kind: Any, solution: Any, frame: Any, project: dict[str, Any], tables: dict[str, Any] | None = None):
        from dclab_core import entities

        super().__init__(kind, solution, frame, project)
        f = self.fields
        self.tables = tables or {}
        self.key, self.target_table, self.horizon, self.absent = f["key_column"], f["target_table"], f["horizon_days"], f["absent"]
        self.specs = EntityTables.event_specs(f, self.tables)
        self.parts = entities.split(self.specs, cutoffs=f["cutoffs"], target_table=self.target_table, horizon_days=self.horizon,
                                    data_end=f["data_end"])
        self._view = None

    def view(self) -> dict[str, Any]:
        from dclab_core import entities

        if self._view is None:
            self._view = entities.training_view(self.frame, self.specs, self.parts, key=self.key, target_table=self.target_table,
                                                horizon_days=self.horizon, absent=self.absent)
        return self._view

    def training_rows(self) -> Any:
        return self.view()["table"]

    def holdout_description(self) -> str:
        return f"the rows at the latest cutoff whose {self.horizon}-day horizon the log reaches, {self.parts['holdout']}"

    def cv_description(self) -> str:
        return f"the latest training cutoff, {self.parts['validation']}, fitted on the cutoffs whose horizon ends by it"

    def data(self, eid: str, previous: dict) -> tuple:
        v, p = self.view(), self.parts
        rows = v["table"]
        per = {str(c.date()): int(n) for c, n in rows.groupby("cutoff").size().items()}
        evidence = {"split": p, "train_rows": int(len(rows)), "holdout_rows": None, "holdout_split": self.holdout_description(),
                    "cv_protocol": self.cv_description(), "rows_per_cutoff": per, "rows_left_out_unknown": v["rows_left_out_unknown"],
                    "joins": v["joins"], "source_rows": int(len(self.frame)), "usable_feature_count": int(len(rows.columns)),
                    "tables": {n: int(len(t)) for n, t in self.tables.items()},
                    "notes_for_person": [{"title": "One row per customer and cutoff, its future in the next days",
                                          "text": f"{len(rows)} training rows over {len(per)} cutoffs; the holdout is {self.holdout_description()}. "
                                                  f"Left out: {', '.join(p['left_out']) or 'none'} ({p['why_left_out']}).", "proof": ["DCLAB-R02"]}]}
        claims = [claim(f"{eid}-C1", "fact", f"{len(rows)} training rows at {', '.join(p['train'])}; holdout at {p['holdout']}.", ["evidence.split"])]
        return evidence, claims, f"{len(rows)} training rows, holdout at {p['holdout']}", None

    def leakage(self, eid: str, previous: dict) -> tuple:
        from dclab_core import entities

        from dclab_rnd.studio.leakage_review import from_moment

        v = self.view()
        later = v["entity_columns_kept_out"]
        said = {i["column"]: i["reason"] for i in from_moment([{"name": c} for c in self.frame.columns], self.solution.prediction_moment, {self.key})}
        forbidden = [x.column for x in self.solution.forbidden]
        out = sorted(set(later["after_the_cutoff"]) | set(said) | set(forbidden) | (set(self.solution.identifiers) - {self.key}))
        refused, checked = {}, []
        for name in self.fields["joined_tables"]:
            try:
                entities.check_join(v["table"], self.tables[name], self.key, name=name)
                checked.append(name)
            except entities.DuplicatingJoin as error:
                refused[name] = str(error)
        why = {**later["why"], **said, **{c: "forbidden in the solution" for c in forbidden}}
        evidence = {"kept_out": out, "why": why, "refused_joins": refused, "declared_leakage_features": out,
                    "joins_checked_not_joined": checked,  # a table that joins cleanly is not added as columns yet: said, not implied
                    "notes_for_person": [{"level": "warning", "title": f"{c} is kept out", "text": why.get(c, "an identifier"), "proof": ["DCLAB-R01"]}
                                         for c in out] + [{"level": "warning", "title": f"{n} is not joined", "text": t, "proof": ["DCLAB-R09"]}
                                                          for n, t in refused.items()]}
        claims = [claim(f"{eid}-C1", "fact", f"Kept out: {', '.join(out) or 'none'}. Refused joins: {', '.join(refused) or 'none'}. "
                        f"Checked but not joined as columns yet: {', '.join(checked) or 'none'}.",
                        ["evidence.kept_out", "evidence.refused_joins", "evidence.joins_checked_not_joined"])]
        return evidence, claims, f"{len(out)} entity columns kept out, {len(refused)} joins refused", None

    def features(self, eid: str, previous: dict) -> tuple:
        from dclab_core import entities

        v = self.view()
        out = self.kept_out(previous)
        whole = entities.build(self.frame, self.specs, key=self.key, cutoffs=self.parts["train"], target_table=self.target_table,
                               horizon_days=self.horizon, absent=self.absent, entity_columns=v["entity_columns"])["table"]
        mine, windows = v["table"], v["window_columns"]  # the columns the windows made, not every name with a table's prefix
        joined = mine[[self.key, "cutoff", *windows]].merge(whole[[self.key, "cutoff", *windows]], on=[self.key, "cutoff"], suffixes=("", "_w"))
        rows_differ = 0
        if windows:  # NaN on both sides is the same value (no event yet), not a difference
            a, b = joined[windows].to_numpy(dtype=float), joined[[c + "_w" for c in windows]].to_numpy(dtype=float)
            rows_differ = int(((a != b) & ~(np.isnan(a) & np.isnan(b))).any(axis=1).sum())
        features = entities.feature_columns(mine, self.key, out)  # exactly what the models read
        recipe = "windows_7_30_90_with_entity_columns"
        about = f"{len(features)} features: counts, sums, days since the last event and trends per event table, and the numeric entity columns kept"
        evidence = {"selected_recipe": recipe, "best_recipe": recipe, "selected_recipe_description": about, "features": features,
                    "rows_that_read_after_the_holdout": rows_differ, "events_after_cutoff_read": v["events_after_cutoff_read"],
                    "stage_results": [{"recipe": recipe, "description": about, "feature_count_mean": len(features),
                                       "metrics": {self.primary_metric: {"mean": None, "std": None}}}], "selection_rule": "one design for every model"}
        decision = {"kind": "recipe", "selected": recipe, "chosen": recipe, "rule": evidence["selection_rule"],
                    "options": [{"id": recipe, "label": recipe, "description": about, "features": len(features)}]}
        return evidence, [claim(f"{eid}-C1", "fact", f"{about}; {v['events_after_cutoff_read']} rows read an event at or after their cutoff, and "
                                f"built from the whole log or from the log before the holdout, {rows_differ} training rows differ.",
                                ["evidence.events_after_cutoff_read", "evidence.rows_that_read_after_the_holdout"])], about, decision

    @staticmethod
    def kept_out(previous: dict) -> list[str]:
        """What the leakage stage kept out: no model reads it."""
        return list(((previous.get("leakage") or {}).get("evidence") or {}).get("kept_out") or [])

    def models(self, eid: str, previous: dict) -> tuple:
        from dclab_core import entities

        picked = entities.choose(self.view(), self.parts, key=self.key, exclude=self.kept_out(previous))
        v = picked["validation"]
        ranking = sorted(({"model": n, "mean": r["roc_auc"], "std": 0.0, "average_precision": r["average_precision"]} for n, r in v.items()),
                         key=lambda r: -r["mean"])
        best = picked["chosen"]
        evidence = {"selected_model": best, "ranking": ranking, "validation": v, "selection_rule": picked["chosen_by"],
                    "fit_rows": picked["fit_rows"], "validation_rows": picked["validation_rows"],
                    "notes_for_person": [{"title": f"{best} chosen on the validation cutoff", "proof": ["DCLAB-R13"],
                                          "text": "ROC-AUC by model: " + "; ".join(f"{r['model']} {r['mean']}" for r in ranking) + ". The holdout is unread."}]}
        decision = {"kind": "model", "selected": best, "chosen": best, "rule": picked["chosen_by"],
                    "options": [{"id": r["model"], "label": r["model"], "mean": r["mean"], "std": r["std"]} for r in ranking]}
        return evidence, [claim(f"{eid}-C1", "fact", "Validation ROC-AUC: " + "; ".join(f"{r['model']} {r['mean']}" for r in ranking)
                                + f". Chosen: {best}.", ["evidence.ranking"])], f"{best} chosen on the validation cutoff", decision

    def final(self, eid: str, previous: dict) -> tuple:
        from dclab_core import entities

        locked = ((previous.get("data") or {}).get("evidence") or {}).get("split") or {}
        if (locked.get("holdout"), locked.get("data_end")) != (self.parts["holdout"], self.parts["data_end"]):
            raise ValueError(f"The holdout was locked at {locked.get('holdout')}, but the tables now put it at {self.parts['holdout']}: rerun the data stage")
        chosen = self.chosen(previous.get("models"))
        scored = entities.score(self.frame, self.specs, self.view(), self.parts, chosen, key=self.key, target_table=self.target_table,
                                horizon_days=self.horizon, absent=self.absent, exclude=self.kept_out(previous))
        mine, base = scored["holdout"][chosen], scored["holdout"]["recency_only"]
        seen, held = scored["holdout_entities_seen_in_training"], scored["holdout_entities"]
        cv = next((r["mean"] for r in ((previous.get("models") or {}).get("evidence") or {}).get("ranking", []) if r["model"] == chosen), None)
        evidence = {"model": chosen, "feature_recipe": self.chosen(previous.get("features")), "holdout_metrics": mine,
                    "holdout_primary_metric_ci": {"low": None, "high": None, "point": mine["roc_auc"], "method": "not computed: one holdout cutoff"},
                    "cv_selected_metric_mean": cv, "cv_to_holdout_gap": round(mine["roc_auc"] - cv, 4) if cv is not None else None,
                    "baselines": {"recency_only": base}, "beats_recency": scored["beats_recency"], "positive_rate": scored["positive_rate"],
                    "holdout_split": self.holdout_description(), "holdout_rows": scored["holdout_rows"], "train_rows": scored["train_rows"],
                    "holdout_entities": held, "holdout_entities_seen_in_training": seen,
                    "kept_but_written_later_at_the_holdout": scored["kept_but_written_later_at_the_holdout"],
                    "selection_rule": "the model chosen on validation, read once here",
                    "notes_for_person": [{"title": f"Holdout ROC-AUC {mine['roc_auc']} for {chosen}", "proof": ["DCLAB-R17"],
                                          "text": f"Recency alone: {base['roc_auc']}. {seen} of the {held} holdout customers also appear in "
                                                  "training: the holdout is mostly the same customers at a later cutoff."}]
                    + [{"level": "warning", "title": f"{c} may tell the holdout's future", "proof": ["DCLAB-R01"],
                        "text": "Kept in training (nothing before the holdout showed it written later), it follows the log after the "
                                "holdout's cutoff: the holdout score may be optimistic. Forbid it and rerun if it is written at export."}
                       for c in scored["kept_but_written_later_at_the_holdout"]]}
        claims = [claim(f"{eid}-C1", "fact", f"At the holdout cutoff {self.parts['holdout']}, {chosen} scores ROC-AUC {mine['roc_auc']} against "
                        f"recency's {base['roc_auc']}.", ["evidence.holdout_metrics", "evidence.baselines"],
                        [f"{seen} of the {held} holdout customers also appear in training: a split in time, not across customers.",
                         *(f"{c} follows the log after the holdout's cutoff: the score may be optimistic."
                           for c in scored["kept_but_written_later_at_the_holdout"]),
                         "Not approved for production from this evidence alone."])]
        return evidence, claims, f"{chosen}: holdout ROC-AUC {mine['roc_auc']}", {
            "kind": "final", "selected": "defaults", "chosen": "defaults", "rule": evidence["selection_rule"], "options": []}


KIND = EntityTables()
