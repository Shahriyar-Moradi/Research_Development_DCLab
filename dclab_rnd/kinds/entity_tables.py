"""Several tables (package 16.8): an entity table, event tables and label tables related by keys; the target and the
features built per entity as of each cutoff. The kind runs its own machine (``dclab_core.entities``), not the five
tabular stages, because a project holds one table today: rows are (entity, cutoff) pairs, features come only from
events before a row's cutoff, and a join that repeats rows is refused."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .base import ProblemKind


class EntityTablesFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key_column: str
    target_table: str  # the event table whose absence (or presence) in the horizon is the target
    horizon_days: int = Field(ge=1, le=730)
    absent: bool = True  # True: the target is "no event in the horizon" (churn); False: "an event in the horizon"
    cutoffs: list[str] = Field(min_length=2, max_length=60)


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
    runs_in_engine = False  # a project holds one table: dclab_core.entities runs it as a library
    Fields = EntityTablesFields

    def applies(self, solution: Any) -> bool:
        from .base import _get

        return _get(solution, "kind") == self.key

    # ------------------------------------------------------------------ acceptance (16.2's harness, its own machine)
    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import shop_customers

        from .acceptance import Case

        r = shop_customers.generate(seed)
        f = r.facts
        late = f["late_cutoff"]  # 10 days before the data ends: its 30-day horizon is not known yet
        return Case(r.customers, "target", "binary", "On each cutoff date: will the customer place no order in the next 30 days?",
                    solution={"kind_fields": {"key_column": f["key"], "target_table": "orders", "horizon_days": f["horizon_days"], "absent": True,
                                              "cutoffs": [*f["cutoffs"], late]},
                              "tables": {"orders": {"frame": r.orders, "time": "ordered_at", "sums": ["amount"]},
                                         "tickets": {"frame": r.tickets, "time": "opened_at"}},
                              "other_tables": {"addresses": r.addresses}}, facts=f)

    def solve(self, case: Any) -> Any:
        from dclab_core import entities

        f = self.fields(case.solution.get("kind_fields"))  # what the person states; never which column leaks
        built = entities.build(case.table, case.solution["tables"], key=f["key_column"], cutoffs=f["cutoffs"], target_table=f["target_table"],
                               horizon_days=f["horizon_days"], absent=f["absent"])
        refused = {}
        for name, frame in (case.solution.get("other_tables") or {}).items():  # a table the person adds as columns: checked first
            try:
                entities.check_join(built["table"], frame, f["key_column"], name=name)
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


KIND = EntityTables()
