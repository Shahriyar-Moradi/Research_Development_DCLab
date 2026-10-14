"""Anomaly detection without labels (package 20.7): nothing is labelled, so nothing is learned from an outcome. The kind
runs its own machine (``dclab_core.anomaly``): the rows to review now are the latest days; detectors (rules, an
isolation forest, nearest neighbours, a row against its entity's usual) are fitted on the earlier rows and chosen by
what they find among anomalies planted in a validation period; the chosen one ranks the latest rows into a review list
of the size the reviewers can take. Every confirmed alert becomes a label (``confirm``); with labels the problem is a
rare-event one."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .base import ProblemKind


class AnomalyFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entity_column: str | None = None
    time_column: str | None = None
    key_column: str | None = None
    budget: float | None = Field(default=None, gt=0, description="Rows to review: a share of the rows (below 1) or a count.")


class Anomaly(ProblemKind):
    key = "anomaly"
    title = "Anomaly detection"
    description = "Find the unusual rows when nothing is labelled yet: a ranked list for people to review, whose answers become labels."
    base_task = "binary"
    task_type = "anomaly"
    specificity = 5  # only declared (no words, no pack): a problem with labels is a rare-event one
    packs = ()
    words = ()
    questions = ("Which column says who did it (a courier, a customer, a machine)?", "When is a row known, and when is it reviewed?",
                 "How many rows a day can the people who review them take?")
    metrics = (("planted_recall_at_budget", "Recall at the review budget, on planted anomalies"),)
    baselines = ("rules",)
    checks = ("PIT-005",)
    reliability = ("what the detectors find among planted shapes", "stability across halves of the training rows",
                   "what no column of the table can show")
    evidence_track = "anomaly-and-fraud-detection"
    aliases = ("anomaly detection", "outlier detection", "unsupervised fraud detection", "novelty detection")
    stratify = False
    runs_in_engine = True
    needs_target = False
    Fields = AnomalyFields
    export = {"notebook": False, "service": False}  # the notebook and the service know rows with outcomes, not a review list

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return "planted_recall_at_budget"

    def own_machine(self) -> Any:
        return AnomalyMachine

    def columns_of(self, solution: Any) -> list[str]:
        from .base import _get

        f = _get(solution, "kind_fields") or {}
        return [c for c in (f.get("entity_column"), f.get("time_column"), f.get("key_column")) if c]

    def check_solution(self, solution: Any) -> None:
        if solution.labels not in ("none", "complete"):
            raise ValueError("With labels, even partial ones, the problem is a rare-event one: declare that kind instead")
        if solution.metric and solution.metric != AnomalyMachine.primary_metric:
            raise ValueError(f"The anomaly machine chooses and reports by {AnomalyMachine.primary_metric}: {solution.metric!r} would be ignored")

    def applies(self, solution: Any) -> bool:
        from .base import _get

        return _get(solution, "kind") == self.key  # only declared: an outcome column or a label file is another kind

    def accept_engine(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check

        records = run.records
        final, models = run.evidence("final"), run.records["models"]
        return [check("the engine runs every stage as the anomaly machine", all((records[s] or {}).get("machine") == self.key for s in records),
                      ", ".join(f"{s}: {(records[s] or {}).get('machine')}" for s in records)),
                check("in the engine the detector chosen on validation ranks the review list, read once",
                      final.get("model") == (models.get("decision") or {}).get("chosen") and final.get("holdout_uses_in_this_project") == 1
                      and bool(final.get("review_list")), f"{final.get('model')}, {len(final.get('review_list') or [])} rows listed"),
                check("in the engine the evaluation on planted anomalies and its limits are in the final record",
                      bool(final.get("planted")) and bool(final.get("limits")), f"planted {final.get('planted')}")]

    # ------------------------------------------------------------------ acceptance (the owner's morning, no label)
    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import delivery_fraud

        from .acceptance import Case

        replica = delivery_fraud.generate(seed)
        rides = replica.rides[replica.rides["status"] == "DELIVERED"].reset_index(drop=True)
        truth = replica.truth
        facts = {"fake_trip": sorted(truth.loc[truth["kind"] == "fake_trip", "order_id"].astype(str)),
                 "self_delivery": sorted(truth.loc[truth["kind"] == "self_delivery", "order_id"].astype(str))}
        return Case(rides, None, "binary", "After delivery, in the daily audit of the day's rides; nothing is labelled yet.",
                    solution={"kind": "anomaly", "labels": "none", "time_column": None,
                              "kind_fields": {"entity_column": "biker_id", "time_column": "created_date", "key_column": "order_id"}},
                    facts=facts)

    def solve(self, case: Any) -> Any:
        from dclab_core import anomaly

        f = self.fields(case.solution.get("kind_fields"))  # what the person states; never which rows are the frauds
        columns = anomaly.numeric_columns(case.table, exclude=[f["entity_column"], f["key_column"], "account_id"])
        result = anomaly.evaluate(case.table, columns, f["entity_column"], f["time_column"], f["budget"], f["key_column"])
        return {"result": result, "columns": columns, "facts": case.facts, "table": case.table}

    def accept(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check

        r, f = run["result"], run["facts"]
        held = set(run["table"].loc[r["holdout_index"], "order_id"].astype(str))
        listed = {x["key"] for x in r["review"]["review_list"]}
        fakes, selfs = set(f["fake_trip"]) & held, set(f["self_delivery"]) & held
        rate = len(fakes) / max(len(held), 1)
        precision = len(fakes & listed) / max(len(listed), 1)
        planted = r["review"]["planted"]
        return [check("a ranked review list of the size the budget allows, from the latest days", len(r["review"]["review_list"]) == r["review"]["budget_rows"] > 0
                      and r["split"]["by"] == "day", f"{len(listed)} rows of {len(held)} from {r['split']['holdout_from']}"),
                check("the detector chosen on planted anomalies of the validation period, before the rows to review are read",
                      r["choice"]["chosen"] in r["choice"]["validation"], f"{r['choice']['chosen']}: {r['choice']['validation'][r['choice']['chosen']]}"),
                check("the review list finds the fake trips well above their rate", precision >= 10 * rate,
                      f"{len(fakes & listed)} of {len(listed)} listed are fake trips; their rate is {rate:.4f}"),
                check("its evaluation on planted anomalies of the rows to review, beside the rules", bool(planted) and all(
                    "chosen" in v and "rules" in v for v in planted.values()), str(planted)),
                check("the honest limits stated: what no column shows (self-deliveries) is not claimed", bool(r["review"]["limits"])
                      and len(selfs & listed) <= max(1, len(selfs) // 2), f"{len(selfs & listed)} of {len(selfs)} self-deliveries listed")]


def _key_text(value: Any) -> str:
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def confirm(store: Any, project_id: str, keys: list[str], key_column: str, rejected: list[str] | None = None) -> dict[str, Any]:
    """The review loop: confirmed alerts (their keys) join the project's known cases, kept across rounds; the cases
    become the project's labels (partial: an alert nobody confirmed is unknown, not normal), and the problem is then a
    rare-event one. Rejected alerts are kept as reviewed, never as labels. The review list each round answered is kept
    on the project (attaching labels clears the stages). A refused round changes nothing."""
    import os

    from dclab_rnd.studio import data as studio_data

    project = store.get(project_id)
    table = studio_data.load_table(studio_data.data_path(store, project))
    if key_column not in table.columns:
        raise ValueError(f"{key_column} is not a column of the table")
    held = {_key_text(v) for v in table[key_column].dropna()}
    unknown = [k for k in [*keys, *(rejected or [])] if str(k) not in held]
    if unknown:  # an alert comes from the review list: a key the table does not hold is a typo, refused before anything is written
        raise ValueError(f"Not in the table's {key_column}: {', '.join(map(str, unknown[:5]))}")
    folder = Path(store.data_dir(project_id))
    path = folder / "reviewed_cases.csv"
    previous = pd.read_csv(path, dtype=str) if path.is_file() else pd.DataFrame(columns=[key_column])
    cases = pd.concat([previous, pd.DataFrame({key_column: [str(k) for k in keys]})]).drop_duplicates(subset=[key_column])
    if cases.empty:
        raise ValueError("No case is confirmed yet: a label file needs at least one")
    final = (store.read_stage(project_id, "final") or {}).get("evidence") or {} if hasattr(store, "read_stage") else {}
    review = dict(project.get("review") or {})
    rounds = list(review.get("rounds") or [])
    rounds.append({"listed": [x.get("key") for x in final.get("review_list") or []], "confirmed": [str(k) for k in keys],
                   "rejected": [str(k) for k in rejected or []], "detector": final.get("model")})
    attached = folder / f"reviewed_cases_r{len(rounds)}.csv"  # this round's label file, attached (and registered) under its own name
    cases.to_csv(attached, index=False)
    try:
        studio_data.attach_labels(store, project_id, attached.name, target="is_case", labels="partial")
    except Exception:
        attached.unlink(missing_ok=True)  # a round the label check refuses changes nothing
        raise
    tmp = path.with_suffix(".tmp")
    cases.to_csv(tmp, index=False)
    os.replace(tmp, path)  # the cases kept across rounds, written only once the round is accepted
    project = store.get(project_id)
    suggestion = dict(project.get("suggestion") or {})
    suggestion.update(kind="rare_event", label_provenance=(
        f"{len(cases)} alert(s) of the review list confirmed by a person over {len(rounds)} round(s): they were picked by the "
        f"detector ({final.get('model') or 'the anomaly machine'}) from the latest days, so a model trained and judged on them "
        "partly measures that detector again")[:600])
    budget = ((project.get("solution") or {}).get("kind_fields") or {}).get("budget") or review.get("budget")
    if budget and budget < 1:
        suggestion.setdefault("alert_budget", budget)
    project["suggestion"] = suggestion
    project["review"] = {"key_column": key_column, "rounds": rounds, "confirmed": int(len(cases)), "budget": budget,
                         "review_list": final.get("review_list") or review.get("review_list") or []}
    store.save(project)
    return project


from .machine import Machine, claim  # noqa: E402


class AnomalyMachine(Machine):
    """Anomaly detection in the engine (20.7). The rows to review are the latest days (locked in the data stage);
    detectors are fitted on the earlier rows and chosen on planted anomalies of a validation period; the chosen one
    ranks the latest rows once, in the final stage, into the review list."""

    task_type = "anomaly"
    primary_metric = "planted_recall_at_budget"
    higher_is_better = True

    def __init__(self, kind: Any, solution: Any, frame: Any, project: dict[str, Any]):
        from dclab_core import anomaly

        super().__init__(kind, solution, frame, project)
        f = self.fields
        self.entity, self.time, self.key, self.budget = f["entity_column"], f["time_column"], f["key_column"], f["budget"]
        self.parts = anomaly.split(frame, self.time)
        if len(self.parts["fit"]) < 100 or len(self.parts["validation"]) < 20:
            raise ValueError("Too few rows before the latest days to fit and check a detector (at least 100 and 20)")

    def training_rows(self) -> Any:
        return self.parts["train"]

    def holdout_description(self) -> str:
        sp = self.parts["split"]
        return (f"the {sp['holdout']} rows of the latest days, from {sp['holdout_from']}" if sp["by"] == "day"
                else f"a seeded {sp['holdout']} rows")

    def cv_description(self) -> str:
        sp = self.parts["split"]
        return f"a validation period of {sp['validation']} rows before them, with planted anomalies"

    def _features(self, previous: dict) -> list[str]:
        return list(((previous.get("leakage") or {}).get("evidence") or {}).get("features") or [])

    def data(self, eid: str, previous: dict) -> tuple:
        sp = self.parts["split"]
        evidence = {"train_rows": sp["train"], "holdout_rows": sp["holdout"], "holdout_split": self.holdout_description(),
                    "cv_protocol": self.cv_description(), "split": sp, "source_rows": int(len(self.frame)),
                    "usable_feature_count": int(len(self.frame.columns)), "labels": "none",
                    "notes_for_person": [{"title": "Nothing is labelled: the latest days are the rows to review",
                                          "text": f"The rows to review now are {self.holdout_description()}; the detectors learn what is "
                                                  f"normal from the {sp['train']} rows before them.", "proof": ["DCLAB-R02"]}]}
        claims = [claim(f"{eid}-C1", "fact", f"{sp['train']} training rows, {sp['holdout']} to review ({self.holdout_description()}).", ["evidence.split"])]
        return evidence, claims, f"{sp['train']} training rows, {self.holdout_description()} to review", None

    def leakage(self, eid: str, previous: dict) -> tuple:
        from dclab_core import anomaly

        from dclab_rnd.studio.leakage_review import from_moment

        own = {c for c in (self.entity, self.time, self.key) if c}
        train = self.parts["train"]
        identifiers = [c for c in train.columns if c not in own and pd.api.types.is_integer_dtype(train[c])
                       and train[c].nunique() == len(train)]  # read on the training rows only
        said = {i["column"]: i["reason"] for i in from_moment([{"name": c} for c in self.frame.columns], self.solution.prediction_moment, own)}
        forbidden = [x.column for x in self.solution.forbidden]
        out = sorted(set(identifiers) | set(said) | set(forbidden) | set(self.solution.identifiers))
        features = [c for c in anomaly.numeric_columns(train, exclude=[*own, *out])]
        if not features and not self.entity:
            raise ValueError("No numeric column to read and no entity: a detector has nothing to score in this table")
        why = {**{c: "a different value on every row: an identifier" for c in identifiers}, **said,
               **{c: "forbidden in the solution" for c in forbidden}, **{c: "an identifier in the solution" for c in self.solution.identifiers}}
        evidence = {"kept_out": out, "why": why, "features": features, "declared_leakage_features": out,
                    "notes_for_person": [{"title": f"{len(features)} numeric columns read, {len(out)} kept out",
                                          "text": "Read: " + ", ".join(features) + ". Kept out: " + (", ".join(out) or "none") + ".",
                                          "proof": ["DCLAB-R01"]}]}
        claims = [claim(f"{eid}-C1", "fact", f"Read: {', '.join(features)}. Kept out: {', '.join(out) or 'none'}.", ["evidence.features", "evidence.kept_out"])]
        return evidence, claims, f"{len(features)} columns read, {len(out)} kept out", None

    def features(self, eid: str, previous: dict) -> tuple:
        from dclab_core import anomaly

        columns = self._features(previous)
        built = list(anomaly.Features(columns, self.entity, self.time).fit(self.parts["train"]).scale)
        recipe = "robust_scaled_with_entity_behaviour" if self.entity else "robust_scaled"
        about = (f"{len(built)} features fitted on the training rows: the numbers robust-scaled (skewed ones logged)"
                 + (f", and each {self.entity}'s behaviour (counts per day, trip shapes, repeated places, a row against its usual)" if self.entity else ""))
        evidence = {"selected_recipe": recipe, "best_recipe": recipe, "selected_recipe_description": about, "features": columns, "built": built[:200],
                    "stage_results": [{"recipe": recipe, "description": about, "feature_count_mean": len(built),
                                       "metrics": {self.primary_metric: {"mean": None, "std": None}}}], "selection_rule": "one design for every detector"}
        decision = {"kind": "recipe", "selected": recipe, "chosen": recipe, "rule": evidence["selection_rule"],
                    "options": [{"id": recipe, "label": recipe, "description": about, "features": len(built)}]}
        return evidence, [claim(f"{eid}-C1", "fact", f"The detectors read {about}.", ["evidence.built"])], about, decision

    def models(self, eid: str, previous: dict) -> tuple:
        from dclab_core import anomaly

        picked = anomaly.choose(self.parts, self._features(previous), self.entity, self.time, self.budget)
        v = picked["validation"]
        ranking = sorted(({"model": n, "mean": r["mean"], "std": 0.0, "by_shape": r["recall_at_budget"]} for n, r in v.items()),
                         key=lambda r: -r["mean"])
        best = picked["chosen"]
        evidence = {"selected_model": best, "ranking": ranking, "validation": v, "selection_rule": picked["chosen_by"],
                    "stability": picked["stability"], "budget_rows": picked["budget_rows"], "shapes": picked["shapes"],
                    "notes_for_person": [{"title": f"{best} chosen on planted anomalies of the validation period", "proof": ["DCLAB-R13"],
                                          "text": "Recall at the budget by detector: " + "; ".join(f"{r['model']} {r['mean']}" for r in ranking)
                                                  + f". Stability across halves of the training rows: {picked['stability']}. The rows to review are unread."}]}
        decision = {"kind": "model", "selected": best, "chosen": best, "rule": picked["chosen_by"],
                    "options": [{"id": r["model"], "label": r["model"], "mean": r["mean"], "std": r["std"]} for r in ranking]}
        claims = [claim(f"{eid}-C1", "fact", "Recall at the review budget on planted anomalies, by detector: "
                        + "; ".join(f"{r['model']} {r['mean']}" for r in ranking) + f". Chosen: {best}.", ["evidence.ranking"])]
        return evidence, claims, f"{best} chosen on planted anomalies", decision

    def final(self, eid: str, previous: dict) -> tuple:
        from dclab_core import anomaly

        sp = self.parts["split"]
        locked = ((previous.get("data") or {}).get("evidence") or {}).get("split") or {}
        if (locked.get("holdout_from"), locked.get("holdout")) != (sp["holdout_from"], sp["holdout"]):
            raise ValueError(f"The rows to review were locked as {locked.get('holdout')} rows from {locked.get('holdout_from')}, but the table now "
                             f"gives {sp['holdout']} from {sp['holdout_from']}: rerun the data stage")
        chosen = self.chosen(previous.get("models"))
        out = anomaly.review(self.parts, chosen, self._features(previous), self.entity, self.time, self.budget, self.key)
        planted = out["planted"]
        mine = round(sum(v["chosen"] for v in planted.values()) / max(len(planted), 1), 4)
        rules = round(sum(v["rules"] for v in planted.values()) / max(len(planted), 1), 4)
        cv = next((r["mean"] for r in ((previous.get("models") or {}).get("evidence") or {}).get("ranking", []) if r["model"] == chosen), None)
        evidence = {"model": chosen, "feature_recipe": self.chosen(previous.get("features")),
                    "holdout_metrics": {self.primary_metric: mine},
                    "holdout_primary_metric_ci": {"low": None, "high": None, "point": mine, "method": "not computed: anomalies planted once"},
                    "cv_selected_metric_mean": cv, "cv_to_holdout_gap": round(mine - cv, 4) if cv is not None else None,
                    "baselines": {"rules": {self.primary_metric: rules}}, "review_list": out["review_list"], "budget_rows": out["budget_rows"],
                    "rows_scored": out["rows_scored"], "planted": planted, "limits": out["limits"],
                    "holdout_split": self.holdout_description(), "holdout_rows": sp["holdout"], "train_rows": sp["train"],
                    "selection_rule": "the detector chosen on validation, ranking the rows to review once here",
                    "notes_for_person": [{"title": f"{len(out['review_list'])} rows to review, ranked by {chosen}", "proof": ["DCLAB-R17"],
                                          "text": f"On anomalies planted in a copy of these rows it finds {mine} at the budget (the rules: {rules}). "
                                                  + " ".join(out["limits"])}]}
        claims = [claim(f"{eid}-C1", "fact", f"{chosen} ranks the {out['rows_scored']} rows to review; on planted anomalies it finds {mine} "
                        f"at the budget of {out['budget_rows']} rows, the rules {rules}.", ["evidence.planted", "evidence.review_list"],
                        out["limits"] + ["Not approved for production from this evidence alone."])]
        return evidence, claims, f"{chosen}: {len(out['review_list'])} rows to review", {
            "kind": "final", "selected": "defaults", "chosen": "defaults", "rule": evidence["selection_rule"], "options": []}


KIND = Anomaly()
