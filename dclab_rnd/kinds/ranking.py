"""Ranking and recommendation (package 16.5): users, items and interactions with time; what to show each user next.
Models read the interactions up to the recommendation moment and the items that existed at it, and are scored on the
window after it against popularity. The engine runs it as its own machine (16.15 B, ``RankingMachine``): the holdout is
the window after the moment, locked in the data stage; the items' columns are checked on a cut that ends at the moment;
the model is chosen at the validation origin one window earlier; the window after the moment is read once, in the final
stage. The items catalogue is the project's related table ``items``."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .base import ProblemKind, _get
from .machine import Machine, claim


class RankingFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_column: str
    item_column: str
    time_column: str
    event_column: str
    relevant_event: str
    moment: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}")  # the recommendation moment: a date (and time) in the log's span
    window_days: int = Field(default=14, ge=1, le=365)
    k: int = Field(default=10, ge=1, le=100)
    created_column: str | None = None
    category_column: str | None = None
    items_table: str = "items"  # 16.15 B: the related table of the items, attached by name


class Ranking(ProblemKind):
    key = "ranking"
    title = "Ranking and recommendation"
    description = "Which items to show each user next, from what users viewed and bought before."
    base_task = "binary"
    task_type = "ranking"
    specificity = 15
    packs = ()
    # an item object is required: "recommend a reply" or "recommend whether to approve" is another kind's problem
    words = (r"\b(recommend\w* (?:\w+ ){0,3}(?:items|products|articles|movies|films|songs|courses|books|content|listings)"
             r"|recommendations? (?:for|to) each (?:user|customer|shopper|reader|viewer)|ranking (?:items|products)"
             r"|(?:items|products) to show)\b")
    questions = ("Which columns name the user, the item, the time and the kind of interaction (a view, a purchase)?",
                 "Which interaction counts as a success (a purchase, a click), and when is the list shown?",
                 "How many items does the list show (k), and over how many days after it is success counted?")
    metrics = (("ndcg_at_k", "NDCG at k"), ("recall_at_k", "Recall at k"), ("coverage", "Catalogue coverage"))
    baselines = ("popularity", "recent_popularity")
    checks = ("PIT-005",)
    reliability = ("users with and without a history", "coverage and novelty", "items created after the moment")
    evidence_track = "recommender-systems"
    aliases = ('recommender systems', 'recommendations', 'learning to rank', 'top-k lists', 'offline evaluation')  # 16.14: the names this kind of problem goes by (read into its evidence record)
    stratify = False
    runs_in_engine = True  # 16.15 B: its own machine, stage by stage
    needs_target = False  # the outcome is an event in the interactions, read in the window after the moment
    export = {"notebook": False, "service": False}  # lists per user: the tabular exports do not fit it
    Fields = RankingFields

    def own_machine(self) -> Any:
        return RankingMachine

    def columns_of(self, solution: Any) -> list[str]:
        f = self.fields(_get(solution, "kind_fields"))
        return [f["user_column"], f["item_column"], f["time_column"], f["event_column"]]

    def tables_of(self, solution: Any) -> list[str]:
        return [self.fields(_get(solution, "kind_fields"))["items_table"]]

    def table_columns_of(self, solution: Any) -> dict[str, list[str]]:
        f = self.fields(_get(solution, "kind_fields"))
        return {f["items_table"]: [c for c in (f["item_column"], f["created_column"], f["category_column"]) if c]}

    def check_solution(self, solution: Any) -> None:
        f = self.fields(solution.kind_fields)
        names = [f["user_column"], f["item_column"], f["time_column"], f["event_column"]]
        if len(set(names)) != len(names):
            raise ValueError("The user, the item, the time and the event must be four different columns")
        if solution.metric and solution.metric not in (m for m, _ in self.metrics):
            raise ValueError(f"The ranking machine is judged by {', '.join(m for m, _ in self.metrics)}, not {solution.metric}")

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return "ndcg_at_k"

    def applies(self, solution: Any) -> bool:
        return _get(solution, "kind") == self.key

    # ------------------------------------------------------------------ acceptance (16.2's harness, its own machine)
    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import shop_interactions

        from .acceptance import Case

        replica = shop_interactions.generate(seed)
        f = replica.facts
        fields = {"user_column": f["user"], "item_column": f["item"], "time_column": f["time"], "event_column": f["event"],
                  "relevant_event": f["relevant"], "moment": f["moment"], "window_days": f["window_days"], "k": f["k"],
                  "created_column": f["created"], "category_column": "category"}
        return Case(replica.interactions, None, "binary", f"On {f['moment']}, the list each customer sees for the next two weeks.",
                    solution={"kind": "ranking", "kind_fields": fields}, tables={"items": replica.items}, facts=f)

    def solve(self, case: Any) -> Any:
        from dclab_core import ranking

        f = self.fields(case.solution.get("kind_fields"))  # what the person states; never which column leaks
        items: pd.DataFrame = case.tables["items"]
        common = dict(user=f["user_column"], item=f["item_column"], time=f["time_column"], event=f["event_column"],
                      relevant=f["relevant_event"], moment=f["moment"])
        check = ranking.item_check(items, case.table, **common, created=f["created_column"])
        numeric = [c for c in items.columns if c not in (f["item_column"], f["created_column"], f["category_column"])
                   and pd.api.types.is_numeric_dtype(items[c])]
        allowed = [c for c in numeric if c not in check["with_the_future"]]
        result = ranking.evaluate(case.table, items, **common, window_days=f["window_days"], k=f["k"], created=f["created_column"],
                                  category=f["category_column"], item_features=allowed)
        created = items.set_index(f["item_column"])[f["created_column"]].astype(str).to_dict() if f["created_column"] else {}
        return {"items_check": check, "evaluation": result, "created": created, "facts": case.facts}

    def accept(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check

        r, c, f = run["evaluation"], run["items_check"], run["facts"]
        k, best, s = r["k"], r["chosen"], r["scores"]
        moment = pd.Timestamp(r["moment"])
        trained = r["ranker_trained_on"]
        window_start = moment - pd.Timedelta(days=r["window_days"])
        read_ok = (not r["read_after_moment"] and pd.Timestamp(r["read_until"]) <= moment and trained.get("why_not") is None
                   and pd.Timestamp(trained["features_until"]) <= window_start and pd.Timestamp(trained["labels_until"]) <= moment)
        created = {i: pd.Timestamp(d) for i, d in run["created"].items()}
        late = sum(1 for lists in r["lists"].values() for ranked in lists.values() for x in ranked if x in created and created[x] > moment)
        return [check("no interaction after the moment is read; the ranker's features end before its training window", read_ok,
                      f"last read {r['read_until']}; the ranker's features until {trained.get('features_until')}, its labels "
                      f"{trained.get('labels_from')} to {trained.get('labels_until')}" if read_ok or trained.get("why_not") is None else trained["why_not"]),
                check(f"`{f['leak']}` is kept out: counted after the moment", f["leak"] in c["with_the_future"] and f["leak"] not in r["item_features"],
                      c["why"].get(f["leak"], "not found")),
                check("no item created after the moment is recommended", c["created_after"] > 0 and late == 0,
                      f"{c['created_after']} items created after the moment; {late} of them in a list"),
                check("every model, the ranker included, is compared with popularity", {"popularity", "recent_popularity", "mf", "ranker"} <= set(s)
                      and all("against_popularity" in v for v in s.values()), ", ".join(f"{m}: NDCG@{k} {v[f'ndcg@{k}']}" for m, v in s.items())),
                check("the model chosen before the moment's window beats popularity, its interval above zero", best in s and best != "popularity"
                      and r["beats_popularity"].get(best, False),
                      f"{best} (chosen at {r['validation']['origin']}): gain {s.get(best, {}).get('against_popularity', {}).get('ndcg_gain')}, "
                      f"90% interval {s.get(best, {}).get('against_popularity', {}).get('ci90')}"),
                check("NDCG and recall at k, coverage and novelty, with and without a history", all(
                    v.get(f"ndcg@{k}") is not None and v.get("coverage") is not None and v.get("novelty") is not None for v in s.values()),
                      f"coverage {s.get(best, {}).get('coverage')}, users with a history {s.get(best, {}).get('with_history', {}).get('users')}")]


    def accept_engine(self, run: Any) -> list[dict[str, Any]]:
        """The decisions on the engine's records of the same case (16.15 B)."""
        from .acceptance import check

        records, f = run.records, run.case.facts
        leakage, models, final = (records[k].get("evidence") or {} for k in ("leakage", "models", "final"))
        chosen, k = (records["models"].get("decision") or {}).get("chosen"), f["k"]
        mine = (final.get("scores") or {}).get(chosen) or {}
        trained = final.get("ranker_trained_on") or {}
        return [check("every stage ran on the ranking machine", all(records[s].get("machine") == self.key for s in records),
                      str({s: records[s].get("machine") for s in records})),
                check(f"`{f['leak']}` is kept out: counted after the moment, found on a cut that ends at it",
                      f["leak"] in leakage.get("with_the_future", []) and f["leak"] not in (records["features"].get("evidence") or {}).get("features", []),
                      (leakage.get("why") or {}).get(f["leak"], "not found")),
                check("nothing after the moment is read before the final stage; the ranker's features end before its window",
                      not final.get("read_after_moment") and trained.get("why_not") is None, f"last read {final.get('read_until')}"),
                check("no item created after the moment is recommended by the model chosen",
                      (final.get("created_after_moment") or 0) > 0 and (final.get("created_after_recommended") or {}).get(chosen) == 0,
                      f"{final.get('created_after_moment')} created after; {(final.get('created_after_recommended') or {}).get(chosen)} recommended"),
                check("the model chosen at the validation origin is the final one, read once, and beats popularity with its interval above zero",
                      final.get("model") == chosen and final.get("holdout_uses_in_this_project") == 1 and bool((final.get("beats_popularity") or {}).get(chosen)),
                      f"{chosen}: gain {(mine.get('against_popularity') or {}).get('ndcg_gain')}, 90% {(mine.get('against_popularity') or {}).get('ci90')}"),
                check("NDCG and recall at k, coverage and novelty, with and without a history", all(
                    v.get(f"ndcg@{k}") is not None and v.get("coverage") is not None for v in (final.get("scores") or {}).values()),
                      f"coverage {mine.get('coverage')}")]


class RankingMachine(Machine):
    """Ranking in the engine (16.15 B). The holdout is the window after the moment (locked in the data stage by the moment,
    the window and what the log holds up to the moment); the items' columns are checked against the window before the
    moment only; the model is chosen at the validation origin; the window after the moment is read once."""

    task_type = "ranking"
    primary_metric = "ndcg_at_k"
    higher_is_better = True
    uses_tables = True

    def __init__(self, kind: Any, solution: Any, frame: Any, project: dict[str, Any], tables: dict[str, Any] | None = None):
        import hashlib

        super().__init__(kind, solution, frame, project)
        f = self.fields
        self.items = (tables or {}).get(f["items_table"])
        if self.items is None:
            raise ValueError(f"The ranking machine reads the items table {f['items_table']!r}: attach it to the project by that name")
        self.at = pd.Timestamp(f["moment"])
        self.validation_at = self.at - pd.Timedelta(days=f["window_days"])
        times = pd.to_datetime(frame[f["time_column"]], errors="coerce")
        before, after = times <= self.at, times > self.at
        if not before.any() or not after.any():
            raise ValueError(f"The log must hold interactions before and after the moment {f['moment']}: nothing to learn from or to score on")
        self.before = frame[before]
        self.lock = {"moment": f["moment"], "window_days": f["window_days"], "rows_before": int(before.sum()), "rows_after": int(after.sum()),
                     "last_before": str(times[before].max())}
        self.lock["fingerprint"] = hashlib.sha256(repr(sorted(self.lock.items())).encode()).hexdigest()[:16]

    def training_rows(self) -> Any:
        return self.before

    def holdout_description(self) -> str:
        return f"the {self.fields['window_days']} days after {self.fields['moment']} ({self.lock['rows_after']} interactions)"

    def cv_description(self) -> str:
        return f"the validation origin {self.validation_at.date()}, one window before the moment, from the log up to the moment"

    def _common(self) -> dict[str, Any]:
        f = self.fields
        return dict(user=f["user_column"], item=f["item_column"], time=f["time_column"], event=f["event_column"], relevant=f["relevant_event"])

    def _ctx(self, previous: dict) -> dict[str, Any]:
        from dclab_core import ranking

        f = self.fields
        allowed = list(((previous.get("features") or {}).get("evidence") or {}).get("features") or [])
        return ranking.setup(self.frame, self.items, **self._common(), moment=f["moment"], window_days=f["window_days"],
                             created=f["created_column"], category=f["category_column"], item_features=allowed)

    def data(self, eid: str, previous: dict) -> tuple:
        f = self.fields
        created = int((pd.to_datetime(self.items[f["created_column"]]) > self.at).sum()) if f["created_column"] else 0
        evidence = {"train_rows": self.lock["rows_before"], "holdout_rows": self.lock["rows_after"], "holdout_split": self.holdout_description(),
                    "cv_protocol": self.cv_description(), "split": self.lock, "users": int(self.before[f["user_column"]].nunique()),
                    "items": int(len(self.items)), "items_created_after_moment": created, "source_rows": int(len(self.frame)),
                    "usable_feature_count": int(len(self.items.columns) - 1),
                    "notes_for_person": [{"title": "The window after the moment is the holdout", "proof": ["DCLAB-R02", "DCLAB-R17"],
                                          "text": f"Models read the log up to {f['moment']}; they are scored once on {self.holdout_description()}. "
                                                  f"{created} items were created after the moment and must not be recommended."}]}
        claims = [claim(f"{eid}-C1", "fact", f"{self.lock['rows_before']} interactions up to {f['moment']} are read; {self.holdout_description()} "
                        "is the holdout.", ["evidence.split"])]
        return evidence, claims, f"{self.lock['rows_before']} interactions read, holdout {self.holdout_description()}", None

    def leakage(self, eid: str, previous: dict) -> tuple:
        from dclab_core import ranking

        f = self.fields
        log = self.frame[pd.to_datetime(self.frame[f["time_column"]], errors="coerce") <= self.at]  # never past the moment
        check = ranking.item_check(self.items, log, **self._common(), moment=str(self.validation_at), created=f["created_column"])
        numeric = [c for c in self.items.columns if c not in (f["item_column"], f["created_column"], f["category_column"])
                   and pd.api.types.is_numeric_dtype(self.items[c])]
        out = sorted(set(check["with_the_future"]) | {x.column for x in self.solution.forbidden if x.column in numeric})
        evidence = {"with_the_future": check["with_the_future"], "why": check["why"], "kept_out": out, "item_columns": numeric,
                    "declared_leakage_features": out, "checked_on": f"the log up to the moment, its after the validation origin {self.validation_at.date()}",
                    "heuristic_review_candidates": [{"feature": c, "reasons": ["counted_after_the_moment"]} for c in check["with_the_future"]],
                    "notes_for_person": [{"level": "warning", "title": f"{c} is kept out", "text": check["why"][c], "proof": ["DCLAB-R01", "DCLAB-R05"]}
                                         for c in check["with_the_future"]]}
        claims = [claim(f"{eid}-C1", "fact", f"Item columns counted later: {', '.join(check['with_the_future']) or 'none'}; kept out of the ranker.",
                        ["evidence.with_the_future"])]
        return evidence, claims, f"{len(out)} item columns kept out", None

    def features(self, eid: str, previous: dict) -> tuple:
        leak = (previous.get("leakage") or {}).get("evidence") or {}
        allowed = [c for c in leak.get("item_columns", []) if c not in leak.get("kept_out", [])]
        recipe = "interaction_counts_and_item_columns"
        about = f"each user's and item's interactions before the list's moment, and the item columns {', '.join(allowed) or 'none'}"
        evidence = {"selected_recipe": recipe, "best_recipe": recipe, "selected_recipe_description": about, "features": allowed,
                    "stage_results": [{"recipe": recipe, "description": about, "feature_count_mean": len(allowed),
                                       "metrics": {self.primary_metric: {"mean": None, "std": None}}}], "selection_rule": "one design for every model"}
        decision = {"kind": "recipe", "selected": recipe, "chosen": recipe, "rule": evidence["selection_rule"],
                    "options": [{"id": recipe, "label": recipe, "description": about, "features": len(allowed)}]}
        return evidence, [claim(f"{eid}-C1", "fact", f"The ranker reads {about}.", ["evidence.features"])], about, decision

    def models(self, eid: str, previous: dict) -> tuple:
        from dclab_core import ranking

        f = self.fields
        ctx = self._ctx(previous)
        picked = ranking.validate(ctx, k=f["k"])
        k, scores, best = f["k"], picked["scores"], picked["chosen"]
        ranking_rows = sorted(({"model": m, "mean": v[f"ndcg@{k}"], "std": 0.0} for m, v in scores.items()), key=lambda r: -(r["mean"] or 0))
        evidence = {"selected_model": best, "ranking": ranking_rows, "validation": {"origin": picked["origin"], "scores": scores},
                    "ranker": picked["ranker"], "kept_out_of_ranker": ctx["kept_out"], "selection_rule": picked["chosen_by"],
                    "notes_for_person": [{"title": f"{best} chosen at the validation origin", "proof": ["DCLAB-R13", "DCLAB-R17"],
                                          "text": f"NDCG@{k} by model: " + "; ".join(f"{r['model']} {r['mean']}" for r in ranking_rows)
                                                  + ". Popularity is what the models are judged against; the window after the moment is sealed."}]}
        decision = {"kind": "model", "selected": best, "chosen": best, "rule": picked["chosen_by"],
                    "options": [{"id": r["model"], "label": r["model"], "mean": r["mean"], "std": r["std"]} for r in ranking_rows if r["model"] != "popularity"]}
        claims = [claim(f"{eid}-C1", "fact", f"NDCG@{k} at the validation origin: " + "; ".join(f"{r['model']} {r['mean']}" for r in ranking_rows)
                        + f". Chosen: {best}.", ["evidence.ranking", "evidence.selected_model"])]
        return evidence, claims, f"{best} chosen at the validation origin", decision

    def final(self, eid: str, previous: dict) -> tuple:
        from dclab_core import ranking

        f = self.fields
        locked = (((previous.get("data") or {}).get("evidence") or {}).get("split") or {}).get("fingerprint")
        if locked != self.lock["fingerprint"]:  # the log or the moment changed since the lock
            raise ValueError("The holdout was locked on another log or moment than the table now gives: rerun the data stage")
        chosen = self.chosen(previous.get("models"))
        if chosen in (None, "popularity"):
            raise ValueError("Popularity is what the models are judged against, not a model to use")
        k = f["k"]
        held = ranking.score(self._ctx(previous), k=k)
        mine = held["scores"][chosen]
        cv = next((r["mean"] for r in ((previous.get("models") or {}).get("evidence") or {}).get("ranking", []) if r["model"] == chosen), None)
        gain = mine["against_popularity"]
        created = int((pd.to_datetime(self.items[f["created_column"]]) > self.at).sum()) if f["created_column"] else 0
        evidence = {"model": chosen, "feature_recipe": self.chosen(previous.get("features")),
                    "holdout_metrics": {"ndcg_at_k": mine[f"ndcg@{k}"], "recall_at_k": mine[f"recall@{k}"], "coverage": mine["coverage"]},
                    "holdout_primary_metric_ci": {"low": None, "high": None, "point": mine[f"ndcg@{k}"],
                                                  "method": f"the gain over popularity carries the interval: {gain['ci90']} (90%, paired over users)"},
                    "cv_selected_metric_mean": cv, "cv_to_holdout_gap": round(mine[f"ndcg@{k}"] - cv, 4) if cv is not None and mine[f"ndcg@{k}"] is not None else None,
                    "baselines": {m: {"ndcg_at_k": held["scores"][m][f"ndcg@{k}"]} for m in ("popularity", "recent_popularity") if m in held["scores"]},
                    "scores": held["scores"], "beats_popularity": held["beats_popularity"], "read_until": held["read_until"],
                    "read_after_moment": held["read_after_moment"], "ranker_trained_on": held["ranker_trained_on"],
                    "created_after_moment": created, "created_after_recommended": held["created_after_recommended"],
                    "users_scored": held["users_scored"], "holdout_split": self.holdout_description(), "holdout_rows": self.lock["rows_after"],
                    "train_rows": self.lock["rows_before"], "selection_rule": "the model chosen at the validation origin, read once here",
                    "notes_for_person": [{"title": f"Holdout NDCG@{k} {mine[f'ndcg@{k}']} for {chosen}", "proof": ["DCLAB-R17", "DCLAB-R18"],
                                          "text": f"Gain over popularity {gain['ndcg_gain']} (90% interval {gain['ci90']}); coverage {mine['coverage']}, "
                                                  f"novelty {mine['novelty']}; users without a history {mine['without_history']}."}]}
        claims = [claim(f"{eid}-C1", "fact", f"On {self.holdout_description()} {chosen} scores NDCG@{k} {mine[f'ndcg@{k}']}, a gain over popularity of "
                        f"{gain['ndcg_gain']} (90% interval {gain['ci90']}).", ["evidence.holdout_metrics", "evidence.baselines"],
                        ["One window after one moment.", "Not approved for production from this evidence alone."])]
        return evidence, claims, f"{chosen}: holdout NDCG@{k} {mine[f'ndcg@{k}']}", {
            "kind": "final", "selected": "defaults", "chosen": "defaults", "rule": evidence["selection_rule"], "options": []}


KIND = Ranking()
