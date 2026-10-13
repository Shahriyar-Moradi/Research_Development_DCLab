"""Ranking and recommendation (package 16.5): users, items and interactions with time; what to show each user next. The
kind runs its own machine (``dclab_core.ranking``), not the five tabular stages: models read the interactions up to the
recommendation moment and the items that existed at it, and are scored on the window after it against popularity."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .base import ProblemKind


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
    stratify = False
    runs_in_engine = False  # the five-stage engine scores rows, not lists: dclab_core.ranking runs it as a library
    Fields = RankingFields

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return "ndcg_at_k"

    def applies(self, solution: Any) -> bool:
        from .base import _get

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
        return Case(replica.interactions, f["relevant"], "binary", f"On {f['moment']}, the list each customer sees for the next two weeks.",
                    solution={"kind_fields": fields, "items": replica.items}, facts=f)

    def solve(self, case: Any) -> Any:
        from dclab_core import ranking

        f = self.fields(case.solution.get("kind_fields"))  # what the person states; never which column leaks
        items: pd.DataFrame = case.solution["items"]
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


KIND = Ranking()
