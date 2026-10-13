"""Segmentation (package 16.6): groups in a table with no outcome to predict. The kind runs its own machine
(``dclab_core.segment``), not the five tabular stages: no target, no holdout score; segments judged by stability under
resampling and separation, described in the data's own columns, with the no-ground-truth limit in the record."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .base import ProblemKind


class SegmentationFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    columns: list[str] | None = Field(default=None, max_length=200)  # None: every column
    outcome_column: str | None = None  # a later outcome to describe the segments with, never to fit them
    max_segments: int = Field(default=8, ge=2, le=20)


class Segmentation(ProblemKind):
    key = "segmentation"
    title = "Segmentation"
    description = "Groups of customers (or items, or places) that behave alike, when there is no outcome to predict."
    base_task = "multiclass"
    task_type = "segmentation"
    specificity = 15
    packs = ()
    words = (r"\b(segment(?:ation)? (?:of )?(?:our |the |my )?(?:customers|users|clients|stores|shoppers|members)"
             r"|(?:customer|user|shopper) segments?|cluster(?:ing)? (?:of )?(?:our |the )?(?:customers|users|clients|stores)"
             r"|personas?|groups? of (?:customers|users|stores) (?:that|who) behave)\b")
    PREDICTS = r"\b(predict\w*|forecast\w*|estimat\w*|will (?:\w+ )?(?:churn|buy|cancel|pay|leave|default))\b"
    questions = ("Which rows are grouped (customers, stores), and which columns describe how they behave?",
                 "Is there a later outcome to describe the groups with (a renewal, a spend)? It will not be used to form them.",
                 "At most how many segments could the team act on?")
    metrics = (("stability", "Bootstrap stability (adjusted Rand)"), ("silhouette", "Silhouette"))
    baselines = ()
    checks = ()
    reliability = ("stability interval per number of segments", "columns no group uses", "density view and its noise share")
    evidence_track = None  # no research track yet
    stratify = False
    runs_in_engine = False  # no target and no holdout: dclab_core.segment runs it as a library
    Fields = SegmentationFields

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return "stability"

    def detect(self, problem: str, profile: dict[str, Any] | None = None) -> str | None:
        import re

        if re.search(self.PREDICTS, (problem or "").lower()):  # "predict churn, reported per segment" is a prediction
            return None
        return super().detect(problem, profile)

    def applies(self, solution: Any) -> bool:
        from .base import _get

        return _get(solution, "kind") == self.key

    # ------------------------------------------------------------------ acceptance (16.2's harness, its own machine)
    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import customer_behaviours

        from .acceptance import Case

        replica = customer_behaviours.generate(seed)
        table = replica.customers.assign(renewed=replica.facts["renewed"])  # a later outcome the person names to describe the groups
        return Case(table, "renewed", "multiclass", "Group the customers by how they buy, to plan a campaign for each group.",
                    solution={"kind_fields": {"outcome_column": "renewed", "max_segments": 7}}, facts=replica.facts)

    def solve(self, case: Any) -> Any:
        from dclab_core import segment

        f = self.fields(case.solution.get("kind_fields"))  # what the person states; never the planted behaviours
        columns = f["columns"] or [c for c in case.table.columns if c != f["outcome_column"]]
        result = segment.segment(case.table, columns=columns, ks=range(2, f["max_segments"] + 1), outcome=f["outcome_column"], resamples=12)
        return {"segmentation": result, "facts": case.facts}

    def accept(self, run: Any) -> list[dict[str, Any]]:
        import numpy as np
        from sklearn.metrics import adjusted_rand_score

        from .acceptance import check

        r, f = run["segmentation"], run["facts"]
        chosen = r.get("chosen") or {}
        truth = np.array(f["behaviour"])
        planted = truth != f["noise_label"]
        ari = adjusted_rand_score(truth[planted], np.array(r["labels"])[planted]) if r.get("labels") else None
        record = r["preprocessing"]
        above = [x for x in r["rejected_as_unstable"] if x["method"] == chosen.get("method") and x["k"] > f["behaviours"]]
        profiled = r.get("profiles") or []
        return [check("the preprocessing is recorded: skewed amounts log-scaled, the identifier left out",
                      "log" in record["numbers"].get("avg_basket", "") and f["identifier"] in record["left_out"],
                      f"avg_basket: {record['numbers'].get('avg_basket')}; {f['identifier']}: {record['left_out'].get(f['identifier'])}"),
                check("columns no group uses are left out, with the reason", set(f["noise_columns"]) <= set(r["passes"]["unused_columns"]),
                      "; ".join(f"{c}: {w}" for c, w in r["passes"]["unused_columns"].items())),
                check("the planted behaviours are recovered, with a stability interval", chosen.get("k") == f["behaviours"] and ari is not None and ari >= 0.8
                      and chosen.get("stability", {}).get("ci90", [0])[0] >= r["min_stability"],
                      f"{chosen.get('method')} k={chosen.get('k')}, stability {chosen.get('stability')}, agreement with the planted behaviours {ari and round(ari, 3)}"),
                check("more segments than planted are rejected for instability, for the chosen method", bool(above),
                      "; ".join(f"{x['method']} k={x['k']}: {x['why']}" for x in above[:3])),
                check("each segment is described in the data's own columns", bool(profiled) and all(len(p["sets_it_apart"]) >= 2 for p in profiled),
                      "; ".join(f"segment {p['segment']} ({p['share']}): " + ", ".join(d["column"] for d in p["sets_it_apart"]) for p in profiled)),
                check("the record says there is no ground truth, and the outcome only describes the segments",
                      "no ground truth" in r["limit"] and "outcome" in r and "does not make them right" in r["outcome"]["note"],
                      str({g: v["rate"] for g, v in (r.get("outcome") or {}).get("by_segment", {}).items()}))]


KIND = Segmentation()
