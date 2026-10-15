"""Segmentation (package 16.6): groups in a table with no outcome to predict, judged by stability under resampling and
separation, described in the data's own columns, with the no-ground-truth limit in the record. The engine runs it as its
own machine (16.15 B, ``SegmentationMachine``): a fifth of the rows held out by their row numbers and locked in the data
stage; the segments chosen on the training rows without the outcome; in the final stage, read once, the held-out rows
are assigned (are they as well separated as on the rows the segments were formed on?) and the person's later outcome
describes the segments."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .base import ProblemKind, _get
from .machine import Machine, claim


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
    evidence_track = "data-science-foundations"  # 16.14: the research track whose evidence agents cite for this kind
    aliases = ('clustering', 'customer segments', 'personas', 'unsupervised grouping')  # 16.14: the names this kind of problem goes by (read into its evidence record)
    stratify = False
    runs_in_engine = True  # 16.15 B: its own machine, stage by stage
    needs_target = False  # no outcome forms the segments; a later one may describe them (kind_fields.outcome_column)
    export = {"notebook": False, "service": False}  # groups, not predictions: the tabular exports do not fit it
    Fields = SegmentationFields

    def own_machine(self) -> Any:
        return SegmentationMachine

    def columns_of(self, solution: Any) -> list[str]:
        f = self.fields(_get(solution, "kind_fields"))
        return [*(f["columns"] or []), *([f["outcome_column"]] if f["outcome_column"] else [])]

    def check_solution(self, solution: Any) -> None:
        f = self.fields(solution.kind_fields)
        if f["outcome_column"] and f["outcome_column"] in (f["columns"] or []):
            raise ValueError("The outcome describes the segments; it cannot also form them: leave it out of the columns")
        if solution.metric and solution.metric not in (m for m, _ in self.metrics):
            raise ValueError(f"The segmentation machine is judged by {', '.join(m for m, _ in self.metrics)}, not {solution.metric}")

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return "silhouette"  # 16.15 B: the held-out rows' separation is what the final stage reads

    def detect(self, problem: str, profile: dict[str, Any] | None = None) -> str | None:
        import re

        if re.search(self.PREDICTS, (problem or "").lower()):  # "predict churn, reported per segment" is a prediction
            return None
        return super().detect(problem, profile)

    def applies(self, solution: Any) -> bool:
        return _get(solution, "kind") == self.key

    # ------------------------------------------------------------------ acceptance (16.2's harness, its own machine)
    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import customer_behaviours

        from .acceptance import Case

        replica = customer_behaviours.generate(seed)
        table = replica.customers.assign(renewed=replica.facts["renewed"])  # a later outcome the person names to describe the groups
        return Case(table, None, "multiclass", "Group the customers by how they buy, to plan a campaign for each group.",
                    solution={"kind": "segmentation", "kind_fields": {"outcome_column": "renewed", "max_segments": 7}}, facts=replica.facts)

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


    def accept_engine(self, run: Any) -> list[dict[str, Any]]:
        """The decisions on the engine's records of the same case (16.15 B)."""
        import numpy as np
        from sklearn.metrics import adjusted_rand_score

        from .acceptance import check

        records, f = run.records, run.case.facts
        models, final = (records[k].get("evidence") or {} for k in ("models", "final"))
        chosen = models.get("chosen") or {}
        truth = np.array(f["behaviour"])
        planted = truth != f["noise_label"]
        labels = np.array(final.get("labels") or [])
        ari = adjusted_rand_score(truth[planted], labels[planted]) if len(labels) == len(truth) else None
        outcome = (run.solution.get("kind_fields") or {}).get("outcome_column")
        return [check("every stage ran on the segmentation machine", all(records[s].get("machine") == self.key for s in records),
                      str({s: records[s].get("machine") for s in records})),
                check("the outcome is never read before the final stage: it is not among the columns the segments are formed on",
                      outcome not in (models.get("columns_used") or []) and final.get("holdout_uses_in_this_project") == 1,
                      f"formed on {models.get('columns_used')}"),
                check("columns no group uses are left out, with the reason", set(f["noise_columns"]) <= set(models.get("unused_columns") or {}),
                      str(models.get("unused_columns"))),
                check("the planted behaviours are recovered on the training rows, with a stability interval",
                      chosen.get("k") == f["behaviours"] and ari is not None and ari >= 0.8,
                      f"{chosen.get('method')} k={chosen.get('k')}, stability {chosen.get('stability')}, agreement {ari and round(ari, 3)}"),
                check("the segments hold on the held-out rows: separated beyond data with no groups", bool(final.get("holds_on_holdout")),
                      f"held-out silhouette {(final.get('holdout_metrics') or {}).get('silhouette')} against "
                      f"{(final.get('holdout_metrics') or {}).get('null_silhouette')} on data with no groups"),
                check("the outcome only describes the segments, and the record says there is no ground truth",
                      "does not make them right" in ((final.get("outcome") or {}).get("note") or "") and "no ground truth" in (final.get("limit") or ""),
                      str({g: v.get("rate") for g, v in ((final.get("outcome") or {}).get("by_segment") or {}).items()}))]


class SegmentationMachine(Machine):
    """Segmentation in the engine (16.15 B). A fifth of the rows is held out by row number (locked in the data stage); the
    segments are chosen on the training rows, never reading the outcome; the final stage assigns the held-out rows and
    lets the outcome describe the segments, once."""

    task_type = "segmentation"
    primary_metric = "silhouette"  # the held-out rows' own separation; the training rows' stability is reported beside it
    higher_is_better = True
    HOLDOUT = 0.2

    def __init__(self, kind: Any, solution: Any, frame: Any, project: dict[str, Any]):
        import hashlib

        import numpy as np

        super().__init__(kind, solution, frame, project)
        f = self.fields
        self.data_frame = frame.reset_index(drop=True)
        self.outcome = f["outcome_column"]
        forbidden = {x.column for x in solution.forbidden}
        self.cols = [c for c in (f["columns"] or self.data_frame.columns) if c != self.outcome and c not in forbidden
                     and c not in solution.identifiers]
        order = np.random.default_rng(0).permutation(len(self.data_frame))  # row numbers only: never a value
        n = int(round(len(order) * self.HOLDOUT))
        self.holdout, self.train = np.sort(order[:n]), np.sort(order[n:])
        if len(self.train) < 30 or n < 10:
            raise ValueError(f"{len(self.data_frame)} rows are too few to form segments and check them on held-out rows")
        self.fingerprint = hashlib.sha256(f"{len(self.data_frame)}:{','.join(map(str, self.holdout.tolist()))}".encode()).hexdigest()[:16]

    def training_rows(self) -> Any:
        return self.data_frame.iloc[self.train]

    def holdout_description(self) -> str:
        return f"{len(self.holdout)} rows held out by their row numbers (a fifth, seeded), assigned to the segments once"

    def cv_description(self) -> str:
        return "the segments' stability under resampling of the training rows (adjusted Rand), and their separation against data with no groups"

    def data(self, eid: str, previous: dict) -> tuple:
        split = {"train": int(len(self.train)), "holdout": int(len(self.holdout)), "fingerprint": self.fingerprint}
        evidence = {"train_rows": split["train"], "holdout_rows": split["holdout"], "holdout_split": self.holdout_description(),
                    "cv_protocol": self.cv_description(), "split": split, "columns": self.cols, "outcome_column": self.outcome,
                    "usable_feature_count": len(self.cols), "source_rows": int(len(self.data_frame)),
                    "notes_for_person": [{"title": "No outcome forms the segments", "proof": ["DCLAB-R17"],
                                          "text": f"The segments are formed on {len(self.cols)} columns of the training rows"
                                                  + (f"; {self.outcome} describes them in the final stage only, and never makes them right."
                                                     if self.outcome else ".")}]}
        claims = [claim(f"{eid}-C1", "fact", f"{split['train']} training rows; {self.holdout_description()}.", ["evidence.split"])]
        return evidence, claims, f"{split['train']} training rows, {split['holdout']} held out", None

    def leakage(self, eid: str, previous: dict) -> tuple:
        from dclab_core import segment

        _, record = segment.prepare(self.training_rows(), self.cols)
        left = dict(record.get("left_out") or {})
        out = sorted([*left, *([self.outcome] if self.outcome else [])])
        why = {**left, **({self.outcome: "the later outcome: it describes the segments, it never forms them"} if self.outcome else {})}
        evidence = {"kept_out": out, "why": why, "declared_leakage_features": [self.outcome] if self.outcome else [],
                    "features": [c for c in self.cols if c not in left], "heuristic_review_candidates": [],
                    "notes_for_person": [{"title": f"{c} is left out", "text": why[c], "proof": ["DCLAB-R01"]} for c in out]}
        claims = [claim(f"{eid}-C1", "fact", f"Left out of the segments: {', '.join(f'{c} ({why[c][:60]})' for c in out) or 'none'}.", ["evidence.kept_out"])]
        return evidence, claims, f"{len(out)} columns left out", None

    def _features(self, previous: dict) -> list[str]:
        return list(((previous.get("leakage") or {}).get("evidence") or {}).get("features") or [])

    def features(self, eid: str, previous: dict) -> tuple:
        from dclab_core import segment

        features = self._features(previous)
        _, record = segment.prepare(self.training_rows(), features)
        recipe = "segment_scaling"
        about = f"{len(features)} columns, each scaled as the data's spread asks (skewed amounts log-scaled), fitted on the training rows"
        evidence = {"selected_recipe": recipe, "best_recipe": recipe, "selected_recipe_description": about, "features": features,
                    "preprocessing": record, "stage_results": [{"recipe": recipe, "description": about, "feature_count_mean": len(features),
                                                                "metrics": {self.primary_metric: {"mean": None, "std": None}}}],
                    "selection_rule": "one design for every grouping"}
        decision = {"kind": "recipe", "selected": recipe, "chosen": recipe, "rule": evidence["selection_rule"],
                    "options": [{"id": recipe, "label": recipe, "description": about, "features": len(features)}]}
        return evidence, [claim(f"{eid}-C1", "fact", f"The segments read {about}.", ["evidence.preprocessing"])], about, decision

    def models(self, eid: str, previous: dict) -> tuple:
        from dclab_core import segment

        f = self.fields
        found = segment.search(self.training_rows().reset_index(drop=True), self._features(previous), range(2, f["max_segments"] + 1), resamples=12)
        table, chosen = found["final"]["table"], found["final"]["chosen"]
        good = sorted((v for v in table.values() if v["stable"] and v["separated"]), key=lambda v: -(v["silhouette"] or -1))
        ranking = [{"model": f"{v['method']}:{v['k']}", "mean": v["stability"]["mean"], "std": 0.0, "silhouette": v["silhouette"]} for v in good]
        best = f"{chosen['method']}:{chosen['k']}" if chosen else None
        why = "the best separated of the stable choices (lower end of the 90% stability interval at least 0.75, silhouette above data with no groups)"
        evidence = {"selected_model": best, "ranking": ranking, "chosen": {k: chosen[k] for k in ("method", "k", "stability", "silhouette", "null_silhouette")}
                    if chosen else None, "columns_used": found["kept"], "unused_columns": found["unused"],
                    "candidates": {k: {kk: v[kk] for kk in ("method", "k", "stability", "silhouette", "null_silhouette", "stable", "separated")}
                                   for k, v in table.items()}, "selection_rule": why,
                    "notes_for_person": [{"title": f"{best or 'no grouping'} chosen on the training rows", "proof": ["DCLAB-R13", "DCLAB-R17"],
                                          "text": ("Stable and separated: " + "; ".join(f"{r['model']} (stability {r['mean']}, silhouette {r['silhouette']})"
                                                                                      for r in ranking)) if ranking else "No grouping is both stable and separated."}]}
        decision = {"kind": "model", "selected": best, "chosen": best, "rule": why,
                    "options": [{"id": r["model"], "label": r["model"], "mean": r["mean"], "std": r["std"]} for r in ranking]}
        claims = [claim(f"{eid}-C1", "fact", f"Chosen on the training rows: {best or 'none: the data shows no stable groups'}.", ["evidence.chosen"])]
        return evidence, claims, f"{best or 'no grouping'} chosen on the training rows", decision

    def final(self, eid: str, previous: dict) -> tuple:
        from sklearn.metrics import silhouette_score

        from dclab_core import segment

        locked = (((previous.get("data") or {}).get("evidence") or {}).get("split") or {}).get("fingerprint")
        if locked != self.fingerprint:  # the table changed since the lock
            raise ValueError("The holdout was locked on other rows than the table now gives: rerun the data stage")
        models = (previous.get("models") or {}).get("evidence") or {}
        chosen = self.chosen(previous.get("models"))
        if not chosen:
            raise ValueError("No grouping was stable and separated on the training rows: there is nothing to assign")
        method, k = chosen.split(":")[0], int(chosen.split(":")[1])
        cols = list(models.get("columns_used") or self._features(previous))
        X_train, record = segment.prepare(self.training_rows(), cols)  # fitted on the training rows only, as the models stage chose
        model = segment._fit(method, k, X_train, 0)
        X = segment.transform(self.data_frame, record)  # every row in the training rows' scaling: nothing learned from the held-out ones
        labels = model.predict(X)
        held = X[self.holdout]
        sil = float(silhouette_score(held, labels[self.holdout])) if len(set(labels[self.holdout])) > 1 else None
        null = segment._null_silhouette(method, k, held, 0)
        holds = sil is not None and null is not None and sil >= null + segment.SEPARATION
        stability = next((r["mean"] for r in models.get("ranking", []) if r["model"] == chosen), None)
        outcome = segment.describe_outcome(self.data_frame, labels, self.outcome)  # the outcome, read once, here
        trained_sil = ((models.get("chosen") or {}).get("silhouette"))
        evidence = {"model": chosen, "feature_recipe": self.chosen(previous.get("features")),
                    "holdout_metrics": {"silhouette": round(sil, 3) if sil is not None else None,
                                        "null_silhouette": round(null, 3) if null is not None else None},
                    "holdout_primary_metric_ci": {"low": None, "high": None, "point": round(sil, 3) if sil is not None else None,
                                                  "method": "not computed: one held-out fifth of the rows; the null silhouette is beside it"},
                    "training_stability": stability, "cv_selected_metric_mean": trained_sil,
                    "cv_to_holdout_gap": round(sil - trained_sil, 3) if sil is not None and trained_sil is not None else None,
                    "baselines": {"no_groups": {"silhouette": round(null, 3) if null is not None else None}},
                    "holds_on_holdout": holds, "labels": labels.tolist(), "profiles": segment.profiles(self.data_frame, labels, cols),
                    "outcome": outcome, "limit": segment.NO_GROUND_TRUTH, "holdout_split": self.holdout_description(),
                    "holdout_rows": int(len(self.holdout)), "train_rows": int(len(self.train)),
                    "selection_rule": "the grouping chosen on the training rows, assigned to the held-out rows once",
                    "notes_for_person": [{"title": f"{chosen}: the held-out rows {'keep' if holds else 'do not keep'} the separation", "proof": ["DCLAB-R17"],
                                          "text": f"Held-out silhouette {sil and round(sil, 3)} against {null and round(null, 3)} on data with no groups. "
                                                  + segment.NO_GROUND_TRUTH}]}
        claims = [claim(f"{eid}-C1", "fact", f"{chosen}, formed on the training rows, assigned to {len(self.holdout)} held-out rows: silhouette "
                        f"{sil and round(sil, 3)} against {null and round(null, 3)} on data with no groups.", ["evidence.holdout_metrics"],
                        [segment.NO_GROUND_TRUTH, "Not approved for production from this evidence alone."])]
        return evidence, claims, f"{chosen}: held-out silhouette {sil and round(sil, 3)}", {
            "kind": "final", "selected": "defaults", "chosen": "defaults", "rule": evidence["selection_rule"], "options": []}


KIND = Segmentation()
