"""Uplift (package 16.7): who responds to an action (an offer, a call), not who would have responded anyway: two arms,
uplift models judged by Qini and AUUC against a response model and random targeting, and the record's wording bound by
randomisation and overlap. The engine runs it as its own machine (16.15 B, ``UpliftMachine``): the holdout stratified by
arm and locked in the data stage (from the arms only), the model chosen on a validation cut of the training rows, the
holdout read once in the final stage; the library runs it whole (``dclab_core.uplift.evaluate``)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .base import ProblemKind, _get
from .machine import Machine, claim


class UpliftFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    treatment_column: str
    randomised: bool  # the person says whether the action was assigned at random; the overlap check tests it
    features: list[str] | None = Field(default=None, max_length=200)  # None: every other column but identifiers


class Uplift(ProblemKind):
    key = "uplift"
    title = "Uplift: who responds to an action"
    description = "Who changes because of an action (an offer, a call), so the action goes where it makes a difference."
    base_task = "binary"
    task_type = "uplift"
    specificity = 15
    packs = ()
    words = (r"\b(uplift|incremental (?:response|effect) of (?:the |an |a )?(?:offer|campaign|call|discount)|persuadable\w*"
             r"|who (?:responds?|reacts?) to (?:the |an |a )?(?:offer|campaign|call|discount|treatment)"
             r"|effect of (?:the |an |a )?(?:discount|offer|campaign|call) on (?:each|which) (?:customer|user)s?)\b")
    questions = ("Which column says who got the action (the offer, the call), and which says the outcome?",
                 "Was the action given at random (an experiment), or chosen by someone?",
                 "What does one action cost, and what is one more outcome worth?")
    metrics = (("qini_coefficient", "Qini coefficient"), ("auuc", "Area under the uplift curve"))
    baselines = ("response_model", "random")
    checks = ()
    reliability = ("randomisation and propensity overlap", "uplift by decile", "the wording the record may use")
    evidence_track = "causal-inference-and-experimentation"  # 16.14: the research track whose evidence agents cite for this kind
    aliases = ('uplift modelling', 'incremental effect', 'treatment effect', 'campaign targeting', 'A/B test')  # 16.14: the names this kind of problem goes by (read into its evidence record)
    stratify = True
    runs_in_engine = True  # 16.15 B: its own machine, stage by stage
    export = {"notebook": False, "service": False}  # two arms and an effect per row: the tabular exports do not fit it
    Fields = UpliftFields

    def own_machine(self) -> Any:
        return UpliftMachine

    def columns_of(self, solution: Any) -> list[str]:
        f = self.fields(_get(solution, "kind_fields"))
        return [f["treatment_column"], *(f["features"] or [])]

    def check_solution(self, solution: Any) -> None:
        f = self.fields(solution.kind_fields)
        if f["treatment_column"] == solution.target:
            raise ValueError("The treatment column and the outcome must be different columns")
        if solution.metric and solution.metric not in (m for m, _ in self.metrics):
            raise ValueError(f"The uplift machine is judged by {', '.join(m for m, _ in self.metrics)}, not {solution.metric}")

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return "qini_coefficient"

    def applies(self, solution: Any) -> bool:
        return _get(solution, "kind") == self.key

    # ------------------------------------------------------------------ acceptance (16.2's harness, its own machine)
    def case(self, seed: int = 0) -> Any:
        """The randomised campaign, and the person's second campaign, chosen by the marketing team (said not random)."""
        from dclab_core.replicas import campaign_uplift

        from .acceptance import Case

        trial, chosen = campaign_uplift.generate(seed, randomised=True), campaign_uplift.generate(seed + 1, randomised=False)
        f = trial.facts
        return Case(trial.customers, f["outcome"], "binary", "When the campaign list is drawn, before any offer is sent.",
                    solution={"kind": "uplift", "kind_fields": {"treatment_column": f["treatment"], "randomised": True}},
                    items={"second_campaign": {"table": chosen.customers, "randomised": False}},  # read by the library's solve only
                    facts={**f, "second_campaign": chosen.facts})

    def solve(self, case: Any) -> Any:
        from dclab_core import uplift

        f = self.fields(case.solution.get("kind_fields"))  # what the person states; never the planted segment
        skip = {f["treatment_column"], case.target}

        def features(table):
            ids = {c for c in table.columns if c not in skip and table[c].nunique() == len(table)}
            return f["features"] or [c for c in table.columns if c not in skip | ids], sorted(ids)
        cols, ids = features(case.table)
        result = uplift.evaluate(case.table, treatment=f["treatment_column"], outcome=case.target, features=cols, randomised=f["randomised"])
        second = (case.items or {}).get("second_campaign")
        observational = None
        if second is not None:  # the same machine on a campaign the person says was chosen, not random: the wording must change
            observational = uplift.evaluate(second["table"], treatment=f["treatment_column"], outcome=case.target,
                                            features=features(second["table"])[0], randomised=bool(second["randomised"]))
        return {"evaluation": result, "observational": observational, "identifiers": ids, "facts": case.facts}

    def accept(self, run: Any) -> list[dict[str, Any]]:
        import numpy as np

        from .acceptance import check

        r, f, obs = run["evaluation"], run["facts"], run["observational"]
        best = r["best"]
        responsive = np.array(f["responsive"])[r["holdout_rows"]]

        def top_share(model: str) -> float:
            s = np.array(r["holdout_uplift"][model])
            return float(responsive[np.argsort(-s, kind="stable")[: max(1, len(s) // 10)]].mean())
        mine, response = top_share(best), top_share("response_model")
        m = r["models"][best]
        chosen_on = set(r["validation_rows"])
        return [check("the arms are compared on a holdout split by arm, the model chosen on validation rows apart from it",
                      r["split"]["stratified_by"] == f["treatment"] and best is not None and len(chosen_on) == r["split"]["validation"]
                      and not chosen_on & set(r["holdout_rows"]) and r["validation"] != {n: v["qini_coefficient"] for n, v in r["models"].items()}
                      and best == max((n for n in r["validation"] if n not in ("response_model", "random")),
                                                                                key=lambda n: r["validation"][n]),
                      f"{best} chosen on {r['split']['validation']} validation rows; {r['split']['holdout']} held out"),
                check("the responsive segment is recovered: the top tenth by predicted uplift is mostly in it", mine >= 0.8 and mine > response,
                      f"{mine:.0%} of {best}'s top tenth are {f['segment_rule']} (all rows {responsive.mean():.0%}; the response model's top tenth {response:.0%})"),
                check("uplift beats random targeting, with an interval", r["best_beats_random"],
                      f"{best}: Qini coefficient {m['qini_coefficient']}, 90% interval {m['qini_coefficient_ci90']}; AUUC {m['auuc']}, "
                      f"interval {m['auuc_ci90']}; random {r['models']['random']['qini_coefficient']}"),
                check("the randomised campaign may speak of effects: the columns do not predict who got the offer", r["wording"]["causal"]
                      and r["overlap"]["overlaps"], f"propensity AUC {r['overlap']['propensity_auc']}; {r['wording']['say']}"),
                check("the observational campaign raises the overlap warning and is never called causal", obs is not None
                      and not obs["overlap"]["overlaps"] and not obs["wording"]["causal"] and "not effects" in obs["wording"]["say"],
                      obs["wording"]["say"] if obs else "the observational version was not run"),
                check("uplift by decile and the Qini curve in the record", len(m["uplift_by_decile"]) == 10 and len(m["qini"]) > 2,
                      f"uplift in the top decile {m['uplift_by_decile'][0]}, the second {m['uplift_by_decile'][1]}")]


    def accept_engine(self, run: Any) -> list[dict[str, Any]]:
        """The decisions on the engine's records of the same case (16.15 B)."""
        import numpy as np

        from .acceptance import check

        records = run.records
        data, models, final = (records[k].get("evidence") or {} for k in ("data", "models", "final"))
        facts, best = run.case.facts, final.get("model")
        chosen = ((records["models"].get("decision") or {}).get("chosen"))
        mine = (final.get("models") or {}).get(best) or {}
        responsive = np.array(facts["responsive"])[final.get("holdout_rows") or []]
        scores = np.array((final.get("holdout_uplift") or {}).get(best) or [])
        top = float(responsive[np.argsort(-scores, kind="stable")[: max(1, len(scores) // 10)]].mean()) if len(scores) else 0.0
        return [check("every stage ran on the uplift machine", all(records[k].get("machine") == self.key for k in records), str({k: records[k].get("machine") for k in records})),
                check("the holdout is split by arm and locked in the data stage, the choice made on validation rows apart from it",
                      data.get("split", {}).get("stratified_by") == facts["treatment"] and best == chosen
                      and final.get("holdout_uses_in_this_project") == 1, f"{chosen} chosen on {data.get('split', {}).get('validation')} validation rows"),
                check("the final model is the one chosen on validation, never the response model", best not in (None, "response_model", "random"), str(best)),
                check("the responsive segment is recovered by the model the engine chose", top >= 0.8, f"{top:.0%} of its top tenth are responsive"),
                check("uplift beats random targeting on the holdout, with an interval", bool(final.get("best_beats_random")),
                      f"Qini {mine.get('qini_coefficient')}, 90% interval {mine.get('qini_coefficient_ci90')}"),
                check("the randomised campaign may speak of effects: the columns do not predict who got the offer",
                      bool((data.get("wording") or {}).get("causal")) and bool((data.get("overlap") or {}).get("overlaps")),
                      str((data.get("wording") or {}).get("say")))]


class UpliftMachine(Machine):
    """Uplift in the engine (16.15 B). The holdout is stratified by arm and fixed by the arms alone (its fingerprint is
    locked in the data stage); the model is chosen on a validation cut of the training rows; the holdout's outcomes are
    read once, in the final stage."""

    task_type = "uplift"
    primary_metric = "qini_coefficient"
    higher_is_better = True

    def __init__(self, kind: Any, solution: Any, frame: Any, project: dict[str, Any]):
        from dclab_core import uplift

        super().__init__(kind, solution, frame, project)
        self.data_frame = frame.reset_index(drop=True)
        self.treatment, self.outcome = self.fields["treatment_column"], solution.target
        self.randomised = bool(self.fields["randomised"])
        self.parts = uplift.split(self.data_frame, treatment=self.treatment)  # the arms only: never the outcome

    def training_rows(self) -> Any:
        return self.data_frame.iloc[self.parts["train"]]

    def holdout_description(self) -> str:
        sz = self.parts["sizes"]
        return f"{sz['holdout']} rows held out, stratified by {self.treatment} (fixed by the arms alone)"

    def cv_description(self) -> str:
        return f"a validation cut of {self.parts['sizes']['validation']} training rows, stratified by {self.treatment}"

    def _features(self, previous: dict) -> list[str]:
        return list(((previous.get("leakage") or {}).get("evidence") or {}).get("features") or [])

    def data(self, eid: str, previous: dict) -> tuple:
        from dclab_core import uplift

        from dclab_rnd.studio.leakage_review import from_moment

        train = self.training_rows()
        own = {self.treatment, self.outcome}
        after = {i["column"] for i in from_moment([{"name": c} for c in self.data_frame.columns], self.solution.prediction_moment, own)}
        out = after | {x.column for x in self.solution.forbidden} | set(self.solution.identifiers)  # never read by the models either
        candidates = [c for c in (self.fields["features"] or self.data_frame.columns) if c not in own and c not in out
                      and self.data_frame[c].nunique() < len(self.data_frame)]
        ov = uplift.overlap(self.data_frame, treatment=self.treatment, features=candidates)  # the arms and the columns, never the outcome
        overlap = {k: v for k, v in ov.items() if k != "propensity"}
        words = uplift.wording(self.randomised, ov)
        arms = {str(k): int(v) for k, v in train[self.treatment].value_counts().items()}
        split = {**self.parts["sizes"], "holdout_fingerprint": self.parts["fingerprint"]}
        evidence = {"train_rows": split["train"], "holdout_rows": split["holdout"], "holdout_split": self.holdout_description(),
                    "cv_protocol": self.cv_description(), "split": split, "arms_in_training": arms, "overlap": overlap, "wording": words,
                    "randomised": self.randomised, "usable_feature_count": len(candidates), "source_rows": int(len(self.data_frame)),
                    "notes_for_person": [{"title": "What the record may say", "text": words["say"], "proof": ["DCLAB-R01", "DCLAB-R17"]}]}
        claims = [claim(f"{eid}-C1", "fact", f"{split['train']} training rows ({arms}), {split['holdout']} held out stratified by {self.treatment}; "
                        f"propensity AUC {overlap.get('propensity_auc')}: {words['say']}", ["evidence.split", "evidence.overlap", "evidence.wording"])]
        return evidence, claims, f"{split['train']} training rows, holdout {self.holdout_description()}", None

    def leakage(self, eid: str, previous: dict) -> tuple:
        from dclab_rnd.studio.leakage_review import from_moment

        own = {self.treatment, self.outcome}
        identifiers = [c for c in self.data_frame.columns if c not in own and self.data_frame[c].nunique() == len(self.data_frame)]
        said = {i["column"]: i["reason"] for i in from_moment([{"name": c} for c in self.data_frame.columns], self.solution.prediction_moment, own)}
        forbidden = [x.column for x in self.solution.forbidden if x.column not in own]
        out = sorted(set(identifiers) | set(said) | set(forbidden) | set(self.solution.identifiers))
        declared = self.fields["features"]
        features = [c for c in (declared or self.data_frame.columns) if c not in own and c not in out]
        why = {**{c: "a different value on every row: an identifier" for c in identifiers}, **said,
               **{c: "forbidden in the solution" for c in forbidden}, **{c: "an identifier in the solution" for c in self.solution.identifiers}}
        evidence = {"kept_out": out, "why": why, "features": features, "declared_leakage_features": [c for c in out if c in said or c in forbidden],
                    "heuristic_review_candidates": [{"feature": c, "reasons": ["after_the_moment"]} for c in said],
                    "notes_for_person": [{"level": "warning", "title": f"{c} is kept out", "text": why[c], "proof": ["DCLAB-R01", "DCLAB-R05"]}
                                         for c in said]}
        claims = [claim(f"{eid}-C1", "fact", f"Kept out: {', '.join(out) or 'none'}. Read: {', '.join(features)}.", ["evidence.kept_out", "evidence.features"])]
        return evidence, claims, f"{len(out)} columns kept out, {len(features)} read", None

    def features(self, eid: str, previous: dict) -> tuple:
        features = self._features(previous)
        recipe = "standardised_numbers_onehot_categories"
        about = f"{len(features)} columns known when the list is drawn, one design for every uplift model, fitted on the training rows"
        evidence = {"selected_recipe": recipe, "best_recipe": recipe, "selected_recipe_description": about, "features": features,
                    "stage_results": [{"recipe": recipe, "description": about, "feature_count_mean": len(features),
                                       "metrics": {self.primary_metric: {"mean": None, "std": None}}}], "selection_rule": "one design for every model"}
        decision = {"kind": "recipe", "selected": recipe, "chosen": recipe, "rule": evidence["selection_rule"],
                    "options": [{"id": recipe, "label": recipe, "description": about, "features": len(features)}]}
        return evidence, [claim(f"{eid}-C1", "fact", f"The models read {about}.", ["evidence.features"])], about, decision

    def models(self, eid: str, previous: dict) -> tuple:
        from dclab_core import uplift

        picked = uplift.validate(self.data_frame, self.parts, treatment=self.treatment, outcome=self.outcome,
                                 features=self._features(previous), randomised=self.randomised)
        v, best = picked["validation"], picked["best"]
        ranking = sorted(({"model": n, "mean": q, "std": 0.0} for n, q in v.items()), key=lambda r: (r["mean"] is None, -(r["mean"] or 0)))
        evidence = {"selected_model": best, "ranking": ranking, "validation": v, "selection_rule": picked["chosen_by"],
                    "notes_for_person": [{"title": f"{best} chosen on the validation rows", "proof": ["DCLAB-R13", "DCLAB-R17"],
                                          "text": "Validation Qini coefficient by model: " + "; ".join(f"{r['model']} {r['mean']}" for r in ranking)
                                                  + ". The response model and random targeting are what it is judged against; the holdout is sealed."}]}
        decision = {"kind": "model", "selected": best, "chosen": best, "rule": picked["chosen_by"],
                    "options": [{"id": r["model"], "label": r["model"], "mean": r["mean"], "std": r["std"]} for r in ranking
                                if r["model"] not in ("response_model", "random")]}
        claims = [claim(f"{eid}-C1", "fact", "Validation Qini coefficient by model: " + "; ".join(f"{r['model']} {r['mean']}" for r in ranking)
                        + f". Chosen: {best}.", ["evidence.ranking", "evidence.selected_model"])]
        return evidence, claims, f"{best} chosen on the validation rows", decision

    def final(self, eid: str, previous: dict) -> tuple:
        from dclab_core import uplift

        locked = (((previous.get("data") or {}).get("evidence") or {}).get("split") or {}).get("holdout_fingerprint")
        if locked != self.parts["fingerprint"]:  # the table changed since the lock
            raise ValueError("The holdout was locked on other rows than the table now gives: rerun the data stage")
        chosen = self.chosen(previous.get("models"))
        if chosen in (None, "response_model", "random"):
            raise ValueError(f"{chosen} is what an uplift model is judged against, not a model to use")
        scored = uplift.score(self.data_frame, self.parts, treatment=self.treatment, outcome=self.outcome, features=self._features(previous),
                              randomised=self.randomised, best=chosen)
        mine = scored["models"][chosen]
        cv = next((r["mean"] for r in ((previous.get("models") or {}).get("evidence") or {}).get("ranking", []) if r["model"] == chosen), None)
        low, high = mine["qini_coefficient_ci95"]
        evidence = {"model": chosen, "feature_recipe": self.chosen(previous.get("features")),
                    "holdout_metrics": {"qini_coefficient": mine["qini_coefficient"], "auuc": mine["auuc"]},
                    "holdout_primary_metric_ci": {"low": low, "high": high, "point": mine["qini_coefficient"],
                                                  "method": "paired bootstrap over the held-out rows, 95%"},
                    "cv_selected_metric_mean": cv, "cv_to_holdout_gap": round(mine["qini_coefficient"] - cv, 4) if cv is not None else None,
                    "baselines": {n: {"qini_coefficient": scored["models"][n]["qini_coefficient"], "auuc": scored["models"][n]["auuc"]}
                                  for n in ("response_model", "random")},
                    "models": scored["models"], "best_beats_random": scored["best_beats_random"],
                    "best_beats_response_model": scored["best_beats_response_model"], "uplift_by_decile": mine["uplift_by_decile"],
                    "holdout_rows": scored["holdout_rows"], "holdout_uplift": {chosen: scored["holdout_uplift"][chosen]},
                    "wording": ((previous.get("data") or {}).get("evidence") or {}).get("wording"),
                    "holdout_split": self.holdout_description(), "train_rows": self.parts["sizes"]["train"],
                    "selection_rule": "the model chosen on validation, read once here",
                    "notes_for_person": [{"title": f"Holdout Qini coefficient {mine['qini_coefficient']} for {chosen}", "proof": ["DCLAB-R17", "DCLAB-R18"],
                                          "text": f"95% interval {low} to {high}; the response model {scored['models']['response_model']['qini_coefficient']}, "
                                                  f"random {scored['models']['random']['qini_coefficient']}; uplift in the top tenth {mine['uplift_by_decile'][0]}."}]}
        claims = [claim(f"{eid}-C1", "fact", f"On the holdout ({self.holdout_description()}) {chosen} scores a Qini coefficient of {mine['qini_coefficient']} "
                        f"(95% interval {low} to {high}) against the response model's {scored['models']['response_model']['qini_coefficient']} "
                        f"and random targeting's {scored['models']['random']['qini_coefficient']}.", ["evidence.holdout_metrics", "evidence.baselines"],
                        ["One holdout, stratified by arm.", "The wording the record may use is bound by randomisation and overlap.",
                         "Not approved for production from this evidence alone."])]
        return evidence, claims, f"{chosen}: holdout Qini {mine['qini_coefficient']}", {
            "kind": "final", "selected": "defaults", "chosen": "defaults", "rule": evidence["selection_rule"], "options": []}


KIND = Uplift()
