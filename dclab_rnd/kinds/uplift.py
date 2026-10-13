"""Uplift (package 16.7): who responds to an action (an offer, a call), not who would have responded anyway. The kind runs
its own machine (``dclab_core.uplift``), not the five tabular stages: two arms, uplift models judged by Qini and AUUC
against a response model and random targeting, and the record's wording bound by randomisation and overlap."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .base import ProblemKind


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
    runs_in_engine = False  # two arms and an effect per row: dclab_core.uplift runs it as a library
    Fields = UpliftFields

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return "qini_coefficient"

    def applies(self, solution: Any) -> bool:
        from .base import _get

        return _get(solution, "kind") == self.key

    # ------------------------------------------------------------------ acceptance (16.2's harness, its own machine)
    def case(self, seed: int = 0) -> Any:
        """The randomised campaign, and the person's second campaign, chosen by the marketing team (said not random)."""
        from dclab_core.replicas import campaign_uplift

        from .acceptance import Case

        trial, chosen = campaign_uplift.generate(seed, randomised=True), campaign_uplift.generate(seed + 1, randomised=False)
        f = trial.facts
        return Case(trial.customers, f["outcome"], "binary", "When the campaign list is drawn, before any offer is sent.",
                    solution={"kind_fields": {"treatment_column": f["treatment"], "randomised": True},
                              "second_campaign": {"table": chosen.customers, "randomised": False}},
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
        second = case.solution.get("second_campaign")
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


KIND = Uplift()
