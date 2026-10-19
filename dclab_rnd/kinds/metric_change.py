"""Why did this number change (package 30.20, the owner's pattern 6): a question about a business number, answered from approved
definitions and tested queries, with the factor that moved it, where, and by how much.

Not a model of an outcome: the answer is a decomposition (sales = footfall x conversion x basket) read from the tables a semantic layer
approves, with a refusal where the data has gaps or does not hold what is asked. It runs as a library (``dclab_core.metric_change``), not in
the five-stage engine: a project of this kind has no target to train on, so the Solution refuses it and the capability matrix says partial.
Its acceptance is the mall sales replica's planted causes on several seeds."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from .base import ProblemKind, _get


class MetricChangeFields(BaseModel):
    model_config = ConfigDict(extra="forbid")
    layer: dict[str, Any] | None = None  # a dclab_core.semantic_layer.v1 document


class MetricChange(ProblemKind):
    key = "metric_change"
    title = "Why a number changed"
    description = "Which factor moved a business number (footfall, conversion, basket), where, by how much, with the query under the answer."
    base_task = "regression"
    task_type = "metric_change"
    specificity = 16
    packs = ()
    words = (r"\b(?:why (?:are|is|did|has|have|were|was) (?:\w+(?:'s)? ){0,4}(?:sales|revenue|footfall|traffic|conversion|basket|bookings|"
             r"spend|orders|visits)\b[^.?]{0,60}\b(?:down|up|drop|dropped|fall|fell|falling|rise|rose|rising|change|changed|lower|higher)\b"
             r"|root[- ]cause (?:of|for) (?:the|a|our) (?:drop|fall|rise|change|decline)|why did (?:this|the|our) (?:number|metric|kpi) (?:change|move|drop))")
    questions = ("Which number moved, for which brand, store or segment, and against which period (last year, the month before)?",
                 "Where are the approved definitions: how is the number computed, and which tables hold its parts?")
    metrics = (("planted_cause_found", "Planted causes found and ranked first"),)
    baselines = ("the metric's change alone (no decomposition)",)
    checks = ()
    reliability = ("every factor's interval over days", "coverage: a gap refuses the why", "the query under every answer")
    evidence_track = "metric-root-cause"
    aliases = ("root cause analysis", "metric decomposition", "why did sales drop", "ask your data", "conversational BI")
    stratify = False
    runs_in_engine = False  # a library: no outcome to train on
    needs_target = False
    export = {"notebook": False, "service": False}
    Fields = MetricChangeFields

    def detect(self, problem: str, profile: dict[str, Any] | None = None) -> str | None:
        """Its words, unless the sentence is a prediction problem that also asks a why (the review: "forecast next week's sales; why
        are sales down in winter?" is forecasting): another specific kind's words match too, or the sentence reads as a task."""
        why = super().detect(problem, profile)
        if not why:
            return None
        from dclab_core.understand import understand

        from . import all_kinds

        if any(k.key != self.key and k.specificity > 0 and k.detect(problem, profile) for k in all_kinds()):
            return None
        return None if understand(problem or "").task.known else why

    def applies(self, solution: Any) -> bool:
        return _get(solution, "kind") == self.key

    def default_metric(self, solution: Any, positive_rate: float | None, pack: dict[str, Any] | None = None) -> str:
        return "planted_cause_found"

    # ------------------------------------------------------------------ acceptance: the mall sales replica's planted causes
    def case(self, seed: int = 0) -> Any:
        from dclab_core.replicas import mall_sales

        return {"seed": seed, "replica": mall_sales.generate(seed), "layer": mall_sales.layer()}

    def solve(self, case: Any) -> Any:
        from dclab_core import metric_change as mc

        from dclab_rnd.metric_change import data_of

        r, layer = case["replica"], case["layer"]
        data = data_of(case)  # the studio's runner: SQLite, so the SQL under each answer is the statement that ran
        answers = []
        for truth in r.truth:
            a, said = mc.ask(truth["question"], layer, {}, data=data, seed=case["seed"])
            named = None
            if truth.get("template"):  # the same question, naming its template: what the model-off path must give
                p = truth["params"]
                q = (f"{truth['template']}: metric={p['metric']}; " + "".join(f"{k}={v}; " for k, v in p["filters"].items())
                     + f"period={p['period'][0]}..{p['period'][1]}; compare={p['compare']}" + (f"; direction={truth['direction']}" if truth.get("direction", "none") != "none" else ""))
                named = mc.ask(q, layer, {}, data=data, seed=case["seed"])[0]
            answers.append({"truth": truth, "answer": a, "said": said, "named": named})
        return {"answers": answers, "planted": r.info["planted"]}

    def accept(self, run: Any) -> list[dict[str, Any]]:
        from .acceptance import check

        by = {x["truth"]["id"]: x for x in run["answers"]}

        def found(case_id: str) -> tuple[bool, str]:
            x = by[case_id]
            return reached(x["truth"], x["answer"]), x["answer"].words[:300]

        out = []
        for case_id, decision in (("footfall_drop", "a footfall drop in one mall: footfall is the factor ranked first"),
                                  ("footfall_where", "the brand's footfall change is placed in the mall it happened in"),
                                  ("conversion_drop", "a conversion drop in one category: conversion first, that category's rate where"),
                                  ("mix_basket", "a basket drop from the mix: the basket first, the cheap category's mix where (not its rate)"),
                                  ("no_change", "a number that moved only within its variation: no factor is said to have moved"),
                                  ("data_gap", "a store whose door counter failed: the why is refused with footfall missing"),
                                  ("competitor", "a question about a competitor: refused, with what the data would need"),
                                  ("margin", "a question about a margin with no cost table: refused, with what the data would need")):
            ok, detail = found(case_id)
            out.append(check(decision, ok, detail))
        same = [x for x in run["answers"] if x["named"] is not None]
        out.append(check("a question naming its template gets the same finding as the question in words (the model-off path)",
                         all(reached(x["truth"], x["named"]) for x in same), f"{len(same)} named"))
        return out


def reached(truth: dict[str, Any], a: Any) -> bool:
    """Whether an answer reaches what the planted cause says a careful analyst finds."""
    if truth.get("refused"):
        return a.status == "refused" and truth["missing"] in a.missing
    if a.status != "answered":
        return False
    if truth["first_factor"] != "any" and a.first_factor != truth["first_factor"]:
        return False
    if truth.get("first_segment") and (a.first_segment or {}).get("value") != truth["first_segment"]["value"]:
        return False
    if truth.get("effect") and (a.first_segment or {}).get("effect") != truth["effect"]:
        return False
    return not (truth["direction"] == "none" and (a.moved or a.first_segment))


KIND = MetricChange()
