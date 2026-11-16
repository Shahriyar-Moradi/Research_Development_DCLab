"""H1 and H3 of the Hierarchical Planning plan (2026-11-16): the model's breakdown of a problem keeps its valid parts,
and the plan is one path from the problem to its solution: sub-problems in the order their dependencies allow, each with
its steps, then the build. Scripted models only."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import task_catalog as T  # noqa: E402
from dclab_rnd import prompts  # noqa: E402
from dclab_rnd.tasks import catalog as C  # noqa: E402
from dclab_rnd.tasks import view  # noqa: E402

TABLES = {"orders": ["order_id", "customer_id", "placed_at", "refund_date", "is_fraud"]}
BREAKDOWN = {"decision": "Which orders to send to manual review for fraud", "target": "is_fraud", "unit": "order", "horizon_days": 7,
             "success_metric": "pr_auc",
             "subproblems": [{"id": "score", "kind": "rare_event", "objective": "Score each order at the moment it is placed", "dependency_ids": ["meaning"]},
                             {"id": "meaning", "kind": "classification", "objective": "Settle what counts as fraud", "dependency_ids": []}],
             "risks": [{"trap_kind": "post_outcome", "table": "orders", "column": "refund_date", "recommended_task": "leakage_probe"}]}
CALLS = [{"task_type": "leakage_probe", "params": {"table": "orders", "column": "refund_date"}, "purpose": "A refund date exists only after the outcome", "subproblem": "score"},
         {"task_type": "ask_owner", "params": {"field": "target", "question": "Does a chargeback count as fraud?", "blocking": True}, "subproblem": "meaning"},
         {"task_type": "grouped_split", "params": {"entity": "customer_id"}, "purpose": "One customer places many orders"}]


class Planner:
    def __init__(self, *answers):
        self.answers, self.seen, self.spent_eur = list(answers), [], 0.0

    def complete(self, messages, tools=None, max_tokens=None, **options):
        self.seen.append({"max_tokens": max_tokens, **options})
        return {"content": json.dumps(self.answers.pop(0)) if self.answers else ""}


class Gateway:
    def __init__(self, client):
        self._client = client

    def client(self, purpose, **kwargs):
        return self._client


class BreakdownTests(unittest.TestCase):
    def test_one_wrong_part_never_throws_the_breakdown_away(self):
        raw = {**BREAKDOWN, "risks": [*BREAKDOWN["risks"], {"trap_kind": "x", "recommended_task": "leakage_probe"},
                                      {"trap_kind": "post_outcome", "table": "orders", "column": "invented", "recommended_task": "leakage_probe"}],
               "subproblems": [BREAKDOWN["subproblems"][0], {**BREAKDOWN["subproblems"][1], "dependency_ids": ["ghost"]}]}
        self.assertIsNone(T.validate_decomposition(raw, TABLES)[0])  # as before, for a caller that wants all or nothing
        kept, issues = T.validate_decomposition(raw, TABLES, keep_valid=True)
        self.assertEqual([r.column for r in kept.risks], ["refund_date"])
        self.assertEqual(kept.subproblems[1].dependency_ids, ())
        self.assertEqual(len(issues), 3)  # each refused part is named

    def test_a_repeated_id_or_a_number_where_a_list_belongs_never_crashes_the_plan(self):
        twice = {**BREAKDOWN, "risks": 5, "subproblems": [BREAKDOWN["subproblems"][1], {**BREAKDOWN["subproblems"][0], "id": "meaning", "dependency_ids": []}]}
        kept, issues = T.validate_decomposition(twice, TABLES, keep_valid=True)  # before: StopIteration in the path, after the model was paid
        self.assertEqual([s.objective for s in kept.subproblems], ["Settle what counts as fraud"])
        self.assertEqual(len(issues), 2)
        self.assertEqual([leg["id"] for leg in T.solution_path({**twice, "risks": []}, CALLS)["legs"]], ["meaning"])

    def test_a_question_for_the_person_with_one_stray_key_is_kept(self):
        valid, issues = T.validate_catalog_plan([{"task_type": "ask_owner", "params": {"field": "target", "question": "Does a chargeback count?", "urgent": True}}], TABLES)
        self.assertEqual((valid[0].params, len(issues)), ({"field": "target", "question": "Does a chargeback count?"}, 1))

    def test_a_step_on_a_column_the_tables_do_not_have_is_refused_alone(self):
        valid, issues = T.validate_catalog_plan([CALLS[0], {"task_type": "leakage_probe", "params": {"table": "orders", "column": "made_up"}}, CALLS[2]], TABLES)
        self.assertEqual([c.task_type for c in valid], ["leakage_probe", "grouped_split"])  # before: the invented column was run as valid
        self.assertIn("made_up", issues[0])

    def test_the_rules_never_guess_the_kind_from_a_name_nor_invent_an_outcome(self):
        self.assertEqual(C.kind_of({"target": "price_band", "task": "multiclass"}), "classification")  # before: "regression", from the word price
        self.assertEqual(C.kind_of({"target": "churned", "task": "regression"}), "regression")
        unknown = C.rules_decomposition("Reduce customer losses before next quarter, across both subscriber tables and the contract history we keep", {})
        self.assertEqual((unknown.target, unknown.horizon_days, [u.field for u in unknown.unknowns]), (None, None, ["target"]))  # before: "target", 0 days
        self.assertIn("contract history we keep", unknown.decision)  # the goal as written: before, cut at 200 characters
        plan = C.plan_with_model("Predict the band", {"listings": ["id", "price_band"]}, {"target": "price_band", "task": "multiclass", "unit": "row",
                                                                                           "horizon_days": 0, "clean_single_table": True})
        self.assertEqual((plan["subproblems"], plan["decomposition"]["success_metric"]), (["classification"], "roc_auc"))


class PathTests(unittest.TestCase):
    def test_the_path_runs_from_the_problem_through_its_sub_problems_to_the_build(self):
        evidence = [{"task_type": "leakage_probe", "traps": ["post_outcome_refund_date"], "findings": ["refund_date set on 3% of rows"]}]
        path = T.solution_path(BREAKDOWN, CALLS, evidence)
        self.assertEqual(path["goal"], BREAKDOWN["decision"])
        self.assertEqual([leg["id"] for leg in path["legs"]], ["meaning", "score"])  # what must be settled first comes first
        self.assertEqual(path["legs"][1]["after"], ["meaning"])
        steps = [(s["n"], s["task"], s["on"], s["for_person"]) for leg in path["legs"] for s in leg["steps"]]
        self.assertEqual(steps, [(1, "ask_owner", "target", True), (2, "grouped_split", "customer_id", False), (3, "leakage_probe", "orders.refund_date", False)])
        self.assertEqual(path["legs"][1]["steps"][0]["found"], ["post_outcome_refund_date"])
        self.assertEqual([(b["n"], b["task"]) for b in path["build"]][:2] + [path["build"][-1]["task"]], [(4, "data"), (5, "features"), "report"])
        self.assertTrue(next(b for b in path["build"] if b["task"] == "holdout")["for_person"])  # the holdout waits for the person

    def test_every_part_of_the_path_carries_its_reason_and_the_next_step_says_why(self):
        path = T.solution_path(BREAKDOWN, CALLS)
        # the reason of a sub-problem's place is written by code from its dependencies: no model is asked to write its reasoning
        self.assertEqual([leg["why"] for leg in path["legs"]], ["Nothing has to be settled before it.",
                                                                 "Comes after: Settle what counts as fraud. That must be settled first."])
        self.assertEqual(path["legs"][1]["steps"][0]["why"], "A refund date exists only after the outcome")  # a step's reason is the plan's own purpose
        self.assertTrue(all(step["why"] for step in path["build"]))  # each stage says why it stands where it does
        self.assertIn("read once", next(b for b in path["build"] if b["task"] == "holdout")["why"])
        waiting = view._path({"path": path}, {})
        self.assertEqual((waiting["next"]["n"], waiting["next"]["who"], waiting["next"]["what"]), (1, "you", "Does a chargeback count as fraud?"))
        answered = view._path({"path": path, "owner_confirmed": True}, {"data": "completed"})
        self.assertEqual((answered["next"]["what"], answered["next"]["who"]), ("Choose the features, inside the training folds", "DCLab"))
        self.assertIn("cannot come back in", answered["next"]["why"])
        done = view._path({"path": path, "owner_confirmed": True}, {b["task"]: "completed" for b in path["build"]})
        self.assertIsNone(done["next"])  # nothing left: no next step is invented

    def test_a_plan_with_no_breakdown_and_a_cycle_still_has_a_path(self):
        self.assertEqual(T.solution_path(None, CALLS[:1])["legs"][0]["id"], "main")
        looped = {**BREAKDOWN, "subproblems": [{**BREAKDOWN["subproblems"][0]}, {**BREAKDOWN["subproblems"][1], "dependency_ids": ["score"]}]}
        self.assertEqual(len(T.solution_path(looped, [])["legs"]), 2)

    def test_the_page_shows_each_step_with_what_the_engine_did_about_it(self):
        plan = {"path": T.solution_path(BREAKDOWN, CALLS, [{"task_type": "leakage_probe", "traps": ["post_outcome_refund_date"], "findings": ["3% of rows"]}]),
                "enforcement": {"applied": [{"field": "forbidden", "value": "refund_date", "by": "human"}], "asked": [{"value": "customer_id"}]}}
        shown = view._path(plan, {"data": "completed"})
        steps = {s["task"]: s for leg in shown["legs"] for s in leg["steps"]}
        self.assertEqual(steps["leakage_probe"]["did"], "enforced: forbidden = refund_date (your decision)")
        self.assertEqual(steps["grouped_split"]["did"], "waiting for your answer")
        self.assertNotIn("findings", steps["leakage_probe"])  # a step's raw findings may count rows: they stay in the record
        self.assertEqual([b["status"] for b in shown["build"]][:2], ["completed", "pending"])


class RoomTests(unittest.TestCase):
    def test_the_model_plans_with_room_and_its_valid_parts_are_kept(self):
        answer = {"decomposition": {**BREAKDOWN, "risks": [*BREAKDOWN["risks"], {"trap_kind": "post_outcome", "table": "orders", "column": "invented",
                                                                                 "recommended_task": "leakage_probe"}]},
                  "calls": [*CALLS, {"task_type": "leakage_probe", "params": {"table": "orders", "column": "invented"}, "subproblem": "score"}],
                  "caught_traps": ["post_outcome_refund_date"]}
        client = Planner(answer, answer)
        plan = C.plan_with_model("Which orders to review for fraud", TABLES, {"target": "is_fraud", "unit": "order", "horizon_days": 7}, gateway=Gateway(client))
        self.assertEqual(plan["routed_to"], "model")  # before: one wrong column swapped the whole plan for the rules'
        self.assertEqual(len(plan["calls"]), 3)
        self.assertEqual(len(plan["refusals"]), 2)
        self.assertEqual([leg["id"] for leg in plan["path"]["legs"]], ["meaning", "score"])
        for seen in client.seen:  # no cap on thinking, one attempt or a token ceiling that stops the gateway asking again
            self.assertEqual(seen, {"max_tokens": C.PLAN_TOKENS})

    def test_the_prompt_gives_room_and_asks_for_no_reasoning_in_the_reply(self):
        text = prompts.text("problem_planner")
        for gone in ("<100 words", "VERY BRIEF", "3 to 6"):
            self.assertNotIn(gone, text)
        for kept in ("no limit on your thinking", "dependency_ids", '"subproblem"', "ask_owner"):
            self.assertIn(kept, text)
        for forbidden in ('"reason"', '"why"', '"reasoning"'):  # the owner's rule: reasoning is read from the thinking summary, never asked for
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
