"""Package 24.4: the solution map. A block for each sub-problem of the problem map (its approach, its workflow, its split), the composition's steps
and ONE holdout design for the whole: every leaf splits as the whole does, no leaf scores the holdout on its own, the composition's last step reads it
once and nothing is chosen on it. A leaf split by time beside one split at random is refused."""

import json
import sys
import unittest
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core import datacard as dc, problem_map as pm, solution_map as sm, solutions as sol
    from dclab_core.contracts import CONTRACTS
    from dclab_core.replicas import delivery_fraud
except ImportError as error:  # pydantic and pandas are not in every environment
    MISSING = str(error)
else:
    MISSING = ""

FRAUD = ("Find the fake rides and fake orders our bikers and customers make. We have a few confirmed cases, nothing else is labelled, "
         "and the team can review 200 a day.")
CHURN = "I want to know which subscribers will cancel at their renewal."
_DATA = {}


def data(events=True):
    """The fraud replica's data card: the rides, the file of known frauds and (with ``events``) a table of what each phone did, which the rings need."""
    key = bool(events)
    if key not in _DATA:
        warnings.filterwarnings("ignore")
        r = delivery_fraud.generate(seed=0)
        tables, roles = {"rides": r.rides, "known_frauds": r.sample}, {"known_frauds": "labels"}
        if events:
            tables["phones"] = r.rides.groupby("biker_phone").size().reset_index(name="rides")
            roles["phones"] = "related"
        _DATA[key] = dc.card_of(tables, name="fraud", roles=roles)
    return _DATA[key]


def fresh(x):
    return sm.SolutionMap.model_validate({**x.model_dump(mode="json"), "fingerprint": ""})


def with_block(x, index, **changes):
    blocks = [b.model_copy(update=changes) if i == index else b for i, b in enumerate(x.blocks)]
    return fresh(x.model_copy(update={"blocks": blocks}))


def with_step(x, index, **changes):
    steps = [s.model_copy(update=changes) if i == index else s for i, s in enumerate(x.steps)]
    return fresh(x.model_copy(update={"steps": steps}))


@unittest.skipIf(MISSING, MISSING)
class BuildTest(unittest.TestCase):
    def setUp(self):
        self.m = pm.decompose(FRAUD)
        self.built = sm.solution_map(self.m, sol.routes(self.m, data()), data())

    def test_the_fraud_map_has_a_block_for_each_part_answering_it(self):
        self.assertEqual([b.sub_problem for b in self.built.blocks], ["sp1", "sp2", "sp3", "sp4"])
        self.assertEqual([b.approach for b in self.built.blocks], ["anomaly", "rare_event", "entity_tables", "review_budgeted"])
        self.assertEqual([b.family for b in self.built.blocks], ["anomaly", "rare_event", "entity_tables", "ranking"])
        self.assertEqual(sm.validate(self.built, self.m), [])

    def test_the_leaves_share_one_split_and_the_composition_reads_the_holdout_once_at_its_end(self):
        splits = {b.split.model_dump_json() for b in self.built.blocks}
        self.assertEqual(len(splits), 1)
        self.assertEqual(self.built.holdout.split, self.built.blocks[0].split)
        self.assertEqual(self.built.holdout.split.strategy, "time_group")  # the data card names entities and a time
        self.assertTrue(self.built.holdout.split.keys and self.built.holdout.split.moment)
        readers = [s for s in self.built.steps if s.reads_holdout]
        self.assertEqual([(s.kind, s.output) for s in readers], [("score", sm.HOLDOUT_STEP)])
        self.assertIs(self.built.steps[-1], readers[0])
        self.assertTrue(all(b.final == "deferred" for b in self.built.blocks))
        self.assertEqual((self.built.holdout.read_by, self.built.holdout.chosen_on), ("composition", "validation"))

    def test_the_steps_are_the_problem_maps_composition_then_the_score(self):
        self.assertEqual([(s.kind, s.inputs, s.output) for s in self.built.steps],
                         [("fusion", ["sp1", "sp2", "sp3"], "risk"), ("budget", ["risk", "sp4"], "review_list"), ("score", ["review_list"], "holdout_score")])
        self.assertEqual(self.built.steps[0].joins, self.m.composition.steps[0].joins)

    def test_each_blocks_workflow_is_the_solution_workflow_of_its_family(self):
        for b in self.built.blocks:
            self.assertEqual(b.workflow.family, b.family)
            self.assertEqual([s.id for s in b.workflow.steps if "holdout" in s.inputs], ["final_once"])

    def test_a_single_problem_keeps_its_own_final_score_and_has_no_steps(self):
        m = pm.decompose(CHURN)
        built = sm.solution_map(m, sol.routes(m, None, detected="binary"), None)
        self.assertEqual((len(built.blocks), built.steps, built.holdout.read_by, built.blocks[0].final, built.blocks[0].approach), (1, [], "leaf", "own", "binary"))
        self.assertEqual(sm.validate(built, m), [])

    def test_without_a_data_card_the_split_is_stratified_and_said(self):
        self.assertEqual(sm.split_for(None), sm.SplitDesign(strategy="stratified", keys=[], moment=None))
        self.assertEqual(sm.split_for(data()).strategy, "time_group")

    def test_without_a_table_of_events_the_rings_have_no_approach_and_the_map_says_so(self):
        m = pm.decompose(FRAUD)
        built = sm.solution_map(m, sol.routes(m, data(events=False)), data(events=False))  # the rides and the cases only
        self.assertIsNone(built.blocks[2].approach)
        self.assertEqual(len(built.blockers), 1)
        self.assertIn("several_tables", built.blockers[0])
        self.assertEqual(sm.validate(built, m), [])  # an honest gap is not an unsound map

    def test_a_part_with_no_approach_is_a_blocker_and_the_map_still_holds_together(self):
        routes = sol.routes(self.m, data())
        routes["sp3"] = routes["sp3"].model_copy(update={"chosen": None, "reason": "no table to join"})
        built = sm.solution_map(self.m, routes, data())
        self.assertEqual(len(built.blockers), 1)
        self.assertIn("sp3", built.blockers[0])
        self.assertIsNone(built.blocks[2].approach)
        self.assertEqual(sm.validate(built, self.m), [])

    def test_routes_may_be_given_as_json(self):
        routes = {k: json.loads(v.model_dump_json()) for k, v in sol.routes(self.m, data()).items()}
        self.assertEqual(sm.solution_map(self.m, routes, data()).fingerprint, self.built.fingerprint)

    def test_the_same_inputs_give_the_same_map(self):
        again = sm.solution_map(self.m, sol.routes(self.m, data()), data())
        self.assertEqual(again.fingerprint, self.built.fingerprint)

    def test_the_contract_round_trips_and_is_registered(self):
        self.assertEqual(sm.SolutionMap.read(self.built.model_dump_json()), self.built)
        self.assertIn("solution_map", CONTRACTS)
        document = json.loads(self.built.model_dump_json())
        document["fingerprint"] = "0" * 16
        with self.assertRaises(ValueError):
            sm.SolutionMap.read(document)


@unittest.skipIf(MISSING, MISSING)
class AdapterTest(unittest.TestCase):
    def test_the_one_holdout_is_a_holdout_plan_proposal_and_a_split_by_time_and_entity_says_it_is_unsupported_there(self):
        from dclab_core.adapters import main_product as mpx, main_product_problem_map as adapter

        m = pm.decompose(FRAUD)
        built = sm.solution_map(m, sol.routes(m, data()), data())
        plan = adapter.holdout_plan(built)
        self.assertEqual(list(plan["fields"]), mpx.fields_of("HoldoutPlan"))
        self.assertEqual((plan["contract"], plan["decision_point"], plan["status"]), ("HoldoutPlan", "split.strategy", "proposal"))
        self.assertEqual(plan["fields"]["strategy"], "unsupported")  # the main product has no verified holdout for time and entity together
        self.assertIn("a person decides", plan["notes"]["strategy"])
        self.assertTrue(plan["fields"]["group_column"] and plan["fields"]["time_column"])
        self.assertEqual(plan["fields"]["evidence"]["read_by"], "composition")

    def test_a_random_split_is_the_main_products_stratified_random_and_an_entity_split_is_group_disjoint(self):
        from dclab_core.adapters import main_product_problem_map as adapter

        m = pm.decompose(CHURN)
        built = sm.solution_map(m, sol.routes(m, None, detected="binary"), None)
        plan = adapter.holdout_plan(built)
        self.assertEqual((plan["fields"]["strategy"], plan["fields"]["stratified"], plan["fields"]["group_column"], plan["fields"]["time_column"]), ("stratified_random", True, None, None))
        grouped = sm.SolutionMap.model_validate({**built.model_dump(mode="json"), "fingerprint": "", "holdout": {**built.holdout.model_dump(mode="json"),
                                                                                                                 "split": {"strategy": "group", "keys": ["account_id"], "moment": None}}})
        out = adapter.holdout_plan(grouped)
        self.assertEqual((out["fields"]["strategy"], out["fields"]["group_column"]), ("group_disjoint", "account_id"))
        timed = sm.SolutionMap.model_validate({**built.model_dump(mode="json"), "fingerprint": "", "holdout": {**built.holdout.model_dump(mode="json"),
                                                                                                                "split": {"strategy": "time", "keys": [], "moment": "created_date"}}})
        self.assertEqual(adapter.holdout_plan(timed)["fields"]["strategy"], "temporal_future")

    def test_the_output_is_plain_json(self):
        from dclab_core.adapters import main_product_problem_map as adapter

        m = pm.decompose(FRAUD)
        plan = adapter.holdout_plan(sm.solution_map(m, sol.routes(m, data()), data()))
        self.assertEqual(json.loads(json.dumps(plan, allow_nan=False)), plan)


@unittest.skipIf(MISSING, MISSING)
class ValidateTest(unittest.TestCase):
    def setUp(self):
        self.m = pm.decompose(FRAUD)
        self.built = sm.solution_map(self.m, sol.routes(self.m, data()), data())

    def issues(self, x, m=None):
        return "\n".join(sm.validate(x, m))

    def test_a_leaf_split_at_random_beside_leaves_split_by_time_is_refused(self):
        odd = with_block(self.built, 1, split=sm.SplitDesign(strategy="stratified"))
        text = self.issues(odd)
        self.assertIn("blk2", text)
        self.assertIn("splits stratified on no key", text)
        self.assertIn("every leaf splits as the whole does", text)
        by_time = with_block(self.built, 0, split=sm.SplitDesign(strategy="time", moment=self.built.holdout.split.moment))
        self.assertIn("blk1", self.issues(by_time))

    def test_a_leaf_that_differs_only_in_a_key_or_the_moment_is_refused_too(self):
        keys = with_block(self.built, 2, split=self.built.blocks[2].split.model_copy(update={"keys": ["order_id"]}))
        self.assertIn("blk3", self.issues(keys))
        moment = with_block(self.built, 3, split=self.built.blocks[3].split.model_copy(update={"moment": "another_time"}))
        self.assertIn("blk4", self.issues(moment))

    def test_a_leaf_that_scores_the_holdout_on_its_own_is_refused_and_so_is_one_that_only_says_it_defers(self):
        text = self.issues(with_block(self.built, 1, final="own", defers=[]))
        self.assertIn("blk2 (Confirmed fraud) scores the holdout on its own", text)
        self.assertIn("the holdout would be read by the leaves too (blk2:final_once)", text)
        with self.assertRaises(ValueError):  # a block that says it defers must defer its final score, and only that
            sm.Block.model_validate({**self.built.blocks[0].model_dump(mode="json"), "defers": []})
        with self.assertRaises(ValueError):
            sm.Block.model_validate({**self.built.blocks[0].model_dump(mode="json"), "final": "own", "defers": ["final_once"]})

    def test_the_holdout_is_read_by_one_step_in_the_whole_map(self):
        readers = [(b.id, s.id) for b in self.built.blocks for s in b.workflow.steps if "holdout" in s.inputs and s.id not in b.defers]
        self.assertEqual(readers, [])  # every leaf's final score is the composition's to run
        self.assertTrue(all(b.defers == ["final_once"] for b in self.built.blocks))
        self.assertEqual([s.output for s in self.built.steps if s.reads_holdout], ["holdout_score"])

    def test_the_holdout_read_before_the_end_is_refused_with_where(self):
        early = with_step(self.built, 0, reads_holdout=True)  # the fusion reads it: its weights would be chosen on what it showed
        text = self.issues(early)
        self.assertIn("exactly one composition step (read by 2)", text)
        moved = fresh(self.built.model_copy(update={"steps": [self.built.steps[0].model_copy(update={"reads_holdout": True}), self.built.steps[1], self.built.steps[2].model_copy(update={"reads_holdout": False})]}))
        self.assertIn("read at step 1 of 3", self.issues(moved))
        self.assertIn("chooses or combines on what the holdout showed", self.issues(moved))

    def test_a_composition_that_never_reads_the_holdout_or_scores_something_else_is_refused(self):
        none = with_step(self.built, 2, reads_holdout=False)
        self.assertIn("exactly one composition step (read by 0)", self.issues(none))
        wrong = with_step(self.built, 2, output="something_else")
        self.assertIn("the map's score and nothing else", self.issues(wrong))

    def test_a_score_that_takes_something_other_than_the_composed_result_is_refused(self):
        for inputs in ([], ["sp1"], ["risk"], ["review_list", "sp1"]):
            self.assertIn("the system that is scored is the one the composition makes", self.issues(with_step(self.built, 2, inputs=inputs)), inputs)

    def test_a_step_with_no_inputs_a_duplicate_output_or_an_output_that_is_a_sub_problems_id_is_refused(self):
        self.assertIn("has no inputs", self.issues(with_step(self.built, 1, inputs=[])))
        self.assertIn("which step 1 already made", self.issues(with_step(self.built, 1, output="risk")))
        self.assertIn("which is a sub-problem's id", self.issues(with_step(self.built, 0, output="sp3")))
        self.assertIn("lists an input twice", self.issues(with_step(self.built, 0, inputs=["sp1", "sp1", "sp2"])))
        self.assertIn("a fusion combines at least two inputs", self.issues(with_step(self.built, 0, inputs=["sp1"])))

    def test_a_result_made_and_never_used_is_left_out_of_what_is_scored(self):
        steps = [sm.SolutionStep(kind="fusion", inputs=["sp1", "sp2"], output="x"), sm.SolutionStep(kind="budget", inputs=["sp3", "sp4"], output="y", joins="the account"),
                 sm.SolutionStep(kind="score", inputs=["y"], output=sm.HOLDOUT_STEP, reads_holdout=True)]
        text = self.issues(fresh(self.built.model_copy(update={"steps": steps})))
        self.assertIn("'x' (step 1) is made and never used", text)

    def test_steps_that_are_not_the_problem_maps_are_refused(self):
        without_budget = fresh(self.built.model_copy(update={"steps": [self.built.steps[0], self.built.steps[2].model_copy(update={"inputs": ["risk"]})]}))
        self.assertIn("the composition's steps are not the problem map's", self.issues(without_budget, self.m))
        self.assertNotIn("the composition's steps are not the problem map's", self.issues(self.built, self.m))

    def test_a_step_that_uses_what_nothing_made_is_refused(self):
        self.assertIn("uses 'nowhere'", self.issues(with_step(self.built, 0, inputs=["sp1", "nowhere"])))
        self.assertIn("uses 'review_list'", self.issues(with_step(self.built, 0, inputs=["review_list", "sp1"])))  # a later step's output

    def test_a_block_whose_workflow_does_not_hold_is_refused_with_its_reason(self):
        b = self.built.blocks[0]
        broken = b.workflow.model_copy(update={"steps": [s for s in b.workflow.steps if s.id != "leakage_audit"]})
        text = self.issues(with_block(self.built, 0, workflow=broken))
        self.assertIn("blk1's workflow", text)
        self.assertIn("leakage_audit", text)

    def test_a_workflow_of_another_family_than_its_block_is_refused(self):
        other = self.built.blocks[1].workflow
        self.assertIn("blk1 is anomaly but its workflow is for rare_event", self.issues(with_block(self.built, 0, workflow=other)))

    def test_a_split_design_must_be_coherent_pinned_and_the_same_for_every_leaf(self):
        for kwargs in ({"strategy": "time"}, {"strategy": "group"}, {"strategy": "time_group", "moment": "t"}, {"strategy": "time_group", "keys": ["a"]},
                       {"strategy": "stratified", "keys": ["a"]}, {"strategy": "stratified", "moment": "t"}):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                sm.SplitDesign(**kwargs)
        for share in (0, 0.5, 1, -0.1):
            with self.assertRaises(ValueError):
                sm.SplitDesign(strategy="stratified", share=share)
        pinned = self.built.holdout.split
        self.assertEqual((pinned.share, pinned.seed), (0.2, 0))  # the fraction and the seed are in the design, not each leaf's own

    def test_a_workflow_that_does_not_group_or_split_as_the_map_does_is_refused(self):
        b = self.built.blocks[0]
        no_group = [s for s in b.workflow.steps if not s.id.startswith("group_folds_")]
        text = self.issues(with_block(self.built, 0, workflow=b.workflow.model_copy(update={"steps": no_group})))
        self.assertIn("does not group the folds by account_id", text)
        no_time = [s for s in b.workflow.steps if s.id != "time_split"]
        self.assertIn("does not split by time", self.issues(with_block(self.built, 0, workflow=b.workflow.model_copy(update={"steps": no_time}))))

    def test_more_than_four_entity_columns_or_several_time_columns_are_noted_not_dropped_silently(self):
        import pandas as pd

        frame = pd.DataFrame({f"e{i}_id": [j % (3 + i) for j in range(90)] for i in range(7)})
        frame["created_at"] = pd.date_range("2026-01-01", periods=90, freq="h")
        frame["updated_at"] = pd.date_range("2026-02-01", periods=90, freq="h")
        frame["y"] = [i % 2 for i in range(90)]
        card = dc.card_of({"t": frame}, name="t", target="y")
        notes: list[str] = []
        split = sm.split_for(card, notes)
        self.assertLessEqual(len(split.keys), 4)
        if len([c for c in card.tables[0].columns if c.purpose == "entity"]) > 4:
            self.assertTrue(any("entity columns" in n for n in notes), notes)
        if len([c for c in card.tables[0].columns if c.purpose == "time"]) > 1:
            self.assertTrue(any("time columns" in n for n in notes), notes)
        built = sm.solution_map(pm.decompose(CHURN), sol.routes(pm.decompose(CHURN), card, detected="binary"), card)
        self.assertEqual(built.notes[:len(notes)], notes)

    def test_a_split_by_time_and_entity_is_noted_as_a_combined_split_a_person_decides(self):
        built = sm.solution_map(self.m, sol.routes(self.m, data()), data())
        self.assertTrue(any("time and by entity together" in n for n in built.notes))

    def test_a_finding_that_would_lock_the_holdout_again_is_a_blocker_and_a_workflow_that_does_it_is_refused(self):
        m = pm.decompose(FRAUD)
        findings = [{"id": "F-entities-biker_id", "topic": "entities", "severity": "warning", "evidence": {"column": "biker_id"}}]
        built = sm.solution_map(m, sol.routes(m, None), None, findings=findings)  # a card that names no entity: the finding is a new split
        self.assertEqual(len(built.blockers), 4)
        self.assertIn("locks the holdout again", built.blockers[0])
        self.assertIn("locks the holdout again on its own", self.issues(built))

    def test_two_blocks_cannot_answer_one_sub_problem(self):
        self.assertIn("answer the same sub-problem", self.issues(fresh(self.built.model_copy(update={"blocks": [self.built.blocks[0], self.built.blocks[0].model_copy(update={"id": "blk9"}), *self.built.blocks[2:]]}))))

    def test_a_single_problem_has_no_steps_and_scores_its_own_holdout(self):
        m = pm.decompose(CHURN)
        built = sm.solution_map(m, sol.routes(m, None, detected="binary"), None)
        self.assertIn("no composition steps", self.issues(fresh(built.model_copy(update={"steps": [sm.SolutionStep(kind="sequence", inputs=["sp1"], output="x")]}))))
        self.assertIn("scores its own holdout once", self.issues(with_block(built, 0, final="deferred", defers=["final_once"])))

    def test_the_fingerprint_follows_the_content_of_a_workflow_step_a_title_and_a_note(self):
        b = self.built.blocks[0]
        step = next(s for s in b.workflow.steps if s.id.startswith("group_folds_"))
        changed = b.workflow.model_copy(update={"steps": [s.model_copy(update={"columns": ["other_column"]}) if s.id == step.id else s for s in b.workflow.steps]})
        self.assertNotEqual(with_block(self.built, 0, workflow=changed).fingerprint, self.built.fingerprint)
        self.assertNotEqual(with_block(self.built, 0, title="Another title").fingerprint, self.built.fingerprint)
        self.assertNotEqual(with_step(self.built, 0, note="another note").fingerprint, self.built.fingerprint)

    def test_every_module_of_the_workflow_and_the_map_imports_first_in_a_fresh_interpreter(self):
        import subprocess

        for module in ("dclab_core.solution_workflow", "dclab_core.solution_map", "dclab_core.problem_map", "dclab_core.solutions", "dclab_core.mapping"):
            done = subprocess.run([sys.executable, "-c", f"import {module}"], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, f"{module}: {done.stderr[-300:]}")

    def test_a_map_changed_after_it_was_made_has_a_stale_fingerprint(self):
        changed = self.built.model_copy(update={"steps": self.built.steps[:-1]})
        self.assertIn("fingerprint is stale", self.issues(changed))

    def test_it_is_checked_against_the_problem_map_it_answers(self):
        self.assertEqual(sm.validate(self.built, self.m), [])
        other = pm.decompose("Route each new support ticket to the team that should handle it, then set how urgent it is.")
        text = self.issues(self.built, other)
        self.assertIn("answers another problem map", text)
        self.assertIn("the blocks answer", text)
        edited = pm.ProblemMap.model_validate({**self.m.model_dump(mode="json"), "fingerprint": ""})  # the same map, made again: the same fingerprint
        self.assertEqual(sm.validate(self.built, edited), [])
        wrong_family = fresh(self.built.model_copy(update={"blocks": [self.built.blocks[0].model_copy(update={"family": "ranking"}), *self.built.blocks[1:]]}))
        self.assertIn("does not match", self.issues(wrong_family, self.m))


if __name__ == "__main__":
    unittest.main()
