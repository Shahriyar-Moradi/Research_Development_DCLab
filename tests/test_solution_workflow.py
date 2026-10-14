"""Package 20.6: the solution workflow from the two cards. For every family the workflow keeps its safety steps (the
holdout locked first, the training rows analysed, a leakage audit before any model, the choice on validation, a
person's gate, one read of the holdout, the limits reported) and the family's own steps; the findings add theirs (a
split by entity, time or area, a leak kept out, the labels joined); a model's proposal is used only when it validates,
and a planted invalid one (the holdout before the choice, no leakage audit) is refused."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core import solution_workflow as sw
    from dclab_core.card import FAMILIES
    from dclab_core.contracts import CONTRACTS
except ImportError as error:  # the agent environment has no pydantic
    MISSING = str(error)
else:
    MISSING = ""


def card(family, labels=None):
    return {"family": {"value": family}, "labels": {"value": labels}}


@unittest.skipIf(MISSING, MISSING)
class TemplateTest(unittest.TestCase):
    def test_every_family_has_its_safety_steps_and_its_own(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                wf = sw.solution_workflow(card(family))
                ids = [s.id for s in wf.steps]
                self.assertEqual([i for i in ids if i in sw.SAFETY_ORDER], list(sw.SAFETY_ORDER))  # all, in order
                for step in sw.FAMILY_STEPS.get(family, ()):
                    self.assertIn(step.id, ids)
                self.assertEqual(sw.validate(wf), [])
                self.assertTrue(all(s.check and s.rule for s in wf.steps))
        self.assertIs(CONTRACTS["solution_workflow"], sw.SolutionWorkflow)

    def findings(self):
        from dclab_core.analyse import analyse
        from dclab_core.replicas import delivery_fraud

        r = delivery_fraud.generate(seed=0)
        rides = r.rides.assign(is_fraud=r.rides["order_id"].isin(r.sample["order_id"]).astype(int))
        return analyse({"rides": rides, "sample": r.sample}, target="is_fraud", labels="sample", main="rides",
                       train_index=list(rides.index[: int(len(rides) * 0.75)]), family="rare_event", entity="biker_id")

    def test_a_split_the_card_names_comes_before_the_lock_and_one_the_analysis_finds_locks_again(self):
        findings = self.findings()
        named = {"tables": [{"role": "main", "columns": [{"name": "biker_id", "purpose": "entity"}]}, {"role": "labels", "columns": []}]}
        wf = sw.solution_workflow(card("rare_event", "file"), named, findings)
        ids = [s.id for s in wf.steps]
        self.assertLess(ids.index("group_folds_biker_id"), ids.index("lock_holdout"))  # from the card, before any analysis
        self.assertLess(ids.index("join_labels"), ids.index("lock_holdout"))
        step = next(s for s in wf.steps if s.id == "group_folds_biker_id")
        self.assertEqual(step.columns, ["biker_id"])
        self.assertTrue(set(step.rests_on) & {"F-entity-biker_id", "F-behaviour-biker_id"})  # the finding confirms it
        unnamed = sw.solution_workflow(card("rare_event", "file"), {"tables": [{"role": "main", "columns": []}]}, findings)
        relock = next(s for s in unnamed.steps if s.id == "relock_holdout")  # found only by the analysis: lock again, read again
        ids = [s.id for s in unnamed.steps]
        self.assertGreater(ids.index("relock_holdout"), ids.index("train_only_analysis"))
        self.assertIn("biker_id", relock.columns)
        self.assertEqual(sw.validate(unnamed), [])

    def test_leaks_are_one_step_naming_their_real_columns(self):
        leaks = [{"id": f"F-leak-{c}", "topic": "leakage", "severity": "critical", "evidence": {"column": c}}
                 for c in ("مبلغ_نهایی", "هزینه_نهایی", "Final Fare", "final_fare", "x" * 500)]
        leaks += [{"id": "F-leak-sum-base_fare-tip", "topic": "leakage", "severity": "critical", "evidence": {"columns": ["base_fare", "tip"]}},
                  {"id": "F-leak-recompute-a-b", "topic": "leakage", "severity": "critical", "evidence": {"later": "b"}}]
        leaks += [{"id": f"F-leak-c{i}", "topic": "leakage", "severity": "warning", "evidence": {"column": f"c{i}"}} for i in range(60)]
        wf = sw.solution_workflow(card("classification"), None, leaks)
        drop = next(s for s in wf.steps if s.id == "drop_leaks")
        self.assertTrue({"مبلغ_نهایی", "هزینه_نهایی", "Final Fare", "final_fare", "base_fare", "tip", "b"} <= set(drop.columns) | {
            c for c in drop.columns})
        self.assertNotIn("sum-base_fare-tip", drop.columns)
        self.assertEqual(sw.validate(wf), [])

    def test_a_family_not_settled_is_refused_never_guessed(self):
        from dclab_core.card import problem_card
        from dclab_core.contracts import ProblemBrief

        unsettled = problem_card(ProblemBrief(), words="I want build anomaly detection")
        with self.assertRaises(ValueError):
            sw.solution_workflow(unsettled)
        with self.assertRaises(ValueError):
            sw.solution_workflow(card("nonsense"))

    def test_the_real_cards_are_read(self):
        from dclab_core import datacard
        from dclab_core.replicas import subscription_churn

        frame = subscription_churn.generate(seed=0).subscribers
        data = datacard.card_of({"subscribers": frame}, target="churned")
        wf = sw.solution_workflow(card("classification"), data)
        self.assertEqual(sw.validate(wf), [])


@unittest.skipIf(MISSING, MISSING)
class ProposalTest(unittest.TestCase):
    def setUp(self):
        self.wf = sw.solution_workflow(card("classification"))

    def steps(self):
        return [s.model_dump() for s in self.wf.steps]

    def test_the_holdout_read_before_the_choice_is_refused(self):
        steps = self.steps()
        final = next(s for s in steps if s["id"] == "final_once")
        steps.remove(final)
        steps.insert([s["id"] for s in steps].index("choose_on_validation"), final)
        kept, problems = sw.propose(self.wf, {"steps": steps})
        self.assertTrue(problems)
        self.assertEqual([s.id for s in kept.steps], [s.id for s in self.wf.steps])  # the current one stays
        self.assertTrue(any("before" in p for p in problems))

    def test_a_leakage_audit_after_the_choice_is_refused(self):
        steps = self.steps()
        audit = next(s for s in steps if s["id"] == "leakage_audit")
        steps.remove(audit)
        steps.insert([s["id"] for s in steps].index("choose_on_validation") + 1, {**audit, "block": "WF-07"})
        _, problems = sw.propose(self.wf, steps)
        self.assertIn("'leakage_audit' must come before 'choose_on_validation'", problems)

    def test_a_workflow_without_a_leakage_audit_is_refused(self):
        steps = [s for s in self.steps() if s["id"] != "leakage_audit"]
        _, problems = sw.propose(self.wf, steps)
        self.assertIn("the safety step 'leakage_audit' is missing", problems)

    def test_a_step_reading_the_holdout_before_the_gate_or_a_second_read_is_refused(self):
        steps = self.steps()
        peek = {**steps[3], "id": "peek", "inputs": ["holdout"], "block": "WF-07"}
        steps.insert(5, peek)
        _, problems = sw.propose(self.wf, steps)
        self.assertTrue(any("read by the final score alone" in p for p in problems))
        no_gate = [{**s, "gate": False} if s["id"] == "holdout_gate" else s for s in self.steps()]
        _, problems = sw.propose(self.wf, no_gate)
        self.assertIn("the holdout gate is not a person's gate", problems)

    def test_the_same_violations_written_otherwise_are_refused(self):
        steps = self.steps()
        hollow = [{**s, "inputs": [], "outputs": [], "check": "none", "decision": None} if s["id"] == "leakage_audit" else s for s in steps]
        self.assertTrue(any("was changed" in p for p in sw.propose(self.wf, hollow)[1]))  # the content is the rule, not the name
        renamed = [{**s, "inputs": ["holdout rows"]} if s["id"] == "choose_on_validation" else s for s in steps]
        self.assertTrue(sw.propose(self.wf, renamed)[1])  # not an artefact of the list: not a workflow
        fit = {"id": "fit_gbm", "block": "WF-05", "name": "Fit", "purpose": "p", "decision": "model.families", "check": "c", "rule": "r",
               "inputs": ["training rows"], "outputs": []}
        early = list(steps)
        early.insert([s["id"] for s in early].index("leakage_audit"), fit)
        self.assertTrue(any("before the leakage audit" in p for p in sw.propose(self.wf, early)[1]))
        sneak = {**fit, "id": "fit_early", "block": "WF-03", "decision": None, "inputs": ["table", "training rows"]}
        before = list(steps)
        before.insert(0, sneak)
        self.assertTrue(any("before the holdout is locked" in p for p in sw.propose(self.wf, before)[1]))
        retune = {**fit, "id": "retune", "block": "WF-10", "inputs": ["holdout score"], "outputs": ["chosen model"]}
        after = list(steps)
        after.insert([s["id"] for s in after].index("final_once") + 1, retune)
        problems = sw.propose(self.wf, after)[1]
        self.assertTrue(any("only the choice on validation chooses" in p for p in problems))
        self.assertTrue(any("may only report it" in p for p in problems))
        back = [{**s, "block": "WF-01"} if s["id"] == "holdout_gate" else s for s in steps]
        self.assertTrue(sw.propose(self.wf, back)[1])

    def test_a_step_in_an_earlier_block_never_comes_later(self):
        steps = self.steps()
        note = {"id": "note_sources", "block": "WF-02", "name": "Note the sources", "purpose": "p", "decision": None, "check": "c",
                "rule": "r", "inputs": ["training rows"], "outputs": []}
        steps.insert([s["id"] for s in steps].index("leakage_audit") + 1, note)
        self.assertTrue(any("never goes back" in p for p in sw.propose(self.wf, steps)[1]))

    def test_a_finding_no_step_answers_is_refused(self):
        claimed = self.wf.model_copy(update={"findings_used": ["F-leak-final_fare"]})
        self.assertIn("findings no step answers: F-leak-final_fare", sw.validate(claimed))

    def test_a_proposal_cannot_drop_what_the_findings_asked(self):
        leak = [{"id": "F-leak-final_fare", "topic": "leakage", "severity": "critical", "evidence": {"column": "final_fare"}}]
        wf = sw.solution_workflow(card("classification"), None, leak)
        without = [s.model_dump() for s in wf.steps if s.id != "drop_leaks"]
        problems = sw.propose(wf, without)[1]
        self.assertTrue(any("drop_leaks" in p for p in problems) or any("no step answers" in p for p in problems))

    def test_a_valid_proposal_is_used_and_marked_as_the_models(self):
        steps = self.steps()
        steps.insert(3, {"id": "explain_columns", "block": "WF-04", "name": "Explain the columns", "purpose": "Say what each column means.",
                         "decision": None, "check": "Every column has a line", "rule": "a person reads the data before the model"})
        used, problems = sw.propose(self.wf, steps)
        self.assertEqual(problems, [])
        self.assertEqual(used.source, "model")
        self.assertIn("explain_columns", [s.id for s in used.steps])
        _, problems = sw.propose(self.wf, [{"id": "x"}])  # not a workflow at all
        self.assertTrue(problems)


if __name__ == "__main__":
    unittest.main()
