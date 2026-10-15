"""Package 24.1: the problem map. A problem as parts and sub-problems with a composition; rules draw it from the person's
words (the owner's fraud problem is four parts, a churn question is one) and judge any map, whoever drew it: a sub-problem
that reads another's outcome before it is known is a leak and is refused with the reason, a read across units says its key,
a cycle is refused, a composition is built in order from what exists and leaves nothing solved out, the fingerprint follows the
structure, and the size matches the parts."""

import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core import problem_map as pm
except ImportError as error:  # pydantic is not in every environment
    MISSING = str(error)
else:
    MISSING = ""

FRAUD = ("Find the fake rides and fake orders our bikers and customers make. We have a few confirmed cases, nothing else is labelled, "
         "and the team can review 200 a day.")
CHURN = "I want to know which subscribers will cancel at their renewal."
ROUTING = "Route each new support ticket to the team that should handle it, then set how urgent it is."
DEMAND = "Forecast the units each store sells of each product over the next 7 days; promotions are planned weeks ahead."


def edit(m, **changes):
    """A copy of a map with its fields changed and nothing re-checked: the rules must judge it, not the constructor."""
    return m.model_copy(update=changes)


def sub(m, sid, **changes):
    return edit(m, sub_problems=[s.model_copy(update=changes) if s.id == sid else s for s in m.sub_problems])


def step(m, index, **changes):
    steps = [c.model_copy(update=changes) if i == index else c for i, c in enumerate(m.composition.steps)]
    return edit(m, composition=m.composition.model_copy(update={"steps": steps}))


def fresh(m):
    """The same map made again, so its fingerprint is current (what a person's edit does)."""
    return pm.ProblemMap.model_validate({**m.model_dump(mode="json"), "fingerprint": ""})


@unittest.skipIf(MISSING, MISSING)
class FraudTest(unittest.TestCase):
    def test_the_fraud_problem_is_four_parts_with_their_families(self):
        m = pm.decompose(FRAUD)
        self.assertEqual((m.size, m.template, len(m.parts)), ("complex", "fraud", 4))
        self.assertEqual([s.family for s in m.sub_problems], ["anomaly", "rare_event", "entity_tables", "ranking"])
        self.assertEqual([s.labels for s in m.sub_problems][:2], ["none", "file"])
        self.assertEqual(pm.validate(m), [])

    def test_the_composition_fuses_the_scores_on_a_key_then_applies_the_budget_to_the_review_list(self):
        m = pm.decompose(FRAUD)
        self.assertEqual([c.kind for c in m.composition.steps], ["fusion", "budget"])
        self.assertEqual(m.composition.steps[0].inputs, ["sp1", "sp2", "sp3"])
        self.assertEqual(m.composition.steps[0].joins, "the account of the ride")
        self.assertEqual(m.composition.steps[1].inputs, ["risk", "sp4"])
        self.assertIn("200", m.composition.budget or "")

    def test_the_classifier_reads_the_screens_score_and_no_part_reads_a_late_outcome(self):
        m = pm.decompose(FRAUD)
        by = {s.id: s for s in m.sub_problems}
        self.assertEqual([(r.sub_problem, r.what) for r in by["sp2"].reads], [("sp1", "score")])
        self.assertTrue(all(r.what == "score" for s in m.sub_problems for r in s.reads))

    def test_the_unit_is_what_the_words_name_not_always_a_ride(self):
        m = pm.decompose("Find fraudulent insurance claims. We have some confirmed cases and the team can review 40 a day.")
        self.assertEqual({p.unit for p in m.parts}, {"claim", "account"})
        self.assertEqual(m.sub_problems[0].title, "Unusual claims")
        self.assertNotIn("ride", m.model_dump_json().lower())
        self.assertEqual(pm.validate(m), [])
        self.assertEqual(pm.decompose("Find fake deliveries made by our couriers.").sub_problems[0].unit, "delivery")
        self.assertEqual(pm.decompose("Find fraud among our drivers.").sub_problems[0].unit, "driver")

    def test_a_problem_on_accounts_needs_no_key(self):
        m = pm.decompose("Find the fraud in our accounts.")
        self.assertEqual(pm.validate(m), [])
        self.assertIsNone(m.composition.steps[0].joins)

    def test_a_budget_is_found_in_the_ways_people_say_it(self):
        for words in ["our reviewers can check about 50 rides a day", "the team can review 200 a day", "we have a review budget of 30",
                      "only 40 cases a day can be handled, so top 40 matter"]:
            self.assertTrue(pm.decompose(f"Find the fake rides our bikers make. {words}").composition.budget, words)

    def test_no_budget_in_the_words_is_none_and_still_a_budget_step(self):
        m = pm.decompose("Find the fake rides our bikers make.")
        self.assertIsNone(m.composition.budget)
        self.assertEqual(m.composition.steps[-1].kind, "budget")

    def test_an_absurd_budget_phrase_is_clipped_not_a_crash(self):
        for words in ["review " + "9" * 700, "top" + "9" * 700, "the team can review " + "x" * 700 + " hundred"]:
            m = pm.decompose(f"Find the fake rides our bikers make. {words}")
            self.assertLessEqual(len(m.composition.budget or ""), pm.MAX_BUDGET if hasattr(pm, "MAX_BUDGET") else 200)
            self.assertEqual(pm.validate(m), [], words[:30])

    def test_confirmed_cases_that_exist_make_a_classifier_and_a_denial_does_not(self):
        yes = ["We have a few confirmed cases, nothing else is labelled.", "Some cases are known."]
        no = ["None of them were confirmed.", "We have not confirmed any yet.", "There are no known cases.", "Nothing has ever been confirmed."]
        for words in yes:
            m = pm.decompose(f"Find the fake rides our bikers make. {words}")
            self.assertEqual([s.labels for s in m.sub_problems][:2], ["none", "file"], words)
        for words in no:
            m = pm.decompose(f"Find the fake rides our bikers make. {words}")
            self.assertEqual([s.family for s in m.sub_problems], ["anomaly", "entity_tables", "ranking"], words)
            self.assertEqual(pm.validate(m), [], words)
            self.assertIn("no case was confirmed", " ".join(m.notes))

    def test_when_the_words_do_not_say_the_classifier_stays_with_its_labels_unknown(self):
        m = pm.decompose("Find the fake rides our bikers make.")
        self.assertEqual([s.family for s in m.sub_problems], ["anomaly", "rare_event", "entity_tables", "ranking"])
        self.assertEqual(m.sub_problems[1].labels, "unknown")
        self.assertIn("question", " ".join(m.notes))

    def test_a_sentence_that_only_shares_words_with_fraud_is_not_given_the_template(self):
        for words in ["Detect abusive comments in orders' reviews", "Predict hospital claims cost; abuse of the first-aid rule",
                      "Predict which transactions are fraudulent; the column is_fraud is the label.",
                      "Predict which payments are fake. The label is the column is_fake."]:
            m = pm.decompose(words)
            self.assertEqual((m.size, m.template), ("single", "single"), words)
            self.assertEqual(pm.validate(m), [], words)


@unittest.skipIf(MISSING, MISSING)
class OtherShapesTest(unittest.TestCase):
    def test_a_churn_question_is_one_problem(self):
        m = pm.decompose(CHURN)
        self.assertEqual((m.size, m.template, len(m.parts), len(m.sub_problems)), ("single", "single", 1, 1))
        self.assertEqual((m.composition.method, m.composition.steps), ("none", []))
        self.assertEqual(pm.validate(m), [])
        self.assertIn("cancel", m.sub_problems[0].quote)

    def test_ticket_routing_in_two_steps_is_a_sequence_that_reads_a_score_not_an_outcome(self):
        m = pm.decompose(ROUTING)
        self.assertEqual((m.size, m.template, m.composition.method), ("complex", "ticket_routing", "sequence"))
        self.assertEqual([(r.sub_problem, r.what) for r in m.sub_problems[1].reads], [("sp1", "score")])
        self.assertEqual(m.sub_problems[0].kind, "multiclass")
        self.assertEqual(pm.validate(m), [])

    def test_a_routing_sentence_must_ask_the_team_and_then_the_urgency(self):
        for words in ["Route each new support ticket to the team that should handle it.",
                      "Predict the priority of each support ticket at creation, using what is known before it is routed.",
                      "Route each ticket to the team and set its urgency.",  # no 'then': one question or two is the person's to say
                      "Set how urgent each ticket is, then route it to the team that should handle it."]:  # reversed order: not drawn by the rule
            self.assertEqual(pm.decompose(words).size, "single", words)

    def test_a_demand_forecast_with_planned_promotions_is_one_forecast_that_needs_the_plan(self):
        m = pm.decompose(DEMAND)
        self.assertEqual((m.size, m.template, m.sub_problems[0].family), ("single", "demand_with_promotions", "forecasting"))
        self.assertIn("features", " ".join(m.notes))
        self.assertEqual(len(m.sub_problems[0].needs), 1)
        self.assertIn("planned", m.sub_problems[0].needs[0].why)

    def test_promotions_whose_timing_is_not_said_become_a_question_and_no_need(self):
        m = pm.decompose("Forecast how many units we will sell per store next week, we run promotions sometimes.")
        self.assertIn("question", " ".join(m.notes))
        self.assertEqual(m.sub_problems[0].needs, [])

    def test_the_card_family_is_never_overridden_by_a_template(self):
        for words in ["Which customers who got the discount will place orders again?", "Which customers will cancel after the retention offer? We have their sales history."]:
            m = pm.decompose(words)
            self.assertEqual(m.template, "single", words)
            self.assertNotEqual(m.sub_problems[0].family, "forecasting", words)

    def test_the_rules_split_none_of_the_dev_journeys_single_problems(self):
        journeys = json.loads((ROOT / "dclab_rnd" / "agent_eval" / "journeys" / "journeys_dev.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(journeys), 15)
        for journey in journeys:
            m = pm.decompose(journey["words"])
            self.assertEqual(m.size, "single", f"{journey['id']}: {journey['words']}")
            self.assertEqual(pm.validate(m), [], journey["id"])

    def test_a_brief_with_parts_becomes_those_parts_in_its_order(self):
        parts = [{"title": "Part 1: churn", "text": "Which customers will cancel next month?"},
                 {"title": "Part 2: demand", "text": "Forecast the weekly units sold per store."}]
        m = pm.decompose("", parts=parts)
        self.assertEqual((m.size, m.template), ("complex", "brief"))
        self.assertEqual([p.title for p in m.parts], ["Part 1: churn", "Part 2: demand"])
        self.assertEqual([s.family for s in m.sub_problems], ["classification", "forecasting"])
        self.assertEqual(pm.validate(m), [])

    def test_one_part_in_a_brief_is_not_a_complex_problem(self):
        self.assertEqual(pm.decompose(CHURN, parts=[{"title": "Part 1", "text": CHURN}]).size, "single")

    def test_a_brief_of_many_parts_is_read_and_one_of_too_many_is_cut_with_a_note(self):
        many = lambda n: [{"title": f"Part {i}", "text": "Which customers will cancel next month?"} for i in range(1, n + 1)]  # noqa: E731
        thirteen = pm.decompose("", parts=many(13))
        self.assertEqual((len(thirteen.parts), pm.validate(thirteen)), (13, []))
        cut = pm.decompose("", parts=many(25))
        self.assertEqual(len(cut.parts), pm.MAX_PARTS)
        self.assertIn("first 20", " ".join(cut.notes))
        self.assertEqual(pm.validate(cut), [])

    def test_a_brief_part_that_is_not_a_part_is_skipped_not_a_crash(self):
        m = pm.decompose("Which customers will cancel?", parts=["Part 1", None, 3])
        self.assertEqual(m.size, "single")
        m = pm.decompose("", parts=["junk", {"title": "A", "text": "Which customers will cancel?"}, 5, {"title": "B", "text": "Forecast the weekly units sold."}])
        self.assertEqual([p.title for p in m.parts], ["A", "B"])

    def test_empty_whitespace_and_very_long_words_do_not_break_it(self):
        self.assertEqual(pm.decompose("").size, "single")
        self.assertEqual(pm.decompose("   \n ").size, "single")
        long = pm.decompose("Which subscribers will cancel? " * 400)
        self.assertEqual(pm.validate(long), [])
        self.assertLessEqual(len(long.sub_problems[0].quote), pm.MAX_TEXT)

    def test_a_blank_title_is_refused_by_the_contract(self):
        m = pm.decompose(CHURN)
        with self.assertRaises(ValueError):
            pm.Part(id="part1", title="   ")
        with self.assertRaises(ValueError):
            pm.SubProblem(id="sp1", part="part1", title="\n", family="classification")
        self.assertEqual(pm.Part(id="part1", title="  A title  ").title, "A title")
        self.assertEqual(m.parts[0].title, "The problem")

    def test_the_same_words_give_the_same_map(self):
        self.assertEqual(pm.decompose(FRAUD).fingerprint, pm.decompose(FRAUD).fingerprint)
        self.assertNotEqual(pm.decompose(FRAUD).fingerprint, pm.decompose(CHURN).fingerprint)


@unittest.skipIf(MISSING, MISSING)
class ValidateTest(unittest.TestCase):
    def setUp(self):
        self.fraud = pm.decompose(FRAUD)

    def issues(self, m):
        return "\n".join(pm.validate(m))

    def test_reading_the_outcome_of_a_later_sub_problem_is_a_leak_and_says_why(self):
        m = fresh(sub(self.fraud, "sp1", reads=[pm.Read(sub_problem="sp2", what="outcome")]))
        text = self.issues(m)
        self.assertIn("sp1 reads the outcome of sp2", text)
        self.assertIn("leak", text)

    def test_an_outcome_that_is_never_known_in_time_cannot_be_read(self):
        m = fresh(sub(self.fraud, "sp4", reads=[pm.Read(sub_problem="sp1", what="outcome")]))
        self.assertIn("never in time", self.issues(m))

    def test_an_outcome_known_before_the_reader_predicts_is_fine(self):
        m = fresh(sub(sub(self.fraud, "sp2", known_rank=2), "sp4", predict_rank=3, known_rank=4, reads=[pm.Read(sub_problem="sp2", what="outcome")]))
        self.assertEqual(pm.validate(m), [])

    def test_an_outcome_known_at_the_readers_own_moment_is_not_before_it(self):
        m = fresh(sub(sub(self.fraud, "sp2", known_rank=2), "sp4", predict_rank=2, known_rank=3, reads=[pm.Read(sub_problem="sp2", what="outcome")]))
        self.assertIn("not before sp4's own moment", self.issues(m))

    def test_an_outcome_known_no_later_than_its_own_prediction_is_nothing_to_predict(self):
        for known in (1, 2):
            self.assertIn("nothing to predict", self.issues(fresh(sub(self.fraud, "sp4", predict_rank=2, known_rank=known))), known)

    def test_a_sub_problem_with_no_label_has_no_outcome_that_becomes_known(self):
        self.assertIn("says no label exists", self.issues(fresh(sub(self.fraud, "sp1", known_rank=3))))

    def test_reading_a_score_made_later_is_refused(self):
        m = fresh(sub(self.fraud, "sp2", reads=[pm.Read(sub_problem="sp4", what="score")]))
        self.assertIn("only made at rank 2", self.issues(m))

    def test_a_cycle_of_reads_is_named(self):
        m = fresh(sub(sub(self.fraud, "sp1", reads=[pm.Read(sub_problem="sp2", what="score")]), "sp2", reads=[pm.Read(sub_problem="sp1", what="score")]))
        self.assertIn("a cycle of reads", self.issues(m))

    def test_a_sub_problem_cannot_read_itself_or_one_that_is_not_there(self):
        text = self.issues(fresh(sub(self.fraud, "sp1", reads=[pm.Read(sub_problem="sp1", what="score"), pm.Read(sub_problem="sp9", what="score")])))
        self.assertIn("reads itself", text)
        self.assertIn("sp9, which is not in this map", text)

    def test_a_read_across_units_says_its_key_and_the_leak_is_reported_beside_it(self):
        m = fresh(sub(self.fraud, "sp2", reads=[pm.Read(sub_problem="sp3", what="outcome")]))  # a ride reads an account, no key, and an outcome never known
        text = self.issues(m)
        self.assertIn("say the key that joins them", text)
        self.assertIn("leak", text)

    def test_the_effective_unit_is_the_sub_problems_own_else_its_parts(self):
        rides = fresh(sub(sub(self.fraud, "sp2", unit=None), "sp2", reads=[pm.Read(sub_problem="sp3", what="score")]))  # sp2 is on its part's unit, the ride
        self.assertIn("say the key that joins them", self.issues(rides))
        keyed = fresh(sub(self.fraud, "sp2", reads=[pm.Read(sub_problem="sp3", what="score", via="the account of the ride")]))
        self.assertEqual(pm.validate(keyed), [])

    def test_a_sub_problem_must_work_on_its_parts_unit_and_moment(self):
        self.assertIn("one part, one unit", self.issues(fresh(sub(self.fraud, "sp1", unit="account"))))
        said = fresh(self.fraud.model_copy(update={"parts": [p.model_copy(update={"moment": "After the ride"}) if p.id == "part1" else p for p in self.fraud.parts]}))
        self.assertIn("moment differs", self.issues(fresh(sub(said, "sp1", moment="Before the ride"))))
        self.assertEqual(pm.validate(fresh(sub(said, "sp1", moment="After the ride"))), [])

    def test_a_missing_unit_or_moment_is_a_question_for_the_reviewer_not_an_unsound_map(self):
        self.assertEqual(pm.validate(pm.decompose(CHURN)), [])
        self.assertIsNone(pm.decompose(CHURN).sub_problems[0].unit)

    def test_a_part_without_a_sub_problem_is_named(self):
        m = fresh(edit(self.fraud, sub_problems=[s for s in self.fraud.sub_problems if s.part != "part3"]))
        self.assertIn("part3", self.issues(m))

    def test_a_sub_problem_of_a_part_that_is_not_there_is_named(self):
        self.assertIn("not a part of this map", self.issues(fresh(sub(self.fraud, "sp1", part="part9"))))

    def test_duplicate_ids_are_named(self):
        m = fresh(edit(self.fraud, sub_problems=[*self.fraud.sub_problems, self.fraud.sub_problems[0]]))
        self.assertIn("same id", self.issues(m))

    def test_the_size_must_match_the_parts_and_a_single_problem_has_nothing_to_compose(self):
        self.assertIn("make a complex problem", self.issues(fresh(edit(self.fraud, size="single"))))
        one = pm.decompose(CHURN)
        self.assertIn("single problem, not a complex", self.issues(fresh(edit(one, size="complex"))))
        steps = [pm.CompositionStep(kind="sequence", inputs=["sp1"], output="x")]
        self.assertIn("nothing to compose", self.issues(fresh(edit(one, composition=pm.Composition(method="sequence", steps=steps)))))

    def test_several_sub_problems_need_a_composition(self):
        self.assertIn("need a composition", self.issues(fresh(edit(self.fraud, composition=pm.Composition()))))


@unittest.skipIf(MISSING, MISSING)
class CompositionTest(unittest.TestCase):
    def setUp(self):
        self.fraud = pm.decompose(FRAUD)

    def issues(self, m):
        return "\n".join(pm.validate(fresh(m)))

    def test_a_step_may_use_an_earlier_steps_output_and_not_a_later_ones(self):
        self.assertEqual(pm.validate(self.fraud), [])
        steps = list(reversed(self.fraud.composition.steps))
        text = self.issues(edit(self.fraud, composition=self.fraud.composition.model_copy(update={"steps": steps})))
        self.assertIn("which a later step makes: steps are in order", text)

    def test_a_step_cannot_feed_on_its_own_output_or_another_that_feeds_on_it(self):
        self.assertIn("neither a sub-problem nor an earlier step's output", self.issues(step(self.fraud, 0, inputs=["sp1", "sp2", "risk"])))
        a = pm.CompositionStep(kind="fusion", inputs=["sp1", "b"], output="a", joins="k")
        b = pm.CompositionStep(kind="fusion", inputs=["sp2", "a"], output="b", joins="k")
        text = self.issues(edit(self.fraud, composition=pm.Composition(method="mixed", steps=[a, b])))
        self.assertIn("uses 'b'", text)

    def test_an_output_cannot_be_a_sub_problems_id_or_an_earlier_output(self):
        self.assertIn("sub-problem's id", self.issues(step(self.fraud, 0, output="sp3")))
        self.assertIn("an earlier step already made", self.issues(step(self.fraud, 1, output="risk")))

    def test_a_step_needs_inputs_and_a_fusion_two(self):
        self.assertIn("has no inputs", self.issues(step(self.fraud, 1, inputs=[])))
        self.assertIn("at least two", self.issues(step(self.fraud, 0, inputs=["sp1"])))
        self.assertIn("lists an input twice", self.issues(step(self.fraud, 0, inputs=["sp1", "sp1", "sp2"])))

    def test_inputs_on_different_units_say_the_key_that_joins_them(self):
        text = self.issues(step(self.fraud, 0, joins=None))
        self.assertIn("different units (account, ride)", text)
        self.assertIn("say the key that joins them", text)

    def test_the_joined_result_is_on_the_first_inputs_unit(self):
        # the fusion is on rides (its first input is the screen), so the budget step over it and the review list needs no key
        self.assertEqual(pm.validate(self.fraud), [])
        m = step(self.fraud, 1, inputs=["sp3", "risk"])  # an account first: the result is on accounts, and the review list is on rides again
        self.assertIn("different units", self.issues(m))

    def test_a_sub_problem_nobody_reads_must_feed_a_step(self):
        left_out = step(self.fraud, 1, inputs=["risk"])  # the review list (sp4) is no longer used
        self.assertIn("sp4 (The review list) is read by no sub-problem and feeds no composition step", self.issues(left_out))
        only_first = edit(self.fraud, composition=pm.Composition(method="mixed", steps=[pm.CompositionStep(kind="sequence", inputs=["sp1"], output="x")]))
        text = self.issues(only_first)
        for sid in ("sp2", "sp3", "sp4"):  # nobody reads them and no step uses them
            self.assertIn(f"{sid} (", text)
        self.assertNotIn("sp1 (", text)  # sp2 reads it, and the step uses it

    def test_a_brief_of_parts_is_one_sequence_that_uses_every_part(self):
        m = pm.decompose("", parts=[{"title": f"Part {i}", "text": "Which customers will cancel?"} for i in range(1, 6)])
        self.assertEqual(m.composition.steps[0].inputs, ["sp1", "sp2", "sp3", "sp4", "sp5"])
        self.assertEqual(pm.validate(m), [])


@unittest.skipIf(MISSING, MISSING)
class FingerprintTest(unittest.TestCase):
    def setUp(self):
        self.fraud = pm.decompose(FRAUD)

    def test_the_fingerprint_is_the_structure_not_the_wording(self):
        m = pm.decompose(ROUTING)
        reworded = pm.ProblemMap.model_validate({**m.model_dump(mode="json"), "fingerprint": "",
                                                 "parts": [{**p.model_dump(mode="json"), "title": "Another title", "question": "Another?"} for p in m.parts]})
        self.assertEqual(reworded.fingerprint, m.fingerprint)

    def test_everything_validate_depends_on_changes_it(self):
        base = self.fraud.fingerprint
        reads = [pm.Read(sub_problem="sp1", what="score")]
        changes = {
            "a read's key": fresh(sub(self.fraud, "sp2", reads=[pm.Read(sub_problem="sp1", what="score", via="the account")])),
            "a read": fresh(sub(self.fraud, "sp2", reads=[])),
            "a unit": fresh(sub(self.fraud, "sp1", unit="ride ")),
            "a moment": fresh(sub(self.fraud, "sp1", moment="Before the ride")),
            "a need": fresh(sub(self.fraud, "sp1", needs=[pm.Need(table="rides", columns=["amount"])])),
            "a need's column": fresh(sub(self.fraud, "sp1", needs=[pm.Need(table="rides", columns=["fare"])])),
            "the budget": fresh(edit(self.fraud, composition=self.fraud.composition.model_copy(update={"budget": "top 5"}))),
            "a step's key": fresh(step(self.fraud, 0, joins="another key")),
            "a rank": fresh(sub(self.fraud, "sp4", predict_rank=3, known_rank=4)),
            "the order of a sequence": fresh(step(self.fraud, 0, inputs=["sp3", "sp2", "sp1"])),
        }
        self.assertTrue(reads)
        seen = {base}
        for what, changed in changes.items():
            self.assertNotIn(changed.fingerprint, seen, what)
            seen.add(changed.fingerprint)

    def test_a_map_changed_after_it_was_made_has_a_stale_fingerprint_that_validate_names_and_the_adapter_refuses(self):
        from dclab_core.adapters import main_product_problem_map as adapter

        changed = sub(self.fraud, "sp1", family="regression")  # model_copy skips the validators
        self.assertEqual(changed.fingerprint, self.fraud.fingerprint)
        self.assertIn("fingerprint is stale", "\n".join(pm.validate(changed)))
        with self.assertRaises(ValueError):
            adapter.problem_specs(changed)
        self.assertEqual(pm.validate(fresh(changed)), [])  # made again, it is a new map with its own fingerprint
        self.assertNotEqual(fresh(changed).fingerprint, self.fraud.fingerprint)

    def test_the_fingerprint_is_in_a_dump_without_defaults(self):
        self.assertEqual(self.fraud.model_dump(exclude_unset=True)["fingerprint"], self.fraud.fingerprint)


@unittest.skipIf(MISSING, MISSING)
class ContractTest(unittest.TestCase):
    def test_a_map_round_trips_through_json(self):
        m = pm.decompose(FRAUD)
        self.assertEqual(pm.ProblemMap.read(m.model_dump_json()), m)

    def test_a_document_without_the_right_version_is_refused(self):
        document = json.loads(pm.decompose(CHURN).model_dump_json())
        document["schema_version"] = "dclab_core.problem_map.v0"
        with self.assertRaises(ValueError):
            pm.ProblemMap.read(document)

    def test_a_map_is_never_edited_in_place_a_wrong_fingerprint_is_refused(self):
        document = json.loads(pm.decompose(CHURN).model_dump_json())
        document["fingerprint"] = "0" * 16
        with self.assertRaises(ValueError):
            pm.ProblemMap.read(document)

    def test_unknown_fields_are_refused(self):
        document = json.loads(pm.decompose(CHURN).model_dump_json())
        document["extra"] = 1
        with self.assertRaises(ValueError):
            pm.ProblemMap.read(document)

    def test_a_family_the_bands_do_not_know_is_refused(self):
        document = json.loads(pm.decompose(CHURN).model_dump_json())
        document["sub_problems"][0]["family"] = "astrology"
        with self.assertRaises(ValueError):
            pm.ProblemMap.read(document)

    def test_the_families_are_the_bands_families(self):
        from dclab_core.bands import FAMILIES
        self.assertEqual(set(pm.Family.__args__), set(FAMILIES))

    def test_nothing_in_a_map_is_unbounded_so_a_table_cannot_be_stored_in_one(self):
        import typing

        def bounded(model, seen=()):
            for name, field in model.model_fields.items():
                if name == "schema_version":
                    continue
                tp, where = field.annotation, f"{model.__name__}.{name}"
                args = [a for a in typing.get_args(tp) if a is not type(None)]
                inner = args[0] if args else tp
                if typing.get_origin(tp) in (typing.Union, types.UnionType):
                    inner = args[0]
                if typing.get_origin(inner) is list or inner is list:
                    self.assertTrue(any(getattr(m, "max_length", None) for m in field.metadata), f"{where} is an unbounded list")
                    item = typing.get_args(inner)[0] if typing.get_args(inner) else None
                    if item is str:
                        pass
                    elif isinstance(item, type) and issubclass(item, pm.Contract) and item not in seen:
                        bounded(item, seen + (item,))
                elif inner is str:
                    limited = any(getattr(m, "max_length", None) for m in field.metadata) or any(getattr(m, "pattern", None) for m in field.metadata)
                    self.assertTrue(limited, f"{where} is an unbounded string")
                elif isinstance(inner, type) and issubclass(inner, pm.Contract) and inner not in seen:
                    bounded(inner, seen + (inner,))

        bounded(pm.ProblemMap)


@unittest.skipIf(MISSING, MISSING)
class AdapterTest(unittest.TestCase):
    """The main product holds one ProblemSpec per problem and no relation between specs: one proposal per sub-problem, the places
    in the map in a free-form constraints key, the composition once, and a map that is not sound maps onto nothing."""

    def setUp(self):
        from dclab_core.adapters import main_product_problem_map as adapter
        self.adapter = adapter
        self.out = adapter.problem_specs(pm.decompose(FRAUD))

    def test_one_spec_proposal_per_sub_problem_with_the_main_products_field_names(self):
        from dclab_core.adapters import main_product as mp
        self.assertEqual(list(self.out["problem_specs"]), ["sp1", "sp2", "sp3", "sp4"])
        for spec in self.out["problem_specs"].values():
            self.assertEqual(spec["contract"], "ProblemSpec")
            self.assertEqual(list(spec["fields"]), mp.fields_of("ProblemSpec"))
            self.assertEqual(spec["status"], "proposal")
            self.assertEqual(spec["decision_point"], "spec.objective")

    def test_the_task_type_is_the_main_products_where_it_has_one_and_says_why_where_it_has_none(self):
        specs = self.out["problem_specs"]
        self.assertEqual(specs["sp2"]["fields"]["task_type"], "binary")
        self.assertIsNone(specs["sp1"]["fields"]["task_type"])
        self.assertIn("unlabelled", specs["sp1"]["notes"]["task_type"])
        self.assertIn("ranking", specs["sp4"]["notes"]["task_type"])

    def test_a_multiclass_kind_keeps_the_main_products_multiclass(self):
        out = self.adapter.problem_specs(pm.decompose(ROUTING))
        self.assertEqual([s["fields"]["task_type"] for s in out["problem_specs"].values()], ["multiclass", "multiclass"])
        self.assertEqual(self.adapter.problem_specs(pm.decompose(CHURN))["problem_specs"]["sp1"]["fields"]["task_type"], "classification")

    def test_each_spec_states_its_own_objective_and_unit(self):
        specs = self.out["problem_specs"]
        self.assertIn("review budget", specs["sp4"]["fields"]["business_objective"])
        self.assertEqual(specs["sp3"]["fields"]["prediction_unit"], "account")

    def test_the_places_in_the_map_are_in_a_free_form_constraints_key(self):
        here = self.out["problem_specs"]["sp2"]["fields"]["constraints"][self.adapter.KEY]
        self.assertEqual(here["fingerprint"], self.out["fingerprint"])
        self.assertEqual([r["sub_problem"] for r in here["reads"]], ["sp1"])
        self.assertEqual(self.out["problem_specs"]["sp1"]["fields"]["constraints"][self.adapter.KEY]["composed_by"], ["risk"])
        self.assertEqual(self.out["problem_specs"]["sp4"]["fields"]["constraints"][self.adapter.KEY]["composed_by"], ["review_list"])

    def test_the_composition_is_given_once_and_what_cannot_be_held_is_said(self):
        self.assertEqual([c["kind"] for c in self.out["composition"]["steps"]], ["fusion", "budget"])
        self.assertEqual(self.out["composition"]["steps"][0]["joins"], "the account of the ride")
        self.assertIn("composition", self.out["not_mapped"])

    def test_a_single_problem_maps_to_one_spec(self):
        out = self.adapter.problem_specs(pm.decompose(CHURN))
        self.assertEqual(list(out["problem_specs"]), ["sp1"])

    def test_a_map_that_is_not_sound_maps_onto_nothing(self):
        bad = fresh(sub(pm.decompose(FRAUD), "sp1", reads=[pm.Read(sub_problem="sp2", what="outcome")]))
        with self.assertRaises(ValueError) as caught:
            self.adapter.problem_specs(bad)
        self.assertIn("leak", str(caught.exception))

    def test_the_output_is_plain_json_and_reads_back(self):
        self.assertEqual(json.loads(json.dumps(self.out, allow_nan=False)), self.out)

    def test_the_contracts_unit_limit_is_under_the_main_products_so_nothing_is_cut(self):
        from dclab_core.adapters import main_product as mp

        field = pm.SubProblem.model_fields["unit"]
        limit = next(m.max_length for m in field.metadata if getattr(m, "max_length", None))
        self.assertLessEqual(limit, mp.recorded()["contracts"]["ProblemSpec"]["max_length"]["prediction_unit"])
        with self.assertRaises(ValueError):
            pm.SubProblem(id="sp1", part="part1", title="T", family="anomaly", unit="x" * (limit + 1))

    def test_the_largest_place_in_a_map_is_far_below_the_main_products_size_limit(self):
        from dclab_core.adapters import main_product as mp

        limit = mp.recorded()["contracts"]["ProblemSpec"]["json_max_bytes"]
        reads = [pm.Read(sub_problem=f"sp{i}", what="score", via="k" * pm.MAX_TEXT) for i in range(1, 11)]
        subs = [pm.SubProblem(id=f"sp{i}", part=f"part{i}", title="T", family="anomaly", labels="none") for i in range(1, 12)]
        subs[10] = subs[10].model_copy(update={"reads": reads})
        parts = [pm.Part(id=f"part{i}", title="T") for i in range(1, 12)]
        steps = [pm.CompositionStep(kind="sequence", inputs=[s.id for s in subs], output="all")]
        big = pm.ProblemMap(size="complex", parts=parts, sub_problems=subs, composition=pm.Composition(method="sequence", steps=steps))
        here = self.adapter.problem_specs(big)["problem_specs"]["sp11"]["fields"]["constraints"]
        self.assertLess(mp._size(here), limit // 2)

    def test_a_sub_problem_with_no_question_and_a_blank_title_still_has_an_objective(self):
        m = pm.decompose(CHURN)
        blank = sub(m, "sp1", question="", title="x")
        out = self.adapter.problem_specs(fresh(blank))
        self.assertTrue(out["problem_specs"]["sp1"]["fields"]["business_objective"])


if __name__ == "__main__":
    unittest.main()
