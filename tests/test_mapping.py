"""Package 24.2: the understanding agents of phase 2. The rules run with no model (a reviewer that asks what the map leaves unsaid, a
coverage of the data card, the person's edits); the decomposer, the map reviewer and the data-need agent are roles behind validators:
every part rests on the person's words or a rule template, a unit and a moment are only the person's words, a model's question never
blocks, and a table or a column the data card does not hold is dropped. A model that fails, abstains or is refused leaves the rules."""

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core import datacard as dc, mapping as mp, problem_map as pm
    from dclab_core.agents import ScriptedAdvisor
    from dclab_core.replicas import delivery_fraud
except ImportError as error:  # pydantic and numpy are not in every environment
    MISSING = str(error)
else:
    MISSING = ""

FRAUD = ("Find the fake rides and fake orders our bikers and customers make. We have a few confirmed cases, nothing else is labelled, "
         "and the team can review 200 a day.")
CHURN = "I want to know which subscribers will cancel at their renewal."
_DATA = {}


def data(labels=True):
    """The fraud replica's data card (its rides, and the file of known frauds)."""
    key = bool(labels)
    if key not in _DATA:
        r = delivery_fraud.generate(seed=0)
        tables = {"rides": r.rides, **({"known_frauds": r.sample} if labels else {})}
        _DATA[key] = dc.card_of(tables, name="fraud")
    return _DATA[key]


def fresh(m):
    return pm.ProblemMap.model_validate({**m.model_dump(mode="json"), "fingerprint": ""})


def sub(m, sid, **changes):
    return m.model_copy(update={"sub_problems": [s.model_copy(update=changes) if s.id == sid else s for s in m.sub_problems]})


def codes(questions):
    return [(q.code, q.target) for q in questions]


@unittest.skipIf(MISSING, MISSING)
class ReviewRulesTest(unittest.TestCase):
    def test_a_sound_single_problem_raises_nothing_the_card_owns_its_questions(self):
        self.assertEqual(mp.review_rules(pm.decompose(CHURN)), [])
        self.assertTrue(mp.review(pm.decompose(CHURN)).ready)

    def test_a_complex_map_is_asked_what_the_parts_leave_unsaid(self):
        got = codes(mp.review_rules(pm.decompose(FRAUD)))
        self.assertIn(("unmarked_is_unknown", "sp2"), got)  # only some cases were found: the rest is unknown, not clean
        self.assertNotIn("budget_missing", [c for c, _ in got])  # the words gave one
        m = pm.decompose("Find the fake rides our bikers make.")
        self.assertIn(("budget_missing", None), codes(mp.review_rules(m)))
        self.assertIn(("labels_unknown", "sp2"), codes(mp.review_rules(m)))

    def test_a_brief_of_parts_is_asked_each_parts_unit_moment_and_labels(self):
        m = pm.decompose("", parts=[{"title": "Part 1: churn", "text": "Which customers will cancel next month?"}, {"title": "Part 2: demand", "text": "Forecast weekly units per store."}])
        got = codes(mp.review_rules(m))
        for part in ("part1", "part2"):
            self.assertIn(("unit_missing", part), got)
        self.assertEqual(got.count(("moment_missing", None)), 1)  # no part says it: one question for them all
        self.assertNotIn(("moment_missing", "part1"), got)
        self.assertIn(("labels_unknown", "sp1"), got)

    def test_a_part_that_says_its_moment_while_another_does_not_is_asked_about_the_other(self):
        m = pm.decompose("", parts=[{"title": "Part 1: churn", "text": "Which customers will cancel next month?"}, {"title": "Part 2: demand", "text": "Forecast weekly units per store."}])
        m = fresh(m.model_copy(update={"parts": [p.model_copy(update={"moment": "At the start of each month"}) if p.id == "part1" else p for p in m.parts]}))
        got = codes(mp.review_rules(m))
        self.assertIn(("moment_missing", "part2"), got)
        self.assertNotIn(("moment_missing", "part1"), got)

    def test_an_unsound_map_is_a_blocking_question_with_the_reason_and_the_review_is_not_ready(self):
        bad = fresh(sub(pm.decompose(FRAUD), "sp1", reads=[pm.Read(sub_problem="sp2", what="outcome")]))
        questions = mp.review_rules(bad)
        blocking = [q for q in questions if q.blocking]
        self.assertTrue(blocking and blocking[0].kind == "leak" and "leak" in blocking[0].question)
        self.assertFalse(mp.review(bad).ready)
        self.assertTrue(mp.review(pm.decompose(FRAUD)).ready)  # sound: nothing blocks, however many questions there are

    def test_a_single_map_that_is_unsound_is_still_raised(self):
        bad = pm.decompose(CHURN).model_copy(update={"size": "complex"})
        self.assertTrue(any(q.blocking for q in mp.review_rules(bad)))

    def test_the_data_card_adds_what_the_tables_lack(self):
        m = pm.decompose(FRAUD)
        self.assertEqual([q for q in mp.review_rules(m, data(labels=True)) if q.kind == "missing_data"], [])
        lacking = [q for q in mp.review_rules(m, data(labels=False)) if q.kind == "missing_data"]
        self.assertEqual([(q.code, q.target) for q in lacking], [("data_labels_absent", "sp2")])
        self.assertIn("file of confirmed cases", lacking[0].question)

    def test_model_questions_are_merged_without_repeating_the_rules_and_never_block(self):
        m = pm.decompose(FRAUD)
        again = mp.MapQuestion(code="model_question", kind="risk", target="sp2", question="Is a ride no one marked as fraud clean, or only not checked yet?", source="model")
        new = mp.MapQuestion(code="model_question", kind="ambiguity", target="part3", question="Are the linked accounts the same as the bikers?", source="model")
        review = mp.review(m, extra=[again, new], source="advisor")
        self.assertEqual(sum(1 for q in review.questions if q.kind == "risk" and q.target == "sp2"), 1)
        self.assertIn("Are the linked accounts the same as the bikers?", [q.question for q in review.questions])
        self.assertTrue(review.ready and review.source == "advisor" and review.map == m.fingerprint)

    def test_the_review_is_a_versioned_contract_that_reads_back(self):
        review = mp.review(pm.decompose(FRAUD), data())
        self.assertEqual(mp.MapReview.read(review.model_dump_json()), review)
        from dclab_core.contracts import CONTRACTS
        self.assertIn("map_review", CONTRACTS)
        with self.assertRaises(ValueError):
            mp.MapQuestion(code="X", kind="risk", question="Is it?")  # a code is a stable lower-case name


@unittest.skipIf(MISSING, MISSING)
class CoverageTest(unittest.TestCase):
    def test_the_fraud_needs_are_met_by_the_replicas_tables(self):
        self.assertEqual(mp.coverage(pm.decompose(FRAUD), data()), [])

    def test_known_cases_without_a_labels_table_and_a_table_nothing_is_like_are_found(self):
        m = pm.decompose(FRAUD)
        self.assertEqual([(g["code"], g["sub_problem"]) for g in mp.coverage(m, data(labels=False))], [("labels_absent", "sp2")])
        two_mains = dc.card_of({"rides": delivery_fraud.generate(seed=0).rides, "payments": delivery_fraud.generate(seed=1).rides}, roles={"payments": "main"})
        gone = pm.decompose(FRAUD)
        gone = fresh(sub(gone, "sp1", needs=[pm.Need(table="complaints")]))
        self.assertEqual([(g["code"], g["table"]) for g in mp.coverage(gone, two_mains) if g["code"] == "table_absent"], [("table_absent", "complaints")])
        self.assertIn(("labels_absent", "sp2"), [(g["code"], g["sub_problem"]) for g in mp.coverage(gone, two_mains)])  # no labels table, and none is assumed

    def test_a_column_a_need_names_that_the_table_lacks_is_listed(self):
        m = fresh(sub(pm.decompose(FRAUD), "sp1", needs=[pm.Need(table="rides", columns=["amount", "created_date", "receiver_phone"])]))
        gap = [g for g in mp.coverage(m, data()) if g["code"] == "columns_absent"]
        self.assertEqual([(g["table"], g["columns"]) for g in gap], [("rides", ["amount", "receiver_phone"])])

    def test_fill_needs_points_each_need_at_a_real_table_and_chooses_only_real_columns(self):
        filled = mp.fill_needs(pm.decompose(FRAUD), data())
        have = {t.name: {c.name for c in t.columns} for t in data().tables}
        tables = {n.table for s in filled.sub_problems for n in s.needs if n.table}
        self.assertTrue(tables <= set(have), tables)
        for s in filled.sub_problems:
            for n in s.needs:
                if n.table:
                    self.assertTrue(set(n.columns) <= have[n.table], (s.id, n.columns))
        self.assertTrue(filled.sub_problems[0].needs[0].columns)  # the screen got columns by purpose
        self.assertEqual(pm.validate(filled), [])
        self.assertNotEqual(filled.fingerprint, pm.decompose(FRAUD).fingerprint)  # needs are in the fingerprint

    def test_the_labels_table_is_found_for_known_cases_and_gets_no_invented_columns(self):
        filled = mp.fill_needs(pm.decompose(FRAUD), data())
        known = next(n for n in filled.sub_problems[1].needs if n.label == "file")
        self.assertEqual((known.table, known.columns), ("known_frauds", []))

    def test_a_map_with_nothing_to_fill_is_returned_as_it_is(self):
        m = pm.decompose(CHURN)
        self.assertIs(mp.fill_needs(m, data()), m)


@unittest.skipIf(MISSING, MISSING)
class EditTest(unittest.TestCase):
    def setUp(self):
        self.fraud = pm.decompose(FRAUD)

    def test_removing_a_part_drops_its_sub_problem_the_reads_of_it_and_its_place_in_the_steps(self):
        m = mp.remove_part(self.fraud, "part2")
        self.assertEqual([s.id for s in m.sub_problems], ["sp1", "sp3", "sp4"])  # ids stay: a decision can still point at one
        self.assertEqual(m.composition.steps[0].inputs, ["sp1", "sp3"])
        self.assertEqual((m.source, m.template, pm.validate(m)), ("person", "edited", []))
        self.assertNotEqual(m.fingerprint, self.fraud.fingerprint)
        self.assertEqual(len(self.fraud.parts), 4)  # the original is untouched

    def test_removing_the_screen_the_classifier_read_drops_that_read(self):
        m = mp.remove_part(self.fraud, "part1")
        self.assertEqual(next(s for s in m.sub_problems if s.id == "sp2").reads, [])
        self.assertEqual(pm.validate(m), [])

    def test_removing_until_one_part_is_left_makes_a_single_problem_with_nothing_to_compose(self):
        m = self.fraud
        for part in ("part1", "part2", "part3"):
            m = mp.remove_part(m, part)
        self.assertEqual((m.size, m.composition.method, m.composition.steps, len(m.sub_problems)), ("single", "none", [], 1))
        self.assertEqual(pm.validate(m), [])
        with self.assertRaises(mp.MapEditRefused):
            mp.remove_part(m, "part4")  # a map keeps one part

    def test_an_edit_the_validator_refuses_says_why_and_changes_nothing(self):
        routing = pm.decompose("Route each new support ticket to the team that should handle it, then set how urgent it is.")
        with self.assertRaises(mp.MapEditRefused) as caught:
            mp.merge_parts(self.fraud, "part1", "part3")  # a ride and an account in one part
        self.assertIn("one part, one unit", str(caught.exception))
        self.assertTrue(caught.exception.issues)
        with self.assertRaises(mp.MapEditRefused) as caught:
            mp.remove_part(self.fraud, "part9")
        self.assertIn("not a part", str(caught.exception))
        self.assertEqual(pm.validate(routing), [])

    def test_two_parts_of_one_unit_and_moment_merge_into_one(self):
        m = mp.merge_parts(self.fraud, "part1", "part2")
        self.assertEqual([p.id for p in m.parts], ["part1", "part3", "part4"])
        self.assertEqual({s.part for s in m.sub_problems if s.id in ("sp1", "sp2")}, {"part1"})
        self.assertIn("and", m.parts[0].title)
        self.assertEqual(pm.validate(m), [])

    def test_the_same_part_or_one_that_is_not_there_cannot_be_merged(self):
        for a, b in (("part1", "part1"), ("part1", "part9")):
            with self.assertRaises(mp.MapEditRefused):
                mp.merge_parts(self.fraud, a, b)

    def test_adding_a_part_joins_it_to_the_first_combining_step_and_gets_new_ids(self):
        m = mp.add_part(self.fraud, "Check the couriers' hours", "anomaly", question="Do some couriers work impossible hours?", unit="ride",
                        moment=self.fraud.parts[0].moment, labels="none")
        self.assertEqual((m.parts[-1].id, m.sub_problems[-1].id), ("part5", "sp5"))
        self.assertIn("sp5", m.composition.steps[0].inputs)
        self.assertEqual((m.source, pm.validate(m)), ("person", []))

    def test_adding_a_part_to_a_single_problem_makes_a_sequence_of_the_two(self):
        m = mp.add_part(pm.decompose(CHURN), "Why they cancel", "classification")
        self.assertEqual((m.size, m.composition.method, [c.kind for c in m.composition.steps], m.composition.steps[0].inputs), ("complex", "sequence", ["sequence"], ["sp1", "sp2"]))
        self.assertEqual(pm.validate(m), [])

    def test_removing_the_scorers_one_by_one_reaches_a_map_that_holds_and_carries_the_key(self):
        m = mp.remove_part(self.fraud, "part1")
        m = mp.remove_part(m, "part2")  # the fusion left with the rings only combines nothing: it goes, and its key stays with the budget
        self.assertEqual([(c.kind, c.inputs, c.joins) for c in m.composition.steps], [("budget", ["sp3", "sp4"], "the account of the ride")])
        self.assertEqual(pm.validate(m), [])

    def test_a_part_on_another_unit_needs_the_key_that_joins_it(self):
        with self.assertRaises(mp.MapEditRefused) as caught:
            mp.add_part(self.fraud, "Customers", "segmentation", unit="customer")
        self.assertIn("say the key that joins them", str(caught.exception))
        m = mp.add_part(self.fraud, "Customers", "segmentation", unit="customer", joins="the account of the ride")
        self.assertEqual(m.composition.steps[0].joins, "the account of the ride; the account of the ride")

    def test_a_value_over_its_limit_or_not_text_is_a_refused_edit_never_a_crash(self):
        for kwargs in ({"unit": "x" * 300}, {"unit": 5}, {"quote": ["a"]}, {"joins": "k" * 700}, {"moment": "m" * 700}, {"question": {"a": 1}}):
            with self.assertRaises(mp.MapEditRefused, msg=str(kwargs)[:40]):
                mp.add_part(self.fraud, "A part", "anomaly", **kwargs)
        m = self.fraud
        for i in range(16):  # twenty parts at most
            m = mp.add_part(m, f"Part number {i}", "anomaly", unit="ride", labels="none") if len(m.parts) < 20 else m
        self.assertEqual(len(m.parts), 20)
        with self.assertRaises(mp.MapEditRefused) as caught:
            mp.add_part(m, "One too many", "anomaly", unit="ride")
        self.assertIn("at most 20", str(caught.exception))

    def test_a_removed_id_is_not_used_again_when_the_caller_says_which_were_removed(self):
        m = mp.remove_part(self.fraud, "part4")
        again = mp.add_part(m, "A new part", "anomaly", unit="ride", moment=None, labels="none", avoid=["part4", "sp4"])
        self.assertEqual((again.parts[-1].id, again.sub_problems[-1].id), ("part5", "sp5"))
        reused = mp.add_part(m, "A new part", "anomaly", unit="ride", labels="none")
        self.assertEqual(reused.parts[-1].id, "part4")  # without the list the next number is the highest plus one

    def test_a_part_can_join_the_step_the_person_names(self):
        m = mp.add_part(self.fraud, "Another score", "anomaly", unit="ride", labels="none", into="risk")
        self.assertIn("sp5", m.composition.steps[0].inputs)  # `risk` is the first step's output
        with self.assertRaises(mp.MapEditRefused):
            mp.add_part(self.fraud, "Another score", "anomaly", unit="ride", into="nowhere")

    def test_two_parts_that_differ_in_unit_or_moment_cannot_merge_even_when_one_says_nothing(self):
        with_moment = fresh(self.fraud.model_copy(update={"parts": [p.model_copy(update={"moment": "At the end of the day"}) if p.id == "part1" else p for p in self.fraud.parts]}))
        with self.assertRaises(mp.MapEditRefused) as caught:
            mp.merge_parts(with_moment, "part1", "part2")  # part2 says no moment: it does not adopt part1's, nor part1 its absence
        self.assertIn("moment", str(caught.exception))
        self.assertEqual([p.moment for p in mp.merge_parts(self.fraud, "part1", "part2").parts][:1], [None])

    def test_a_part_needs_a_title_and_a_family_and_labels_of_the_vocabulary(self):
        for kwargs in ({"title": "  ", "family": "anomaly"}, {"title": "x", "family": "astrology"}, {"title": "x", "family": "anomaly", "labels": "maybe"}):
            with self.assertRaises(mp.MapEditRefused):
                mp.add_part(self.fraud, **kwargs)


@unittest.skipIf(MISSING, MISSING)
class DecomposerTest(unittest.TestCase):
    def reply(self, **changes):
        base = {"parts": [{"title": "Find unusual rides", "question": "Which rides look unlike the rest?", "unit": "rides", "quote": "Find the fake rides and fake orders"},
                          {"title": "Choose what to review", "question": "Which cases first?", "quote": "the team can review 200 a day"}],
                "sub_problems": [{"part": 1, "family": "anomaly", "labels": "none", "predict_rank": 1, "rule": "fraud"},
                                 {"part": 2, "family": "ranking", "labels": "unknown", "predict_rank": 2, "known_rank": 3, "reads": [{"sub_problem": 1, "what": "score"}],
                                  "quote": "the team can review 200 a day"}],
                "composition": {"method": "sequence", "budget": "the team can review 200 a day", "steps": [{"kind": "sequence", "inputs": ["sp1", "sp2"], "output": "list"}]}}
        base.update(changes)
        return base

    def draw(self, reply, words=FRAUD):
        return mp.draw(words, advisor=ScriptedAdvisor({"decomposer": reply}))

    def test_a_valid_reply_is_the_map_and_it_comes_from_the_advisor(self):
        made, source, _ = self.draw(self.reply())
        self.assertEqual((source, made.source, made.template, made.size, len(made.parts)), ("advisor", "model", "model", "complex", 2))
        self.assertEqual(pm.validate(made), [])
        self.assertIn("200", made.composition.budget)

    def test_every_part_rests_on_the_persons_words_or_a_rule_template(self):
        reply = self.reply()
        reply["parts"][1]["quote"] = "a made up claim nobody wrote"
        reply["sub_problems"][1]["quote"] = "also invented words"
        made, source, _ = self.draw(reply)
        self.assertEqual((source, len(made.parts)), ("advisor", 1))  # the second part is dropped with its sub-problem and what read it
        self.assertEqual(made.size, "single")

    def test_nothing_grounded_is_refused_whole_and_the_rules_stand(self):
        reply = {"parts": [{"title": "Invented", "quote": "words that are not there"}], "sub_problems": [{"part": 1, "family": "anomaly", "quote": "words that are not there"}]}
        made, source, reason = self.draw(reply)
        self.assertEqual((source, made.fingerprint), ("fallback", pm.decompose(FRAUD).fingerprint))
        self.assertEqual(reason, "invalid")

    def test_a_step_that_names_a_dropped_sub_problem_never_reads_it_as_another(self):
        q = "Find the fake rides and fake orders"
        reply = {"parts": [{"title": f"P{i}", "quote": q if i != 2 else "words nobody wrote"} for i in range(1, 5)],
                 "sub_problems": [{"part": 1, "family": "anomaly", "quote": q}, {"part": 2, "family": "rare_event", "quote": "words nobody wrote"},
                                  {"part": 3, "family": "entity_tables", "quote": q}, {"part": 4, "family": "ranking", "quote": q}],
                 "composition": {"method": "mixed", "steps": [{"kind": "fusion", "inputs": ["sp1", "sp2", "sp4"], "output": "risk"},
                                                              {"kind": "budget", "inputs": ["risk", "sp3"], "output": "list"}]}}
        made, source, _ = self.draw(reply)
        self.assertEqual(source, "advisor")
        self.assertEqual([s.family for s in made.sub_problems], ["anomaly", "entity_tables", "ranking"])  # the second was dropped; the others renumbered
        # the model's sp1, sp3 and sp4 are our sp1, sp2 and sp3; its sp2 is gone and is not read as anyone's
        self.assertEqual([(c.kind, c.inputs) for c in made.composition.steps], [("fusion", ["sp1", "sp3"]), ("budget", ["risk", "sp2"])])

    def test_a_rule_template_is_named_only_when_the_rules_found_it_in_the_words(self):
        reply = {"parts": [{"title": "A fraud screen", "rule": "fraud"}, {"title": "Cancelling", "quote": "which subscribers will cancel"}],
                 "sub_problems": [{"part": 1, "family": "anomaly", "rule": "fraud"}, {"part": 2, "family": "classification", "quote": "which subscribers will cancel"}],
                 "composition": {"method": "sequence", "steps": [{"kind": "sequence", "inputs": ["sp1", "sp2"], "output": "x"}]}}
        made, source, _ = mp.draw(CHURN, advisor=ScriptedAdvisor({"decomposer": reply}))
        self.assertEqual((source, made.size, [s.family for s in made.sub_problems]), ("advisor", "single", ["classification"]))  # no fraud part from churn words
        made, source, _ = self.draw(self.reply())
        self.assertEqual(source, "advisor")  # the fraud words do have the template: naming it is allowed

    def test_what_decides_a_leak_is_never_the_models_to_write(self):
        q = "Find the fake rides and fake orders"
        reply = {"parts": [{"title": "Groups", "quote": q}, {"title": "Review", "quote": q}],
                 "sub_problems": [{"part": 1, "family": "segmentation", "labels": "column", "predict_rank": 1, "known_rank": 1, "quote": q},
                                  {"part": 2, "family": "ranking", "labels": "column", "predict_rank": 1, "known_rank": 1, "reads": [{"sub_problem": 1, "what": "outcome"}], "quote": q}],
                 "composition": {"method": "sequence", "steps": [{"kind": "sequence", "inputs": ["sp1", "sp2"], "output": "x"}]}}
        made, source, reason = self.draw(reply)
        self.assertEqual((source, reason), ("fallback", "invalid"))  # the model said the groups' outcome is known at rank 1: that is not its to say, and it is never known in time
        reply["sub_problems"][1]["reads"] = [{"sub_problem": 1, "what": "score"}]
        made, source, _ = self.draw(reply)
        self.assertEqual(source, "advisor")
        by = {s.family: s for s in made.sub_problems}
        self.assertEqual((by["segmentation"].labels, by["segmentation"].known_rank, by["segmentation"].predict_rank), ("unknown", None, 0))  # a family the rules did not draw: nothing the model wrote; it reads nothing, so it predicts first
        self.assertEqual((by["ranking"].labels, by["ranking"].known_rank, by["ranking"].predict_rank), ("unknown", 3, 1))  # the rules' own reading of a ranking (its outcome known late); it predicts after what it reads

    def test_a_problem_and_the_offer_that_follows_it_are_drawn_in_the_order_they_read(self):
        # the live dev run's CD-04: the second part reads the first's score; the rules know only the second's family, at moment 0
        reply = {"parts": [{"title": "Cancellation", "quote": "which subscribers will cancel next month"}, {"title": "Offer", "quote": "which retention offer would change their mind"}],
                 "sub_problems": [{"part": 1, "family": "classification", "quote": "which subscribers will cancel next month"},
                                  {"part": 2, "family": "uplift", "reads": [{"sub_problem": 1, "what": "score"}], "quote": "which retention offer would change their mind"}],
                 "composition": {"method": "sequence", "steps": [{"kind": "sequence", "inputs": ["sp1", "sp2"], "output": "offer"}]}}
        words = "Predict which subscribers will cancel next month, and for those, which retention offer would change their mind."
        made, source, _ = mp.draw(words, advisor=ScriptedAdvisor({"decomposer": reply}))
        self.assertEqual((source, [s.family for s in made.sub_problems], [s.predict_rank for s in made.sub_problems]), ("advisor", ["classification", "uplift"], [0, 1]))
        reply["sub_problems"][0]["reads"] = [{"sub_problem": 2, "what": "score"}]  # each reading the other: a cycle
        _, source, reason = mp.draw(words, advisor=ScriptedAdvisor({"decomposer": reply}))
        self.assertEqual((source, reason), ("fallback", "invalid"))

    def test_a_unit_and_a_moment_are_kept_only_as_the_persons_words(self):
        reply = self.reply()
        reply["parts"][0].update(unit="a taxi trip", moment="the next morning at nine")  # neither is in the words
        made, _, _ = self.draw(reply)
        self.assertEqual((made.parts[0].unit, made.parts[0].moment), (None, None))
        words = FRAUD + " They are needed the next morning."
        reply = self.reply()
        reply["parts"][0].update(unit="fake rides", moment="the next morning")
        made, _, _ = mp.draw(words, advisor=ScriptedAdvisor({"decomposer": reply}))
        self.assertEqual((made.parts[0].unit, made.parts[0].moment), ("fake rides", "the next morning"))

    def test_a_family_outside_the_vocabulary_drops_the_sub_problem(self):
        reply = self.reply()
        reply["sub_problems"][1]["family"] = "astrology"
        made, _, _ = self.draw(reply)
        self.assertEqual([s.family for s in made.sub_problems], ["anomaly"])

    def test_a_leak_in_the_models_map_refuses_it(self):
        reply = self.reply()
        reply["sub_problems"][0]["reads"] = [{"sub_problem": 2, "what": "outcome"}]
        made, source, reason = self.draw(reply)
        self.assertEqual((source, reason), ("fallback", "invalid"))
        self.assertEqual(made.template, "fraud")

    def test_a_budget_the_person_did_not_state_is_dropped(self):
        reply = self.reply()
        reply["composition"]["budget"] = "ten reviewers a day"
        made, _, _ = self.draw(reply)
        self.assertIsNone(made.composition.budget)

    def test_a_reply_of_the_wrong_shape_or_an_abstention_leaves_the_rules(self):
        for reply in ({"parts": "x", "sub_problems": []}, {}, None):
            made, source, _ = self.draw(reply)
            self.assertEqual((source, made.template), ("fallback", "fraud"))
        made, source, reason = mp.draw(FRAUD)
        self.assertEqual((source, reason, made.template), ("fallback", "no advisor", "fraud"))

    def test_at_most_eight_parts_are_read(self):
        reply = self.reply(parts=[{"title": f"P{i}", "quote": "Find the fake rides and fake orders"} for i in range(12)],
                           sub_problems=[{"part": i, "family": "anomaly", "quote": "Find the fake rides and fake orders", "predict_rank": 1} for i in range(1, 13)],
                           composition={"method": "sequence", "steps": []})
        made, source, _ = self.draw(reply)
        self.assertEqual(source, "fallback")  # eight sub-problems and no steps do not hold together: the validator refuses
        reply["composition"]["steps"] = [{"kind": "sequence", "inputs": [f"sp{i}" for i in range(1, 9)], "output": "all"}]
        made, source, _ = self.draw(reply)
        self.assertEqual((source, len(made.parts)), ("advisor", 8))

    def test_the_evidence_is_names_and_never_rows_and_passes_the_advisors_check(self):
        advisor = ScriptedAdvisor({"decomposer": None})
        mp.draw(FRAUD, advisor=advisor, data=data(), card={"family": "rare_event", "labels": "file"})
        (purpose, evidence), = advisor.requests
        self.assertEqual(purpose, "decomposer")
        self.assertIn("rides", evidence["tables"])
        self.assertTrue(all(":" in c for c in evidence["tables"]["rides"]))  # column:purpose
        self.assertEqual(evidence["card"], {"family": "rare_event", "labels": "file"})
        self.assertIn("fraud", evidence["templates"])
        self.assertNotIn("rides.csv", json.dumps(evidence))

    def test_a_single_problem_stays_one_when_the_model_agrees(self):
        reply = {"parts": [{"title": "Churn", "quote": "which subscribers will cancel"}], "sub_problems": [{"part": 1, "family": "classification", "labels": "column", "quote": "which subscribers will cancel"}]}
        made, source, _ = mp.draw(CHURN, advisor=ScriptedAdvisor({"decomposer": reply}))
        self.assertEqual((source, made.size, made.composition.steps), ("advisor", "single", []))


@unittest.skipIf(MISSING, MISSING)
class ReviewerTest(unittest.TestCase):
    def setUp(self):
        self.m = pm.decompose(FRAUD)

    def ask(self, reply, **kw):
        return mp.review_with(self.m, advisor=ScriptedAdvisor({"map_reviewer": reply}), words=FRAUD, **kw)

    def item(self, **changes):
        base = {"kind": "ambiguity", "target": "part3", "question": "Are linked accounts the same as the bikers?", "quote": "our bikers and customers"}
        base.update(changes)
        return base

    def test_a_question_that_rests_on_the_words_or_the_map_is_kept_and_never_blocks(self):
        review, source, _ = self.ask({"questions": [self.item(), self.item(question="Does part two read part one?", quote=None, ref="sp2", target="sp2", kind="leak")]})
        model = [q for q in review.questions if q.source == "model"]
        self.assertEqual((source, len(model)), ("advisor", 2))
        self.assertFalse(any(q.blocking for q in model))
        self.assertIn("our bikers and customers", model[0].why)

    def test_a_question_that_rests_on_nothing_or_is_not_a_question_or_points_nowhere_is_dropped(self):
        bad = [self.item(quote="words nobody wrote"), self.item(question="Check the bikers."), self.item(target="part9"), self.item(kind="gossip"), self.item(question="x?")]
        review, _, _ = self.ask({"questions": bad})
        self.assertEqual([q for q in review.questions if q.source == "model"], [])

    def test_a_table_or_a_column_of_the_data_card_is_something_to_rest_on(self):
        item = self.item(quote=None, ref="biker_phone", question="Does one phone belong to several bikers?", target="part3", kind="risk")
        review, _, _ = self.ask({"questions": [item]}, data=data())
        self.assertEqual(len([q for q in review.questions if q.source == "model"]), 1)
        review, _, _ = self.ask({"questions": [item]})  # without the data card the column is not known
        self.assertEqual(len([q for q in review.questions if q.source == "model"]), 0)

    def test_at_most_six_model_questions(self):
        many = [self.item(question=f"Is question number {i} about the bikers?") for i in range(15)]
        review, _, _ = self.ask({"questions": many})
        self.assertEqual(len([q for q in review.questions if q.source == "model"]), 6)

    def test_a_failing_advisor_leaves_the_rules_questions(self):
        class Fails:
            def ask(self, purpose, evidence):
                raise RuntimeError("provider down")

        review, source, reason = mp.review_with(self.m, advisor=Fails(), words=FRAUD)
        self.assertEqual((source, reason, review.source), ("fallback", "failed", "rules"))
        self.assertEqual(codes(review.questions), codes(mp.review_rules(self.m)))

    def test_the_evidence_lists_what_the_rules_already_ask_so_the_model_does_not_repeat_it(self):
        advisor = ScriptedAdvisor({"map_reviewer": {"questions": []}})
        mp.review_with(self.m, advisor=advisor, words=FRAUD, data=data())
        evidence = advisor.requests[0][1]
        self.assertIn("unmarked_is_unknown:sp2", evidence["rules_ask"])
        self.assertTrue(all(isinstance(line, str) for line in evidence["map"]))


@unittest.skipIf(MISSING, MISSING)
class DataNeedTest(unittest.TestCase):
    def setUp(self):
        self.m = pm.decompose(FRAUD)

    def reply(self, *needs):
        return {"needs": list(needs)}

    def test_a_need_on_a_table_and_columns_the_data_holds_is_added_to_the_rules(self):
        advisor = ScriptedAdvisor({"data_need": self.reply({"sub_problem": "sp1", "table": "rides", "columns": ["biker_id", "created_date", "no_such_column"], "label": None, "why": "who and when"})})
        filled, source, _ = mp.needs_for(self.m, data(), advisor)
        self.assertEqual(source, "advisor")
        needs = filled.sub_problems[0].needs
        columns = {c for n in needs for c in n.columns}
        self.assertIn("biker_id", columns)
        self.assertNotIn("no_such_column", columns)  # a column the table lacks is dropped, the need stays
        self.assertEqual(pm.validate(filled), [])

    def test_a_table_the_data_card_does_not_hold_drops_the_need_and_everything_unsupported_leaves_the_rules(self):
        advisor = ScriptedAdvisor({"data_need": self.reply({"sub_problem": "sp1", "table": "complaints", "columns": ["a"]})})
        filled, source, reason = mp.needs_for(self.m, data(), advisor)
        self.assertEqual((source, reason), ("fallback", "invalid"))  # 'complaints' is like no table here, and the one main table is not assumed for a model
        two = dc.card_of({"rides": delivery_fraud.generate(seed=0).rides, "payments": delivery_fraud.generate(seed=1).rides}, roles={"payments": "main"})
        again, source2, reason2 = mp.needs_for(self.m, two, ScriptedAdvisor({"data_need": self.reply({"sub_problem": "sp1", "table": "complaints", "columns": ["a"]})}))
        self.assertEqual((source2, reason2), ("fallback", "invalid"))
        self.assertEqual(again.fingerprint, mp.fill_needs(self.m, two).fingerprint)

    def test_a_table_nothing_is_like_is_never_assumed_to_be_the_only_table(self):
        advisor = ScriptedAdvisor({"data_need": self.reply({"sub_problem": "sp1", "table": "phone_calls", "columns": ["caller", "callee", "biker_id"]})})
        filled, source, reason = mp.needs_for(self.m, data(labels=False), advisor)  # one main table: 'rides'
        self.assertEqual((source, reason), ("fallback", "invalid"))  # the only need named a table that is not there
        self.assertIsNone(mp.find_table("phone_calls", data(labels=False)))
        self.assertEqual(mp.find_table("phone_calls", data(labels=False), assume_main=True).name, "rides")  # only the rules' own needs may assume it

    def test_a_column_the_data_card_suspects_of_leaking_is_never_taken(self):
        from dclab_core.replicas import delivery_time

        orders = delivery_time.generate(seed=0, n=600).orders
        card = dc.card_of({"orders": orders}, name="orders", target="actual_minutes")
        leaks = {c.name for c in card.tables[0].columns if c.purpose == "leak_suspect"}
        self.assertTrue(leaks)
        advisor = ScriptedAdvisor({"data_need": self.reply({"sub_problem": "sp1", "table": "orders", "columns": ["order_time", *sorted(leaks)]})})
        filled, source, _ = mp.needs_for(self.m, card, advisor)
        taken = {c for s in filled.sub_problems for n in s.needs for c in n.columns}
        self.assertEqual(source, "advisor")
        self.assertIn("order_time", taken)
        self.assertEqual(taken & leaks, set())

    def test_a_sub_problem_that_is_not_in_the_map_is_ignored(self):
        advisor = ScriptedAdvisor({"data_need": self.reply({"sub_problem": "sp9", "table": "rides", "columns": ["biker_id"]})})
        _, source, reason = mp.needs_for(self.m, data(), advisor)
        self.assertEqual((source, reason), ("fallback", "invalid"))

    def test_with_no_advisor_the_rules_fill_the_needs(self):
        filled, source, reason = mp.needs_for(self.m, data())
        self.assertEqual((source, reason), ("fallback", "no advisor"))
        self.assertEqual(filled.fingerprint, mp.fill_needs(self.m, data()).fingerprint)

    def test_the_evidence_is_column_names_with_their_purposes_and_grains_never_values(self):
        advisor = ScriptedAdvisor({"data_need": None})
        mp.needs_for(self.m, data(), advisor)
        evidence = advisor.requests[0][1]
        self.assertEqual(set(evidence["tables"]), {"rides", "known_frauds"})
        self.assertTrue(all(isinstance(c, str) and ":" in c for cols in evidence["tables"].values() for c in cols))
        self.assertIn("rides", evidence["grains"])


@unittest.skipIf(MISSING, MISSING)
class RolesTest(unittest.TestCase):
    def test_the_three_roles_are_registered_with_their_prompts(self):
        from dclab_core import prompts
        from dclab_core.agents import roles

        for name in ("decomposer", "map_reviewer", "data_need"):
            self.assertIn(name, roles())
            self.assertIn(name, prompts.names())
        self.assertIn("copied exactly", " ".join(prompts.text("decomposer").split()))
        self.assertIn("never block", " ".join(prompts.text("map_reviewer").split()))

    def test_a_prompt_names_the_rules_that_keep_each_role_safe(self):
        from dclab_core import prompts

        text = {n: " ".join(prompts.text(n).split()) for n in ("decomposer", "map_reviewer", "data_need")}
        self.assertIn("a moment you wrote is a moment the person did not give", text["decomposer"])
        self.assertIn("You never change the map", text["map_reviewer"])
        self.assertIn("A table or a column that is not there is discarded", text["data_need"])


if __name__ == "__main__":
    unittest.main()
