"""T1 of the team workspace (2026-11-23): feasibility means what Business asked.

The pure rules of ``dclab_core.target_contract``: the person's words read as an event code only when they are clear
(a negation is read, a conflict or an unknown wording stays open: never a guess), and a confirmed target agreeing with
what a table's label column is declared to record, or naming exactly what is missing. ``TeamTask`` v2 carries the
confirmed target and still reads v1; ``meet.check`` raises ``target_event`` when the two disagree.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import target_contract as tc  # noqa: E402
from dclab_core import team_work as tw  # noqa: E402

CANCEL, LAPSE = "explicit_cancellation", "non_renewal_by_date"
BOTH = (CANCEL, LAPSE)


def contract(event=LAPSE, words="Failure to renew by the contractual renewal date.", **kw):
    return tc.TargetContract(**{"event": event, "words": words, "brief_revision": 1, "confirmed_by": "sara", **kw})


def label(event=LAPSE, column="renewed_by_date", **kw):
    return tc.LabelDeclaration(**{"column": column, "event": event, "declared_by": "fixture", **kw})


class ReadEventTest(unittest.TestCase):
    def test_clear_words_are_read_as_their_event(self):
        cases = {
            "Failure to renew by the contractual renewal date.": LAPSE,
            "failure to renew by the contractual date": LAPSE,
            "Non-renewal by the contract date": LAPSE,
            "They simply do not renew when the contract ends": LAPSE,
            "The contract lapses": LAPSE,
            "Cancellation.": CANCEL,
            "Cancellation of the subscription.": CANCEL,
            "Only explicit cancellation counts": CANCEL,
            "when the customer cancels": CANCEL,
        }
        for words, event in cases.items():
            with self.subTest(words=words):
                self.assertEqual(tc.read_event(words, BOTH), event)

    def test_a_negation_is_read_not_matched_as_a_keyword(self):
        cases = {
            "explicit cancellation only, not non-renewal": CANCEL,
            "Not a failure to renew: an explicit cancellation": CANCEL,
            "Cancellation. Non-renewal does not count.": CANCEL,
            "Failure to renew, not a cancellation": LAPSE,
            "Non-renewal by the date; a cancellation is not required": LAPSE,
            "Customers who cancel do not count, only failure to renew": LAPSE,
            "We mean non-renewal rather than cancellation": LAPSE,
        }
        for words, event in cases.items():
            with self.subTest(words=words):
                self.assertEqual(tc.read_event(words, BOTH), event)

    def test_conflicting_unknown_or_unsure_words_stay_open(self):
        for words in ("Cancellation or failure to renew", "Both cancellation and non-renewal", "Either one",
                      "Not just failure to renew, cancellation", "Not only cancellation", "I am not sure, maybe cancellation",
                      "Does cancellation count?", "They stop paying", "Customers who leave", "", "   ",
                      "Cancellation, I mean not cancellation", "not cancellation", "Customers who cancel do not renew",
                      "whatever the data has"):
            with self.subTest(words=words):
                self.assertIsNone(tc.read_event(words, BOTH))

    def test_every_spelling_of_a_negation_sets_its_event_aside_never_affirms_it(self):
        """The review of T1: "we don't mean cancellation" was read as a cancellation (the wrong event, silently). A clause
        with one event and any negation in it never affirms that event; a lone "no", "too" or "also", or a cancelled
        renewal, leaves the question open."""
        for words in ("we don't mean cancellation", "don't count lapses", "we dont mean cancellations", "cancellations can't count",
                      "cancellation not required", "cancellation isnt needed", "cancellation is irrelevant", "forget cancellations",
                      "leave out cancellations", "apart from cancellations", "cancellation: no", "cancellations, no", "cancellation never",
                      "cancelled their auto-renewal", "renewal cancelled", "non-renewal is fine too", "cancellation also"):
            with self.subTest(words=words):
                self.assertIsNone(tc.read_event(words, BOTH))
        for words, event in {"Cancellation does not count, non-renewal does.": LAPSE, "cancellation rather than non-renewal": CANCEL,
                             "let the contract expire without renewing": LAPSE,
                             "Customers who cancel before the renewal date": CANCEL}.items():
            with self.subTest(words=words):
                self.assertEqual(tc.read_event(words, BOTH), event)

    def test_only_an_offered_event_is_returned(self):
        self.assertIsNone(tc.read_event("Cancellation.", (LAPSE,)))
        self.assertEqual(tc.read_event("Cancellation.", (CANCEL,)), CANCEL)
        self.assertIsNone(tc.read_event("Cancellation.", ()))
        self.assertIsNone(tc.read_event("Cancellation.", ("late_payment",)))  # a code the module has no words for

    def test_every_event_has_its_words_and_reads_its_own_label(self):
        for code, event in tc.EVENTS.items():
            self.assertRegex(code, tc.CODE)
            self.assertEqual(tc.read_event(event.label, tuple(tc.EVENTS)), code, event.label)
            self.assertTrue(tc.event_label(code))
        self.assertEqual(tc.event_label("late_payment"), "late payment")


class AgreementTest(unittest.TestCase):
    def test_a_target_agrees_with_a_mature_label_of_the_same_event(self):
        verdict = tc.agreement(contract(), [label()])
        self.assertEqual((verdict.ok, verdict.kind, verdict.column, verdict.missing), (True, "agrees", "renewed_by_date", None))

    def test_another_event_names_the_record_that_is_missing(self):
        verdict = tc.agreement(contract(CANCEL, "Only explicit cancellation counts"), [label()])
        self.assertEqual((verdict.ok, verdict.kind, verdict.missing, verdict.recorded), (False, "event_mismatch", CANCEL, [LAPSE]))
        self.assertIn("explicit cancellation", verdict.reason)
        self.assertIn("renewed_by_date", verdict.reason)
        self.assertNotIn("Only explicit", verdict.reason)  # codes and column names: never the person's words

    def test_it_passes_once_the_record_exists(self):
        both = [label(), label(CANCEL, "cancelled_at")]
        verdict = tc.agreement(contract(CANCEL, "Only explicit cancellation counts"), both)
        self.assertEqual((verdict.ok, verdict.column), (True, "cancelled_at"))
        self.assertEqual(tc.agreement(contract(), both).column, "renewed_by_date")

    def test_no_target_no_label_and_immature_outcomes_each_say_so(self):
        self.assertEqual(tc.agreement(None, [label()]).kind, "no_target")
        none = tc.agreement(contract(), [])
        self.assertEqual((none.kind, none.missing), ("no_label", LAPSE))
        young = tc.agreement(contract(), [label(mature=False)])
        self.assertEqual((young.ok, young.kind, young.missing, young.column), (False, "immature", LAPSE, "renewed_by_date"))
        self.assertTrue(tc.agreement(contract(), [label(mature=False), label(column="renewed_flag")]).ok)  # a mature one wins
        self.assertEqual(tc.agreement(contract(), label()).kind, "agrees")  # one declaration, not in a list

    def test_the_contracts_are_frozen_versioned_and_round_trip(self):
        for doc in (contract(entity="account", prediction_time="seven days before renewal", horizon="by the renewal date",
                             label_source="sample:renewal_v1", maturity_rule="known once the renewal date has passed", response="r-1"),
                    label(source="sample:renewal_v1")):
            again = type(doc).read(doc.model_dump_json())
            self.assertEqual(again, doc)
            with self.assertRaises(Exception):
                type(doc).model_validate({**json.loads(doc.model_dump_json()), "extra": 1})
        self.assertEqual(contract().schema_version, "dclab_core.target_contract.v1")
        self.assertEqual(label().schema_version, "dclab_core.label_declaration.v1")
        with self.assertRaises(Exception):
            contract(event="Not A Code")
        with self.assertRaises(Exception):
            contract(words=" ")

    def test_a_changed_answer_is_seen_and_the_same_answer_is_not(self):
        first = contract()
        self.assertFalse(tc.changed(first, contract(words="Non-renewal by the date")))  # the same event in other words
        self.assertTrue(tc.changed(first, contract(CANCEL, "Cancellation")))
        self.assertTrue(tc.changed(None, first))
        self.assertFalse(tc.changed(None, None))


class TeamTaskTest(unittest.TestCase):
    BASE = dict(id="T-1", workspace_id="W1", project_id="P-1", problem_id="PB-1", brief_revision=1,
                requester={"team": "business", "member": "sara", "thread_id": "th-1"}, owner_team="data_science",
                question="Can the data rank accounts by churn risk?", expected_output="a feasibility answer",
                done_criteria=["says whether the data can answer"], created_at="2026-11-23T10:00:00+00:00")

    def test_a_task_carries_its_confirmed_target_and_is_v2(self):
        task = tw.TeamTask(**self.BASE, target=contract())
        self.assertEqual((task.schema_version, task.target.event), ("dclab_core.team_task.v2", LAPSE))
        self.assertEqual(tw.TeamTask.read(task.model_dump_json()), task)
        self.assertIsNone(tw.TeamTask(**self.BASE).target)

    def test_a_v1_task_still_reads_alone_and_inside_a_bundle(self):
        old = {**json.loads(tw.TeamTask(**self.BASE).model_dump_json()), "schema_version": "dclab_core.team_task.v1"}
        old.pop("target")
        self.assertEqual(tw.TeamTask.read(old).schema_version, "dclab_core.team_task.v2")
        self.assertIsNone(tw.TeamTask.model_validate(old).target)
        for version in ("v1", "v2"):
            bundle = tw.TeamWork.read({"schema_version": f"dclab_core.team_work.{version}", "problem_id": "PB-1", "tasks": [old]})
            self.assertEqual((bundle.schema_version, bundle.tasks[0].schema_version), ("dclab_core.team_work.v3", "dclab_core.team_task.v2"))
        self.assertEqual(tw.CONTRACT, "dclab_core.team_work.v3")

    def test_a_targets_brief_revision_is_never_ahead_of_its_task(self):
        with self.assertRaises(Exception):
            tw.TeamTask(**self.BASE, target=contract(brief_revision=2))
        self.assertEqual(tw.TeamTask(**{**self.BASE, "brief_revision": 2}, target=contract(brief_revision=2)).target.brief_revision, 2)


class MeetTest(unittest.TestCase):
    def cards(self):
        import pandas as pd

        from dclab_core.datacard import card_of

        table = pd.DataFrame({"account_id": [f"a{i}" for i in range(60)], "seats": [i % 9 for i in range(60)],
                              "renewed_by_date": [i % 2 for i in range(60)]})
        return card_of({"accounts": table}, target="renewed_by_date")

    def problem(self, definition):
        sys.path.insert(0, str(ROOT / "tests"))
        from test_problem_meets_data import problem

        return problem("Rank accounts by churn risk", "classification", target_column="renewed_by_date", target_definition=definition)

    def test_check_raises_target_event_when_the_column_records_another_event(self):
        from dclab_core import meet

        data = self.cards()
        labels = [label()]
        found = meet.check(self.problem("only an explicit cancellation counts"), data, labels=labels)
        [m] = [m for m in found.mismatches if m.code == "target_event"]
        self.assertTrue(m.blocking)
        self.assertFalse(found.ready)
        self.assertEqual((m.evidence["missing"], m.evidence["recorded"], m.evidence["columns"]), (CANCEL, [LAPSE], ["renewed_by_date"]))
        self.assertEqual(found.schema_version, "dclab_core.problem_data_check.v2")
        # the confirmed contract wins over the card's words; the same event agrees; unreadable words raise nothing
        agreed = meet.check(self.problem("only an explicit cancellation counts"), data, labels=labels, target=contract())
        self.assertNotIn("target_event", [m.code for m in agreed.mismatches])
        self.assertNotIn("target_event", [m.code for m in meet.check(self.problem("failure to renew by the date"), data, labels=labels).mismatches])
        self.assertNotIn("target_event", [m.code for m in meet.check(self.problem("customers who leave"), data, labels=labels).mismatches])
        self.assertNotIn("target_event", [m.code for m in meet.check(self.problem("only an explicit cancellation counts"), data).mismatches])
        old = {**json.loads(found.model_dump_json()), "schema_version": "dclab_core.problem_data_check.v1", "mismatches": []}
        self.assertEqual(meet.ProblemDataCheck.read(old).schema_version, "dclab_core.problem_data_check.v2")
        verdict = meet.fit(self.problem("only an explicit cancellation counts"), data, found)
        self.assertEqual((verdict.verdict, verdict.missing, verdict.back_to), ("does_not_fit", ["a record of explicit cancellation"], "data"))


if __name__ == "__main__":
    unittest.main()
