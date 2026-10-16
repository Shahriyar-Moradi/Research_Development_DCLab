"""The twin contract (package 25.1): the system and an AI-first agent on one step, the differences, the verdict, and
where each approach is best by twin_rule_v1."""

from __future__ import annotations

import sys
import unittest
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from pydantic import ValidationError

    from dclab_core import twin as T
    from dclab_core.adapters import main_product_twin as mpt
    from dclab_core.adapters.main_product import recorded
    from dclab_core.contracts import CONTRACTS
    MISSING = None
except ImportError as error:  # pragma: no cover
    MISSING = f"dclab_core needs pydantic: {error}"

AT = "2026-10-16T09:00:00+00:00"


def arm(name, value, status="answered", valid=None, cost=0.0):
    return T.ArmAnswer(arm=name, answerer="rules" if name == "system" else "ai_first@v1", value=value, status=status, valid=valid, cost_eur=cost)


def made(n, system, ai, truth, step="solution.columns", valid=None):
    """n twin steps where each arm's answer is right or wrong as asked (on the identifiers field)."""
    right, wrong = {"identifiers": ["id"]}, {"identifiers": ["amount"]}
    out = []
    for i in range(n):
        s = right if system(i) else wrong
        a = right if ai(i) else wrong
        out.append(T.step(f"t{i}", "s", step, AT, {"i": i}, arm("system", s), arm("ai_first", a, valid=valid(i) if valid else None),
                          truth=T.Truth(value=dict(truth), source="planted")))
    return out


@unittest.skipIf(MISSING, MISSING)
class TwinStepTest(unittest.TestCase):
    def test_equal_answers_have_no_differences(self):
        value = {"target": "churned", "task": "binary", "identifiers": ["customer_id"], "leakage": [], "time": ["signup"], "group": []}
        record = T.step("t1", "s1", "solution.columns", AT, {"columns": 9}, arm("system", value), arm("ai_first", dict(value)))
        self.assertTrue(all(d.kind in ("same", "neither") for d in record.differences))
        self.assertEqual(record.system.input_fingerprint, record.ai_first.input_fingerprint)
        self.assertEqual(record.verdict.winner, "unjudged")
        self.assertIn("twin_step", CONTRACTS)
        self.assertEqual(T.TwinStep.read(record.model_dump(mode="json")), record)

    def test_a_set_field_says_what_each_arm_alone_gave(self):
        record = T.step("t1", "s1", "solution.columns", AT, {}, arm("system", {"leakage": ["final_fare", "rating"]}),
                        arm("ai_first", {"leakage": ["final_fare", "duration"]}))
        leak = next(d for d in record.differences if d.field == "leakage")
        self.assertEqual((leak.kind, leak.only_system, leak.only_ai), ("differs", ["rating"], ["duration"]))
        only = next(d for d in T.compare("home.ready", {"ready": True}, {}) if d.field == "ready")
        self.assertEqual(only.kind, "only_system")

    def test_text_is_compared_by_key_terms_and_mappings_key_by_key(self):
        self.assertTrue(T.same("text", "Whether the customer churns in the next 30 days", "will the customer churn in the next 30 days?"))
        self.assertTrue(T.same("text", "customer churns next month", "the customer churns next month"))
        self.assertFalse(T.same("text", "delivery time in minutes", "fraud on the account"))
        d = T.difference("routes", "mapping", {"screen": "isolation_forest", "rings": "graph"}, {"screen": "isolation_forest", "rings": "rules"})
        self.assertEqual((d.kind, d.only_system, d.only_ai), ("differs", ["rings"], ["rings"]))
        self.assertTrue(T.same("numeric", 0.31, 0.34))
        self.assertFalse(T.same("numeric", 0.31, 0.5))
        self.assertTrue(T.same("ranked", ["a", "b"], ["A", "b"]))
        self.assertFalse(T.same("ranked", ["a", "b"], ["b", "a"]))

    def test_two_inputs_are_refused(self):
        system = arm("system", {"pack": "churn"}).model_copy(update={"input_fingerprint": T.fingerprint({"words": "a"})})
        with self.assertRaises(ValueError):
            T.step("t1", "s1", "home.pack", AT, {"words": "b"}, system, arm("ai_first", {"pack": "churn"}))
        record = T.step("t1", "s1", "home.pack", AT, {"words": "b"}, arm("system", {"pack": "churn"}), arm("ai_first", {"pack": "fraud"}))
        data = record.model_dump(mode="json")
        data["ai_first"]["input_fingerprint"] = T.fingerprint({"words": "c"})
        with self.assertRaises(ValidationError):
            T.TwinStep.model_validate(data)

    def test_a_field_the_step_does_not_have_is_refused(self):
        with self.assertRaises(ValidationError):
            T.step("t1", "s1", "home.pack", AT, {}, arm("system", {"pack": "churn", "rows": 3}), arm("ai_first", {"pack": "churn"}))
        with self.assertRaises(ValueError):
            T.spec("home.nothing")
        self.assertEqual(list(T.spec("decision:column.is_identifier").fields), ["value"])

    def test_the_verdict_follows_the_truth_and_an_arm_that_failed_is_wrong(self):
        record = T.step("t1", "s1", "home.understanding", AT, {}, arm("system", {"family": "classification", "labels": "column"}),
                        arm("ai_first", {"family": "anomaly"}, status="failed"),
                        truth=T.Truth(value={"family": "classification"}, source="planted"))
        self.assertEqual((record.verdict.winner, record.verdict.system, record.verdict.ai_first),
                         ("system", {"family": True}, {"family": False}))
        picked = T.with_truth(T.step("t2", "s1", "home.pack", AT, {}, arm("system", {"pack": "churn"}), arm("ai_first", {"pack": "fraud"})),
                              T.Truth(value={"pack": "fraud"}, source="person_pick"))
        self.assertEqual(picked.verdict.winner, "ai_first")


@unittest.skipIf(MISSING, MISSING)
class ReviewFindingsTest(unittest.TestCase):
    """The 25.1 review's findings, each as a test."""

    def test_a_stored_verdict_or_difference_that_does_not_follow_is_refused_and_a_missing_one_is_computed(self):
        record = T.step("t1", "s1", "home.pack", AT, {}, arm("system", {"pack": "churn"}), arm("ai_first", {"pack": "fraud"}),
                        truth=T.Truth(value={"pack": "fraud"}, source="planted"))
        data = record.model_dump(mode="json")
        data["verdict"]["winner"] = "system"
        with self.assertRaises(ValidationError):
            T.TwinStep.model_validate(data)
        bare = {k: v for k, v in record.model_dump(mode="json").items() if k not in ("differences", "verdict")}
        rebuilt = T.TwinStep.model_validate(bare)
        self.assertEqual((rebuilt.verdict.winner, rebuilt.differences[0].kind), ("ai_first", "differs"))

    def test_an_empty_fingerprint_is_refused(self):
        data = T.step("t1", "s1", "home.pack", AT, {}, arm("system", {"pack": "churn"}), arm("ai_first", {"pack": "churn"})).model_dump(mode="json")
        data.update(input_fingerprint="")
        data["system"]["input_fingerprint"] = data["ai_first"]["input_fingerprint"] = ""
        with self.assertRaises(ValidationError):
            T.TwinStep.model_validate(data)

    def test_persian_text_is_compared_by_its_words(self):
        self.assertFalse(T.same("text", "ریزش مشتری", "تقلب"))
        self.assertTrue(T.same("text", "ریزش مشتری", "ریزش مشتری"))
        self.assertFalse(T.same("text", "?", "!"))
        record = T.step("t1", "s1", "home.understanding", AT, {}, arm("system", {"outcome": "تقلب"}), arm("ai_first", {"outcome": "ریزش مشتری"}),
                        truth=T.Truth(value={"outcome": "ریزش مشتری"}, source="planted"))
        self.assertEqual(record.verdict.winner, "ai_first")

    def test_records_compare_whatever_their_key_order_and_a_flag_is_not_a_number(self):
        self.assertTrue(T.same("set", [{"a": 1, "b": "X"}], [{"b": "x", "a": 1}]))
        self.assertTrue(T.same("ranked", [{"wf": "WF-01", "n": 1}, {"wf": "WF-02"}], [{"n": 1, "wf": "wf-01"}, {"wf": "WF-02"}]))
        self.assertFalse(T.same("equal", True, 1))
        d = T.difference("parts", "set", [{"t": "screen"}, {"t": "rings"}], [{"t": "rings"}])
        self.assertEqual((d.only_system, d.only_ai), ([{"t": "screen"}], []))

    def test_the_margin_is_counted_in_whole_steps_at_any_base_rate(self):
        for system, ai in ((2, 3), (7, 8), (1, 2)):
            with self.subTest(system=system, ai=ai):
                self.assertEqual(T.mode_of({"judged": 10, "system_right": system, "ai_right": ai, "checked_right": ai})[0], "ai")

    def test_an_ai_that_failed_where_the_system_was_right_is_not_credited_to_the_validator(self):
        records = []
        for i in range(12):
            s = {"identifiers": ["id"]} if i < 6 or i >= 9 else {"identifiers": ["x"]}
            a = arm("ai_first", {"identifiers": ["id"]}) if i < 9 else arm("ai_first", {}, status="failed", cost=0.4)
            records.append(T.step(f"t{i}", "s", "solution.columns", AT, {"i": i}, arm("system", s), a,
                                  truth=T.Truth(value={"identifiers": ["id"]}, source="planted")))
        e = T.effect("solution.columns", records)
        self.assertEqual((e.mode, e.ai_right, e.checked_right, e.system_right), ("ai", 12, 12, 9))
        self.assertEqual(e.cost_eur["ai_first"], 0.0)  # nine free answers and three failures that spent 0.4: the median
        self.assertEqual(e.ai_failed, 3)

    def test_a_disagreement_on_an_unjudged_field_is_not_a_disagreement(self):
        records = []
        for i in range(12):
            s = {"identifiers": ["id"] if i < 9 else ["x"], "time": ["a"]}
            a = {"identifiers": ["id"] if i < 9 else ["x"], "time": ["b"]}  # agree on what is judged; differ on time
            records.append(T.step(f"t{i}", "s", "solution.columns", AT, {"i": i}, arm("system", s), arm("ai_first", a),
                                  truth=T.Truth(value={"identifiers": ["id"]}, source="planted")))
        e = T.effect("solution.columns", records)
        self.assertIsNone(e.review_precision)
        self.assertEqual((e.review_recall, e.mode), (0.0, "software"))

    def test_a_row_cannot_enter_a_twin_record_and_a_long_reply_is_cut(self):
        with self.assertRaises(ValidationError):
            T.step("t1", "s1", "home.pack", AT, {}, arm("system", {"pack": "churn"}), arm("ai_first", {"pack": "churn"}),
                   input_summary={"rows": [{"email": "a@b.c", "amount": 12.5}]})
        with self.assertRaises(ValidationError):
            arm("ai_first", {"pack": "x" * 500})
        with self.assertRaises(ValidationError):
            T.step("t1", "s1", "decision:column.is_identifier", AT, {}, arm("system", {"value": True}), arm("ai_first", {"value": "Jane Roe, 12 Main St"}))
        long = T.ArmAnswer(arm="ai_first", answerer="claude", said="x" * 5000, reasons=[f"r{i}" for i in range(20)])
        self.assertEqual((len(long.said), len(long.reasons), long.dropped), (T.MAX_TEXT, 12, {"said": 5000 - T.MAX_TEXT, "reasons": 8}))

    def test_a_longer_right_text_is_judged_right(self):
        record = T.step("t1", "s1", "home.understanding", AT, {}, arm("system", {"outcome": "fraud"}),
                        arm("ai_first", {"outcome": "customer churns within 30 days"}), truth=T.Truth(value={"outcome": "churn"}, source="planted"))
        self.assertEqual(record.verdict.winner, "ai_first")


@unittest.skipIf(MISSING, MISSING)
class RuleTest(unittest.TestCase):
    truth = {"identifiers": ["id"]}

    def mode(self, records):
        return T.effect("solution.columns", records)

    def test_fewer_than_ten_judged_is_undecided(self):
        e = self.mode(made(9, lambda i: True, lambda i: False, self.truth))
        self.assertEqual(e.mode, "undecided")

    def test_software_when_the_ai_is_not_better(self):
        e = self.mode(made(12, lambda i: i < 10, lambda i: i < 9, self.truth))
        self.assertEqual((e.mode, e.system_right, e.ai_right), ("software", 10, 9))

    def test_ai_when_it_is_better_and_the_validator_adds_nothing(self):
        e = self.mode(made(12, lambda i: i < 6, lambda i: i < 11, self.truth))
        self.assertEqual(e.mode, "ai")
        self.assertEqual(e.ai_only_right, 5)

    def test_ai_checked_when_the_validator_catches_its_errors(self):
        # the AI is right on 0-9, wrong on 10-11 where the validator refuses it and the system is right
        e = self.mode(made(12, lambda i: i < 4 or i >= 10, lambda i: i < 10, self.truth, valid=lambda i: i < 10))
        self.assertEqual((e.mode, e.checked_right, e.ai_right, e.system_right), ("ai_checked", 12, 10, 6))

    def test_software_reviewed_when_the_ai_catches_the_systems_errors(self):
        # the system right on 0-8 (9 of 12); the AI right on 0-4 and 9-11: better nowhere overall but right on all three system errors
        e = self.mode(made(12, lambda i: i < 9, lambda i: i < 5 or i >= 9, self.truth))
        self.assertEqual((e.mode, e.review_recall), ("software_reviewed", 1.0))
        self.assertGreaterEqual(e.review_precision, T.REVIEW_PRECISION)

    def test_the_matrix_orders_steps_and_sums_cost(self):
        records = made(10, lambda i: True, lambda i: True, self.truth) + [
            T.step("p", "s", "home.pack", AT, {}, arm("system", {"pack": "churn"}), arm("ai_first", {"pack": "churn"}, cost=0.02))]
        matrix = T.effectiveness(records, "unit test")
        self.assertEqual([r.step for r in matrix.steps], ["home.pack", "solution.columns"])
        self.assertEqual(matrix.totals["cost_eur_ai_first"], 0.02)
        self.assertEqual(matrix.steps[1].agreement, 1.0)
        self.assertEqual(T.TwinMatrix.read(matrix.model_dump(mode="json")), matrix)


@unittest.skipIf(MISSING, MISSING)
class MainProductTwinTest(unittest.TestCase):
    def row(self, **values):
        columns = recorded()["experience_tables"]["semantic_decision_answers"]
        return {c: None for c in columns} | {
            "id": str(uuid.uuid4()), "workspace_id": str(uuid.uuid4()), "project_id": str(uuid.uuid4()), "llm_invocation_id": str(uuid.uuid4()),
            "purpose": "column.is_identifier", "decision_point_key": "column.is_identifier", "release_version": "1", "model_id": "jev-1.13.0",
            "question_key": "customer_id", "question_digest": "a" * 64, "data_class": "metadata", "evidence_partition": "metadata",
            "primitive": "noul", "answer": {"value": 0.96}, "confidence": 0.96, "in_acting_band": True, "rule_answer": {"value": False},
            "agreement": "disagree", "level": 1, "policy_outcome": "rule", "value_used": {"value": False}, "created_at": AT, "labels_version": 0} | values

    def test_a_rule_and_jev_pair_is_a_twin_step_judged_by_its_label(self):
        answer = self.row()
        label = self.row(label_source="benchmark", ground_truth={"value": True}, labeled_at=AT, labels_version=1, supersedes_label_id=answer["id"])
        steps = mpt.twin_steps([answer, label], session="mp")
        self.assertEqual(len(steps), 1)
        s = steps[0]
        self.assertEqual((s.step, s.system.value, s.ai_first.value, s.ai_first.answerer), ("decision:column.is_identifier", {"value": False},
                                                                                           {"value": True}, "jev-1.13.0@1"))
        self.assertEqual((s.truth.source, s.verdict.winner), ("label", "ai_first"))
        self.assertEqual(s.input_summary["question_digest"], "a" * 64)

    def test_the_latest_label_is_the_truth_whatever_the_row_order_and_a_broken_chain_is_left_out(self):
        answer = self.row()
        l1 = self.row(label_source="benchmark", ground_truth={"value": True}, labeled_at=AT, labels_version=1, supersedes_label_id=answer["id"])
        l2 = self.row(label_source="blind", ground_truth={"value": False}, labeled_at=AT, labels_version=2, supersedes_label_id=l1["id"],
                      labeled_by_user_id=str(uuid.uuid4()))
        for order in ([answer, l1, l2], [answer, l2, l1]):
            with self.subTest(order=[r["labels_version"] for r in order]):
                self.assertEqual(mpt.twin_steps(order, session="mp")[0].verdict.winner, "system")
        alone = mpt.twin_steps([answer, l2], session="mp")  # v2 without v1: the decision stays unjudged, the batch is read
        self.assertEqual(alone[0].verdict.winner, "unjudged")

    def test_jevs_abstain_is_an_abstained_arm(self):
        s = mpt.twin_steps([self.row(answer={"value": 0.5}, confidence=0.5, agreement="abstain", in_acting_band=False)], session="mp")[0]
        self.assertEqual((s.ai_first.status, s.ai_first.value), ("abstained", {}))

    def test_an_unavailable_jev_is_a_skipped_arm(self):
        steps = mpt.twin_steps([self.row(answer={}, confidence=None, in_acting_band=False, agreement="unavailable")], session="mp")
        self.assertEqual(steps[0].ai_first.status, "skipped")
        self.assertEqual(mpt.twin_steps([self.row(purpose="nothing.here", decision_point_key="nothing.here")], session="mp"), [])


if __name__ == "__main__":
    unittest.main()
