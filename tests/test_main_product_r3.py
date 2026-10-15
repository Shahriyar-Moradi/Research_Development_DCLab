"""Package 22.7, adoption by the main product: a DCLab decision classifier answers in the main product's own shape (its
state, its bands, its answer tuple), an R3-shaped run compares it with the rule on the same cases (by partition and by
label source, with the ai_off ablation) as an r3_runs row, the cases become semantic_decision_answers rows, and the
labels leave in its label format only through consent. The answer row no longer writes a classifier's confident no as
a yes."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import ledger as L  # noqa: E402
from dclab_core.adapters import main_product_ledger as mpl  # noqa: E402
from dclab_core.adapters import main_product_r3 as r3  # noqa: E402
from dclab_core.adapters.main_product import recorded  # noqa: E402


def record(point, answers, **over):
    return L.DecisionRecord(id="dr" + "0" * 12, point=point, subject=L.Subject(kind="column", key="customer_id"), state={},
                            options=["yes", "no"] if point != "column.semantic_role" else ["numeric", "identifier"],
                            answers=answers, value_used=answers[0].value, used="rule", level=0, workspace="w1", at="2026-10-15T00:00:00+00:00", **over)


class Says:
    by = "classifier"

    def __init__(self, point, value, confidence, name="fake"):
        self.name, self.version, self.points, self.value, self.confidence = name, "v1", frozenset({point}), value, confidence
        self.states = []

    def answer(self, point, state, options):
        self.states.append(dict(state))
        value = self.value(state) if callable(self.value) else self.value
        return None if value is None else L.Answer(by="classifier", answerer=f"{self.name}@v1", value=value, raw=self.confidence, confidence=self.confidence)


def cases(point, n=40, rule=False, label=True):
    return [{"point": point, "dataset": f"t{i % 8}", "partition": "dev", "fields": {"column": f"c{i}", "dtype": "integer", "uniqueness": "all"},
             "context": {"target": "y"}, "options": ["yes", "no"], "rule": rule, "label": label, "label_source": "benchmark", "column": f"c{i}"}
            for i in range(n)]


class AnswerRowTest(unittest.TestCase):
    def test_a_classifiers_confident_no_is_a_probability_near_zero_never_a_yes(self):
        rule = L.Answer(by="rule", answerer="rule:id_like", value=False)
        no = L.Answer(by="classifier", answerer="m@1", value=False, raw=0.97, confidence=0.97)
        row = mpl.to_semantic_answer(record("column.is_identifier", [rule, no]), "m", "1", "b" * 64)
        self.assertEqual((row["answer"], row["confidence"], row["in_acting_band"], row["agreement"]), ({"value": 0.03}, None, True, "agree"))

    def test_the_main_products_band_decides_and_a_choice_is_its_label(self):
        rule = L.Answer(by="rule", answerer="rule:id_like", value=False)
        unsure = L.Answer(by="classifier", answerer="m@1", value=True, raw=0.85, confidence=0.85)  # below its yes band, 0.90
        row = mpl.to_semantic_answer(record("column.is_identifier", [rule, unsure]), "m", "1", "b" * 64)
        self.assertEqual((row["answer"], row["in_acting_band"], row["agreement"]), ({"value": 0.85}, False, "abstain"))
        role = L.Answer(by="classifier", answerer="m@1", value="identifier", raw=0.93, confidence=0.93)
        row = mpl.to_semantic_answer(record("column.semantic_role", [L.Answer(by="rule", answerer="r", value="numeric"), role]), "m", "1", "b" * 64)
        self.assertEqual((row["answer"], row["confidence"], row["agreement"]), ({"value": "identifier"}, 0.93, "disagree"))


class R3Test(unittest.TestCase):
    def test_the_state_is_what_the_main_product_sends(self):
        state = {"column": "refund_amount", "name_tokens": ["refund", "amount"], "dtype": "float", "uniqueness": "high", "cardinality": "high", "target": "churned"}
        sent = r3.project("feature.leakage_suspect", state)
        self.assertEqual(sorted(sent), ["column", "dtype", "name_tokens", "target"])  # its release's fields, the tokens read from the name
        self.assertNotIn("uniqueness", r3.project("column.semantic_role", state))

    def test_an_answerer_answers_as_the_main_products_answerer(self):
        model = Says("column.is_identifier", lambda s: s["column"] != "c1", 0.96)
        got = r3.R3Answerer(model)({"purpose": "column.is_identifier"}, cases("column.is_identifier", 2))
        self.assertEqual(got, [({"value": 0.96}, None), ({"value": 0.04}, None)])
        self.assertNotIn("uniqueness", Says("feature.leakage_suspect", True, 0.99).states or r3.project("feature.leakage_suspect", cases("x", 1)[0]["fields"]))

    def test_a_run_compares_the_model_with_the_rule_as_an_r3_runs_row(self):
        right = r3.R3Answerer(Says("feature.leakage_suspect", True, 0.99))
        at, rid = "2026-10-15T00:00:00+00:00", "8f2b3c3e-1111-4222-8333-444455556666"
        row = r3.run(cases("feature.leakage_suspect"), right, candidate="dclab:fake@v1", model_id="fake", at=at, run_id=rid)
        self.assertTrue(set(row) <= set(recorded()["r3"]["runs_columns"]))
        point = row["report"]["points"]["feature.leakage_suspect"]
        everything = point["sources"]["all"]
        self.assertEqual((everything["rule_accuracy_all"], everything["ai_policy_accuracy_all"], everything["gap"]["point"]), (0.0, 1.0, 1.0))
        self.assertEqual(everything["gap"]["method"], "t_cluster_means")  # 8 tables: below 20 clusters, a t interval
        self.assertEqual(sorted(point["sources"]), ["all", "development"])
        self.assertEqual(list(point["by_label_source"]), ["benchmark"])
        self.assertTrue(row["report"]["ablation"]["ai_off_equals_rule"])
        report = row["report"]  # verified as the main product verifies a stored run
        self.assertEqual((report["run_id"], row["id"], report["created_at"]), (rid, rid, at))
        self.assertEqual(report["digest"], r3.report_digest(report))
        self.assertEqual(report["run_digest"], r3.run_digest(report))
        again = r3.run(cases("feature.leakage_suspect"), right, candidate="dclab:fake@v1", model_id="fake", at=at, run_id=rid)
        self.assertEqual((again["content_digest"], again["run_digest"]), (row["content_digest"], row["run_digest"]))
        self.assertIn("llm_invocation_id", report["note"])
        with self.assertRaises(ValueError):  # never sliced
            r3.run(cases("feature.leakage_suspect"), right, candidate="x" * 129, model_id="fake")

    def test_the_ablation_notices_an_off_arm_that_still_answers(self):
        from unittest import mock

        right = r3.R3Answerer(Says("feature.leakage_suspect", True, 0.99))
        with mock.patch.object(r3, "ai_off", lambda release, cases: [({"value": 0.99}, None)] * len(cases)):
            row = r3.run(cases("feature.leakage_suspect"), right, candidate="c", model_id="m")
        self.assertFalse(row["report"]["ablation"]["ai_off_equals_rule"])

    def test_an_unsure_model_abstains_and_the_rule_stands(self):
        unsure = r3.R3Answerer(Says("feature.leakage_suspect", True, 0.6))
        everything = r3.run(cases("feature.leakage_suspect"), unsure, candidate="c", model_id="m")["report"]["points"]["feature.leakage_suspect"]["sources"]["all"]
        self.assertEqual((everything["abstain_rate"], everything["ai_policy_accuracy_all"], everything["rule_accuracy_all"]), (1.0, 0.0, 0.0))

    def test_answer_and_label_rows_in_the_main_products_columns(self):
        rows = r3.answer_rows(cases("column.is_identifier", 3), r3.R3Answerer(Says("column.is_identifier", True, 0.99)), model_id="m",
                              release_version="v1", workspace="w1")
        self.assertTrue(all(set(r) <= set(recorded()["experience_tables"]["semantic_decision_answers"]) for r in rows))
        self.assertEqual({(r["llm_invocation_id"], r["policy_outcome"], r["level"]) for r in rows}, {(None, "rule", 0)})
        labels, left = r3.label_rows([{"id": "m", "source": "measured", "value": True, "at": "t0"}, {"id": "l1", "source": "benchmark", "value": True, "at": "t1"},
                                      {"id": "l2", "source": "blind", "value": False, "at": "t2"}], answer_row=rows[0], answer_id="a1")
        self.assertEqual([(r["labels_version"], r["supersedes_label_id"], r["label_source"]) for r in labels],
                         [(1, "a1", "benchmark"), (2, "l1", "blind")])  # numbered after the measured one is left out, each superseding the one before
        self.assertEqual((labels[0]["answer"], labels[1]["ground_truth"]), (rows[0]["answer"], {"value": False}))  # the answer row's columns copied
        self.assertEqual(left, {"measured": 1})
        with self.assertRaises(ValueError):
            r3.label_rows([{"id": "l", "source": "benchmark", "value": True}])  # no time: refused


class ReviewFixesTest(unittest.TestCase):
    def test_at_semantic_role_a_model_overrides_only_between_modeled_roles_the_validator_accepts(self):
        exclusion = r3.score("column.semantic_role", "categorical_code", ({"value": "identifier"}, 0.9), "categorical_code")
        self.assertEqual((exclusion["agreement"], exclusion["eligible"], exclusion["policy_correct"]), ("disagree", False, True))  # the rule stands
        modeled = r3.score("column.semantic_role", "categorical_code", ({"value": "numeric"}, 0.9), "numeric")
        self.assertEqual((modeled["eligible"], modeled["policy_correct"]), (True, True))
        refused = r3.score("column.semantic_role", "categorical_code", ({"value": "numeric"}, 0.9), "numeric", accepts=("categorical_code",))
        self.assertEqual((refused["eligible"], refused["policy_correct"]), (False, False))  # the validator would refuse it there

    def test_the_column_comes_from_the_case_as_its_harness_builds_one(self):
        from dataclasses import dataclass, field

        @dataclass(frozen=True)
        class Case:  # its Case: the column apart from the fields
            column: str
            fields: dict = field(default_factory=dict)
            context: dict = field(default_factory=dict)

        model = Says("column.semantic_role", "numeric", 0.95)
        r3.R3Answerer(model)({"purpose": "column.semantic_role"}, [Case("refund_amount", {"dtype": "float", "cardinality": "high"})])
        self.assertEqual((model.states[0]["column"], model.states[0]["name_tokens"]), ("refund_amount", ["refund", "amount"]))

    def test_every_statistic_counts_answered_cases_only(self):
        broken = 0

        class Breaks:
            by, name, version, points = "classifier", "b", "v1", frozenset({"feature.leakage_suspect"})

            def answer(self, point, state, options):
                nonlocal broken
                broken += 1
                if broken % 2:
                    raise RuntimeError("no answer")
                return L.Answer(by="classifier", answerer="b@v1", value=True, confidence=0.99)

        everything = r3.run(cases("feature.leakage_suspect"), r3.R3Answerer(Breaks()), candidate="c", model_id="m")["report"]["points"]["feature.leakage_suspect"]["sources"]["all"]
        self.assertEqual((everything["total"], everything["n"], everything["unavailable_rate"]), (40, 20, 0.5))
        self.assertEqual((everything["gap"]["point"], everything["disagreement_rate"]), (1.0, 1.0))  # over the 20 answered

    def test_an_answer_without_a_value_is_none_and_a_plain_yes_is_a_probability_of_one(self):
        rule = L.Answer(by="rule", answerer="r", value=False)
        self.assertEqual(mpl.as_raw("column.is_identifier", L.Answer(by="classifier", answerer="m@1", value=None)), (None, None))
        model = L.Answer(by="model_role", answerer="m@1", value=True)
        row = mpl.to_semantic_answer(record("column.is_identifier", [rule, model]), "m", "1", "b" * 64)
        self.assertEqual((row["answer"], row["in_acting_band"], row["agreement"]), ({"value": 1.0}, True, "disagree"))
        self.assertEqual(mpl.as_raw("column.is_identifier", L.Answer(by="classifier", answerer="m@1", value="yes", confidence=0.95)), (0.95, None))


class ExportTest(unittest.TestCase):
    def test_the_label_export_leaves_only_through_consent(self):
        try:
            from dclab_rnd import consent
        except ImportError as error:
            self.skipTest(str(error))
        rule = L.Answer(by="rule", answerer="r", value=False)
        recs = [record("column.is_identifier", [rule]), L.DecisionRecord(**{**record("column.time", [rule]).model_dump(), "id": "dr" + "1" * 12})]
        recs = [r.model_copy(update={"consent": "train"}) for r in recs]  # 23.1: written under train; an operate record never leaves
        labels = [L.DecisionLabel(id="dl" + "0" * 12, record=recs[0].id, point="column.is_identifier", value=True, source="benchmark", by="t",
                                  at="2026-10-15T00:00:00+00:00"),
                  L.DecisionLabel(id="dl" + "1" * 12, record=recs[0].id, point="column.is_identifier", value=True, source="measured", by="t",
                                  at="2026-10-15T00:00:01+00:00")]
        with self.assertRaises(consent.ConsentRefused):
            consent.export_main_product(recs, labels, {"train": False}, {})
        out = consent.export_main_product(recs, labels, {"train": True}, {})
        self.assertEqual([d["decision"]["point"] for d in out["decisions"]], ["column.is_identifier"])  # a Jev purpose only
        self.assertEqual(out["decisions"][0]["labels"][0]["label_source"], "benchmark")
        self.assertEqual(out["left_out"], {"measured": 1})

    def test_the_run_reads_dev_rows_only(self):
        try:
            from dclab_rnd import decision_r3
        except ImportError as error:
            self.skipTest(str(error))
        if not (ROOT / "research/llm-fine-tuning/experiments/decisions/generated_v1/decisions/column.is_identifier/rows.jsonl").exists():
            self.skipTest("the generated sets are not on this machine (make decisions-gen)")
        drawn = decision_r3.cases(per_point=50)
        self.assertEqual({c["partition"] for c in drawn}, {"dev"})
        self.assertEqual({c["point"] for c in drawn}, set(r3.releases()))


if __name__ == "__main__":
    unittest.main()
