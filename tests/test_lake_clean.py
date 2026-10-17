"""Package 27.5: every gold set cleaned (pure states, personal data, near duplicates, evaluation tables), scored and carded."""

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from dclab_core import clean
from dclab_core import lake as L


def prov(group="dataset:a", split="train", origin="construction", licence=None):
    return L.Provenance(source="test", origin=origin, consent="ours", group=group, split=split, licence=licence)


def sample(i, features=None, text="{}", **kw):
    return L.DecisionSample(sample_id=f"s{i}", decision_point="column.is_identifier", subject_kind="column", subject_ref="c",
                            features=features if features is not None else {"dtype": "integer", "uniqueness": "all"}, state_text=text,
                            label=True, provenance=prov(**kw))


def example(i, user="Which column is the time?", answer='{"time_column": "ordered_at"}', family="split_design", verified=True, **kw):
    return L.TrainingExample(example_id=f"e{i}", family=family, messages=[{"role": "system", "content": "s"}, {"role": "user", "content": user},
                                                                          {"role": "assistant", "content": answer}],
                             verification=L.Verification(kind="construction", passed=verified), provenance=prov(**kw))


class CoreTests(unittest.TestCase):
    def test_near_duplicates_compare_without_case_or_punctuation_but_keep_their_numbers(self):
        a = [{"role": "user", "content": "Rows: 1200. Which column leaks?"}, {"role": "assistant", "content": "refund_at"}]
        b = [{"role": "system", "content": "other"}, {"role": "user", "content": "rows  1200 which COLUMN leaks"}, {"role": "assistant", "content": "Refund_at."}]
        self.assertEqual(clean.near_key(a), clean.near_key(b))
        self.assertNotEqual(clean.near_key(a), clean.near_key([*a[:1], {"role": "assistant", "content": "settled_at"}]))
        other = [{"role": "user", "content": "Rows: 9900. Which column leaks?"}, {"role": "assistant", "content": "refund_at"}]
        self.assertNotEqual(clean.near_key(a), clean.near_key(other))  # other numbers teach other things

    def test_a_value_in_the_features_is_impure_and_a_table_level_sample_is_checked_column_by_column(self):
        self.assertEqual(clean.impure({"dtype": "integer", "target": None}), [])  # an empty field is no field
        self.assertTrue(clean.impure({"dtype": "integer", "value": "John Smith"}))
        good = {"columns": [{"name_tokens": ["order", "id"], "dtype": "integer", "uniqueness": "all"}]}
        self.assertEqual(clean.impure(good), [])
        self.assertTrue(clean.impure({"columns": [{"name_tokens": ["order"], "sample_values": ["a@b.com"]}]}))
        table = {"rows": "tiny", "target": "class", "columns": 18,  # a table-level sample: a count of columns and the outcome's state
                 "target_state": {"column": "class", "name_tokens": ["class"], "dtype": "categorical", "cardinality": "low"}}
        self.assertEqual(clean.impure(table), [])
        self.assertTrue(clean.impure({**table, "rows": 148}))  # a raw row count is not a band
        self.assertTrue(clean.impure({**table, "target_state": {"column": "class", "top_values": ["yes"]}}))

    def test_personal_data_is_found_and_a_measured_number_is_not_it(self):
        self.assertEqual(clean.personal(["no one here", None]), {})
        self.assertIn("EMAIL", clean.personal(["write to jane.doe@example.com"]))
        self.assertEqual(clean.personal(['{"accuracy": 0.9941796008869179, "auc": 0.99369521734112}']), {})  # sixteen digits of a metric
        self.assertIn("EMAIL", clean.personal(['{"note": "mail jane.doe@example.com"}']))
        self.assertEqual(clean.personal(["284807 transactions, features V1-V28, a $50 fee (September-2013)"]), {})  # a study's content

    def test_quality_rewards_verification_and_a_citation_where_evidence_is_the_answer(self):
        msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "Leakage first (DCLAB-R04)."}]
        self.assertEqual(clean.quality("evidence_answer", msgs, True, reasoned=True), 1.0)
        self.assertLess(clean.quality("evidence_answer", msgs, True), 1.0)  # no verified reasoning
        self.assertLess(clean.quality("evidence_answer", [msgs[0], {"role": "assistant", "content": "Leakage first."}], True), 1.0)
        self.assertLess(clean.quality("split_design", msgs, False), 0.6)


class LakeCleanTests(unittest.TestCase):
    def test_a_duplicated_example_across_splits_leaves_train(self):
        from dclab_rnd.lake import clean_examples

        rows = [example(1, user="Rows: 100. Which column is the time?", group="selfplay:binary:12", split="train"),
                example(2, user="rows 100 which column is the TIME", group="selfplay:binary:13", split="sealed"),
                example(3, user="Which column is the time? (seed 14)", answer='{"time_column": "created_at"}', group="selfplay:binary:14", split="train")]
        kept, why = clean_examples(rows, set())
        self.assertEqual(sorted(r.example_id for r in kept), ["e2", "e3"])
        self.assertEqual(why["train row (nearly) identical to an evaluation row"], 1)

    def test_an_example_of_a_table_an_evaluation_reads_never_trains(self):
        from dclab_rnd.lake import clean_examples, eval_seeds

        seeds = eval_seeds()
        self.assertTrue({0, 1, 3, 4, 5} <= seeds)  # benchmark v4's and the journeys' seeds, the workflow models' sealed seed
        rows = [example(1, group="selfplay:binary:3", split="train"), example(2, answer="x", group="selfplay:binary:3", split="sealed"),
                example(3, answer="y", group="selfplay:binary:12", split="train")]
        kept, why = clean_examples(rows, seeds)
        self.assertEqual(sorted(r.example_id for r in kept), ["e2", "e3"])
        self.assertEqual(why["train example of a table an evaluation reads"], 1)
        self.assertTrue(all(r.quality > 0.5 for r in kept))

    def test_a_cell_value_or_personal_data_never_reaches_a_decision_set(self):
        from dclab_rnd.lake import clean_decisions

        rows = [sample(1), sample(2, features={"dtype": "integer", "value": "4111 1111 1111 1111"}, group="dataset:b"),
                sample(3, text="contact jane.doe@example.com", features={"dtype": "string"}, group="dataset:c")]
        kept, why = clean_decisions(rows)
        self.assertEqual([r.sample_id for r in kept], ["s1"])
        self.assertEqual((why["features not a pure state (a value)"], why["personal data"]), (1, 1))

    def test_every_written_set_has_its_card(self):
        from dclab_rnd.lake import write_set

        folder = Path(tempfile.mkdtemp()) / "solver" / "split_design" / "v1"
        got = write_set(folder, "training_example", "solver/split_design", [example(1, licence="CC-BY 4.0")], [], Counter({"duplicate": 2}))
        self.assertTrue(got["written"])
        card = json.loads((folder / "card.json").read_text())
        self.assertEqual((card["rows"], card["licences"], card["dropped"]), (1, {"CC-BY 4.0": 1}, {"duplicate": 2}))
        self.assertTrue(card["can_claim"] and card["cannot_claim"])
        self.assertIn("What it cannot claim", (folder / "CARD.md").read_text())


if __name__ == "__main__":
    unittest.main()
