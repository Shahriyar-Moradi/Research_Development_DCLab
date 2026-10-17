"""Package 26.9: another answerer on every lake decision, every detail kept (a scripted model here: no torch, no weights)."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dclab_core import lake as L


def sample(i, point="column.is_identifier", split="dev", features=None, label=True, options=("yes", "no"), rule=True):
    return L.DecisionSample(sample_id=f"s{i}", decision_point=point, subject_kind="column", subject_ref="c",
                            features=features or {"column": f"col_{i}", "name_tokens": ["col"], "dtype": "integer", "uniqueness": "all"},
                            state_text="{}", options=list(options), label=label, rule_answer=rule,
                            provenance=L.Provenance(source="test", origin="construction", consent="ours", group=f"g{i}", split=split))


class Scripted:
    """Says yes to everything a column is asked, with a confidence; counts what it was asked."""

    def __init__(self):
        self.asked = []

    def batch_classify_text(self, texts, schema, include_confidence=False):
        self.asked += texts
        labels = schema["answer"]["labels"]
        first = next(iter(labels)) if isinstance(labels, dict) else labels[0]
        return [{"answer": {"label": first, "confidence": 0.75}} for _ in texts]


def write_gold(lake, rows):
    for r in rows:
        folder = lake / "gold" / "decisions" / r.decision_point / "v1"
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f"{r.provenance.split}.jsonl").open("a") as fh:
            fh.write(r.model_dump_json() + "\n")


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.lake = Path(tempfile.mkdtemp())
        self.rows = [sample(1), sample(2, label=False, rule=True), sample(3, split="sealed"), sample(4, split="train")]
        write_gold(self.lake, self.rows)

    def capture(self, model, **kw):
        from dclab_rnd.lake import answers

        with mock.patch.object(answers, "MANIFEST", self.lake / "MANIFEST.json"):
            return answers.capture("gliner2.5-decide", ("lake",), lake=self.lake, model=model, log=lambda *_: None,
                                   identity={"revision": "abc123", "library": "gliner2 2.1.0", "machine": "test"}, **kw)

    def test_every_detail_of_every_answer_is_kept(self):
        from dclab_rnd.lake import answers

        self.capture(Scripted())
        rec = answers.stored("gliner2.5-decide", "lake", "column.is_identifier", "dev", self.lake)[0]
        for key in ("sample_id", "input_text", "schema", "input_fingerprint", "output", "value", "confidence", "model", "revision",
                    "library", "machine", "render", "seconds", "at", "licence", "use", "group"):
            self.assertIn(key, rec)
        self.assertEqual((rec["value"], rec["confidence"], rec["revision"], rec["use"]), (True, 0.75, "abc123", "train"))
        self.assertIn("col_1", rec["input_text"])
        self.assertNotIn("label", rec)  # the labels stay in gold

    def test_a_second_run_asks_nothing_and_a_changed_decision_is_asked_again(self):
        first = Scripted()
        self.capture(first)
        self.assertEqual(len(first.asked), 4)
        again = Scripted()
        self.capture(again)
        self.assertEqual(again.asked, [])
        (self.lake / "gold" / "decisions" / "column.is_identifier" / "v1" / "dev.jsonl").write_text(
            sample(1, features={"column": "col_1", "name_tokens": ["col"], "dtype": "string", "uniqueness": "all"}).model_dump_json() + "\n")
        changed = Scripted()
        self.capture(changed)
        self.assertEqual(len(changed.asked), 1)

    def test_the_summary_scores_dev_only_never_the_sealed_split(self):
        from dclab_rnd.lake import answers

        self.capture(Scripted())
        with mock.patch.object(answers, "MANIFEST", self.lake / "MANIFEST.json"):
            doc = answers.summary(self.lake)
        point = doc["answerers"]["gliner2.5-decide"]["lake"]["column.is_identifier"]
        self.assertEqual(point["dev"]["agrees_with_label"], 0.5)  # yes on two decisions, one of which is no
        self.assertEqual(point["dev"]["rule_agrees_with_label"], 0.5)
        self.assertNotIn("agrees_with_label", point["sealed"])
        self.assertEqual(point["sealed"]["answered"], 1)

    def test_a_target_column_is_asked_with_its_own_options(self):
        from dclab_rnd.lake import answers

        table = sample(9, point="target.column", features={"columns": [{"name_tokens": ["churn"], "dtype": "boolean", "cardinality": "binary"}]},
                       options=("churn", "plan"), label="churn", rule=None)
        write_gold(self.lake, [table])
        self.capture(Scripted(), points=["target.column"])
        rec = answers.stored("gliner2.5-decide", "lake", "target.column", "dev", self.lake)[0]
        self.assertEqual((rec["schema"]["answer"]["labels"], rec["value"]), (["churn", "plan"], "churn"))

    def test_answers_reach_their_decisions_in_the_lake_and_never_the_label(self):
        from dclab_rnd.lake import answers

        self.capture(Scripted())
        out, why = answers.attach(self.rows, self.lake)
        self.assertEqual(out[1].other_answers["gliner2.5-decide"]["value"], True)
        self.assertIs(out[1].label, False)  # the label is untouched
        self.assertEqual(why["gliner2.5-decide: attached"], 4)
        moved = self.rows[0].model_copy(update={"features": {"column": "col_1", "dtype": "float"}})
        _, why = answers.attach([moved], self.lake)
        self.assertEqual(why["gliner2.5-decide: answered on another state (ask again)"], 1)

    def test_the_first_layout_moves_under_its_source(self):
        from dclab_rnd.lake import answers

        old = self.lake / "bronze" / "answers" / "gliner2.5-decide" / "column.time"
        old.mkdir(parents=True)
        (old / "dev.jsonl").write_text('{"sample_id": "s1"}\n')
        self.assertEqual(answers.migrate(self.lake), ["column.time"])
        self.assertTrue((self.lake / "bronze" / "answers" / "gliner2.5-decide" / "lake" / "column.time" / "dev.jsonl").is_file())

    def test_a_labelled_decision_in_gliner2s_training_format(self):
        from dclab_rnd.lake import answers

        d = {"decision_point": "column.semantic_role", "text": "Column 'zone'.", "label": "categorical_code",
             "schema": {"answer": {"labels": {"numeric": "a quantity", "categorical_code": "a code"}}}}
        row = answers.gliner2_row(d)
        task = row["output"]["classifications"][0]
        self.assertEqual((row["input"], task["task"], task["labels"], task["true_label"]), ("Column 'zone'.", "column.semantic_role",
                                                                                           ["numeric", "categorical_code"], ["categorical_code"]))
        self.assertEqual(task["label_descriptions"]["categorical_code"], "a code")
        yes = answers.gliner2_row({"decision_point": "column.time", "text": "t", "label": True, "schema": {"answer": {"labels": ["yes", "no"], "prompt": "q?"}}})
        self.assertEqual((yes["output"]["classifications"][0]["true_label"], yes["output"]["classifications"][0]["prompt"]), (["yes"], "q?"))

    def test_the_export_keeps_the_sealed_decisions_apart(self):
        from dclab_rnd.lake import answers

        got = answers.export_gliner2(("lake",), self.lake, self.lake / "export")
        self.assertEqual(sorted(got["counts"]["files"]), ["lake/column.is_identifier/dev.jsonl", "lake/column.is_identifier/sealed.jsonl",
                                                          "lake/column.is_identifier/train.jsonl"])

    def test_journeys_are_asked_from_the_persons_own_words(self):
        from dclab_rnd.lake import answers

        rows = list(answers.tests("journeys", self.lake))
        self.assertTrue(rows)
        one = next(r for r in rows if r["sample_id"].startswith("journey:dev:"))
        self.assertEqual(one["decision_point"], "problem.family_from_words")
        self.assertIn(one["label"], one["schema"]["answer"]["labels"])
        multi = [r for r in rows if r["schema"]["answer"].get("multi_label")]
        self.assertTrue(multi and all(isinstance(r["label"], list) for r in multi))  # a programme journey: several families

    def test_every_answerer_says_what_its_answers_may_be_used_for(self):
        from dclab_rnd.lake import answers

        for name, info in answers.ANSWERERS.items():
            self.assertIn(info["use"], ("train", "eval_only"), name)
            self.assertTrue(info["licence"] and info["why"])
        text, schema = answers.render("feature.leakage_suspect", {"column": "refund_at", "dtype": "datetime", "target": "fraud"}, ["yes", "no"])
        self.assertIn("refund_at", text)
        self.assertIn("fraud", text)
        self.assertEqual(schema["answer"]["labels"], ["yes", "no"])


if __name__ == "__main__":
    unittest.main()
