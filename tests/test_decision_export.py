"""Package 22.5, open-source decision models: the decision sets exported in three formats from the same rows (a table,
chat JSONL with its dataset card, typed questions), the sealed partition never written; a language model as an answerer
through the same interface as the rule and the classifiers, compared on the dev partition only."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import decision_export as de  # noqa: E402
from dclab_core.answerers import ModelAnswerer, compare  # noqa: E402


def row(n, split="train", value=False, point="feature.leakage_suspect"):
    return {"record": f"dg{n:020d}", "point": point, "subject": {"kind": "column", "key": f"col_{n}"},
            "state": {"column": f"refund_{n}", "name_tokens": ["refund", str(n)], "dtype": "float", "uniqueness": "over_99pct", "target": "churned"},
            "options": ["yes", "no"], "answers": [{"by": "rule", "answerer": "datacard:purposes", "value": value, "confidence": None}],
            "value_used": value, "used": "rule", "label": {"value": value, "source": "benchmark"}, "group": f"dataset:t{n % 3}", "split": split}


class Advisor:
    def __init__(self, replies):
        self.replies, self.asked = list(replies), []

    def ask(self, purpose, evidence):
        self.asked.append((purpose, evidence))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.rows = [row(i, "train", i % 2 == 0) for i in range(6)] + [row(10, "dev", True)] + [row(20, "sealed", True), {**row(30, "dev"), "label": None}]

    def test_three_formats_hold_the_same_records_and_never_a_sealed_one(self):
        chat, table, typed = de.chat(self.rows), de.tabular(self.rows), de.typed(self.rows)
        records = [c["metadata"]["record"] for c in chat]
        self.assertEqual(records, [t["record"] for t in table])
        self.assertEqual(records, [t["record"] for t in typed])
        self.assertEqual(len(records), 7)  # the sealed row and the unlabelled one are left out
        self.assertNotIn(row(20)["record"], records)

    def test_the_chat_shape_and_the_answer_are_the_label(self):
        c = de.chat(self.rows)[0]
        self.assertEqual([m["role"] for m in c["messages"]], ["system", "user", "assistant"])
        self.assertEqual(json.loads(c["messages"][1]["content"])["state"]["dtype"], "float")
        self.assertEqual(json.loads(c["messages"][2]["content"]), {"answer": "yes"})  # row 0's label: True, and no reason learned
        self.assertFalse(any("label says so" in m["content"] for x in de.chat(self.rows) for m in x["messages"]))

    def test_a_model_is_asked_exactly_as_it_was_trained(self):
        from dclab_rnd.core_advisor import GatewayAdvisor

        sent = []

        class Client:
            def complete(self, messages, tools=None, max_tokens=0, **kw):
                sent.append(messages)
                return {"content": '{"answer": "yes"}', "finish_reason": "stop"}

        class Gateway:
            def client(self, purpose, **kw):
                return Client()

        r = self.rows[0]
        ModelAnswerer(GatewayAdvisor(Gateway()), {r["point"]}).answer(r["point"], r["state"], r["options"])
        self.assertEqual(sent[0], de.chat([r])[0]["messages"][:2])

    def test_the_table_and_typed_questions(self):
        t = de.tabular(self.rows)[0]
        self.assertEqual((t["state_dtype"], t["state_name_tokens"], t["label"], t["label_source"]), ("float", "refund 0", "yes", "benchmark"))
        self.assertNotIn("state_column", t)  # a whole name never: its tokens are there
        q = de.typed(self.rows)[0]
        self.assertEqual((q["primitive"], q["answer"], q["options"]), ("noul", "yes", ["yes", "no"]))  # the answer as one of the options

    def test_the_dataset_card_has_its_header_and_says_what_is_not_in_it(self):
        text = de.card("Test decisions", self.rows, "benchmark labels", ["Invented tables."])
        self.assertTrue(text.startswith("---\npretty_name: Test decisions\n"))
        self.assertIn("path: train.chat.jsonl", text)
        self.assertIn("sealed partition is not here", text)
        self.assertIn("- Invented tables.", text)

    def test_the_writer_never_opens_the_sealed_file(self):
        from dclab_rnd import decision_export

        with tempfile.TemporaryDirectory() as tmp:
            sets = Path(tmp) / "sets" / "feature.leakage_suspect"
            sets.mkdir(parents=True)
            (sets / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows if r["split"] != "sealed"))
            (sets / "sealed.jsonl").write_text("not json: reading this file would fail\n")
            manifest = decision_export.build(sets.parent, Path(tmp) / "out")
            self.assertTrue(manifest["same_records_in_every_format"])
            self.assertEqual(manifest["rows"], 7)
            self.assertTrue((Path(tmp) / "out" / "README.md").read_text().startswith("---"))
            (sets / "rows.jsonl").write_text(json.dumps(row(40, "sealed")) + "\n")  # a sealed row where it must not be
            with self.assertRaises(ValueError):
                decision_export.build(sets.parent, Path(tmp) / "out2")


class ModelAnswererTest(unittest.TestCase):
    def test_an_option_is_its_answer_and_anything_else_abstains(self):
        advisor = Advisor([{"answer": "Yes", "reason": "written after the outcome"}, {"answer": "maybe"}, None, RuntimeError("cut"), {"answer": False}])
        m = ModelAnswerer(advisor, {"feature.leakage_suspect"}, name="llm:test", version="p1")
        state = row(1)["state"]
        got = m.answer("feature.leakage_suspect", state, ["yes", "no"])
        self.assertEqual((got.value, got.by, got.answerer, got.reason), (True, "model_role", "llm:test@p1", "written after the outcome"))
        self.assertIsNone(m.answer("feature.leakage_suspect", state, ["yes", "no"]))  # not an option
        self.assertIsNone(m.answer("feature.leakage_suspect", state, ["yes", "no"]))  # no reply
        self.assertIsNone(m.answer("feature.leakage_suspect", state, ["yes", "no"]))  # a failed request
        self.assertEqual(m.errors, 1)
        self.assertFalse(m.answer("feature.leakage_suspect", state, ["yes", "no"]).value)  # a JSON false is a no
        self.assertIsNone(m.answer("column.time", state, ["yes", "no"]))  # not its point: not asked
        self.assertEqual(len(advisor.asked), 5)
        with self.assertRaises(ValueError):  # its purpose may see a column's state, not a project's
            ModelAnswerer(advisor, {"task.framing"})
        purpose, evidence = advisor.asked[0]
        self.assertEqual((purpose, set(evidence)), ("decision_answer", {"point", "question", "state", "options"}))

    def test_it_is_compared_beside_the_rule_on_the_same_decisions(self):
        from dclab_rnd import decision_model_compare as dmc

        rows = [row(i, "dev", i % 2 == 0) for i in range(6)]
        records, labels = dmc.as_records(rows)
        m = ModelAnswerer(Advisor([{"answer": "yes"}] * 6), {"feature.leakage_suspect"}, name="llm:test", version="p1")
        from dclab_core.answerers import RecordedRule

        result = compare(records, labels, [RecordedRule(), m])
        mine = result["llm:test@p1"]["all"]
        self.assertEqual((mine["answered"], mine["accuracy"]), (6, 0.5))  # always yes: right on the three planted leaks

    def test_the_comparison_never_falls_back_to_a_paid_model(self):
        from dclab_rnd import decision_model_compare as dmc

        key = "k" + "x" * 8
        remote = {"DCLAB_TIER_CHEAP_BASE_URL": "https://api.example.com/v1", "DCLAB_TIER_CHEAP_API_KEY": key, "DCLAB_TIER_CHEAP_MODEL": "m"}
        with mock.patch.dict("os.environ", remote):
            with self.assertRaises(SystemExit):
                dmc.local_gateway(None, False)
        with mock.patch.dict("os.environ", {**remote, "OPENAI_API_KEY": key, "DCLAB_TIER_TUNED_BASE_URL": "", "DCLAB_TIER_TUNED_MODEL": ""}):
            with self.assertRaises(SystemExit):  # no tuned model: never the cheap tier in its place
                dmc.local_gateway(None, True)
        tuned = {"DCLAB_TIER_TUNED_BASE_URL": "https://tuned.example.com/v1", "DCLAB_TIER_TUNED_MODEL": "mine", "DCLAB_TIER_TUNED_API_KEY": key}
        with mock.patch.dict("os.environ", tuned):
            with self.assertRaises(SystemExit):  # a remote tuned model only with a cap
                dmc.local_gateway(None, True)
        from dclab_rnd.models.gateway import Gateway

        local_tuned = {**remote, "OPENAI_API_KEY": key, "DCLAB_TIER_TUNED_BASE_URL": "http://localhost:8000/v1", "DCLAB_TIER_TUNED_MODEL": "mine"}
        with mock.patch.dict("os.environ", local_tuned), mock.patch.object(Gateway, "_tier_ready", lambda self, name: name != "tuned"):
            with self.assertRaises(SystemExit):  # a tuned tier that cannot answer falls back to another in the gateway: refused here
                dmc.local_gateway(None, True)

    def test_dev_rows_read_dev_only(self):
        from dclab_rnd import decision_model_compare as dmc

        with tempfile.TemporaryDirectory() as tmp:
            sets = Path(tmp) / "feature.leakage_suspect"
            sets.mkdir()
            (sets / "rows.jsonl").write_text("".join(json.dumps(row(i, "dev" if i < 4 else "train")) + "\n" for i in range(8)))
            (sets / "sealed.jsonl").write_text("not json\n")
            got = dmc.dev_rows(10, sets=Path(tmp))
        self.assertEqual(sorted(r["split"] for r in got), ["dev"] * 4)


if __name__ == "__main__":
    unittest.main()
