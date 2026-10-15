"""Claude helps label (2026-10-15): a model answers each drawn decision's blind view through the gateway; its labels are
source ``model``, named, weighted, never the truth, and kept apart from a person's. No model is reached here."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_core import ledger as contract
    from dclab_rnd import ledger
    from dclab_rnd.ledger import labels, model_labels
except ImportError as error:  # the agent environment may lack pydantic
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class ModelLabelsTest(unittest.TestCase):
    def setUp(self):
        self.book = ledger.FileLedger(Path(tempfile.mkdtemp()) / "ledger", "w1")
        for i in range(80):
            ledger.decide("column.is_identifier", ("column", f"c{i}"), answers=[ledger.answer("rule", "r", i % 2 == 0)],
                          value_used=i % 2 == 0, ledger=self.book, project="p1")

    def test_the_model_sees_the_blind_view_only_and_its_labels_are_kept_apart(self):
        seen = []

        def ask(view):
            seen.append(view)
            return {"answer": "yes", "confidence": "high", "reason": "unique on every row"}

        counts = model_labels.to_label(self.book, ask, "claude-sonnet-5-5")
        drawn = [r for r in self.book.records() if contract.blind_draw(r, model_labels.SEED)]
        self.assertEqual((counts["asked"], counts["labelled"]), (len(drawn), len(drawn)))
        shown = {k for v in seen for k in v}
        self.assertTrue(shown <= {"record", "point", "question", "answer", "subject", "state", "options", "weight", "seed"}, shown)
        written = [x for x in self.book.labels() if x.source == "model"]
        self.assertTrue(all(x.by == "model:claude-sonnet-5-5" and x.weight and x.value is True for x in written))
        record = next(r for r in self.book.records() if r.id == written[0].record)
        self.assertIsNone(contract.effective_label(record, self.book.labels()))  # a model's label is never the truth
        self.assertEqual(labels.summary(self.book)["by_source"]["model"], len(written))
        self.assertEqual(model_labels.to_label(self.book, ask, "claude-sonnet-5-5")["asked"], 0)  # once per model

    def test_unsure_and_off_list_answers_write_nothing(self):
        drawn = sum(1 for r in self.book.records() if contract.blind_draw(r, model_labels.SEED))  # ids are random: so is the draw's size
        counts = model_labels.to_label(self.book, lambda view: {"answer": None, "reason": "cannot tell"}, "m")
        self.assertEqual((counts["unsure"], counts["labelled"]), (drawn, 0))
        counts = model_labels.to_label(self.book, lambda view: {"answer": "maybe"}, "m")
        self.assertEqual((counts["refused"], counts["labelled"]), (drawn, 0))

    def test_a_model_label_names_its_model_and_v1_labels_still_read(self):
        with self.assertRaises(ValueError):
            contract.DecisionLabel(id="dl0000001", record="r1", point="column.is_identifier", value=True, source="model", by="someone",
                                   at="2026-10-15T00:00:00+00:00")
        old = contract.DecisionLabel.model_validate({"schema_version": "dclab_core.decision_label.v1", "id": "dl0000002", "record": "r1",
                                                     "point": "column.is_identifier", "value": True, "source": "benchmark", "by": "replica",
                                                     "at": "2026-10-15T00:00:00+00:00"})
        self.assertEqual(old.schema_version, "dclab_core.decision_label.v2")


@unittest.skipIf(MISSING, MISSING)
class StoredAndExportedTest(unittest.TestCase):
    def test_a_stored_v1_label_is_still_read_on_files(self):
        import json

        book = ledger.FileLedger(Path(tempfile.mkdtemp()) / "ledger", "w1")
        r = ledger.decide("column.is_identifier", ("column", "c1"), answers=[ledger.answer("rule", "r", True)], value_used=True, ledger=book)
        ledger.label(r, True, "benchmark", "replica:x", ledger=book)
        self.assertEqual(len(book.labels()), 1)
        for path in Path(book.root).rglob("*.jsonl"):  # the label as a v1 writer stored it
            text = path.read_text(encoding="utf-8").replace("dclab_core.decision_label.v2", "dclab_core.decision_label.v1")
            path.write_text(text, encoding="utf-8")
        self.assertEqual(len(book.labels()), 1)
        self.assertEqual(contract.DecisionLabel.read(json.loads(json.dumps({"schema_version": "dclab_core.decision_label.v1", "id": "dl0000009",
                         "record": "r1", "point": "column.is_identifier", "value": True, "source": "blind", "by": "reviewer", "weight": 10.0,
                         "at": "2026-10-15T00:00:00+00:00"}))).schema_version, "dclab_core.decision_label.v2")

    def test_a_model_label_goes_through_the_export_into_the_decision_set(self):
        from dclab_rnd import consent

        self.assertEqual(consent._by("model:claude-sonnet-5-5"), "model:claude-sonnet-5-5")  # a model's name is not a person
        self.assertEqual(consent._by("person:u1"), "person")

    def test_a_refusal_stops_the_run_and_says_why(self):
        book = ledger.FileLedger(Path(tempfile.mkdtemp()) / "ledger", "w1")
        for i in range(60):
            ledger.decide("column.is_identifier", ("column", f"c{i}"), answers=[ledger.answer("rule", "r", True)], value_used=True, ledger=book)

        def capped(view):
            raise RuntimeError("BudgetExceeded: this run has used its budget")

        counts = model_labels.to_label(book, capped, "m")
        self.assertEqual((counts["stopped"], counts["asked"], counts["labelled"]), (1, 1, 0))
        self.assertIn("BudgetExceeded", counts["why_stopped"])


if __name__ == "__main__":
    unittest.main()
