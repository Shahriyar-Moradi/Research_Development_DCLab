"""Package 32.11: the decision sets.

Every placed decision with labels is in its model type's folder with a manifest and a card; its best label wins (a person's
over a teacher's), a teacher's is silver with its weight; a decision's sealed rows never appear in its train or dev and the
sealed split is written once; a decision with no group is kept out; a decision a student answered beside its teachers is a
pair too.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)

AT = "2026-10-20T10:00:00+00:00"


@unittest.skipIf(MISSING, MISSING)
class DecisionSetsTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store import Layout, Store, curators, shapers

        self.layout = Layout(Path(tempfile.mkdtemp()) / "store")
        store = Store(self.layout)
        decisions, labels = [], []
        for i in range(40):
            answers = [{"by": "rule", "value": i % 2 == 0}] + ([{"by": "classifier", "value": True}] if i < 5 else [])
            decisions.append({"schema_version": "dclab_core.decision_record.v2", "id": f"dg{i:020d}", "point": "column.is_identifier", "at": AT,
                              "project": f"p{i}", "state": {"rows": "medium"}, "answers": answers, "used": "rule", "value_used": i % 2 == 0})
            labels.append({"schema_version": "dclab_core.decision_label.v2", "id": f"dl{i}", "record": f"dg{i:020d}", "point": "column.is_identifier",
                           "value": i % 2 == 0, "source": "model", "at": AT})
            if i < 10:
                labels.append({"schema_version": "dclab_core.decision_label.v2", "id": f"dp{i}", "record": f"dg{i:020d}", "point": "column.is_identifier",
                               "value": i % 3 == 0, "source": "user_decision_exposed", "at": AT})
        decisions.append({"schema_version": "dclab_core.decision_record.v2", "id": "dgnogroup000000000000", "point": "column.is_identifier", "at": AT,
                          "state": {}, "answers": [{"by": "rule", "value": True}], "used": "rule", "value_used": True})
        store.keep_log("decisions", "dev", "ledger/decisions/2026/10/20.jsonl", "".join(json.dumps(d) + "\n" for d in decisions).encode(), consent="train")
        store.keep_log("decisions", "dev", "ledger/labels/2026/10/20.jsonl", "".join(json.dumps(x) + "\n" for x in labels).encode(), consent="train")
        shapers.shape(self.layout)
        curators.curate(self.layout, run="r1")

    def rows(self, model_type, folder, split):
        path = self.layout.sets(model_type, *folder.split("/"), "v1") / f"{split}.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def test_the_sets_hold_the_best_labels_split_by_group_with_a_manifest_and_a_card(self):
        from dclab_rnd.store.sets import decisions

        report = decisions.build(self.layout)
        self.assertIn("m3_ml_decisions/general/column.is_identifier", report["sets"])
        self.assertIn("m1_students/column.is_identifier", report["sets"])
        self.assertIn("m1_students/column.is_identifier/pairs", report["sets"])
        quarantined = [p for p in (self.layout.root / "quarantine").rglob("*.jsonl")]
        self.assertTrue(any("missing" in p.parent.name or "broken" in p.parent.name for p in quarantined))  # the decision with no group: kept out
        base = self.layout.sets("m1_students", "column.is_identifier", "v1")
        manifest = json.loads((base / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(manifest["counts"][s] for s in ("train", "dev", "sealed")), 40)
        self.assertTrue((base / "CARD.md").exists())
        every = {s: self.rows("m1_students", "column.is_identifier", s) for s in ("train", "dev", "sealed")}
        groups = {s: {r["provenance"]["project"] for r in rows} for s, rows in every.items()}
        self.assertFalse(groups["sealed"] & (groups["train"] | groups["dev"]))  # a sealed group never in train or dev
        by_id = {r["id"].split(":", 1)[1]: r for rows in every.values() for r in rows}
        person = [r for r in by_id.values() if r["label_source"] == "person"]
        self.assertEqual(len(person), 10)  # the person's label wins over the model's
        self.assertTrue(all(r["weight"] == 1.0 for r in person))
        silver = [r for r in by_id.values() if r["label_source"] == "teacher"]
        self.assertTrue(silver and all(r["weight"] < 1 for r in silver))
        self.assertEqual(len(self.rows("m1_students", "column.is_identifier/pairs", "train") + self.rows("m1_students", "column.is_identifier/pairs", "dev")
                             + self.rows("m1_students", "column.is_identifier/pairs", "sealed")), 5)

    def test_the_sealed_split_is_written_once(self):
        from dclab_rnd.store.sets import decisions

        decisions.build(self.layout)
        sealed = self.layout.sets("m1_students", "column.is_identifier", "v1") / "sealed.jsonl"
        before = sealed.read_bytes()
        sealed.write_bytes(before)  # untouched by the next build
        decisions.build(self.layout)
        self.assertEqual(sealed.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
