"""Package 29.6 (the rest): the small processes' choices recorded where they are made: the reader the structure reader
chose (with what it looked at, never a line), every cleaning step applied per column it changed (kept by the rules), the
design a synthetic table follows (whose it is) and the record an evidence search ranked first (the question as a hash)."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas  # noqa: F401

    HAVE_ML = True
except ImportError:  # the agent environment
    HAVE_ML = False

SECRET = "Zanzibar-7731"  # a value of the table: it must never reach a record


class LedgerTest(unittest.TestCase):
    def setUp(self):
        from dclab_rnd import ledger

        self.tmp = Path(tempfile.mkdtemp())
        self.book = ledger.FileLedger(self.tmp / "ledger", "w")
        ledger.install(self.book)
        self.addCleanup(ledger.install, None)

    def records(self, point):
        return [r for r in self.book.records() if r.point == point]


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class ReaderAndCleaningTests(LedgerTest):
    def test_the_reader_chosen_is_recorded_with_what_it_looked_at(self):
        from dclab_core.ledger import check_state
        from dclab_rnd.draft import structure
        from dclab_rnd.ledger import capture

        files = {"a.csv": "id,city\n1,%s\n2,Rome\n" % SECRET, "b.tsv": "id\tcity\n1\t%s\n2\tRome\n" % SECRET,
                 "c.json": json.dumps([{"id": 1, "city": SECRET}, {"id": 2, "city": "Rome"}])}
        for i, (name, text) in enumerate(files.items()):
            path = self.tmp / name
            path.write_text(text)
            _, info = structure.to_table(path)
            capture.structure_read(f"d{i}", path, info["reader"])
        capture.structure_read("unknown", path, "a_reader_nobody_has")  # not one of the readers: nothing recorded
        odd = self.tmp / "report.acmecorp"
        odd.write_text("id,city\n1,Rome\n")
        self.assertEqual(structure.evidence(odd)["suffix"], "other")  # never the uploader's own word
        got = {r.draft: r for r in self.records("data.structure_format")}
        self.assertEqual([got[f"d{i}"].value_used for i in range(3)], ["csv", "tsv", "json"])
        self.assertEqual((got["d0"].state["suffix"], got["d0"].state["delimiter"], got["d0"].state["signature"]), ("csv", "comma", "text"))
        self.assertEqual(got["d1"].state["delimiter"], "tab")
        self.assertTrue(got["d2"].state["json_head"])
        for r in got.values():
            self.assertEqual(check_state(r.state), [])
            self.assertNotIn(SECRET, r.model_dump_json())
            self.assertFalse({"format", "reader", "parser", "parse_rate"} & set(r.state))  # a state never holds its own answer or its result

    def test_every_cleaning_step_applied_is_recorded_per_column_it_changed(self):
        import pandas as pd

        from dclab_core.ledger import check_state
        from dclab_rnd.draft import clean
        from dclab_rnd.ledger import capture
        from dclab_rnd.studio.data import profile_table

        raw = pd.DataFrame({"amount": ["1200", "350", "n/a", "78", "90"] * 20, "when": ["2024-01-0%d" % (i % 9 + 1) for i in range(100)],
                            "city": [f" {SECRET} ", "Rome", "Oslo", "Lima", "Kyiv"] * 20})
        frame, log = clean.clean(raw.copy())
        capture.cleaning("d1", log, len(frame), profile=profile_table(raw))
        applied = {(e["step"], c) for e in log for c in (e.get("columns") or ["*"])}
        kept = self.records("data.cleaning_keep")
        self.assertTrue(applied)
        self.assertEqual({(r.state["cleaning"], r.subject.key) for r in kept}, applied)  # no column renamed here
        self.assertTrue(all(r.value_used is True and r.used == "rule" and check_state(r.state) == [] for r in kept))
        self.assertTrue(all(SECRET not in r.model_dump_json() for r in kept))
        first = self.records("data.cleaning_step")  # 19.7's point, unchanged: each column's first step or none
        self.assertEqual({r.subject.key for r in first if r.subject.key != "*"}, set(raw.columns))


    def test_a_renamed_column_keeps_its_own_raw_bands(self):
        import pandas as pd

        from dclab_rnd.draft import clean
        from dclab_rnd.ledger import capture
        from dclab_rnd.studio.data import profile_table

        raw = pd.DataFrame({"Total  Amount": ["12", "n/a", "30", "41"] * 25, "Total Amount": list(range(100))})
        frame, log = clean.clean(raw.copy())
        capture.cleaning("d2", log, len(frame), profile=profile_table(raw))
        text = [r for r in self.records("data.cleaning_keep") if r.state.get("cleaning") in ("missing_markers", "numbers")]
        self.assertTrue(text and all(r.subject.key == "Total  Amount" and r.state["dtype"] == "string" for r in text))  # its own bands, not the integers'
        first = {r.subject.key: r.value_used for r in self.records("data.cleaning_step") if "dtype" in r.state}  # each column's first step
        self.assertEqual((first["Total  Amount"], first["Total Amount"]), ("missing_markers", "none"))

    def test_without_a_profile_each_step_is_recorded_with_the_table_size_only(self):
        from dclab_rnd.ledger import capture

        capture.cleaning("d3", [{"step": "duplicates", "columns": []}, {"step": "trim", "columns": ["city"]}], 50, profile=None)
        kept = self.records("data.cleaning_keep")
        self.assertEqual(sorted((r.subject.key, r.state["cleaning"]) for r in kept), [("*", "duplicates"), ("city", "trim")])
        self.assertTrue(all(set(r.state) == {"rows", "cleaning"} for r in kept))


class SyntheticAndEvidenceTests(LedgerTest):
    def test_a_synthetic_design_is_recorded_as_whose_it_is(self):
        from dclab_rnd.ledger import capture

        capture.synthetic_shape("p", {"source": "template", "template": "churn", "chosen_by": "user"}, 5000, True)
        capture.synthetic_shape("m", {"source": "model", "attempts": 1}, 2000, True)
        capture.synthetic_shape("f", {"source": "template", "template": "churn", "attempts": 2, "fallback_reason": "no valid spec"}, 2000, True)
        capture.synthetic_shape("x", {"source": "unknown"}, 2000, False)  # no design named: nothing recorded
        got = {r.draft: r for r in self.records("data.synthetic_shape")}
        self.assertEqual(set(got), {"p", "m", "f"})
        self.assertEqual([(got[k].value_used, got[k].used) for k in "pmf"], [("churn", "person"), ("model", "model_role"), ("churn", "rule")])
        self.assertTrue(all(set(r.state) <= {"rows", "model_available"} for r in got.values()))  # never what the design came to
        self.assertEqual([(a.by, a.value) for a in got["f"].answers], [("model_role", None), ("rule", "churn")])  # the model gave none

    def test_an_evidence_search_records_its_first_record_and_never_the_question(self):
        from dclab_rnd.ledger import capture

        question = f"Why does {SECRET} leak?"
        hits = [{"record_id": "EXP-001", "score": 3.2, "method": "bm25"}, {"record_id": "EXP-007", "score": 1.1, "method": "bm25"}]
        capture.evidence_search(question, hits, weak=2.0)
        capture.evidence_search("nothing", [], weak=2.0)  # no record came back: nothing decided
        hits = list(reversed(hits)) + []  # ranked EXP-007 first: the options never say which
        capture.evidence_search(question, hits, weak=2.0)
        first, second = self.records("evidence.records")
        self.assertEqual((first.value_used, first.subject.kind, first.state["hits"], first.state["weak"]), ("EXP-001", "conversation", 2, False))
        self.assertEqual((first.options, second.options, second.value_used), (["EXP-001", "EXP-007"], ["EXP-001", "EXP-007"], "EXP-007"))
        self.assertNotEqual(first.subject.key, second.subject.key)  # each search its own subject, never the question's hash
        r = first
        self.assertNotIn(SECRET, r.model_dump_json())
        self.assertNotIn("leak", r.model_dump_json())

    @unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
    def test_the_evidence_page_records_its_search(self):
        from dclab_rnd.agentic.pages import evidence

        out = evidence.ask("How should the holdout be used?")
        recorded = self.records("evidence.records")
        if out["records"]:
            self.assertEqual([r.value_used for r in recorded], [out["records"][0]["id"]])
        else:
            self.assertEqual(recorded, [])


if __name__ == "__main__":
    unittest.main()
