"""Package 30.23: one data foundation (the owner's pattern 10).

The same customer in four systems resolves into one view (every record with its own person, the pair F1 with its interval); a table's
catalog entry says its owner and a failed check; a table read as of a date returns no later row, and a history table the row valid then;
the resolution is kept whole beside its record.
"""

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
except ImportError as error:  # .venv-agent
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class FoundationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_rnd import foundation

        cls.doc = foundation.run(0)

    def test_four_systems_resolve_into_one_view_of_each_person(self):
        d = self.doc
        self.assertEqual(d["records"], sum(e["rows"] for e in d["entries"]))
        self.assertLessEqual(d["entities_mixing_people"], d["people"] * 0.01)  # almost never two people merged
        self.assertLessEqual(abs(d["entities"] - d["people"]), d["people"] * 0.02)
        self.assertGreaterEqual(d["pair_f1"], 0.9)
        self.assertEqual({e["owner"] for e in d["entries"]}, {"sales operations", "finance", "customer care", "e-commerce"})

    def test_an_entry_says_a_failed_check_and_as_of_never_returns_a_later_row(self):
        from dclab_core import foundation

        frame = pd.DataFrame({"id": [1, 1, None], "since": ["2026-01-01", "2026-02-01", "not a date"]})
        e = foundation.entry(frame, name="t", system="s", owner="o", keys=["id"], date_column="since")
        self.assertFalse(e.healthy)
        self.assertEqual({c.name for c in e.checks if not c.passed}, {"id never empty", "id unique", "since readable"})
        events = pd.DataFrame({"at": ["2026-01-01", "2026-03-01", "2026-05-01"], "v": [1, 2, 3]})
        self.assertEqual(list(foundation.as_of(events, "2026-03-01", date_column="at")["v"]), [1, 2])
        history = pd.DataFrame({"customer": ["c", "c"], "tier": ["silver", "gold"], "from": ["2025-01-01", "2026-02-01"], "to": ["2026-02-01", None]})
        self.assertEqual(list(foundation.as_of(history, "2026-01-15", valid_from="from", valid_to="to")["tier"]), ["silver"])
        self.assertEqual(list(foundation.as_of(history, "2026-06-01", valid_from="from", valid_to="to")["tier"]), ["gold"])
        backdated = history.assign(written=["2025-01-01", "2026-07-01"])  # gold valid from February, but written in July
        self.assertEqual(list(foundation.as_of(backdated, "2026-03-01", date_column="written", valid_from="from", valid_to="to")["tier"]), [])
        self.assertEqual(list(foundation.as_of(backdated, "2026-08-01", date_column="written", valid_from="from", valid_to="to")["tier"]), ["gold"])
        self.assertEqual(foundation.TableEntry.read(e.model_dump_json()), e)

    def test_the_resolution_is_kept_whole(self):
        from dclab_rnd import foundation

        with tempfile.TemporaryDirectory() as tmp:
            path = foundation.write(self.doc, campaign=Path(tmp) / "foundation_v1")
            record = json.loads(path.read_text(encoding="utf-8"))
            with gzip.open(path.parent.parent / record["detail"], "rt", encoding="utf-8") as fh:
                kept = [json.loads(line) for line in fh]
            self.assertEqual(len(kept), self.doc["records"])
            self.assertNotIn("entity", record)


if __name__ == "__main__":
    unittest.main()
