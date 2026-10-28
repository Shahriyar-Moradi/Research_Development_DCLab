"""The name rule for leakage (decide.rule, feature.leakage_suspect) measured on every real column name we hold.

Measured 2026-10-29: on 269 column names of 12 real datasets the rule flags 6, all of them known leaks; 4 of the 6 are
caught by the word "duration". Dropping that word to quiet a harmless name such as contract_duration (seen only in a
demo notebook) would lose them, so the words stay. A change to the leak words must keep this test passing.
Only the files' column names are read, never a row.
"""
import csv
import glob
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def real_columns() -> dict[tuple[str, str], None]:
    import pyarrow.parquet as pq

    seen: dict[tuple[str, str], None] = {}
    for path in sorted(glob.glob(str(ROOT / "data/public/*/*.parquet"))):
        for column in pq.read_schema(path).names:
            seen.setdefault((Path(path).parent.name, column), None)
    for path in sorted(glob.glob(str(ROOT / "data/project/*/*.csv"))):
        with open(path, newline="", encoding="utf-8", errors="replace") as handle:
            for column in next(csv.reader(handle)):
                seen.setdefault((Path(path).parent.name, column), None)
    return seen


class NameRuleOnRealColumnsTests(unittest.TestCase):
    def test_every_flag_is_a_known_leak_and_the_known_name_leaks_are_caught(self):
        from dclab_core.bands import name_tokens

        from dclab_rnd import decide
        from dclab_rnd.copilot.analyzer import blocked_columns

        columns = real_columns()
        if len(columns) < 200:
            self.skipTest("the real datasets are not all here")
        known = {c for names in blocked_columns().values() for c in names}
        flagged = {c for (_, c) in columns if decide.rule("feature.leakage_suspect", {"name_tokens": name_tokens(c)}).value}
        self.assertEqual(flagged - known, set(), "a harmless real column is flagged")
        must = {"duration", "final_customer_fare", "final_biker_fare", "Administrative_Duration",
                "Informational_Duration", "ProductRelated_Duration"}
        self.assertEqual(must - flagged, set(), "a real leak the name shows is no longer caught")


if __name__ == "__main__":
    unittest.main()
