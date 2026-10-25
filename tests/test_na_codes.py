""""NA" as Namibia's code or as a missing value (the owner, 2026-10-25): kept in a column of two-letter codes, missing
everywhere else, the same way in an upload, a project's table, the cleaner and the code DCLab exports."""

import io
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.studio import na_codes as N  # noqa: E402

CSV = "country,amount,flag,note\nNA,1.5,True,a\nZA,NA,False,NA\nKE,2,NA,b\nBW,3,True,c\nZM,4,False,d\n"


def read(text: str):
    return N.na_as_read(pd.read_csv(io.StringIO(text), keep_default_na=False, na_values=N.READ_NA))


class RuleTest(unittest.TestCase):
    def test_na_is_a_code_among_two_letter_codes_and_missing_elsewhere(self):
        frame, kept = read(CSV)
        self.assertEqual(kept, ["country"])
        self.assertEqual(frame["country"].tolist(), ["NA", "ZA", "KE", "BW", "ZM"])  # Namibia kept
        self.assertTrue(np.isnan(frame["amount"][1]))  # R's missing number: missing
        self.assertTrue(pd.isna(frame["flag"][2]) and pd.isna(frame["note"][1]))

    def test_every_other_column_reads_as_pandas_would_have_read_it(self):
        frame, _ = read(CSV)
        default = pd.read_csv(io.StringIO(CSV))
        self.assertEqual({c: str(t) for c, t in frame.dtypes.items()}, {c: str(t) for c, t in default.dtypes.items()})
        for column in ("amount", "flag", "note"):
            self.assertEqual(frame[column].isna().tolist(), default[column].isna().tolist())
        self.assertEqual(frame["flag"].dropna().tolist(), default["flag"].dropna().tolist())  # true and false, not their words

    def test_a_column_is_codes_only_when_nearly_all_its_values_are_two_capital_letters(self):
        self.assertTrue(N.code_column(["NA", "ZA", "KE", "BW"]))
        self.assertFalse(N.code_column(["NA", "ZA", "NA", "ZA"]))  # two codes are too few to say
        self.assertFalse(N.code_column(["NA", "apple", "pear", "ZA", "KE"]))  # words, not codes
        self.assertTrue(N.code_column(["NA", "ZA", "KE", "BW", None, "N/A"] * 3))  # other missing markers do not count

    def test_na_itself_never_counts_toward_the_codes(self):  # the review: R's missing values in a column of two codes
        self.assertFalse(N.code_column(["NA"] * 95 + ["JR", "SR", "JR", "SR", "see note"]))
        self.assertFalse(N.code_column(["NA", "ZA", "KE"]))  # two other codes are too few
        self.assertTrue(N.code_column(pd.Series(["NA"] * 50 + ["ZA", "KE", "BW"] * 10)))  # counted over distinct values too

    def test_pandas_text_columns_of_the_string_dtype_are_read_the_same(self):  # the review: pandas 3 reads text as "string"
        frame = pd.DataFrame({"country": pd.array(["NA", "ZA", "KE", "BW"], dtype="string"), "amount": pd.array(["1.5", "NA", "2", "3"], dtype="string")})
        frame, kept = N.na_as_read(frame)
        self.assertEqual(kept, ["country"])
        self.assertEqual(str(frame["amount"].dtype), "float64")

    def test_the_edges_read_as_pandas_reads_them(self):  # the review: all "NA", a spreadsheet's dates, 20-digit ids
        dates = pd.Series([pd.Timestamp("2024-01-01"), "NA", pd.Timestamp("2024-01-03")], dtype=object)
        frame, _ = N.na_as_read(pd.DataFrame({"none": ["NA", "NA", "NA"], "when": dates, "big": ["12345678901234567890", "NA", "98765432109876543210"]}))
        self.assertEqual(str(frame["none"].dtype), "float64")
        self.assertTrue(str(frame["when"].dtype).startswith("datetime64"))
        self.assertEqual(frame["big"][0], "12345678901234567890")  # never a float that loses digits
        self.assertTrue(pd.isna(frame["big"][1]))

    def test_the_exported_copy_is_the_same_code_and_runs_on_its_own(self):
        space: dict = {}
        exec(N.SOURCE, space)
        frame, kept = space["na_as_read"](pd.read_csv(io.StringIO(CSV), keep_default_na=False, na_values=space["READ_NA"]))
        self.assertEqual(kept, ["country"])
        pd.testing.assert_frame_equal(frame, read(CSV)[0])


class PathsTest(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore")
        rng = np.random.default_rng(0)
        n = 120
        self.frame = pd.DataFrame({"country": rng.choice(["NA", "ZA", "BW", "KE"], n), "word": rng.choice(["NA", "apple", "pear"], n),
                                   "amount": np.where(rng.random(n) < 0.1, "NA", rng.normal(100, 10, n).round(2).astype(str)),
                                   "row": np.arange(n)})
        self.namibia = int((self.frame["country"] == "NA").sum())
        self.folder = Path(tempfile.mkdtemp())

    def test_an_upload_keeps_namibia_and_says_so(self):
        from dclab_rnd.draft import structure

        path = self.folder / "t.csv"
        self.frame.to_csv(path, index=False)
        frame, info = structure.to_table(path)
        self.assertEqual(int((frame["country"] == "NA").sum()), self.namibia)
        self.assertTrue(frame["word"].isna().any())
        self.assertEqual(str(frame["amount"].dtype), "float64")
        self.assertTrue(any("Namibia" in n for n in info.get("notes") or []))

    def test_a_projects_table_reads_namibia_back(self):
        from dclab_rnd.studio.data import load_table

        for name, sep in (("t.csv", ","), ("t.tsv", "\t")):
            path = self.folder / name
            self.frame.to_csv(path, index=False, sep=sep)
            frame = load_table(path)
            self.assertEqual(int((frame["country"] == "NA").sum()), self.namibia, name)
            self.assertEqual(str(frame["amount"].dtype), "float64", name)

    def test_the_cleaner_keeps_namibia_and_marks_other_nas_missing(self):
        from dclab_rnd.draft.clean import clean

        frame, log = clean(self.frame.copy())
        self.assertEqual(int((frame["country"] == "NA").sum()), self.namibia)
        self.assertFalse((frame["word"] == "NA").any())
        markers = next(e for e in log if e["step"] == "missing_markers")
        self.assertNotIn("country", markers["columns"])

    def test_a_spreadsheet_and_a_file_with_bad_lines_keep_namibia_too(self):
        from dclab_rnd.draft import structure

        path = self.folder / "t.xlsx"
        self.frame.to_excel(path, index=False)
        frame, info = structure.to_table(path)
        self.assertEqual(int((frame["country"] == "NA").sum()), self.namibia)
        self.assertTrue(frame["word"].isna().any())
        bad = self.folder / "bad.csv"
        bad.write_text(self.frame.to_csv(index=False) + "ZA,pear,1.0,999,extra,field\n")
        frame, info = structure.to_table(bad)
        self.assertEqual(int((frame["country"] == "NA").sum()), self.namibia)
        self.assertTrue(any("Skipped" in n for n in info["notes"]) and any("NA" in n for n in info["notes"]))

    def test_the_cleaning_log_says_namibia_was_kept(self):
        from dclab_rnd.draft.clean import clean

        _, log = clean(self.frame.copy())
        entry = next(e for e in log if e["step"] == "na_codes")
        self.assertEqual((entry["columns"], entry["changed"]), ([], self.namibia))  # a table-wide entry: the column's own step stays its own
        self.assertIn("`country`", entry["detail"])

    def test_the_scoring_service_reads_a_csv_as_training_did(self):
        sys.path.insert(0, str(ROOT / "dclab_rnd" / "studio" / "service_template"))
        self.addCleanup(sys.path.remove, str(ROOT / "dclab_rnd" / "studio" / "service_template"))
        import importlib

        scorer = importlib.import_module("scorer")
        one = self.frame.head(1).assign(country="NA", amount="NA").to_csv(index=False).encode()
        frame = scorer.read_csv(one, {"na_codes": N.codes_kept(N.na_as_read(self.frame.copy())[0])})
        self.assertEqual(frame["country"][0], "NA")  # one row: decided on the training table, never on the batch
        self.assertTrue(np.isnan(frame["amount"][0]))
        self.assertEqual(N.codes_kept(self.frame), ["country"])

    def test_the_exported_code_and_notebook_read_as_the_studio_reads(self):
        from dclab_rnd.studio import code, export

        self.assertIn(N.SOURCE, code.HEADER)
        self.assertIn("na_as_read(pd.read_csv(path, low_memory=False, keep_default_na=False, na_values=READ_NA))[0]", code.HEADER)
        self.assertTrue(export._read_code("data.csv").startswith("na_as_read(pd.read_csv('data.csv'"))
        self.assertEqual(export._read_code("data.parquet"), "pd.read_parquet('data.parquet')")


if __name__ == "__main__":
    unittest.main()
