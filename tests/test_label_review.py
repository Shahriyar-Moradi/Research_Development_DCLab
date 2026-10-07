"""Package 14.4: the reviewers' kit: blank forms without the answers, a key, and the agreement report."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dclab_rnd.agent_eval import benchmark, review  # noqa: E402


def fill(path, decide):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for r in rows:
        r["decision"] = decide(r)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=review.FORM_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class ReviewKitTests(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.info = review.export(self.out, ["A", "B"], per_trap=1)
        self.key = json.loads((self.out / "key.json").read_text())["key"]
        self.cases = {c.id: c for c in [*benchmark.split("dev"), *benchmark.split("test")]}

    def test_the_forms_hold_no_answer_and_no_case_name(self):
        text = (self.out / "form_A.csv").read_text()
        for case_id in self.cases:
            self.assertNotIn(case_id, text)  # references only
        with (self.out / "form_A.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(set(rows[0]), set(review.FORM_COLUMNS))
        self.assertTrue(all(r["decision"] == "" for r in rows))
        self.assertNotIn("target", {r["column"] for r in rows})
        splits = {k["split"] for k in self.key.values()}
        self.assertEqual(splits, {"dev", "test"})  # both sets are sampled, every trap family at most once per set

    def test_the_columns_of_a_case_do_not_show_which_one_was_added(self):
        """Every generator appends its planted column last: in the form the order is shuffled, the same on every form."""
        import csv as _csv
        with (self.out / "form_A.csv").open(newline="") as handle:
            rows = list(_csv.DictReader(handle))
        with (self.out / "form_B.csv").open(newline="") as handle:
            self.assertEqual([(r["case_ref"], r["column"]) for r in rows], [(r["case_ref"], r["column"]) for r in _csv.DictReader(handle)])
        last_is_planted = 0
        planted_cases = 0
        for ref, k in self.key.items():
            case = self.cases[k["case"]]
            columns = [r["column"] for r in rows if r["case_ref"] == ref]
            natural = [c for c in case.table().columns if c != "target"]
            self.assertEqual(sorted(columns), sorted(natural))
            if case.leaks:
                planted_cases += 1
                last_is_planted += columns[-1] == case.leaks[-1] and natural[-1] == case.leaks[-1]
        self.assertLess(last_is_planted, planted_cases * 0.6)  # in table order it was nearly always the last one

    def test_a_case_whose_planted_column_was_left_blank_is_not_confirmed(self):
        leak_ref = next(ref for ref, k in self.key.items() if self.cases[k["case"]].leaks)
        leak_col = self.cases[self.key[leak_ref]["case"]].leaks[0]

        def right(row):
            case = self.cases[self.key[row["case_ref"]]["case"]]
            return review.expected(case).get(row["column"], "usable")
        for name in ("A", "B"):
            fill(self.out / f"form_{name}.csv", lambda row: "" if (row["case_ref"], row["column"]) == (leak_ref, leak_col) else right(row))
        report = review.agreement(self.out / "form_A.csv", self.out / "form_B.csv", self.out / "key.json")
        self.assertNotIn(self.key[leak_ref]["case"], report["cases_confirmed_by_both"])
        self.assertIn(leak_col, [u["column"] for u in report["labelled_columns_not_decided_by_both"]])

    def test_the_references_do_not_follow_the_benchmarks_order(self):
        refs_in_order = [k["case"] for _, k in sorted(self.key.items())]
        self.assertNotEqual(refs_in_order, sorted(refs_in_order))

    def test_a_form_or_a_key_is_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            review.export(self.out, ["A", "B"], per_trap=1)

    def test_two_reviewers_who_match_the_benchmark_agree_perfectly(self):
        def right(row):
            case = self.cases[self.key[row["case_ref"]]["case"]]
            return review.expected(case).get(row["column"], "usable")
        for name in ("A", "B"):
            fill(self.out / f"form_{name}.csv", right)
        report = review.agreement(self.out / "form_A.csv", self.out / "form_B.csv", self.out / "key.json")
        self.assertEqual(report["reviewers"]["agreement"], 1.0)
        self.assertEqual(report["reviewer_a_vs_benchmark"]["agreement"], 1.0)
        self.assertEqual(report["disagreements"], [])
        self.assertEqual(len(report["cases_confirmed_by_both"]), self.info["cases"])

    def test_a_disagreement_is_listed_with_all_three_views(self):
        def right(row):
            case = self.cases[self.key[row["case_ref"]]["case"]]
            return review.expected(case).get(row["column"], "usable")
        leak_ref = next(ref for ref, k in self.key.items() if self.cases[k["case"]].leaks)
        leak_col = self.cases[self.key[leak_ref]["case"]].leaks[0]
        fill(self.out / "form_A.csv", right)
        fill(self.out / "form_B.csv", lambda row: "usable" if (row["case_ref"], row["column"]) == (leak_ref, leak_col) else right(row))
        report = review.agreement(self.out / "form_A.csv", self.out / "form_B.csv", self.out / "key.json")
        self.assertEqual(len(report["disagreements"]), 1)
        d = report["disagreements"][0]
        self.assertEqual((d["column"], d["reviewer_a"], d["reviewer_b"], d["benchmark"]), (leak_col, "keep_out", "usable", "keep_out"))
        self.assertNotIn(d["case"], report["cases_confirmed_by_both"])
        self.assertLess(report["reviewers"]["agreement"], 1.0)

    def test_a_blank_is_not_an_agreement_and_a_wrong_word_is_refused(self):
        fill(self.out / "form_A.csv", lambda row: "usable")
        report = review.agreement(self.out / "form_A.csv", self.out / "form_B.csv", self.out / "key.json")  # B is still blank
        self.assertEqual(report["rows_both_reviewers"], 0)
        fill(self.out / "form_B.csv", lambda row: "maybe")
        with self.assertRaises(ValueError):
            review.read_form(self.out / "form_B.csv")

    def test_kappa_is_chance_corrected(self):
        self.assertEqual(review.kappa(["keep_out", "usable"], ["keep_out", "usable"]), 1.0)
        self.assertEqual(review.kappa(["usable", "usable"], ["usable", "usable"]), None)  # no variation: undefined, not 1
        self.assertLess(review.kappa(["keep_out", "usable", "usable", "usable"], ["usable", "usable", "usable", "keep_out"]), 0.1)


if __name__ == "__main__":
    unittest.main()
