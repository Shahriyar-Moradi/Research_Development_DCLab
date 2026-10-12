"""Benchmark v3 (package 15.9): v2 untouched, rare-event bases and traps added as a new version with its own sealed
fingerprints; each rare trap is really planted; the sealed set refuses to run unless frozen as committed."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class BenchmarkV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from dclab_rnd.agent_eval import benchmark as v2, benchmark_v3 as v3
        except ImportError as error:
            raise unittest.SkipTest(f"pandas or scikit-learn not installed: {error}")
        cls.v2, cls.v3 = v2, v3

    def test_v2_is_untouched(self):
        sealed = json.loads(self.v2.SEALED.read_text(encoding="utf-8"))
        self.assertEqual(sealed["version"], "benchmark_v2")
        self.assertEqual(self.v2.sealed_fingerprints(), sealed["cases"])

    def test_a_new_version_split_by_source_with_rare_bases_on_both_sides(self):
        v3 = self.v3
        dev, test = v3.split("dev"), v3.split("test")
        self.assertEqual(v3.VERSION, "benchmark_v3")
        self.assertGreaterEqual(len(test), 100)
        self.assertFalse({c.tags[1] for c in dev} & {c.tags[1] for c in test})
        self.assertEqual(len({c.id for c in v3.CASES}), len(v3.CASES))
        self.assertFalse({c.id for c in v3.CASES} & {c.id for c in self.v2.CASES})  # new ids, new seeds: new tables
        self.assertFalse({c.seed for c in v3.CASES} & {c.seed for c in self.v2.CASES})
        for split in (dev, test):
            rare = {c.trap for c in split if c.tags[3] == "rare"}
            self.assertTrue({"label_by_key", "partial_labels", "rare_entity_cluster", "oversampled_positives"} <= rare)

    def test_the_sealed_set_is_frozen_and_unchanged(self):
        sealed = json.loads(self.v3.SEALED.read_text(encoding="utf-8"))
        self.assertEqual(sealed["version"], "benchmark_v3")
        self.assertEqual(self.v3.sealed_fingerprints(), sealed["cases"],
                         "a sealed v3 case changed: make benchmark_v4 instead of editing this one")

    def test_the_rare_traps_are_planted(self):
        cases = {c.id: c for c in self.v3.CASES}
        clean = cases["BV3-PAYM-clean"].table()
        self.assertLess(clean["target"].mean(), 0.03)  # a rare event
        flagged = cases["BV3-PAYM-label_by_key"].table()
        self.assertTrue((flagged.loc[flagged["target"] == 1, "card_flagged"] == 1).all())  # every case's key is flagged
        self.assertGreater(flagged["card_flagged"].mean(), flagged["target"].mean())  # and its other rows too: inflation
        partial = cases["BV3-PAYM-partial_labels"]
        self.assertEqual(partial.labels, "partial")
        self.assertIn("never checked", partial.moment)
        sealed_sentences = {c.moment for c in self.v3.split("test") if c.trap == "partial_labels"}
        self.assertEqual(len(sealed_sentences), 3)  # each sealed base words it its own way
        self.assertTrue(cases["BV3-PAYM-rare_entity_cluster"].strict_group)
        free = cases["BV3-PAYM-unclustered_entity"]
        self.assertIn("terminal_ref", free.innocent)
        self.assertNotIn("BV3-PAYM-time_random_split", cases)  # it would overwrite the rare target
        self.assertFalse(any(c.tags[1] in ("diabetes", "wine") for c in self.v3.CASES))  # real tables cannot be reseeded
        self.assertLess(partial.table()["target"].mean(), clean["target"].mean())
        cluster = cases["BV3-PAYM-rare_entity_cluster"].table()
        share = cluster.groupby("card_ref")["target"].sum().sort_values(ascending=False)
        self.assertGreater(share.head(int(len(share) * 0.1)).sum() / max(share.sum(), 1), 0.6)  # a tenth of entities hold most cases
        over = cases["BV3-PAYM-oversampled_positives"]
        self.assertTrue(over.flag_duplicates)
        self.assertGreater(over.table().duplicated().sum(), 0)

    def test_partial_labels_are_scored_and_earlier_fingerprints_ignore_the_new_field(self):
        from dclab_rnd.agent_eval.cases import CASES as SUITE
        from dclab_rnd.agent_eval.run import score

        case = next(c for c in self.v3.CASES if c.trap == "partial_labels")
        project = {"solution": {"labels": "complete"}}
        self.assertFalse(score(case, project, {}, [], [], set())["leak_caught"])
        project = {"solution": {"labels": "partial"}}
        self.assertTrue(score(case, project, {}, [], [], set())["leak_caught"])
        self.assertTrue(all(c.labels is None for c in SUITE))  # the judgment suite's fingerprints did not move

    def test_the_moment_says_partial_and_the_tool_says_so(self):
        from dclab_core.understand import label_coverage

        dev = next(b for b in self.v3.RARE_BASES if b.split == "dev")
        self.assertEqual(label_coverage(dev.names["partial"])[0], "partial")  # the dev wording only: the sealed ones are not checked
        for ordinary in ("When the payment is authorised, before anyone reviews it.", "Readings that were never checked by hand are kept.",
                         "Tickets never reviewed by a person are routed automatically.",
                         "At the start of each month. Use the 2023 rows only; the older rows were never checked for typos."):
            self.assertIsNone(label_coverage(ordinary), ordinary)
        self.assertEqual(label_coverage("Most frauds are never found by anyone, so an unmarked payment is not a clean one.")[0], "partial")

    def test_a_sealed_run_without_a_record_or_a_frozen_set_is_refused(self):
        import tempfile
        from unittest import mock

        with self.assertRaises(SystemExit):
            self.v3.main(["--split", "test"])  # no record asked for
        out = Path(tempfile.mkdtemp()) / "BEN3-999_test_scripted.json"
        with mock.patch.object(self.v3, "frozen_problem", return_value="a case changed"), self.assertRaises(SystemExit):
            self.v3.main(["--split", "test", "--output", str(out)])
        self.assertFalse(out.exists())  # refused before anything ran or was written

    def test_the_standard_plan_keeps_the_moment_the_person_wrote(self):
        from dclab_rnd.intern.loop import _stated_moment

        task = "Build a leakage-safe model. The prediction is made: When the order is placed; the outcome is lateness."
        self.assertEqual(_stated_moment(task), "When the order is placed; the outcome is lateness.")
        self.assertEqual(_stated_moment("The prediction is made: at the start of each month.\n\nThen export it."),
                         "at the start of each month.")
        self.assertIsNone(_stated_moment("Build a model."))


if __name__ == "__main__":
    unittest.main()
