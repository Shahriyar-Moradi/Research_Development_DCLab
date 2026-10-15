"""Package 16.14, its open halves: every benchmark case filed by its problem kind and the results reported per kind
with intervals (benchmark v4), each tabular kind's pitfalls measured on its replica, lessons filed by kind, and a
research track per kind with the kind's evidence in its index; v3 stays as it was."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas  # noqa: F401

    from dclab_rnd import kinds
    from dclab_rnd.agent_eval import benchmark_v3, benchmark_v4, by_kind
except ImportError as error:  # pandas is not in every environment
    MISSING = str(error)
else:
    MISSING = ""


@unittest.skipIf(MISSING, MISSING)
class BenchmarkByKindTest(unittest.TestCase):
    def test_every_case_is_filed_by_its_kind_and_v3_is_untouched(self):
        filed = {c.tags[-1] for c in benchmark_v4.CASES}
        self.assertEqual(filed, {"binary", "regression", "rare_event", "text_tabular"})
        self.assertTrue(all(c.id.startswith("BV3-") for c in benchmark_v3.CASES))
        self.assertEqual(len([c for c in benchmark_v4.CASES if c.id.startswith("BV4-") and not c.id.startswith("BV4-KIND")]), len(benchmark_v3.CASES))
        reseeded = {c.id.replace("BV4-", "BV3-", 1): c.seed for c in benchmark_v4.CASES if not c.id.startswith("BV4-KIND")}
        self.assertTrue(all(reseeded[c.id] != c.seed for c in benchmark_v3.CASES))  # every table differs from v3's

    def test_the_kinds_replica_cases_carry_their_planted_leaks(self):
        replica = {c.id: c for c in benchmark_v4.CASES if c.id.startswith("BV4-KIND")}
        self.assertEqual(replica["BV4-KIND-binary-0"].leaks, ("final_refund_amount",))
        self.assertEqual(set(replica["BV4-KIND-regression-3"].leaks), {"delivered_at", "courier_rating"})
        self.assertEqual(replica["BV4-KIND-text_tabular-0"].tags[0], "dev")
        self.assertEqual(replica["BV4-KIND-text_tabular-4"].tags[0], "test")
        self.assertIn("target", replica["BV4-KIND-binary-0"].table().columns)

    def test_a_replica_case_carries_its_time_its_entity_and_its_clean_columns_and_no_seed_already_read(self):
        from dclab_rnd.kinds import campaign

        replica = {c.id: c for c in benchmark_v4.CASES if c.id.startswith("BV4-KIND")}
        self.assertEqual(replica["BV4-KIND-binary-3"].time_column, "signup_time")
        self.assertEqual(replica["BV4-KIND-regression-3"].group_column, "restaurant_id")
        self.assertNotIn("order_id", replica["BV4-KIND-regression-3"].innocent)
        self.assertIn("monthly_fee", replica["BV4-KIND-binary-3"].innocent)
        self.assertFalse(set(benchmark_v4.REPLICA_SEEDS["test"]) & set(campaign.SEEDS))  # the kinds' acceptance read those

    def test_the_sealed_set_is_frozen(self):
        frozen = json.loads(benchmark_v4.SEALED.read_text(encoding="utf-8"))["cases"]
        self.assertEqual(frozen, benchmark_v4.sealed_fingerprints())

    def test_results_per_kind_carry_intervals(self):
        results = [{"case": "BV4-KIND-binary-0", "policy": "standard", "leak_caught": True, "false_alarm": False},
                   {"case": "BV4-KIND-binary-1", "policy": "standard", "leak_caught": False, "false_alarm": None}]
        cell = benchmark_v4.by_kind(results, benchmark_v4.CASES)["binary"]["standard"]
        self.assertEqual((cell["leaks_caught"], cell["false_alarms"]), ([1, 2], [0, 1]))
        self.assertEqual(len(cell["leaks_caught_ci95"]), 2)

    def test_a_kind_the_policies_cannot_judge_says_so(self):
        row = by_kind.row("forecasting")
        self.assertIsInstance(row["benchmark"], str)
        self.assertIn("not judged", row["benchmark"])
        self.assertIn("acceptance", row)


@unittest.skipIf(MISSING, MISSING)
class PitfallsByKindTest(unittest.TestCase):
    def test_a_kinds_leak_and_its_split_are_measured_on_its_replica(self):
        from dclab_rnd import pitfalls_by_kind

        binary = pitfalls_by_kind.measure("binary", 0)
        self.assertGreater(binary["pitfalls"]["leak_kept_in"]["gap"], 0.1)  # the refund paid after a cancellation
        self.assertEqual(binary["right_split"], "the latest rows by time")
        places = pitfalls_by_kind.measure("places", 0)
        self.assertIn("blocks", places["right_split"])
        self.assertIn("random_split", places["pitfalls"])

    def test_the_text_and_a_time_valued_leak_are_modelled_and_no_effect_is_said_as_none(self):
        from dclab_rnd import pitfalls_by_kind

        text = pitfalls_by_kind.measure("text_tabular", 0)["pitfalls"]["leak_kept_in"]
        self.assertGreater(text["right"], 0.7)  # the text is read: the baseline is the kind's, not a broken one
        regression = pitfalls_by_kind.measure("regression", 0)["pitfalls"]["leak_kept_in"]
        self.assertIn("delivered_at", regression["leaks"])
        doc = pitfalls_by_kind.result("binary", 999, seeds=(0, 1, 2))
        split = next(c for c in doc["claims"] if "random split" in c["statement"])
        self.assertIn("no difference", split["statement"])
        self.assertFalse(doc["evidence"]["random_split"]["measured"])

    def test_the_records_are_filed_by_kind(self):
        from dclab_rnd import tools

        found = [r for r in tools._index().records if r["type"] == "pitfall" and r["metadata"].get("problem_kind")]
        self.assertTrue({"binary", "regression", "places"} <= {r["metadata"]["problem_kind"] for r in found})
        checks = tools.kind_checks("binary")
        self.assertTrue(checks["pitfalls"] and "results" in checks and "lessons" in checks)


@unittest.skipIf(MISSING, MISSING)
class LessonsAndTracksTest(unittest.TestCase):
    def test_a_lesson_is_filed_by_its_projects_kind(self):
        from dclab_rnd import lessons

        project = {"id": "p1", "name": "P", "solution": {"target": "y", "task": "regression", "kind": "regression"}}
        scope = lessons.scope_of(project, {"task_type": "regression", "evidence": {}})
        self.assertEqual(scope["problem_kind"], "regression")
        lesson = {"id": "l1", "claim": "c", "against": "a", "next_test": "n", "scope": scope, "status": "accepted", "cites": []}
        self.assertEqual(lessons.as_record(lesson)["metadata"]["problem_kind"], "regression")
        store = lessons.FileLessons(Path(tempfile.mkdtemp()) / "lessons.json")
        self.assertEqual(lessons.by_kind("regression", store), [])

    def test_every_kind_has_a_track_of_its_own_subject_and_its_index_names_it(self):
        from dclab_rnd import research_map

        self.assertEqual(kinds.get("regression").evidence_track, "tabular-regression")
        self.assertEqual(kinds.get("places").evidence_track, "places-and-spatial")
        for kind in kinds.all_kinds():
            readme = ROOT / "research" / kind.evidence_track / "README.md"
            self.assertTrue(readme.exists(), kind.key)
        data = research_map.track_data(ROOT / "research" / "tabular-regression", research_map.tracked_files(fresh=True))
        self.assertEqual([k["key"] for k in data["kinds"]], ["regression"])
        self.assertIn("### ", research_map.render_index(data))


if __name__ == "__main__":
    unittest.main()
