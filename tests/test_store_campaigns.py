"""The campaigns keeper (2026-10-31, the owner: keep every costly run for the seven models): every campaign's results and
labels kept, a practice run whole, a sealed or test run as its numbers and labels only, Home's runs not kept twice."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.store.keepers import campaigns  # noqa: E402


class CampaignsKeeperTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        res = self.root / "journeys_v1" / "results"
        res.mkdir(parents=True)
        (res / "JRN-001_dev.json").write_text(json.dumps({"id": "JRN-001", "split": "dev", "rows": [{"said": "a long sentence the person wrote to us"}]}))
        (res / "JRN-002_sealed.json").write_text(json.dumps({"id": "JRN-002", "split": "sealed", "summary": {"passed": 7, "of": 10},
                                                             "rows": [{"case": "s-1", "said": "a long sentence of a sealed person, never kept"}]}))
        (res / "JRN-003.json").write_text(json.dumps({"id": "JRN-003", "split": "test", "verdict": "pass"}))
        home = self.root / "home_v1" / "results"
        home.mkdir(parents=True)
        (home / "HOME-001.json").write_text("{}")
        (self.root / "journeys_v1" / "labels").mkdir()
        (self.root / "journeys_v1" / "labels" / "x_owner_todo.json").write_text("[]")
        self.found = {s.source: s for s in campaigns.sources({"campaigns": str(self.root)})}

    def test_every_result_and_label_is_kept_and_home_is_not_kept_twice(self):
        self.assertEqual(set(self.found), {"campaigns/journeys_v1/results/JRN-001_dev.json", "campaigns/journeys_v1/results/JRN-002_sealed.json",
                                           "campaigns/journeys_v1/results/JRN-003.json", "campaigns/journeys_v1/labels/x_owner_todo.json"})
        self.assertIsNone(self.found["campaigns/journeys_v1/results/JRN-001_dev.json"].transform)  # practice: whole

    def test_a_sealed_or_test_run_keeps_numbers_and_labels_never_words(self):
        for name in ("JRN-002_sealed.json", "JRN-003.json"):
            s = self.found[f"campaigns/journeys_v1/results/{name}"]
            kept = json.loads(s.transform(Path(s.data).read_bytes()))
            self.assertIn("shape_only", kept)
            self.assertNotIn("never kept", json.dumps(kept))
        kept = json.loads(self.found["campaigns/journeys_v1/results/JRN-002_sealed.json"].transform(
            (self.root / "journeys_v1" / "results" / "JRN-002_sealed.json").read_bytes()))
        self.assertEqual((kept["summary"], kept["rows"][0]["case"]), ({"passed": 7, "of": 10}, "s-1"))

    def test_the_keeper_is_installed(self):
        from dclab_rnd.store.keepers import standard

        self.assertIn("campaigns", standard.install({}))


class HumanLabelsTests(unittest.TestCase):
    def test_a_persons_marks_on_practice_papers_become_gold_judge_lessons_and_sealed_ones_never(self):
        from dclab_rnd.store.sets import others

        base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, base, True)
        for run in ("CEX-cal-1", "CEX-013_sealed_v1"):
            (base / f"{run}_items").mkdir()
            (base / f"{run}_items" / "r1.json").write_text(json.dumps({"ref": "r1", "person": {"truth": {"goal": "keep members"}, "must_understand": ["the goal"]},
                                                                       "transcript": [{"role": "person", "text": "we lose members"}, {"role": "assistant", "text": "So you want to keep members?"}]}))
            (base / f"{run}_owner_labels.json").write_text(json.dumps({"by": "the owner", "entries": [{"ref": "r1", "owner": {"understood": [True], "options_fit": None,
                                                                                                                             "options_differ": None, "trap_handled": None}}]}))
        out = others.human_labels(base)
        self.assertEqual([e.check["run"] for e in out], ["CEX-cal-1"])
        self.assertEqual((out[0].weight, out[0].stage, json.loads(out[0].messages[-1].content)["understood"]), (1.0, "human_labels", [True]))
        self.assertTrue(out[0].messages[-1].no_reasoning)


if __name__ == "__main__":
    unittest.main()
