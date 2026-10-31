"""The conversation exam's benchmark: every run's arms in one table, the final verdict after the owner's labels, and the
exams' calls counted with their thinking, a vault by shape only."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval.conversation_exam import benchmark  # noqa: E402


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        self.results = self.home / "results"
        self.results.mkdir()
        arm = {"people": 40, "understanding": 0.85, "options_fit": 4.25, "route_reached": 0.9, "already_answered": 17,
               "safety_failures": 0, "traps": {"a": 1.0, "b": 0.5}, "silent": 0, "seconds_per_turn": 33.5, "spent_eur": 8.0}
        (self.results / "CEX-013_sealed_v1.json").write_text(json.dumps({"id": "CEX-013", "split": "sealed_v1", "summary": {"case": arm, "plain": arm},
                                                                         "verdict": {"status": "pending_owner_labels"}}))
        (self.results / "CEX-013_sealed_v1_owner_labels.json").write_text(json.dumps({"id": "CEX-013", "judge_owner_agreement": 0.66,
                                                                                      "owner_labels": {"n": 10}, "verdict": {"status": "fail"}}))
        work = self.home / "ws"
        (work / "model_exchanges" / "2026").mkdir(parents=True)
        (work / "model_exchanges" / "2026" / "31.jsonl").write_text(json.dumps({"purpose": "exam_judge", "whole": True, "thinking": [{"text": "t"}]}) + "\n")
        vault = work / "exam_vault" / "CEX-013_sealed_v1" / "model_exchanges"
        vault.mkdir(parents=True)
        (vault / "31.jsonl").write_text(json.dumps({"purpose": "case_agent", "whole": True, "thinking": [{"text": "SEALED WORDS"}],
                                                    "messages": [{"content": "SEALED WORDS"}]}) + "\n")
        self.doc = benchmark.build(self.results, work)

    def test_every_arm_has_a_row_and_the_owners_verdict_replaces_pending(self):
        rows = {(r["run"], r["arm"]): r for r in self.doc["runs"]}
        self.assertEqual(set(rows), {("CEX-013", "case"), ("CEX-013", "plain")})
        self.assertEqual(rows[("CEX-013", "case")]["verdict"], "fail")
        self.assertEqual(rows[("CEX-013", "case")]["traps"], 0.75)
        self.assertEqual(self.doc["labels"][0]["agreement"], 0.66)

    def test_thinking_is_counted_and_a_vault_gives_counts_never_words(self):
        self.assertEqual(self.doc["capture"]["workspace"]["exam_judge"], {"calls": 1, "whole": 1, "thinking": 1})
        self.assertEqual(self.doc["capture"]["vaults"]["CEX-013_sealed_v1"]["case_agent"]["thinking"], 1)
        self.assertNotIn("SEALED WORDS", json.dumps(self.doc) + benchmark.render(self.doc))


if __name__ == "__main__":
    unittest.main()
