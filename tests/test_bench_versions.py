"""DCLab Bench B5 (2026-11-03): each prompt's versions, the scripted checks that recorded each hash, the live results placed in
the version in force on their date, and whether the version in force now was measured live."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.bench import versions  # noqa: E402

HISTORY = {"case_agent": [{"commit": "aaa", "date": "2026-10-30T10:00:00+04:00", "subject": "first", "hash": "h1"},
                          {"commit": "bbb", "date": "2026-11-01T10:00:00+04:00", "subject": "second", "hash": "h2"},
                          {"commit": "ccc", "date": "2026-11-02T10:00:00+04:00", "subject": "act first", "hash": "h3"}]}


def write(folder: Path, suite: str, ref: str, doc: dict) -> None:
    (folder / suite / "results").mkdir(parents=True, exist_ok=True)
    (folder / suite / "results" / f"{ref}.json").write_text(json.dumps(doc))


class VersionTests(unittest.TestCase):
    def setUp(self):
        self.campaigns = Path(tempfile.mkdtemp())
        write(self.campaigns, "conversation_exam_v1", "CEX-001_dev", {"at": "2026-10-31T12:00:00+00:00", "verdict": "fail", "spent_eur": 2.0, "arms": ["case"]})
        write(self.campaigns, "conversation_exam_v1", "CEX-002_dev", {"at": "2026-11-01T12:00:00+00:00", "verdict": "pass", "spent_eur": 1.0, "arms": ["case"]})
        write(self.campaigns, "agent_eval_v1", "AEV-001", {"at": "2026-11-02T12:00:00+00:00", "verdict": "pass",
                                                          "prompt_hashes": {"prompt:case_agent": "h3"}})
        self.patch = mock.patch.object(versions, "history", return_value=HISTORY)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def agent(self, now):
        return versions.timeline("case_agent", campaigns=self.campaigns, now=now)[0]

    def test_a_live_result_is_placed_in_the_version_in_force_on_its_date(self):
        a = self.agent({"prompt:case_agent": "h3"})
        by = {v["hash"]: v for v in a["versions"]}
        self.assertEqual([r["ref"] for r in by["h1"]["live"]], ["CEX-001_dev"])
        self.assertEqual([r["ref"] for r in by["h2"]["live"]], ["CEX-002_dev"])
        self.assertEqual((by["h1"]["failed"], by["h2"]["passed"], by["h1"]["spent_eur"]), (1, 1, 2.0))

    def test_a_scripted_check_is_a_check_never_a_measurement(self):
        a = self.agent({"prompt:case_agent": "h3"})
        current = a["versions"][-1]
        self.assertEqual((current["checks"], current["checks_passed"], current["live"]), (1, 1, []))
        self.assertTrue(a["committed"])
        self.assertFalse(a["measured_now"])  # the version in force now has no live result
        self.assertEqual(a["last_pass"]["hash"], "h2")
        self.assertEqual(a["since_last_pass"], 1)

    def test_an_uncommitted_edit_is_neither_committed_nor_measured(self):
        a = self.agent({"prompt:case_agent": "edited"})
        self.assertFalse(a["committed"])
        self.assertFalse(a["measured_now"])

    def test_every_mapped_prompt_exists(self):
        names = {p.stem for p in (ROOT / "dclab_rnd" / "prompts").glob("*.md")}
        for suite, prompts in versions.SUITE_PROMPTS.items():
            self.assertTrue(set(prompts) <= names, suite)



class ArmTests(unittest.TestCase):
    """Review finding 4: every exam result was credited to every exam prompt, even one whose arm never ran."""

    def test_an_exam_result_measures_only_the_prompts_of_its_arms(self):
        sent = versions._sent
        self.assertEqual(sent("conversation_exam_v1", {"arms": ["plain"]}, versions.SUITE_PROMPTS["conversation_exam_v1"]),
                         ("exam_plain", "exam_person", "exam_judge"))
        self.assertIn("case_agent", sent("conversation_exam_v1", {"scores": {"case": [], "plain": []}}, versions.SUITE_PROMPTS["conversation_exam_v1"]))
        self.assertEqual(sent("conversation_exam_v1", {"verdict": "relabelled"}, versions.SUITE_PROMPTS["conversation_exam_v1"]), ())

if __name__ == "__main__":
    unittest.main()
