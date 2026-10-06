"""Prompts as code (package A4.3): versioned files, pinned to the scripted suite run, compared with a live baseline."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import prompts  # noqa: E402
from dclab_rnd.agent_eval import live  # noqa: E402

RESULTS = ROOT / "evidence/campaigns/agent_eval_v1/results"


class PromptGateTests(unittest.TestCase):
    def test_every_prompt_and_tool_schema_was_run_through_the_suite(self):
        from dclab_rnd.agent_eval import CASES, POLICIES
        latest = sorted(RESULTS.glob("AEV-*_scripted.json"))[-1]
        stored = json.loads(latest.read_text())
        # only a complete run vouches: every policy on every case, no policy erroring
        self.assertEqual((stored.get("policies"), [c["id"] for c in stored["cases"]]), (list(POLICIES), [c.id for c in CASES]), latest.name)
        self.assertTrue(all(s["errors"] == 0 for s in stored["summary"].values()), latest.name)
        recorded = stored.get("prompt_hashes") or {}
        now = prompts.fingerprints()
        changed = sorted(k for k in set(now) | set(recorded) if now.get(k) != recorded.get(k))
        self.assertEqual(changed, [], f"changed since {latest.name}: {changed}. A prompt or a tool schema is code: run `make agent-eval` "
                                      "(it stores a new result with the new hashes) and look at the scores before you commit.")

    def test_the_prompts_the_agents_send_are_the_files(self):
        from dclab_rnd.copilot import fixes
        from dclab_rnd.draft import chat
        from dclab_rnd.intern import loop
        from dclab_rnd.studio import explain
        self.assertEqual(loop.POLICY, prompts.text("intern"))
        self.assertEqual(chat.POLICY, prompts.text("home_agent"))
        self.assertIn(f"at most {explain.WORDS} words", explain.PROMPT)  # a constant fills its placeholder: the number lives in one place
        self.assertIn(f"at most {fixes.MAX_SENTENCES} sentences", fixes.PROMPT)
        self.assertGreaterEqual(len(prompts.names()), 10)
        self.assertIn("validator:graph", prompts.fingerprints())  # the validator's rules are part of the gate too
        with self.assertRaises(KeyError):
            prompts.text("stage_explain")  # a placeholder left unfilled is an error, not a "${words}" sent to a model

    def test_a_live_run_is_compared_with_the_last_one_of_the_same_model(self):
        then = {"by_case": {"AE-04": {"leak_caught": [1, 5], "false_alarm": [0, 5]}, "AE-18": {"leak_caught": [0, 0], "false_alarm": [1, 5]}},
                "scores": {"leaks_caught": {"mean": 0.2}, "false_alarms": {"mean": 0.2}}}
        now = {"by_case": {"AE-04": {"leak_caught": [4, 5], "false_alarm": [0, 5]}, "AE-18": {"leak_caught": [0, 0], "false_alarm": [1, 5]}},
               "scores": {"leaks_caught": {"mean": 0.8}, "false_alarms": {"mean": 0.2}}}
        change = live.compare(then, now)
        self.assertEqual(change["cases"], {"AE-04": {"leak_caught": [0.2, 0.8]}})  # only what moved
        self.assertEqual(change["scores"]["leaks_caught"], [0.2, 0.8])
        from dclab_rnd.agent_eval.cases import by_id
        found = live.baseline("qwen2.5-coder:1.5b", [by_id(i) for i in ("AE-04", "AE-11", "AE-18")])
        self.assertIsNotNone(found)
        self.assertIn("live_qwen2-5-coder-1-5b", found[0].name)
        self.assertIsNone(live.baseline("qwen2.5-coder:1.5b", [by_id("AE-04")]))  # another case set is never a baseline

    def test_no_code_writes_the_prompt_files(self):
        for path in (ROOT / "dclab_rnd").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotRegex(text, r"prompts/[\w.]*\.md[\"']?\)?\s*\.write_text|HERE\s*/[^\n]*write_", str(path))  # prompts are edited by people


if __name__ == "__main__":
    unittest.main()
