"""The owner's decision on a provider model's outputs (2026-10-25): stored for evaluation and for training, silver and apart.

A set of Claude's or GPT's outputs takes its consent from ``dclab_rnd/store/owner_decisions.json``; without a recorded yes
it is evaluation only (``improve``), which an export for training refuses.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ProviderOutputsTest(unittest.TestCase):
    def test_the_recorded_decision_sets_the_consent(self):
        from dclab_rnd.store import provider_outputs

        d = Path(tempfile.mkdtemp())
        yes, no, bad = d / "yes.json", d / "no.json", d / "bad.json"
        yes.write_text(json.dumps({"provider_outputs": {"for_training": True}}), encoding="utf-8")
        no.write_text(json.dumps({"provider_outputs": {"for_training": False}}), encoding="utf-8")
        bad.write_text("{", encoding="utf-8")
        self.assertEqual(provider_outputs.consent(yes), "train")
        self.assertEqual(provider_outputs.consent(no), "improve")
        self.assertEqual(provider_outputs.consent(bad), "improve")  # no readable yes: evaluation only
        self.assertEqual(provider_outputs.consent(d / "missing.json"), "improve")

    def test_the_owners_record_says_yes_with_its_conditions(self):
        from dclab_rnd.store import provider_outputs

        record = json.loads(provider_outputs.DECISIONS.read_text(encoding="utf-8"))["provider_outputs"]
        self.assertTrue(record["for_training"])
        self.assertEqual(record["decided"], "2026-10-25")
        self.assertIn("silver", record["kept"])
        self.assertIn("terms", record["check_before_use"])

    def test_openrouter_qwen_trains_at_full_weight_by_the_owners_record(self):
        """The owner, 2026-11-11: no restriction on OpenRouter Qwen's outputs and reasoning; full weight, apart; its licence is
        checked before any fine-tune. Claude's record is unchanged (silver)."""
        from dclab_rnd.store import provider_outputs as P

        record = json.loads(P.DECISIONS.read_text(encoding="utf-8"))[P.OPENROUTER_QWEN]
        self.assertEqual((record["for_training"], record["weight"], record["source_class"], record["decided"]),
                         (True, 1.0, "open_weights_model", "2026-11-11"))
        self.assertIn("there is no restriction on openrouter qwen model", record["said"])
        self.assertIn("licence", record["check_before_use"])
        self.assertEqual((P.consent(decision=P.OPENROUTER_QWEN), P.weight(decision=P.OPENROUTER_QWEN)), ("train", 1.0))
        self.assertEqual(P.weight(), 0.5)  # Claude's: silver, unsaid weight
        d = Path(tempfile.mkdtemp())
        only_claude = d / "c.json"
        only_claude.write_text(json.dumps({"provider_outputs": {"for_training": True}}), encoding="utf-8")
        self.assertEqual(P.consent(only_claude, P.OPENROUTER_QWEN), "improve")  # no record of a yes for Qwen: evaluation only
        quoted = d / "q.json"
        quoted.write_text(json.dumps({"openrouter_qwen": {"for_training": "yes"}}), encoding="utf-8")
        self.assertEqual(P.consent(quoted, P.OPENROUTER_QWEN), "improve")  # only a recorded true is a yes

    def test_which_models_fall_under_the_qwen_decision(self):
        from dclab_rnd.store import provider_outputs as P

        self.assertTrue(P.openrouter_qwen("qwen/qwen3.8-flash"))  # OpenRouter's slug
        self.assertTrue(P.openrouter_qwen("qwen/qwen3.8-flash:free"))
        self.assertTrue(P.openrouter_qwen("openrouter/qwen/qwen3.8-flash"))
        self.assertTrue(P.openrouter_qwen("qwen3.8-flash", "openrouter", "open_weights_model"))  # the gateway's labels win
        self.assertFalse(P.openrouter_qwen("qwen/qwen3.8-flash", "together"))  # another host
        self.assertFalse(P.openrouter_qwen("qwen/qwen3.8-flash", "openrouter", "commercial_model"))
        self.assertFalse(P.openrouter_qwen("Qwen/Qwen2.5-72B-Instruct:novita"))  # a Hugging Face id: another host's
        self.assertFalse(P.openrouter_qwen("qwen2.5-coder:1.5b"))  # an Ollama tag: this machine's
        self.assertFalse(P.openrouter_qwen("claude-sonnet-5-5", "openrouter"))
        self.assertFalse(P.openrouter_qwen(None))
        self.assertEqual(P.answerer_name("qwen/qwen3.8-flash"), "qwen3.8-flash")
        self.assertEqual(P.answerer_name("openrouter/qwen/qwen3.8-flash:free"), "qwen3.8-flash")
        self.assertEqual(P.answerer_name("qwen2.5-coder:1.5b"), "qwen2.5-coder-1.5b")
        self.assertEqual(P.answerer_name(""), "unknown-model")


if __name__ == "__main__":
    unittest.main()
