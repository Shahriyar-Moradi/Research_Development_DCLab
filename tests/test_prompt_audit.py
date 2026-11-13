"""Package 37.14 (docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): no prompt asks a model to write its reasoning in its answer.
The owner's rule: reasoning is read from the model's summarized thinking (thinking on, display "summarized"), never asked for
in the reply's text: no <thinking> tags, no "reasoning" or "reason" field, no "show your steps". A new prompt that asks for it
fails here. The prompts written before the rule that still ask for a reason are listed with what they are, for the owner
to decide (they are not silently allowed: the list may only shrink)."""
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import prompts  # noqa: E402

ASKS_REASONING = re.compile(
    r"(?<!do not )(?<!never )(?:explain|show|give|write|include|state|list)[a-z ]{0,12}(?:your |the )?(?:reasoning|thought process|chain of thought|thinking)"
    r"|<thinking>|\"(?:reasoning|rationale|reason|why)\"\s*:|step by step", re.I)
BEFORE_THE_RULE = {  # prompt -> why it may still ask; the list may only shrink (the owner, 2026-10-31: reasoning from the thinking summary)
    "leakage_review": "a user-facing claim, not reasoning: each verdict's reason must quote the prediction moment and code checks it",
    "journey_checker": "a user-facing claim, not reasoning: the failed check shown to the owner when a written person is flagged",
    "reasoning_teacher": "teacher-written reasoning (the owner cancelled that backfill on 2026-10-28; the prompt is unused)",
}


class PromptAuditTests(unittest.TestCase):
    def test_no_prompt_asks_for_reasoning_in_its_answer_but_the_listed_ones(self):
        asking = {name for name in prompts.names() if ASKS_REASONING.search(prompts.text.__wrapped__(name) if hasattr(prompts.text, "__wrapped__") else (prompts.HERE / f"{name}.md").read_text(encoding="utf-8"))}
        self.assertEqual(asking - set(BEFORE_THE_RULE), set(), "a prompt asks the model to write its reasoning: read its thinking instead")

    def test_the_list_only_shrinks(self):
        for name in BEFORE_THE_RULE:
            self.assertIn(name, prompts.names(), f"{name} is gone: remove it from the list")
            text = (prompts.HERE / f"{name}.md").read_text(encoding="utf-8")
            self.assertTrue(ASKS_REASONING.search(text), f"{name} no longer asks for a reason: remove it from the list")

    def test_phase_13_prompts_ask_for_none(self):
        for name in ("case_agent", "case_specialist", "exam_plain", "exam_person", "exam_judge"):
            self.assertIsNone(ASKS_REASONING.search((prompts.HERE / f"{name}.md").read_text(encoding="utf-8")), name)


if __name__ == "__main__":
    unittest.main()
