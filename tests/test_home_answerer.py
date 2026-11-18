"""The owner, 2026-11-18: no provider output trains until the rights review. A Home conversation named no model, so 1,276
rows passed the export's hold uncounted as anyone's. Each one now says who wrote it: the models whose requests name its
draft, an unknown hosted model when the draft shows a model ran, or DCLab's rules."""
import itertools
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

AT = "2026-11-10T10:00:00+00:00"
IDS = itertools.count(1)
CLAUDE = {"model": "claude-sonnet-5-5"}
QWEN_LOG = {"model": "qwen/qwen3.8-flash"}  # the request log names no provider
QWEN_EXCHANGE = {"model": "qwen/qwen3.8-flash", "provider": "openrouter", "source_class": "open_weights_model"}


def request(draft, workspace="dev", purpose="home_agent", outcome="ok", **body):
    from dclab_core.store import Provenance, StoreRecord

    return StoreRecord(id=f"q{next(IDS)}", kind="model_requests", type="request", at=AT,
                       body={"draft": draft, "purpose": purpose, "outcome": outcome, **body}, split="train",
                       provenance=Provenance(workspace=workspace, source="model_requests.jsonl", entry="e:1", at=AT, sha256="0" * 64, consent="train"))


def draft(**extra):
    from dclab_core.store import Provenance, StoreRecord

    body = {"id": "d1", "intent": {"track": "prediction"}, "updated": AT,
            "messages": [{"role": "person", "text": "Which customers leave?"}, {"role": "agent", "text": "When is the prediction made?"}], **extra}
    return StoreRecord(id="r1", kind="conversations", type="draft", at=AT, body=body, split="train",
                       provenance=Provenance(workspace="dev", source="drafts/d1/draft.json", entry="e:d1", at=AT, sha256="0" * 64, consent="train", draft="d1"))


class HomeAnswererTests(unittest.TestCase):
    def check(self, *requests, **extra):
        from dclab_rnd.store.sets import llm

        return llm.from_draft(draft(**extra), {}, llm.home_models(requests)).check

    def test_a_claude_request_for_the_draft_names_claude(self):
        c = self.check(request("d1", **CLAUDE))
        self.assertEqual((c["answerer"], c["model"], c["models"]), ("claude:claude-sonnet-5-5", "claude-sonnet-5-5", ["claude:claude-sonnet-5-5"]))

    def test_the_log_and_the_exchange_of_one_qwen_request_are_one_answerer(self):
        c = self.check(request("d1", **QWEN_LOG), request("d1", **QWEN_EXCHANGE))
        self.assertEqual(c["answerer"], "openrouter:qwen/qwen3.8-flash")
        self.assertEqual((c["provider"], c["source_class"]), ("openrouter", "open_weights_model"))  # the copy naming its provider wins

    def test_claude_and_qwen_on_one_draft_are_mixed_and_claude_is_its_model(self):
        c = self.check(request("d1", **QWEN_EXCHANGE), request("d1", purpose="case_agent", **CLAUDE))
        self.assertEqual(c["answerer"], "mixed:claude:claude-sonnet-5-5+openrouter:qwen/qwen3.8-flash")
        self.assertEqual(c["model"], "claude-sonnet-5-5")  # the most restrictive, so the reasoning sets treat it as Claude's

    def test_requests_that_wrote_no_turn_do_not_count(self):
        c = self.check(request("d1", outcome="the provider refused the key", **CLAUDE), request("d1", shadow=True, **CLAUDE),
                       request("d1", purpose="routing_labeler", **CLAUDE), request("d1", purpose="exam_judge", **CLAUDE),
                       request("d1", workspace="other", **CLAUDE), request("d2", **CLAUDE))
        self.assertEqual(c["answerer"], "rules")
        self.assertNotIn("model", c)

    def test_a_draft_that_shows_a_model_ran_without_its_requests_is_an_unknown_model(self):
        signs = [{"agent": {"mode": "model"}}, {"case_state": {"stage": "problem"}},
                 {"messages": [{"role": "person", "text": "Hi"}, {"role": "agent", "by": "case_agent", "text": "Hello"}]},
                 {"messages": [{"role": "person", "text": "Hi"}, {"role": "agent", "by": "model", "text": "Hello"}]},
                 {"questions": [{"id": "q1", "source": "critic"}]},
                 {"messages": [{"role": "person", "text": "Hi"}, {"role": "agent", "text": "Why?", "question": {"id": "q2", "source": "advisory"}}]}]
        for extra in signs:
            c = self.check(**extra)
            self.assertEqual(c["answerer"], "frontier:unknown", extra)
            self.assertNotIn("model", c)

    def test_a_rules_conversation_is_the_rules_and_never_a_model_called_rules(self):
        c = self.check(messages=[{"role": "person", "text": "Hi"}, {"role": "agent", "by": "code", "text": "Hello"}])
        self.assertEqual(c["answerer"], "rules")
        self.assertNotIn("model", c)
        self.assertIn("answerer_basis", c)  # the owner can read why it was judged the rules'

    def test_the_export_holds_claude_and_the_unknown_model_and_keeps_qwen_and_the_rules(self):
        from dclab_rnd.store.export_chat import provider_of

        held = lambda *r, **e: provider_of({"check": self.check(*r, **e)})
        self.assertEqual(held(request("d1", **CLAUDE)), ("claude:claude-sonnet-5-5", True))
        self.assertEqual(held(agent={"mode": "model"}), ("frontier:unknown", True))
        self.assertEqual(held(request("d1", **QWEN_EXCHANGE), request("d1", **CLAUDE))[0],
                         "mixed:claude:claude-sonnet-5-5+openrouter:qwen/qwen3.8-flash")
        self.assertEqual(held(request("d1", **QWEN_LOG)), (None, True))  # the owner's yes of 2026-11-11
        self.assertEqual(held(), (None, True))  # DCLab's rules: recorded, and not held


if __name__ == "__main__":
    unittest.main()
