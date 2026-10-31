"""Package 37.17 (docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): finished cases become case-file examples with their source,
never edited: an exam's written person is simulated, at half weight; a person's draft is real."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core.store import Provenance  # noqa: E402
from dclab_rnd.store.sets import cases  # noqa: E402

PROV = dict(workspace="w", entry="e:1", at="2026-10-31T00:00:00+00:00", sha256="0" * 64, consent="train")
CASE = {"version": 4, "goal": {"value": "keep members", "status": "said", "quote": "keep members"}, "route": {"track": "prediction", "tool": "ml_project"}}


def record(rid, source, body):
    return SimpleNamespace(id=rid, provenance=Provenance(source=source, **PROV), body=body)


class CaseSetTests(unittest.TestCase):
    def test_an_exam_case_is_simulated_and_a_draft_case_is_real_each_with_its_source(self):
        exam = record("r1", "live_tests/2026-10-31-conversation-exam/CEX-012/case/d-01/conversation.json",
                      {"id": "d-01", "transcript": [{"role": "person", "text": "we lose members"}, {"role": "assistant", "text": "Who acts?"}],
                       "notes": {"case_file": CASE}})
        draft = record("r2", "drafts/abc/draft.json", {"id": "abc", "case_state": {"case": CASE, "turns": [
            {"role": "user", "content": "we lose members"}, {"role": "assistant", "content": "Who acts?"}]}})
        other = record("r3", "drafts/x/draft.json", {"id": "x"})
        out = cases.from_records([exam, draft, other])
        self.assertEqual([e.check["origin"] for e in out], ["simulated", "real"])
        self.assertEqual([e.weight for e in out], [0.5, 1.0])
        self.assertEqual(out[0].stage, "case_files")
        self.assertIn("keep members", out[0].messages[-1].content)
        self.assertTrue(out[0].messages[-1].no_reasoning)  # a record built by code: no reasoning claimed
        self.assertEqual(out[0].provenance.source, exam.provenance.source)

    def test_an_exam_case_as_the_store_really_keeps_it(self):
        """2026-10-31: the store keeps an exam's conversation as an 'other' record whose source is only 'conversation.json'
        and whose workspace names the exam folder; the set found none of the 75 practice cases."""
        body = {"id": "d-40", "arm": "case", "transcript": [{"role": "person", "text": "we lose members"}, {"role": "assistant", "text": "Who acts?"}],
                "notes": {"case_file": CASE}}
        real = SimpleNamespace(id="other:live_tests__2026-10-30-conversation-exam__cex-012__case__d-40:d8:v1#0", kind="other",
                               provenance=Provenance(source="conversation.json", **{**PROV, "workspace": "live_tests__2026-10-30-conversation-exam__cex-012__case__d-40"}),
                               body=body)
        out = cases.from_records([real])
        self.assertEqual([(e.check["origin"], e.weight) for e in out], [("simulated", 0.5)])
        stray = SimpleNamespace(id="x", kind="other", provenance=Provenance(source="conversation.json", **{**PROV, "workspace": "some_project"}), body=body)
        self.assertEqual(cases.from_records([stray]), [])  # a conversation.json from elsewhere is not an exam's


if __name__ == "__main__":
    unittest.main()
