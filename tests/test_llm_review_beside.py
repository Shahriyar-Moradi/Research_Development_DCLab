"""Package 29.18: a second reviewer of the past experiments (Claude, beside the campaign's reviewer): every critique kept
whole in its own files (what was sent, the raw reply), no result file, shared memory or report changed, nothing
earlier deleted, a review kept once never asked again."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import llm_review as R  # noqa: E402


class Client:
    def __init__(self):
        self.asked = 0

    def complete(self, messages, **kw):
        self.asked += 1
        return {"content": json.dumps({"decision": "keep", "overall_confidence": "medium", "challenges_to_claims": [{"claim_id": "C1", "issue": "x" * 600}]}),
                "usage": {"input_tokens": 900, "output_tokens": 300}, "finish_reason": "stop"}


class BesideTests(unittest.TestCase):
    def test_a_second_reviewer_changes_nothing_of_the_campaign_and_keeps_everything_whole(self):
        root = Path(tempfile.mkdtemp())
        results = root / R.CAMPAIGN_DIR / "results"
        results.mkdir(parents=True)
        for i in (1, 2):
            (results / f"EXP-00{i}_d_kind.json").write_text(json.dumps({"experiment_id": f"EXP-00{i}", "dataset": "d", "kind": "kind", "status": "completed",
                                                                         "claims": [{"claim_id": "C1", "statement": "s"}], "evidence": {"n": 5},
                                                                         "llm_review": {"model": "gpt-4o-mini", "review": {"decision": "earlier"}}}))
        shared = root / R.CAMPAIGN_DIR / R.MEMORY_NAME
        shared.write_text('{"earlier": true}\n')
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in results.glob("*.json")}
        client = Client()
        with mock.patch.object(R, "_client", return_value=client), mock.patch.object(R, "sync_campaign_outputs", side_effect=AssertionError("report changed")):
            self.assertEqual(R.run_llm_reviews(root, model="claude-sonnet-5-5", beside=True, sleep_s=0), 0)
            self.assertEqual(R.run_llm_reviews(root, model="claude-sonnet-5-5", beside=True, sleep_s=0), 0)  # nothing left to ask
        self.assertEqual(client.asked, 2)
        self.assertEqual({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in results.glob("*.json")}, before)  # results untouched
        self.assertEqual(shared.read_text(), '{"earlier": true}\n')  # the shared memory untouched
        kept = json.loads((root / R.CAMPAIGN_DIR / "reviews" / "claude-sonnet-5-5" / "EXP-001.json").read_text())
        self.assertEqual(len(json.loads(kept["raw_reply"])["challenges_to_claims"][0]["issue"]), 600)  # whole, never cut
        self.assertEqual(kept["sent"]["claims"][0]["claim_id"], "C1")
        self.assertEqual(kept["usage"], {"input_tokens": 900, "output_tokens": 300})
        memory = (root / R.CAMPAIGN_DIR / R.memory_name_for_model("claude-sonnet-5-5")).read_text().splitlines()
        self.assertEqual(len(memory), 2)


if __name__ == "__main__":
    unittest.main()
