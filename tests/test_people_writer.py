"""Written people for the understanding students (S8 B1, 2026-10-25): the request holds a kind's column names and never
a value; the truth (kind, family) is the code's, never the writer's; a person the checker says does not fit is kept
apart; every request and reply is kept whole; a batch already answered is not asked again; a cap stops it."""

import json
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import people_writer as P  # noqa: E402


class FakeClient:
    model = "claude-sonnet-5-5"

    def __init__(self, replies):
        self.replies, self.spent_eur, self.sent = replies, 0.0, []

    def complete(self, messages, tools=None, max_tokens=None):
        self.sent.append(messages)
        self.spent_eur += 0.01
        return {"content": self.replies.pop(0), "finish_reason": "stop", "usage": {"output_tokens": 10}}


PEOPLE = json.dumps({"people": [
    {"words": "need AI to see who cancels", "answers": {"target": "churned", "made_up_key": "x", "family": "yes or no"}},
    {"words": "we want clustering of our subscribers by price", "answers": {}},
    {"words": "", "answers": {}},  # refused: no words
    {"words": "predict churn, kind=regression please", "kind": "regression", "family": "regression", "answers": {}}]})
CHECKS = json.dumps({"checks": [{"index": 0, "fits": True, "why": "churn"}, {"index": 1, "fits": False, "why": "segments, not churn"},
                                {"index": 2, "fits": True, "why": "the person still asks about churn"}]})


class WriterTest(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore")
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"})
        env.start()
        self.addCleanup(env.stop)
        self.home = Path(tempfile.mkdtemp())
        out = mock.patch.multiple(P, OUT=self.home / "homework.json", FLAGGED=self.home / "flagged.json", BATCHES=1)
        out.start()
        self.addCleanup(out.stop)

    def test_a_request_names_columns_never_values(self):
        from dclab_rnd import kinds

        messages = P.request("binary", 0)
        sent = messages[1]["content"]
        facts = json.loads(sent)["problem"]
        self.assertEqual(facts["outcome_column"], "churned")
        self.assertIn("monthly_fee", facts["columns"])
        table = kinds.get("binary").case(0).table
        values = {str(v) for c in table.columns for v in table[c].dropna().astype(str).unique()[:50] if len(str(v)) > 3}
        self.assertFalse([v for v in values if v in sent and v not in facts["columns"] and v not in ("churned",)][:5])  # no cell value sent
        self.assertEqual(len(json.loads(sent)["their_businesses"]), P.PER_BATCH)
        self.assertNotEqual(P.businesses("binary", 0), P.businesses("binary", 1))  # each batch its own businesses

    def test_the_truth_is_the_codes_and_a_person_that_does_not_fit_is_kept_apart(self):
        writer, checker = FakeClient([PEOPLE]), FakeClient([CHECKS])
        with mock.patch.object(P, "_client", side_effect=[writer, checker]):
            got = P.write(1.0, self.home, kinds_=["binary"])
        self.assertEqual((got["people"], got["flagged"]), (2, 1))  # the empty one refused; the clustering one kept apart
        people = json.loads((self.home / "homework.json").read_text())
        self.assertEqual({(p["kind"], json.dumps(p["family"])) for p in people}, {("binary", '"classification"')})  # never the writer's
        self.assertNotIn("made_up_key", people[0]["answers"])
        flagged = json.loads((self.home / "flagged.json").read_text())
        self.assertEqual(flagged[0]["check"], "segments, not churn")
        kept = json.loads((self.home / "replies" / "binary-0.json").read_text())
        self.assertEqual((kept["reply"]["content"], kept["check_reply"]["content"]), (PEOPLE, CHECKS))  # whole
        with mock.patch.object(P, "_client", side_effect=AssertionError("asked again")):
            P.write(1.0, self.home, kinds_=["binary"])  # resumable: nothing sent twice

    def test_the_cap_stops_before_a_request(self):
        (self.home / "replies").mkdir()
        (self.home / "replies" / "x-0.json").write_text(json.dumps({"spent_eur": 2.0, "kind": "binary", "batch": 9, "prompt": "p",
                                                                    "reply": {"content": "{}"}, "check_reply": None}))
        with mock.patch.object(P, "_client", side_effect=AssertionError("sent past the cap")):
            got = P.write(1.0, self.home, kinds_=["binary"])
        self.assertIn("cap", got["stopped"])


if __name__ == "__main__":
    unittest.main()
