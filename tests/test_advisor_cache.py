"""An advisor answers the same question once: an identical ask (same purpose, same evidence) is a route recomputed with nothing changed, and paying for it twice
is how a measurement runs out of its cap. An error is never kept."""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd.core_advisor import GatewayAdvisor
except ImportError as error:  # the agent environment
    MISSING = str(error)
else:
    MISSING = ""


class Client:
    spent_eur = 0.0

    def __init__(self, replies):
        self.replies, self.sent = list(replies), 0

    def complete(self, messages, tools=None, max_tokens=0):
        self.sent += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return {"content": reply, "finish_reason": "stop"}


class Gate:
    def __init__(self, client):
        self._client = client

    def client(self, purpose, **kw):
        return self._client


@unittest.skipIf(MISSING, MISSING)
class AdvisorCacheTest(unittest.TestCase):
    def test_an_identical_ask_is_answered_once_and_a_different_one_is_sent(self):
        client = Client([json.dumps({"choice": "anomaly"}), json.dumps({"choice": "rare_event"})])
        advisor = GatewayAdvisor(Gate(client))
        a = advisor.ask("solution_router", {"sub": "sp1", "route": {"chosen": "anomaly"}})
        b = advisor.ask("solution_router", {"route": {"chosen": "anomaly"}, "sub": "sp1"})  # the same evidence, its keys in another order
        c = advisor.ask("solution_router", {"sub": "sp2", "route": {"chosen": "anomaly"}})
        self.assertEqual((a, b, c), ({"choice": "anomaly"}, {"choice": "anomaly"}, {"choice": "rare_event"}))
        self.assertEqual(client.sent, 2)
        self.assertEqual((advisor.calls, advisor.hits), ({"solution_router": 2}, {"solution_router": 1}))

    def test_the_same_evidence_for_another_purpose_is_another_question(self):
        client = Client([json.dumps({"x": 1}), json.dumps({"y": 2})])
        advisor = GatewayAdvisor(Gate(client))
        advisor.ask("decomposer", {"words": "w"})
        advisor.ask("map_reviewer", {"words": "w"})
        self.assertEqual(client.sent, 2)

    def test_an_error_is_never_kept_and_is_asked_again(self):
        client = Client([RuntimeError("provider down"), json.dumps({"ok": True})])
        advisor = GatewayAdvisor(Gate(client))
        with self.assertRaises(RuntimeError):
            advisor.ask("decomposer", {"words": "w"})
        self.assertEqual(advisor.ask("decomposer", {"words": "w"}), {"ok": True})
        self.assertEqual(client.sent, 2)

    def test_no_model_serving_is_not_kept_either(self):
        advisor = GatewayAdvisor(Gate(None))
        self.assertIsNone(advisor.ask("decomposer", {"words": "w"}))
        self.assertEqual(advisor.hits, {})


if __name__ == "__main__":
    unittest.main()
