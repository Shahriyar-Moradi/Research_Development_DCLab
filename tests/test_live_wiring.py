"""The live tests of 2026-10-19 (HRN-022) and their review: a live run keeps each play's record, counts the requests that
failed beside what it spent, and reports both however it ends. No model is asked: scripted clients only."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas  # noqa: F401
    import pydantic  # noqa: F401
except ImportError as error:  # .venv-agent has no pandas
    MISSING = str(error)
else:
    MISSING = ""


class Failing:
    spent_eur, run_limit_eur = 0.02, None

    def complete(self, messages, tools=None, max_tokens=700):
        raise RuntimeError("no credit")


class Answering(Failing):
    def complete(self, messages, tools=None, max_tokens=700):
        return {"content": '{"ok": true}', "finish_reason": "stop"}


@unittest.skipIf(MISSING, MISSING)
class LiveWiringTest(unittest.TestCase):
    def test_an_advisor_counts_the_requests_that_failed(self):
        from dclab_rnd.core_advisor import GatewayAdvisor

        clients = {"understanding_critic": Failing(), "solution_router": Answering()}
        advisor = GatewayAdvisor(SimpleNamespace(client=lambda purpose, **_: clients.get(purpose)), cap_eur=0.5)
        with self.assertRaises(RuntimeError):
            advisor.ask("understanding_critic", {"card": {}})
        self.assertEqual(advisor.ask("solution_router", {"map": {}}), {"ok": True})
        self.assertEqual(advisor.failures, 1)

    def test_a_run_reports_its_spend_and_failures_however_it_ends(self):
        from dclab_rnd import real_problems as rp

        read = SimpleNamespace(advisor=SimpleNamespace(spent_eur=lambda: 0.1, failures=0), failures=2)
        critic, team = SimpleNamespace(spent_eur=lambda: 0.05, failures=1), None  # the run stopped before the team
        book, log = SimpleNamespace(records=lambda: [1, 2]), SimpleNamespace(read=lambda limit: [1])
        out = rp._spent({"result": "not built"}, True, read, critic, team, book, log)
        self.assertEqual((out["spent_eur"], out["failed_requests"], out["decisions"], out["events"]), (0.15, 3, 2, 1))
        self.assertNotIn("spent_eur", rp._spent({}, False, None, None, None, book, log))  # nothing live: nothing spent

    def test_each_play_is_kept_in_its_own_folder(self):
        from dclab_rnd.agent_eval import regression

        keep = Path(tempfile.mkdtemp())
        rows = regression.run(only=["JD-02"], workers=1, keep=keep)
        self.assertEqual([r["id"] for r in rows], ["JD-02"])
        self.assertTrue((keep / "JD-02" / "row.json").is_file())
        self.assertTrue((keep / "JD-02" / "events.jsonl").is_file())
        self.assertTrue((keep / "JD-02" / "workspace").is_dir())


if __name__ == "__main__":
    unittest.main()
