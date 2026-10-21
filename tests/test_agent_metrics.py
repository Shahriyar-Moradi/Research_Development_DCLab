"""The agents' metrics ledger (the owner, 2026-10-20): every request, step, check, proposal and rating of an agent kept under
the hood, per prompt version, with intervals; the evaluation suites' results read in once each; one scorecard route."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd import agent_metrics as am  # noqa: E402


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())

    def test_emit_appends_a_line_and_never_raises(self):
        self.assertTrue(am.emit("home_agent", "request_ok", 1, model="m", extra={"seconds": 1.5, "cost_eur": 0.01}, home=self.home))
        rows = am.series(self.home)
        self.assertEqual((rows[0]["agent"], rows[0]["metric"], rows[0]["value"], rows[0]["source"]), ("home_agent", "request_ok", 1.0, "live"))
        self.assertTrue(rows[0]["prompt"])  # the prompt version it ran with: a content hash
        self.assertFalse(am.emit("x", "y", 1, home=None))  # no workspace known: dropped, not an error
        self.assertFalse(am.emit("x", "y", 1, home="/dev/null/not-a-folder"))  # cannot be written: False, never an exception

    def test_the_scorecard_gives_rates_with_intervals_latency_and_prompts(self):
        for ok in (1, 1, 1, 0):
            am.emit("intern", "step_ok", ok, home=self.home)
        for sec in (1.0, 2.0, 3.0, 10.0):
            am.emit("intern", "request_ok", 1, model="m1", prompt="abc12345", extra={"seconds": sec, "cost_eur": 0.5, "tokens_in": 10, "tokens_out": 5}, home=self.home)
        am.emit("intern", "correction", 1, home=self.home)
        card = am.scorecard(self.home)
        a = next(x for x in card["agents"] if x["agent"] == "intern")
        self.assertEqual(a["rates"]["step_ok"]["n"], 4)
        self.assertEqual(a["rates"]["step_ok"]["rate"], 0.75)
        lo, hi = a["rates"]["step_ok"]["interval"]
        self.assertTrue(lo < 0.75 < hi and 0 <= lo and hi <= 1)  # a Wilson interval around a small sample is wide
        self.assertEqual((a["requests"], a["eur"], a["tokens"], a["corrections"]), (4, 2.0, 60, 1))
        self.assertEqual(a["p50_seconds"], 3.0)
        self.assertEqual(a["prompts"]["abc12345"], 4)

    def test_the_evaluation_results_are_read_in_once_each(self):
        results = Path(tempfile.mkdtemp())
        (results / "suite_v1" / "results").mkdir(parents=True)
        (results / "suite_v1" / "results" / "AEV-001.json").write_text(json.dumps({"at": "2026-10-19T10:00:00+00:00", "verdict": "pass", "spent_eur": 0.2,
                                                                                   "summary": {"cases": 20, "right": 17, "note": "text"}}))
        (results / "suite_v1" / "results" / "AEV-002.json").write_text(json.dumps({"at": "2026-10-20T10:00:00+00:00", "verdict": "fail", "summary": {"cases": 20, "right": 9}}))
        added = am.sync_eval(self.home, results)
        self.assertGreaterEqual(added, 6)
        self.assertEqual(am.sync_eval(self.home, results), 0)  # again: nothing new
        suite = am.scorecard(self.home)["suites"][0]
        self.assertEqual((suite["suite"], suite["runs"], suite["passed"], suite["failed"]), ("suite_v1", 2, 1, 1))
        self.assertEqual(suite["latest"]["ref"], "AEV-002")
        self.assertEqual(suite["latest"]["summary"]["right"], 9.0)
        self.assertNotIn("note", suite["latest"]["summary"])  # only numbers

    def test_wilson_is_the_known_interval(self):
        lo, hi = am.wilson(8, 10)
        self.assertAlmostEqual(lo, 0.4902, places=3)
        self.assertAlmostEqual(hi, 0.9433, places=3)
        self.assertEqual(am.wilson(0, 0), (0.0, 1.0))


class HooksTests(unittest.TestCase):
    def test_a_model_request_a_check_a_proposal_and_a_correction_are_written_where_they_happen(self):
        from dclab_rnd import corrections
        from dclab_rnd.models.gateway import Gateway
        from dclab_rnd.models.usage import FileUsage

        home = Path(tempfile.mkdtemp())
        gateway = Gateway(FileUsage(home / "u.jsonl"))
        gateway.home = home
        gateway.record("home_agent", "claude-sonnet-5-5", {"input_tokens": 100, "output_tokens": 20}, 1.25, 1, "ok")
        gateway.record("home_agent", "claude-sonnet-5-5", None, 9.0, 2, "error")
        gateway.check("synthetic_schema", "m", False, "the design failed validation")
        corrections.record("notebook_cell", {"answer": None}, {"answer": "accepted"}, by="p1", project="proj1", home=home)
        corrections.record("notebook_cell", {"answer": None}, {"answer": "rejected"}, by="p1", project="proj1", home=home)
        corrections.record("solution", {"a": 1}, {"a": 2}, home=home)
        card = am.scorecard(home)
        by = {a["agent"]: a for a in card["agents"]}
        self.assertEqual(by["home_agent"]["rates"]["request_ok"]["k"], 1.0)
        self.assertEqual(by["home_agent"]["rates"]["request_ok"]["n"], 2)
        self.assertEqual(by["synthetic_schema"]["rates"]["output_check"]["rate"], 0.0)
        self.assertEqual(by["notebook_cell"]["rates"]["proposal_accepted"]["rate"], 0.5)
        self.assertEqual(by["solution"]["corrections"], 1)
        self.assertEqual(by["home_agent"]["models"], {"claude-sonnet-5-5": 2})


class RouteTests(unittest.TestCase):
    def test_the_scorecard_route(self):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd.agentic.server import create_app
        except ImportError as error:
            raise unittest.SkipTest(str(error))
        home = Path(tempfile.mkdtemp())
        client = TestClient(create_app(home))
        am.emit("intern", "step_ok", 1, home=home)
        got = client.get("/api/models/scorecard")
        self.assertEqual(got.status_code, 200, got.text)
        body = got.json()
        self.assertIn("intern", [a["agent"] for a in body["agents"]])
        if (ROOT / "evidence" / "campaigns").is_dir():  # the repository's own evaluation results, read in (absent from the open edition)
            self.assertTrue(body["suites"])
        self.assertIn("never", body["note"].lower())


if __name__ == "__main__":
    unittest.main()
