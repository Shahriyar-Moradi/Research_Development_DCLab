"""The live judgment suite (package A4.2), with a scripted model: nothing starts without --yes, caps hold, scores carry intervals."""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agent_eval import live  # noqa: E402
from dclab_rnd.models.gateway import Gateway  # noqa: E402
from dclab_rnd.models.usage import FileUsage  # noqa: E402

# DCLAB_NO_LIVE_MODELS stays on: scripted gateways are not blocked by it, a stray default one would be
KEYED = {"OPENAI_API_KEY": "k", "DCLAB_NO_LIVE_MODELS": "1", **{f"DCLAB_TIER_{t}_{n}": "" for t in ("STRONG", "STANDARD", "CHEAP") for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")}}


def reply(text="", calls=()):
    calls = [{"id": f"c{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(calls)]
    return {"content": text, "tool_calls": calls, "usage": {"input_tokens": 100, "output_tokens": 20},
            "assistant_message": {"role": "assistant", "content": text, "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]}}


class Careful:
    """A scripted 'model' that follows the audit: propose, forbid what it flags, finish. It reads the conversation."""
    sent = 0

    def complete(self, messages, tools=None, max_tokens=None, **_):
        Careful.sent += 1
        pid = re.search(r"existing project ([0-9a-f]{12})", messages[1]["content"]).group(1)
        last = messages[-1]
        if last["role"] != "tool":
            return reply(calls=[("propose_solution", {"project_id": pid, "target": "target"})])
        result = json.loads(last["content"])
        if "forbidden" in result and "task" in result:
            ids = result.get("identifiers") or []
            forbid = [{"column": f["column"], "reason": f["reason"]} for f in result["forbidden"] if f["column"] not in ids]
            return reply(calls=[("set_solution", {"project_id": pid, "target": "target", "task": result["task"], "forbidden": forbid, "identifiers": ids,
                                                  "prediction_moment": "At the monthly snapshot, before the outcome is known."})])
        return reply(calls=[("finish", {"report": "Solution saved."})])


def gateway():
    return Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"), transport=lambda tier, purpose: Careful())


def cli(*args, gw=None):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), mock.patch.dict(os.environ, KEYED):
        code = live.main(list(args), gateway=gw or gateway())
    return code, out.getvalue(), err.getvalue()


class LiveTests(unittest.TestCase):
    def setUp(self):
        Careful.sent = 0
        self.out = Path(tempfile.mkdtemp()) / "AEV-900_judgment_v1_live_test.json"

    def test_nothing_is_sent_without_yes_and_the_plan_is_printed(self):
        code, out, err = cli("--cases", "AE-01,AE-18", "--repeats", "3")
        self.assertEqual(code, 2)
        self.assertEqual(Careful.sent, 0)
        self.assertIn("2 cases × 3 repeats = 6 runs", out)
        self.assertIn("At most 108 requests, all by the intern", out)  # 6 runs × (16 tool calls + 2)
        self.assertIn("input tokens", out)
        self.assertIn("not about any model's readiness for production", out)
        self.assertIn("--yes", err)

    def test_without_a_model_nothing_starts(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            code = live.main(["--yes", "--cases", "AE-01"], gateway=Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl")))
        self.assertEqual(code, 2)
        self.assertIn("No model serves the intern", err.getvalue())

    def test_a_run_scores_each_case_with_an_interval_and_is_stored_once(self):
        code, out, err = cli("--yes", "--cases", "AE-04,AE-18", "--repeats", "2", "--output", str(self.out))
        self.assertEqual(code, 0, err)
        stored = json.loads(self.out.read_text())
        self.assertEqual((stored["kind"], stored["experiment_id"], stored["model"] is not None, len(stored["runs"])), ("judgment_suite_live", "AEV-900", True, 4))
        self.assertEqual(stored["summary"]["by_case"]["AE-04"]["leak_caught"], [2, 2])  # the careful model forbids the noisy copy
        self.assertEqual(stored["summary"]["scores"]["leaks_caught"]["mean"], 1.0)
        self.assertEqual(stored["summary"]["scores"]["leaks_caught"]["n"], 2)
        self.assertEqual(stored["requests"], Careful.sent)
        self.assertEqual(stored["spent"]["input_tokens"], 100 * Careful.sent)
        self.assertIn("readiness for production", stored["limitations"][0])
        self.assertIn("95% interval", out)
        with self.assertRaises(FileExistsError):  # a result is never overwritten
            cli("--yes", "--cases", "AE-04", "--repeats", "1", "--output", str(self.out))

    def test_the_request_cap_stops_the_suite_before_it_is_passed(self):
        code, out, err = cli("--yes", "--cases", "AE-04,AE-05,AE-06", "--repeats", "2", "--max-requests", "5", "--output", str(self.out))
        self.assertEqual(code, 0, err)
        stored = json.loads(self.out.read_text())
        self.assertLessEqual(Careful.sent, 5)
        self.assertEqual(stored["status"], "stopped")
        self.assertIn("request cap", stored["stopped"])
        self.assertTrue(any(not r["ran"] for r in stored["runs"]))  # the rest are reported, not run

    def test_only_the_intern_asks_the_model_its_tools_use_their_deterministic_path(self):
        class Reviewing(Careful):  # first asks for the leakage review, a tool that sends its own model request
            def complete(self, messages, tools=None, max_tokens=None, **_):
                if str(messages[0].get("content", "")).startswith("You review a machine-learning table"):  # the review's own request
                    Careful.sent += 1
                    return reply('{"columns": []}')
                if messages[-1]["role"] != "tool":
                    Careful.sent += 1
                    pid = re.search(r"existing project ([0-9a-f]{12})", messages[1]["content"]).group(1)
                    return reply(calls=[("review_leakage", {"project_id": pid, "target": "target", "prediction_moment": "At the monthly snapshot, before the outcome."})])
                if "consider" in messages[-1]["content"] and "forbid" in messages[-1]["content"]:
                    Careful.sent += 1
                    pid = re.search(r"existing project ([0-9a-f]{12})", messages[1]["content"]).group(1)
                    return reply(calls=[("propose_solution", {"project_id": pid, "target": "target"})])
                return super().complete(messages, tools, max_tokens)
        gw = Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"), transport=lambda tier, purpose: Reviewing())
        code, out, err = cli("--yes", "--cases", "AE-04", "--repeats", "1", "--max-requests", "50", "--output", str(self.out), gw=gw)
        self.assertEqual(code, 0, err)
        stored = json.loads(self.out.read_text())
        self.assertEqual(stored["requests"], Careful.sent)  # every request is the intern's: the review asked no model
        self.assertEqual(stored["spent"]["input_tokens"], 100 * Careful.sent)  # so the session's accounting covers them all
        self.assertTrue(stored["summary"]["by_case"]["AE-04"]["leak_caught"][1] == 1)

    def test_a_run_that_fails_or_is_capped_is_not_scored_and_partial_repeats_stay_out_of_the_interval(self):
        class Broken:
            def complete(self, *a, **k):
                raise RuntimeError("APIConnectionError: the model server is down")
        gw = Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"), transport=lambda tier, purpose: Broken())
        code, out, err = cli("--yes", "--cases", "AE-04,AE-05", "--repeats", "2", "--output", str(self.out), gw=gw)
        self.assertEqual(code, 0, err)
        stored = json.loads(self.out.read_text())
        self.assertTrue(all(not r["ran"] for r in stored["runs"]))  # an error is not a missed leak
        self.assertEqual((stored["status"], stored["summary"]["scores"]["leaks_caught"]["n"]), ("stopped", 0))
        rows = [{"case": "AE-04", "repeat": 0, "ran": True, "leak_caught": True, "false_alarm": False, "valid_moves": 1.0, "citations_exist": True},
                {"case": "AE-05", "repeat": 0, "ran": True, "leak_caught": True, "false_alarm": False, "valid_moves": 1.0, "citations_exist": True},
                {"case": "AE-04", "repeat": 1, "ran": True, "leak_caught": False, "false_alarm": False, "valid_moves": 1.0, "citations_exist": True},
                {"case": "AE-05", "repeat": 1, "ran": False, "why": "a cap stopped it"}]
        from dclab_rnd.agent_eval.cases import by_id
        summary = live.summarize(rows, [by_id("AE-04"), by_id("AE-05")], 2)
        self.assertEqual((summary["scores"]["leaks_caught"]["mean"], summary["complete_repeats"]), (1.0, 1))  # the half repeat is left out
        self.assertEqual(summary["by_case"]["AE-04"]["leak_caught"], [1, 2])  # but its scored run is counted per case

    def test_arguments_that_would_not_cap_anything_are_refused(self):
        for args in (["--max-requests", "0"], ["--max-steps", "1"], ["--cap-eur", "2"]):
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), mock.patch.dict(os.environ, KEYED):
                try:
                    code = live.main(["--yes", "--cases", "AE-01", *args], gateway=gateway())
                except SystemExit as stop:
                    code = stop.code
            self.assertEqual(code, 2, args)  # an unpriced model cannot be capped in euros
        self.assertEqual(Careful.sent, 0)

    def test_the_interval_is_a_t_interval_over_repeats(self):
        self.assertEqual(live.interval([1.0, 0.5, 0.75])["mean"], 0.75)
        self.assertEqual(live.interval([1.0])["ci95"], None)
        lo, hi = live.interval([0.6, 0.8, 0.7, 0.9, 0.5])["ci95"]
        self.assertTrue(lo < 0.7 < hi)



class ToolsAsTextScoringTests(unittest.TestCase):
    def test_a_model_that_writes_its_calls_as_text_is_scored_not_cut_short(self):
        """8.1: such a session ends "failed"; it is the model's own behaviour, so the live suite scores it."""
        from dclab_rnd.agent_eval.live import _cut_short

        self.assertIsNone(_cut_short({"status": "failed", "ended": "tools_as_text", "error": "wrote its tool calls as text"}))
        self.assertIn("the run failed", _cut_short({"status": "failed", "error": "ConnectionError"}))


if __name__ == "__main__":
    unittest.main()
