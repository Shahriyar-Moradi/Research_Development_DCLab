"""End-to-end comparison (2026-11-18): cases made by code, a future set no arm sees, a locked Python box, a plain agent,
DCLab's arms read the same way, and scores computed by code."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class CaseTests(unittest.TestCase):
    def test_a_case_is_made_by_code_and_its_future_has_no_leak(self):
        from dclab_rnd.agent_eval.e2e import cases

        for kind in ("ticket_routing", "delivery_time"):
            case = cases.make(kind, 11)
            self.assertEqual(len(case.leaks), 2, kind)
            self.assertIn(case.target, case.table.columns)
            self.assertIn(case.target, case.future.columns)  # the scorer needs the truth of the future cases
            for column in case.leaks:
                self.assertTrue(case.future[column].isna().all(), column)  # blank, as at the prediction moment
                self.assertFalse(case.table[column].isna().all(), column)  # the trap is in the visible table
            self.assertIn(case.filename, case.request())
            self.assertIn(case.moment, case.request())

    def test_the_same_seed_gives_the_same_case_and_the_future_is_new(self):
        from dclab_rnd.agent_eval.e2e import cases

        a, b = cases.make("delivery_time", 12), cases.make("delivery_time", 12)
        self.assertTrue(a.table.equals(b.table))
        self.assertFalse(a.table.head(50).reset_index(drop=True).equals(a.future.head(50).reset_index(drop=True)))


class ReferenceTests(unittest.TestCase):
    def test_a_leak_looks_good_on_the_table_and_fails_on_the_future(self):
        from dclab_rnd.agent_eval.e2e import cases, reference

        case = cases.make("delivery_time", 11)
        honest = [c for c in case.table.columns if c not in case.leaks and c != case.target and not c.endswith("_id")]
        with_leak = honest + case.leaks
        self.assertLess(reference.real_score(case, honest), reference.real_score(case, with_leak))  # MAE: lower is better
        self.assertIsNone(reference.real_score(case, [case.target]))

    def test_the_score_is_the_same_twice(self):
        from dclab_rnd.agent_eval.e2e import cases, reference

        case = cases.make("ticket_routing", 11)
        features = [c for c in case.table.columns if c not in case.leaks and c != case.target][:5]
        self.assertEqual(reference.real_score(case, features), reference.real_score(case, features))


class SandboxTests(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.agent_eval.e2e import sandbox

        if not Path(sandbox.SANDBOX).is_file():
            self.skipTest("no sandbox-exec on this machine")
        self.box = sandbox
        self.folder = Path(tempfile.mkdtemp())

    def test_code_runs_and_reads_its_table(self):
        (self.folder / "data.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        out = self.box.run_python("import pandas as pd\nprint(pd.read_csv('data.csv')['a'].sum())", self.folder, 1)
        self.assertEqual((out["returncode"], out["stdout"].strip()), (0, "4"))

    def test_no_write_outside_its_folder(self):
        outside = Path(tempfile.mkdtemp()) / "escaped.txt"
        out = self.box.run_python(f"open({str(outside)!r}, 'w').write('x')", self.folder, 2)
        self.assertNotEqual(out["returncode"], 0)
        self.assertFalse(outside.exists())

    def test_no_key_reaches_the_box(self):
        with mock.patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-test-not-real"}):
            out = self.box.run_python("import os\nprint(os.environ.get('OPENROUTER_API_KEY'))", self.folder, 3)
        self.assertEqual(out["stdout"].strip(), "None")

    def test_a_long_run_is_stopped(self):
        out = self.box.run_python("import time\ntime.sleep(5)", self.folder, 4, timeout=1)
        self.assertIn("stopped after", out["error"])


class ScriptedClient:
    """Answers with the next scripted reply; records what it was sent."""

    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def complete(self, messages, tools, max_tokens=None):
        self.sent.append({"messages": list(messages), "tools": tools})
        return self.replies.pop(0)


def call(name, args, n=1):
    return {"content": "", "tool_calls": [{"id": f"c{n}", "name": name, "arguments": args}]}


class PlainArmTests(unittest.TestCase):
    def test_the_plain_arm_runs_its_code_and_returns_its_answer(self):
        from dclab_rnd.agent_eval.e2e import cases, plain, sandbox

        if not Path(sandbox.SANDBOX).is_file():
            self.skipTest("no sandbox-exec on this machine")
        case = cases.make("delivery_time", 11)
        answer = {"target": "actual_minutes", "task": "regression", "metric": "mae", "value": 6.1, "low": 5.8, "high": 6.4,
                  "features": ["distance_km"], "report": "MAE about 6.1"}
        client = ScriptedClient([call("run_python", {"code": f"import pandas as pd\nprint(len(pd.read_csv('{case.filename}')))"}, 1),
                                 call("finish", answer, 2)])
        out = plain.run(case, client, Path(tempfile.mkdtemp()))
        self.assertEqual(out["finished"]["features"], ["distance_km"])
        self.assertEqual(out["seen"][0].strip(), str(len(case.table)))
        self.assertEqual([c["tool"] for c in out["calls"]], ["run_python", "finish"])

    def test_its_last_call_offers_only_finish(self):
        from dclab_rnd.agent_eval.e2e import cases, plain

        case = cases.make("delivery_time", 11)
        replies = [{"content": "thinking aloud", "tool_calls": []}] * plain.MAX_CALLS
        client = ScriptedClient(replies)
        out = plain.run(case, client, Path(tempfile.mkdtemp()))
        self.assertIsNone(out["finished"])
        self.assertEqual([t["function"]["name"] for t in client.sent[-1]["tools"]], ["finish"])


class DclabOutcomeTests(unittest.TestCase):
    def test_a_finished_rules_project_reads_into_the_shared_shape(self):
        from dclab_rnd import kinds
        from dclab_rnd.agent_eval.e2e import dclab
        from dclab_rnd.kinds import acceptance

        kind = kinds.get("regression")
        run = acceptance.engine_run(kind.case(0))
        out = dclab.outcome("rules", {"solution": run.solution, "holdout_uses": 1}, run.records)
        self.assertEqual(out["finished"]["target"], "actual_minutes")
        self.assertEqual(out["finished"]["metric"], "mae")
        self.assertNotIn("delivered_at", out["finished"]["features"])  # the rules keep this leak out (the kind's acceptance)
        self.assertIsNotNone(out["finished"]["low"])


class ScoreTests(unittest.TestCase):
    def test_a_leaky_arm_is_caught_and_its_claim_is_not_honest(self):
        from dclab_rnd.agent_eval.e2e import cases, score

        case = cases.make("delivery_time", 11)
        honest_cols = [c for c in case.table.columns if c not in case.leaks and c != case.target and not c.endswith("_id")]
        leaky = {"arm": "plain", "seen": ["MAE 1.20"], "finished": {"target": case.target, "metric": "mae", "value": 1.2, "low": 1.1,
                 "high": 1.3, "features": honest_cols + case.leaks, "report": "MAE 1.20, about 42.5 percent better"}}
        row = score.score(case, leaky)
        self.assertEqual(row["leaks_used"], sorted(case.leaks))
        self.assertFalse(row["leaks_kept_out"])
        self.assertFalse(row["real_inside_interval"])  # it promised far better than it does on new cases
        self.assertEqual(row["numbers_nobody_gave"], 1)  # 42.5 came from no output

    def test_an_unfinished_arm_scores_unfinished(self):
        from dclab_rnd.agent_eval.e2e import cases, score

        self.assertEqual(score.score(cases.make("ticket_routing", 11), {"arm": "dclab_agent", "finished": None})["finished"], False)


class CommandTests(unittest.TestCase):
    def test_a_claude_model_is_refused_and_a_run_without_yes_only_plans(self):
        from dclab_rnd.agent_eval.e2e import __main__ as cli

        self.assertIsNotNone(cli.refuse_model("anthropic/claude-sonnet-5.5"))
        self.assertIsNone(cli.refuse_model("qwen/qwen3.8-flash"))
        self.assertEqual(cli.main(["run", "--model", "claude-sonnet-5-5", "--yes", "--cap-eur", "0.5"]), 2)
        with mock.patch.dict("os.environ", {"DCLAB_NO_LIVE_MODELS": "1"}):
            self.assertEqual(cli.main(["run", "--yes", "--cap-eur", "0.5"]), 2)
        self.assertEqual(cli.main(["plan"]), 0)


class GateTests(unittest.TestCase):
    def test_the_gate_is_pinned(self):
        import hashlib

        folder = ROOT / "dclab_rnd" / "agent_eval" / "e2e"
        pin = (folder / "gate_v1.sha256").read_text(encoding="utf-8").strip()
        self.assertEqual(hashlib.sha256((folder / "gate_v1.json").read_bytes()).hexdigest(), pin)
