"""Package 14.5: the prompted base model, the benchmark split in the live runner, and the report against the gate."""
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dclab_rnd.agent_eval import baselines, benchmark, gate  # noqa: E402
import importlib  # noqa: E402

suite = importlib.import_module("dclab_rnd.agent_eval.run")
ROOT = Path(__file__).resolve().parents[1]


class FakeClient:
    """A scripted model: answers with what the test says, and keeps what it was shown."""
    model, spent_eur = "fake", 0.0

    def __init__(self, answer):
        self.answer, self.seen = answer, []

    def complete(self, messages, tools=None, max_tokens=None, **options):
        self.seen.append(messages)
        self.spent_eur += 0.001
        return {"content": self.answer(messages) if callable(self.answer) else self.answer}


def run_one(case, client, max_requests=5, max_eur=None):
    counter = {"requests": 0, "eur": 0.0}
    suite.POLICIES["base_model"] = baselines.base_model_policy(client, counter, max_requests, max_eur)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            return suite.run_case(case, "base_model", Path(tmp), suite._known_ids()), counter
    finally:
        suite.POLICIES.pop("base_model", None)


def dev_case(trap):
    return next(c for c in benchmark.split("dev") if c.trap == trap)


class BaseModelTests(unittest.TestCase):
    def test_a_base_model_that_names_the_leak_keeps_it_out_and_never_sees_a_value(self):
        case = dev_case("post_outcome_name")
        client = FakeClient(json.dumps({"forbidden": [{"column": c, "reason": "written after"} for c in case.leaks], "identifiers": []}))
        result, counter = run_one(case, client)
        self.assertIsNone(result.error)
        self.assertTrue(result.leak_caught)
        self.assertEqual(counter["requests"], 1)
        shown = json.loads(client.seen[0][1]["content"].split("Columns:\n", 1)[1])
        self.assertTrue(all(set(c) == {"name", "kind", "missing", "unique"} for c in shown))  # names and summaries, never a value or a flag
        self.assertNotIn("target", [c["name"] for c in shown])  # the target is not offered as a column to decide on

    def test_a_clean_column_it_forbids_is_a_false_alarm_and_unknown_names_are_dropped(self):
        case = next(c for c in benchmark.split("dev") if c.clean_inputs)
        client = FakeClient(json.dumps({"forbidden": [{"column": case.clean_inputs[0], "reason": "sounds late"}, {"column": "not_a_column", "reason": "x"}]}))
        result, _ = run_one(case, client)
        self.assertTrue(result.false_alarm)
        self.assertNotIn("not_a_column", result.kept_out)

    def test_an_answer_that_is_not_json_is_a_recorded_error_not_a_crash(self):
        result, _ = run_one(benchmark.split("dev")[0], FakeClient("I think income might leak."))
        self.assertIsNotNone(result.error)

    def test_a_cap_stops_before_the_request_is_sent(self):
        client = FakeClient("{}")
        result, counter = run_one(benchmark.split("dev")[0], client, max_requests=0)
        self.assertIn("CapReached", result.error)
        self.assertEqual((counter["requests"], client.seen), (0, []))
        result, _ = run_one(benchmark.split("dev")[0], client, max_eur=0.0)
        self.assertIn("CapReached", result.error)


class SummaryTests(unittest.TestCase):
    def test_a_case_that_did_not_run_counts_against_the_policy(self):
        cases = benchmark.split("test")[:4]
        traps = {c.id: c.trap for c in cases}
        rows = [{"case": c.id, "ran": True, "leak_caught": True, "false_alarm": False, "valid_moves": 1.0, "citations_exist": True} for c in cases[:3]]
        rows.append({"case": cases[3].id, "ran": False, "error": "a cap stopped it", "leak_caught": None, "false_alarm": None})
        summary, _ = baselines.rows_summary(rows, 4, traps)
        self.assertEqual((summary["cases"], summary["errors"]), (4, 1))
        self.assertFalse(gate.verdict(summary)["pass"])  # "every case ran" fails

    def test_the_report_sets_the_sealed_scripted_baselines_against_the_gate(self):
        scripted = json.loads((ROOT / "evidence/campaigns/benchmark_v2/results/BEN-002_test_scripted.json").read_text())
        report = baselines.build_report(scripted, None, None)
        self.assertEqual(sorted(report["policies"]), ["audit", "bad", "standard"])
        self.assertEqual(report["passes"], [])
        self.assertEqual(report["gate"], gate.VERSION)
        self.assertIn(baselines.SCOPE, report["limitations"])  # the verdict is stated as evidence about these policies, never a production claim


class InputChecks(unittest.TestCase):
    def test_a_result_that_is_not_the_whole_sealed_set_is_refused_not_judged(self):
        sealed = json.loads((ROOT / "evidence/campaigns/benchmark_v2/results/BEN-002_test_scripted.json").read_text())
        dev = json.loads((ROOT / "evidence/campaigns/benchmark_v2/results/BEN-001_dev_scripted.json").read_text())
        with self.assertRaises(ValueError):
            baselines.build_report(dev, None, None)  # a dev run
        subset = {**sealed, "cases": sealed["cases"][:50]}
        with self.assertRaises(ValueError):
            baselines.build_report(subset, None, None)  # a subset would pass "every case ran" against its own count
        changed = {**sealed, "cases": [{**c, "fingerprint": "0" * 16} if i == 0 else c for i, c in enumerate(sealed["cases"])]}
        with self.assertRaises(ValueError):
            baselines.build_report(changed, None, None)  # a table that is no longer the frozen one
        with self.assertRaises(ValueError):
            baselines.build_report(sealed, {**dev, "summary": {"base_model": {}}, "by_family": {"base_model": {}}}, None)
        report = baselines.build_report(sealed, None, None)
        self.assertEqual(report["policies"]["audit"]["summary"]["cases"], 70)
        for p in report["policies"].values():
            self.assertEqual(len(p["false_alarms_on_controls"]), 2)  # both denominators are reported
        self.assertTrue(any("control" in line for line in report["limitations"]))


class LiveSplitTests(unittest.TestCase):
    def test_the_live_runner_takes_the_benchmark_split_and_stops_without_yes(self):
        import io
        from contextlib import redirect_stdout

        from dclab_rnd.agent_eval import live

        class Gateway:  # no model: the plan is printed, nothing is sent
            def available(self, purpose): return False
            def route(self, purpose): return ("strong", None)
            def client(self, *a, **k): raise AssertionError("no request may be made")

        out = io.StringIO()
        with redirect_stdout(out):
            code = live.main(["--split", "test", "--repeats", "1"], gateway=Gateway())
        self.assertEqual(code, 2)
        self.assertIn("70 cases", out.getvalue())


if __name__ == "__main__":
    unittest.main()
