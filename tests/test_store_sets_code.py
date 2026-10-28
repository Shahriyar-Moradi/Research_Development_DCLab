"""36.6: verified code. Only an exported pipeline that reproduced its recorded number in a clean Python becomes an m5
example: the problem as stated (never a result), then the code. Its assistant turn says who wrote the code and carries no
reasoning a model did not give (the owner, 2026-10-28). The split follows the replica's own group; each kind comes from
its newest finished run in which it passed; a test's store never reads the owner's real runs."""

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import lake  # noqa: E402
from dclab_core.datasets import split_of  # noqa: E402
from dclab_core.store import ChatExample  # noqa: E402
from dclab_rnd.store.layout import Layout  # noqa: E402
from dclab_rnd.store.sets import code as code_set  # noqa: E402

PROBLEM = {"kind": "binary", "task": "binary", "moment": "At the monthly snapshot, before the outcome is known.", "target": "churned",
           "columns": ["age", "plan", "churned"], "solution": {"target": "churned", "forbidden": []}}


class VerifiedCodeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.layout = Layout(self.tmp / "store")
        self.run = self.tmp / "runs" / "20261028T100000"

    def kind(self, key, same=True, value=0.81, problem=True):
        d = self.run / key
        d.mkdir(parents=True)
        (d / "pipeline.py").write_text(f"# the {key} pipeline\nprint('ok')\n")
        (d / "README.md").write_text(f"# Replica: recomputes the holdout and exits 1 when it differs from the recorded {value}.\n")
        if problem:
            (d / "problem.json").write_text(json.dumps({**PROBLEM, "kind": key}))
        (d / "result.json").write_text(json.dumps({"kind": key, "metric": "auc", "recorded": value, "reproduced": value if same else 0.5, "same": same}))

    def summary(self, status="passed", at="2026-10-28T10:00:00Z"):
        (self.run / "summary.json").write_text(json.dumps({"status": status, "python": "3.12.13", "finished_at": at}))

    def test_only_a_pipeline_that_reproduced_its_number_becomes_an_example(self):
        self.kind("binary")
        self.kind("streams", same=False)
        self.summary("failed")
        self.assertEqual([e.check["kind"] for e in code_set.examples(self.run)], ["binary"])

    def test_the_question_is_the_problem_with_no_result_and_the_answer_has_no_invented_reasoning(self):
        self.kind("binary", value=0.8137)
        self.summary()
        ex = code_set.examples(self.run)[0]
        ChatExample.model_validate(ex.model_dump())
        user, answer = ex.messages
        for said in ("churned", "At the monthly snapshot", "age, plan, churned"):
            self.assertIn(said, user.content)
        self.assertNotIn("0.8137", user.content)  # the score is the code's to compute, never the question's
        self.assertEqual(answer.content, "# the binary pipeline\nprint('ok')\n")
        self.assertIsNone(answer.reasoning)
        self.assertIsNone(answer.reasoning_source)
        self.assertIn("DCLab's engine", answer.no_reasoning)

    def test_a_kind_without_its_stated_problem_is_left_out(self):
        self.kind("binary", problem=False)  # a run from before problem.json: never a question made from the README
        self.summary()
        self.assertEqual(code_set.examples(self.run), [])

    def test_a_run_without_its_summary_is_refused(self):
        self.kind("binary")
        with self.assertRaises(ValueError):
            code_set.examples(self.run)

    def test_the_split_is_the_replicas_own_group_by_the_stricter_rule(self):
        for key in ("regression", "places", "matching", "uplift", "multiclass", "binary"):
            group = f"selfplay:{key}:0"
            strict = max(split_of(group), lake.split_of(group), key={"train": 0, "dev": 1, "sealed": 2}.__getitem__)
            self.assertEqual(code_set.split_for(key), strict, key)
            if "sealed" in (split_of(group), lake.split_of(group)):
                self.assertEqual(code_set.split_for(key), "sealed", key)

    def test_a_later_failed_run_never_removes_a_verified_kind(self):
        self.kind("binary")
        self.kind("streams")
        self.summary(at="2026-10-28T10:00:00Z")
        self.run = self.tmp / "runs" / "rerun"  # finished later, and binary failed there
        self.kind("binary", same=False)
        self.summary("failed", at="2026-10-28T12:00:00Z")
        got = {e.check["kind"]: e.check["run"] for e in code_set.latest(self.tmp / "runs")}
        self.assertEqual(got, {"binary": "20261028T100000", "streams": "20261028T100000"})

    def test_the_newest_passing_run_wins_by_its_finish_stamp_not_its_name(self):
        self.kind("binary")
        self.summary(at="2026-10-28T12:00:00Z")
        newer = self.run
        self.run = self.tmp / "runs" / "zzz-old"
        self.kind("binary")
        self.summary(at="2026-10-27T09:00:00Z")
        self.assertEqual(code_set.latest(self.tmp / "runs")[0].check["run"], newer.name)

    def test_a_test_store_never_reads_the_real_runs(self):
        self.assertEqual(code_set.runs_for(self.layout), self.layout.root / "code_runs")
        self.assertEqual(code_set.build_latest(self.layout), {"sets": {}})

    def test_the_set_is_written_from_the_newest_passing_runs(self):
        self.kind("binary")
        self.summary()
        out = code_set.build_latest(self.layout, self.tmp / "runs")
        self.assertEqual(out["runs"], ["20261028T100000"])
        self.assertEqual(sum(out["sets"]["m5_solver_llm/verified_code"].get(s, 0) for s in ("train", "dev", "sealed")), 1)


class KeepArgTest(unittest.TestCase):
    def load(self):
        spec = importlib.util.spec_from_file_location("code_check", ROOT / "scripts" / "code_check.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_keep_takes_its_folder_and_a_missing_folder_is_a_usage_error(self):
        mod = self.load()
        self.assertEqual(mod.keep_arg(["binary", "--keep", "out", "streams"]), (Path("out"), ["binary", "streams"]))
        self.assertEqual(mod.keep_arg(["binary"]), (None, ["binary"]))
        with self.assertRaises(SystemExit) as stop:
            mod.keep_arg(["binary", "--keep"])
        self.assertEqual(stop.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
