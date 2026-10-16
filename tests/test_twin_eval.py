"""25.8: the measurement of the twin runs — blind runs on studied samples, judged against the planted truth, stored once."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dclab_core import twin as T


def answer(arm: str, target: str, leakage: list[str], cost: float = 0.0) -> T.ArmAnswer:
    return T.ArmAnswer(arm=arm, answerer="rules" if arm == "system" else "claude", status="answered",
                       value={"target": target, "leakage": leakage}, cost_eur=cost)


def write_run(root: Path, name: str, sample: str, *, blind: bool, ai_leak: list[str]) -> None:
    folder = root / "twin" / "runs" / name
    folder.mkdir(parents=True)
    (folder / "run.json").write_text(json.dumps({"run_id": name, "seed": f"sample:{sample}", "blind": blind, "model": "claude-scripted"}))
    (folder / "compare.json").write_text(json.dumps({
        "status": {"workflow": "completed", "ai_first": "completed"}, "seconds": {"workflow": 10.0, "ai_first": 100.0},
        "holdout": {"metric": "roc_auc", "workflow": 0.8, "ai_first": 0.79},
        "model": {"workflow": {"eur": 0.0, "requests": 0}, "ai_first": {"eur": 1.0, "requests": 20}},
        "judged": [{"step": "solution.columns", "winner": "both", "differs": []}]}))
    truth = T.Truth(value={"target": "y", "leakage": ["leak"]}, source="planted")
    record = T.step(f"{name}:solution.columns", name, "solution.columns", "2026-10-16T00:00:00+00:00", {"run": name},
                    answer("system", "y", ["leak"]), answer("ai_first", "y", ai_leak, 1.0), truth=truth)
    (folder / "steps.jsonl").write_text(record.model_dump_json() + "\n" + "not a step\n")
    design = T.step(f"{name}:expert:v1", name, "solution.columns", "2026-10-16T00:00:00+00:00", {"run": name},
                    answer("system", "y", ["leak"]), answer("ai_first", "y", ["leak"]), truth=truth)
    (folder / "expert_steps.jsonl").write_text(design.model_dump_json() + "\n")  # the same system answer beside a version of the record


class TwinEvalTests(unittest.TestCase):
    def test_only_blind_sample_runs_are_judged_and_the_result_is_stored_once_per_measurement(self):
        from dclab_rnd.twin import evaluate

        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            write_run(home, "r1", "bank_marketing", blind=True, ai_leak=["leak"])
            write_run(home, "r2", "hyperack", blind=True, ai_leak=[])  # Claude missed the leak here
            write_run(home, "r3", "telco_churn", blind=False, ai_leak=["leak"])  # given the solution: never judged
            measured = evaluate.measure(home)
            self.assertEqual([r["sample"] for r in measured["runs"]], ["bank_marketing", "hyperack"])
            self.assertEqual(measured["steps"], 2)  # the malformed line is skipped, never guessed
            row = next(s for s in measured["matrix"]["steps"] if s["step"] == "solution.columns")
            self.assertEqual((row["judged"], row["system_right"], row["ai_right"]), (2, 2, 1))  # each run's answers once (TWN-001 pooled the record's)
            expert = next(s for s in measured["matrix"]["steps"] if s["step"] == "solution.columns")
            self.assertEqual(measured["expert_steps"], 2)
            self.assertEqual(next(s for s in measured["expert_matrix"]["steps"] if s["step"] == "solution.columns")["ai_right"], 2)
            self.assertIsNot(expert, None)
            self.assertEqual(row["mode"], "undecided")  # below the rule's minimum of judged cases
            with mock.patch.object(evaluate, "CAMPAIGN", home / "c"), mock.patch.object(evaluate, "RESULTS", home / "c" / "results"):
                first, second = evaluate.store(measured), evaluate.store(measured, {"id": "TWN-001", "why": "pooled"})
                self.assertEqual(json.loads(second.read_text())["supersedes"], {"id": "TWN-001", "why": "pooled"})
                self.assertEqual((first.name, second.name), ("TWN-001.json", "TWN-002.json"))  # a new file each time, none rewritten
                doc = json.loads(first.read_text())
                self.assertEqual((doc["campaign_id"], doc["spent_eur"], doc["summary"]["samples"]), ("twin_v1", 2.0, ["bank_marketing", "hyperack"]))
                self.assertTrue((home / "c" / "manifest.json").exists())
            self.assertIn("hyperack", evaluate.report(measured))


if __name__ == "__main__":
    unittest.main()
