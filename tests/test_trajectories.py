"""Trajectory export (package A6.1): finished runs as training records, only from projects their owner opted in."""
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agents.traces import open_traces  # noqa: E402
from dclab_rnd.expansion.runner import fast_profile  # noqa: E402
from dclab_rnd.intern import Intern  # noqa: E402
from dclab_rnd.intern.tools import Toolbox  # noqa: E402
from dclab_rnd.storage import open_stores  # noqa: E402

TELCO = ROOT / "data" / "project" / "telco" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def telco_values() -> set[str]:
    """Every text cell of the Telco table longer than three characters (Yes, No and 0 appear in any English text)."""
    frame = pd.read_csv(TELCO, dtype=str)
    return {v for c in frame.columns for v in frame[c].dropna().unique() if len(v.strip()) > 3 and v not in frame.columns}


class TrajectoryExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.home = Path(tempfile.mkdtemp())
        cls.projects, _, sessions = open_stores(cls.home)  # files, or PostgreSQL under test-pg
        intern = Intern(sessions, Toolbox(cls.projects), None, traces=open_traces(cls.home))  # the standard plan: no model
        ids = []
        with fast_profile():
            for task in ("Predict which telco customers churn", "Predict which telco customers churn, a second look"):
                s = intern.start(task)
                ids.append(intern.run(s["id"])["project_id"])
        cls.shared, cls.private = ids
        from fastapi.testclient import TestClient
        from dclab_rnd.agentic.server import create_app

        with TestClient(create_app(cls.home)) as c:
            h = {"X-DCLab-Token": c.get("/api/config").json()["csrf"]}
            cls.default = c.get(f"/api/projects/{cls.private}").json()["settings"].get("share_for_training")
            cls.patched = c.patch(f"/api/projects/{cls.shared}", json={"settings": {"share_for_training": True}}, headers=h)
            cls.policy = c.get("/api/learn/policy").json()
        from dclab_rnd.studio import sft

        cls.out = cls.home / "export"
        cls.code = sft.main(["--trajectories", "--workspace", str(cls.home), "--out", str(cls.out)])
        cls.lines = (cls.out / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()
        cls.records = [json.loads(line) for line in cls.lines]
        cls.manifest = json.loads((cls.out / "manifest.json").read_text(encoding="utf-8"))

    def test_only_the_opted_in_project_is_exported(self):
        self.assertIs(self.default, False)  # off by default
        self.assertEqual(self.patched.status_code, 200, self.patched.text)
        self.assertIn("training_opt_in_changed", json.dumps(self.projects.activity(self.shared)))  # the owner's choice is logged
        self.assertEqual(self.code, 0)
        self.assertTrue(self.records)
        self.assertEqual({r["project_id"] for r in self.records}, {self.shared})
        self.assertEqual((self.manifest["projects"], self.manifest["not_opted_in"]), (1, 1))
        self.assertIn("opted in", json.dumps(self.policy).lower())  # the Policy model page says so

    def test_a_record_has_the_state_the_choices_the_move_the_verdict_the_evidence_and_what_followed(self):
        kinds = {r["kind"] for r in self.records}
        self.assertEqual(kinds, {"move", "step"})  # the validator's transitions and the agent's tool calls
        for r in self.records:
            self.assertTrue(r["state"], r)
            self.assertIn("allowed", r)
            self.assertTrue(r["move"] if r["kind"] == "move" else r["tool"], r)
            self.assertIn(r["verdict"], ("allowed", "blocked", "needs_approval", "ok", "error"))
            self.assertIsInstance(r["cited"], list)
            self.assertIn("next", r)
        moves = [r for r in self.records if r["kind"] == "move"]
        self.assertTrue(any(r["allowed"] for r in moves))  # the moves the graph allowed at that state
        self.assertTrue(any(r["cited"] for r in moves))  # the rules and records the validator cited
        self.assertTrue(any(r["next"]["state"] for r in moves[:-1]))
        steps = [r for r in self.records if r["kind"] == "step"]
        self.assertIn("set_solution", {r["tool"] for r in steps})
        self.assertTrue(all(r["allowed"] for r in steps))  # the tools the agent could call

    def test_no_cell_value_and_no_free_text_is_exported(self):
        text = "\n".join(self.lines)
        # a value as a whole token: a leak is a JSON string or a word of its own, while a random hex id may contain "6465"
        leaked = sorted(v for v in telco_values() if re.search(r"(?<![\w.-])" + re.escape(v) + r"(?![\w.-])", text))
        self.assertEqual(leaked, [])
        self.assertNotIn("positive_label", text)  # a value of the target
        self.assertNotIn("prediction_moment", text)  # free text the person or the agent wrote
        solution = next(r for r in self.records if r.get("tool") == "set_solution")
        self.assertEqual(solution["arguments"]["target"], "Churn")  # a column name is kept: the schema's non-free-text fields

    def test_a_value_a_model_put_in_a_name_field_is_not_exported(self):
        from dclab_rnd.studio import trajectories as tr

        project = self.projects.get(self.shared)
        vocab = tr.vocabulary(project, self.projects.records(self.shared), ["s1"])
        rows = [  # what a model could send: a customer id as an identifier, a cell as a column, a cell as a choice
            {"agent": "intern", "n": 1, "reply": 0, "state": "dd--------", "tool": "set_solution", "verdict": "error",
             "arguments": {"project_id": self.shared, "target": "Churn", "identifiers": ["7590-VHVEG", "customerID"],
                           "forbidden": [{"column": "Contract=Month-to-month", "reason": "x"}, {"column": "TotalCharges", "reason": "after"}]},
             "result": "error: no column Month-to-month; see PRJ-7590-VHVEG and DCLAB-R04"},
            {"agent": "intern", "n": 2, "reply": 1, "state": "dd--------", "tool": "approve_stage", "verdict": "error",
             "arguments": {"project_id": self.shared, "stage": "features", "choice": "Month-to-month"}, "result": "error"},
            {"agent": "intern", "n": 3, "reply": 2, "state": "dd--------", "tool": "finish", "verdict": "ok", "arguments": {"summary": "done"}, "result": "{}"},
            {"agent": "home", "n": 1, "reply": 0, "state": "x-x-", "tool": "get_profile", "verdict": "ok",
             "arguments": {"columns": ["Contract", "Fiber optic"]}, "result": "{}"},
        ]
        out = tr.steps(project, "s1", rows, {}, vocab)
        text = json.dumps(out)
        for value in ("7590-VHVEG", "Month-to-month", "Fiber optic", "PRJ-7590-VHVEG", "summary"):
            self.assertNotIn(value, text)
        self.assertEqual(out[0]["arguments"]["identifiers"], ["customerID"])  # a real column is kept
        self.assertEqual(out[0]["arguments"]["forbidden"], [{"column": "TotalCharges"}])
        self.assertEqual(out[0]["cited"], ["DCLAB-R04"])  # an evidence record, not a code that looks like an id
        self.assertNotIn("choice", out[1]["arguments"])
        self.assertIn("finish", out[2]["allowed"])  # the chosen tool is among those offered
        self.assertEqual(out[3]["arguments"], {"columns": ["Contract"]})

    def test_the_export_never_writes_into_the_v3_corpus(self):
        from dclab_rnd.studio import sft

        v3 = ROOT / "research" / "llm-fine-tuning" / "experiments" / "sft" / "out_v3" / "trajectories"
        with self.assertRaises(SystemExit):
            sft.main(["--trajectories", "--workspace", str(self.home), "--out", str(v3)])
        self.assertFalse(v3.exists())
        cased = Path(str(v3).replace("out_v3", "OUT_V3"))  # macOS file systems ignore letter case
        with self.assertRaises(SystemExit):
            sft.main(["--trajectories", "--workspace", str(self.home), "--out", str(cased)])
        self.assertFalse(v3.exists())


if __name__ == "__main__":
    unittest.main()
