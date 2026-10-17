"""Package 27.10: a company's own language model's training files, from its own sessions, kept in its own folder."""

import json
import tempfile
import unittest
from pathlib import Path


def record(i, project, label="yes"):
    return {"id": f"dr{i:020d}", "point": "column.is_identifier", "subject": {"kind": "column", "key": f"c{i}"},
            "state": {"column": f"c{i}", "dtype": "integer", "uniqueness": "all"}, "options": ["yes", "no"],
            "answers": [{"by": "rule", "answerer": "rule:id", "value": True}], "value_used": True, "used": "rule", "project": project,
            "at": "2026-10-17T00:00:00+00:00", "consent": "operate",
            "labels": [{"id": f"lb{i:020d}", "record": f"dr{i:020d}", "point": "column.is_identifier", "value": label == "yes",
                        "source": "blind", "by": "reviewer", "weight": 1.0, "at": "2026-10-17T00:00:00+00:00"}]}


class OwnLlmTests(unittest.TestCase):
    def test_a_workspaces_own_sessions_become_its_training_files_in_its_folder(self):
        from dclab_rnd import own_llm

        home = Path(tempfile.mkdtemp())
        records = [record(i, f"p{i % 12}") for i in range(24)]
        items = [{"draft": "d1", "project": "p1", "problem_card": {"family": "classification"}, "data_card": {"tables": []},
                  "workflow": {"source": "rules", "nodes": [{"wf": "WF-01"}]}, "solution": {"target": "churn", "task": "binary"},
                  "pipeline": [{"step": "data", "run": "r1"}]},
                 {"draft": "d2", "project": "p2", "problem_card": {"family": "regression"}, "data_card": {"tables": []},
                  "workflow": {"source": "model"}, "solution": {"target": "price", "task": "regression"}, "pipeline": [{"step": "data", "run": "r2"}]}]
        m = own_llm.export(home, events=[], records=records, workflow_items=items)
        out = home / "own_models" / "llm"
        self.assertTrue((out / "MANIFEST.json").is_file())
        self.assertEqual(m["by_family"].get("solution_design"), 2)
        self.assertGreater(m["by_family"].get("typed_decision", 0), 0)
        self.assertEqual(m["counts"]["eval_only"], 1)  # the workflow a model drew is not a training target
        rows = [json.loads(x) for name in ("train", "val", "sealed", "eval_only") for x in (out / f"{name}.chat.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["messages"][-1]["role"] == "assistant" for r in rows))
        self.assertEqual(sum(m["counts"].values()), len(rows))


if __name__ == "__main__":
    unittest.main()
