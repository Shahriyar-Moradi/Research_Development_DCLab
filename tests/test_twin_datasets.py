"""Capture into training data (package 25.7): what the system and Claude answered on the same inputs becomes the twin's
sets (every step, the preference pairs a truth decided, the expert's records, and Claude's answers as a teacher's beside
the rule's at the ledger's points, one set per use), built only under the workspace's train consent, never a row of data."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import datasets as builders  # noqa: E402
from dclab_core import twin as T  # noqa: E402

AT = "2026-10-16T12:00:00+00:00"


def step(i, system, ai, step_name="home.pack", truth=None, said=None):
    return json.loads(T.step(f"s{i}", "draft-d1", step_name, AT, {"i": i}, T.ArmAnswer(arm="system", answerer="home", value=system),
                             T.ArmAnswer(arm="ai_first", answerer="ai_first@home", model="claude-sonnet-5-5", value=ai, said=said),
                             truth=T.Truth(value=truth, source="person_pick") if truth else None).model_dump_json())


class BuilderTest(unittest.TestCase):
    def test_the_sets_and_the_teacher_rows_by_use(self):
        items = [{"step": step(1, {"pack": "tabular"}, {"pack": "churn"}, truth={"pack": "churn"}, said="Write to jane.roe" + "@" + "example.com"), "where": {"draft": "d1"}},
                 {"step": step(2, {"split": "random", "metric": "roc_auc"}, {"split": "time", "metric": "roc_auc"}, "stage.data"), "where": {"run": "r1"}}]
        decided = {("d1", "problem.pack"): [{"id": "dr-pack-0001", "at": "2026-10-16T11:00:00+00:00", "subject": {"kind": "draft", "key": "d1"},
                                             "state": {"family": "classification"}, "options": None,
                                             "answers": [{"by": "rule", "answerer": "rule:pack_detect", "value": "tabular"}]}]}
        sets = builders.twin_sets(items, decided, traces=[{"trace": {"version": 1, "checkpoint": "initial"}, "run": "r1", "blind": True,
                                                            "columns_winner": "ai_first", "outcome": {"holdout": 0.8}}], consent="train")
        self.assertEqual(set(sets), {"twin.steps", "twin.preferences", "twin.expert", "twin.teacher.technical", "twin.teacher.workflow"})
        steps, _ = sets["twin.steps"]
        self.assertEqual(len(steps), 2)
        self.assertNotIn("example.com", json.dumps(steps))  # Claude's words go through the redaction
        pairs, m = sets["twin.preferences"]
        self.assertEqual([(p["chosen"]["arm"], p["chosen"]["value"]) for p in pairs], [("ai_first", {"pack": "churn"})])
        self.assertEqual(m.schema_version, "dclab_core.dataset_manifest.v6")
        workflow, _ = sets["twin.teacher.workflow"]
        self.assertEqual([(r["point"], r["rule"]["answerer"], r["teacher"]["value"], r["truth"], r["state"]) for r in workflow],
                         [("problem.pack", "rule:pack_detect", "churn", "churn", {"family": "classification"})])
        technical, _ = sets["twin.teacher.technical"]
        self.assertEqual(sorted(r["point"] for r in technical), ["metric.primary", "split.strategy"])
        self.assertEqual(next(r for r in technical if r["point"] == "split.strategy")["rule"], {"answerer": "system", "value": "random"})
        experts, _ = sets["twin.expert"]
        self.assertTrue(experts[0]["verified"])


@unittest.skipIf(False, "")
class WorkspaceTest(unittest.TestCase):
    def test_built_only_under_train_consent_and_only_for_allowed_drafts(self):
        try:
            from fastapi.testclient import TestClient

            from dclab_rnd import consent, datasets
            from dclab_rnd.agentic.server import create_app
        except ImportError as error:  # pragma: no cover
            self.skipTest(str(error))
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            client = TestClient(create_app(Path(tempfile.mkdtemp())))
        s = client.app.state.services
        home = Path(s.store.home)
        d = s.drafts.create("Predict which customers cancel next month")
        folder = home / "twin" / "sessions" / "2026" / "10" / "16" / f"draft-{d['id']}"
        folder.mkdir(parents=True)
        (folder / "steps.jsonl").write_text(json.dumps(step(1, {"pack": "tabular"}, {"pack": "churn"}, truth={"pack": "churn"})) + "\n")
        (home / "twin" / "index.jsonl").write_text(json.dumps({"session": f"draft-{d['id']}", "draft_id": d["id"], "path": str(folder.relative_to(home / "twin"))}) + "\n")
        out = datasets.build(s)
        self.assertFalse((home / "datasets" / "twin").exists())  # the train switch is off: nothing
        consent.set_workspace_consent(home, "train", True)
        out = datasets.build(s)
        built = {b["name"]: b["rows"] for b in out["built"] if b["builder"] == "twin"}
        self.assertEqual((built["twin.steps"], built["twin.preferences"]), (1, 1))
        self.assertTrue((home / "datasets" / "twin" / "twin.steps" / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
